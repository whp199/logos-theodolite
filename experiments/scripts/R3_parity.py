#!/usr/bin/env python3
"""R3 parity (CPU only; no GPU, no network): the positive control. R3's fit code against jpp_lens's own stages at the
pinned commit, on a tiny Gemma 4 that keeps the real config's flags (gelu_pytorch_tanh, hidden_size_per_layer_input
0, attention_k_eq_v, the 30.0 softcap), on the first WikiText records of the cache.

jpp_lens's side runs in its own environment (experiments/R3_cache/env) as a subprocess of this file (--jpp-side):
the stage functions of scripts/jpp_cli.py (fit_router, fit_shard, merge_experts, the pooled lens, fit_weights) with
the trainers, router, merge and weight fitter they call, exactly as at d7efc0de. pandas is not installed there, so
the subprocess puts a stand-in module under the name "pandas" that raises on any use: the stages run only if they
never touch it (deviation PD1).

R3's side runs here, with R3_fitlib (its router with the streamed fp64 PCA, its batched-row estimator under R3_lrp's
rules, its per-layer shard files and streamed merge, its item code) and jpp_lens's fitter for the weights.

Every compared quantity must agree within 1e-5 (max |R3 - jpp| <= 1e-5 * max(1, max |jpp|)), the integer ones
exactly. Any remaining difference is listed as a numbered deviation. Also checked, on the real tokenizer: R3's
encoding of the 1,000 records equals jpp_lens's (HFLensModel with force_bos) and the instrument's (BOS + tokens).

Writes experiments/results/R3_parity.json (numbers only: no text from WikiText or the items) and exits 1 on failure.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_parity.py
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import argparse
import datetime
import json
import random
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402

T0 = time.time()

# The tiny run's sizes (the recipe's where they fit a 32-wide model; PD2 lists the rest)
SPEC = {
    "seed": 0,
    "source_layers": [1, 2, 3, 4],            # the tiny model has 6 blocks; the target is block 5's raw output
    "n_experts": C.N_EXPERTS,                 # 8, as registered
    "pca_dim": 16,                            # registered 64 > d_model 32
    "skip_first": C.SKIP_FIRST,               # 16, as registered
    "max_seq_len": 48,                        # registered 128; a character tokenizer fills 48 from any record
    "rows_per_pass": 8,
    "n_router": 96,                           # registered 1,000
    "n_fit": 32,                              # registered 64
    "n_shards": 4,                            # registered 8 (halves A/B: shards 0, 2 and 1, 3)
    "min_kept": C.MIN_KEPT,                   # 50, as registered
    "min_kept_probe": 120,                    # a second merge in which some experts fall back to the pooled lens
    "readout_max_seq_len": C.READOUT_MAX_SEQ_LEN,
    "fit_fraction": C.FIT_FRACTION,
    "split_seed": C.SPLIT_SEED,
    "items_per_eval": 15,
}
TOL = 1e-5
MACRO = list(C.JPP_MACRO_EVALS)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def synthetic_items(n: int) -> Dict[str, List[Dict[str, Any]]]:
    """Item files in J++'s format for the tiny model's character tokenizer (single letters are its only one-token
    words): made up here, so no J++ item text is used or written."""
    rng = random.Random(7)
    letters = "abcdefghijklmnopqrstuvwxyz"

    def words(k):
        return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(2, 6))) for _ in range(k))

    out = {}
    for slug in C.JPP_EVALS:
        items = []
        for i in range(n):
            inter = [rng.choice(letters), rng.choice(letters)]
            if slug == "order-ops":
                inter = [rng.choice(["3", "4", "addition", "7"]), rng.choice(letters)]
            prompt = words(rng.randint(3, 6))
            if slug == "poetry":
                prompt = words(3) + "\n" + words(2)
            it = {"name": f"{slug}-{i:02d}", "prompt": prompt, "intermediates": inter}
            if slug in ("multihop", "multilingual", "order-ops"):
                it["target"] = rng.choice(letters)
            items.append(it)
        out[slug] = items
    return out


def write_items(d: Path, items: Dict[str, List[Dict[str, Any]]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for slug, its in items.items():
        (d / f"lens-eval-{slug}.json").write_text(json.dumps({"items": its}), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ jpp_lens's side
def jpp_side(tmp: Path) -> None:
    """Runs in jpp_lens's environment. Writes tmp/jpp.pt."""
    # transformers probes for pandas while it imports, so everything that is not jpp_lens's own code is imported
    # before the stand-in goes in
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin  # noqa: F401
    import workspace_lens  # noqa: F401

    def _unusable(name):
        def __init__(self, *a, **k):
            raise RuntimeError(f"pandas.{name} was built; R3_parity runs jpp_lens's stages without pandas")
        return type(name, (), {"__init__": __init__})

    class _NoPandas(types.ModuleType):
        """Stands in for pandas: the two type names that jpp_lens's signatures evaluate at import
        (readout_eval_scoring.py has no postponed annotations) are classes that raise when built; any other
        attribute raises when it is looked up."""

        def __getattr__(self, name):
            if name in ("DataFrame", "Series"):
                return _unusable(name)
            raise RuntimeError(f"pandas.{name} was used; R3_parity runs jpp_lens's stages without pandas")

    sys.modules["pandas"] = _NoPandas("pandas")
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("jpp_cli", C.JPP_DIR / "scripts" / "jpp_cli.py")
    cli = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(cli)
    import jlens
    import workspace_lens
    from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items, non_semantic_token_ids
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.condense_experts import CondenseConfig, ExpertJacobians
    from workspace_lens.tests.tokenizers import CharTokenizer as JppCharTokenizer
    from workspace_lens.utils import vocab_size_of

    S = json.loads((tmp / "spec.json").read_text())
    records = C.wikitext_records()
    hf = C.build_tiny_gemma4(seed=S["seed"])
    model = jlens.from_hf(hf, JppCharTokenizer())
    layers = S["source_layers"]
    out: Dict[str, Any] = {"src": str(Path(workspace_lens.__file__).resolve()), "cli": str(Path(cli.__file__))}
    router = cli.fit_router(model, records[:S["n_router"]], layers=layers, num_clusters=S["n_experts"],
                            projection_dim=S["pca_dim"], skip_first_n_positions=S["skip_first"],
                            max_seq_len=S["max_seq_len"], seed=0)
    out["router"] = {l: {"mean": r.pca_mean_N, "components": r.pca_components_ND, "centroids": r.centroids_ED}
                     for l, r in router.layer_routers_L_dict.items()}
    cfg = LensConfig(hf_model_name="tiny", checkpoint_name="parity", artifacts_base_dir=str(tmp / "jpp_artifacts"),
                     source_layers=layers, relative_end_transport_layer=-1, jacobian_rows_per_pass=S["rows_per_pass"],
                     max_seq_len=S["max_seq_len"], skip_first_n_positions=S["skip_first"],
                     checkpoint_every_n_prompts=None, lrp_mode=C.LRP_MODE)
    fit_prompts = records[:S["n_fit"]]
    dirs = [cli.fit_shard(model, fit_prompts, routers=[router], shard_idx=i, num_shards=S["n_shards"], config=cfg)
            for i in range(S["n_shards"])]
    out["shards"] = []
    for d in dirs:
        st = torch.load(Path(d) / f"experts_K{S['n_experts']}_checkpoint.pt", map_location="cpu", weights_only=True)
        out["shards"].append({"layer_sums": st["layer_sums"], "position_records": st["position_records"],
                              "lrp_rules": st["lrp_rules"]})
    experts = cli.merge_experts(dirs, num_clusters=S["n_experts"], min_kept_positions=S["min_kept"],
                                checkpoint_name="parity_experts", device="cpu")
    probe = cli.merge_experts(dirs, num_clusters=S["n_experts"], min_kept_positions=S["min_kept_probe"],
                              checkpoint_name="parity_experts_probe", device="cpu")
    out["probe_experts"] = {l: e.half() for l, e in probe.experts_L_dict_EFN.items()}
    out["probe_fallback"] = probe.fallback_L_dict_Bool_E
    out["experts_fp32"] = experts.experts_L_dict_EFN
    out["fallback"] = experts.fallback_L_dict_Bool_E
    out["counts"] = experts.position_counts_L_dict_E
    out["pooled"] = experts.pooled_lens(checkpoint_name="parity_pooled").jacobians_L_dict_FN
    experts.save(str(tmp / "jpp_experts.pt"))
    loaded = ExpertJacobians.load(str(tmp / "jpp_experts.pt"))       # fit-weights reads the saved (fp16) file
    out["experts_fp16"] = loaded.experts_L_dict_EFN
    data_dir = str(tmp / "items")
    items = load_readout_eval_items(data_dir, slugs=MACRO)
    runner = cli.readout_runner(model, layers=layers, max_seq_len=S["readout_max_seq_len"])
    scoreable = runner.scoreable_items(items)
    fitting = cli.select_items(scoreable, f"fit:{S['split_seed']}", fit_fraction=S["fit_fraction"])
    excluded = non_semantic_token_ids(model.tokenizer, vocab_size=vocab_size_of(model),
                                      items=load_readout_eval_items(data_dir))
    res = runner.record_residuals(scoreable, hf_model_name="tiny")
    weighting, lens = cli.fit_weights(model, loaded, scoreable, layers=layers, excluded_token_ids=excluded,
                                      cache_path=None, checkpoint_name="parity_jpp", hf_model_name="tiny",
                                      max_seq_len=S["readout_max_seq_len"],
                                      fitting_item_names=[it.name for it in fitting], condense_config=CondenseConfig())
    out.update({
        "scoreable": [it.name for it in scoreable], "fitting": [it.name for it in fitting], "excluded": excluded,
        "residuals": res.residuals_L_dict_PN, "candidates": res.candidate_ids_P_list, "res_names": res.item_names,
        "weights": {l: {"w": lw.weights_E, "train_macro_pass": lw.train_macro_pass, "init": lw.init_name,
                        "step": lw.step} for l, lw in weighting.learned_weights_L_dict.items()},
        "lens": lens.jacobians_L_dict_FN,
    })
    tok = C.R2.load_tokenizer()
    real = jlens.from_hf(hf, tok)                                     # force_bos=True, as every J++ fit
    out["real_encode_sha"] = [C.text_sha256(json.dumps(real.encode(t, max_length=C.MAX_SEQ_LEN)[0].tolist()))
                              for t in records]
    torch.save(out, tmp / "jpp.pt")


# ------------------------------------------------------------------------------------------------ R3's side
def r3_side(tmp: Path, jpp_router: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    import R3_fitlib as F

    S = json.loads((tmp / "spec.json").read_text())
    records = C.wikitext_records()
    layers = S["source_layers"]
    dec = F.Decoder(C.build_tiny_gemma4(seed=S["seed"]), C.CharTokenizer())
    kw = dict(skip=S["skip_first"], max_seq_len=S["max_seq_len"])
    out: Dict[str, Any] = {}
    out["router"] = F.fit_router(lambda: F.kept_points(dec, records[:S["n_router"]], layers, **kw), layers,
                                 n_experts=S["n_experts"], pca_dim=S["pca_dim"], seed=0)
    jr = {l: F.LayerRouter(v["mean"], v["components"], v["centroids"]) for l, v in jpp_router.items()}
    # the pool's routing under both routers
    out["assign_r3"], out["assign_jpp"] = {l: [] for l in layers}, {l: [] for l in layers}
    for pts in F.kept_points(dec, records[:S["n_router"]], layers, **kw):
        for l in layers:
            out["assign_r3"][l].append(out["router"][l].assign(pts[l]))
            out["assign_jpp"][l].append(jr[l].assign(pts[l]))
    fit_prompts = records[:S["n_fit"]]
    slices = F.shard_slices(len(fit_prompts), S["n_shards"])

    def run(routers, mode, tag):
        dirs, recs, sums = [], [], []
        for s, sl in enumerate(slices):
            sm, rc = F.fit_shard(dec, fit_prompts[sl], routers, layers, n_experts=S["n_experts"],
                                 rows_per_pass=S["rows_per_pass"], mode=mode, lrp=True, first_idx=sl.start,
                                 target=dec.n_layers - 1, **kw)
            d = tmp / f"r3_{tag}" / f"shard{s}"
            F.write_shard(d, sm, rc)
            dirs.append(d)
            recs += rc
            sums.append(sm)
        merged = {l: F.merge_layer(dirs, l, min_kept=S["min_kept"]) for l in layers}
        if tag == "jr":
            out["probe"] = {l: F.merge_layer(dirs, l, min_kept=S["min_kept_probe"], halves=False) for l in layers}
        return sums, recs, merged

    # the estimator and merge with jpp_lens's router (isolates them from the PCA precision), in both batching modes
    out["sums_jr"], out["recs"], out["merged_jr"] = run(jr, "replicate", "jr")
    out["sums_jr_vmap"], _, _ = run(jr, "vmap", "jr_vmap")
    # end to end with R3's own router
    out["sums_own"], _, out["merged_own"] = run(out["router"], "replicate", "own")
    raw = {s: json.loads((tmp / "items" / f"lens-eval-{s}.json").read_text())["items"] for s in C.JPP_EVALS}
    items = F.items_from_raw(raw, MACRO)
    all_items = F.items_from_raw(raw, C.JPP_EVALS)
    scoreable = F.scoreable(items, dec.tokenizer)
    fitting, _ = F.split_items(scoreable, seed=S["split_seed"], fit_fraction=S["fit_fraction"])
    excluded = F.excluded_ids(dec.tokenizer, F.vocab_size(dec), all_items)
    res = F.record_residuals(dec, scoreable, layers, S["readout_max_seq_len"])
    out.update({"scoreable": [it.name for it in scoreable], "fitting": [it.name for it in fitting],
                "excluded": excluded, "residuals": res.residuals_L_dict_PN, "candidates": res.candidate_ids_P_list,
                "res_names": res.item_names})
    for tag in ("jr", "own"):
        merged = out[f"merged_{tag}"]
        wt = F.fit_weights(dec, {l: merged[l].experts_fp16 for l in layers}, res, excluded, layers,
                           [it.name for it in fitting])
        out[f"weights_{tag}"] = {l: {"w": lw.weights_E, "train_macro_pass": lw.train_macro_pass,
                                     "init": lw.init_name, "step": lw.step}
                                 for l, lw in wt.learned_weights_L_dict.items()}
        out[f"lens_{tag}"] = {l: F.combine(merged[l].experts_fp16, wt.learned_weights_L_dict[l].weights_E)
                              for l in layers}
    tok = C.R2.load_tokenizer()
    rdec = F.Decoder(C.build_tiny_gemma4(seed=S["seed"]), tok)
    out["real_encode_sha"] = [C.text_sha256(json.dumps(rdec.encode(t, max_length=C.MAX_SEQ_LEN)[0].tolist()))
                              for t in records]
    out["instrument_encode_sha"] = [C.text_sha256(json.dumps(C.R2.text_ids(tok, t)[:C.MAX_SEQ_LEN])) for t in records]
    return out


# ------------------------------------------------------------------------------------------------ comparison
def close(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return {"ok": False, "shape": [list(a.shape), list(b.shape)]}
    d = float((a - b).abs().max()) if a.numel() else 0.0
    scale = float(b.abs().max()) if b.numel() else 0.0
    return {"ok": d <= TOL * max(1.0, scale), "max_abs_diff": d, "max_abs_ref": scale, "bit_identical": d == 0.0}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--jpp-side":
        jpp_side(Path(sys.argv[2]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(C.PARITY_JSON))
    a = ap.parse_args()
    import R3_fitlib as F

    C.verify_jpp_checkout()
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, **detail) -> bool:
        checks.append({"no": len(checks) + 1, "check": name, "ok": bool(ok), "detail": detail})
        log(("ok    " if ok else "FAIL  ") + name, json.dumps(detail, default=str)[:240] if detail else "")
        return ok

    with tempfile.TemporaryDirectory(prefix="r3_parity_") as tmpd:
        tmp = Path(tmpd)
        (tmp / "spec.json").write_text(json.dumps(SPEC))
        write_items(tmp / "items", synthetic_items(SPEC["items_per_eval"]))
        log("jpp_lens's stages, in its own environment ...")
        p = C.env_subprocess([Path(__file__).resolve(), "--jpp-side", tmp], cwd=str(C.JPP_DIR))
        if p.returncode != 0:
            print(p.stdout[-3000:], p.stderr[-6000:])
            check("jpp_lens's stages ran (fit_router, fit_shard x4, merge_experts, pooled lens, fit_weights)", False)
            return finish(a.out, checks, None)
        J = torch.load(tmp / "jpp.pt", map_location="cpu", weights_only=False)
        check("jpp_lens's stages ran in its environment from the pinned checkout, pandas never used",
              J["src"].startswith(str(C.JPP_SRC.resolve())), src=J["src"])
        log("R3's side ...")
        R = r3_side(tmp, J["router"])
        layers = SPEC["source_layers"]

        # P: encoding
        check("real tokenizer: R3's encoding of the 1,000 records equals jpp_lens's (HFLensModel, force_bos)",
              R["real_encode_sha"] == J["real_encode_sha"], n=len(R["real_encode_sha"]))
        check("real tokenizer: and equals the instrument's (BOS + tokens, truncated to 128)",
              R["real_encode_sha"] == R["instrument_encode_sha"])
        # router
        r_det: Dict[str, Any] = {}
        ok_r = True
        for l in layers:
            jr, rr = J["router"][l], R["router"][l]
            sign = torch.sign((rr.components * jr["components"]).sum(0))
            cos = float(((rr.components * jr["components"]).sum(0).abs()).min())
            mean = close(rr.mean, jr["mean"])
            cent = close(rr.centroids * sign, jr["centroids"])
            same = int(sum(int((x == y).sum()) for x, y in zip(R["assign_r3"][l], R["assign_jpp"][l])))
            tot = int(sum(len(x) for x in R["assign_r3"][l]))
            r_det[str(l)] = {"min_abs_cos": cos, "mean": mean["max_abs_diff"], "centroids": cent["max_abs_diff"],
                             "assignments_equal": same, "of": tot}
            ok_r &= cos >= 1 - TOL and mean["ok"] and cent["ok"] and same == tot
        check("router: R3's streamed fp64 PCA and k-means equal jpp_lens's fit_router (components up to sign, "
              "centroids, every pool position's expert)", ok_r, by_layer=r_det)
        # positions
        jrec = [r for s in J["shards"] for r in s["position_records"]]
        check("kept positions and token ids of every fit prompt equal jpp_lens's position records",
              [(r["token_ids_sha256"], r["source_positions"]) for r in jrec]
              == [(r.token_ids_sha256, r.positions) for r in R["recs"]], n=len(jrec))
        check("jpp_lens's shards were fitted under the rlens rules (ln, identity, half)",
              all(s["lrp_rules"]["ln_rule"] and s["lrp_rules"]["identity_rule"] and s["lrp_rules"]["half_rule"]
                  for s in J["shards"]))
        # shard sums
        for tag, label in (("sums_jr", "replicated batch"), ("sums_jr_vmap", "one forward, batched backward (vmap)"),
                           ("sums_own", "R3's own router")):
            det, ok = {}, True
            for s, (js, rs) in enumerate(zip(J["shards"], R[tag])):
                for l in layers:
                    jl = js["layer_sums"][l]
                    c = close(rs[l].row_sum, jl["weighted_jacobian_row_sum_EFN"])
                    cnt = torch.equal(rs[l].count, jl["position_count_E"]) and torch.equal(rs[l].weight, jl["weight_sum_E"])
                    ok &= c["ok"] and cnt
                    det[f"shard{s} L{l}"] = {"row_sum_max_abs_diff": c["max_abs_diff"], "scale": c["max_abs_ref"],
                                             "counts_equal": cnt}
            worst = max(v["row_sum_max_abs_diff"] for v in det.values())
            check(f"expert sums of every shard and layer equal jpp_lens's fit_shard ({label}); worst max |diff| "
                  f"{worst:.2e}", ok, worst=worst, by_shard_layer=det)
        # merge
        for tag, label in (("merged_jr", "jpp_lens's router"), ("merged_own", "R3's own router")):
            det, ok = {}, True
            for l in layers:
                m = R[tag][l]
                e32 = close(m.experts_fp16.float(), J["experts_fp16"][l].float())
                fb = torch.equal(m.fallback, J["fallback"][l])
                cn = torch.equal(m.count, J["counts"][l])
                pl = close(m.pooled, J["pooled"][l])
                ok &= e32["ok"] and fb and cn and pl["ok"]
                det[str(l)] = {"experts_fp16_max_abs_diff": e32["max_abs_diff"], "fallback_equal": fb,
                               "fallback": m.fallback.int().tolist(), "counts": m.count.tolist(),
                               "pooled_max_abs_diff": pl["max_abs_diff"]}
            check(f"streamed per-layer merge ({label}): experts (fp16, as jpp_lens's experts file), the experts that "
                  f"fall back to pooled (fewer than {SPEC['min_kept']} positions), counts and the pooled lens equal "
                  f"jpp_lens's merge_experts", ok, by_layer=det)
        det, ok = {}, True
        for l in layers:
            m = R["probe"][l]
            same = torch.equal(m.experts_fp16, J["probe_experts"][l]) and torch.equal(m.fallback, J["probe_fallback"][l])
            ok &= same
            det[str(l)] = {"fallback": m.fallback.int().tolist(), "equal": same}
        n_fb = sum(sum(v["fallback"]) for v in det.values())
        check(f"the fallback branch: merged with min kept {SPEC['min_kept_probe']}, the {n_fb} experts that fall back "
              f"to the pooled Jacobian and every expert matrix equal jpp_lens's merge_experts", ok and n_fb > 0,
              by_layer=det)
        # items
        check("items: the scoreable items and the fitting split (seed 0, 20% per eval) equal jpp_lens's",
              R["scoreable"] == J["scoreable"] and R["fitting"] == J["fitting"],
              n_scoreable=len(R["scoreable"]), n_fitting=len(R["fitting"]))
        check("items: the Readout Filtering exclusion list (no letter or digit, sparing every candidate of the six "
              "evals) equals jpp_lens's non_semantic_token_ids", R["excluded"] == J["excluded"],
              n_excluded=len(R["excluded"]))
        ok = R["res_names"] == J["res_names"] and R["candidates"] == J["candidates"]
        det = {str(l): close(R["residuals"][l], J["residuals"][l])["max_abs_diff"] for l in layers}
        ok &= all(close(R["residuals"][l], J["residuals"][l])["ok"] for l in layers)
        check("items: readout residuals and candidate ids equal jpp_lens's record_residuals", ok, by_layer=det)
        # weights and the combined lens
        for tag, label in (("jr", "jpp_lens's router"), ("own", "R3's own router")):
            det, ok = {}, True
            for l in layers:
                rw, jw = R[f"weights_{tag}"][l], J["weights"][l]
                c = close(rw["w"], jw["w"])
                same = rw["train_macro_pass"] == jw["train_macro_pass"] and rw["init"] == jw["init"] and rw["step"] == jw["step"]
                lc = close(R[f"lens_{tag}"][l], J["lens"][l])
                ok &= c["ok"] and same and lc["ok"]
                det[str(l)] = {"weights_max_abs_diff": c["max_abs_diff"], "same_iterate": same,
                               "train_macro_pass": rw["train_macro_pass"], "lens_max_abs_diff": lc["max_abs_diff"]}
            check(f"expert weights (pass@10 surrogate, 3 starts x 200 steps) and the combined J++ lens equal "
                  f"jpp_lens's fit_weights ({label})", ok, by_layer=det)
    return finish(a.out, checks, SPEC)


DEVIATIONS = [
    "PD1. jpp_lens's stages run as the library functions of scripts/jpp_cli.py (loaded by path), not through its "
    "command line, and with a stand-in module named pandas (pandas is not installed; no download beyond the three "
    "approved sources is approved): any attribute raises when looked up, except DataFrame and Series, which "
    "jpp_lens's signatures name at import and which raise when built. No stage used it. Not run: grade_model_correctness and the "
    "correctness CSV (they need pandas); R3 grades correctness itself (D8) with jpp_lens's own "
    "next_token_matches_target on the greedy token, so its grading is jpp_lens's by construction, not by test.",
    "PD2. The tiny run's sizes differ from the registered ones where a 32-wide model forces it or to keep it to "
    "seconds: PCA 16 (64 registered), 96 router records (1,000), 32 fit prompts (64) in 4 shards (8), 48 tokens "
    "(128), 8 rows per pass, a character tokenizer, made-up items in J++'s format. E = 8, skip 16, min kept 50, "
    "the R-lens rules, the split rule and the condense defaults are the registered ones.",
    "PD3. The weights are fitted by jpp_lens's own ExpertWeightingFitter, imported, not reimplemented, so the "
    "weight parity tests R3's inputs to it (fp16 experts, residuals, candidates, exclusions, the split).",
    "Matched, not deviations (the parity decides, R3_PREREG §2): (a) the experts are rounded to fp16 before the "
    "weights are fitted and combined, because jpp_lens's fit-weights stage reads the fp16 experts file; the pooled "
    "R-lens comes from the in-memory fp32 sums (merge-experts --pooled-lens-out); (b) the weight surrogate sees "
    "softcapped logits (HFLensModel.unembed applies final_logit_softcapping); (c) the final norm is patched, as "
    "jpp_lens patches it; it is off the path of the block-output target (R3_lrp --selftest); (d) the exclusion list "
    "spares every candidate id of the six evals' items (readout_eval_items.py:191-212); (e) BOS is forced on raw "
    "text (HFLensModel force_bos): without it this tokenizer adds no BOS.",
    "Measured, a registered deviation (D4): R3's PCA is a streamed fp64 covariance where jpp_lens's fit_pca is fp32 "
    "on stored points; the router check above gives its size on the tiny run.",
]


def finish(out: str, checks: List[Dict[str, Any]], spec) -> int:
    ok = bool(checks) and all(c["ok"] for c in checks)
    rec = {"schema": "R3_parity/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "jpp_lens_commit": C.JPP_COMMIT, "tolerance": f"max |R3 - jpp| <= {TOL} * max(1, max |jpp|)",
           "spec": spec, "ok": ok, "checks": checks, "deviations": DEVIATIONS,
           "torch": torch.__version__}
    Path(out).write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    log(f"{'PARITY HOLDS' if ok else 'PARITY FAILS'}: {sum(c['ok'] for c in checks)}/{len(checks)} checks; wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
