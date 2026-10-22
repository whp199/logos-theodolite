#!/usr/bin/env python3
"""R3 04 (GPU, short): J++'s expert weights on the instruct model, and R3's lens files (D8, D11, D15).

1. The items. J++'s five macro evals (association, multihop, multilingual, poetry, typo; order-ops is outside J++'s
   macro) from the pinned checkout. Correctness (D8): one greedy step of the instruct model on the bare item prompt
   (BOS, no chat template), jpp_lens's next_token_matches_target against the item's correctness target; evals with
   no question (association, typo) are kept whole, as jpp_lens keeps their NA cells. Then the scoreable items (an
   intermediate with a one-token candidate), and jpp_lens's split: 20% of each eval fit, seed 0 (D15).
2. The per-eval floor, fixed before grading (R3_PREREG.md §2): an eval keeping fewer than 10 scoreable items
   (correct, or ungraded, and with a one-token candidate) or fewer than 2 fitting items is dropped from the macro.
   Association and typo have no correctness target, so they are never graded and always pass it. The primary macro
   is J++'s own Gemma convention, without poetry (koayon/jpp-lenses README); the full macro, poetry in, is a
   sensitivity analysis with its own weights. Two registered stops, each before any lens is written: fewer than 2
   graded evals (multihop, multilingual) left in the primary macro, or fewer than 20 fitting (item, intermediate)
   pairs in it.
3. The readout residuals (the block output at each scoreable item's readout position, code layers 30-58), and
   J++'s exclusion list (no letter or digit, over the LM head's width, sparing every candidate of all six evals).
4. Per layer, one at a time: the merged fp16 experts (R3_03), jpp_lens's own ExpertWeightingFitter with the
   registered CondenseConfig (signed unit-L2 weights, Adam, three starts, the exact macro pass@10 on the fitting
   pairs), and the combined lens J_l = sum_e w_e J_{l,e} in fp32. The same weights applied to each half's experts give
   the half-lenses; the pooled Jacobian (and each half's) is the R-lens lens without weights (rl-it).
5. Writes the lens files (bf16, the shipped J++'s layout {"J": {l: [5376, 5376]}, "source_layers", "d_model",
   "n_prompts", "provenance"}, read h @ J.T), R3_04_residuals.pt (gitignored, for R3_05's held-out recall) and
   R3_04_weights.json: per eval the counts at each step, which evals the floor kept, the fitting and held-out items as
   indices into the pinned item files (no item text or name is written), each layer's signed weights, their
   in-sample macro and start, and every lens file's sha256.

Runs only after R3_03's merge, the pilot's "proceed", the registered files committed, the server stopped:
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_04_weights.py --run
Every check before the model:  CUDA_VISIBLE_DEVICES="" ... R3_04_weights.py --check-only
Self-test (CPU, a tiny Gemma 4 and made-up items in J++'s format):  CUDA_VISIBLE_DEVICES="" ... --selftest
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--check-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # the CPU modes never see the GPU
sys.dont_write_bytecode = True

import argparse
import datetime
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_fitlib as FL  # noqa: E402
from R3_03_fit import load_merged  # noqa: E402

T0 = time.time()
RESIDUALS_PT = C.RESULTS / "R3_04_residuals.pt"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Amendment(Exception):
    """The registered stop: too few fitting pairs remain; no lens is written."""


def n_pairs(items: Sequence[FL.Item], tok) -> int:
    return sum(len(FL.item_candidates(tok, it)) for it in items)


def select_items(dec: FL.Decoder, raw: Dict[str, List[Dict[str, Any]]], graded: Dict[Tuple[str, str], Any], *,
                 floor: Tuple[int, int] = (C.EVAL_FLOOR_CORRECT, C.EVAL_FLOOR_FITTING)):
    """The item steps 1-2 above: (correct, scoreable, fitting, held-out, floors, counts). `floor` is the registered
    (10 correct, 2 fitting) except in R3_00's dry run, which grades six items per eval."""
    macro = FL.items_from_raw(raw, C.JPP_MACRO_EVALS)
    correct = [it for it in macro if graded[(it.eval_slug, it.name)] in (True, None)]
    scoreable = FL.scoreable(correct, dec.tokenizer)
    fitting, heldout = FL.split_items(scoreable)
    fk = dict(min_correct=floor[0], min_fitting=floor[1])
    floors = {"primary": FL.eval_floor(scoreable, fitting, C.PRIMARY_MACRO_EVALS, **fk),
              "full": FL.eval_floor(scoreable, fitting, C.JPP_MACRO_EVALS, **fk)}
    counts = {}
    for s in C.JPP_MACRO_EVALS:
        g = [graded[(it.eval_slug, it.name)] for it in macro if it.eval_slug == s]
        counts[s] = {"items": len(g), "graded_correct": sum(v is True for v in g),
                     "graded_wrong": sum(v is False for v in g), "ungradeable_kept": sum(v is None for v in g),
                     "scoreable": sum(it.eval_slug == s for it in scoreable),
                     "fitting": sum(it.eval_slug == s for it in fitting),
                     "heldout": sum(it.eval_slug == s for it in heldout),
                     "fitting_pairs": n_pairs([it for it in fitting if it.eval_slug == s], dec.tokenizer)}
    return correct, scoreable, fitting, heldout, floors, counts


def index_of(raw: Dict[str, List[Dict[str, Any]]], items: Sequence[FL.Item]) -> Dict[str, List[int]]:
    """Items as indices into their pinned files, per eval (what is committed instead of names)."""
    pos = {(s, it["name"]): i for s, its in raw.items() for i, it in enumerate(its)}
    out: Dict[str, List[int]] = {}
    for it in items:
        out.setdefault(it.eval_slug, []).append(pos[(it.eval_slug, it.name)])
    return out


def lens_payload(J: Dict[int, torch.Tensor], name: str, provenance: Dict[str, Any], n_prompts: int) -> Dict[str, Any]:
    return {"J": {int(l): m.to(torch.bfloat16).contiguous() for l, m in J.items()}, "source_layers": sorted(J),
            "d_model": int(next(iter(J.values())).shape[0]), "n_prompts": n_prompts,
            "provenance": dict(provenance, lens=name)}


def fit_all(dec: FL.Decoder, raw: Dict[str, List[Dict[str, Any]]], merged_dir: Path, layers: Sequence[int],
            out_paths: Dict[str, Path], residuals_path: Path, provenance: Dict[str, Any], *,
            condense: Optional[Dict[str, Any]] = None, min_dev_pairs: int = C.MIN_DEV_PAIRS,
            graded: Optional[Dict[Tuple[str, str], Any]] = None,
            floor: Tuple[int, int] = (C.EVAL_FLOOR_CORRECT, C.EVAL_FLOOR_FITTING),
            min_graded_evals: int = C.MIN_GRADED_EVALS) -> Dict[str, Any]:
    """Steps 1-5 on any model (the tiny one in the self-test). Returns the record R3_04_weights.json holds."""
    layers = sorted(layers)
    all_items = FL.items_from_raw(raw, C.JPP_EVALS)
    if graded is None:
        t = time.time()
        graded = FL.grade(dec, FL.items_from_raw(raw, C.JPP_MACRO_EVALS))
        log(f"graded {len(graded)} items in {time.time() - t:.1f}s")
    correct, scoreable, fitting, heldout, floors, counts = select_items(dec, raw, graded, floor=floor)
    rec: Dict[str, Any] = {"counts": counts, "floors": floors, "floor": list(floor), "condense": condense or C.CONDENSE,
                           "correct_by_eval": {s: [graded[(s, it["name"])] for it in raw[s]]
                                               for s in C.JPP_MACRO_EVALS},
                           "fitting_idx": index_of(raw, fitting), "heldout_idx": index_of(raw, heldout)}
    prim = [it for it in fitting if it.eval_slug in floors["primary"]["kept"]]
    full = [it for it in fitting if it.eval_slug in floors["full"]["kept"]]
    rec["dev_pairs"] = {"primary": n_pairs(prim, dec.tokenizer), "full": n_pairs(full, dec.tokenizer),
                        "primary_graded_evals": n_pairs([it for it in prim if it.eval_slug not in C.UNGRADED_EVALS],
                                                        dec.tokenizer),
                        "min": min_dev_pairs}
    graded_kept = [s for s in floors["primary"]["kept"] if s not in C.UNGRADED_EVALS]
    rec["graded_evals_kept"] = {"primary": graded_kept, "min": min_graded_evals,
                                "ungraded_always_kept": list(C.UNGRADED_EVALS)}
    if len(graded_kept) < min_graded_evals:
        rec["stopped"] = (f"{len(graded_kept)} graded evals ({graded_kept}) survive the floor in the primary macro, "
                          f"fewer than {min_graded_evals}: the weights would be fitted on the ungraded evals alone; "
                          f"stop for an amendment (R3_PREREG.md §9, D16)")
        raise Amendment(json.dumps(rec, default=str))
    if rec["dev_pairs"]["primary"] < min_dev_pairs:
        rec["stopped"] = (f"{rec['dev_pairs']['primary']} fitting pairs in the primary macro, fewer than "
                          f"{min_dev_pairs}: stop for an amendment (R3_PREREG.md §9)")
        raise Amendment(json.dumps(rec, default=str))
    excluded = FL.excluded_ids(dec.tokenizer, FL.vocab_size(dec), all_items)
    rec["excluded"] = {"n": len(excluded), "sha256": C.text_sha256(json.dumps(excluded))}
    t = time.time()
    res = FL.record_residuals(dec, scoreable, layers)
    log(f"residuals of {len(res.item_names)} items at {len(layers)} layers in {time.time() - t:.1f}s")
    torch.save({"residuals": res.residuals_L_dict_PN, "eval_slugs": res.eval_slugs, "item_names": res.item_names,
                "candidates": res.candidate_ids_P_list, "excluded": excluded,
                "fitting": [it.name for it in fitting], "heldout": [it.name for it in heldout]}, residuals_path)
    J: Dict[str, Dict[int, torch.Tensor]] = {k: {} for k in out_paths}
    rec["layers"] = {}
    for l in layers:
        m = load_merged(merged_dir, l)
        ex = m["experts_fp16"]
        lw = FL.fit_weights(dec, {l: ex}, res, excluded, [l], [it.name for it in prim],
                            condense).learned_weights_L_dict[l]
        lf = FL.fit_weights(dec, {l: ex}, res, excluded, [l], [it.name for it in full],
                            condense).learned_weights_L_dict[l]
        J["jpp-it"][l] = FL.combine(ex, lw.weights_E)
        J["jpp-it-full-macro"][l] = FL.combine(ex, lf.weights_E)
        J["rl-it"][l] = m["pooled"].float()
        for h in ("A", "B"):
            J[f"jpp-it-{h}"][l] = FL.combine(m["halves"][h]["experts_fp16"], lw.weights_E)
            J[f"rl-it-{h}"][l] = m["halves"][h]["pooled"].float()
        rec["layers"][str(l)] = {
            "weights": [round(float(x), 6) for x in lw.weights_E], "train_macro_pass": round(lw.train_macro_pass, 4),
            "init": lw.init_name, "step": lw.step,
            "full_macro": {"weights": [round(float(x), 6) for x in lf.weights_E],
                           "train_macro_pass": round(lf.train_macro_pass, 4), "init": lf.init_name, "step": lf.step},
            "fallback": m["fallback"].tolist(), "count": m["count"].tolist()}
        del m, ex
        log(f"  layer {l}: weights fitted (in-sample macro {lw.train_macro_pass:.3f}, {lw.init_name})")
    rec["lens_files"] = {}
    n_prompts = int(provenance.get("n_prompts", C.N_FIT))
    for name, path in out_paths.items():
        torch.save(lens_payload(J[name], name, provenance, n_prompts), path)
        rec["lens_files"][name] = {"path": str(path), "sha256": C.sha256_file(path)}
    rec["residuals_pt"] = {"path": str(residuals_path), "sha256": C.sha256_file(residuals_path)}
    return rec


def provenance_for(merged: Dict[str, Any]) -> Dict[str, Any]:
    return {"method": "jpp", "lrp_mode": C.LRP_MODE, "fitted_by": "R3 (experiments/scripts/R3_03_fit.py, "
            "R3_04_weights.py)", "jpp_lens_commit": C.JPP_COMMIT,
            "source_config": {"hf_model_name": C.MODEL_ID, "snapshot": C.MODEL_SNAPSHOT,
                              "source_layers": list(C.FIT_LAYERS), "target_layer": C.TARGET_LAYER,
                              "max_seq_len": C.MAX_SEQ_LEN, "skip_first_n_positions": C.SKIP_FIRST,
                              "num_clusters": C.N_EXPERTS, "pca_dim": C.PCA_DIM, "min_kept_positions": C.MIN_KEPT,
                              "capture_point": "block_output"},
            "fit_data": {"dataset": C.WIKITEXT_ID, "config": C.WIKITEXT_CONFIG, "revision": C.WIKITEXT_REVISION,
                         "n_fit_prompts": C.N_FIT, "n_router_records": C.N_ROUTER},
            "n_prompts": merged.get("n_prompts_fitted", C.N_FIT), "router_sha256": merged["settings"]["router_sha256"]}


# ------------------------------------------------------------------------------------------------ checks
def check_only() -> Dict[str, Any]:
    import R3_gpu as G

    out: Dict[str, Any] = {"failed": [], "pending": []}
    raw = C.jpp_items()
    out["items"] = {s: len(v) for s, v in raw.items()}
    names = [(s, it["name"]) for s, v in raw.items() for it in v]
    if len(set(n for _, n in names)) != len(names):
        out["failed"].append("item names are not unique across the evals (the fitter selects by name)")
    out["pilot"] = G.pilot_decision()
    if out["pilot"]["decision"] != "proceed":
        out["pending"].append(f"R3_00's pilot: {out['pilot']['why']}")
    mj = C.MERGED_DIR / "merged.json"
    if not mj.is_file():
        out["pending"].append(f"no {mj.relative_to(C.REPO)} yet (R3_03 --merge)")
    else:
        rec = json.loads(mj.read_text(encoding="utf-8"))
        missing = [l for l in C.FIT_LAYERS if str(l) not in rec["layers"]]
        if missing:
            out["failed"].append(f"merged.json lacks layers {missing}")
    existing = [p.name for p in C.LENS_FILES.values() if p.exists()]
    if existing:
        out["pending"].append(f"lens files exist ({existing}): a run would need --force")
    return out


# ------------------------------------------------------------------------------------------------ self-test
def selftest(record: bool = False) -> int:
    import R3_03_fit as F3
    from R3_parity import synthetic_items

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    torch.manual_seed(0)
    dec = FL.Decoder(C.build_tiny_gemma4(), C.CharTokenizer(), name="tiny")
    records = C.wikitext_records()[:24]
    layers, E, kw = [1, 2, 3], 4, dict(target=5, skip=16, max_seq_len=40)
    routers = FL.fit_router(lambda: FL.kept_points(dec, records, layers, skip=16, max_seq_len=40), layers,
                            n_experts=E, pca_dim=8)
    raw = synthetic_items(20)
    # an eval that the floor must drop: every multilingual item graded wrong
    graded_real = FL.grade(dec, FL.items_from_raw(raw, C.JPP_MACRO_EVALS))
    graded = {k: (False if k[0] == "multilingual" else (True if v is not None else None)) for k, v in graded_real.items()}
    check(all(v is None for k, v in graded_real.items() if k[0] in ("association", "typo"))
          and all(isinstance(v, bool) for k, v in graded_real.items() if k[0] in ("multihop", "multilingual",
                                                                                   "poetry")),
          "grading: a greedy step against the target for multihop, multilingual and poetry; none for association "
          "and typo (kept whole)")
    small = dict(C.CONDENSE, steps=20)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dirs = []
        for s, sl in enumerate(FL.shard_slices(8, 2)):
            d = tmp / f"shard{s}"
            F3.fit_shard_job(dec, records[:8][sl], sl.start, routers, layers, d, {"r": 1}, rows_per_pass=8,
                             mode="replicate", n_experts=E, **kw)
            dirs.append(d)
        F3.merge_all(dirs, layers, tmp / "merged", min_kept=5, expect_prompts=8)
        outs = {k: tmp / f"{k}.pt" for k in C.LENS_FILES}
        prov = {"method": "jpp", "lrp_mode": C.LRP_MODE, "n_prompts": 8, "source_config": {"hf_model_name": "tiny"}}
        try:
            fit_all(dec, raw, tmp / "merged", layers, {k: tmp / f"g{k}.pt" for k in C.LENS_FILES}, tmp / "g.pt",
                    prov, condense=small, graded=graded)
            check(False, "fewer than two graded evals surviving the floor stops the run for an amendment")
        except Amendment as e:
            st = json.loads(str(e))
            check(st["graded_evals_kept"]["primary"] == ["multihop"] and "graded evals" in st["stopped"]
                  and not any((tmp / f"g{k}.pt").exists() for k in C.LENS_FILES),
                  f"with multilingual dropped, one graded eval survives ({st['graded_evals_kept']['primary']}; "
                  f"association and typo always pass, ungraded): the run stops for an amendment, before any lens")
        # the rest of the self-test lets one graded eval through, to exercise the fit
        rec = fit_all(dec, raw, tmp / "merged", layers, outs, tmp / "res.pt", prov, condense=small, graded=graded,
                      min_graded_evals=1)
        fl = rec["floors"]
        check("multilingual" in fl["primary"]["dropped"] and "poetry" not in fl["primary"]["counts"]
              and "poetry" in fl["full"]["counts"] and rec["counts"]["multilingual"]["scoreable"] == 0,
              f"the per-eval floor drops an eval with no correct item (primary kept {fl['primary']['kept']}; the full "
              f"macro adds poetry: kept {fl['full']['kept']})")
        res_saved = torch.load(tmp / "res.pt", weights_only=False)
        res = FL.Residuals(res_saved["eval_slugs"], res_saved["item_names"], res_saved["residuals"],
                           res_saved["candidates"], "tiny", C.READOUT_MAX_SEQ_LEN)
        prim = [n for n, s in zip(res.item_names, res.eval_slugs) if s in fl["primary"]["kept"]
                and n in set(res_saved["fitting"])]
        experts = {l: load_merged(tmp / "merged", l)["experts_fp16"] for l in layers}
        at_once = FL.fit_weights(dec, experts, res, res_saved["excluded"], layers, prim, small)
        same = all([round(float(x), 6) for x in at_once.learned_weights_L_dict[l].weights_E]
                   == rec["layers"][str(l)]["weights"] for l in layers)
        check(same, "weights fitted one layer at a time (each layer's experts loaded alone) equal all layers at once")
        # D35: the 31B is loaded from the pinned local snapshot, so its config.name_or_path is C.MODEL_DIR while the
        # run names the decoder C.MODEL_ID; jpp_lens's identity check must accept that pairing and no other
        hf = dec._hf_model
        saved_name, saved_path = dec.name, getattr(hf.config, "name_or_path", "")
        one = {layers[0]: experts[layers[0]]}
        tiny_cfg = dict(small, steps=5)
        try:
            dec.name, hf.config.name_or_path = C.MODEL_ID, str(C.MODEL_DIR)
            try:
                FL.fit_weights(dec, one, res, res_saved["excluded"], [layers[0]], prim, tiny_cfg)
                ok_pinned = True
            except ValueError:
                ok_pinned = False
            hf.config.name_or_path = str(C.MODEL_DIR.parent / "another-snapshot")
            try:
                FL.fit_weights(dec, one, res, res_saved["excluded"], [layers[0]], prim, tiny_cfg)
                ok_other = False
            except ValueError:
                ok_other = True
        finally:
            dec.name, hf.config.name_or_path = saved_name, saved_path
        check(ok_pinned and ok_other, "the weight fit accepts the 31B as loaded (name_or_path the pinned snapshot "
                                      "directory, the decoder named C.MODEL_ID) and refuses any other snapshot (D35)")
        lens = torch.load(outs["jpp-it"], weights_only=True)
        ok = sorted(lens["J"]) == layers and all(lens["J"][l].dtype == torch.bfloat16 for l in layers)
        ok &= lens["provenance"]["lens"] == "jpp-it" and lens["provenance"]["lrp_mode"] == "rlens"
        h = torch.randn(5, dec.d_model)
        for l in layers:
            w = at_once.learned_weights_L_dict[l].weights_E
            want = sum(float(w[e]) * (h @ experts[l][e].float().T) for e in range(E))
            got = h @ lens["J"][l].float().T
            ok &= float((got - want).abs().max()) <= 2e-2 * float(want.abs().max())
        check(ok, "the jpp-it file: bf16 {'J': {layer: [N, N]}} with provenance, read h @ J.T, equals the weighted "
                  "sum of the experts' transports (to bf16 rounding)")
        ok = True
        for l in layers:
            mp = FL.macro_pass(dec, res, l, FL.combine(experts[l], at_once.learned_weights_L_dict[l].weights_E),
                               [res.item_names.index(n) for n in prim], res_saved["excluded"])
            ok &= abs(mp["macro"] - at_once.learned_weights_L_dict[l].train_macro_pass) <= 1e-6
        check(ok, "R3_fitlib.macro_pass (J++'s readout eval, R3_05's held-out recall) gives the fitter's own "
                  "in-sample macro for the combined lens at every layer")
        rl = torch.load(outs["rl-it"], weights_only=True)
        mg = load_merged(tmp / "merged", layers[0])
        check(torch.equal(rl["J"][layers[0]], mg["pooled"].to(torch.bfloat16))
              and all(outs[k].is_file() for k in C.LENS_FILES)
              and rec["lens_files"]["jpp-it-A"]["sha256"] != rec["lens_files"]["jpp-it-B"]["sha256"],
              "rl-it is the pooled Jacobian; all seven lens files are written (the halves differ)")
        js = json.dumps(rec, default=str)
        check(not any(f'"{it["name"]}"' in js for v in raw.values() for it in v),
              "the record holds item indices and counts, never an item's name or text")
        try:
            fit_all(dec, raw, tmp / "merged", layers, {k: tmp / f"z{k}.pt" for k in C.LENS_FILES}, tmp / "z.pt",
                    prov, condense=small, graded=graded, min_dev_pairs=10 ** 6, min_graded_evals=1)
            check(False, "too few fitting pairs stops the run for an amendment, before any lens is written")
        except Amendment:
            check(not any((tmp / f"z{k}.pt").exists() for k in C.LENS_FILES),
                  "too few fitting pairs stops the run for an amendment, before any lens is written")
    co = check_only()
    check(not co["failed"] and co["items"] == {"multihop": 93, "multilingual": 107, "order-ops": 55, "poetry": 98,
                                                "association": 102, "typo": 96},
          f"--check-only: the six pinned item files ({sum(co['items'].values())} items, names unique); waiting for: "
          f"{co['pending']}")
    print(f"R3_04_weights self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the run
def run(args) -> int:
    import R3_gpu as G

    existing = [str(p) for p in C.LENS_FILES.values() if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"{existing} exist; --force to refit (the earlier run stays in the runs log)")
    G.require_pilot_proceed()
    merged = json.loads((C.MERGED_DIR / "merged.json").read_text(encoding="utf-8"))
    with G.gpu_job("R3_04_weights", sys.argv, args.allow_uncommitted, log) as job:
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        raw = C.jpp_items()
        try:
            rec = fit_all(dec, raw, C.MERGED_DIR, C.FIT_LAYERS, C.LENS_FILES, RESIDUALS_PT, provenance_for(merged))
        except Amendment as e:
            rec = json.loads(str(e))
            log(rec["stopped"])
        rec.update({"schema": "R3_04/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "git": job["git"], "cap": job["cap"], "merged_json_sha256": C.sha256_file(C.MERGED_DIR / "merged.json")})
        C.WEIGHTS_JSON.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(C.WEIGHTS_JSON)
    log(f"wrote {C.WEIGHTS_JSON}")
    return 1 if "stopped" in rec else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="the GPU run (server stopped)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.record)
    if a.check_only:
        co = check_only()
        print(json.dumps(co, indent=1, default=str))
        print("check-only:", "FAILED" if co["failed"] else ("PASS, waiting for: " + "; ".join(co["pending"])
                                                            if co["pending"] else "PASS"))
        return 1 if co["failed"] else 0
    if a.run:
        return run(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
