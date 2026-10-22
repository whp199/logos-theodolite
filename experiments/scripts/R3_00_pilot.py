#!/usr/bin/env python3
"""R3 00 (GPU): the dry run on the local Gemma 4 12B, then the pilot on the served 31B, before any fit (R3_PREREG.md
§10). Neither reads a lens on an evaluation prompt: every text read here is a WikiText fit record.

--dry-run-12b (GPU, bf16 12B, blocks below the device map's first GPU block on the CPU, the job cap held): the whole
  pipeline once, at three source layers, on R3's own code: the LRP coverage report, R3_02's resumable router on 16
  records, two R3_03 shard jobs of one prompt each and the streamed merge, R3_04's grading, split, floor (1 and 1
  here), residuals and weights on ten items of each eval, a readout of one record through the new lens with R2's
  filter, R3_Q1's direction mode, and the parity check below. Writes R3_00_dry_run.json. The pilot refuses to start
  without it.
--pilot (GPU, the 31B; at most 30 minutes; on WikiText fit prompts 0 and 1):
  (a) LRP coverage (the patched module names, counted) and the forward under the rules: the top-1 word-piece of
      every position must be unchanged at >= 99.9% of positions.
  (b) parity: with the rules off, R3's batched rows for six output dimensions (R1's "unit" directions) must equal
      an independent backward, one direction at a time from the graph rooted at block 0 (R3_gpu), at cos >= 0.99
      at every fitted layer, in both batching modes; with the rules on it is recorded too.
  (c) memory and seconds per backward pass for is_grads_batched passes at B in {8, 16, 32, 64, 128} and for a
      replicated batch at B in {1, 2, 3, 4}, each under the job cap (an out-of-memory error is a result, not a
      failure); the forward with and without the graph; the Q1 direction mode.
  (d) the choice (the fastest configuration per prompt) and the projection of GPU-hours for every R3 GPU job
      (shards, router, weights, Q1, H4's states, the evaluation), times a registered margin of 1.25, against
      R3_common.APPROVED_GPU_HOURS. A projection over the approved budget, a budget not yet approved, or a failed
      check (a) or (b) writes "stop" (an amendment is needed); only "proceed" lets R3_02-R3_05 and Q1 run.
  Writes R3_00_pilot.json, and the card's memory before and after (nvidia-smi and torch).
Bounds stated beside the projection: about 1.29e18 FLOP for 64 prompts x 5376 rows x 128 tokens through blocks 30-59
(3.6-14 h at a sustained 25-100 TFLOP/s), and rows one at a time at R1's 0.79 s per backward pass (75 h).

Every check before the model (CPU):
  CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_00_pilot.py --check-only
Self-test (CPU: the pilot's measurement, the projection and decision branches and the dry-run pipeline on a tiny
Gemma 4 with genuine 4-bit linears):  CUDA_VISIBLE_DEVICES="" ... R3_00_pilot.py --selftest
GPU runs (server stopped, in this order):
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_00_pilot.py --dry-run-12b
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_00_pilot.py --pilot
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
import math
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_fitlib as FL  # noqa: E402
import R3_gpu as G  # noqa: E402
import R3_lrp  # noqa: E402

T0 = time.time()
MARGIN = 1.25                                          # registered safety factor on the projection
DRY_LAYERS = (24, 36, 46)                              # the 12B's source layers (of 48); its target is block 47
DRY_ITEMS_PER_EVAL = 10                                # 2 fitting items per eval; the dry run's floor is 1 and 1


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def _cuda(dev) -> bool:
    return torch.cuda.is_available() and torch.device(dev).type == "cuda"


def _peak(dev) -> Optional[float]:
    return round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3) if _cuda(dev) else None


def _reset(dev) -> None:
    if _cuda(dev):
        G.peak_reset(torch.device(dev))


def _sync(dev) -> None:
    if _cuda(dev):
        torch.cuda.synchronize(dev)


def _oom_types() -> Tuple[type, ...]:
    return (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)


# ------------------------------------------------------------------------------------------------ (a) the rules
@torch.no_grad()
def lrp_identity(dec: FL.Decoder, texts: Sequence[str], max_seq_len: int = C.MAX_SEQ_LEN) -> Dict[str, Any]:
    """The share of positions whose top-1 word-piece is the same with and without the rules in the forward."""
    same = total = 0
    last = dec.n_layers - 1
    for t in texts:
        _, a = FL.block_outputs(dec, t, [last], max_seq_len)
        with R3_lrp.rlens(dec._hf_model):
            _, b = FL.block_outputs(dec, t, [last], max_seq_len)
        ta, tb = dec.unembed(a[last]).argmax(-1), dec.unembed(b[last]).argmax(-1)
        same += int((ta == tb).sum())
        total += int(ta.numel())
    return {"positions": total, "same_top1": same, "share": round(same / max(1, total), 6)}


def coverage(hf) -> Dict[str, Any]:
    rep = R3_lrp.plan(hf)
    return {**rep.summary(), "ln_rule_names_sha256": C.text_sha256(json.dumps(rep.ln_rule)),
            "mlp_rule_names_sha256": C.text_sha256(json.dumps(rep.mlp_rule)),
            "first_names": {"ln_rule": rep.ln_rule[:5], "mlp_rule": rep.mlp_rule[:3],
                            "attention_norms_kept": rep.attention_norms_kept[:3]}}


# ------------------------------------------------------------------------------------------------ (b) parity
def parity(dec: FL.Decoder, text: str, layers: Sequence[int], dims: Sequence[int], *, linears_ctx, lrp: bool,
           target: int, skip: int, max_seq_len: int) -> Dict[str, Any]:
    """Per batching mode, the cosine per layer and dimension of R3's batched rows (dims_rows) against the
    independent single-direction backward (R3_gpu.reference_direction_covectors); its minimum."""
    layers = sorted(layers)
    E = torch.zeros(len(dims), dec.d_model)
    E[torch.arange(len(dims)), torch.as_tensor(list(dims))] = 1.0
    ref = G.reference_direction_covectors(dec, text, E, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                          lrp=lrp, linears_ctx=linears_ctx())
    out: Dict[str, Any] = {"dims": list(dims), "lrp": lrp}
    for mode in ("replicate", "vmap"):
        with linears_ctx():
            fw = FL.fit_forward(dec, text, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                rows_per_pass=len(dims), mode=mode, lrp=lrp)
            rows = FL.dims_rows(fw, dims, keep=False)
        del fw
        cos = torch.stack([torch.nn.functional.cosine_similarity(rows[i].cpu().float(), ref[:, i].float(), dim=-1)
                           for i in range(len(layers))])                 # [L, dims]
        out[mode] = {"min_cos": round(float(cos.min()), 6), "per_layer_min": [round(float(x), 6)
                                                                             for x in cos.min(1).values]}
    out["min_cos"] = min(out["replicate"]["min_cos"], out["vmap"]["min_cos"])
    return out


# ------------------------------------------------------------------------------------------------ (c) timing
def time_config(dec: FL.Decoder, text: str, layers: Sequence[int], mode: str, B: int, *, linears_ctx, passes: int,
                target: int, skip: int, max_seq_len: int, lrp: bool = True) -> Dict[str, Any]:
    dev = dec.input_device
    rec: Dict[str, Any] = {"mode": mode, "B": B}
    _reset(dev)
    try:
        t = time.time()
        with linears_ctx():
            fw = FL.fit_forward(dec, text, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                rows_per_pass=B, mode=mode, lrp=lrp)
            _sync(dev)
            rec["forward_s"] = round(time.time() - t, 3)
            rec["peak_after_forward_gib"] = _peak(dev)
            secs = []
            gen = FL.backward_rows(fw, B)
            for _ in range(passes):
                t = time.time()
                ds, nb, rows = next(gen)
                _sync(dev)
                secs.append(time.time() - t)
                del rows
            del gen, fw
        rec.update(ok=True, s_per_pass=round(statistics.median(secs), 4), passes_timed=len(secs),
                   peak_gib=_peak(dev), s_per_row=round(statistics.median(secs) / B, 5))
    except _oom_types() as e:
        rec.update(ok=False, oom=True, error=f"{type(e).__name__}: {str(e)[:160]}", peak_gib=_peak(dev))
    finally:
        if _cuda(dev):
            torch.cuda.empty_cache()
    return rec


def time_q1(dec: FL.Decoder, text: str, layers: Sequence[int], dirs: torch.Tensor, *, linears_ctx, target, skip,
            max_seq_len) -> Dict[str, Any]:
    dev = dec.input_device
    _reset(dev)
    try:
        t = time.time()
        with linears_ctx():
            fw = FL.fit_forward(dec, text, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                rows_per_pass=C.Q1_DIM_BATCH, mode="vmap", lrp=False)
            _sync(dev)
            tf = time.time() - t
            t = time.time()
            FL.direction_rows(fw, dirs.to(fw.target.device), dim_batch=C.Q1_DIM_BATCH)
            _sync(dev)
        return {"ok": True, "forward_s": round(tf, 3), "directions_s": round(time.time() - t, 3),
                "n_directions": int(dirs.shape[0]), "peak_gib": _peak(dev)}
    except _oom_types() as e:
        return {"ok": False, "oom": True, "error": f"{type(e).__name__}: {str(e)[:160]}", "peak_gib": _peak(dev)}
    finally:
        if _cuda(dev):
            torch.cuda.empty_cache()


@torch.no_grad()
def time_forward(dec: FL.Decoder, text: str, layers: Sequence[int], max_seq_len: int) -> float:
    dev = dec.input_device
    _sync(dev)
    t = time.time()
    FL.block_outputs(dec, text, layers, max_seq_len)
    _sync(dev)
    return round(time.time() - t, 4)


# ------------------------------------------------------------------------------------------------ (d) the projection
def choose(table: Sequence[Dict[str, Any]], d_model: int, cap_gib: float) -> Optional[Dict[str, Any]]:
    """The configuration with the least time per prompt (forward + ceil(d / B) passes), among those that ran within
    the cap."""
    ok = [r for r in table if r.get("ok") and (r.get("peak_gib") is None or r["peak_gib"] <= cap_gib)]
    if not ok:
        return None
    for r in ok:
        r["s_per_prompt"] = round(r["forward_s"] + math.ceil(d_model / r["B"]) * r["s_per_pass"], 2)
    best = min(ok, key=lambda r: r["s_per_prompt"])
    return {"mode": best["mode"], "rows_per_pass": best["B"], "s_per_prompt": best["s_per_prompt"]}


def project(choice: Dict[str, Any], t: Dict[str, float], *, n_fit: int = C.N_FIT, n_router: int = C.N_ROUTER,
            n_items: int, n_q1_prompts: int, n_h4_seqs: int, r2_eval_s: float, n_r3_prompts: int = 320,
            n_r2_prompts: int = 220) -> Dict[str, Any]:
    """GPU-hours per job from the measured seconds: shards 64 x s_per_prompt; the router two forward passes per
    record; weights two forwards per item (grading and residuals); Q1 a forward with the graph and 4 direction
    batches per prompt; H4 one forward per held-out sequence; the evaluation R2's measured run scaled by the prompt
    count (R3 reads 320 prompts and R2's 220 again, at 30 layers against R2's 20, with three more lenses). Total times
    MARGIN."""
    h = {
        "shards": n_fit * choice["s_per_prompt"] / 3600,
        "router": 2 * n_router * t["forward_nograd_s"] / 3600,
        "weights": 2 * n_items * t["forward_nograd_s"] / 3600,
        "q1": n_q1_prompts * (t["q1_forward_s"] + t["q1_directions_s"]) / 3600,
        "h4_states": n_h4_seqs * t["forward_nograd_long_s"] / 3600,
        "evaluation": r2_eval_s * (n_r3_prompts + n_r2_prompts) / n_r2_prompts * 1.5 / 3600,
    }
    total = sum(h.values()) * MARGIN
    return {"hours": {k: round(v, 3) for k, v in h.items()}, "margin": MARGIN, "total_hours": round(total, 2)}


def decide(proj: Optional[Dict[str, Any]], checks: Dict[str, bool], approved: Optional[float]) -> Dict[str, Any]:
    failed = [k for k, v in checks.items() if not v]
    base = {"approved_gpu_hours": approved, "projected_gpu_hours": proj["total_hours"] if proj else None,
            "checks": checks}
    if failed:
        return dict(base, decision="stop", why=f"pilot checks failed: {failed}; an amendment is needed")
    if proj is None:
        return dict(base, decision="stop", why="no batching configuration ran within the job cap")
    if approved is None:
        return dict(base, decision="stop", why=(f"projected {proj['total_hours']} GPU-hours; no budget is approved yet "
                                                "(R3_common.APPROVED_GPU_HOURS [NOT YET APPROVED])"))
    if proj["total_hours"] > approved:
        return dict(base, decision="stop", why=(f"projected {proj['total_hours']} GPU-hours, over the approved "
                                                f"{approved}: stop for an amendment before any fit"))
    return dict(base, decision="proceed", why=f"projected {proj['total_hours']} of {approved} approved GPU-hours")


def pilot_measure(dec: FL.Decoder, texts: Sequence[str], *, layers: Sequence[int], target: int, skip: int,
                  max_seq_len: int, linears_ctx: Callable[[], Any], parity_dims: Sequence[int], q1_dirs: torch.Tensor,
                  vmap_B: Sequence[int], replicate_B: Sequence[int], passes: int, budget_s: float,
                  cap_gib: float) -> Dict[str, Any]:
    """(a)-(c) and the choice, on any model (the tiny one in the self-test)."""
    t_start = time.time()
    kw = dict(target=target, skip=skip, max_seq_len=max_seq_len)
    out: Dict[str, Any] = {"coverage": coverage(dec._hf_model), "lrp_identity": lrp_identity(dec, texts, max_seq_len)}
    log(f"(a) rules: {out['coverage']['ln_rule']} norms, {out['coverage']['mlp_rule']} MLPs; top-1 unchanged at "
        f"{out['lrp_identity']['share']:.4%} of {out['lrp_identity']['positions']} positions")
    out["parity"] = {"plain": parity(dec, texts[0], layers, parity_dims, linears_ctx=linears_ctx, lrp=False, **kw),
                     "rlens": parity(dec, texts[0], layers, parity_dims, linears_ctx=linears_ctx, lrp=True, **kw)}
    log(f"(b) parity: min cos plain {out['parity']['plain']['min_cos']:.5f}, R-lens {out['parity']['rlens']['min_cos']:.5f}")
    table, skipped = [], []
    for mode, Bs in (("replicate", replicate_B), ("vmap", vmap_B)):
        for B in Bs:
            if time.time() - t_start > budget_s:
                skipped.append({"mode": mode, "B": B, "why": "the pilot's time budget was spent"})
                continue
            r = time_config(dec, texts[1 % len(texts)], layers, mode, B, linears_ctx=linears_ctx, passes=passes, **kw)
            table.append(r)
            log(f"(c) {mode} B={B}: " + (f"{r['s_per_pass']} s/pass, peak {r['peak_gib']} GiB" if r.get("ok")
                                        else f"out of memory (peak {r.get('peak_gib')} GiB)"))
    out["timing"] = table
    out["timing_skipped"] = skipped
    out["q1"] = time_q1(dec, texts[0], layers, q1_dirs, linears_ctx=linears_ctx, **kw)
    out["forward_nograd_s"] = time_forward(dec, texts[0], layers, max_seq_len)
    out["choice"] = choose(table, dec.d_model, cap_gib)
    out["seconds"] = round(time.time() - t_start, 1)
    return out


# ------------------------------------------------------------------------------------------------ the dry run
def dry_pipeline(dec: FL.Decoder, *, layers: Sequence[int], target: int, router_texts: Sequence[str],
                 fit_texts: Sequence[str], raw_items: Dict[str, List[Dict[str, Any]]], readout_text: str,
                 content_mask: Optional[torch.Tensor], linears_ctx: Callable[[], Any], n_experts: int, pca_dim: int,
                 rows_per_pass: int, condense: Dict[str, Any], tmp: Path, skip: int = C.SKIP_FIRST,
                 max_seq_len: int = C.MAX_SEQ_LEN) -> Dict[str, Any]:
    """Every R3 stage once, on R3's own code, in tmp; each stage timed, with its peak memory."""
    import R3_02_router as R2r
    import R3_03_fit as R3f
    import R3_04_weights as R4w
    import R3_Q1_jlens as Q1

    dev = dec.input_device
    st: Dict[str, Any] = {}

    def stage(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        _reset(dev)
        t = time.time()
        try:
            r = dict(fn() or {}, ok=True)
        except Exception as e:  # noqa: BLE001 - every stage's failure is recorded, then the run fails
            r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r.update(seconds=round(time.time() - t, 2), peak_gib=_peak(dev))
        st[name] = r
        log(f"dry run: {name} {'ok' if r['ok'] else 'FAILED ' + r['error']} ({r['seconds']}s)")
        return r

    stage("lrp_coverage", lambda: coverage(dec._hf_model))
    holder: Dict[str, Any] = {}

    def router():
        with torch.no_grad():
            holder["routers"], rec = R2r.fit_router_resumable(dec, router_texts, layers, tmp / "router_ckpt.pt",
                                                              every=4, n_experts=n_experts, pca_dim=pca_dim,
                                                              skip=skip, max_seq_len=max_seq_len)
        return {"cluster_sizes": rec["cluster_sizes"], "skipped": rec["skipped_records"]}
    stage("router", router)

    def shards():
        dirs = []
        for s, t in enumerate(fit_texts):
            d = tmp / "shards" / f"shard{s}"
            R3f.fit_shard_job(dec, [t], s, holder["routers"], layers, d, {"dry": True}, rows_per_pass=rows_per_pass,
                              mode="vmap", lrp=True, linears_ctx=linears_ctx, n_experts=n_experts, target=target,
                              skip=skip, max_seq_len=max_seq_len)
            dirs.append(d)
        rec = R3f.merge_all(dirs, layers, tmp / "merged", expect_prompts=len(fit_texts))
        return {"fallback": {l: v["fallback"] for l, v in rec["layers"].items()}}
    stage("shards_and_merge", shards)

    def weights():
        sub = {s: v[:DRY_ITEMS_PER_EVAL] for s, v in raw_items.items()}
        outs = {k: tmp / f"{k}.pt" for k in C.LENS_FILES}
        rec = R4w.fit_all(dec, sub, tmp / "merged", layers, outs, tmp / "res.pt",
                          {"method": "jpp", "lrp_mode": C.LRP_MODE, "n_prompts": len(fit_texts), "dry_run": True},
                          condense=condense, min_dev_pairs=1, floor=(1, 1))
        return {"counts": rec["counts"], "floors": {k: v["kept"] for k, v in rec["floors"].items()},
                "train_macro_pass": {l: v["train_macro_pass"] for l, v in rec["layers"].items()}}
    stage("weights", weights)

    @torch.no_grad()
    def readout():
        lens = torch.load(tmp / "jpp-it.pt", weights_only=True)
        ids, outs = FL.block_outputs(dec, readout_text, layers, max_seq_len)
        top = {}
        for l in layers:
            h = outs[l][-1:].float()
            lg = dec.unembed(h @ lens["J"][l].to(h.device).float().T)[0].float()
            if content_mask is not None:
                lg = lg.masked_fill(~content_mask.to(lg.device), float("-inf"))
            top[l] = torch.topk(lg, 5).indices.tolist()
        return {"filtered": content_mask is not None, "top5_ids": top}
    stage("readout", readout)

    def q1():
        g = torch.Generator().manual_seed(0)
        dirs = torch.randn(4, dec.d_model, generator=g)
        s = Q1.fit_set(dec, fit_texts[:1], dirs, layers, tmp / "q1.pt", dim_batch=2, linears_ctx=linears_ctx,
                       target=target, skip=skip, max_seq_len=max_seq_len)
        return {"n": s["n"]}
    stage("q1_directions", q1)
    stage("parity", lambda: parity(dec, fit_texts[0], layers, [0, 1, 2], linears_ctx=linears_ctx, lrp=False,
                                   target=target, skip=skip, max_seq_len=max_seq_len))
    st["ok"] = all(v.get("ok") for v in st.values() if isinstance(v, dict)) and st["parity"].get("min_cos", 0) >= \
        C.PILOT_PARITY_COS_MIN
    return st


# ------------------------------------------------------------------------------------------------ checks
def check_only() -> Dict[str, Any]:
    out: Dict[str, Any] = {"failed": [], "pending": []}
    recs = C.wikitext_records()
    out["pilot_prompts"] = {"n": C.PILOT_N_PROMPTS, "sha256": [C.text_sha256(t) for t in recs[:C.PILOT_N_PROMPTS]]}
    if not (C.MODEL_DIR / "config.json").is_file():
        out["failed"].append(f"no model snapshot at {C.MODEL_DIR}")
    try:
        dm = G.dry_device_map(must_be_on_gpu_from=min(DRY_LAYERS))
        out["dry_run_device_map"] = {k: v for k, v in dm.items() if k != "device_map"}
        if not dm["covered"]:
            out["failed"].append("the dry run's device map leaves weights unplaced")
    except SystemExit as e:
        out["failed"].append(f"the dry run's device map: {e}")
    except FileNotFoundError:
        out["failed"].append(f"no dry-run checkpoint at {C.DRY_RUN_MODEL_DIR}")
    import transformers
    from transformers import AutoConfig

    for name, path in (("31B", C.MODEL_DIR), ("12B", C.DRY_RUN_MODEL_DIR)):
        cfg = AutoConfig.from_pretrained(str(path), local_files_only=True)
        with torch.device("meta"):
            m = getattr(transformers, cfg.architectures[0])(cfg)
        try:
            out[f"lrp_plan_{name}"] = R3_lrp.plan(m).summary()
        except RuntimeError as e:
            out["failed"].append(f"the R-lens rules cannot patch the {name}: {e}")
        del m
    q1 = C.check_pinned_file(C.R1_DIRS_PATH, C.R1_DIRS_SHA256, C.R1_DIRS_SIZE, full=False)
    if not q1["size_ok"]:
        out["failed"].append("R1_02_dirs.pt (the parity dimensions and Q1's directions) is missing")
    out["approved_gpu_hours"] = C.APPROVED_GPU_HOURS
    if C.APPROVED_GPU_HOURS is None:
        out["pending"].append("no GPU budget approved yet (R3_common.APPROVED_GPU_HOURS): the pilot would write stop")
    if not C.DRY_RUN_JSON.is_file():
        out["pending"].append("the 12B dry run has not run (R3_00_dry_run.json)")
    r2 = json.loads((C.RESULTS / "R2_02_eval.json").read_text(encoding="utf-8"))["meta"]["timing_s"]["total"]
    out["r2_eval_seconds"] = r2
    return out


def parity_dims_from_r1() -> List[int]:
    """R1's "unit" directions (the highest-energy output dimensions at layer 59, R1_02_prep.py): the first six."""
    DD = torch.load(C.R1_DIRS_PATH, weights_only=False, map_location="cpu")
    return [int(n[4:]) for n in DD["names"] if n.startswith("unit")][:C.PILOT_PARITY_DIMS]


# ------------------------------------------------------------------------------------------------ self-test
def selftest(record: bool = False) -> int:
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
    hf = C.build_tiny_gemma4(seed=4)
    nq = G.quantise_tiny(hf)
    dec = FL.Decoder(hf, C.CharTokenizer(), name="tiny-w4a16")
    recs = C.wikitext_records()[:20]
    layers, kw = [1, 2, 3], dict(target=5, skip=16, max_seq_len=40)
    g = torch.Generator().manual_seed(1)
    pm = pilot_measure(dec, recs[:2], layers=layers, linears_ctx=lambda: G.grad_linears(hf), parity_dims=[0, 7, 31],
                       q1_dirs=torch.randn(8, 32, generator=g), vmap_B=(8, 16), replicate_B=(1, 2), passes=2,
                       budget_s=600, cap_gib=C.JOB_CAP_GB, **kw)
    check(pm["coverage"]["ln_rule"] == 25 and pm["coverage"]["mlp_rule"] == 6 and pm["lrp_identity"]["share"] == 1.0,
          f"(a) on a tiny Gemma 4 with {nq} genuine 4-bit linears: the rules' coverage (25 norms, 6 MLPs) and the "
          f"forward's top-1 unchanged at every position ({pm['lrp_identity']['positions']})")
    check(pm["parity"]["plain"]["min_cos"] >= 0.99999 and pm["parity"]["rlens"]["min_cos"] >= 0.99999,
          f"(b) parity through the recompute-in-backward linears, both modes: min cos plain "
          f"{pm['parity']['plain']['min_cos']}, R-lens {pm['parity']['rlens']['min_cos']} (>= 0.99 on the GPU)")
    check(len(pm["timing"]) == 4 and all(r["ok"] and r["s_per_pass"] > 0 for r in pm["timing"])
          and pm["q1"]["ok"] and pm["choice"] is not None,
          f"(c) the timing table runs every configuration (CPU seconds here; memory only on the GPU); the choice: "
          f"{pm['choice']}")
    # (d) the projection and every decision branch
    choice = {"mode": "vmap", "rows_per_pass": 32, "s_per_prompt": 600.0}
    t = {"forward_nograd_s": 0.5, "q1_forward_s": 1.0, "q1_directions_s": 3.0, "forward_nograd_long_s": 1.0}
    pr = project(choice, t, n_items=450, n_q1_prompts=64, n_h4_seqs=64, r2_eval_s=369.7)
    want = (64 * 600 + 2 * 1000 * 0.5 + 2 * 450 * 0.5 + 64 * 4.0 + 64 * 1.0 + 369.7 * 540 / 220 * 1.5) / 3600 * 1.25
    check(abs(pr["total_hours"] - round(want, 2)) < 1e-9, f"(d) the projection adds every job and the margin "
                                                          f"({pr['total_hours']} h for 600 s per prompt)")
    ok_checks = {"lrp_identity": True, "parity": True}
    d0 = decide(pr, ok_checks, None)
    d1 = decide(pr, ok_checks, 5.0)
    d2 = decide(pr, ok_checks, 24.0)
    d3 = decide(pr, {"lrp_identity": True, "parity": False}, 24.0)
    d4 = decide(None, ok_checks, 24.0)
    check(d0["decision"] == "stop" and "NOT YET APPROVED" in d0["why"] and d1["decision"] == "stop"
          and d2["decision"] == "proceed" and d3["decision"] == "stop" and d4["decision"] == "stop",
          "(d) the decision: stop with no approved budget, stop over budget, proceed within it, stop on a failed "
          "check or when nothing ran within the cap")
    tbl = [{"ok": True, "mode": "replicate", "B": 2, "forward_s": 1.0, "s_per_pass": 1.0, "peak_gib": 10},
           {"ok": True, "mode": "vmap", "B": 64, "forward_s": 1.0, "s_per_pass": 20.0, "peak_gib": 30},
           {"ok": True, "mode": "vmap", "B": 32, "forward_s": 1.0, "s_per_pass": 4.0, "peak_gib": 20},
           {"ok": False, "oom": True, "mode": "vmap", "B": 128, "peak_gib": 24}]
    ch = choose(tbl, 5376, 24.0)
    check(ch == {"mode": "vmap", "rows_per_pass": 32, "s_per_prompt": 1.0 + 168 * 4.0},
          "the choice: the least time per prompt among configurations that ran within the cap (an over-cap or "
          "out-of-memory one is never chosen)")
    try:
        G.require_pilot_proceed()
        check(False, "with no pilot result, every fit job refuses to start")
    except SystemExit:
        check(True, "with no pilot result, every fit job refuses to start")
    # the dry-run pipeline on the tiny model (made-up items in J++'s format; no filter: a character tokenizer)
    with tempfile.TemporaryDirectory() as tmp:
        st = dry_pipeline(dec, layers=layers, target=5, router_texts=recs[:16], fit_texts=recs[16:18],
                          raw_items=synthetic_items(20), readout_text=recs[18], content_mask=None,
                          linears_ctx=lambda: G.grad_linears(hf), n_experts=4, pca_dim=8, rows_per_pass=8,
                          condense=dict(C.CONDENSE, steps=10), tmp=Path(tmp), skip=16, max_seq_len=40)
    names = [k for k in st if isinstance(st[k], dict)]
    check(st["ok"] and names == ["lrp_coverage", "router", "shards_and_merge", "weights", "readout", "q1_directions",
                                 "parity"],
          f"the dry-run pipeline runs every stage on R3's own code through the 4-bit linears: {names}")
    co = check_only()
    check(not co["failed"] and co["lrp_plan_31B"]["ln_rule"] == 241 and co["lrp_plan_12B"]["ln_rule"] == 193,
          f"--check-only: the pilot's prompts, the 12B's device map and both module trees; waiting for: "
          f"{co['pending']}")
    print(f"R3_00_pilot self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the GPU runs
def run_dry(args) -> int:
    with G.gpu_job("R3_00 dry run (12B)", sys.argv, args.allow_uncommitted, log) as job:
        hf, tok, plan = G.load_dry_model(DRY_LAYERS)
        dec = FL.Decoder(hf, tok, name=str(C.DRY_RUN_MODEL_DIR))
        recs = C.wikitext_records()
        mask = C.R2.content_mask(tok)
        if len(mask) != FL.vocab_size(dec):
            mask = None
        with tempfile.TemporaryDirectory(dir=C.RESULTS) as tmp:
            st = dry_pipeline(dec, layers=DRY_LAYERS, target=dec.n_layers - 1, router_texts=recs[:16],
                              fit_texts=recs[:2], raw_items=C.jpp_items(), readout_text=recs[2], content_mask=mask,
                              linears_ctx=nullcontext, n_experts=C.N_EXPERTS, pca_dim=C.PCA_DIM, rows_per_pass=32,
                              condense=dict(C.CONDENSE, steps=20), tmp=Path(tmp))
        out = {"schema": "R3_00_dry/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "model": str(C.DRY_RUN_MODEL_DIR), "layers": list(DRY_LAYERS), "target": dec.n_layers - 1,
               "device_map": {k: v for k, v in plan.items() if k != "device_map"}, "git": job["git"],
               "cap": job["cap"], "vram": G.vram_record(torch.device("cuda:0")), "stages": st, "ok": st["ok"]}
        C.DRY_RUN_JSON.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(C.DRY_RUN_JSON)
    log(f"wrote {C.DRY_RUN_JSON}: {'ok' if out['ok'] else 'FAILED'}")
    return 0 if out["ok"] else 1


def run_pilot(args) -> int:
    if not C.DRY_RUN_JSON.is_file() or not json.loads(C.DRY_RUN_JSON.read_text(encoding="utf-8")).get("ok"):
        raise SystemExit("the 12B dry run has not passed (R3_00_dry_run.json); run --dry-run-12b first")
    if C.PILOT_JSON.exists() and not args.force:
        raise SystemExit(f"{C.PILOT_JSON} exists; --force to rerun (the earlier run stays in the runs log)")
    recs = C.wikitext_records()
    texts = recs[:C.PILOT_N_PROMPTS]
    DD = torch.load(C.R1_DIRS_PATH, weights_only=False, map_location="cpu")
    with G.gpu_job("R3_00 pilot (31B)", sys.argv, args.allow_uncommitted, log) as job:
        dev = torch.device("cuda:0")
        vram_before = G.vram_record(dev)
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        vram_loaded = G.vram_record(dev)
        kw = dict(target=C.TARGET_LAYER, skip=C.SKIP_FIRST, max_seq_len=C.MAX_SEQ_LEN)
        pm = pilot_measure(dec, texts, layers=C.FIT_LAYERS, linears_ctx=lambda: G.grad_linears(hf),
                           parity_dims=parity_dims_from_r1(), q1_dirs=DD["dirs"].float(),
                           vmap_B=C.PILOT_VMAP_B, replicate_B=C.PILOT_REPLICATE_B, passes=C.PILOT_PASSES,
                           budget_s=C.PILOT_MAX_MINUTES * 60 * 0.8, cap_gib=C.JOB_CAP_GB, **kw)
        # a forward of R1_10's longest held-out sequence's length (H4), read on a long record cut to that length
        import R3_05_eval as E5

        seqs = E5.r1_10_sequences(tok)
        longest = max(len(s["ids"]) for s in seqs)
        long_text = "\n".join(recs[2:12])
        t_long = time_forward(dec, long_text, [C.TARGET_LAYER], longest)
        checks = {"lrp_identity": pm["lrp_identity"]["share"] >= C.PILOT_LRP_TOP1_MIN,
                  "parity": pm["parity"]["plain"]["min_cos"] >= C.PILOT_PARITY_COS_MIN}
        proj = None
        if pm["choice"] is not None and pm["q1"].get("ok"):
            mp, _ = __import__("R3_Q1_jlens").miru_prompts()
            proj = project(pm["choice"], {"forward_nograd_s": pm["forward_nograd_s"],
                                          "q1_forward_s": pm["q1"]["forward_s"],
                                          "q1_directions_s": pm["q1"]["directions_s"],
                                          "forward_nograd_long_s": t_long},
                           n_items=sum(len(v) for v in C.jpp_items(C.JPP_MACRO_EVALS).values()),
                           n_q1_prompts=C.N_FIT + (len(mp) if mp else 0), n_h4_seqs=len(seqs),
                           r2_eval_s=json.loads((C.RESULTS / "R2_02_eval.json").read_text())["meta"]["timing_s"]["total"])
        decision = decide(proj, checks, C.APPROVED_GPU_HOURS)
        out = {"schema": "R3_00_pilot/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "prompts": {"which": f"WikiText fit records 0-{C.PILOT_N_PROMPTS - 1}",
                           "sha256": [C.text_sha256(t) for t in texts]},
               "vram": {"before_load": vram_before, "after_load": vram_loaded, "end": G.vram_record(dev)},
               "cap": job["cap"], "git": job["git"], **pm, "forward_nograd_long_s": t_long,
               "h4_longest_ids": longest, "projection": proj, "decision": decision, "choice": pm["choice"],
               "bounds": {"flop_lower_hours": [3.6, 14.0], "b1_bound_hours": C.B1_BOUND_HOURS,
                          "pilot_max_minutes": C.PILOT_MAX_MINUTES}}
        C.PILOT_JSON.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(C.PILOT_JSON)
    log(f"wrote {C.PILOT_JSON}: {decision['decision']} ({decision['why']})")
    return 0 if decision["decision"] == "proceed" else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run-12b", action="store_true")
    ap.add_argument("--pilot", action="store_true")
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
    if a.dry_run_12b:
        return run_dry(a)
    if a.pilot:
        return run_pilot(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
