#!/usr/bin/env python3
"""R3 03 (GPU, one shard of fit prompts per job, resumable; the merge runs on the CPU): J++'s expert Jacobians on the
W4A16 instruct model, with the R-lens rules (D1-D3, D5, D12).

A shard job (--shard i, i = 0..7) fits the 8 fit prompts of slice i of the 64 (jpp_lens's shard_prompt_slice). Per
prompt: one forward under the rules with the graph rooted at block 30 and the recompute-in-backward 4-bit linears;
then ceil(5376 / B) backward passes, each a one-hot cotangent at B output dimensions set at every kept position of
block 59's raw output; the rows at the 29 source layers are routed by R3_02's frozen router and added, per layer and
expert, to fp32 sums on the CPU (29 x 8 x 5376^2 x 4 B = 26.8 GB). B and the batching mode (a replicated batch, or
is_grads_batched passes) are the pilot's choice (R3_00_pilot.json). The sums are written one file per layer
(L{layer}.pt: row_sum, count, weight) with a meta.json, into experiments/results/R3_03_shards/shard{i}/state/;
every --checkpoint-every prompts the whole state is rewritten beside the old one and swapped in, so a killed job
resumes after its last checkpoint. A finished shard says "complete" in its meta.json.

The merge (--merge, CPU) reads the eight shards one layer at a time (memory-mapped), forms the total and the halves A
(even shards) and B (odd shards), replaces an expert with fewer than 50 kept positions by the layer's pooled
Jacobian, rounds the experts to fp16 (the file jpp_lens's fit-weights reads), and writes R3_03_merged/L{layer}.pt
and merged.json (counts, fallbacks, sha256 of each file). --delete-shards then removes the shard files, only after
every merged file has been read back and checked.

Runs only after R3_00's pilot says "proceed" and R3_02's router exists, with the registered files committed and the
server stopped:
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_03_fit.py --shard I [--checkpoint-every 2]
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_03_fit.py --merge [--delete-shards]
Every check before the model:  CUDA_VISIBLE_DEVICES="" ... R3_03_fit.py --check-only
Self-test (CPU, a tiny Gemma 4: the streamed merge against the in-memory merge, resuming, refusals):
  CUDA_VISIBLE_DEVICES="" ... R3_03_fit.py --selftest
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--check-only" in sys.argv or "--merge" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # the CPU modes never see the GPU
sys.dont_write_bytecode = True

import argparse
import datetime
import json
import shutil
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

T0 = time.time()
SHARD_BYTES = len(C.FIT_LAYERS) * C.N_EXPERTS * C.D_MODEL ** 2 * 4      # one shard's sums on disk (26.8 GB)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Interrupted(Exception):
    """Raised by the self-test's stop_after: a job killed between prompts."""


# ------------------------------------------------------------------------------------------------ one shard
def _state_dirs(shard_dir: Path) -> Tuple[Path, Path, Path]:
    return shard_dir / "state", shard_dir / "state.tmp", shard_dir / "state.old"


def load_state(shard_dir: Path, layers: Sequence[int], E: int, F: int, N: int):
    """(sums, meta) of a shard's last checkpoint, or fresh zeros and None. A half-written state.tmp is discarded;
    a swap interrupted after state was moved aside is completed backwards."""
    st, tmp, old = _state_dirs(shard_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    if old.exists() and not st.exists():
        os.replace(old, st)
    elif old.exists():
        shutil.rmtree(old)
    if not (st / "meta.json").is_file():
        return {l: FL.LayerSums.zeros(E, F, N) for l in layers}, None
    meta = json.loads((st / "meta.json").read_text(encoding="utf-8"))
    sums = {}
    for l in layers:
        d = torch.load(st / f"L{l:02d}.pt", map_location="cpu", mmap=True, weights_only=True)
        sums[l] = FL.LayerSums(d["row_sum"].clone(), d["count"].clone(), d["weight"].clone())
        del d
    return sums, meta


def save_state(shard_dir: Path, sums: Dict[int, FL.LayerSums], meta: Dict[str, Any]) -> None:
    """Write the whole state beside the old one (state.tmp), then swap it in: the old state is valid until then."""
    st, tmp, old = _state_dirs(shard_dir)
    FL.write_shard(tmp, sums, [], **{k: v for k, v in meta.items() if k not in ("layers", "records")})
    (tmp / "meta.json").write_text(json.dumps(dict(meta, layers=sorted(sums)), indent=1) + "\n", encoding="utf-8")
    if st.exists():
        os.replace(st, old)
    os.replace(tmp, st)
    if old.exists():
        shutil.rmtree(old)


def fit_shard_job(dec: FL.Decoder, prompts: Sequence[str], first_idx: int, routers: Dict[int, FL.LayerRouter],
                  layers: Sequence[int], shard_dir: Path, settings: Dict[str, Any], *, rows_per_pass: int,
                  mode: str, lrp: bool = True, linears_ctx: Callable[[], Any] = nullcontext,
                  checkpoint_every: int = 2, n_experts: int = C.N_EXPERTS, target: int = C.TARGET_LAYER,
                  skip: int = C.SKIP_FIRST, max_seq_len: int = C.MAX_SEQ_LEN,
                  stop_after: Optional[int] = None) -> Dict[str, Any]:
    """Fit one shard (resuming from its last checkpoint) and return its final meta. `settings` (router sha256,
    layers, model, rules, ...) must equal those of a checkpoint it resumes from."""
    layers = sorted(layers)
    sums, meta = load_state(shard_dir, layers, n_experts, dec.d_model, dec.d_model)
    if meta is None:
        meta = {"first_idx": first_idx, "n_prompts": len(prompts), "settings": settings, "done": [], "records": [],
                "seconds": [], "complete": False, "lrp_report": None}
    elif meta["settings"] != settings or meta["first_idx"] != first_idx or meta["n_prompts"] != len(prompts):
        raise SystemExit(f"{shard_dir} holds a checkpoint made with other settings; move it away to refit")
    if meta["complete"]:
        log(f"  {shard_dir.name}: already complete")
        return meta
    seen = 0
    for j, text in enumerate(prompts):
        i = first_idx + j
        if i in meta["done"]:
            continue
        t = time.time()
        with linears_ctx():
            try:
                fw = FL.fit_forward(dec, text, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                    rows_per_pass=rows_per_pass, mode=mode, lrp=lrp)
            except ValueError:
                fw = None
            if fw is not None:
                FL.accumulate_prompt(sums, routers, fw, rows_per_pass)
                meta["records"].append(FL.PromptRecord(i, fw.seq_len, fw.positions.cpu().tolist(),
                                                       FL.token_ids_sha256(fw.input_ids)).__dict__)
                meta["lrp_report"] = fw.lrp_report
                del fw
        meta["done"].append(i)
        meta["seconds"].append(round(time.time() - t, 2))
        seen += 1
        log(f"  {shard_dir.name}: prompt {i} in {time.time() - t:.1f}s ({len(meta['done'])}/{len(prompts)})")
        last = len(meta["done"]) == len(prompts)
        if last or len(meta["done"]) % checkpoint_every == 0:
            meta["complete"] = last
            save_state(shard_dir, sums, meta)
        if stop_after is not None and seen >= stop_after and not last:
            raise Interrupted(f"stopped after {seen} prompts")
    return meta


# ------------------------------------------------------------------------------------------------ the merge
def merge_all(shard_dirs: Sequence[Path], layers: Sequence[int], out_dir: Path, *, min_kept: int = C.MIN_KEPT,
              expect_prompts: Optional[int] = None) -> Dict[str, Any]:
    """Every shard must be complete and fitted with the same settings; then per layer R3_fitlib.merge_layer and one
    file out_dir/L{layer}.pt, and out_dir/merged.json."""
    metas = []
    for d in shard_dirs:
        p = Path(d) / "state" / "meta.json"
        if not p.is_file():
            raise SystemExit(f"{d}: no state")
        m = json.loads(p.read_text(encoding="utf-8"))
        if not m["complete"]:
            raise SystemExit(f"{d}: not complete ({len(m['done'])}/{m['n_prompts']} prompts)")
        metas.append(m)
    if any(m["settings"] != metas[0]["settings"] for m in metas):
        raise SystemExit("the shards were fitted with different settings (router, model, rules or layers)")
    done = sorted(i for m in metas for i in m["done"])
    if done != list(range(len(done))) or (expect_prompts is not None and len(done) != expect_prompts):
        raise SystemExit(f"the shards cover prompts {done[:3]}..{done[-3:]}, not 0..{(expect_prompts or 0) - 1}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: Dict[str, Any] = {"layers": {}, "shards": [str(d) for d in shard_dirs], "settings": metas[0]["settings"],
                           "n_prompts_fitted": sum(len(m["records"]) for m in metas),
                           "prompts_skipped": sorted(set(done) - {r["prompt_idx"] for m in metas
                                                                   for r in m["records"]}),
                           "token_ids_sha256": [r["token_ids_sha256"] for m in metas for r in m["records"]],
                           "min_kept": min_kept}
    for l in layers:
        ml = FL.merge_layer([Path(d) / "state" for d in shard_dirs], l, min_kept=min_kept)
        payload = {"layer": l, "experts_fp16": ml.experts_fp16, "pooled": ml.pooled, "fallback": ml.fallback,
                   "count": ml.count, "weight": ml.weight,
                   "halves": {k: {"experts_fp16": v[0].half(), "pooled": v[1], "fallback": v[2], "count": v[3],
                                  "weight": v[4]} for k, v in ml.halves.items()}}
        path = out_dir / f"L{l:02d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        rec["layers"][str(l)] = {"sha256": C.sha256_file(path), "count": ml.count.tolist(),
                                 "fallback": ml.fallback.tolist(),
                                 "halves": {k: {"count": v[3].tolist(), "fallback": v[2].tolist()}
                                            for k, v in ml.halves.items()}}
        del ml, payload
    rec["written_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    (out_dir / "merged.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def load_merged(out_dir: Path, layer: int, verify: bool = True) -> Dict[str, Any]:
    """One merged layer, checked against merged.json's sha256 (R3_04 reads the fit through this)."""
    rec = json.loads((out_dir / "merged.json").read_text(encoding="utf-8"))
    path = out_dir / f"L{layer:02d}.pt"
    if verify and C.sha256_file(path) != rec["layers"][str(layer)]["sha256"]:
        raise SystemExit(f"{path} is not the file merged.json records")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def delete_shards(shard_dirs: Sequence[Path], out_dir: Path, layers: Sequence[int]) -> List[str]:
    """Remove the shard directories, only after every merged layer has been read back and matched its sha256."""
    for l in layers:
        m = load_merged(out_dir, l)
        if tuple(m["experts_fp16"].shape[0:1]) != tuple(m["count"].shape):
            raise SystemExit(f"merged layer {l} is malformed; the shards are kept")
    gone = []
    for d in shard_dirs:
        shutil.rmtree(d)
        gone.append(str(d))
    return gone


# ------------------------------------------------------------------------------------------------ checks
def check_only() -> Dict[str, Any]:
    import R3_gpu as G

    out: Dict[str, Any] = {"failed": [], "pending": []}
    recs = C.wikitext_records()
    out["fit_prompts"] = min(len(recs), C.N_FIT)
    if len(recs) < C.N_FIT:
        out["failed"].append(f"only {len(recs)} records")
    sl = FL.shard_slices(C.N_FIT, C.N_SHARDS)
    out["shards"] = [[s.start, s.stop] for s in sl]
    if sorted(i for s in sl for i in range(s.start, s.stop)) != list(range(C.N_FIT)):
        out["failed"].append("the shard slices do not cover the 64 fit prompts once each")
    free = shutil.disk_usage(C.RESULTS).free
    need = C.N_SHARDS * SHARD_BYTES * 1.1 + len(C.FIT_LAYERS) * C.N_EXPERTS * C.D_MODEL ** 2 * 2 * 3
    out["disk"] = {"free_gb": round(free / 1e9, 1), "needed_gb": round(need / 1e9, 1)}
    if free < need:
        out["failed"].append(f"{free / 1e9:.0f} GB free on {C.RESULTS}, the shards and the merge need {need / 1e9:.0f}")
    out["pilot"] = G.pilot_decision()
    if out["pilot"]["decision"] != "proceed":
        out["pending"].append(f"R3_00's pilot: {out['pilot']['why']}")
    if not C.ROUTER_PT.is_file():
        out["pending"].append(f"no {C.ROUTER_PT.name} yet (R3_02)")
    done = [d.name for d in sorted(C.SHARD_DIR.glob("shard*")) if (d / "state" / "meta.json").is_file()]
    out["shards_on_disk"] = done
    return out


# ------------------------------------------------------------------------------------------------ self-test
def selftest(record: bool = False) -> int:
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
    prompts = records[:9] + ["too short"]
    layers, E, kw = [1, 2, 3], 4, dict(target=5, skip=16, max_seq_len=40)
    routers = FL.fit_router(lambda: FL.kept_points(dec, records, layers, skip=16, max_seq_len=40), layers,
                            n_experts=E, pca_dim=8)
    settings = {"router_sha256": "selftest", "layers": layers, "lrp": True, "model": "tiny"}
    mem, _ = FL.fit_shard(dec, prompts, routers, layers, n_experts=E, rows_per_pass=8, mode="replicate", lrp=True,
                          **kw)
    slices = FL.shard_slices(len(prompts), 3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dirs = []
        for s, sl in enumerate(slices):
            d = tmp / "shards" / f"shard{s}"
            job = dict(rows_per_pass=8, mode="replicate", checkpoint_every=1, n_experts=E, **kw)
            if s == 0:                    # killed twice, then resumed
                for stop in (1, 1):
                    try:
                        fit_shard_job(dec, prompts[sl], sl.start, routers, layers, d, settings, stop_after=stop, **job)
                    except Interrupted:
                        pass
                part = json.loads((d / "state" / "meta.json").read_text())
                check(len(part["done"]) == 2 and not part["complete"],
                      "a shard job killed after each prompt leaves a valid checkpoint of the prompts done")
                (d / "state.tmp").mkdir()            # a half-written next state is discarded on resume
            fit_shard_job(dec, prompts[sl], sl.start, routers, layers, d, settings, **job)
            dirs.append(d)
        # the same shards fitted at once in memory
        ok = True
        for s, sl in enumerate(slices):
            one, _ = FL.fit_shard(dec, prompts[sl], routers, layers, n_experts=E, rows_per_pass=8, lrp=True, **kw)
            back, meta = load_state(dirs[s], layers, E, dec.d_model, dec.d_model)
            ok &= meta["complete"] and all(torch.equal(back[l].count, one[l].count)
                                           and float((back[l].row_sum - one[l].row_sum).abs().max()) <= 1e-6
                                           for l in layers)
        check(ok, "each shard job (resumed or not) equals the same prompts fitted at once in memory")
        meta2 = json.loads((dirs[2] / "state" / "meta.json").read_text())
        check(meta2["done"][-1] == 9 and all(r["prompt_idx"] != 9 for r in meta2["records"]),
              "a prompt too short to keep a position is marked done and skipped, as jpp_lens skips it")
        merged_dir = tmp / "merged"
        rec = merge_all(dirs, layers, merged_dir, min_kept=5, expect_prompts=len(prompts))
        ok = True
        for l in layers:
            ex, pooled, fb = FL.build_experts(mem[l], 5)
            m = load_merged(merged_dir, l)
            ok &= torch.equal(m["experts_fp16"], ex.half()) or float((m["experts_fp16"].float() - ex.half().float())
                                                                     .abs().max()) <= 1e-3 * float(ex.abs().max())
            ok &= float((m["pooled"] - pooled).abs().max()) <= 1e-6 * max(1.0, float(pooled.abs().max()))
            ok &= torch.equal(m["fallback"], fb) and torch.equal(m["count"], mem[l].count)
            hA, hB = m["halves"]["A"], m["halves"]["B"]
            ok &= torch.equal(hA["count"] + hB["count"], m["count"])
            ok &= float(((hA["pooled"] * hA["weight"].sum() + hB["pooled"] * hB["weight"].sum())
                         / (hA["weight"].sum() + hB["weight"].sum()) - m["pooled"]).abs().max()) <= 1e-6
        check(ok, "the streamed per-layer merge of the shard files equals the in-memory merge of one run over all "
                  "prompts (fp16 experts, pooled, fallback, counts); halves A (shards 0, 2) and B (shard 1) add up")
        fb_any = rec["layers"][str(layers[0])]
        rec_hi = merge_all(dirs, layers, tmp / "merged_hi", min_kept=10 ** 6)
        mh = load_merged(tmp / "merged_hi", layers[0])
        check(all(rec_hi["layers"][str(l)]["fallback"] == [True] * E for l in layers)
              and all(torch.equal(mh["experts_fp16"][e], mh["pooled"].half()) for e in range(E)),
              f"the fallback branch: with a floor no expert reaches, every expert is the pooled Jacobian (at the "
              f"floor of 5 here: fallback {fb_any['fallback']})")
        # vmap-mode shards equal replicate-mode shards
        dv = tmp / "vmap" / "shard0"
        fit_shard_job(dec, prompts[slices[0]], 0, routers, layers, dv, settings, rows_per_pass=8, mode="vmap",
                      checkpoint_every=4, n_experts=E, **kw)
        bv, _ = load_state(dv, layers, E, dec.d_model, dec.d_model)
        br, _ = load_state(dirs[0], layers, E, dec.d_model, dec.d_model)
        d = max(float((bv[l].row_sum - br[l].row_sum).abs().max()) for l in layers)
        check(d <= 1e-5, f"a shard fitted in vmap mode equals the replicated-batch shard (max |diff| {d:.1e})")
        # refusals
        refused = []
        bad = tmp / "bad" / "shard0"
        try:
            fit_shard_job(dec, prompts[slices[0]], 0, routers, layers, bad, settings, stop_after=1, rows_per_pass=8,
                          mode="replicate", checkpoint_every=1, n_experts=E, **kw)
        except Interrupted:
            pass
        _set_settings(dv, {"router_sha256": "another router"})
        for what, fn in (("an incomplete shard is not merged", lambda: merge_all([bad] + dirs[1:], layers, tmp / "x")),
                         ("shards fitted with another router are not merged",
                          lambda: merge_all([dv, dirs[1], dirs[2]], layers, tmp / "y")),
                         ("a checkpoint made with other settings is not resumed",
                          lambda: fit_shard_job(dec, prompts[slices[0]], 0, routers, layers, bad,
                                                dict(settings, lrp=False), rows_per_pass=8, mode="replicate",
                                                n_experts=E, **kw))):
            try:
                fn()
                refused.append(False)
                check(False, what)
            except SystemExit:
                refused.append(True)
                check(True, what)
        gone = delete_shards(dirs, merged_dir, layers)
        check(len(gone) == 3 and not any(Path(g).exists() for g in gone) and (merged_dir / "merged.json").is_file(),
              "--delete-shards removes the shard files only after reading back every merged layer, and keeps the "
              "merged files")
        (merged_dir / f"L{layers[0]:02d}.pt").write_bytes(b"x")
        try:
            load_merged(merged_dir, layers[0])
            check(False, "a merged file that is not the one merged.json records is refused")
        except SystemExit:
            check(True, "a merged file that is not the one merged.json records is refused")
    co = check_only()
    check(not co["failed"] and co["shards"] == [[8 * i, 8 * i + 8] for i in range(8)],
          f"--check-only: 8 shards of 8 fit prompts, disk {co['disk']}; waiting for: {co['pending']}")
    print(f"R3_03_fit self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


def _set_settings(shard_dir: Path, change: Dict[str, Any]) -> None:
    p = shard_dir / "state" / "meta.json"
    m = json.loads(p.read_text())
    m["settings"] = dict(m["settings"], **change)
    p.write_text(json.dumps(m))


# ------------------------------------------------------------------------------------------------ the run
def run_shard(args) -> int:
    import R3_gpu as G

    if not 0 <= args.shard < C.N_SHARDS:
        raise SystemExit(f"--shard must be 0..{C.N_SHARDS - 1}")
    pilot = G.require_pilot_proceed()
    choice = dict(pilot["choice"] or {})
    deviation = {}
    for k, v in (("mode", args.mode), ("rows_per_pass", args.rows_per_pass)):
        if v is not None and v != choice.get(k):
            deviation[k] = {"pilot": choice.get(k), "used": v}
            choice[k] = v
    rj = json.loads(C.ROUTER_JSON.read_text(encoding="utf-8"))
    if C.sha256_file(C.ROUTER_PT) != rj["router_sha256"]:
        raise SystemExit(f"{C.ROUTER_PT} is not the router R3_02_router.json records")
    routers = FL.load_router(C.ROUTER_PT)
    sl = FL.shard_slices(C.N_FIT, C.N_SHARDS)[args.shard]
    prompts = C.wikitext_records()[:C.N_FIT][sl]
    settings = {"router_sha256": rj["router_sha256"], "layers": list(C.FIT_LAYERS), "lrp": C.LRP_MODE,
                "model": C.MODEL_ID, "snapshot": C.MODEL_SNAPSHOT, "target": C.TARGET_LAYER,
                "skip_first": C.SKIP_FIRST, "max_seq_len": C.MAX_SEQ_LEN, "n_experts": C.N_EXPERTS}
    with G.gpu_job(f"R3_03_fit shard {args.shard}", sys.argv, args.allow_uncommitted, log) as job:
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        routers = {l: FL.LayerRouter(r.mean.to(dec.input_device), r.components.to(dec.input_device),
                                     r.centroids.to(dec.input_device)) for l, r in routers.items()}
        meta = fit_shard_job(dec, prompts, sl.start, routers, C.FIT_LAYERS, C.SHARD_DIR / f"shard{args.shard}",
                             settings, rows_per_pass=int(choice["rows_per_pass"]), mode=choice["mode"], lrp=True,
                             linears_ctx=lambda: G.grad_linears(hf), checkpoint_every=args.checkpoint_every)
        meta_path = C.SHARD_DIR / f"shard{args.shard}" / "state" / "meta.json"
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        m.update({"choice": choice, "deviation_from_pilot": deviation, "git": job["git"], "cap": job["cap"]})
        meta_path.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        job["out"] = str(meta_path)
    log(f"shard {args.shard}: {'complete' if meta['complete'] else 'not complete'}")
    return 0


def run_merge(args) -> int:
    dirs = [C.SHARD_DIR / f"shard{i}" for i in range(C.N_SHARDS)]
    rec = merge_all(dirs, C.FIT_LAYERS, C.MERGED_DIR, expect_prompts=C.N_FIT)
    log(f"merged {len(rec['layers'])} layers into {C.MERGED_DIR}; fallbacks: "
        + ", ".join(f"{l}:{sum(v['fallback'])}" for l, v in rec["layers"].items() if any(v["fallback"])))
    if args.delete_shards:
        gone = delete_shards(dirs, C.MERGED_DIR, C.FIT_LAYERS)
        log(f"deleted {len(gone)} shard directories")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, help="the GPU job for shard I (server stopped)")
    ap.add_argument("--merge", action="store_true", help="merge the 8 shards (CPU)")
    ap.add_argument("--delete-shards", action="store_true", help="with --merge: remove the shards after checking")
    ap.add_argument("--checkpoint-every", type=int, default=2, help="prompts between checkpoints (26.8 GB each)")
    ap.add_argument("--mode", choices=("replicate", "vmap"), help="override the pilot's choice (a deviation)")
    ap.add_argument("--rows-per-pass", type=int, help="override the pilot's choice (a deviation)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
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
    if a.merge:
        return run_merge(a)
    if a.shard is not None:
        return run_shard(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
