#!/usr/bin/env python3
"""R3 02 (GPU, forward passes only, resumable): J++'s router at the fitted layers (D4, D14), from the 1,000 WikiText
records of the cache.

For each of code layers 30-58 (pages 31-59): the block outputs at every record's kept positions (16 .. T-2, after BOS
and truncation to 128 word-pieces), unit-normalised.
  Pass 1 streams their mean and covariance in fp64 (29 x 5376^2 x 8 B = 6.7 GB on the CPU), checkpointed every
         --checkpoint-every records; eigh gives the top 64 components (fit_pca's eigh route, without storing points).
  Pass 2 projects every record's points on them (824 MB in all), checkpointed the same way.
  Then jpp_lens's own k-means++ (seed 0, 8 clusters, up to 100 iterations, tolerance 1e-6) per layer.
Writes experiments/results/R3_02_router.pt (gitignored: mean, components and centroids per layer) and
R3_02_router.json (counts, cluster sizes, the components' share of variance, the records skipped, the file's sha256,
timing, memory). A record too short to keep any position is skipped, as jpp_lens skips it.

Runs only after R3_00's pilot says "proceed" (the budget), with the registered files committed, the server stopped.
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_02_router.py --run [--checkpoint-every 100]
Every check before the model (CPU):  ... R3_02_router.py --check-only
Self-test (CPU, a tiny Gemma 4 on the cached records): ... R3_02_router.py --selftest
(prefix the CPU modes with CUDA_VISIBLE_DEVICES="")
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_fitlib as FL  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Interrupted(Exception):
    """Raised by the self-test's stop_after to stand for a job killed between records."""


def _save(obj: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit_router_resumable(dec: FL.Decoder, texts: Sequence[str], layers: Sequence[int], ckpt: Path, *,
                         every: int = 100, n_experts: int = C.N_EXPERTS, pca_dim: int = C.PCA_DIM,
                         seed: int = C.ROUTER_SEED, skip: int = C.SKIP_FIRST, max_seq_len: int = C.MAX_SEQ_LEN,
                         stop_after: Optional[int] = None) -> Tuple[Dict[int, FL.LayerRouter], Dict[str, Any]]:
    """R3_fitlib.fit_router's two passes, with a checkpoint every `every` records so that a killed job resumes
    where it stopped. Returns the routers and the record (counts, skipped records, token hashes, sizes, shares)."""
    jc = FL._jpp().cluster
    layers = sorted(layers)
    st: Dict[str, Any]
    if ckpt.is_file():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        log(f"resuming from {ckpt.name}: pass {st['pass']}, {st['done']} records done")
    else:
        st = {"pass": 1, "done": 0, "n_texts": len(texts), "skipped": [], "token_sha": [], "n_kept": [],
              "pca": None, "params": None, "projected": {l: [] for l in layers}}
    if st["n_texts"] != len(texts):
        raise SystemExit(f"{ckpt} was written for {st['n_texts']} records, not {len(texts)}")
    seen = 0

    def records(start: int):
        for i in range(start, len(texts)):
            ids, outs = FL.block_outputs(dec, texts[i], layers, max_seq_len)
            try:
                pos = FL.kept_positions(ids.shape[1], skip)
            except ValueError:
                yield i, ids, None
                continue
            yield i, ids, {l: outs[l][pos.to(outs[l].device)] for l in layers}

    if st["pass"] == 1:
        pca = FL.StreamedPCA(layers, dec.d_model)
        if st["pca"] is not None:
            pca.n, pca.s1, pca.s2 = st["pca"]["n"], st["pca"]["s1"], st["pca"]["s2"]
        for i, ids, pts in records(st["done"]):
            st["token_sha"].append(FL.token_ids_sha256(ids[0]))
            if pts is None:
                st["skipped"].append(i)
                st["n_kept"].append(0)
            else:
                st["n_kept"].append(int(next(iter(pts.values())).shape[0]))
                for l in layers:
                    pca.add(l, pts[l])
            st["done"] = i + 1
            seen += 1
            if st["done"] % every == 0 or st["done"] == len(texts):
                st["pca"] = {"n": pca.n, "s1": pca.s1, "s2": pca.s2}
                _save(st, ckpt)
                log(f"  pass 1: {st['done']}/{len(texts)} records (checkpoint)")
            if stop_after is not None and seen >= stop_after:
                raise Interrupted(f"stopped after {seen} records")
        params, shares = {}, {}
        for l in layers:
            mean, comps = pca.components(l, pca_dim)
            n = pca.n[l]
            cov = (pca.s2[l] - n * torch.outer(pca.s1[l] / n, pca.s1[l] / n)) / max(n - 1, 1)
            ev = torch.linalg.eigvalsh(cov)
            shares[l] = float(ev[-pca_dim:].sum() / ev.sum())
            params[l] = (mean.float(), comps.float())
        st.update({"pass": 2, "done": 0, "params": params, "shares": shares, "pca_n": dict(pca.n), "pca": None})
        _save(st, ckpt)
        del pca
    for i, ids, pts in records(st["done"]):
        if FL.token_ids_sha256(ids[0]) != st["token_sha"][i]:
            raise SystemExit(f"record {i} encodes differently in pass 2")
        if pts is not None:
            for l in layers:
                st["projected"][l].append(jc.normalize_center_project(pts[l].float().cpu(), *st["params"][l]))
        st["done"] = i + 1
        seen += 1
        if st["done"] % every == 0 or st["done"] == len(texts):
            _save(st, ckpt)
            log(f"  pass 2: {st['done']}/{len(texts)} records (checkpoint)")
        if stop_after is not None and seen >= stop_after:
            raise Interrupted(f"stopped after {seen} records")
    routers, sizes = {}, {}
    for l in layers:
        cent, assign = jc.kmeans(torch.cat(st["projected"][l]), n_experts, seed=seed,
                                 max_iterations=C.KMEANS_MAX_ITER, tolerance=C.KMEANS_TOL)
        routers[l] = FL.LayerRouter(st["params"][l][0], st["params"][l][1], cent)
        sizes[l] = torch.bincount(assign, minlength=n_experts).tolist()
    rec = {"n_records": len(texts), "skipped_records": st["skipped"], "n_points": st["pca_n"],
           "kept_per_record": {"min": min(st["n_kept"]), "max": max(st["n_kept"]), "sum": sum(st["n_kept"])},
           "token_ids_sequence_sha256": C.sequence_sha256(st["token_sha"]), "cluster_sizes": sizes,
           "pca_variance_share": {l: round(v, 6) for l, v in st["shares"].items()}}
    return routers, rec


# ------------------------------------------------------------------------------------------------ checks
def check_only() -> Dict[str, Any]:
    """Everything before the model, on the CPU: the cache, the records, the pins and what the run waits for."""
    out: Dict[str, Any] = {"failed": [], "pending": []}
    recs = C.wikitext_records()
    if len(recs) != C.N_ROUTER:
        out["failed"].append(f"{len(recs)} WikiText records, not {C.N_ROUTER}")
    out["records"] = len(recs)
    if not (C.MODEL_DIR / "config.json").is_file():
        out["failed"].append(f"no model snapshot at {C.MODEL_DIR}")
    import R3_gpu as G

    out["pilot"] = G.pilot_decision()
    if out["pilot"]["decision"] != "proceed":
        out["pending"].append(f"R3_00's pilot: {out['pilot']['why']}")
    if C.ROUTER_PT.exists():
        out["pending"].append(f"{C.ROUTER_PT.name} exists: a run would need --force (every run is logged)")
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
    texts = C.wikitext_records()[:40] + ["too short"]
    layers, kw = [1, 2, 3], dict(skip=16, max_seq_len=48)
    spec = dict(n_experts=4, pca_dim=8, **kw)
    ref = FL.fit_router(lambda: FL.kept_points(dec, texts, layers, **kw), layers, n_experts=4, pca_dim=8)
    with tempfile.TemporaryDirectory() as tmp:
        ck1 = Path(tmp) / "a.pt"
        r1, rec1 = fit_router_resumable(dec, texts, layers, ck1, every=7, **spec)
        same = all(torch.equal(r1[l].centroids, ref[l].centroids) and torch.equal(r1[l].components, ref[l].components)
                   and torch.equal(r1[l].mean, ref[l].mean) for l in layers)
        check(same, "the resumable two-pass router equals R3_fitlib.fit_router (the one R3_parity tied to jpp_lens) "
                    "bit for bit: mean, components, centroids")
        check(rec1["skipped_records"] == [40] and rec1["n_records"] == 41,
              "a record too short to keep a position is skipped and recorded (as jpp_lens skips it)")
        ok = True
        for stop in (9, 30, 12):          # killed in pass 1, in pass 2, and again (each run passes a checkpoint)
            ck = Path(tmp) / f"b{stop}.pt"
            resumed = None
            for attempt in range(20):
                try:
                    resumed = fit_router_resumable(dec, texts, layers, ck, every=7, stop_after=stop, **spec)
                    break
                except Interrupted:
                    continue
            assert resumed is not None
            r2, rec2 = resumed
            ok &= all(torch.equal(r2[l].centroids, r1[l].centroids) and torch.equal(r2[l].components,
                                                                                   r1[l].components) for l in layers)
            ok &= rec2["token_ids_sequence_sha256"] == rec1["token_ids_sequence_sha256"]
            ok &= rec2["cluster_sizes"] == rec1["cluster_sizes"]
        check(ok, "a job killed between records (in pass 1 and in pass 2, several times) resumes from its "
                  "checkpoint and ends with the same router and record")
        FL.save_router(r1, Path(tmp) / "router.pt", note="selftest")
        back = FL.load_router(Path(tmp) / "router.pt")
        pts = next(iter(FL.kept_points(dec, texts[:1], layers, **kw)))
        check(all(torch.equal(back[l].assign(pts[l]), r1[l].assign(pts[l])) for l in layers),
              "the router file round-trips: the loaded router assigns as the fitted one")
        sz = rec1["cluster_sizes"]
        check(all(sum(v) == rec1["n_points"][l] for l, v in sz.items()) and all(0 < s <= 1 for s in
                                                                                 rec1["pca_variance_share"].values()),
              "cluster sizes add up to the points per layer; the components' variance share lies in (0, 1]")
    co = check_only()
    check(not co["failed"] and any("pilot" in p for p in co["pending"]),
          f"--check-only passes everything before the model, and says what the run waits for: {co['pending']}")
    print(f"R3_02_router self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the run
def run(args) -> int:
    import R3_gpu as G

    if C.ROUTER_PT.exists() and not args.force:
        raise SystemExit(f"{C.ROUTER_PT} exists; --force to refit (the earlier run stays in the runs log)")
    G.require_pilot_proceed()
    with G.gpu_job("R3_02_router", sys.argv, args.allow_uncommitted, log) as job:
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        texts = C.wikitext_records()
        t = time.time()
        with torch.no_grad():
            routers, rec = fit_router_resumable(dec, texts, C.FIT_LAYERS, C.ROUTER_CKPT, every=args.checkpoint_every)
        FL.save_router(routers, C.ROUTER_PT, model=C.MODEL_ID, snapshot=C.MODEL_SNAPSHOT, layers=list(C.FIT_LAYERS),
                       n_experts=C.N_EXPERTS, pca_dim=C.PCA_DIM, seed=C.ROUTER_SEED)
        out = {"schema": "R3_02/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "router_pt": str(C.ROUTER_PT), "router_sha256": C.sha256_file(C.ROUTER_PT),
               "settings": {"layers": list(C.FIT_LAYERS), "n_experts": C.N_EXPERTS, "pca_dim": C.PCA_DIM,
                            "seed": C.ROUTER_SEED, "kmeans_max_iter": C.KMEANS_MAX_ITER, "kmeans_tol": C.KMEANS_TOL,
                            "skip_first": C.SKIP_FIRST, "max_seq_len": C.MAX_SEQ_LEN},
               "records_sequence_sha256": C.load_manifest()["wikitext"]["records_sequence_sha256"],
               "seconds": round(time.time() - t, 1), "git": job["git"], "cap": job["cap"], **rec}
        C.ROUTER_JSON.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(C.ROUTER_JSON)
    C.ROUTER_CKPT.unlink(missing_ok=True)
    log(f"wrote {C.ROUTER_PT} and {C.ROUTER_JSON}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="the GPU run (server stopped)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R3_selftests.json")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.check_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
