#!/usr/bin/env python3
"""R3 Q1 (GPU, resumable; about 4 backward passes per prompt): the 4-bit J-lens arm of H6, fine-tuning against
quantisation (R3_PREREG.md §1, §8).

What is fitted: a plain J-lens on the served W4A16 instruct model, for R1's 128 final-space directions only (the
directions Q0 compares): per prompt, d_k^T J_l = the gradient of sum over kept targets p' of <h_59[p'], d_k> with
respect to block l's output, read at each kept source position p (so the sum runs over p' >= p: Miru's
"summed_current_and_future_targets"), averaged over the kept positions 16 .. T-2; then a uniform mean over prompts.
Plain gradients (no R-lens rules), target block 59's raw output, the batched backward in batches of Miru's
dim_batch 32 (four passes for 128 directions), code layers 30-58. Two halves by prompt parity give the split-half
ceiling (Miru's own ceiling is unknown).

The prompts, two sets:
- "miru": Miru's 425 processed prompts, if R3_01_check.py rebuilt them from its header's description and both of
  its hashes matched. Then bf16 against W4A16 is the only change between the Miru file and this arm.
- "jpp64": J++'s 64 fit prompts (the first 64 WikiText records), the data yardstick; and, while Miru's prompts are
  not rebuilt, the registered fallback, so that the arm also changes the prompts (WikiText both, Miru's 425
  rechunked passages against J++'s first 64 records). The output says which set every number comes from.
Known unknowns of Miru's estimator, not in its header: whether its last position is kept (here, as J++: no), and
its gradient mode beyond "no LRP recorded".

After the fit (and with --analyze-only, on the CPU from the saved sums): cosines in q10 and loud coordinates per
band against the Q0 pullbacks (Miru, J++, JLens) and R1's IT covectors, beside Q0's Miru-against-JLens row, and the
split-half ceiling. Writes experiments/results/R3_Q1_jlens.pt (gitignored sums) and R3_Q1_jlens.json.

Runs only after the pilot's "proceed", the registered files committed, the server stopped:
  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_Q1_jlens.py --run [--set miru|jpp64|both]
Every check before the model:  CUDA_VISIBLE_DEVICES="" ... R3_Q1_jlens.py --check-only
Self-test (CPU, a tiny Gemma 4):  CUDA_VISIBLE_DEVICES="" ... R3_Q1_jlens.py --selftest
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--check-only" in sys.argv or "--analyze-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # the CPU modes never see the GPU
sys.dont_write_bytecode = True

import argparse
import datetime
import itertools
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_fitlib as FL  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Interrupted(Exception):
    pass


# ------------------------------------------------------------------------------------------------ the prompts
def miru_prompts() -> Tuple[Optional[List[str]], str]:
    """(Miru's 425 processed prompts, how they were found) if R3_01_check recorded a rebuild and it reproduces
    here with the processed-prefix hash; else (None, why)."""
    if not C.CHECK_JSON.is_file():
        return None, f"no {C.CHECK_JSON.name}: run R3_01_check.py"
    mp = json.loads(C.CHECK_JSON.read_text(encoding="utf-8")).get("miru_prompts") or {}
    if not mp.get("rebuilt"):
        return None, ("Miru's prompts were not rebuilt (R3_01_check.json): the arm runs on J++'s 64 prompts only "
                      "(the registered fallback; the prompts change with the checkpoint)")
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    import R3_01_check as K                        # it hides the GPU when imported; the run still needs it

    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved

    fit = C.miru_fit_metadata()
    sel = fit["provenance"]["corpus"]["selection"]
    want, n = fit["fit"]["processed_prompt_prefix_sha256"], fit["fit"]["processed_prompts"]
    rows = C.wikitext_rows()
    for hit in mp["hits"]:
        for strip, sep, algo in itertools.product((True, False), ("\n", " ", "\n\n", ""),
                                                  ("pack, long paragraphs split at a space", "pack, no split",
                                                   "stream cut at a space")):
            name = f"{algo}; paragraphs {'stripped' if strip else 'as stored'}; joined by {sep!r}"
            if name != hit["reading"]:
                continue
            paras = [(t.strip() if strip else t) for t in rows
                     if t.strip() and not (sel["exclude_headings"] and K.HEADING.match(t))]
            ps = list(K._streamed(paras, sep, sel["min_chars"], sel["max_chars"]) if algo.startswith("stream")
                      else K._packed(paras, sep, sel["min_chars"], sel["max_chars"], algo.startswith("pack, long")))
            if len(ps) >= n and want in K.sequence_hashes(ps[:n]).values():
                return ps[:n], f"rebuilt ({name}; processed-prefix sha256 matches)"
    return None, "R3_01_check.json records a rebuild that does not reproduce here"


def prompt_sets(which: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    mp, why = miru_prompts()
    if which in ("miru", "both"):
        if mp is not None:
            out["miru"] = {"texts": mp, "why": why}
        else:
            log(f"miru: {why}")
    if which in ("jpp64", "both") or not out:
        out["jpp64"] = {"texts": C.wikitext_records()[:C.N_FIT],
                        "why": "J++'s 64 fit prompts" + ("" if mp is not None else f"; {why}")}
    return out


# ------------------------------------------------------------------------------------------------ the fit
def fit_set(dec: FL.Decoder, texts: Sequence[str], dirs_KF: torch.Tensor, layers: Sequence[int], ckpt: Path, *,
            mode: str = "vmap", dim_batch: int = C.Q1_DIM_BATCH, linears_ctx: Callable[[], Any] = nullcontext,
            every: int = 8, target: int = C.TARGET_LAYER, skip: int = C.SKIP_FIRST,
            max_seq_len: int = C.MAX_SEQ_LEN, stop_after: Optional[int] = None) -> Dict[str, Any]:
    """Per-half sums over prompts of the per-prompt covectors [L, K, N] (fp32), resumable from `ckpt`."""
    layers = sorted(layers)
    K, N = dirs_KF.shape[0], dec.d_model
    st = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.is_file() else {
        "sum": torch.zeros(2, len(layers), K, N), "n": [0, 0], "done": [], "skipped": [], "token_sha": {},
        "seconds": [], "dirs_sha256": C.text_sha256(json.dumps(dirs_KF.tolist())), "layers": layers}
    if st["layers"] != layers or st["dirs_sha256"] != C.text_sha256(json.dumps(dirs_KF.tolist())):
        raise SystemExit(f"{ckpt} was written for other layers or directions")
    seen = 0
    for i, t in enumerate(texts):
        if i in st["done"]:
            continue
        t0 = time.time()
        with linears_ctx():
            try:
                fw = FL.fit_forward(dec, t, layers, target=target, skip=skip, max_seq_len=max_seq_len,
                                    rows_per_pass=dim_batch, mode=mode, lrp=False)
            except ValueError:
                fw = None
            if fw is None:
                st["skipped"].append(i)
            else:
                rows = FL.direction_rows(fw, dirs_KF.to(fw.target.device), dim_batch=dim_batch)
                h = i % 2
                for j, r in enumerate(rows):
                    st["sum"][h, j] += r.cpu()
                st["n"][h] += 1
                st["token_sha"][i] = FL.token_ids_sha256(fw.input_ids)
                del fw, rows
        st["done"].append(i)
        st["seconds"].append(round(time.time() - t0, 2))
        seen += 1
        if len(st["done"]) % every == 0 or len(st["done"]) == len(texts):
            tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
            torch.save(st, tmp)
            os.replace(tmp, ckpt)
        if stop_after is not None and seen >= stop_after and len(st["done"]) < len(texts):
            raise Interrupted()
    return st


def means(st: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """{"all", "A", "B"}: [L, K, N] uniform means over prompts (A: even prompts, B: odd)."""
    n0, n1 = st["n"]
    return {"all": (st["sum"][0] + st["sum"][1]) / max(1, n0 + n1), "A": st["sum"][0] / max(1, n0),
            "B": st["sum"][1] / max(1, n1)}


# ------------------------------------------------------------------------------------------------ the comparison
def compare(q1: Dict[str, torch.Tensor], layers: Sequence[int], refs: Dict[str, torch.Tensor],
            masks: Dict[str, torch.Tensor], seed: int = C.BOOT_SEED) -> Dict[str, Any]:
    """Cosines (R1_04's masked cosine) of the 4-bit J-lens covectors against each reference ([K, 59, N] pullbacks
    over code layers 0-58), per band of the fitted layers, 98 token pairs; and the split-half ceiling."""
    import R3_Q0_pullbacks as Q0

    layers = list(layers)
    rng = np.random.default_rng(seed)
    bands = [(b, a, z) for b, a, z in C.Q0_BANDS if a >= layers[0] and z <= layers[-1]]

    def summ(cos_KL: torch.Tensor) -> Dict[str, Any]:
        x = cos_KL.double().numpy()
        tp = list(C.TOKPAIR_DIRS)
        idx = rng.integers(0, len(tp), size=(2000, len(tp)))
        out = {}
        for name, a, z in bands:
            cols = [layers.index(l) for l in range(a, z + 1)]
            v = x[:, cols].mean(1)[tp]
            out[name] = {"tokpair98": round(float(v.mean()), 4),
                         "ci95": [round(float(q), 4) for q in np.percentile(v[idx].mean(1), [2.5, 97.5])]}
        return out

    A = q1["all"].permute(1, 0, 2)                         # [K, L, N]
    out: Dict[str, Any] = {"bands": [b for b, _, _ in bands], "vs": {}}
    for name, R in refs.items():
        Rl = R[:, layers]
        out["vs"][name] = {mk: summ(Q0.mcos(A, Rl, masks[mk][layers])) for mk in ("all", "loud", "q10")}
    hA, hB = q1["A"].permute(1, 0, 2), q1["B"].permute(1, 0, 2)
    out["split_half"] = {mk: summ(Q0.mcos(hA, hB, masks[mk][layers])) for mk in ("all", "loud", "q10")}
    return out


def analyze(st_by_set: Dict[str, Dict[str, Any]], layers: Sequence[int]) -> Dict[str, Any]:
    import R3_Q0_pullbacks as Q0

    q0 = torch.load(C.Q0_PT, map_location="cpu", weights_only=False)
    refs = {k: v for k, v in q0["pb"].items()}
    G = torch.load(C.R1_GBAR_PATH, map_location="cpu", mmap=True, weights_only=False)
    refs["r1_it"] = G["mix"][:, :59].float()
    masks = {k: v[:59] for k, v in Q0.quiet_masks().items()}
    q0j = json.loads(C.Q0_JSON.read_text(encoding="utf-8"))
    out = {"q0_fine_tuning_axis_copied": q0j["comparisons"].get(
        "fp32: miru vs jlens (fine-tuning axis: bf16 instruct vs bf16 base, plain J-lenses)")}
    for name, st in st_by_set.items():
        out[name] = {"n_prompts": st["n"], "skipped": st["skipped"], "compare": compare(means(st), layers, refs,
                                                                                       masks)}
    return out


# ------------------------------------------------------------------------------------------------ checks
def check_only() -> Dict[str, Any]:
    import R3_gpu as G

    out: Dict[str, Any] = {"failed": [], "pending": []}
    mp, why = miru_prompts()
    out["miru_prompts"] = {"rebuilt": mp is not None, "why": why}
    if mp is None:
        out["pending"].append("Miru's prompts: " + why)
    for name, path, sha, size in (("R1_02_dirs.pt", C.R1_DIRS_PATH, C.R1_DIRS_SHA256, C.R1_DIRS_SIZE),
                                  ("R1_04_gbar.pt", C.R1_GBAR_PATH, C.R1_GBAR_SHA256, C.R1_GBAR_SIZE)):
        rec = C.check_pinned_file(path, sha, size, full=False)
        if not rec["size_ok"]:
            out["failed"].append(f"{name} missing or not the pinned size")
    if not C.Q0_PT.is_file() or not json.loads(C.Q0_JSON.read_text(encoding="utf-8")).get("reproduced_r1"):
        out["failed"].append("Q0 has not reproduced R1's numbers (run R3_Q0_pullbacks.py)")
    out["pilot"] = G.pilot_decision()
    if out["pilot"]["decision"] != "proceed":
        out["pending"].append(f"R3_00's pilot: {out['pilot']['why']}")
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
    texts = C.wikitext_records()[:7] + ["short"]
    layers, kw = [1, 2, 3], dict(target=5, skip=16, max_seq_len=40)
    g = torch.Generator().manual_seed(2)
    dirs = torch.randn(10, dec.d_model, generator=g)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        st = fit_set(dec, texts, dirs, layers, tmp / "a.pt", dim_batch=4, every=1, **kw)
        # brute force: full plain rows per prompt, projected, averaged over prompts
        want = torch.zeros(len(layers), 10, dec.d_model)
        n = 0
        for t in texts[:7]:
            fw = FL.fit_forward(dec, t, layers, rows_per_pass=8, mode="replicate", lrp=False, **kw)
            rows = FL.prompt_mean_rows(fw, 8)
            for j, l in enumerate(layers):
                want[j] += dirs @ rows[l]
            n += 1
        m = means(st)
        d = float((m["all"] - want / n).abs().max())
        check(d <= 1e-5 and st["skipped"] == [7] and st["n"] == [4, 3],
              f"the arm's covectors are the plain estimator's full rows projected on each direction, averaged "
              f"uniformly over prompts (max |diff| {d:.1e}); a too-short prompt is skipped; halves 4 + 3")
        check(torch.allclose((m["A"] * 4 + m["B"] * 3) / 7, m["all"], atol=1e-6),
              "the halves (even and odd prompts) average to the whole")
        st_r = fit_set(dec, texts, dirs, layers, tmp / "r.pt", mode="replicate", dim_batch=4, **kw)
        check(float((means(st_r)["all"] - m["all"]).abs().max()) <= 1e-5,
              "vmap and replicated-batch modes give the same covectors (batches of dim_batch directions)")
        for _ in range(20):
            try:
                st2 = fit_set(dec, texts, dirs, layers, tmp / "b.pt", dim_batch=4, every=2, stop_after=3, **kw)
                break
            except Interrupted:
                continue
        check(torch.equal(st2["sum"], st["sum"]) and st2["n"] == st["n"],
              "a job killed every 3 prompts resumes from its checkpoint and ends with the same sums")
        try:
            fit_set(dec, texts, dirs[:5], layers, tmp / "a.pt", dim_batch=4, **kw)
            check(False, "a checkpoint for other directions is refused")
        except SystemExit:
            check(True, "a checkpoint for other directions is refused")
    # the comparison on synthetic covectors: a reference equal to the arm gives cosine 1, the halves their ceiling
    K, L, N = 128, 59, 24
    ref = torch.randn(K, L, N, generator=g)
    lay = list(range(30, 59))
    q1 = {"all": ref[:, lay].permute(1, 0, 2).clone(), "A": ref[:, lay].permute(1, 0, 2) + 0.1,
          "B": ref[:, lay].permute(1, 0, 2) - 0.1}
    masks = {"all": torch.ones(L, N), "q10": (torch.arange(N) < 3).float().expand(L, N), "loud": None}
    masks["loud"] = 1 - masks["q10"]
    cmp = compare(q1, lay, {"same": ref, "other": torch.randn(K, L, N, generator=g)}, masks)
    check(cmp["bands"] == ["L30-39", "L40-49", "L50-58"]
          and all(abs(cmp["vs"]["same"][mk][b]["tokpair98"] - 1) < 1e-6 for mk in ("all", "q10") for b in cmp["bands"])
          and all(abs(cmp["vs"]["other"]["all"][b]["tokpair98"]) < 0.2 for b in cmp["bands"]),
          "the comparison: an identical reference gives cosine 1 in every band and mask, an unrelated one about 0")
    co = check_only()
    check(not co["failed"], f"--check-only passes the pins and Q0; prompts: {co['miru_prompts']['why'][:60]}...; "
                            f"waiting for: {[p[:40] for p in co['pending']]}")
    print(f"R3_Q1_jlens self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, done, time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the run
def run(args) -> int:
    import R3_gpu as G

    G.require_pilot_proceed()
    sets = prompt_sets(args.set)
    DD = torch.load(C.R1_DIRS_PATH, weights_only=False, map_location="cpu")
    dirs = DD["dirs"].float()
    states: Dict[str, Dict[str, Any]] = {}
    with G.gpu_job("R3_Q1_jlens", sys.argv, args.allow_uncommitted, log) as job:
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        for name, s in sets.items():
            log(f"{name}: {len(s['texts'])} prompts ({s['why']})")
            states[name] = fit_set(dec, s["texts"], dirs, C.Q1_LAYERS, C.RESULTS / f"R3_Q1_{name}_ckpt.pt",
                                   mode=args.mode, linears_ctx=lambda: G.grad_linears(hf))
        torch.save({k: {"sum": v["sum"], "n": v["n"], "skipped": v["skipped"], "layers": v["layers"],
                        "token_sha": v["token_sha"], "seconds": v["seconds"]} for k, v in states.items()}, C.Q1_PT)
        out = {"schema": "R3_Q1/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "sets": {k: v["why"] for k, v in sets.items()}, "git": job["git"], "cap": job["cap"],
               "settings": {"layers": list(C.Q1_LAYERS), "dim_batch": C.Q1_DIM_BATCH, "mode": args.mode,
                            "target": C.TARGET_LAYER, "skip_first": C.SKIP_FIRST, "max_seq_len": C.MAX_SEQ_LEN,
                            "lrp": None, "directions": "R1_02_dirs.pt"},
               "analysis": analyze(states, C.Q1_LAYERS)}
        C.Q1_JSON.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(C.Q1_JSON)
    log(f"wrote {C.Q1_PT} and {C.Q1_JSON}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="the GPU run (server stopped)")
    ap.add_argument("--set", choices=("miru", "jpp64", "both"), default="both")
    ap.add_argument("--mode", choices=("vmap", "replicate"), default="vmap")
    ap.add_argument("--analyze-only", action="store_true", help="recompute the comparison from R3_Q1_jlens.pt (CPU)")
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
    if a.analyze_only:
        st = torch.load(C.Q1_PT, map_location="cpu", weights_only=False)
        out = json.loads(C.Q1_JSON.read_text(encoding="utf-8"))
        out["analysis"] = analyze(st, C.Q1_LAYERS)
        C.Q1_JSON.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        return 0
    if a.run:
        return run(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
