#!/usr/bin/env python3
"""R4 00: the replication record, made before registration (R4_PREREG.md §4.2). R4's reading code reads R3's 320
prompts again, on the backend as it is now, and is compared with R3's own evaluation (R3_05_eval.json). No R4 prompt
is read.

GPU (server stopped; an owner-approved run): R4_02_eval's functions, R3_05's backend.
  (a) Teacher-forced: every R3 prompt as R2_common composes it (the mid-reply prompts followed by R3's stored reply),
      read at R3's 930 stored positions, batched as R3 batched them (8 prompts a forward): J++ and R1 at 55 and 58, the
      naked eye at 55, 58 and 59.
  (b) Regenerated: R3's prompts generated again (8-token continuations, 48-token mid-reply replies) and their read
      positions placed again with R3's draw seed (20261021), then read.
  Writes R4_00_reread.json and appends every run to R4_00_runs.jsonl (started/finished/failed).
CPU (--compare): the record, R4_00_replication.json, passes when all three hold:
  (a) for each of the seven readouts, the filtered top-1 id equals R3's recorded one at 99% or more of the stored
      positions (R3's own re-read of R2's prompts matched every pair);
  (b) 95% or more of R3's read positions are found again on regeneration, with their draws, next token, content
      target and context (R3 §8.6(a)): this checks the model's own words, which define the targets;
  (c) R3's prompts, with the teacher-forced readouts, under R4's design with R3's bootstrap seed (20261022): R1's
      rules at 55 and 58 (rule 4 on the raw p) equal R3's; and when every top-1 of (a) matched, every mean, interval,
      sign count and raw p equals R3's to its stored rounding.
  The record also keeps what it was made with: the pinned files, the snapshot's blobs, the library versions and the
  sha256 of R4_common.py, R4_02_eval.py and this script. R4_02 refuses to run unless the record passed and matches the
  runtime and R4 scripts it finds (so an R4 script changed after the record calls for a new record).
On a failure nothing is registered until the cause is found (§4.2).

Run (GPU):  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R4_00_replicate.py [--batch 8]
                [--force]
Then (CPU): CUDA_VISIBLE_DEVICES="" ... R4_00_replicate.py --compare
Self-test (CPU; the comparison on R3's own file and perturbed copies, and the pipeline on R3_05's stand-in):
            CUDA_VISIBLE_DEVICES="" ... R4_00_replicate.py --selftest [--record]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--compare" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import argparse
import copy
import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402
import R3_05_eval as E  # noqa: E402
import R4_common as C  # noqa: E402
import R4_02_eval as R4E  # noqa: E402

T0 = time.time()
MIN_TOP1 = 0.99                                       # (a) per readout
MIN_FOUND = 0.95                                      # (b) R3's read positions found again
READOUTS: Tuple[Tuple[str, int], ...] = tuple((k, l) for k in C.KINDS for l in C.LENS_LAYERS[k])   # seven


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ the comparison
def _key(p: Dict[str, Any]) -> Tuple[Any, ...]:
    return (p["pid"], p["cond"], p["pos"], p.get("k"))


def compare_teacher_forced(tf: Dict[str, Any], r3: Dict[str, Any]) -> Dict[str, Any]:
    """(a): per readout, the share of R3's stored positions whose filtered top-1 id is R3's (and the unfiltered one,
    reported)."""
    if [_key(p) for p in tf["positions"]] != [_key(p) for p in r3["positions"]]:
        return {"ok": False, "why": "the teacher-forced positions are not R3's stored positions, in R3's order"}
    # the rebuilt input: the token read at each position must be the one R3 read there
    tok_differs = [{"pid": p["pid"], "pos": p["pos"], "k": p.get("k"), "r3": q.get("read_tok"),
                    "now": p.get("read_tok_again")} for p, q in zip(tf["positions"], r3["positions"])
                   if p.get("read_tok_again") != q.get("read_tok")]
    per: Dict[str, Any] = {}
    mism: List[Dict[str, Any]] = []
    for kind, l in READOUTS:
        n = same_f = same_u = 0
        for tp, sp in zip(tf["positions"], r3["positions"]):
            a, b = tp["r"][kind][str(l)], sp["r"][kind][str(l)]
            n += 1
            same_u += int(a["u"][0] == b["u"][0])
            if a["f"][0] == b["f"][0]:
                same_f += 1
            elif len(mism) < 300:
                mism.append({"pid": sp["pid"], "pos": sp["pos"], "k": sp.get("k"), "kind": kind, "layer": l,
                             "r3_f0": b["f"][0], "now_f0": a["f"][0]})
        per[f"{kind}@{l}"] = {"positions": n, "filtered_top1_same": same_f, "filtered_share": same_f / n,
                              "unfiltered_share": same_u / n, "ok": same_f / n >= MIN_TOP1}
    return {"ok": all(v["ok"] for v in per.values()) and not tok_differs, "readouts": per, "tolerance": MIN_TOP1,
            "all_identical": all(v["filtered_top1_same"] == v["positions"] for v in per.values()),
            "read_token_differs": tok_differs[:50], "n_read_token_differs": len(tok_differs),
            "mismatches_listed": mism}


def compare_regenerated(regen: Dict[str, Any], r3: Dict[str, Any]) -> Dict[str, Any]:
    """(b): R3's read positions found again (R3_06.replication_readouts' rule: the prompt, condition, position,
    draws, next token, content target and continuation, and for mid-reply the reply up to the read position's next
    token); the readouts at the found positions, reported."""
    reply_now = {p["pid"]: p.get("reply_ids") for p in regen["prompts"]}
    reply_r3 = {p["pid"]: p.get("reply_ids") for p in r3["prompts"]}
    mine = {_key(p): p for p in regen["positions"]}
    found, lost = [], []
    for q in r3["positions"]:
        p = mine.get(_key(q))
        same = p is not None and all(p.get(k) == q.get(k) for k in ("draws", "next_id", "ct_id", "cont_ids"))
        if same and q["cond"] == "mid-reply":
            k = q["k"]
            same = (reply_now.get(q["pid"]) or [])[:k + 2] == (reply_r3.get(q["pid"]) or [])[:k + 2]
        (found if same else lost).append((q, p))
    share = len(found) / len(r3["positions"]) if r3["positions"] else 0.0
    readouts = {}
    for kind, l in READOUTS:
        n = sum(1 for _ in found)
        same = sum(int(p["r"][kind][str(l)]["f"][0] == q["r"][kind][str(l)]["f"][0]) for q, p in found)
        readouts[f"{kind}@{l}"] = {"positions": n, "filtered_share": same / n if n else None}
    by_cond = {c: {"r3": sum(q["cond"] == c for q in r3["positions"]),
                   "found": sum(q["cond"] == c for q, _ in found)} for c in C.CONDITIONS}
    return {"ok": share >= MIN_FOUND, "r3_positions": len(r3["positions"]), "found_again": len(found),
            "found_share": share, "by_condition": by_cond, "tolerance": MIN_FOUND,
            "lost": [{"pid": q["pid"], "cond": q["cond"], "pos": q["pos"], "k": q.get("k")} for q, _ in lost][:200],
            "readouts_at_found_positions": readouts}


def compare_verdicts(tf: Dict[str, Any], r3_analysis: Dict[str, Any], exact_required: bool) -> Dict[str, Any]:
    """(c): R4's design (R3's seed) on R3's prompts with the teacher-forced readouts, against R3's record of R1."""
    import R4_03_analyze as R43                   # CPU only: R3_06 hides the GPU when imported

    import R3_06_analyze as A

    _t, _b, bands, _d, _r = A.decision_suite(R43.r3_design(), R43.restrict(tf))
    cmp = R43.compare_with_r3({"bands": bands}, r3_analysis)
    cmp["exact_required"] = exact_required
    cmp["ok"] = cmp["verdict_same"] and (cmp["exact"] or not exact_required)
    return cmp


def compare(reread: Dict[str, Any], r3: Dict[str, Any], r3_analysis: Dict[str, Any]) -> Dict[str, Any]:
    a = compare_teacher_forced(reread["teacher_forced"], r3)
    b = compare_regenerated(reread["regenerated"], r3)
    c = compare_verdicts(reread["teacher_forced"], r3_analysis, exact_required=bool(a.get("all_identical")))
    return {"ok": bool(a["ok"] and b["ok"] and c["ok"]), "a_teacher_forced": a, "b_regenerated": b,
            "c_verdicts": c}


# ------------------------------------------------------------------------------------------------ the GPU run
def made_with() -> Dict[str, Any]:
    rs = C.runtime_state()
    return {"pinned_unchanged": C.PINNED_UNCHANGED if not rs["pinned_differ"] else None,
            "pinned_differ": rs["pinned_differ"], "snapshot_blobs": C.snapshot_blobs(), "versions": rs["versions"],
            "environment": rs["environment"], "r4_scripts": R4E.files_sha256(), "runtime_ok": rs["ok"]}


def run(args) -> int:
    import R3_gpu as G

    r3 = json.loads(C3.EVAL_JSON.read_bytes())
    if args.batch != r3["meta"]["constants"]["batch"]:
        raise SystemExit(f"R3 read with batch {r3['meta']['constants']['batch']}; the replication reads as R3 did "
                         f"(--batch {args.batch} refused)")
    tok = R2.load_tokenizer()
    mw = made_with()
    if not mw["runtime_ok"]:
        raise SystemExit(f"the runtime to be registered does not hold: {mw}")
    r1, _ = C.find_r1_lens(full=True)
    if r1 is None or Path(r1) != C.R1_STABLE:
        raise SystemExit(f"R1's pinned lens is not at the stable path {C.R1_STABLE} (found {r1}); copy it there (§0)")
    jr = C3.check_pinned_file(C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE, full=True)
    if not jr["size_ok"] or not jr["sha256_ok"]:
        raise SystemExit(f"J++'s file is not the pinned file: {jr}")
    if C.REREAD_JSON.exists() and not args.force:
        raise SystemExit(f"{C.REREAD_JSON.name} exists: rerun with --force")
    r3_prompts = json.loads(C3.PROMPTS_PATH.read_text(encoding="utf-8"))
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    man = R4E.lens_manifest(Path(r1))
    gst = C.git_state(C.registered_files())
    meta = {"git": gst, "made_with": mw, "lenses": {k: {"path": str(v["path"]), "sha256": v["sha256"]}
                                                    for k, v in man.items()},
            "r3_eval_sha256": C.sha256_file(C3.EVAL_JSON), "r3_prompts_sha256": C.sha256_file(C3.PROMPTS_PATH),
            "forced": bool(args.force)}
    G.log_run({"event": "started", "argv": sys.argv, "git": gst, "forced": bool(args.force)}, C.REPLICATION_RUNS_LOG)
    try:
        G.set_job_cap(torch.device("cuda:0"))
        backend = E.GpuBackend(man)
        meta["made_with"]["loaded_code"] = C.loaded_code()
        with torch.no_grad():
            tf = R4E.teacher_forced(backend, tok, r3_prompts, r3, mask, batch=args.batch, meta_extra={})
            regen = R4E.run_set(backend, tok, r3_prompts, mask, special, seed=C.R3_DRAW_SEED, batch=args.batch,
                                limit=None, meta_extra={})
        out = {"schema": "R4_00/1", "meta": dict(meta, written_at=datetime.datetime.now().isoformat(timespec="seconds"),
                                                 backend=backend.info()),
               "teacher_forced": tf, "regenerated": regen}
        tmp = C.REREAD_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
        os.replace(tmp, C.REREAD_JSON)
    except BaseException as e:
        G.log_run({"event": "failed", "error": f"{type(e).__name__}: {e}"}, C.REPLICATION_RUNS_LOG)
        raise
    G.log_run({"event": "finished", "out": str(C.REREAD_JSON), "out_sha256": C.sha256_file(C.REREAD_JSON),
               "vram": G.vram_record(torch.device("cuda:0"))}, C.REPLICATION_RUNS_LOG)
    log(f"wrote {C.REREAD_JSON}; now run --compare on the CPU")
    return 0


def write_record(reread_path: Path = C.REREAD_JSON, out: Path = C.REPLICATION_JSON) -> Dict[str, Any]:
    raw = reread_path.read_bytes()
    reread = json.loads(raw)
    if reread.get("schema") != "R4_00/1":
        raise SystemExit(f"unexpected schema {reread.get('schema')}")
    runs = [json.loads(x) for x in C.REPLICATION_RUNS_LOG.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if C.REPLICATION_RUNS_LOG.is_file() else []
    if not runs or runs[-1].get("event") != "finished" or runs[-1].get("out_sha256") != hashlib.sha256(raw).hexdigest():
        raise SystemExit("the re-read is not the output of R4_00's last finished run (R4_00_runs.jsonl)")
    r3 = json.loads(C3.EVAL_JSON.read_bytes())
    r3a = json.loads(C3.ANALYSIS_JSON.read_text(encoding="utf-8"))
    rec = compare(reread, r3, r3a)
    compared_with = R4E.files_sha256()
    same_code = compared_with == reread["meta"]["made_with"].get("r4_scripts")
    rec = {"schema": "R4_00_replication/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "ok": bool(rec["ok"] and same_code),
           "reread": {"path": str(reread_path.relative_to(C.REPO)), "sha256": hashlib.sha256(raw).hexdigest()},
           "made_with": reread["meta"]["made_with"], "compared_with": compared_with,
           "same_code_for_reading_and_comparison": same_code, "git_at_run": reread["meta"]["git"], **rec}
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    return rec


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

    import R4_03_analyze as R43

    r3 = json.loads(C3.EVAL_JSON.read_bytes())
    r3a = json.loads(C3.ANALYSIS_JSON.read_text(encoding="utf-8"))
    own = R43.restrict(r3)
    own["positions"] = [dict(p, read_tok_again=p["read_tok"]) for p in own["positions"]]   # as teacher forcing records
    same = compare({"teacher_forced": own, "regenerated": own}, r3, r3a)
    check(same["ok"] and same["a_teacher_forced"]["all_identical"] and same["b_regenerated"]["found_share"] == 1.0
          and same["c_verdicts"]["exact"] and len(same["a_teacher_forced"]["readouts"]) == 7,
          "R3's own reading against itself: all seven readouts identical, every position found again, R1's record "
          "reproduced exactly")
    flip = copy.deepcopy(own)
    n_flip = 0
    for p in flip["positions"][::40]:                 # 24 of 930: 2.6%
        r = p["r"]["r1"]["58"]
        r["f"], r["fs"] = [r["f"][1]] + r["f"][1:], [r["fs"][1]] + r["fs"][1:]
        n_flip += 1
    fa = compare_teacher_forced(flip, r3)
    check(not fa["ok"] and not fa["readouts"]["r1@58"]["ok"] and fa["readouts"]["r1@55"]["ok"]
          and fa["readouts"]["r1@58"]["positions"] - fa["readouts"]["r1@58"]["filtered_top1_same"] == n_flip,
          f"{n_flip} changed top-1s of R1 at 58 fail that readout's 99% and no other")
    fc = compare_verdicts(flip, r3a, exact_required=False)
    check(not fc["ok"] and not fc["verdict_same"] and "58 rule 5: R4 False, R3 True" in fc["mismatches"],
          "(c) fails when R1's top-1 changes at 24 of R3's positions at code 58: R1's margin over the naked eye "
          "there falls below 0.05 and rule 5's verdict differs from R3's")
    few = copy.deepcopy(own)
    for p in few["positions"][::300]:                 # 4 positions
        r = p["r"]["r1"]["55"]
        r["f"], r["fs"] = [r["f"][1]] + r["f"][1:], [r["fs"][1]] + r["fs"][1:]
    fs = compare_verdicts(few, r3a, exact_required=False)
    fs_exact = compare_verdicts(few, r3a, exact_required=True)
    check(fs["ok"] and fs["verdict_same"] and fs["mismatches"] and not fs_exact["ok"]
          and all(m.startswith("55 ") for m in fs["mismatches"]),
          "(c) with R1's top-1 changed at 4 positions of code 55: the verdicts still match, which is all (c) asks "
          f"when (a) found changed top-1s; the {len(fs['mismatches'])} numbers that moved at 55 are listed, and the "
          "same file fails when every number must match")
    lost = copy.deepcopy(own)
    lost["positions"] = [p for i, p in enumerate(lost["positions"]) if i % 15 != 0]     # 62 of 930 lost: 93.3%
    fb = compare_regenerated(lost, r3)
    other = copy.deepcopy(own)
    for p in other["prompts"]:
        if p["cond"] == "mid-reply":
            p["reply_ids"] = [p["reply_ids"][0] + 1] + p["reply_ids"][1:]
    fb2 = compare_regenerated(other, r3)
    check(not fb["ok"] and fb["found_again"] == 930 - 62 and not fb2["ok"]
          and fb2["by_condition"]["mid-reply"]["found"] == 0,
          "(b) fails when 7% of R3's positions are not found again, and when every mid-reply reply differs")
    check(not compare_teacher_forced(dict(own, positions=own["positions"][1:]), r3)["ok"],
          "(a) refuses teacher-forced positions that are not R3's, in R3's order")
    # the pipeline on R3_05's stand-in: R3's prompts teacher-forced and regenerated, then compared (and failing)
    tok = R2.load_tokenizer()
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    backend = E.FakeBackend(tok)
    r3p = json.loads(C3.PROMPTS_PATH.read_text(encoding="utf-8"))
    tf = R4E.teacher_forced(backend, tok, r3p, r3, mask, batch=8, meta_extra={"selftest": True})
    regen = R4E.run_set(backend, tok, r3p, mask, special, seed=C.R3_DRAW_SEED, batch=8, limit=2,
                        meta_extra={"selftest": True})
    fake = compare({"teacher_forced": tf, "regenerated": regen}, r3, r3a)
    check(len(tf["positions"]) == 930 and fake["a_teacher_forced"]["n_read_token_differs"] == 0
          and not fake["ok"] and fake["b_regenerated"]["found_share"] < 0.05,
          "on the stand-in, all 930 of R3's positions are rebuilt (the token read is R3's at each, as (a) requires) "
          "and read; the stand-in's readouts and words are not the model's, and the comparison says so")
    wrong_tok = copy.deepcopy(own)
    wrong_tok["positions"][7]["read_tok_again"] = "<not R3's>"
    check(not compare_teacher_forced(wrong_tok, r3)["ok"],
          "(a) fails when a rebuilt input reads another token than R3 read at that position")
    mw = made_with()
    check(mw["runtime_ok"] and set(mw["r4_scripts"]) == set(R4E.R4_SCRIPTS) and mw["environment"].get("numpy"),
          "the record keeps the runtime, the environment and the sha256 of R4_common, R4_02_eval, R4_00_replicate and "
          "R4_03_analyze (the reading and the comparison)")
    print(f"R4_00 self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, 'CUDA_VISIBLE_DEVICES="" .venv/bin/python -I '
                          'experiments/scripts/R4_00_replicate.py --selftest', done, time.time() - t_start)
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R4_selftests.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.record)
    if a.compare:
        rec = write_record()
        ta = rec["a_teacher_forced"]
        print("(a) teacher-forced, filtered top-1 as R3's: " + ", ".join(
            f"{k} {v['filtered_share']:.4f}" for k, v in (ta.get("readouts") or {}).items()))
        print(f"(b) regenerated: {rec['b_regenerated']['found_again']} of {rec['b_regenerated']['r3_positions']} found "
              f"again ({rec['b_regenerated']['found_share']:.4f})")
        print(f"(c) R1's verdicts as R3's: {rec['c_verdicts']['verdict_same']}; numbers exact: "
              f"{rec['c_verdicts']['exact']} ({len(rec['c_verdicts']['mismatches'])} differ)")
        print(f"replication record: {'PASSES' if rec['ok'] else 'FAILS'}; wrote {C.REPLICATION_JSON}")
        return 0 if rec["ok"] else 1
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
