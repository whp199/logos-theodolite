#!/usr/bin/env python3
"""R4 02 (GPU): read R4's 640 fresh prompts through J++, the naked eye and R1's lens at code 55 and 58 (the naked eye at
59 too), filtered and not (R4_PREREG.md §4). The analysis is R4_03_analyze.py.

R3_05_eval.py's reading, narrowed to R4's three readouts and three layers, through R3_05's own pieces, imported
unchanged: the backends (GpuBackend loads the served checkpoint itself with R3_gpu.load_w4a16; FakeBackend is the CPU
stand-in), greedy decoding by full forwards, and the read positions (R2's targets, the mid-reply draws with a seed).
R3_06_analyze is not imported here: it hides the GPU when imported.
  1. Checks before the model: R4_prompts.json validates with R4's counts and composition, and its sha256 is the one a
     passing R4_01_check.json recorded; R2's filter has R2_01_check.json's class counts; J++'s and R1's lens files
     have their pinned size and full sha256; every file R4 imports or reads has its pinned sha256, the model
     snapshot's files point to their pinned blobs, torch and transformers are the registered versions; the
     replication record (R4_00_replication.json) exists, passed, and was made at this runtime with these R4 scripts;
     R4's registered files are committed and unchanged at HEAD (else --allow-uncommitted, a deviation); no earlier
     output (else --force). Every run past these checks is appended to R4_02_runs.jsonl (started/finished/failed).
  2. Greedy continuations by full forwards (no KV cache), as R2 and R3: 8 tokens after each chat-no-answer, chat-begun
     and bare prompt, 48-token replies to the mid-reply questions; the model's own word is the argmax of the final
     norm, W_U and softcap of resid_post[59].
  3. Read positions as R2 and R3 (R2_common.first_content, R2's stop rule); the mid-reply draws are
     default_rng([20261028, mid-reply index, draw]), draw 0 "content_next" (3 positions, reported only) and draw 1
     "any" (6 positions, the decision condition).
  4. resid_post at code 55, 58 and 59 at the read positions, one batched forward per group of prompts.
  5. Readouts through the instrument's own functions (jpp_readout_logits / unembed_gemma4, bf16): J++ and R1 at 55 and
     58, the naked eye at 55, 58 and 59. Per readout: top-5 ids and texts unfiltered ("u", "us") and with R2's filter
     ("f", "fs"); the rank of the next word-piece unfiltered ("ur") and of the content target filtered ("fr"); whether
     the unfiltered top-1 is content ("u0c"). No J++-filtered readout is recorded, so R3's S5 is not run.
Writes R4_02_eval.json (schema R4_02/1, in R3_05_eval.json's layout). The same functions read R3's prompts again for
the replication record (R4_00_replicate.py), teacher-forced and regenerated.

Run (GPU, server stopped):  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R4_02_eval.py
                                [--batch 8] [--force] [--allow-uncommitted]
Every check before the model (CPU):  CUDA_VISIBLE_DEVICES="" ... R4_02_eval.py --check-only
Self-test (CPU, R3_05's tiny random stand-in for the model with the real tokenizer and filter; R3's prompts stand in for
R4's until R4_prompts.json is written):  CUDA_VISIBLE_DEVICES="" ... R4_02_eval.py --selftest --out X.json [--record]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
if "--selftest" in sys.argv or "--check-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # the CPU modes never see the GPU
sys.dont_write_bytecode = True

import argparse
import copy
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402
import R3_05_eval as E  # noqa: E402  (R3's reading pieces, unchanged)
import R4_common as C  # noqa: E402

T0 = time.time()
# the code a replication record must have used: its GPU reading and its comparison (R4_00 --compare runs R4_03's)
R4_SCRIPTS = ("R4_common.py", "R4_02_eval.py", "R4_00_replicate.py", "R4_03_analyze.py")
STATES_PT = C.RESULTS / "R4_02_states.pt"              # gitignored: resid_post at 55, 58, 59 at every read position


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ the reading
def lens_manifest(r1_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """The two lens files R4 reads, with their pins (the naked eye has none)."""
    return {"jpp": {"path": C.JPP_BASE_PATH, "sha256": C.JPP_BASE_SHA256, "size": C.JPP_BASE_SIZE, "format": "pt"},
            "r1": {"path": r1_path, "sha256": C.R1_SHA256, "size": C.R1_SIZE, "format": "pt"}}


def prepare(tok, data: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every prompt in file order with the text read and its ids (``limit``: the first N per condition)."""
    items = R2.prompt_items(data)
    if limit:
        items = [it for c in C.CONDITIONS for it in [x for x in items if x["cond"] == c][:limit]]
    for it in items:
        it["text_read"] = R2.prompt_text(tok, it)
        it["ids"] = R2.text_ids(tok, it["text_read"])
    return items


def generate(backend, items: List[Dict[str, Any]], batch: int) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """R3_05's greedy decoding: 8 tokens after every non-mid-reply prompt, 48-token replies to the mid-reply ones."""
    first = [it for it in items if it["cond"] != "mid-reply"]
    mids = [it for it in items if it["cond"] == "mid-reply"]
    conts = {it["pid"]: c for it, c in zip(first, E.greedy(backend, [it["ids"] for it in first], C.CONT_WINDOW, batch,
                                                             "continuations"))}
    replies = {it["pid"]: r for it, r in zip(mids, E.greedy(backend, [it["ids"] for it in mids], C.MID_NEW_TOKENS,
                                                            batch, "mid-reply replies"))}
    return conts, replies


def read_states(backend, pos_recs: List[Dict[str, Any]], batch: int) -> torch.Tensor:
    """resid_post at READ_LAYERS for every read position [N, L, D] (bf16): one forward per group of `batch` prompts,
    the prompts in the order they first appear, each read at all its positions (as R3_05)."""
    by_pid: Dict[str, List[int]] = {}
    for j, p in enumerate(pos_recs):
        by_pid.setdefault(p["pid"], []).append(j)
    pids = list(by_pid)
    H = None
    for b0 in range(0, len(pids), batch):
        grp = pids[b0:b0 + batch]
        st = backend.states([pos_recs[by_pid[pid][0]]["seq"] for pid in grp],
                            [[pos_recs[j]["pos"] for j in by_pid[pid]] for pid in grp], C.READ_LAYERS)
        if H is None:
            H = torch.empty(len(pos_recs), len(C.READ_LAYERS), st[0].shape[-1], dtype=torch.bfloat16)
        for pid, s in zip(grp, st):
            for q, j in enumerate(by_pid[pid]):
                H[j] = s[:, q].to(torch.bfloat16)
    return H


def readouts(backend, tok, H: torch.Tensor, pos_recs: List[Dict[str, Any]], mask: torch.Tensor,
             s_of) -> List[Dict[str, Dict[str, Any]]]:
    """Every readout of KINDS at its layers, per position: R3_05's record without J++'s filter (u, us, f, fs, ur, fr,
    u0c)."""
    is_content = mask.numpy()
    dev = backend.dev
    mask_dev = mask.to(dev)
    N = len(pos_recs)
    next_ids = torch.tensor([p["next_id"] for p in pos_recs], device=dev)
    ct_ids = torch.tensor([p["ct"][1] if p["ct"] else -1 for p in pos_recs], device=dev)
    R: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(N)]
    CH = 64
    for li, l in enumerate(C.READ_LAYERS):
        for kind in [k for k in C.KINDS if l in C.LENS_LAYERS[k]]:
            J = backend.lens_matrix(kind, l)
            for s0 in range(0, N, CH):
                lg = backend.readout_logits(kind, l, H[s0:s0 + CH, li], J)
                u = torch.topk(lg, C.TOPK, dim=-1).indices
                lgf = lg.masked_fill(~mask_dev, float("-inf"))
                f = torch.topk(lgf, C.TOPK, dim=-1).indices
                nx, ct = next_ids[s0:s0 + CH], ct_ids[s0:s0 + CH]
                ur = (lg > lg.gather(1, nx[:, None])).sum(1) + 1
                fr = (lgf > lgf.gather(1, ct.clamp(min=0)[:, None])).sum(1) + 1
                u_l, f_l, ur_l, fr_l = u.tolist(), f.tolist(), ur.tolist(), fr.tolist()
                for q in range(len(u_l)):
                    R[s0 + q].setdefault(kind, {})[str(l)] = {
                        "u": u_l[q], "us": [s_of(i) for i in u_l[q]], "f": f_l[q], "fs": [s_of(i) for i in f_l[q]],
                        "ur": int(ur_l[q]), "fr": int(fr_l[q]) if int(ct[q]) >= 0 else None,
                        "u0c": bool(is_content[u_l[q][0]])}
                del lg, lgf
            backend.release(kind, J)
        log(f"  readouts at layer {l}")
    return R


def position_record(j: int, p: Dict[str, Any], r: Dict[str, Dict[str, Any]], s_of, is_content) -> Dict[str, Any]:
    """R3_05's position record."""
    m = r["naive"]["59"]
    ct = p["ct"]
    return {"i": j, "pid": p["pid"], "cond": p["cond"], "draws": p["draws"], "k": p["k"], "pos": p["pos"],
            "read_tok": s_of(p["seq"][p["pos"]]), "next_id": p["next_id"], "next_str": s_of(p["next_id"]),
            "next_content": bool(is_content[p["next_id"]]), "cont_ids": p["cont_ids"],
            "cont_strs": [s_of(i) for i in p["cont_ids"]], "ct_id": ct[1] if ct else None,
            "ct_str": s_of(ct[1]) if ct else None, "ct_offset": ct[0] if ct else None,
            "ct_prev_str": s_of(p["ct_prev"]) if ct else None,
            "ct_collapse": bool(ct) and R2.word_key(s_of(ct[1])) in R2.COLLAPSE_TARGETS,
            "ct_formatlike": bool(ct) and R2.formatlike_target(s_of(ct[1]), s_of(p["ct_prev"])),
            "model_top1_id": m["u"][0], "model_top1_str": m["us"][0], "model_agrees": m["u"][0] == p["next_id"],
            "r": r}


def string_cache(tok):
    strs: Dict[int, str] = {}

    def s_of(i: int) -> str:
        i = int(i)
        if i not in strs:
            strs[i] = R2.token_text(tok, i)
        return strs[i]
    return s_of


def summary_of(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    agree: Dict[str, List[bool]] = {}
    for rec in positions:
        agree.setdefault(rec["cond"], []).append(rec["model_agrees"])
    return {
        "n_positions": {c: sum(p["cond"] == c for p in positions) for c in C.CONDITIONS},
        "n_mid_any": sum(1 for p in positions if p["draws"] and "any" in p["draws"]),
        "n_with_content_target": {c: sum(p["cond"] == c and p["ct_id"] is not None for p in positions)
                                  for c in C.CONDITIONS},
        "n_collapse_target": {c: sum(p["cond"] == c and p["ct_collapse"] for p in positions) for c in C.CONDITIONS},
        "model_agrees_with_next": {c: round(sum(v) / len(v), 4) for c, v in agree.items()},
    }


def constants(seed: int, batch: int, limit: Optional[int]) -> Dict[str, Any]:
    return {"READ_LAYERS": list(C.READ_LAYERS), "LENS_LAYERS": {k: list(v) for k, v in C.LENS_LAYERS.items()},
            "KINDS": list(C.KINDS), "BANDS": [[n, list(ls)] for n, ls in C.BANDS], "CONT_WINDOW": C.CONT_WINDOW,
            "MID_NEW_TOKENS": C.MID_NEW_TOKENS, "MID_DRAW_SIZES": C.MID_DRAW_SIZES, "DRAW_SEED": seed,
            "STOP_IDS": list(R2.STOP_IDS), "TOPK": C.TOPK, "COLLAPSE_TARGETS": list(R2.COLLAPSE_TARGETS),
            "batch": batch, "limit": limit}


def run_set(backend, tok, data: Dict[str, Any], mask: torch.Tensor, is_special, *, seed: int, batch: int,
            limit: Optional[int], meta_extra: Dict[str, Any], states_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate, place the read positions, read: the whole reading of one prompts file (R3_05's run_set, R4's
    readouts and layers)."""
    is_content = mask.numpy()
    if len(is_content) != backend.vocab_rows:
        raise SystemExit(f"the mask has {len(is_content)} ids, W_U has {backend.vocab_rows} rows")
    items = prepare(tok, data, limit)
    checked = set()
    for it in items:
        if it["cond"] not in checked:
            backend.check_text(it, it["text_read"], it["ids"])
            checked.add(it["cond"])
    log(f"{len(items)} prompts: " + ", ".join(f"{c} {sum(it['cond'] == c for it in items)}" for c in C.CONDITIONS))
    t = time.time()
    conts, replies = generate(backend, items, batch)
    t_gen = time.time() - t
    pos_recs = E.build_positions(items, conts, replies, is_content, is_special, seed)
    t = time.time()
    H = read_states(backend, pos_recs, batch)
    t_read = time.time() - t
    t = time.time()
    s_of = string_cache(tok)
    R = readouts(backend, tok, H, pos_recs, mask, s_of)
    t_lens = time.time() - t
    positions = [position_record(j, p, R[j], s_of, is_content) for j, p in enumerate(pos_recs)]
    if states_path is not None:                    # for later readings on the CPU, as R3_05 kept its states
        torch.save({"H": H, "layers": list(C.READ_LAYERS),
                    "positions": [[p["pid"], p["pos"], p["k"]] for p in positions]}, states_path)
    prompts_out = []
    for it in items:
        d = {k: it[k] for k in ("pid", "cond", "user", "begun", "text", "kind", "format_likely", "about_model", "topic")
             if k in it}
        d.update(n_ids=len(it["ids"]), read_tok=s_of(it["ids"][-1]))
        if it["cond"] == "mid-reply":
            d.update(reply_ids=it["reply_ids"], reply_text=tok.decode(it["reply_ids"], skip_special_tokens=False),
                     reply_stopped=it["reply_stopped"], draws=it["draws"])
        else:
            d.update(cont_ids=conts[it["pid"]], cont_text=tok.decode(conts[it["pid"]], skip_special_tokens=False))
        prompts_out.append(d)
    return {"schema": "R4_02/1", "meta": dict(meta_extra, **{
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
        "argv": sys.argv, "backend": backend.info(), "constants": constants(seed, batch, limit),
        "timing_s": {"generation": round(t_gen, 1), "reading": round(t_read, 1), "readouts": round(t_lens, 1)},
        "summary": summary_of(positions)}), "prompts": prompts_out, "positions": positions}


def teacher_forced(backend, tok, data: Dict[str, Any], stored: Dict[str, Any], mask: torch.Tensor, *, batch: int,
                   meta_extra: Dict[str, Any]) -> Dict[str, Any]:
    """Read a stored reading's own inputs again (R4_PREREG.md §4.2(a)): each prompt as R2_common composes it, the
    mid-reply prompts followed by the stored reply, read at the stored positions, batched as R3_05 batched them. The
    stored positions are kept whole (their targets, draws and context); only their readouts ("r") and the naked eye's
    top-1 at 59 (model_top1_*, model_agrees) are new. Refuses when a prompt's ids are not the stored count."""
    sp = {p["pid"]: p for p in stored["prompts"]}
    items = {it["pid"]: it for it in prepare(tok, data) if it["pid"] in sp}
    bad = [pid for pid, p in sp.items() if pid not in items or p.get("n_ids") != len(items[pid]["ids"])]
    if bad:
        raise SystemExit(f"the prompts do not rebuild to the stored inputs: {bad[:5]}")
    pos_recs = []
    for p in stored["positions"]:
        ids = items[p["pid"]]["ids"]
        seq = ids + list(sp[p["pid"]]["reply_ids"]) if p["cond"] == "mid-reply" else ids
        pos_recs.append({"pid": p["pid"], "seq": seq, "pos": p["pos"], "next_id": p["next_id"],
                         "ct": (p["ct_offset"], p["ct_id"]) if p.get("ct_id") is not None else None})
    t = time.time()
    H = read_states(backend, pos_recs, batch)
    s_of = string_cache(tok)
    R = readouts(backend, tok, H, pos_recs, mask, s_of)
    positions = []
    for j, (p, rec) in enumerate(zip(stored["positions"], pos_recs)):
        q = copy.deepcopy({k: v for k, v in p.items() if k != "r"})
        m = R[j]["naive"]["59"]
        q.update(r=R[j], model_top1_id=m["u"][0], model_top1_str=m["us"][0],
                 model_agrees=m["u"][0] == p["next_id"],
                 read_tok_again=s_of(rec["seq"][rec["pos"]]))
        positions.append(q)
    return {"schema": "R4_00_tf/1", "meta": dict(meta_extra, **{
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "backend": backend.info(),
        "constants": constants(None, batch, None), "timing_s": {"reading": round(time.time() - t, 1)},
        "teacher_forced": True}), "prompts": copy.deepcopy(stored["prompts"]), "positions": positions}


# ------------------------------------------------------------------------------------------------ checks
def files_sha256(names: Sequence[str] = R4_SCRIPTS) -> Dict[str, str]:
    return {n: C.sha256_file(C.SCRIPTS / n) for n in names if (C.SCRIPTS / n).is_file()}


def replication_state(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Whether the replication record (R4_00_replication.json) exists, passed, and is the comparison of the re-read
    that R4_00's last finished run wrote (sha256 chain), made at this runtime, environment and snapshot with these
    R4 scripts (R4_PREREG.md §4.1, §4.2)."""
    if not C.REPLICATION_JSON.is_file():
        return {"ok": False, "missing": True, "why": "R4_00_replication.json is not written yet (§4.2)"}
    rec = json.loads(C.REPLICATION_JSON.read_text(encoding="utf-8"))
    why = []
    if not rec.get("ok"):
        why.append("the replication record did not pass")
    last = C.last_finished(C.REPLICATION_RUNS_LOG)
    reread_sha = C.sha256_file(C.REREAD_JSON) if C.REREAD_JSON.is_file() else None
    if not (reread_sha and (rec.get("reread") or {}).get("sha256") == reread_sha and last
            and last.get("out_sha256") == reread_sha):
        why.append("the record is not the comparison of R4_00_reread.json as R4_00's last finished run wrote it")
    made = rec.get("made_with") or {}
    if made.get("pinned_unchanged") != C.PINNED_UNCHANGED or made.get("versions") != runtime.get("versions"):
        why.append("the replication record was made at another runtime (pinned files or library versions)")
    if made.get("environment") != runtime.get("environment"):
        why.append(f"the replication record was made in another environment: {made.get('environment')} against "
                   f"{runtime.get('environment')}")
    if made.get("snapshot_blobs") != C.SNAPSHOT_BLOBS:
        why.append("the replication record was made on another model snapshot")
    if made.get("r4_scripts") != files_sha256() or rec.get("compared_with") != files_sha256():
        why.append("the replication record was made or compared with other R4 scripts: "
                   f"{sorted(k for k, v in files_sha256().items() if (made.get('r4_scripts') or {}).get(k) != v)}")
    return {"ok": not why, "missing": False, "why": "; ".join(why), "record": rec, "sha256":
            C.sha256_file(C.REPLICATION_JSON)}


def checks_before_model(full_hash: bool = True, prompts_required: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"failed": [], "pending": []}
    tok = R2.load_tokenizer()
    out["tokenizer_ids"] = len(tok)
    if C.PROMPTS_PATH.is_file():
        val = C.validate(C.load_prompts(), tok)
        if not val["ok"]:
            out["failed"].append(f"R4_prompts.json invalid: {val['problems'][:3]}")
        chk = json.loads(C.CHECK_JSON.read_text(encoding="utf-8")) if C.CHECK_JSON.is_file() else {}
        if not chk.get("ok") or chk.get("prompts_sha256") != C.sha256_file(C.PROMPTS_PATH):
            out["failed" if prompts_required else "pending"].append(
                "R4_01_check.json does not pass this exact R4_prompts.json; rerun R4_01_check.py")
    else:
        (out["failed"] if prompts_required else out["pending"]).append(
            "R4_prompts.json is not written yet (the registration's prompts)")
    fr = R2.filter_report(tok)
    r2chk = json.loads(R2.CHECK_JSON.read_text(encoding="utf-8"))
    if not fr["ok"] or fr["counts"] != (r2chk.get("filter") or {}).get("counts"):
        out["failed"].append("R2's filter fails its spot checks or differs from R2_01_check.json's counts")
    out["r2_filter_counts"] = fr["counts"]
    jr = C3.check_pinned_file(C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE, full=full_hash)
    r1, seen = C.find_r1_lens(full=full_hash)
    out["lens_files"] = {"jpp": jr, "r1": {"found": str(r1) if r1 else None, "candidates": seen}}
    if not jr["size_ok"] or jr["sha256_ok"] is False:
        out["failed"].append(f"J++: not the pinned file ({jr})")
    if r1 is None:
        out["failed"].append("R1's lens: no candidate is the pinned file")
    elif Path(r1) != C.R1_STABLE:
        out["pending"].append(f"R1's lens is read from {r1}: copy it to the stable path {C.R1_STABLE} (§0)")
    rs = C.runtime_state()
    out["runtime"] = rs
    if not rs["ok"]:
        out["failed"].append(f"the registered runtime does not hold: pinned files differ {rs['pinned_differ']}, "
                             f"snapshot files differ {rs['snapshot_differ']}, versions {rs['versions']}")
    rep = replication_state(rs)
    out["_replication_record"] = rep.pop("record", None)
    out["replication"] = rep
    if not rep["ok"]:
        (out["pending"] if rep.get("missing") and not prompts_required else out["failed"]).append(rep["why"])
    if C.EVAL_JSON.exists():
        out["pending"].append(f"{C.EVAL_JSON.name} exists: a run would need --force")
    out["_tok"] = tok
    out["_r1"] = r1
    return out


def require_registered(allow: bool) -> Dict[str, Any]:
    """R4's registered files committed and unchanged at HEAD (else --allow-uncommitted, a recorded deviation)."""
    gst = C.git_state(C.registered_files() + [C.REPO / r for r in C.PINNED_UNCHANGED])
    if gst["dirty"] or gst["head"] is None:
        msg = f"registered files not committed at HEAD {gst['head']}: {gst['dirty'][:8]}"
        if not allow:
            raise SystemExit(msg + "; commit them first (R4_PREREG.md §0), or pass --allow-uncommitted (a deviation)")
        log("WARNING (a deviation): " + msg)
        gst["deviation"] = msg
    return gst


# ------------------------------------------------------------------------------------------------ self-test
def selftest(out_path: Path, limit: int, record: bool = False) -> int:
    import R3_06_analyze as A                     # the CPU self-test only: R3_06 hides the GPU when imported

    fails: List[str] = []
    done: List[Tuple[bool, str]] = []
    t_start = time.time()

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        done.append((bool(ok), what))
        if not ok:
            fails.append(what)

    if out_path.resolve().parent == C.RESULTS.resolve():
        raise SystemExit("--selftest must write outside experiments/results (--out)")
    co = checks_before_model(full_hash=False, prompts_required=False)
    tok = co.pop("_tok")
    co.pop("_r1"), co.pop("_replication_record")
    check(not co["failed"], f"every check before the model passes on the CPU (sizes only here); waiting for: "
                            f"{co['pending']}; failed: {co['failed']}")
    # the draws: R3_05's function with R4's seed; draw 0 is content_next (3), draw 1 is any (6)
    rng = np.random.default_rng(5)
    is_c = rng.random(len(tok)) < 0.9
    ok_draws = True
    for i, n in enumerate((0, 2, 9, 48, 30, 48)):
        r = list(rng.integers(0, len(tok), size=n))
        got = E.draw_mid_positions(r, is_c, i, C.DRAW_SEED)
        elig = list(range(max(0, n - 1)))
        cn = [k for k in elig if bool(is_c[int(r[k + 1])])]
        want: Dict[int, List[str]] = {}
        for d, (name, pool, size) in enumerate((("content_next", cn, 3), ("any", elig, 6))):
            if pool:
                g = np.random.default_rng([20261028, i, d])
                for p in sorted(int(x) for x in g.choice(len(pool), size=min(size, len(pool)), replace=False)):
                    want.setdefault(pool[p], []).append(name)
        ok_draws &= got == dict(sorted(want.items()))
    check(ok_draws, "the mid-reply draws are default_rng([20261028, mid-reply index, draw]) with draw 0 content_next "
                    "(3) and draw 1 any (6)")
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    backend = E.FakeBackend(tok)
    data = C.load_prompts() if C.PROMPTS_PATH.is_file() else json.loads(C3.PROMPTS_PATH.read_text(encoding="utf-8"))
    out = run_set(backend, tok, data, mask, special, seed=C.DRAW_SEED, batch=4, limit=limit,
                  meta_extra={"selftest": True})
    out_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    pos = out["positions"]
    ok = bool(pos) and {p["cond"] for p in pos} == set(C.CONDITIONS)
    for p in pos:
        ok &= set(p["r"]) == set(C.KINDS)
        for k in C.KINDS:
            ok &= set(p["r"][k]) == {str(l) for l in C.LENS_LAYERS[k]}
            for r in p["r"][k].values():
                ok &= all(key in r for key in ("u", "us", "f", "fs", "ur", "fr", "u0c")) and "fj" not in r
                ok &= all(bool(mask[i]) for i in r["f"])
    check(ok, f"{len(pos)} read positions in every condition: J++ and R1 at 55 and 58, the naked eye at 55, 58 and 59, "
              f"each with top-5 unfiltered and under R2's filter (content ids only), the ranks, no J++-filtered "
              f"readout")
    D = A.Design(**C.DESIGN_FIELDS, overlap=dict(C.FIT_OVERLAP))
    t = A.Table(D, out)
    _t, _b, bands, decision, ref = A.decision_suite(D, out)
    check(set(bands) == {"r1"} and set(bands["r1"]) == {"55", "58"} and not ref
          and all(ev["holm_m"] == 2 for ev in bands["r1"].values()) and len(t.values("f_word", "r1", 55)) == len(pos),
          "R3_06's rule reads the output under R4's design: R1 at bands 55 and 58, Holm over 2, nothing reference-only")
    # teacher forcing: the output's own inputs read again at its own positions give the same readouts
    tf = teacher_forced(backend, tok, data, dict(out, prompts=[dict(p) for p in out["prompts"]]), mask, batch=3,
                        meta_extra={"selftest": True})
    same = all(tp["r"][k][l]["f"] == p["r"][k][l]["f"] and tp["r"][k][l]["u"] == p["r"][k][l]["u"]
               and tp["read_tok_again"] == p["read_tok"]
               for tp, p in zip(tf["positions"], pos) for k in C.KINDS for l in p["r"][k])
    check(same and len(tf["positions"]) == len(pos) and tf["positions"][0]["ct_id"] == pos[0]["ct_id"],
          "teacher forcing rebuilds every stored input (the prompt, and the stored reply for mid-reply), keeps the "
          "stored targets, and reads the stored positions again with the same readouts (the stand-in ignores "
          "batching, so this says nothing about batch invariance on the GPU)")
    try:
        bad = dict(out, prompts=[dict(p, n_ids=p["n_ids"] + 1) if i == 0 else p for i, p in enumerate(out["prompts"])])
        teacher_forced(backend, tok, data, bad, mask, batch=3, meta_extra={})
        refused = False
    except SystemExit:
        refused = True
    check(refused, "teacher forcing refuses a stored reading whose prompt ids differ from the rebuilt ones")
    print(f"R4_02_eval self-test: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    if record:
        C.record_selftest(Path(__file__).name, 'CUDA_VISIBLE_DEVICES="" .venv/bin/python -I '
                          f'experiments/scripts/R4_02_eval.py --selftest --out <scratch> --limit {limit}', done,
                          time.time() - t_start)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ the run
def run(args) -> int:
    import R3_gpu as G

    co = checks_before_model(full_hash=True, prompts_required=True)
    tok, r1, rep = co.pop("_tok"), co.pop("_r1"), co.pop("_replication_record")
    waiting = [p for p in co["pending"] if not (p.endswith("would need --force") and args.force)]
    if co["failed"] or waiting:
        raise SystemExit(f"checks before the model: failed {co['failed']}; waiting for {waiting}")
    gst = require_registered(args.allow_uncommitted)
    man = lens_manifest(Path(r1))
    cls, _ = R2.classify_vocab(tok)
    mask = torch.from_numpy(cls == 0)
    special = R2.special_mask(cls)
    meta_extra = {"git": gst, "lenses": {k: {"path": str(v["path"]), "sha256": v["sha256"]} for k, v in man.items()},
                  "checks_before_model": co, "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                  "prompts_sha256": C.sha256_file(C.PROMPTS_PATH), "files_sha256": files_sha256(),
                  "replication_sha256": co["replication"]["sha256"], "forced": bool(args.force)}
    G.log_run({"event": "started", "argv": sys.argv, "git": gst, "forced": bool(args.force)}, C.RUNS_LOG)
    try:
        G.set_job_cap(torch.device("cuda:0"))
        backend = E.GpuBackend(man)
        lc = C.loaded_code()
        meta_extra["loaded_code"] = lc
        if lc != (rep.get("made_with") or {}).get("loaded_code"):
            raise SystemExit("the code loaded for this run differs from the code the replication record ran: "
                             f"{sorted(set(lc.items()) ^ set(((rep.get('made_with') or {}).get('loaded_code') or {}).items()))[:6]}")
        with torch.no_grad():
            out = run_set(backend, tok, C.load_prompts(), mask, special, seed=C.DRAW_SEED, batch=args.batch,
                          limit=None, meta_extra=meta_extra, states_path=STATES_PT)
        tmp = C.EVAL_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
        os.replace(tmp, C.EVAL_JSON)
        log(f"wrote {C.EVAL_JSON} ({C.EVAL_JSON.stat().st_size / 1e6:.1f} MB)")
    except BaseException as e:
        G.log_run({"event": "failed", "error": f"{type(e).__name__}: {e}"}, C.RUNS_LOG)
        raise
    G.log_run({"event": "finished", "out": str(C.EVAL_JSON), "out_sha256": C.sha256_file(C.EVAL_JSON),
               "vram": G.vram_record(torch.device("cuda:0"))}, C.RUNS_LOG)
    log("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="--selftest: where to write the output (outside experiments/results)")
    ap.add_argument("--limit", type=int, default=0, help="--selftest: the first N prompts per condition")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true", help="with --selftest: keep the result in R4_selftests.json")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        if not a.out:
            raise SystemExit("--selftest needs --out (outside experiments/results)")
        return selftest(Path(a.out), a.limit or 3, a.record)
    if a.limit:
        raise SystemExit("--limit is for --selftest only: the registered run reads every prompt")
    if a.check_only:
        co = checks_before_model(full_hash=True, prompts_required=False)
        co.pop("_tok"), co.pop("_r1"), co.pop("_replication_record")
        print(json.dumps(co, indent=1, default=str))
        print("check-only:", "FAILED" if co["failed"] else ("PASS, waiting for: " + "; ".join(co["pending"])
                                                            if co["pending"] else "PASS"))
        return 1 if co["failed"] else 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
