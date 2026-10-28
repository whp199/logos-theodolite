#!/usr/bin/env python3
"""R4 01 (CPU only; loads no model, uses no GPU, no network): the checks made before registration.

1. Pins. R1's pinned lens is found by its full sha256 at the stable path outside every worktree (R4_PREREG.md §0); J++'s
   file has its pinned size; every file R4 imports or reads has its pinned sha256, the model snapshot's files point to
   their pinned blobs, and torch and transformers are the registered versions (R4_common.runtime_state).
2. R2's filter has the class counts R2_01_check.json recorded and passes its spot checks.
3. Freshness. Every R4 prompt (each question, begun answer, question + begun answer, bare text) against: every string
   literal of experiments/scripts/*.py and every string of experiments/results/*.json in the main checkout and in
   every worktree under .claude/worktrees (R4's own files excluded; files deduplicated by content), which holds R2's
   and R3's results, R3's greedy replies and R1's corpus; R2's and R3's prompt files; the N1B corpus (decoded); every
   cached WikiText row and every J++ item string (R3's cache, checked against R3_sources.json); and the other R4
   prompts. Matching is R2_common.freshness_check, R2's rule unchanged. Matches inside third-party texts are recorded
   by source and sha256, never by text.
4. The prompts. R4_prompts.json validates with R4's counts and composition (R4_common.validate), and FIT_OVERLAP
   names R4 prompts only.

Writes experiments/results/R4_01_check.json and exits 1 if any check fails.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R4_01_check.py
       [--before-prompts] [--out PATH] [--corpus-cache PATH]
A part of the prompts while they are written (one or more conditions; written nowhere under experiments/results):
       ... R4_01_check.py --fragment PART.json [--corpus-cache PATH]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import argparse
import datetime
import glob
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402
import R4_common as C  # noqa: E402

T0 = time.time()
THIRD_PARTY = ("wikitext:", "jpp_items:", "results/N1B_corpus.jsonl")


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def checkouts() -> List[Path]:
    """The main checkout, this one if it differs, and every worktree under .claude/worktrees."""
    roots = [C.MAIN_CHECKOUT]
    if C.REPO.resolve() != C.MAIN_CHECKOUT.resolve():
        roots.append(C.REPO)
    roots += sorted(p for p in C.WORKTREES.glob("*") if (p / "experiments").is_dir())
    return roots


def build_corpus(tok) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    corpus: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}                              # file sha256 -> where it was read
    n_py = n_js = 0
    files_py = files_js = 0
    for root in checkouts():
        for pattern, kind in (("experiments/scripts/*.py", "py"), ("experiments/results/*.json", "json")):
            for f in sorted(glob.glob(str(root / pattern))):
                name = os.path.basename(f)
                if name.startswith("R4_"):
                    continue
                sha = C.sha256_file(Path(f))
                if sha in seen:
                    continue
                seen[sha] = f
                if kind == "py":
                    strs = R2.python_string_literals(Path(f))
                    files_py += 1
                    n_py += len(strs)
                    corpus += [(f"scripts/{name}", s) for s in strs]
                else:
                    try:
                        obj = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
                    except ValueError:
                        continue
                    strs: List[str] = []
                    R2.collect_strings(obj, strs)
                    files_js += 1
                    n_js += len(strs)
                    corpus += [(f"results/{name}", s) for s in strs]
    n_prompt_files = 0
    for path in (R2.PROMPTS_PATH, C3.PROMPTS_PATH):
        q = C.queries_of(json.loads(path.read_text(encoding="utf-8")))
        corpus += [(f"scripts/{path.name}", s) for _, s in q]
        n_prompt_files += len(q)
    n1b = C.MAIN_CHECKOUT / "experiments" / "results" / "N1B_corpus.jsonl"
    n_n1b = 0
    if n1b.is_file():
        for line in n1b.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            strs = []
            if isinstance(d, dict) and isinstance(d.get("ids"), list):
                strs.append(tok.decode([int(i) for i in d["ids"]], skip_special_tokens=False))
            R2.collect_strings({k: v for k, v in d.items() if k != "ids"} if isinstance(d, dict) else d, strs)
            n_n1b += len(strs)
            corpus += [("results/N1B_corpus.jsonl", s) for s in strs]
    rows, items = C.r3_cache_rows_and_items()
    n_wt = 0
    for i, t in enumerate(rows):
        if len(t.split()) >= 3:
            corpus.append((f"wikitext:row{i}", t))
            n_wt += 1
    n_it = 0
    for slug, its in items.items():                        # items by their place in the pinned file, not by name
        for j, it in enumerate(its):
            for k, v in (("prompt", it.get("prompt")), ("target", it.get("target"))):
                if isinstance(v, str):
                    corpus.append((f"jpp_items:{slug}/#{j}/{k}", v))
                    n_it += 1
            for v in it.get("intermediates", []):
                corpus.append((f"jpp_items:{slug}/#{j}/intermediate", v))
                n_it += 1
    sources = {"checkouts": [str(r) for r in checkouts()], "scripts_py_files": files_py, "scripts_strings": n_py,
               "results_json_files": files_js, "results_strings": n_js, "r2_r3_prompt_strings": n_prompt_files,
               "n1b_corpus_strings": n_n1b, "n1b_corpus_present": n1b.is_file(), "wikitext_rows": n_wt,
               "jpp_item_strings": n_it, "r3_cache": str(C.R3_CHECKOUT / "experiments" / "R3_cache")}
    return corpus, sources


def corpus_cached(tok, cache: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    if cache and Path(cache).is_file():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    out = build_corpus(tok)
    if cache:
        if Path(cache).resolve().parent == C.RESULTS.resolve():
            raise SystemExit("--corpus-cache must be outside experiments/results")
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out


def redact(rec: Dict[str, Any], key: str = "match") -> Dict[str, Any]:
    """A flag or closest-match record with any third-party text replaced by its sha256."""
    src = rec.get("source") or ""
    if any(src.startswith(p) for p in THIRD_PARTY) and rec.get(key) is not None:
        rec = dict(rec)
        rec[key + "_sha256"] = C.text_sha256(rec.pop(key))
    return rec


def freshness(data: Dict[str, Any], corpus: List[Tuple[str, str]]) -> Dict[str, Any]:
    q = C.queries_of(data)
    fc = R2.freshness_check(q, corpus)
    fi = R2.freshness_check(q, list(q))
    internal = [f for f in fi["flags"] if f["source"].split(":")[0] != f["query"].split(":")[0]]
    return {"n_queries": fc["n_queries"], "flags": [redact(f) for f in fc["flags"]], "internal_flags": internal,
            "closest_10": [redact(c) for c in fc["closest"][:10]],
            "max_ratio": fc["closest"][0]["ratio"] if fc["closest"] else 0.0,
            "rule": "R2_common.freshness_check: exact | contained | contains | similar (difflib >= 0.80 or word "
                    "Jaccard >= 0.75), after NFKC, case folding and dropping punctuation"}


def fragment(path: Path, cache: str) -> int:
    """Check a part of the prompts ({"conditions": {cond: [...]}}) against the corpus and itself; print the flags."""
    tok = R2.load_tokenizer()
    part = json.loads(path.read_text(encoding="utf-8"))
    data = {"conditions": {c: (part.get("conditions") or {}).get(c, []) for c in C.CONDITIONS}}
    only = [c for c in C.CONDITIONS if data["conditions"][c]]
    val = C.validate(data, tok, only=only)
    for p in val["problems"]:
        print("INVALID", p)
    corpus, _ = corpus_cached(tok, cache)
    fr = freshness(data, corpus)
    for f in fr["flags"]:
        print("FLAG  ", f["query"], f["kind"], "|", f["source"], "|", (f.get("match") or f.get("match_sha256"))[:120])
    for f in fr["internal_flags"]:
        print("INNER ", f["query"], f["kind"], "|", f["source"], "|", f["match"][:120])
    for c in fr["closest_10"]:
        print("close ", c["query"], c["ratio"], "|", c["source"])
    print(f"{fr['n_queries']} queries: {len(fr['flags'])} flags against the corpus, {len(fr['internal_flags'])} "
          f"among these prompts; {len(val['problems'])} validation problems ({', '.join(only)})")
    return 0 if not fr["flags"] and not fr["internal_flags"] and not val["problems"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before-prompts", action="store_true", help="check everything but the prompts")
    ap.add_argument("--out", default=str(C.CHECK_JSON))
    ap.add_argument("--corpus-cache", default="", help="a pickle of the corpus (outside experiments/results)")
    ap.add_argument("--fragment", default="", help="check a part of the prompts and print the flags")
    a = ap.parse_args()
    if a.fragment:
        return fragment(Path(a.fragment), a.corpus_cache)
    if a.corpus_cache:
        raise SystemExit("--corpus-cache is for --fragment only: the record is checked against a corpus built now")
    rec: Dict[str, Any] = {"schema": "R4_01/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds")}
    fails: List[str] = []

    def check(ok: bool, what: str) -> bool:
        log(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails.append(what)
        return ok

    # 1. pins
    r1, seen = C.find_r1_lens(full=True)
    rec["r1_lens"] = {"found": str(r1) if r1 else None, "candidates": seen}
    check(r1 is not None and Path(r1) == C.R1_STABLE,
          f"R1's pinned lens (sha256 {C.R1_SHA256[:12]}, {C.R1_SIZE:,} bytes) at the stable path {C.R1_STABLE}: "
          f"found at {r1}")
    jr = C3.check_pinned_file(C.JPP_BASE_PATH, C.JPP_BASE_SHA256, C.JPP_BASE_SIZE, full=False)
    rec["jpp_lens"] = jr
    check(bool(jr["size_ok"]), f"J++'s file has its pinned size ({C.JPP_BASE_SIZE:,} bytes; R4_02 checks its sha256)")
    rs = C.runtime_state()
    rec["runtime"] = rs
    check(rs["ok"], f"the runtime: pinned files differ {rs['pinned_differ']}, snapshot files differ "
                    f"{rs['snapshot_differ']}, versions {rs['versions']}")
    # 2. R2's filter
    tok = R2.load_tokenizer()
    fr = R2.filter_report(tok)
    r2c = json.loads(R2.CHECK_JSON.read_text(encoding="utf-8"))["filter"]
    rec["filter"] = {"counts": fr["counts"], "spot_checks_ok": fr["ok"]}
    check(fr["counts"] == r2c["counts"] and fr["ok"],
          f"R2's filter: class counts {fr['counts']} equal R2_01_check.json's, spot checks pass")
    # 3. freshness
    corpus, sources = corpus_cached(tok, a.corpus_cache)
    log(f"freshness corpus: {len(corpus):,} strings from {json.dumps(sources)[:300]}")
    r3p = json.loads(C3.PROMPTS_PATH.read_text(encoding="utf-8"))
    rows, items = C.r3_cache_rows_and_items(verify=False)
    planted = [("planted:wikitext", " ".join(next(t for t in rows if len(t.split()) > 40).split()[:12])),
               ("planted:jpp_item", items["multihop"][0]["prompt"]),
               ("planted:r2_prompt", json.loads(R2.PROMPTS_PATH.read_text())["conditions"]["bare"][0]["text"]),
               ("planted:r3_prompt", r3p["conditions"]["mid-reply"][5]["user"]),
               ("planted:fresh", "Would a lighthouse keeper's tortoise rather take the stairs or the lift on a "
                                 "windy Tuesday?")]
    pc = R2.freshness_check(planted, corpus)
    flagged = {f["query"] for f in pc["flags"]}
    check({"planted:wikitext", "planted:jpp_item", "planted:r2_prompt", "planted:r3_prompt"} <= flagged
          and "planted:fresh" not in flagged,
          "the corpus flags planted copies of a WikiText row, a J++ item, an R2 prompt and an R3 prompt, and not a "
          "fresh one")
    rec["freshness"] = {"sources": sources, "n_corpus_strings": pc["n_corpus_strings"],
                        "planted": {lab: lab in flagged for lab, _ in planted}}
    # 4. the prompts
    if C.PROMPTS_PATH.is_file():
        data = C.load_prompts()
        rec["prompts_file"] = str(C.PROMPTS_PATH.relative_to(C.REPO))
        rec["prompts_sha256"] = C.sha256_file(C.PROMPTS_PATH)
        val = C.validate(data, tok)
        rec["validation"] = val
        check(val["ok"], f"R4_prompts.json is valid with R4's counts and composition ({val['problems'][:3]})")
        fz = freshness(data, corpus)
        rec["freshness"].update({k: v for k, v in fz.items()})
        check(not fz["flags"] and not fz["internal_flags"],
              f"freshness: {len(fz['flags'])} flags against the corpus, {len(fz['internal_flags'])} among the R4 "
              f"prompts")
        pids = {it["pid"] for it in R2.prompt_items(data)}
        check(set(C.FIT_OVERLAP) <= pids, f"FIT_OVERLAP names {len(C.FIT_OVERLAP)} R4 prompts, all of them R4's")
        rec["fit_overlap"] = {"n": len(C.FIT_OVERLAP),
                              "by_condition": {c: sum(p.startswith(f"r4-{C.PREFIX[c]}-") for p in C.FIT_OVERLAP)
                                               for c in C.CONDITIONS}}
    elif a.before_prompts:
        rec["prompts_file"] = None
        log("R4_prompts.json not written yet (--before-prompts)")
    else:
        check(False, "R4_prompts.json exists (use --before-prompts until it is written)")
    rec["ok"] = not fails
    rec["failures"] = fails
    rec["before_prompts"] = bool(a.before_prompts and not C.PROMPTS_PATH.is_file())
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    log(f"{'ALL CHECKS PASS' if not fails else str(len(fails)) + ' CHECK(S) FAIL'}; wrote {a.out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
