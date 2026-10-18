#!/usr/bin/env python3
"""R2 01 (CPU only; loads no model): the checks made before the GPU run.

1. The token filter (R2_common.content_mask): its size, examples of each class, the pre-registered spot checks
   (" **", "**", "<strong>", "</h2>", "\\n", " ", "<|channel>", "<channel|>", "\\x00", " U+200B", "✔️" NON-CONTENT;
   " Paris", " Ottawa", " the", "The", "8", " kind", " same" CONTENT) and the edges of its definition.
2. The prompt file (R2_prompts.json): counts, fields, begun answers, template ends, token lengths.
3. Freshness: every prompt (each question, begun answer, question + begun answer, bare text) against
   - the earlier held-out set: the QUESTIONS, BEGUN and BARE lists of the scratchpad heldout_eval.py (parsed, not run);
   - every string literal of experiments/scripts/*.py (all *_common.py prompt lists included) in this checkout and
     the main checkout, R2_* excluded;
   - every string in experiments/results/*.json of both checkouts (the study's stimuli, generated chats and corpora
     as saved), R2_* excluded, and the main checkout's N1B_corpus.jsonl (gitignored third-party docs) if present;
   - the other R2 prompts (no two R2 prompts may be near-duplicates of each other).
   Matching is R2_common.freshness_check: exact, contained, contains, similar (difflib >= 0.80 or Jaccard >= 0.75).

Writes experiments/results/R2_01_check.json and exits 1 if any check fails.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_01_check.py
       [--heldout PATH] [--main PATH]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import datetime
import glob
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import R2_common as C  # noqa: E402

DEFAULT_HELDOUT = ("/tmp/claude-1000/-home-eric-bin-ai-theodolite2/3bfa7298-f5bc-4efa-943d-4ad1d77f7b92/scratchpad/"
                   "heldout/heldout_eval.py")
DEFAULT_MAIN = "/home/eric/bin/ai/theodolite2"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def heldout_strings(path: Path):
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("QUESTIONS", "BEGUN", "BARE"):
                out[name] = [n.value for n in ast.walk(node.value)
                             if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return out


def queries_of(data):
    q = []
    for it in C.prompt_items(data):
        if it["cond"] == "bare":
            q.append((f"{it['pid']}:text", it["text"]))
        else:
            q.append((f"{it['pid']}:user", it["user"]))
        if it["cond"] == "chat-begun":
            q.append((f"{it['pid']}:begun", it["begun"]))
            q.append((f"{it['pid']}:user+begun", it["user"] + " " + it["begun"]))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--main", default=DEFAULT_MAIN, help="the main checkout (for gitignored corpora and scripts)")
    ap.add_argument("--out", default=str(C.CHECK_JSON))
    args = ap.parse_args()
    rec = {"schema": "R2_01/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "prompts_file": str(C.PROMPTS_PATH.relative_to(C.REPO)), "prompts_sha256": C.sha256_file(C.PROMPTS_PATH)}
    ok = True

    # 1. the filter
    tok = C.load_tokenizer()
    fr = C.filter_report(tok)
    rec["filter"] = fr
    log(f"filter: vocab {fr['vocab_size']:,}; CONTENT {fr['n_content']:,}; NON-CONTENT {fr['n_noncontent']:,} "
        f"{fr['counts']} (html_tag includes {fr['html_tag_of_which_unused_placeholders']:,} <unusedN> placeholders)")
    for c in fr["checks"]:
        log(f"  {'ok ' if c['ok'] else 'BAD'} {c['text']!r:14} want {c['want']:11} text class {c['text_class']:12} "
            f"vocab ids {c['vocab_ids']}  encodes as {c['encodes_as_text']}")
    for cls, ex in fr["examples"].items():
        log(f"  examples {cls}: " + ", ".join(repr(t) for _, t in ex))
    for k, v in fr["edges"].items():
        log(f"  edge {k}: {json.dumps(v, ensure_ascii=True)[:300]}")
    ok &= fr["ok"]

    # 2. the prompt file
    data = C.load_prompts()
    val = C.validate_prompts(data, tok)
    rec["validation"] = val
    log(f"prompts: {'valid' if val['ok'] else 'INVALID'}; {json.dumps(val['stats'])}")
    for p in val["problems"]:
        log("  problem:", p)
    ok &= val["ok"]
    # what the read token is in each condition (first prompt), for the record
    rec["read_tokens"] = {}
    for it in C.prompt_items(data):
        if it["cond"] in rec["read_tokens"]:
            continue
        ids = C.text_ids(tok, C.prompt_text(tok, it))
        rec["read_tokens"][it["cond"]] = {"pid": it["pid"], "n_tokens": len(ids),
                                          "last_tokens": [C.token_text(tok, i) for i in ids[-4:]]}
    log("read tokens:", json.dumps(rec["read_tokens"], ensure_ascii=False))

    # 3. freshness
    corpus = []
    sources = {}
    held = Path(args.heldout)
    if not held.is_file():
        log(f"held-out script not found at {held}")
        ok = False
        rec["freshness_error"] = f"held-out script not found: {held}"
    else:
        hs = heldout_strings(held)
        for name, strs in hs.items():
            corpus += [(f"heldout:{name}", s) for s in strs]
        sources["heldout_eval.py"] = {k: len(v) for k, v in hs.items()}
        if sorted(hs) != ["BARE", "BEGUN", "QUESTIONS"]:
            log("held-out lists not all found:", sorted(hs))
            ok = False
    roots = [C.REPO] + ([Path(args.main)] if Path(args.main).resolve() != C.REPO.resolve() else [])
    seen_py, seen_json = {}, {}
    for root in roots:
        for f in sorted(glob.glob(str(root / "experiments" / "scripts" / "*.py"))):
            name = os.path.basename(f)
            if name.startswith("R2_") or name in seen_py:
                continue
            seen_py[name] = f
        for f in sorted(glob.glob(str(root / "experiments" / "results" / "*.json"))):
            name = os.path.basename(f)
            if name.startswith("R2_") or name in seen_json:
                continue
            seen_json[name] = f
    n_py = 0
    for name, f in seen_py.items():
        strs = C.python_string_literals(Path(f))
        n_py += len(strs)
        corpus += [(f"scripts/{name}", s) for s in strs]
    n_js = 0
    for name, f in seen_json.items():
        try:
            obj = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        strs = []
        C.collect_strings(obj, strs)
        n_js += len(strs)
        corpus += [(f"results/{name}", s) for s in strs]
    n1b = Path(args.main) / "experiments" / "results" / "N1B_corpus.jsonl"
    n_n1b = 0
    if n1b.is_file():
        # chunks are stored as token ids ({"cid", "src", "doc", "ids"}): decode them with the same tokenizer
        for line in n1b.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            strs = []
            if isinstance(d, dict) and isinstance(d.get("ids"), list):
                strs.append(tok.decode([int(i) for i in d["ids"]], skip_special_tokens=False))
            C.collect_strings({k: v for k, v in d.items() if k != "ids"} if isinstance(d, dict) else d, strs)
            n_n1b += len(strs)
            corpus += [("results/N1B_corpus.jsonl", s) for s in strs]
    sources.update({"scripts_py_files": len(seen_py), "scripts_strings": n_py, "results_json_files": len(seen_json),
                    "results_strings": n_js, "n1b_corpus_strings": n_n1b, "n1b_corpus_present": n1b.is_file()})
    log(f"corpus: {len(corpus):,} strings from {json.dumps(sources)}")
    queries = queries_of(data)
    fc = C.freshness_check(queries, corpus)
    # the R2 prompts against each other (a prompt's own entries are not matches)
    own = [(lab, t) for lab, t in queries]
    fi = C.freshness_check(queries, own)
    internal = [f for f in fi["flags"] if f["source"].split(":")[0] != f["query"].split(":")[0]]
    rec["freshness"] = {
        "sources": sources,
        "n_queries": fc["n_queries"],
        "n_corpus_strings": fc["n_corpus_strings"],
        "flags": fc["flags"],
        "internal_flags": internal,
        "closest_10": fc["closest"][:10],
        "max_ratio": fc["closest"][0]["ratio"] if fc["closest"] else 0.0,
        "rule": "exact | contained (query >= 4 words inside a corpus string) | contains (corpus string >= 5 words "
                "inside the query) | similar (difflib ratio >= 0.80 or word Jaccard >= 0.75), after NFKC, case "
                "folding and dropping punctuation",
    }
    log(f"freshness: {len(fc['flags'])} flags against the corpus, {len(internal)} among the R2 prompts; "
        f"closest ratio {rec['freshness']['max_ratio']}")
    for f in fc["flags"][:60]:
        log(f"  FLAG {f['kind']:9} {f['query']:18} {f['text']!r} ~ {f['source']}: {f['match'][:120]!r}")
    for f in internal[:30]:
        log(f"  INTERNAL {f['kind']:9} {f['query']:18} ~ {f['source']}: {f['match'][:100]!r}")
    for c in fc["closest"][:10]:
        log(f"  closest {c['ratio']:.3f} {c['query']:18} ~ {c['source']}: {str(c['match'])[:100]!r}")
    ok &= not fc["flags"] and not internal and "freshness_error" not in rec
    rec["ok"] = bool(ok)
    Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAIL'}; wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
