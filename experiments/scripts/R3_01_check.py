#!/usr/bin/env python3
"""R3 01 (CPU only; loads no model, uses no GPU, no network): the checks made before any R3 GPU work.

1. Pins. R3_common's pins for R1's lens and the Miru J-lens equal R2's; R1's pinned lens is found (the main
   checkout's copy has the same matrices but an older provenance note and is not the pinned file); the sha256 and
   size of the Miru file, R1's 128 directions and the quiet-dimension statistics; the model snapshot's config.
2. The sources. The jpp_lens checkout is the pinned commit, file by file; J++'s six item files and the 1,000 WikiText
   records match the manifest (R3_sources.json). The 64 fit prompts' token counts (BOS, truncation 128) and kept
   positions are recorded (J++'s 7,104 = 64 x 111 is for Qwen's tokenizer), and the router's kept positions.
3. The token filters. R2's filter (R2_common.content_mask) has the class counts R2_01_check.json recorded (special
   24, blank 104, punct_symbol 7,746, html_tag 6,297) and passes its spot checks. J++'s Readout Filtering mask (no
   letter or digit, jpp_lens's own all_non_semantic_token_ids over the LM-head width, minus every candidate id of the
   six evals' items) is counted, and crossed with R2's classes.
4. Miru's prompts. Its header describes its selection (concatenate_and_rechunk, exclude_headings, 200-2000 chars,
   1,000 prompts at revision b08601e0) and records two hashes: the 1,000-prompt sequence and the 425 processed
   prompts. A bounded set of readings of that description (paragraph packing or stream cutting, stripped or not, four
   separators) is rebuilt from the cached rows and hashed in twelve ways. A match fixes Miru's prompts for the Q1 arm;
   no match is recorded, with the fallback (R3_PREREG.md §3), and is not a failure of this check.
5. Freshness. Every R3 prompt (each question, begun answer, question + begun answer, bare text) against R2's sources
   (string literals of experiments/scripts/*.py and every string in experiments/results/*.json of this checkout and
   the main one, R3_* excluded, so R2_prompts.json, R2's results and R1's corpus are in; the N1B corpus), every cached
   WikiText row (the 1,000 records and the rows between them, which hold the start of Miru's prompt stream), every J++
   eval item (prompt, target, intermediates), and the other R3 prompts. Matching is R2_common.freshness_check. Before
   the prompts exist (--before-prompts), the corpus is built and checked with planted queries only.
   Matches inside third-party texts are recorded by source and sha256, never by text. R2's 220 prompts (R3's
   replication set) are checked the same way against R3's fit sources (the WikiText rows and J++'s items), which R2
   could not do; that is recorded, not a failure.

Writes experiments/results/R3_01_check.json and exits 1 if any check fails (without --before-prompts, a missing
R3_prompts.json is a failure).
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_01_check.py
       [--before-prompts] [--main PATH] [--out PATH] [--quick]
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
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C  # noqa: E402

T0 = time.time()
THIRD_PARTY = ("wikitext:", "jpp_items:", "results/N1B_corpus.jsonl")


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ 4. Miru's prompts
HEADING = re.compile(r"^\s*=+.*=+\s*$")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sequence_hashes(ps: List[str]) -> Dict[str, str]:
    """Twelve ways a prompt sequence could have been hashed."""
    out = {}
    for ea in (True, False):
        for sep in (None, (",", ":")):
            out[f"json.dumps ensure_ascii={ea} separators={sep}"] = _sha(json.dumps(ps, ensure_ascii=ea,
                                                                                      separators=sep).encode())
    hx = [_sha(p.encode()) for p in ps]
    out["per-prompt sha256, newline-joined"] = _sha("\n".join(hx).encode())
    out["per-prompt sha256, concatenated"] = _sha("".join(hx).encode())
    out["texts newline-joined"] = _sha("\n".join(ps).encode())
    out["texts NUL-joined"] = _sha("\0".join(ps).encode())
    out["texts concatenated"] = _sha("".join(ps).encode())
    out["JSON lines"] = _sha("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ps).encode())
    for order in ("little", "big"):
        h = hashlib.sha256()
        for p in ps:
            b = p.encode()
            h.update(len(b).to_bytes(8, order))
            h.update(b)
        out[f"length-prefixed ({order}-endian u64)"] = h.hexdigest()
    return out


def _packed(paras: List[str], sep: str, mn: int, mx: int, split_long: bool):
    cur = ""
    for p in paras:
        cand = p if not cur else cur + sep + p
        if len(cand) <= mx:
            cur = cand
            continue
        if cur and len(cur) >= mn:
            yield cur
        cur = p
        while split_long and len(cur) > mx:
            cut = cur.rfind(" ", 0, mx + 1)
            cut = cut if cut > 0 else mx
            piece, cur = cur[:cut].rstrip(), cur[cut:].lstrip()
            if len(piece) >= mn:
                yield piece
    if cur and len(cur) >= mn:
        yield cur


def _streamed(paras: List[str], sep: str, mn: int, mx: int):
    text, i = sep.join(paras), 0
    while i < len(text):
        j = min(len(text), i + mx)
        if j < len(text):
            k = text.rfind(" ", i, j + 1)
            j = k if k > i else j
        piece = text[i:j].strip()
        if len(piece) >= mn:
            yield piece
        i = j


def miru_rebuild(rows: List[str], fit: Dict[str, Any]) -> Dict[str, Any]:
    sel = fit["provenance"]["corpus"]["selection"]
    mn, mx = sel["min_chars"], sel["max_chars"]
    want_full = fit["provenance"]["prompt_sequence_sha256"]
    want_prefix = fit["fit"]["processed_prompt_prefix_sha256"]
    n_full, n_prefix = fit["provenance"]["selected_prompts"], fit["fit"]["processed_prompts"]
    tried, hits, produced = 0, [], {}
    for strip, sep, algo in itertools.product((True, False), ("\n", " ", "\n\n", ""),
                                              ("pack, long paragraphs split at a space", "pack, no split",
                                               "stream cut at a space")):
        paras = [(t.strip() if strip else t) for t in rows if t.strip() and not (sel["exclude_headings"]
                                                                                 and HEADING.match(t))]
        ps = list(_streamed(paras, sep, mn, mx) if algo.startswith("stream")
                  else _packed(paras, sep, mn, mx, algo.startswith("pack, long")))
        name = f"{algo}; paragraphs {'stripped' if strip else 'as stored'}; joined by {sep!r}"
        produced[name] = len(ps)
        if len(ps) >= n_prefix:
            for k, v in sequence_hashes(ps[:n_prefix]).items():
                tried += 1
                if v == want_prefix:
                    hits.append({"reading": name, "hash": k, "which": f"processed prefix ({n_prefix})"})
        if len(ps) >= n_full:
            for k, v in sequence_hashes(ps[:n_full]).items():
                tried += 1
                if v == want_full:
                    hits.append({"reading": name, "hash": k, "which": f"full sequence ({n_full})"})
    return {
        "selection_in_header": sel, "prompt_budget": fit["provenance"]["prompt_budget"],
        "processed_prompts": n_prefix, "dataset_revision": fit["provenance"]["corpus"]["dataset_revision"],
        "readings_tried": len(produced), "hash_comparisons": tried, "prompts_per_reading": produced,
        "max_prompts_from_cache": max(produced.values()), "rebuilt": bool(hits), "hits": hits,
        "outcome": ("rebuilt: the Q1 4-bit J-lens arm is fitted on exactly Miru's processed prompts" if hits else
                    "not rebuilt. No reading of the header's description reproduced either hash. Miru Tracer's own "
                    "selection code (github.com/returnmoe/miru-tracer) is not one of the approved sources, and every "
                    f"reading tried yields at most {max(produced.values())} prompts from the cached rows, fewer than "
                    f"the {n_full} the full-sequence hash covers. Fallback (R3_PREREG.md §3): the Q1 arm is fitted on "
                    "J++'s 64 prompts only, so bf16 against W4A16 also changes the prompts (WikiText both, Miru's "
                    "425 rechunked passages against J++'s first 64 records); this is stated wherever Q1 is "
                    "reported. Rebuilding needs the user to approve Miru Tracer's code at a pinned commit (and, "
                    "probably, more WikiText rows)."),
    }


# ------------------------------------------------------------------------------------------------ 5. freshness
def queries_of(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    q = []
    for it in R2.prompt_items(data):
        if it["cond"] == "bare":
            q.append((f"{it['pid']}:text", it["text"]))
        else:
            q.append((f"{it['pid']}:user", it["user"]))
        if it["cond"] == "chat-begun":
            q.append((f"{it['pid']}:begun", it["begun"]))
            q.append((f"{it['pid']}:user+begun", it["user"] + " " + it["begun"]))
    return q


def build_corpus(main: Path, tok, rows: List[str], items: Dict[str, List[Dict[str, Any]]]):
    corpus: List[Tuple[str, str]] = []
    roots = [C.REPO] + ([main] if main.resolve() != C.REPO.resolve() else [])
    seen_py, seen_json = {}, {}
    for root in roots:
        for f in sorted(glob.glob(str(root / "experiments" / "scripts" / "*.py"))):
            n = os.path.basename(f)
            if not n.startswith("R3_") and n not in seen_py:
                seen_py[n] = f
        for f in sorted(glob.glob(str(root / "experiments" / "results" / "*.json"))):
            n = os.path.basename(f)
            if not n.startswith("R3_") and n not in seen_json:
                seen_json[n] = f
    n_py = n_js = 0
    for n, f in seen_py.items():
        strs = R2.python_string_literals(Path(f))
        n_py += len(strs)
        corpus += [(f"scripts/{n}", s) for s in strs]
    for n, f in seen_json.items():
        try:
            obj = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        strs: List[str] = []
        R2.collect_strings(obj, strs)
        n_js += len(strs)
        corpus += [(f"results/{n}", s) for s in strs]
    r2p = json.loads(R2.PROMPTS_PATH.read_text(encoding="utf-8"))
    corpus += [("scripts/R2_prompts.json", s) for _, s in queries_of(r2p)]
    n1b = main / "experiments" / "results" / "N1B_corpus.jsonl"
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
    sources = {"scripts_py_files": len(seen_py), "scripts_strings": n_py, "results_json_files": len(seen_json),
               "results_strings": n_js, "r2_prompt_strings": len(queries_of(r2p)), "n1b_corpus_strings": n_n1b,
               "n1b_corpus_present": n1b.is_file(), "wikitext_rows": n_wt, "jpp_item_strings": n_it,
               "heldout_eval_py": "not available: R2's earlier held-out script lived in another session's scratchpad "
                                  "and no longer exists (R2_01_check.json records only its list sizes, 40/60/30)"}
    return corpus, sources


def redact(rec: Dict[str, Any], key: str = "match") -> Dict[str, Any]:
    """A flag or closest-match record with any third-party text replaced by its sha256."""
    src = rec.get("source") or ""
    if any(src.startswith(p) for p in THIRD_PARTY) and rec.get(key) is not None:
        rec = dict(rec)
        rec[key + "_sha256"] = C.text_sha256(rec.pop(key))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before-prompts", action="store_true",
                    help="check everything but the prompts (R3_prompts.json not written yet)")
    ap.add_argument("--main", default=str(C.MAIN_CHECKOUT))
    ap.add_argument("--out", default=str(C.CHECK_JSON))
    ap.add_argument("--quick", action="store_true", help="skip the sha256 of the 3.4 GB Miru file (size only)")
    a = ap.parse_args()
    rec: Dict[str, Any] = {"schema": "R3_01/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds")}
    fails: List[str] = []

    def check(ok: bool, what: str) -> bool:
        log(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails.append(what)
        return ok

    # 1. pins
    pins: Dict[str, Any] = {}
    check((C.R1_SHA256, C.R1_SIZE, C.MIRU_SHA256, C.MIRU_SIZE, C.MIRU_REVISION) ==
          (R2.R1_SHA256, R2.R1_SIZE, R2.MIRU_SHA256, R2.MIRU_SIZE, R2.MIRU_REVISION),
          "R3_common pins R1's lens and the Miru J-lens exactly as R2_common does")
    r1, seen = C.find_r1_lens()
    pins["r1_lens"] = {"found": str(r1) if r1 else None, "candidates": seen}
    check(r1 is not None, f"R1's pinned lens (sha256 {C.R1_SHA256[:12]}, {C.R1_SIZE:,} bytes) found at {r1}")
    mainr1 = [s for s in seen if s["path"].startswith(str(C.MAIN_CHECKOUT / "experiments"))]
    if mainr1 and mainr1[0]["exists"] and mainr1[0]["sha256_ok"] is False:
        pins["r1_lens"]["main_copy_note"] = ("the main checkout's R1_18_it_lens.pt has the pinned size but another "
                                             "sha256: same J matrices, an older provenance note; not used")
    for name, path, sha, size in (("miru", C.MIRU_PATH, C.MIRU_SHA256, C.MIRU_SIZE),
                                  ("r1_dirs", C.R1_DIRS_PATH, C.R1_DIRS_SHA256, C.R1_DIRS_SIZE),
                                  ("quiet_stats", C.QUIET_STATS_PATH, C.QUIET_STATS_SHA256, C.QUIET_STATS_SIZE)):
        r = C.check_pinned_file(path, sha, size, full=not (a.quick and name == "miru"))
        pins[name] = r
        check(r["size_ok"] and r["sha256_ok"] is not False,
              f"{name}: {size:,} bytes" + ("" if r["sha256_ok"] is None else f", sha256 {sha[:12]}"))
    cfg = C.real_text_config()
    check(cfg["num_hidden_layers"] == C.N_LAYERS and cfg["hidden_size"] == C.D_MODEL
          and cfg["final_logit_softcapping"] == 30.0,
          f"model snapshot {C.MODEL_SNAPSHOT[:8]}: 60 layers, d_model 5376, softcap 30")
    rec["pins"] = pins

    # 2. sources
    try:
        C.verify_jpp_checkout()
        check(True, f"jpp_lens checkout at {C.JPP_COMMIT[:12]}, every tracked file as the manifest pins it")
    except C.CacheError as e:
        check(False, f"jpp_lens checkout: {e}")
    items = C.jpp_items()
    check({k: len(v) for k, v in items.items()} == {s: C.load_manifest()["jpp_eval_items"]["files"][s]["n_items"]
                                                    for s in C.JPP_EVALS},
          "J++'s six item files match the manifest: " + ", ".join(f"{k} {len(v)}" for k, v in items.items()))
    records = C.wikitext_records()
    rows = C.wikitext_rows()
    check(len(records) == C.N_ROUTER, f"the {C.N_ROUTER:,} WikiText records match the manifest (row, length, sha256)")
    tok = R2.load_tokenizer()
    lens_fit = [len(R2.text_ids(tok, t)[:C.MAX_SEQ_LEN]) for t in records[:C.N_FIT]]
    kept_fit = sum(n - 1 - C.SKIP_FIRST for n in lens_fit)
    lens_router = [len(R2.text_ids(tok, t)[:C.MAX_SEQ_LEN]) for t in records]
    kept_router = sum(max(0, n - 1 - C.SKIP_FIRST) for n in lens_router)
    rec["fit_positions"] = {"fit_prompts": C.N_FIT, "fit_tokens_each": sorted(set(lens_fit)),
                            "fit_kept_positions": kept_fit, "router_records": len(records),
                            "router_kept_positions": kept_router, "router_min_tokens": min(lens_router)}
    short = {i: n for i, n in enumerate(lens_fit) if n < C.MAX_SEQ_LEN}
    rec["fit_positions"]["fit_prompts_under_128_tokens"] = short
    check(min(lens_fit) > C.SKIP_FIRST + 1 and kept_fit == sum(n - 1 - C.SKIP_FIRST for n in lens_fit),
          f"the 64 fit prompts (BOS, truncated to 128): {sum(n == C.MAX_SEQ_LEN for n in lens_fit)} at 128 tokens"
          + (", " + ", ".join(f"record {i} at {n}" for i, n in short.items()) if short else "")
          + f"; {kept_fit:,} kept positions (J++'s 7,104 = 64 x 111 is for Qwen's tokenizer)")
    log(f"router: {kept_router:,} kept positions over {len(records):,} records (shortest {min(lens_router)} tokens)")

    # 3. filters
    fr = R2.filter_report(tok)
    r2c = json.loads(R2.CHECK_JSON.read_text(encoding="utf-8"))["filter"]
    check(fr["counts"] == r2c["counts"] and fr["ok"],
          f"R2's filter: class counts {fr['counts']} equal R2_01_check.json's, spot checks pass")
    C.import_jpp()
    from lens_evals import eval_utils as EU

    import R3_fitlib as F

    V = int(cfg["vocab_size"])                                # the LM-head width: tied embeddings, vocab_size rows
    check(cfg.get("tie_word_embeddings") is True and V == len(tok),
          f"LM-head width {V:,} (tied embeddings), the tokenizer's {len(tok):,} ids")
    allns = EU.all_non_semantic_token_ids(tok, V)
    all_items = F.items_from_raw(items, C.JPP_EVALS)
    excl = set(F.excluded_ids(tok, V, all_items))
    cls, _ = R2.classify_vocab(tok)
    cross = {}
    for k, name in enumerate(R2.CLASSES):
        ids = set(int(i) for i in (cls == k).nonzero()[0])
        cross[name] = {"n": len(ids), "jpp_excluded": len(ids & excl), "jpp_kept": len(ids - excl)}
    rec["filters"] = {
        "r2": {"counts": fr["counts"], "n_content": fr["n_content"], "n_noncontent": fr["n_noncontent"],
               "spot_checks_ok": fr["ok"]},
        "jpp": {"vocab_size": V, "no_letter_or_digit": len(allns), "spared_item_candidates": len(allns - excl),
                "excluded": len(excl), "rule": "lens_evals.eval_utils.is_non_semantic_token (no str.isalnum char); "
                                               "minus every candidate id of the six evals' items"},
        "r2_class_by_jpp": cross,
        "differ": {"r2_noncontent_jpp_kept": sum(v["jpp_kept"] for k, v in cross.items() if k != "content"),
                   "r2_content_jpp_excluded": cross["content"]["jpp_excluded"]},
    }
    check(len(excl) > 0 and cross["content"]["n"] == fr["n_content"],
          f"J++'s mask: {len(allns):,} ids with no letter or digit, {len(allns - excl)} spared as item candidates, "
          f"{len(excl):,} excluded; R2-non-content kept by J++ {rec['filters']['differ']['r2_noncontent_jpp_kept']:,}, "
          f"R2-content excluded by J++ {cross['content']['jpp_excluded']:,}")

    # 4. Miru's prompts
    fit = C.miru_fit_metadata()
    want = {"prompt_sequence_sha256": fit["provenance"]["prompt_sequence_sha256"],
            "processed_prompt_prefix_sha256": fit["fit"]["processed_prompt_prefix_sha256"],
            "processed_prompts": fit["fit"]["processed_prompts"], "dim_batch": fit["fit"]["dim_batch"],
            "target_layer": fit["fit"]["target_layer"], "selection": fit["provenance"]["corpus"]["selection"],
            "dataset_revision": fit["provenance"]["corpus"]["dataset_revision"]}
    check(all(C.MIRU_FIT[k] == v for k, v in want.items()),
          "Miru's header (fit_metadata) records what R3_common pins: both prompt hashes, 425 prompts, dim_batch 32, "
          "target 59, the selection, revision b08601e0")
    mr = miru_rebuild(rows, fit)
    rec["miru_prompts"] = mr
    log(f"Miru's prompts: {mr['readings_tried']} readings, {mr['hash_comparisons']} hash comparisons -> "
        f"{'REBUILT' if mr['rebuilt'] else 'not rebuilt'}")

    # 5. freshness
    corpus, sources = build_corpus(Path(a.main), tok, rows, items)
    log(f"freshness corpus: {len(corpus):,} strings from {json.dumps(sources)[:400]}")
    planted = [("planted:wikitext", " ".join(records[0].split()[:12])),
               ("planted:jpp_item", items["multihop"][0]["prompt"]),
               ("planted:r2_prompt", json.loads(R2.PROMPTS_PATH.read_text())["conditions"]["bare"][0]["text"]),
               ("planted:fresh", "Which colour would a heron choose for its winter coat, if asked politely?")]
    pc = R2.freshness_check(planted, corpus)
    flagged = {f["query"] for f in pc["flags"]}
    src_of = {f["query"]: f["source"].split(":")[0] for f in pc["flags"]}
    check({"planted:wikitext", "planted:jpp_item", "planted:r2_prompt"} <= flagged and "planted:fresh" not in flagged
          and src_of.get("planted:wikitext") == "wikitext" and src_of.get("planted:jpp_item") == "jpp_items",
          "the corpus flags planted copies of a WikiText record, a J++ item and an R2 prompt, and not a fresh one")
    rec["freshness"] = {"sources": sources, "n_corpus_strings": pc["n_corpus_strings"],
                        "planted": {lab: lab in flagged for lab, _ in planted}}
    # R2's 220 prompts are R3's replication set (secondary). R2 could not check them against WikiText (R2_PREREG.md
    # §3.2); here they are checked against R3's fit sources: every cached WikiText row and every J++ item string.
    # Recorded, not a failure: the replication set is reported whatever this finds.
    fit_src = [(s, t) for s, t in corpus if s.startswith(("wikitext:", "jpp_items:"))]
    r2q = queries_of(json.loads(R2.PROMPTS_PATH.read_text(encoding="utf-8")))
    rc = R2.freshness_check(r2q, fit_src)
    rec["r2_set_against_fit_sources"] = {
        "n_queries": rc["n_queries"], "n_fit_source_strings": rc["n_corpus_strings"],
        "flags": [redact(f) for f in rc["flags"]], "closest_5": [redact(c) for c in rc["closest"][:5]],
        "max_ratio": rc["closest"][0]["ratio"] if rc["closest"] else 0.0}
    log(f"R2's prompts against R3's fit sources (WikiText rows, J++ items): {len(rc['flags'])} flags over "
        f"{rc['n_queries']} queries and {rc['n_corpus_strings']:,} strings (recorded, not a failure)")
    if C.PROMPTS_PATH.is_file():
        data = json.loads(C.PROMPTS_PATH.read_text(encoding="utf-8"))
        rec["prompts_file"] = str(C.PROMPTS_PATH.relative_to(C.REPO))
        rec["prompts_sha256"] = C.sha256_file(C.PROMPTS_PATH)
        with mock.patch.dict(R2.N_PROMPTS, C.N_PROMPTS):
            val = R2.validate_prompts(data, tok)
        rec["validation"] = val
        check(val["ok"], f"R3_prompts.json is valid ({json.dumps(val['stats'])[:200]})")
        q = queries_of(data)
        fc = R2.freshness_check(q, corpus)
        fi = R2.freshness_check(q, list(q))
        internal = [f for f in fi["flags"] if f["source"].split(":")[0] != f["query"].split(":")[0]]
        rec["freshness"].update({"n_queries": fc["n_queries"], "flags": [redact(f) for f in fc["flags"]],
                                 "internal_flags": internal, "closest_10": [redact(c) for c in fc["closest"][:10]],
                                 "max_ratio": fc["closest"][0]["ratio"] if fc["closest"] else 0.0,
                                 "rule": "R2_common.freshness_check: exact | contained | contains | similar (difflib "
                                         ">= 0.80 or word Jaccard >= 0.75), after NFKC, case folding and dropping "
                                         "punctuation"})
        check(not fc["flags"] and not internal,
              f"freshness: {len(fc['flags'])} flags against the corpus, {len(internal)} among the R3 prompts")
        check(not C.FIT_OVERLAP or set(C.FIT_OVERLAP) <= {it["pid"] for it in R2.prompt_items(data)},
              f"FIT_OVERLAP names {len(C.FIT_OVERLAP)} R3 prompts")
    elif a.before_prompts:
        rec["prompts_file"] = None
        rec["freshness"]["prompts"] = "not yet written: R3_prompts.json is part of the registration still to come"
        log("R3_prompts.json not written yet (--before-prompts): the prompts' freshness is checked when it is")
    else:
        check(False, "R3_prompts.json exists (use --before-prompts until it is written)")
    rec["ok"] = not fails
    rec["failures"] = fails
    rec["before_prompts"] = bool(a.before_prompts and not C.PROMPTS_PATH.is_file())
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    log(f"{'ALL CHECKS PASS' if not fails else str(len(fails)) + ' CHECK(S) FAIL'}"
        f"{' (before the prompts)' if rec['before_prompts'] else ''}; wrote {a.out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
