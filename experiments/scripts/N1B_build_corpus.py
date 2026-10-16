#!/usr/bin/env python3
"""N1B step 1: build the activation corpus (CPU only, run locally with .venv python; no internet).

Sources: (a) model-generated chat transcripts (N1B_gen_raw.jsonl, from N1B_01_gen.py; full chat incl. answer),
(b) local English prose/docs: package METADATA long descriptions + *.md/*.rst/*.txt in site-packages, READMEs/NEWS/FAQ
in /usr/share/doc (bz2-decompressed), license boilerplate filtered, (c) Python source files from site-packages.
Each document is tokenized (no special tokens) and cut into windows of <=127 tokens, each prefixed with BOS (2) -> <=128.
Output: experiments/results/N1B_corpus.jsonl, one line per chunk: {"cid", "src", "doc", "ids"}.
Usage: python N1B_build_corpus.py [--disk-only]
"""
import bz2, glob, json, os, random, re, sys

sys.path.insert(0, ".")
from transformers import AutoTokenizer

R = "./experiments/results/"
SP = "./.venv/lib/python3.12/site-packages/"
tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
BOS = 2
WIN = 127
MIN_LAST = 24
rng = random.Random(1234)

BOILER = re.compile(r"permission is hereby granted|without warranty|gnu (lesser |library )?general public license|"
                    r"redistribution and use in source|licensed under the apache|all rights reserved|"
                    r"warranties of merchantability|copyright \(c\)|spdx-license", re.I)
SKIPNAME = re.compile(r"licen[cs]e|copying|notice|authors|contributors|changelog|entry_points|top_level|record|"
                      r"requires|sources|dependency_links|vocab|merges|wheel$|installer|thanks|credits", re.I)


def clean_prose(text):
    paras = re.split(r"\n\s*\n", text)
    keep = [p for p in paras if not BOILER.search(p) and len(p.strip()) > 40]
    return "\n\n".join(keep)


def read_any(path):
    try:
        if path.endswith(".bz2"):
            return bz2.open(path, "rt", errors="replace").read()
        return open(path, errors="replace").read()
    except Exception:
        return ""


def chunk_doc(ids, max_chunks):
    out = []
    for s in range(0, len(ids), WIN):
        w = ids[s:s + WIN]
        if len(w) < MIN_LAST and s > 0:
            break
        if len(w) < MIN_LAST:
            break
        out.append([BOS] + w)
        if len(out) >= max_chunks:
            break
    return out


def metadata_desc(path):
    t = read_any(path)
    # long description follows the first blank line after headers
    k = t.find("\n\n")
    return t[k + 2:] if k >= 0 else ""


def collect_prose(target_tokens):
    docs = []
    for p in sorted(glob.glob(SP + "*.dist-info/METADATA")):
        d = metadata_desc(p)
        if len(d) > 800:
            docs.append(("prose_pkgmeta", os.path.relpath(p, SP), d))
    for ext in ("md", "rst", "txt"):
        for p in sorted(glob.glob(SP + f"**/*.{ext}", recursive=True)):
            if SKIPNAME.search(os.path.basename(p)) or os.path.getsize(p) < 1500:
                continue
            docs.append(("prose_pkgdoc", os.path.relpath(p, SP), read_any(p)))
    for p in sorted(glob.glob("/usr/share/doc/*/*")):
        b = os.path.basename(p)
        if SKIPNAME.search(b) or not re.match(r"(README|NEWS|FAQ|TODO|HACKING|INSTALL|BUGS|USAGE|.*\.md|.*\.txt|.*\.rst)", b, re.I):
            continue
        if b.lower().endswith((".pdf", ".html", ".gz", ".png")):
            continue
        docs.append(("prose_usrdoc", p, read_any(p)))
    rng.shuffle(docs)
    chunks, n = [], 0
    seen = set()
    for src, name, text in docs:
        text = clean_prose(text)
        if len(text) < 600:
            continue
        key = text[:400]
        if key in seen:
            continue
        seen.add(key)
        ids = tok.encode(text, add_special_tokens=False)
        for c in chunk_doc(ids, max_chunks=6):
            chunks.append({"src": src, "doc": name, "ids": c})
            n += len(c)
        if n >= target_tokens:
            break
    return chunks, n


def collect_code(target_tokens):
    files = [p for p in glob.glob(SP + "**/*.py", recursive=True)
             if "/tests/" not in p and "/test_" not in p and os.path.getsize(p) > 3000
             and "_pb2" not in p and "/cuda/" not in p]
    files.sort()
    rng.shuffle(files)
    chunks, n, pkgs = [], 0, {}
    for p in files:
        pkg = os.path.relpath(p, SP).split("/")[0]
        if pkgs.get(pkg, 0) >= 6:  # diversity: <= 6 files per top-level package
            continue
        text = read_any(p)
        # drop leading license header comments
        lines = text.split("\n")
        i = 0
        while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
            i += 1
        text = "\n".join(lines[i:])
        if len(text) < 1500:
            continue
        pkgs[pkg] = pkgs.get(pkg, 0) + 1
        ids = tok.encode(text, add_special_tokens=False)
        # take chunks from a random offset so we don't only see imports
        start = rng.randrange(0, max(1, len(ids) - 4 * WIN)) if len(ids) > 6 * WIN else 0
        for c in chunk_doc(ids[start:], max_chunks=4):
            chunks.append({"src": "code_py", "doc": os.path.relpath(p, SP), "ids": c})
            n += len(c)
        if n >= target_tokens:
            break
    return chunks, n


def collect_chat():
    path = R + "N1B_gen_raw.jsonl"
    chunks, n = [], 0
    if not os.path.exists(path):
        return chunks, n
    for k, line in enumerate(open(path)):
        d = json.loads(line)
        ids = d["prompt_ids"] + d["gen_ids"]  # starts with BOS
        body = ids[1:]
        for c in chunk_doc(body, max_chunks=4):
            chunks.append({"src": "chat_gen", "doc": f"gen{k:03d}", "ids": c})
            n += len(c)
    return chunks, n


if __name__ == "__main__":
    disk_only = "--disk-only" in sys.argv
    out = []
    stats = {}
    if not disk_only:
        c, n = collect_chat(); out += c; stats["chat_gen"] = (len(c), n)
    c, n = collect_prose(90000); out += c; stats["prose"] = (len(c), n)
    c, n = collect_code(45000); out += c; stats["code_py"] = (len(c), n)
    with open(R + "N1B_corpus.jsonl", "w") as f:
        for i, ch in enumerate(out):
            ch = {"cid": i, **ch}
            f.write(json.dumps(ch) + "\n")
    tot = sum(len(ch["ids"]) for ch in out)
    print(json.dumps({"stats_chunks_tokens": stats, "n_chunks": len(out), "n_tokens": tot}, indent=1))
    from collections import Counter
    print(Counter(ch["src"] for ch in out))
