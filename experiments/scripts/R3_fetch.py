#!/usr/bin/env python3
"""R3 fetch (network, run once; loads no model, uses no GPU): the three sources the user approved for R3.

1. jpp_lens (github.com/safety-research/jpp_lens) at commit d7efc0de, a shallow checkout of that one commit.
2. J++'s six eval item files, lens-eval-{association,multihop,multilingual,order-ops,poetry,typo}.json. They are part
   of that checkout (data/jlens/evaluations/), so they come with step 1 and are pinned on their own.
3. WikiText-103 (Salesforce/wikitext, wikitext-103-raw-v1, train) at revision b08601e0, Miru's pin: the stream from
   row 0 up to the row of the 1,000th record of at least 600 characters, the records J++'s loader keeps
   (src/jlens/examples.py:9-27: len(text.strip()) >= 600, stream order). The rows in between (empty lines,
   headings, short paragraphs) are kept too, so that the selection can be checked and Miru's own prompts can be
   rebuilt from the same stream later. The rows are read from the first parquet shard at the pinned revision by
   range requests, one row group at a time, with the small parquet reader below; the shard itself is not stored.

Everything goes into experiments/R3_cache/ (gitignored). jpp_lens is installed, editable and without dependencies
(uv, --offline, with the shared .venv's setuptools as build backend), into its own environment
experiments/R3_cache/env, which sees the shared .venv's site-packages read-only through a .pth file: torch,
transformers, numpy, huggingface_hub, safetensors, accelerate and pytest are reused, not downloaded, and the shared
.venv is never written to.

What jpp_lens also asks for and neither the shared .venv nor the uv cache holds for Python 3.12 (pandas, pyarrow,
datasets) is NOT installed, because nothing beyond the three sources may be downloaded. Without pandas,
scripts/jpp_cli.py and the lens_evals package do not import; jlens and workspace_lens (the router, PCA, k-means,
the expert and condense fitters, the LRP rules) do. datasets would only serve jlens/examples.py's unpinned stream,
which the pinned rows replace; pyarrow is replaced, for reading WikiText, by the reader below.

Writes experiments/scripts/R3_sources.json, the committed manifest: commits, revisions, sha256 and size of every
fetched file, the WikiText record indices and per-record sha256, and the environment's versions. Texts and items
never enter git. A step whose cache is already present is not fetched again; it is checked against what it holds.
Re-verify later, offline, with experiments/scripts/R3_sources_check.py.

Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_fetch.py
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import datetime
import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------------------------------------ pins and paths
REPO = Path(__file__).resolve().parents[2]            # the checkout this file lives in (worktree or main)
SCRIPTS = REPO / "experiments" / "scripts"
CACHE = REPO / "experiments" / "R3_cache"             # gitignored
MANIFEST = SCRIPTS / "R3_sources.json"

JPP_URL = "https://github.com/safety-research/jpp_lens"
JPP_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"
JPP_DIR = CACHE / "jpp_lens"
ITEMS_SUBDIR = "data/jlens/evaluations"
ITEM_SLUGS = ("association", "multihop", "multilingual", "order-ops", "poetry", "typo")

WT_DATASET = "Salesforce/wikitext"
WT_CONFIG = "wikitext-103-raw-v1"
WT_SPLIT = "train"
WT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WT_SHARD = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
WT_N_RECORDS = 1000
WT_MIN_CHARS = 600
WT_DIR = CACHE / "wikitext"
WT_ROWS = WT_DIR / "rows.jsonl"                        # one {"row": i, "text": ...} per raw row, from row 0
WT_REMOTE = WT_DIR / "remote.json"                     # what the hub said about the revision and the shard

ENV_DIR = CACHE / "env"
SYSTEM_PYTHON = "/usr/bin/python3.12"                 # the interpreter the shared .venv was made from
SHARED_SITE = Path("/home/eric/bin/ai/theodolite2/.venv/lib/python3.12/site-packages")
SHARED_PTH = "_r3_shared_venv.pth"
# Reused read-only from the shared .venv, beside jpp_lens's own pins (pyproject.toml at JPP_COMMIT) for the record
REUSED = {"torch": "2.13.0 (extra cpu) / 2.11.0 (extra cu128)", "transformers": "5.14.1", "numpy": "2.5.1",
          "huggingface_hub": "1.26.0", "safetensors": None, "accelerate": "1.12.0 (extras cpu, cu128)",
          "pytest": "9.1.1 (extra dev)"}
NOT_INSTALLED = {
    "pandas": "pin 3.0.5; no Python 3.12 build in the shared .venv or the uv cache, and no download beyond the three "
              "sources is approved. scripts/jpp_cli.py and lens_evals import it at the top, so neither imports here.",
    "pyarrow": "pin 25.0.0; same reason. WikiText is read by R3_fetch.py's own parquet reader instead.",
    "datasets": "pin 5.0.1; only jlens/examples.py uses it, to stream WikiText unpinned; the pinned rows replace it.",
}

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sequence_sha256(hashes: List[str]) -> str:
    """The hash of an ordered list of records: sha256 of their sha256 hex digests joined by newlines."""
    return sha256_bytes("\n".join(hashes).encode("ascii"))


def run(cmd: List[str], **kw) -> str:
    log("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        raise SystemExit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def git(*args: str) -> str:
    return run(["git", "-C", str(JPP_DIR), *args]).strip()


# ------------------------------------------------------------------------------------------------ 1-2. jpp_lens, items
def fetch_jpp() -> Dict[str, Any]:
    if not (JPP_DIR / ".git").exists():
        JPP_DIR.mkdir(parents=True, exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", JPP_URL)
        git("fetch", "-q", "--depth", "1", "origin", JPP_COMMIT)
        git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
    else:
        log("jpp_lens checkout present; not fetched again")
    head = git("rev-parse", "HEAD")
    if head != JPP_COMMIT:
        raise SystemExit(f"jpp_lens HEAD is {head}, not {JPP_COMMIT}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"jpp_lens checkout has local changes:\n{dirty}")
    files = {}
    for rel in git("ls-files", "-z").split("\0"):
        if rel:
            p = JPP_DIR / rel
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    items = {}
    for slug in ITEM_SLUGS:
        rel = f"{ITEMS_SUBDIR}/lens-eval-{slug}.json"
        items[slug] = {"path": rel, "sha256": files[rel]["sha256"], "size": files[rel]["size"],
                       "n_items": len(json.loads((JPP_DIR / rel).read_text(encoding="utf-8"))["items"])}
    log(f"jpp_lens at {head}: {len(files)} tracked files; items", {s: v["n_items"] for s, v in items.items()})
    return {
        "jpp_lens": {
            "url": JPP_URL, "commit": head,
            "commit_date": git("log", "-1", "--format=%cI"), "commit_subject": git("log", "-1", "--format=%s"),
            "fetch": "git init; git fetch --depth 1 origin <commit>; git checkout FETCH_HEAD",
            "cache_dir": str(JPP_DIR.relative_to(REPO)), "license": "Apache-2.0 (LICENSE and NOTICE at the commit)",
            "files": files,
        },
        "jpp_eval_items": {
            "from": "jpp_lens", "commit": head, "dir": ITEMS_SUBDIR, "files": items,
            "n_items_total": sum(v["n_items"] for v in items.values()),
            "note": "The items are read from the gitignored checkout; only these hashes and counts are committed.",
        },
    }


# ------------------------------------------------------------------------------------------------ environment
def env_python() -> Path:
    return ENV_DIR / "bin" / "python"


def env_site() -> Path:
    return ENV_DIR / "lib" / "python3.12" / "site-packages"


ENV_PROBE = r'''
import importlib.metadata as m, json, sys
import torch
names = ["jpp-lens", "torch", "transformers", "numpy", "huggingface_hub", "safetensors", "accelerate", "pytest"]
print(json.dumps({"python": sys.version.split()[0], "versions": {n: m.version(n) for n in names},
                  "locations": {n: str(m.distribution(n).locate_file("")) for n in names},
                  "jpp_lens_src": __import__("workspace_lens").__file__, "cuda_available": torch.cuda.is_available()}))
'''


def build_env() -> Dict[str, Any]:
    if not env_python().exists():
        run(["uv", "venv", "--offline", "--python", SYSTEM_PYTHON, str(ENV_DIR)])
    (env_site() / SHARED_PTH).write_text(str(SHARED_SITE) + "\n", encoding="utf-8")
    # --no-build-isolation: the build backend (setuptools) is the shared .venv's, seen through the .pth file
    run(["uv", "pip", "install", "--offline", "--no-deps", "--no-build-isolation", "--python", str(env_python()),
         "-e", str(JPP_DIR)])
    # -B: no byte-code is written, so the shared .venv's __pycache__ directories are left as they are
    info = json.loads(run([str(env_python()), "-I", "-B", "-c", ENV_PROBE],
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}))
    if info["cuda_available"]:
        raise SystemExit("the GPU is visible to the env; it must stay hidden")
    for n in REUSED:
        if not info["locations"][n].startswith(str(SHARED_SITE)):
            raise SystemExit(f"{n} does not come from the shared .venv: {info['locations'][n]}")
    if not info["jpp_lens_src"].startswith(str(JPP_DIR / "src")):
        raise SystemExit(f"workspace_lens is not imported from the checkout: {info['jpp_lens_src']}")
    if git("status", "--porcelain"):
        raise SystemExit("the editable install left files in the jpp_lens checkout")
    log("env:", info["versions"])
    return {
        "dir": str(ENV_DIR.relative_to(REPO)), "python": info["python"], "interpreter": SYSTEM_PYTHON,
        "shared_site_packages": str(SHARED_SITE), "shared_pth": SHARED_PTH,
        "jpp_lens": {"version": info["versions"]["jpp-lens"],
                     "install": "uv pip install --offline --no-deps --no-build-isolation -e <checkout> "
                                "(src/ on the path)"},
        "reused_read_only": {n: info["versions"][n] for n in REUSED},
        "jpp_lens_pins": {n: v for n, v in REUSED.items() if v},
        "not_installed": NOT_INSTALLED,
        "deviations": [
            "torch 2.14.1, transformers 5.20.0, numpy 2.5.4, huggingface_hub 2.2.0 and accelerate 1.15.0 are the "
            "shared .venv's, not jpp_lens's pins (reused, not downloaded); pytest 9.1.1 matches its pin.",
            "pandas, pyarrow and datasets are not installed (see not_installed): scripts/jpp_cli.py and "
            "lens_evals cannot be imported until pandas is, which needs a download not yet approved.",
        ],
    }


# ------------------------------------------------------------------------------------------------ 3. WikiText
# A minimal parquet reader for this one file: a single optional BYTE_ARRAY column ("text"), SNAPPY pages, a PLAIN
# dictionary page and RLE_DICTIONARY (or PLAIN) data pages, v1 or v2. Anything else raises. Every page's decompressed
# size, value count and every row group's row count are checked against the file's own metadata.
def _varint(b: bytes, i: int) -> Tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, i


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _thrift_value(b: bytes, i: int, t: int) -> Tuple[Any, int]:
    """One value of Thrift's compact protocol; structs come back as {field id: value}."""
    if t in (1, 2):
        return t == 1, i
    if t == 3:
        return struct.unpack_from("b", b, i)[0], i + 1
    if t in (4, 5, 6):
        v, i = _varint(b, i)
        return _zigzag(v), i
    if t == 7:
        return struct.unpack_from("<d", b, i)[0], i + 8
    if t == 8:
        n, i = _varint(b, i)
        return bytes(b[i:i + n]), i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = _varint(b, i)
        out = []
        for _ in range(n):
            if et in (1, 2):
                v, i = b[i] == 1, i + 1
            else:
                v, i = _thrift_value(b, i, et)
            out.append(v)
        return out, i
    if t == 12:
        return _thrift_struct(b, i)
    raise ValueError(f"thrift type {t} not handled")


def _thrift_struct(b: bytes, i: int) -> Tuple[Dict[int, Any], int]:
    out: Dict[int, Any] = {}
    last = 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return out, i
        delta, t = h >> 4, h & 0x0F
        if delta:
            fid = last + delta
        else:
            v, i = _varint(b, i)
            fid = _zigzag(v)
        last = fid
        out[fid], i = _thrift_value(b, i, t)


def _snappy(b: bytes) -> bytes:
    """Raw (unframed) Snappy decompression."""
    n, i = _varint(b, 0)
    out = bytearray()
    while i < len(b):
        tag = b[i]
        i += 1
        kind = tag & 3
        if kind == 0:                                  # literal
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(b[i:i + nb], "little")
                i += nb
            ln += 1
            out += b[i:i + ln]
            i += ln
            continue
        if kind == 1:
            ln, off = 4 + ((tag >> 2) & 7), ((tag >> 5) << 8) | b[i]
            i += 1
        elif kind == 2:
            ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 2], "little")
            i += 2
        else:
            ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 4], "little")
            i += 4
        if off <= 0 or off > len(out):
            raise ValueError("bad snappy offset")
        start = len(out) - off
        for k in range(ln):                            # copies may overlap their own output
            out.append(out[start + k])
    if len(out) != n:
        raise ValueError(f"snappy: {len(out)} bytes, header says {n}")
    return bytes(out)


def _rle_hybrid(b: bytes, i: int, end: int, width: int, count: int) -> List[int]:
    """Parquet's RLE / bit-packed hybrid, read until `count` values."""
    out: List[int] = []
    nbytes = (width + 7) // 8
    mask = (1 << width) - 1
    while len(out) < count and i < end:
        h, i = _varint(b, i)
        if h & 1:                                      # bit-packed groups of 8
            groups = h >> 1
            bits = int.from_bytes(b[i:i + groups * width], "little")
            i += groups * width
            out.extend((bits >> (k * width)) & mask for k in range(groups * 8))
        else:
            v = int.from_bytes(b[i:i + nbytes], "little")
            i += nbytes
            out.extend([v] * (h >> 1))
    if len(out) < count:
        raise ValueError("RLE run ended early")
    return out[:count]


def _plain_strings(b: bytes, n: int) -> List[bytes]:
    out, i = [], 0
    for _ in range(n):
        ln = struct.unpack_from("<I", b, i)[0]
        out.append(b[i + 4:i + 4 + ln])
        i += 4 + ln
    return out


def _column_chunk(chunk: bytes, n_values: int) -> List[str]:
    i, dictionary, values = 0, None, []
    while len(values) < n_values:
        hdr, i = _thrift_struct(chunk, i)
        ptype, usize, csize = hdr[1], hdr[2], hdr[3]
        body = chunk[i:i + csize]
        i += csize
        if ptype == 2:                                 # dictionary page
            dph = hdr[7]
            if dph.get(2, 0) not in (0, 2):
                raise ValueError("dictionary page not PLAIN")
            page = _snappy(body)
            if len(page) != usize:
                raise ValueError("dictionary page size")
            dictionary = _plain_strings(page, dph[1])
            continue
        if ptype == 0:                                 # data page v1: levels inside the compressed body
            dph = hdr[5]
            nv, enc = dph[1], dph[2]
            page = _snappy(body)
            if len(page) != usize:
                raise ValueError("data page size")
            dl_len = struct.unpack_from("<I", page, 0)[0]
            defs = _rle_hybrid(page, 4, 4 + dl_len, 1, nv)
            data = page[4 + dl_len:]
        elif ptype == 3:                               # data page v2: levels uncompressed, before the body
            dph = hdr[8]
            nv, enc, dl_len, rl_len = dph[1], dph[4], dph[5], dph.get(6, 0)
            if rl_len:
                raise ValueError("repetition levels in a flat column")
            defs = _rle_hybrid(body, 0, dl_len, 1, nv)
            rest = body[dl_len:]
            data = _snappy(rest) if dph.get(7, True) else rest
        else:
            raise ValueError(f"page type {ptype} not handled")
        present = sum(defs)
        if present != nv:
            raise ValueError("null text values")
        if enc == 0:
            strings = _plain_strings(data, present)
        elif enc in (2, 8):
            width = data[0]
            idx = _rle_hybrid(data, 1, len(data), width, present)
            strings = [dictionary[k] for k in idx]
        else:
            raise ValueError(f"encoding {enc} not handled")
        values.extend(s.decode("utf-8") for s in strings)
    if len(values) != n_values:
        raise ValueError("column chunk value count")
    return values


def read_wikitext_prefix() -> Tuple[List[str], Dict[str, Any]]:
    """Stream the pinned shard's row groups in order until the 1,000th record of >= 600 characters."""
    from huggingface_hub import HfApi, HfFileSystem

    info = HfApi().dataset_info(WT_DATASET, revision=WT_REVISION, files_metadata=True)
    sib = {s.rfilename: s for s in info.siblings}[WT_SHARD]
    card = info.card_data.to_dict() if info.card_data else {}
    fs = HfFileSystem()
    path = f"datasets/{WT_DATASET}@{WT_REVISION}/{WT_SHARD}"
    size = fs.info(path)["size"]
    rows: List[str] = []
    kept, groups_read, bytes_read = 0, 0, 0
    with fs.open(path, "rb", block_size=1 << 16) as f:
        f.seek(size - 8)
        tail = f.read(8)
        if tail[4:] != b"PAR1":
            raise SystemExit("not a parquet file")
        flen = struct.unpack("<I", tail[:4])[0]
        f.seek(size - 8 - flen)
        meta, _ = _thrift_struct(f.read(flen), 0)
        bytes_read += 8 + flen
        schema = [(e.get(4), e.get(1), e.get(3)) for e in meta[2]]
        if schema[1:] != [(b"text", 6, 1)]:
            raise SystemExit(f"unexpected schema {schema}")
        for rg in meta[4]:
            (cc,) = rg[1]
            cm = cc[3]
            if cm[4] != 1:
                raise SystemExit(f"codec {cm[4]} is not SNAPPY")
            start = cm[11] if cm.get(11) else cm[9]  # the dictionary page, when there is one, comes first
            f.seek(start)
            chunk = f.read(cm[7])
            bytes_read += cm[7]
            texts = _column_chunk(chunk, cm[5])
            if len(texts) != rg[3]:
                raise SystemExit("row group row count")
            groups_read += 1
            for text in texts:
                rows.append(text)
                if len(text.strip()) >= WT_MIN_CHARS:
                    kept += 1
                    if kept == WT_N_RECORDS:
                        break
            if kept == WT_N_RECORDS:
                break
    remote = {
        "resolved_revision": info.sha, "license": card.get("license"),
        "shard_lfs_sha256": sib.lfs.sha256 if sib.lfs else None, "shard_size": size,
        "shard_rows": meta[3], "shard_row_groups": len(meta[4]), "created_by": meta.get(6, b"").decode(),
        "row_groups_read": groups_read, "bytes_read": bytes_read, "kept": kept, "rows": len(rows),
    }
    return rows, remote


def fetch_wikitext() -> Dict[str, Any]:
    WT_DIR.mkdir(parents=True, exist_ok=True)
    if not WT_ROWS.exists():
        if os.environ.get("HF_HUB_OFFLINE"):
            raise SystemExit("HF_HUB_OFFLINE is set; the WikiText fetch needs the network once")
        rows, remote = read_wikitext_prefix()
        if remote["resolved_revision"] != WT_REVISION:
            raise SystemExit(f"the hub resolved {remote['resolved_revision']}, not {WT_REVISION}")
        if remote["kept"] != WT_N_RECORDS:
            raise SystemExit(f"only {remote['kept']} records of >= {WT_MIN_CHARS} characters in the first shard")
        tmp = WT_ROWS.with_suffix(".partial")
        with open(tmp, "w", encoding="utf-8") as f:
            for i, text in enumerate(rows):
                f.write(json.dumps({"row": i, "text": text}, ensure_ascii=False) + "\n")
        tmp.replace(WT_ROWS)
        WT_REMOTE.write_text(json.dumps(remote, indent=1) + "\n", encoding="utf-8")
        log(f"WikiText: read {remote['row_groups_read']} of {remote['shard_row_groups']} row groups "
            f"({remote['bytes_read']:,} bytes)")
    else:
        log("WikiText rows present; not fetched again")
    remote = json.loads(WT_REMOTE.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in WT_ROWS.read_text(encoding="utf-8").splitlines()]
    if [r["row"] for r in rows] != list(range(len(rows))):
        raise SystemExit("the cached rows are not numbered 0..n-1")
    records = [[r["row"], len(r["text"]), text_sha256(r["text"])] for r in rows
               if len(r["text"].strip()) >= WT_MIN_CHARS]
    if len(records) != WT_N_RECORDS or records[-1][0] != len(rows) - 1:
        raise SystemExit("the cached rows do not end at the 1,000th kept record")
    log(f"WikiText: {len(rows)} raw rows cached, {len(records)} records kept (last at row {records[-1][0]})")
    return {
        "dataset_id": WT_DATASET, "config": WT_CONFIG, "split": WT_SPLIT, "revision": WT_REVISION,
        "resolved_revision": remote["resolved_revision"], "license": remote["license"],
        "shard": WT_SHARD, "shard_lfs_sha256": remote["shard_lfs_sha256"], "shard_size": remote["shard_size"],
        "shard_rows": remote["shard_rows"], "shard_row_groups": remote["shard_row_groups"],
        "shard_created_by": remote["created_by"],
        "fetch": "huggingface_hub.HfFileSystem at the revision, range requests: the footer, then row groups in "
                 "order, decoded by R3_fetch.py's parquet reader; the shard is not stored",
        "row_groups_read": remote["row_groups_read"], "bytes_read": remote["bytes_read"],
        "selection": f"the first {WT_N_RECORDS} rows with len(text.strip()) >= {WT_MIN_CHARS}, in stream order, "
                     "text kept unstripped (jpp_lens src/jlens/examples.py:9-27)",
        "cache_file": str(WT_ROWS.relative_to(REPO)), "cache_file_sha256": sha256_file(WT_ROWS),
        "cache_file_size": WT_ROWS.stat().st_size, "raw_rows_cached": len(rows),
        "raw_rows_sequence_sha256": sequence_sha256([text_sha256(r["text"]) for r in rows]),
        "records_sequence_sha256": sequence_sha256([r[2] for r in records]),
        "sequence_sha256_rule": "sha256 of the records' sha256 hex digests joined by newlines, in order",
        "fit_prompts": "the first 64 records (J++'s 64 fit prompts); all 1,000 feed the router",
        "records_columns": ["row", "chars", "sha256 of the UTF-8 text"],
        "records": records,
        "note": "Only row indices and hashes are committed; the texts stay in the gitignored cache. Whether this "
                "prefix holds all of Miru's 1,000 rechunked prompts is checked when they are rebuilt.",
    }


# ------------------------------------------------------------------------------------------------ manifest
def dumps_compact(obj: Any, indent: int = 0) -> str:
    """JSON with one line per list of scalars (a WikiText record) and per small dict of scalars (a file's sha256 and
    size), so that the manifest diffs line by line."""
    pad, inner = " " * indent, " " * (indent + 1)
    scalar = (str, int, float, bool, type(None))
    if isinstance(obj, list) and all(isinstance(v, scalar) for v in obj):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, dict) and len(obj) <= 4 and all(isinstance(v, scalar) for v in obj.values()):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return "[\n" + ",\n".join(inner + dumps_compact(v, indent + 1) for v in obj) + "\n" + pad + "]"
    if isinstance(obj, dict):
        return "{\n" + ",\n".join(f"{inner}{json.dumps(k)}: {dumps_compact(v, indent + 1)}"
                                   for k, v in obj.items()) + "\n" + pad + "}"
    return json.dumps(obj, ensure_ascii=False)


def write_manifest(m: Dict[str, Any]) -> None:
    MANIFEST.write_text(dumps_compact(m) + "\n", encoding="utf-8")
    if json.loads(MANIFEST.read_text(encoding="utf-8")) != m:
        raise SystemExit("the manifest does not read back as written")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    if not shutil.which("uv"):
        raise SystemExit("uv is needed to install jpp_lens into its env from the local cache")
    m: Dict[str, Any] = {
        "schema": 1,
        "written_by": "experiments/scripts/R3_fetch.py",
        "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cache_dir": str(CACHE.relative_to(REPO)),
        "approved": "user decision (D-lens-research): fetch once, pinned, into a gitignored cache: the first 1,000 "
                    "WikiText-103 records at b08601e0, jpp_lens at d7efc0de installed into its own env, J++'s six "
                    "eval item files. Nothing else is downloaded.",
        "check": "CUDA_VISIBLE_DEVICES= .venv/bin/python -I experiments/scripts/R3_sources_check.py [--tests]",
    }
    m.update(fetch_jpp())
    m["environment"] = build_env()
    m["wikitext"] = fetch_wikitext()
    write_manifest(m)
    log("wrote", MANIFEST.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
