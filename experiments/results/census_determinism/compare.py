#!/usr/bin/env python3
"""Is a census run the same bit for bit across processes on one card? Reads the runs `run_det.sh` wrote on the home
RTX 5090 on 29 October 2026 (kept in atlas_data/offline/census_determinism/, the run files being over 5 MB) and
compares, row by row and field by field:
- the unbroken run against the same run taken in two shards and merged (`census_tools.py merge`);
- the two canaries written in separate processes (`--canary-only`, CANA and CANB), and shard 1's own canary of plan
  batch 0, which the merge compared with shard 0's rows.
CPU only; writes nothing but its report to stdout.

    CUDA_VISIBLE_DEVICES= /home/eric/bin/ai/theodolite2/.venv/bin/python -I compare.py > compare.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUNS = Path("/home/eric/bin/ai/theodolite2/atlas_data/offline/census_determinism")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def compare(a: Path, b: Path) -> dict:
    ra, rb = rows(a), rows(b)
    fields = sorted({k for r in ra for k in r})
    differ = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
    return {"a": str(a.relative_to(RUNS)), "b": str(b.relative_to(RUNS)), "rows": [len(ra), len(rb)],
            "fields": fields, "rows_differing": len(differ) + abs(len(ra) - len(rb)), "first_differing": differ[:5],
            "bytes_identical": a.read_bytes() == b.read_bytes(), "sha256": [sha256(a), sha256(b)]}


out = {
    "unbroken_vs_merged": compare(RUNS / "census/DET.jsonl", RUNS / "sharded/DET.jsonl"),
    "canary_A_vs_B": compare(RUNS / "census/CANA.canary.jsonl", RUNS / "census/CANB.canary.jsonl"),
    "canary_A_vs_shard1": compare(RUNS / "census/CANA.canary.jsonl", RUNS / "sharded/DET.s1of2.canary.jsonl"),
    "files": {str(p.relative_to(RUNS)): sha256(p) for p in sorted(RUNS.rglob("*")) if p.is_file()},
}
print(json.dumps(out, indent=1))
