#!/usr/bin/env python3
"""R3 00, context only (GPU, the 31B, forward only; not a gate, not registered as one): the pilot's identity check (a)
read under jpp_lens's OWN R-lens patcher (workspace_lens.lrp.apply_lrp_rules_for_mode(..., "rlens"), imported read
only from the pinned checkout at d7efc0de), beside R3's rules as registered (R3_lrp.py at e1733fe, the code the first
pilot ran) and R3's rules under D33 (R3_lrp.py at HEAD), on the same positions: WikiText fit records 0-15, the 31B
loaded as the pilot loads it (R3_gpu.load_w4a16), the top-1 word-piece of block 59's output read through the
instrument's unembed, each condition against the stock forward. A second stock forward shows whether the stock
forward repeats bit for bit. Writes R3_00_jpp_identity.json: the flips per condition, the positions that flip, their
overlaps, and the hashes of the code that ran. Nothing here gates anything (A14 notes it as context).

  /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_00_jpp_identity.py --run
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.dont_write_bytecode = True

import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R3_common as C  # noqa: E402
import R3_fitlib as FL  # noqa: E402
import R3_gpu as G  # noqa: E402
import R3_lrp  # noqa: E402

T0 = time.time()
OUT = C.RESULTS / "R3_00_jpp_identity.json"
REGISTERED_COMMIT = "e1733fe4056600a6bc0f6fcb23ad549c9a862802"   # HEAD of the first pilot (R3_gpu_runs.jsonl)
FIRST_PILOT_DIFFER = 90                                            # R3_00_pilot.json at 8996fb5


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def registered_lrp() -> Tuple[Any, str]:
    """R3_lrp.py as the first pilot ran it (git show at REGISTERED_COMMIT), loaded as its own module."""
    src = subprocess.run(["git", "-C", str(C.REPO), "show", f"{REGISTERED_COMMIT}:experiments/scripts/R3_lrp.py"],
                         capture_output=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="r3_lrp_registered_"))
    p = d / "R3_lrp_registered.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("R3_lrp_registered", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # its dataclasses look their module up while they are built
    spec.loader.exec_module(mod)
    return mod, sha256_bytes(src)


@torch.no_grad()
def read(dec: FL.Decoder, text: str, ctx) -> Tuple[torch.Tensor, torch.Tensor]:
    last = dec.n_layers - 1
    with ctx:
        _, o = FL.block_outputs(dec, text, [last], C.MAX_SEQ_LEN)
    h = o[last]
    return dec.unembed(h).argmax(-1).cpu(), h.cpu()


def run(args) -> int:
    C.import_jpp()                                     # verifies the pinned checkout first
    from workspace_lens.lrp.lrp import apply_lrp_rules_for_mode

    reg, reg_sha = registered_lrp()
    recs = C.wikitext_records()
    texts = recs[:C.PILOT_LRP_N_PROMPTS]
    with G.gpu_job("R3_00 jpp_lens identity context (31B)", sys.argv, args.allow_uncommitted, log) as job:
        dev = torch.device("cuda:0")
        vram_before = G.vram_record(dev)
        hf, tok = G.load_w4a16()
        dec = FL.Decoder(hf, tok, name=C.MODEL_ID)
        with apply_lrp_rules_for_mode(hf, "rlens") as jrep:
            jn = jrep.patched_names()
        rep = R3_lrp.plan(hf)
        conds = {"stock_repeat": lambda: nullcontext(),
                 "jpp_lens_rlens": lambda: apply_lrp_rules_for_mode(hf, "rlens"),
                 "r3_registered": lambda: reg.rlens(hf),
                 "r3_d33": lambda: R3_lrp.rlens(hf)}
        flips: Dict[str, Set[Tuple[int, int]]] = {k: set() for k in conds}
        maxdiff = {k: 0.0 for k in conds}
        total = 0
        for i, t in enumerate(texts):
            t0, h0 = read(dec, t, nullcontext())
            total += t0.numel()
            for k, mk in conds.items():
                tk, hk = read(dec, t, mk())
                flips[k] |= {(i, int(p)) for p in torch.nonzero(tk != t0).flatten()}
                maxdiff[k] = max(maxdiff[k], float((hk - h0).abs().max()))
            log(f"record {i}: " + ", ".join(f"{k} {len(v)}" for k, v in flips.items()))
        J, R, D = flips["jpp_lens_rlens"], flips["r3_registered"], flips["r3_d33"]
        allowed = int((1.0 - C.PILOT_LRP_TOP1_MIN) * total + 1e-9)
        out = {
            "schema": "R3_00_jpp_identity/1", "context_only": True,
            "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "what": ("the pilot's identity check (a), forward only, on WikiText fit records 0-15, each condition's "
                     "top-1 word-piece at block 59 (the instrument's unembed) against the stock forward"),
            "positions": total, "gate_a_allowed_to_differ_for_reference": allowed,
            "prompts_sha256": [C.text_sha256(t) for t in texts],
            "differ": {k: len(v) for k, v in flips.items()},
            "max_abs_block59_diff": {k: v for k, v in maxdiff.items()},
            "overlap": {"jpp_and_r3_registered": len(J & R), "jpp_only": len(J - R), "r3_registered_only": len(R - J),
                        "jpp_and_r3_d33": len(J & D), "r3_registered_and_r3_d33": len(R & D)},
            "r3_registered_reproduces_first_pilot": len(R) == FIRST_PILOT_DIFFER,
            "flip_positions": {k: sorted(v) for k, v in flips.items()},
            "jpp_patched": {k: len(v) for k, v in jn.items()},
            "jpp_patches_same_modules_as_r3": sorted(jn["ln_rule"]) == sorted(rep.ln_rule)
            and sorted(jn["mlp_rule"]) == sorted(rep.mlp_rule),
            "code": {"jpp_lens_commit": C.JPP_COMMIT,
                     "jpp_lens_lrp_py_sha256": C.sha256_file(C.JPP_SRC / "workspace_lens" / "lrp" / "lrp.py"),
                     "this_script_sha256": C.sha256_file(Path(__file__)),
                     "r3_lrp_head_sha256": C.sha256_file(HERE / "R3_lrp.py"),
                     "r3_lrp_registered_commit": REGISTERED_COMMIT, "r3_lrp_registered_sha256": reg_sha},
            "git": job["git"], "cap": job["cap"],
            "vram": {"before_load": vram_before, "end": G.vram_record(dev)},
            "seconds": round(time.time() - T0, 1),
        }
        OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
        job["out"] = str(OUT)
    log(f"wrote {OUT}: differ {out['differ']}, overlap {out['overlap']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    a = ap.parse_args()
    if not a.run:
        ap.print_help()
        return 2
    if OUT.exists() and not a.force:
        raise SystemExit(f"{OUT} exists; --force to rerun (the earlier run stays in the runs log)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
