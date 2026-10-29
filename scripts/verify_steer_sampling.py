#!/usr/bin/env python3
"""The push written with the instrument's sampler (`POST /api/atlas/steer` with `sampler` and `samples`), checked on
the real model through the running server: do the three voices stay alike at dose 0 in bf16, as they did on the
stand-in in float32, and what do eight samples cost against one?

Each step of a sampled /steer is one draw shared by the three voices of a sample (an exponential race seeded
seed + s), so the voices part only where the push moves the odds. At dose 0 the push moves nothing, so the three rows
of a sample must write the same words, unless the rows of one batch are not computed alike on the card. A sample s of
several is drawn with seed + s, but in a batch of 3k rows where one sample alone is a batch of 3; the writer's floor
(scripts/verify_atlas_writer.py) found that batch shapes can part an unpushed answer from itself, so whether sample s
of eight is the answer a single sample at seed + s writes is measured here, not assumed.

Fixed before the run (29 October 2026), on one question (QUESTION, a chat turn), at most MAX_NEW word-pieces:
- dose 0, greedy, one sample: up, baseline and down the same text;
- dose 0, sampled (temperature 1, top-k 64, top-p 0.95, seed SEED), one sample and eight: in every sample the three
  voices the same text; the eight-sample request again: the same eight texts;
- each single sample at seed SEED + s, s = 0..7, against sample s of the eight: the word-pieces they share before
  they first part (all of them where they never do), as text, read back through the tokenizer-free comparison of the
  texts' characters (reported, not a pass or fail);
- dose 0.8 on one neuron (layer 45, counted from 0; n drawn with --seed), eight samples: the share of samples whose
  pushed voices part from the baseline (a sanity check that the push acts; reported);
- the wall time of each request, as the page waits for it.
Exits 0 when every "the same" above holds.

Stop nothing: this talks to the server on :8765. It writes experiments/results/atlas_steer_sampling/steer_sampling.json
(--out) with the commit and the server's /api/health.

Run: .venv/bin/python -I scripts/verify_steer_sampling.py [--base http://127.0.0.1:8765]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT = ROOT_DIR / "experiments" / "results" / "atlas_steer_sampling" / "steer_sampling.json"

QUESTION = "Describe a walk through a forest in early spring."
MAX_NEW = 64
SEED = 1234
SAMPLER = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
LAYER = 45


def shared_chars(a: str, b: str) -> int:
    """Characters held in common before the two texts first part (all of the shorter where one begins the other)."""
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    with httpx.Client(base_url=args.base, timeout=1800) as c:
        health = c.get("/api/health").json()
        arch = c.get("/api/atlas/arch").json()
        n = random.Random(args.seed).randrange(int(arch["d_ff"]))
        timings: Dict[str, float] = {}

        def steer(tag: str, dose: float, sampler: Optional[Dict[str, Any]], samples: int) -> Dict[str, Any]:
            body: Dict[str, Any] = {"prompt": QUESTION, "layer": LAYER, "n": n, "dose": dose, "max_new": MAX_NEW, "samples": samples}
            if sampler is not None:
                body["sampler"] = sampler
            t = time.time()
            r = c.post("/api/atlas/steer", json=body)
            timings[tag] = round(time.time() - t, 2)
            if r.status_code != 200:
                raise RuntimeError(f"{tag}: {r.status_code} {r.text[:400]}")
            print(f"{tag}: {timings[tag]} s", flush=True)
            return r.json()

        def voices(rep: Dict[str, Any]) -> List[List[str]]:
            k = rep["up"] if isinstance(rep["up"], list) else [rep["up"]]
            b = rep["baseline"] if isinstance(rep["baseline"], list) else [rep["baseline"]]
            d = rep["down"] if isinstance(rep["down"], list) else [rep["down"]]
            return [[u, m, w] for u, m, w in zip(k, b, d)]

        checks: Dict[str, bool] = {}
        g = steer("greedy_1", 0.0, None, 1)
        checks["greedy, dose 0: the three voices the same"] = len(set(voices(g)[0])) == 1
        s1 = steer("sampled_1", 0.0, {**SAMPLER, "seed": SEED}, 1)
        checks["sampled, one sample, dose 0: the three voices the same"] = len(set(voices(s1)[0])) == 1
        s8 = steer("sampled_8", 0.0, {**SAMPLER, "seed": SEED}, 8)
        v8 = voices(s8)
        checks["sampled, eight samples, dose 0: in every sample the three voices the same"] = all(len(set(v)) == 1 for v in v8)
        s8b = steer("sampled_8_again", 0.0, {**SAMPLER, "seed": SEED}, 8)
        checks["sampled, eight samples again: the same eight texts"] = voices(s8b) == v8
        alone = []
        for s in range(8):
            one = steer(f"sampled_1_seed+{s}", 0.0, {**SAMPLER, "seed": SEED + s}, 1)
            a = voices(one)[0][1]
            alone.append({"s": s, "same": a == v8[s][1], "shared_chars": shared_chars(a, v8[s][1]),
                          "chars_alone": len(a), "chars_in_eight": len(v8[s][1])})
        p8 = steer("sampled_8_dose_0.8", 0.8, {**SAMPLER, "seed": SEED}, 8)
        vp = voices(p8)
        parted = sum(1 for u, m, w in vp if u != m or w != m)
    out = {"what": __doc__.split("\n\n")[0],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip(),
           "health": {k: health.get(k) for k in ("model", "model_id", "unpacker", "device", "gpu", "runtime") if k in health},
           "question": QUESTION, "max_new": MAX_NEW, "sampler": {**SAMPLER, "seed": SEED}, "push": {"layer": LAYER, "n": n, "layers_from": 0},
           "checks": checks, "timings_s": timings,
           "sample_s_of_eight_against_seed_plus_s_alone": alone,
           "dose_0.8": {"samples": len(vp), "pushed_voices_parted": parted},
           "texts": {"greedy_1": g, "sampled_1": s1, "sampled_8": s8, "sampled_8_dose_0.8": p8},
           "sampling": s8.get("sampling")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    print("sample s of eight equals seed + s alone:", sum(x["same"] for x in alone), "of 8;",
          "shared characters", [x["shared_chars"] for x in alone])
    print(f"dose 0.8: {parted} of {len(vp)} samples' pushed voices parted from the baseline")
    print("timings (s):", timings)
    print(f"wrote {args.out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
