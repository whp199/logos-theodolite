#!/usr/bin/env python3
"""The two nulls' controls, dose by dose, read the trial's way on the CPU: each answer cut at 48 word-pieces as the
trial's reader cuts it (trial_read.load), each control's mass from readlib's counts and masses against the other
controls. A control's mass depends on the controls alone, so the neurons are not loaded. Writes control_tails.json
beside this file: each null's p50, p95, p99, fifth largest and largest at all four doses, at +-1 and at +-2, and the
strongest control of each with its words.

Run: CUDA_VISIBLE_DEVICES= .venv/bin/python -I experiments/results/census_L45_columns/control_tails.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT / "atlas_data" / "census" / "trials" / "prereg"
sys.path.insert(0, str(P))
sys.path.insert(0, str(ROOT))

import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from backend import census as cz  # noqa: E402

RUNS = {"isotropic_seed7": ROOT / "atlas_data" / "census" / "runs" / "L45-break_rel.jsonl",
        "columns_seed11": ROOT / "atlas_data" / "census" / "nulls" / "L45-break_rel-columns-controls.jsonl"}
DOSES = {"all": (-2.0, -1.0, 1.0, 2.0), "pm1": (-1.0, 1.0), "pm2": (-2.0, 2.0)}
CUT = 48


def controls_rows(path: Path, tok) -> list:
    """The run's unpushed answers and controls, cut as trial_read.load cuts them (its neurons skipped)."""
    out = []
    for r in cz.read_rows(path, tolerate_tail=True):
        if r["kind"] not in ("baseline", "random"):
            continue
        ids, lp, text, fin = r["token_ids"], r["logprob"], r["text"], r["finish"]
        if len(ids) > CUT:
            body = ids[:-1] if fin == "stop" else ids
            ids, lp, text, fin = ids[:CUT], lp[:CUT], tok.decode(body[:CUT], skip_special_tokens=True), "length"
        out.append({"layer": r["layer"], "kind": r["kind"], "neuron": int(r["neuron"]), "prompt": r["prompt"],
                    "dose": float(r["dose"]), "finish": fin, "text": text, "token_ids": ids, "lp": lp})
    return out


def main() -> None:
    tok = Tokenizer.from_file(T.TOK)
    out = {"cut": CUT, "columns": ["p50", "p95", "p99", "fifth_largest", "largest"], "nulls": {}}
    for name, path in RUNS.items():
        rows = controls_rows(path, tok)
        out["nulls"][name] = {}
        for label, doses in DOSES.items():
            keep = lambda r, d=doses: any(abs(r["dose"] - x) < 1e-9 for x in d)  # noqa: E731
            m = {u: v for u, v in R.mass_from_counts(R.counts(rows, keep)).items() if u[0] == "random"}
            c = np.array([v[0] for v in m.values()])
            top = max(m, key=lambda u: m[u][0])
            out["nulls"][name][label] = {
                "mass": [round(float(x), 2) for x in (*np.percentile(c, [50, 95, 99]), np.sort(c)[-5], c.max())],
                "strongest": {"control": top[1], "mass": round(m[top][0], 2),
                              "words": [[w, round(float(z), 2)] for z, w in m[top][1]]}}
        print(name, {k: v["mass"] for k, v in out["nulls"][name].items()}, flush=True)
    cz.write_json_atomic(HERE / "control_tails.json", out)


if __name__ == "__main__":
    main()
