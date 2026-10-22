#!/usr/bin/env python3
"""R3 power (CPU, before any R3 data): what R2's realised numbers say about R3's rule 4 and rule 5 at R3's counts
(R3_PREREG.md §7.3). No model, no GPU; R2's own files only.

From R2_02_eval.json under R2's design (R3_06_analyze's Table, the code that reproduces R2_03 exactly):
1. Per R2 lens-band (R1 and Miru at R2's bands), against J++ and the naked eye: the prompt-level band-score
   differences, their sign counts (positive, negative, tied: ties are dropped by the sign test), the discordant share,
   and the realised pooled interval half-widths, as R2_03 recorded them.
2. Rule 4 by resampling: R2's valid prompts drawn with replacement within each condition to R3's expected counts
   (80 chat-no-answer, 80 chat-begun, 52 bare, 80 mid-reply: 292), 4,000 times; each draw's two exact one-sided
   sign tests, the larger p against 0.025 / 14 (Holm's first step: sufficient for any test, and necessary for one
   that passes alone); and rule 5 (both pooled differences at least 0.05) on the same draws.
3. Rule 4 for a given effect, exactly: with n_d = round(q x 292) discordant prompts at the band type's median
   discordant share q, and each discordant prompt +1 or -1 (exact for one layer and one position, an approximation
   otherwise), a true pooled difference d gives a sign probability (q + d) / (2q); the power is the binomial tail at
   the smallest count whose exact p is below 0.025 / 14. One baseline only: both must pass, so this is an upper
   bound for the pair.
4. Rule 5 at R3's n: with the standard error of the pooled difference against the naked eye from R2's interval
   (half-width / 1.96) scaled by sqrt(199 / 292), the chance that the point estimate reaches 0.05 for a true
   difference of +0.046 (R1 at code 58 in R2) and +0.05, under a normal approximation.
Writes experiments/results/R3_power.json.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R3_power.py
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import argparse
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C  # noqa: E402
import R3_06_analyze as A  # noqa: E402
from R2_03_analyze import binom_sf_half  # noqa: E402

R3_EXPECTED = {"chat-no-answer": 80, "chat-begun": 80, "bare": 52, "mid-reply": 80}   # §7.3: R2's shares, R3's 80s
N_SIM = 4000
SIM_SEED = 20261024
EFFECTS = (0.05, 0.063, 0.075)
HOLM_FIRST = C.ALPHA / C.HOLM_M
BAND_TYPES = {"single layer (50, 55, 58)": ("50", "55", "58"), "51-54": ("51-54",), "56-57": ("56-57",),
              "40-49 (and so 30-39, not read by R2)": ("40-49",)}


def r4(x):
    return None if x is None else round(float(x), 4)


def k_crit(n: int, alpha: float) -> int:
    """The smallest k whose exact one-sided sign-test p (R2_03's binom_sf_half) is below alpha (n + 1 if none)."""
    for k in range(n + 1):
        if binom_sf_half(k, n) < alpha:
            return k
    return n + 1


def binom_tail(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(C.RESULTS / "R3_power.json"))
    a = ap.parse_args()
    raw = R2.EVAL_JSON.read_bytes()
    data = json.loads(raw)
    ref = json.loads(R2.ANALYSIS_JSON.read_text(encoding="utf-8"))
    D = A.R2_DESIGN
    t = A.Table(D, data)
    boot = A.Boot(t, D.n_boot, D.boot_seed)
    conds = D.conditions
    valid = {c: boot.idx(c, 0)[0] for c in conds}
    n_r2 = sum(len(v) for v in valid.values())
    n_r3 = sum(R3_EXPECTED.values())
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SIM_SEED)
    draws = {c: rng.integers(0, len(valid[c]), size=(N_SIM, R3_EXPECTED[c])) for c in conds}
    checks = []
    for lens, d in ref["bands"].items():
        for band, ev in d.items():
            layers = ev["layers"]
            diffs = {}
            for b in ("jpp", "naive"):
                diffs[b] = {c: (t.prompt_scores(A.PRIMARY, lens, layers, c)[valid[c]]
                                - t.prompt_scores(A.PRIMARY, b, layers, c)[valid[c]]) for c in conds}
            row: Dict[str, Any] = {"lens": lens, "band": band, "r2_failed_rules": [k for k, v in ev["rules"].items()
                                                                                  if not v]}
            for b in ("jpp", "naive"):
                x = np.concatenate([diffs[b][c] for c in conds])
                npos, nneg = int((x > 1e-9).sum()), int((x < -1e-9).sum())
                sg = ev[f"sign_{b}"]
                checks.append(npos == sg["n_pos"] and nneg == sg["n_neg"])
                pooled = ev[f"diff_{b}"]["pooled"]
                hw = (pooled["hi_raw"] - pooled["lo_raw"]) / 2
                row[b] = {"n_pos": npos, "n_neg": nneg, "n_tie": int(len(x) - npos - nneg),
                          "discordant_share": r4((npos + nneg) / len(x)), "pooled_mean": r4(pooled["mean_raw"]),
                          "pooled_half_width": r4(hw), "half_width_scaled_to_r3": r4(hw * math.sqrt(n_r2 / n_r3))}
            # resampling at R3's counts
            p_max = np.zeros(N_SIM)
            r5 = np.ones(N_SIM, bool)
            for b in ("jpp", "naive"):
                xs = np.concatenate([diffs[b][c][draws[c]] for c in conds], 1)      # [N_SIM, 292]
                npos = (xs > 1e-9).sum(1)
                nneg = (xs < -1e-9).sum(1)
                p = np.array([binom_sf_half(int(u), int(u + v)) for u, v in zip(npos, nneg)])
                p_max = np.maximum(p_max, p)
                r5 &= xs.mean(1) >= C.MIN_EFFECT - 1e-12
            row["resampled_at_r3_counts"] = {"rule4_power": r4((p_max < HOLM_FIRST).mean()),
                                             "rule5_share": r4(r5.mean()),
                                             "rules4_and_5": r4(((p_max < HOLM_FIRST) & r5).mean())}
            rows.append(row)
    exact: Dict[str, Any] = {}
    for name, bands in BAND_TYPES.items():
        qs = [row["naive"]["discordant_share"] for row in rows if row["band"] in bands] + \
             [row["jpp"]["discordant_share"] for row in rows if row["band"] in bands]
        q = float(np.median(qs))
        nd = int(round(q * n_r3))
        k = k_crit(nd, HOLM_FIRST)
        exact[name] = {"median_discordant_share": r4(q), "discordant_prompts_at_292": nd,
                       "k_needed_of_n": [k, nd], "power": {f"{e:+.3f}": r4(binom_tail(k, nd, min(1.0, (q + e) / (2 * q))))
                                                         for e in EFFECTS}}
    se = {}
    for row in rows:
        se[f"{row['lens']} {row['band']}"] = row["naive"]["half_width_scaled_to_r3"] / 1.96
    se_r1_58 = se["r1 58"]
    se_med = float(np.median(list(se.values())))

    def p_reach(delta, s):
        return r4(0.5 * math.erfc(-(delta - C.MIN_EFFECT) / s / math.sqrt(2)))
    rule5 = {"se_naked_eye_at_r3": {"r1 58": r4(se_r1_58), "median over R2's lens-bands": r4(se_med)},
             "p_point_estimate_reaches_0.05": {
                 "true +0.046, SE of r1 58": p_reach(0.046, se_r1_58), "true +0.046, median SE": p_reach(0.046, se_med),
                 "true +0.050, any SE": 0.5}}
    hws = [row[b]["pooled_half_width"] for row in rows for b in ("jpp", "naive")]
    head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"schema": "R3_power/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "head": head,
           "input": {"R2_02_eval.json": hashlib.sha256(raw).hexdigest(),
                     "R2_03_analysis.json": C.sha256_file(R2.ANALYSIS_JSON)},
           "ok": all(checks), "check": "the sign counts recomputed here equal R2_03's for every lens-band and baseline",
           "n_valid_r2": n_r2, "n_expected_r3": n_r3, "r3_expected_by_condition": R3_EXPECTED,
           "holm_first_step_alpha": HOLM_FIRST, "n_sim": N_SIM, "sim_seed": SIM_SEED,
           "r2_realised_pooled_half_widths": {"min": r4(min(hws)), "median": r4(float(np.median(hws))),
                                              "max": r4(max(hws)),
                                              "scaled_to_r3": [r4(min(hws) * math.sqrt(n_r2 / n_r3)),
                                                               r4(max(hws) * math.sqrt(n_r2 / n_r3))]},
           "lens_bands": rows, "rule4_exact_by_band_type": exact, "rule5_at_r3_n": rule5}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['lens']:5} {row['band']:6} naked {row['naive']['n_pos']}+/{row['naive']['n_neg']}-/"
              f"{row['naive']['n_tie']}= J++ {row['jpp']['n_pos']}+/{row['jpp']['n_neg']}-  resampled: rule 4 "
              f"{row['resampled_at_r3_counts']['rule4_power']}, rule 5 {row['resampled_at_r3_counts']['rule5_share']}")
    print(json.dumps({"exact": exact, "rule5": rule5, "half_widths": out["r2_realised_pooled_half_widths"]}, indent=1))
    print(f"R3_power: {'ok' if out['ok'] else 'SIGN COUNTS DIFFER FROM R2_03'}; wrote {a.out}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
