#!/usr/bin/env python3
"""R4 power (CPU, before any R4 data): what R2's and R3's realised readings of R1 say about R4's chance of passing at
code 55 and 58, all five rules jointly, at R4's counts (R4_PREREG.md §7). No model, no GPU; R2's and R3's files only.

1. Integrity: from R2_02_eval.json and R3_05_eval.json under R4's design (R3_06_analyze's Table), the sign counts of
   R1 against J++ and the naked eye at 55 and 58 must equal R2_03_analysis.json's bands.r1 and R3_06_analysis.json's
   reference_bands.r1.
2. The population: per condition, R2's and R3's valid prompts together (a prompt's band scores for R1, J++ and the
   naked eye, and its read positions' filtered top-1 words and targets, for rule 3).
3. Resampling at R4's expected valid counts, twice the counts R3 realised (160, 160, 86, 160: 566), 4,000 draws
   within each condition, in two modes:
     point       the population taken as the truth (prompts drawn from it with replacement);
     predictive  the population itself resampled first (at its own size), then R4's prompts drawn from that, so the
                 chance allows for the uncertainty of R2's and R3's own estimates.
   Each draw applies the five rules as R4_03 does, except that the 95% intervals of rules 1 and 2 are the normal
   approximation of the stratified prompt bootstrap (mean +/- 1.96 standard errors, per condition and pooled) rather
   than 2,000 resamples inside each draw. Rule 3 is the stuck-word share of the drawn prompts' positions per condition;
   rule 4 is the larger exact sign-test p of the two baselines, Holm-adjusted over the two bands; rule 5 is both pooled
   differences at least 0.05.
4. The operating characteristic: the chance that the pooled R1 - naked eye estimate reaches 0.05 for a true margin of
   +0.03 to +0.07, at 320 and 640 prompts (normal approximation, the standard error from the point draws).
Writes experiments/results/R4_power.json.
Run: CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R4_power.py
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import argparse
import datetime
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402
import R3_06_analyze as A  # noqa: E402
import R4_common as C  # noqa: E402
import R4_03_analyze as R43  # noqa: E402
from R2_03_analyze import binom_sf_half, holm  # noqa: E402

R4_EXPECTED = {"chat-no-answer": 160, "chat-begun": 160, "bare": 86, "mid-reply": 160}   # R3's 80/80/43/80, doubled
N_SIM = 4000
SIM_SEED = 20261030
Z = 1.96
EFFECTS = (0.03, 0.04, 0.05, 0.06, 0.07)


@functools.lru_cache(maxsize=None)
def p_sign(k: int, n: int) -> float:
    return binom_sf_half(k, n)


def population(D, data: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """Per condition: the valid prompts' band scores (R1, J++, naked eye) at 55 and 58, and per band each prompt's
    positions as (filtered top-1 word, usable target) pairs; under R2's metric ("r2") and under S6 ("s6", without the
    positions whose target is a function word: a prompt left with no scored position has no S6 score, NaN)."""
    t = A.Table(D, data)
    t6 = A.Table(D, R43.without_function_word_targets(data))
    boot = A.Boot(t, 1, 0)
    out: Dict[str, Any] = {}
    for c in D.conditions:
        ok = boot.idx(c, 0)[0]
        pids = [t.prompts[c][i] for i in ok]
        rec: Dict[str, Any] = {"pids": [f"{tag}:{p}" for p in pids]}
        for band, layers in D.bands:
            L = list(layers)
            rec[band] = {}
            for key, tab in (("r2", t), ("s6", t6)):
                at = {pid: i for i, pid in enumerate(tab.prompts[c])}
                full = {k: tab.prompt_scores(A.PRIMARY, k, L, c) for k in ("r1", "jpp", "naive")}
                sc = {k: np.array([full[k][at[pid]] if pid in at else np.nan for pid in pids])
                      for k in ("r1", "jpp", "naive")}
                words = []
                for pid in pids:
                    ws = []
                    for j in tab.prompt_pos[c].get(pid, []):
                        r = tab.pos[j]["r"].get("r1", {}).get(str(L[0]))
                        if r is not None:
                            ws.append((R2.word_key(r["fs"][0]), tab.ct_key[j]))
                    words.append(ws)
                rec[band][key] = {"d_jpp": sc["r1"] - sc["jpp"], "d_naive": sc["r1"] - sc["naive"], "words": words}
        out[c] = rec
    return out


def merge(a: Dict[str, Any], b: Dict[str, Any], bands) -> Dict[str, Any]:
    out = {}
    for c in a:
        rec = {"pids": a[c]["pids"] + b[c]["pids"]}
        for band, _ in bands:
            rec[band] = {key: {"d_jpp": np.concatenate([a[c][band][key]["d_jpp"], b[c][band][key]["d_jpp"]]),
                               "d_naive": np.concatenate([a[c][band][key]["d_naive"], b[c][band][key]["d_naive"]]),
                               "words": a[c][band][key]["words"] + b[c][band][key]["words"]}
                         for key in ("r2", "s6")}
        out[c] = rec
    return out


def word_matrix(words: List[List[Tuple[str, Any]]]) -> Tuple[np.ndarray, np.ndarray]:
    """[prompts, words]: how many of a prompt's positions have that filtered top-1 word, not counting a position whose
    word is its own usable target (rule 3's numerator); and each prompt's number of positions (its denominator)."""
    vocab: Dict[str, int] = {}
    for ws in words:
        for w, ct in ws:
            if w != ct:
                vocab.setdefault(w, len(vocab))
    M = np.zeros((len(words), max(1, len(vocab))))
    n = np.zeros(len(words))
    for i, ws in enumerate(words):
        n[i] = len(ws)
        for w, ct in ws:
            if w != ct:
                M[i, vocab[w]] += 1
    return M, n


def draws(pop: Dict[str, Any], expected: Dict[str, int], mode: str, rng) -> Dict[str, np.ndarray]:
    """Per condition, how often each population prompt is drawn in each simulation [N_SIM, n_pop]."""
    counts = {}
    for c in pop:
        n_pop = len(pop[c]["pids"])
        W = np.zeros((N_SIM, n_pop))
        for s in range(N_SIM):
            if mode == "point":
                idx = rng.integers(0, n_pop, expected[c])
            else:
                first = rng.integers(0, n_pop, n_pop)
                idx = first[rng.integers(0, n_pop, expected[c])]
            W[s] = np.bincount(idx, minlength=n_pop)
        counts[c] = W
    return counts


def apply_rules(pop: Dict[str, Any], bands, counts: Dict[str, np.ndarray], key: str) -> Dict[str, Any]:
    """The five rules on each simulation's drawn prompts, for R2's metric ("r2") or S6 ("s6", where a drawn prompt
    with no score drops out, so the counts vary from draw to draw). Returns per-draw arrays [N_SIM, bands]."""
    conds = list(pop)
    nb = len(bands)
    p4 = np.ones((N_SIM, nb))
    rules = {k: np.ones((N_SIM, nb), bool) for k in ("1", "2", "3", "5")}
    half5 = {b: np.ones((N_SIM, nb), bool) for b in ("jpp", "naive")}
    means = {b: np.zeros((N_SIM, nb)) for b in ("jpp", "naive")}
    n_valid = np.zeros(N_SIM)
    for bi, (band, _) in enumerate(bands):
        p_max = np.zeros(N_SIM)
        for b in ("jpp", "naive"):
            tot = np.zeros(N_SIM)
            var_pooled = np.zeros(N_SIM)
            npos = np.zeros(N_SIM)
            nneg = np.zeros(N_SIM)
            N = np.zeros(N_SIM)
            for c in conds:
                d = pop[c][band][key][f"d_{b}"]
                valid = ~np.isnan(d)
                dz = np.where(valid, d, 0.0)
                W = counts[c]
                n_c = W @ valid
                s1, s2 = W @ dz, W @ (dz * dz)
                with np.errstate(invalid="ignore", divide="ignore"):
                    m_c = s1 / n_c
                    v_c = np.maximum(s2 / n_c - m_c ** 2, 0.0)
                    se_c = np.sqrt(v_c / n_c)
                rules["2"][:, bi] &= (n_c > 0) & ~(m_c + Z * se_c < 0)     # no data in a condition fails rule 2
                tot += s1
                var_pooled += np.where(n_c > 0, n_c * v_c, 0.0)
                npos += W @ (dz > 1e-9)
                nneg += W @ (dz < -1e-9)
                N += n_c
            with np.errstate(invalid="ignore", divide="ignore"):
                m = tot / N
                se = np.sqrt(var_pooled) / N
            means[b][:, bi] = m
            n_valid = N
            rules["1"][:, bi] &= (m - Z * se) > 0
            half5[b][:, bi] = m >= C.MIN_EFFECT - 1e-12
            rules["5"][:, bi] &= half5[b][:, bi]
            p = np.array([p_sign(int(u), int(u + v)) for u, v in zip(npos, nneg)])
            p_max = np.maximum(p_max, p)
        p4[:, bi] = p_max
        for c in conds:
            M, npos_c = word_matrix(pop[c][band][key]["words"])
            W = counts[c]
            share = (W @ M).max(1) / np.maximum(W @ npos_c, 1)
            rules["3"][:, bi] &= share <= C.STUCK_MAX + 1e-12
    adj = np.zeros_like(p4)
    for s in range(N_SIM):
        h = holm({band: float(p4[s, bi]) for bi, (band, _) in enumerate(bands)})
        adj[s] = [h[band] for band, _ in bands]
    r4ok = adj < C.ALPHA
    passes = r4ok & rules["1"] & rules["2"] & rules["3"] & rules["5"]
    return {"rules": rules, "r4ok": r4ok, "half5": half5, "means": means, "passes": passes, "n_valid": n_valid}


def summary(x: Dict[str, Any], bands) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    rules, r4ok, half5, means, passes = x["rules"], x["r4ok"], x["half5"], x["means"], x["passes"]
    for bi, (band, _) in enumerate(bands):
        res[band] = {"page": R43.PAGE[band],
                     "rule_shares": {"1": A.r4(rules["1"][:, bi].mean()), "2": A.r4(rules["2"][:, bi].mean()),
                                     "3": A.r4(rules["3"][:, bi].mean()), "4": A.r4(r4ok[:, bi].mean()),
                                     "5": A.r4(rules["5"][:, bi].mean())},
                     "rule5_half": {"r1_minus_jpp": A.r4(half5["jpp"][:, bi].mean()),
                                    "r1_minus_naked_eye": A.r4(half5["naive"][:, bi].mean())},
                     "passes_all_five": A.r4(passes[:, bi].mean()),
                     "mean_of_pooled_diff": {"jpp": A.r4(np.nanmean(means["jpp"][:, bi])),
                                             "naive": A.r4(np.nanmean(means["naive"][:, bi]))},
                     "sd_of_pooled_diff": {"jpp": A.r4(np.nanstd(means["jpp"][:, bi])),
                                           "naive": A.r4(np.nanstd(means["naive"][:, bi]))}}
    res["both_bands_pass"] = A.r4(passes.all(1).mean())
    res["at_least_one_band_passes"] = A.r4(passes.any(1).mean())
    res["mean_valid_prompts"] = A.r4(float(np.mean(x["n_valid"])))
    return res


def simulate(pop: Dict[str, Any], bands, expected: Dict[str, int], mode: str, rng) -> Dict[str, Any]:
    """One mode's draws, with R2's rule, S6 and R4's decision (both) on the same drawn prompts."""
    counts = draws(pop, expected, mode, rng)
    r2 = apply_rules(pop, bands, counts, "r2")
    s6 = apply_rules(pop, bands, counts, "s6")
    served = r2["passes"] & s6["passes"]
    out = {"r2_rule": summary(r2, bands), "s6": summary(s6, bands), "served": {}}
    for bi, (band, _) in enumerate(bands):
        out["served"][band] = {"page": R43.PAGE[band], "served": A.r4(served[:, bi].mean())}
    out["served"]["both_bands"] = A.r4(served.all(1).mean())
    out["served"]["at_least_one_band"] = A.r4(served.any(1).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(C.POWER_JSON))
    a = ap.parse_args()
    D = R43.R4_DESIGN
    raw2, raw3 = R2.EVAL_JSON.read_bytes(), C3.EVAL_JSON.read_bytes()
    d2, d3 = R43.restrict(json.loads(raw2)), R43.restrict(json.loads(raw3))
    ref2 = json.loads(R2.ANALYSIS_JSON.read_text(encoding="utf-8"))["bands"]["r1"]
    ref3 = json.loads(C3.ANALYSIS_JSON.read_text(encoding="utf-8"))["reference_bands"]["r1"]
    checks = []
    for data, ref, name in ((d2, ref2, "R2"), (d3, ref3, "R3")):
        t = A.Table(D, data)
        boot = A.Boot(t, 1, 0)
        for band, layers in D.bands:
            for b in ("jpp", "naive"):
                st = A.sign_test(t, boot, "r1", b, list(layers), D.conditions)
                checks.append({"set": name, "band": band, "baseline": b, "recomputed": [st["n_pos"], st["n_neg"]],
                               "recorded": [ref[band][f"sign_{b}"]["n_pos"], ref[band][f"sign_{b}"]["n_neg"]]})
    ok = all(c["recomputed"] == c["recorded"] for c in checks)
    p2, p3 = population(D, d2, "R2"), population(D, d3, "R3")
    pop = merge(p2, p3, D.bands)
    sizes = {c: {"R2": len(p2[c]["pids"]), "R3": len(p3[c]["pids"]), "together": len(pop[c]["pids"])}
             for c in D.conditions}
    def weighted(p, band, key, b) -> Optional[float]:
        """The mean difference over a population's valid prompts, weighted by condition to R4's expected counts (in
        S6, each condition's expected count scaled by its share of prompts with a score)."""
        num = den = 0.0
        for c in D.conditions:
            d = p[c][band][key][f"d_{b}"]
            v = d[~np.isnan(d)]
            if len(v):
                w = R4_EXPECTED[c] * len(v) / len(d)
                num += float(v.mean()) * w
                den += w
        return A.r4(num / den) if den else None

    record_means = {}
    for band, _ in D.bands:
        record_means[band] = {}
        for key in ("r2", "s6"):
            for b in ("jpp", "naive"):
                record_means[band][f"{key}_{b}"] = {
                    "by_condition": {c: A.r4(float(np.nanmean(pop[c][band][key][f"d_{b}"]))) for c in D.conditions},
                    "together": weighted(pop, band, key, b), "r2_alone": weighted(p2, band, key, b),
                    "r3_alone": weighted(p3, band, key, b)}
    s6_valid = {c: int((~np.isnan(pop[c]["55"]["s6"]["d_naive"])).sum()) for c in D.conditions}
    rng = np.random.default_rng(SIM_SEED)
    point = simulate(pop, D.bands, R4_EXPECTED, "point", rng)
    pred = simulate(pop, D.bands, R4_EXPECTED, "predictive", rng)
    oc: Dict[str, Any] = {}
    for band, _ in D.bands:
        se640 = point["r2_rule"][band]["sd_of_pooled_diff"]["naive"]
        rows = {}
        for e in EFFECTS:
            rows[f"{e:+.2f}"] = {f"{n} prompts": A.r4(0.5 * math.erfc(-(e - C.MIN_EFFECT) / (se640 * math.sqrt(640 / n))
                                                                     / math.sqrt(2)))
                                 for n in (320, 640)}
        oc[band] = {"se_r1_minus_naked_at_640": se640, "p_estimate_reaches_0.05": rows}
    head = C.git_state([])["head"]
    out = {"schema": "R4_power/1", "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "head": head,
           "input": {"R2_02_eval.json": hashlib.sha256(raw2).hexdigest(),
                     "R3_05_eval.json": hashlib.sha256(raw3).hexdigest(),
                     "R2_03_analysis.json": C.sha256_file(R2.ANALYSIS_JSON),
                     "R3_06_analysis.json": C.sha256_file(C3.ANALYSIS_JSON)},
           "ok": ok, "check": "the sign counts recomputed here equal R2_03's bands.r1 and R3_06's reference_bands.r1",
           "sign_counts": checks, "population_sizes": sizes, "r4_expected_valid": R4_EXPECTED,
           "n_sim": N_SIM, "sim_seed": SIM_SEED,
           "intervals": "rules 1 and 2 by the normal approximation of the stratified prompt bootstrap (mean +/- 1.96 "
                        "SE); the registered rule takes 2,000 bootstrap resamples",
           "s6": {"why": R43.FW_WHY, "population_with_a_score": s6_valid,
                  "note": "drawn on the same prompts as R2's rule; a drawn prompt whose every target is a function word "
                          "has no S6 score and drops out of S6 in that draw"},
           "decision": R43.GATE_RULE,
           "record_means": record_means, "point": point, "predictive": pred, "operating_characteristic": oc}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "sizes": sizes, "s6_valid": s6_valid, "record_means": record_means}, indent=1))
    for mode, r in (("point", point), ("predictive", pred)):
        for key in ("r2_rule", "s6"):
            for band, _ in D.bands:
                x = r[key][band]
                print(f"{mode:10} {key:7} page {x['page']}: all five {x['passes_all_five']}, rules {x['rule_shares']}, "
                      f"rule-5 halves {x['rule5_half']}")
        print(f"{mode:10} SERVED (both): {r['served']}")
    print(json.dumps(oc, indent=1))
    print(f"R4_power: {'ok' if ok else 'SIGN COUNTS DIFFER FROM R2_03 / R3_06'}; wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
