#!/usr/bin/env python3
"""Which subsets of the calibration doses best tell neurons from controls in the census's own ranking.

For every layer's calib_rel and calib_ceil run (atlas_data/census/runs/L<layer>-<design>.jsonl), the run is re-ranked
by backend/census.compute_units (the statistic census_tools.py rank writes, unchanged) on the answers of a chosen subset
of its doses, the unpushed answers always kept (compute_units leaves out every word they use). Per subset, the census's
own figures: the neurons' and the controls' median null_pct, how many neurons beat every control (null_pct 1.0), the
least q. Beside them, summaries over the same masses (not a new statistic):
- auc: P(a neuron's mass > a control's), ties half, with its one-sided Mann-Whitney p. A unit's mass does not depend on
  which units are labelled controls (compute_units' word mean is over every unit), so this is the exact label-permutation
  test of the neurons' lead, and it is not fooled by the mass-0 ties that pull both medians of null_pct to 0 on sparse
  or wrecked subsets (null_pct counts a tie as a loss);
- p_top: the chance of at least that many neurons above every control were the units exchangeable,
  prod_{i<k} (N - i) / (N + C - i);
- auc_words, auc_resid: a covariate check. words/answer is a unit's mean count of new words (not in the unpushed
  answers) per counted answer. auc_words is the neurons' lead on it alone; auc_resid the lead of mass once its rank is
  regressed (OLS) on the rank of words/answer. Where auc_words >= auc and auc_resid ~ 0.5, the lead is that the neurons'
  text keeps more words under the push than the controls', not that their words recur more than wordiness predicts.
Rank correlations (Spearman) of the neurons' null_pct between subsets, and between the two designs on the same subset
(the same 64 neurons and the same control seeds, independent answers), each beside the controls' (of mass).

max_rel: rank's default (1.6 of the stream) drops answers pushed harder; a subset is ranked on its own answers, so
subsets are ranked with max_rel = inf, and the full set both ways (at 1.6 it reproduces summary.json).

    CUDA_VISIBLE_DEVICES= /home/eric/bin/ai/theodolite2/.venv/bin/python -I subset_rank.py [--out subset_ranking.json]
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/eric/bin/ai/theodolite2")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from backend import census as cz  # noqa: E402

RUNS = ROOT / "atlas_data" / "census" / "runs"
SUMMARY = ROOT / "experiments" / "results" / "atlas_calibration" / "summary.json"
HERE = Path(__file__).resolve().parent
LAYERS = (20, 30, 40, 45, 50, 57)
DESIGNS = ("calib_rel", "calib_ceil")
INF = math.inf
FULL16 = "full (rank default, max_rel 1.6)"
TYPES = ("full", "+ only", "- only", "under break", "at/above break", "smallest 2", "smallest 3")


def subset_rows(full: cz.SlimRows, doses) -> cz.SlimRows:
    """The run's unpushed answers and its answers at `doses`, as SlimRows sharing the full run's vocabulary."""
    doses = set(doses)
    s = cz.SlimRows()
    s.vocab, s._ids = full.vocab, full._ids
    s.extend(r for r in full if r["kind"] == "baseline" or r["dose"] in doses)
    return s


def p_top(k: int, n: int, c: int) -> float:
    """P(at least k neurons above every control) were the n + c units exchangeable."""
    out = 1.0
    for i in range(k):
        out *= (n - i) / (n + c - i)
    return out


def auc_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(np.concatenate([a, b]) == a[0]):
        return 0.5, 1.0
    mw = stats.mannwhitneyu(a, b, alternative="greater")
    return float(mw.statistic / (len(a) * len(b))), float(mw.pvalue)


def measure(rows: cz.SlimRows, max_rel: float) -> dict:
    recs = cz.compute_units(rows, max_rel=max_rel)  # the census's own ranking, unchanged
    neu = [u for u in recs if u["kind"] == "neuron"]
    ctl = [u for u in recs if u["kind"] == "random"]
    nm = np.array([u["mass"] for u in neu])
    cm = np.array([u["mass"] for u in ctl])
    auc, p = auc_p(nm, cm)
    k = sum(u["null_pct"] == 1.0 for u in neu)
    # the covariate: new words per counted answer, over the answers compute_units counts
    base = set().union(*(r["wid"] for r in rows if r["kind"] == "baseline"))
    nw = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline" and r["rel_push"] <= max_rel:
            nw[(r["kind"], int(r["neuron"]))].append(sum(1 for i in r["wid"] if i not in base))
    units = [(u["kind"], int(u["neuron"])) for u in neu + ctl]
    x = np.array([np.mean(nw[u]) if nw[u] else 0.0 for u in units])
    y = np.concatenate([nm, cm])
    is_n = np.array([True] * len(neu) + [False] * len(ctl))
    auc_w, p_w = auc_p(x[is_n], x[~is_n])
    if np.std(x) > 0 and np.std(y) > 0:
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        res = ry - np.polyval(np.polyfit(rx, ry, 1), rx)
        auc_r, p_r = auc_p(res[is_n], res[~is_n])
        rho_w = float(stats.spearmanr(x, y).statistic)
    else:
        auc_r, p_r, rho_w = float("nan"), float("nan"), float("nan")
    per_unit = sum(len(v) for v in nw.values()) / max(len(units), 1)
    return {
        "neurons_median_null_pct": round(float(np.median([u["null_pct"] for u in neu])), 4),
        "controls_median_null_pct": round(float(np.median([u["null_pct"] for u in ctl])), 4),
        "neurons_mean_null_pct": round(float(np.mean([u["null_pct"] for u in neu])), 4),
        "neurons_beating_every_control": int(k),
        "p_top_exchangeable": float(p_top(k, len(neu), len(ctl))),
        "min_q": float(min(u["q"] for u in neu)),
        "neurons_q_le_0.1": int(sum(u["q"] <= 0.1 for u in neu)),
        "neurons_q_le_0.25": int(sum(u["q"] <= 0.25 for u in neu)),
        "auc": round(auc, 4), "mwu_p": p,
        "auc_words": round(auc_w, 4), "mwu_p_words": p_w,
        "auc_resid": round(auc_r, 4), "mwu_p_resid": p_r, "rho_mass_words": round(rho_w, 3),
        "zero_mass": {"neuron": int((nm == 0).sum()), "random": int((cm == 0).sum())},
        "answers_counted_per_unit": round(per_unit, 2),
        "n_neurons": len(neu), "n_controls": len(ctl),
        "top_neurons": sorted(int(u["neuron"]) for u in neu if u["null_pct"] == 1.0),
        "_null_pct": {int(u["neuron"]): u["null_pct"] for u in neu},
        "_mass": {int(u["neuron"]): u["mass"] for u in neu},
        "_ctl_mass": {int(u["neuron"]): u["mass"] for u in ctl},
    }


def public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def breaks_of(summary: dict, layer: int, design: str) -> dict:
    """Per sign, the smaller |dose| at which the neurons' or the controls' text breaks (summary.json: the census's
    dose_window, distinct 4-grams more than 0.1 below the unpushed answers')."""
    b = summary["layers"][str(layer)][design]["breaks"]
    out = {}
    for sign in ("+", "-"):
        vals = [abs(v) for v in b.get(sign, {}).values() if v is not None]
        out[sign] = min(vals) if vals else None
    return out


def subsets_for(doses: list, brk: dict) -> dict:
    """{name: (doses, max_rel)} in report order."""
    mags = sorted({abs(d) for d in doses})
    pos = [d for d in doses if d > 0]
    neg = [d for d in doses if d < 0]

    def under(d):
        b = brk["+" if d > 0 else "-"]
        return b is None or abs(d) < b

    def both(ms):
        return [d for d in doses if abs(d) in ms]

    S = {FULL16: (doses, 1.6), "full": (doses, INF)}
    for m in mags:
        S[f"+-{m:g}"] = (both({m}), INF)
    S["+ only"] = (pos, INF)
    S["- only"] = (neg, INF)
    S["under break"] = ([d for d in doses if under(d)], INF)
    S["at/above break"] = ([d for d in doses if not under(d)], INF)
    S["smallest 2"] = (both(set(mags[:2])), INF)
    S["smallest 3"] = (both(set(mags[:3])), INF)
    return S


def spearman(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def stouffer(ps) -> tuple:
    ps = [min(max(p, 1e-300), 1 - 1e-16) for p in ps if p == p]
    z = float(np.sum(stats.norm.isf(ps)) / math.sqrt(len(ps)))
    return round(z, 2), float(stats.norm.sf(z))


def ds_str(ds) -> str:
    return ",".join(f"{d:+g}" for d in sorted(ds, key=lambda d: (abs(d), d)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "subset_ranking.json"))
    ap.add_argument("--table", default=str(HERE / "subset_table.json"))
    a = ap.parse_args(argv)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report = {"what": __doc__.split("\n\n")[0], "runs": str(RUNS), "layers": {}, "checks": {}, "across_layers": {}}
    table = []
    res_by = {}
    for layer in LAYERS:
        report["layers"][str(layer)] = {}
        for design in DESIGNS:
            run = RUNS / f"L{layer}-{design}.jsonl"
            full = cz.read_slim(run)
            doses = sorted({r["dose"] for r in full if r["kind"] != "baseline"})
            brk = breaks_of(summary, layer, design)
            res = {}
            for name, (ds, mr) in subsets_for(doses, brk).items():
                m = measure(subset_rows(full, ds), mr)
                m["doses"], m["max_rel"] = [float(d) for d in ds], ("inf" if mr == INF else mr)
                m["halves"] = {}
                if mr == INF and any(d > 0 for d in ds) and any(d < 0 for d in ds):
                    h = {s: measure(subset_rows(full, [d for d in ds if (d > 0) == (s == "+")]), mr) for s in "+-"}
                    m["halves_rank_corr"] = round(spearman(h["+"]["_null_pct"], h["-"]["_null_pct"]), 3)
                    m["halves"] = {s: public(v) for s, v in h.items()}
                if mr == INF and any(r["rel_push"] > 1.6 for r in full if r["kind"] != "baseline" and r["dose"] in set(ds)):
                    m["at_max_rel_1.6"] = public(measure(subset_rows(full, ds), 1.6))
                res[name] = m
            signed = {f"{d:+g}": public(measure(subset_rows(full, [d]), INF)) for d in doses}
            names = [n for n in res if n != FULL16]
            corr = {f"{x} | {y}": {"neurons_null_pct": round(spearman(res[x]["_null_pct"], res[y]["_null_pct"]), 3),
                                   "neurons_mass": round(spearman(res[x]["_mass"], res[y]["_mass"]), 3),
                                   "controls_mass": round(spearman(res[x]["_ctl_mass"], res[y]["_ctl_mass"]), 3)}
                    for x, y in itertools.combinations(names, 2)}
            res_by[(layer, design)] = res
            s = summary["layers"][str(layer)][design]["ranking"]
            m = res[FULL16]
            ok = (abs(m["neurons_median_null_pct"] - s["neurons_median_null_pct"]) < 5e-4
                  and abs(m["controls_median_null_pct"] - s["controls_median_null_pct"]) < 5e-4
                  and m["neurons_beating_every_control"] == s["neurons_beating_every_control"]
                  and abs(m["min_q"] - s["min_q"]) < 5e-4)
            report["checks"][f"L{layer}-{design}"] = {"reproduces_summary": ok, "summary": s, "here": {k: m[k] for k in s}}
            report["layers"][str(layer)][design] = {"breaks_used": brk, "subsets": {n: public(v) for n, v in res.items()},
                                                    "signed_single": signed, "rank_corr_between_subsets": corr}
            print(f"\nL{layer} {design}  breaks used +{brk['+']:g} -{brk['-']:g}  (full at max_rel 1.6 reproduces summary.json: {ok})")
            print(f"  {'subset':16s} {'doses':40s} {'a/u':>4} {'neuMd':>5} {'ctlMd':>5} {'top':>3} {'minq':>5} {'AUC':>5} {'p':>7} "
                  f"{'AUCw':>5} {'AUCr':>5} {'p_r':>7} | {'+ AUC/p':>12} | {'- AUC/p':>12} | r+-")
            for n, v in res.items():
                hs = [f"{v['halves'][s]['auc']:.2f}/{v['halves'][s]['mwu_p']:.0e}" if v["halves"] else "" for s in "+-"]
                label = "full@1.6" if n == FULL16 else n
                print(f"  {label:16s} {ds_str(v['doses'])[:40]:40s} {v['answers_counted_per_unit']:4.1f} {v['neurons_median_null_pct']:5.3f} "
                      f"{v['controls_median_null_pct']:5.3f} {v['neurons_beating_every_control']:3d} {v['min_q']:5.3f} {v['auc']:5.3f} "
                      f"{v['mwu_p']:7.1e} {v['auc_words']:5.3f} {v['auc_resid']:5.3f} {v['mwu_p_resid']:7.1e} | {hs[0]:>12} | {hs[1]:>12} | "
                      f"{v.get('halves_rank_corr', float('nan')):+.2f}")
                # compact: a/u answers counted a unit; AUC (MWU p); w = AUC on words/answer alone; r = AUC of mass beyond
                # words/answer (p); 0 = units at mass 0, neurons/controls; +/- = each sign's half AUC (p); rho = their
                # neurons' null_pct rank correlation
                note = (f"a/u {v['answers_counted_per_unit']:g}; AUC {v['auc']:.3f} (p {v['mwu_p']:.1g}); w {v['auc_words']:.2f}; "
                        f"r {v['auc_resid']:.3f} (p {v['mwu_p_resid']:.1g}); 0 {v['zero_mass']['neuron']}/{v['zero_mass']['random']}")
                if v["halves"]:
                    hp, hm = v["halves"]["+"], v["halves"]["-"]
                    note += (f"; + {hp['auc']:.2f} (p {hp['mwu_p']:.1g}), - {hm['auc']:.2f} (p {hm['mwu_p']:.1g}), "
                             f"rho {v['halves_rank_corr']:+.2f}")
                if n == FULL16:
                    note = f"rank default; = summary.json: {ok}; " + note
                elif "at_max_rel_1.6" in v:
                    w = v["at_max_rel_1.6"]
                    note += (f"; at max_rel 1.6: a/u {w['answers_counted_per_unit']:g}, med {w['neurons_median_null_pct']:.3f}, "
                             f"top {w['neurons_beating_every_control']}, q {w['min_q']:.3f}")
                if n == "under break" or n == "at/above break":
                    note = f"break +{brk['+']:g}/-{brk['-']:g}; " + note
                table.append({"layer": layer, "design": design, "doses": f"{label}: {ds_str(v['doses'])}",
                              "neurons_median_null_pct": v["neurons_median_null_pct"],
                              "controls_median_null_pct": v["controls_median_null_pct"],
                              "neurons_beating_every_control": v["neurons_beating_every_control"],
                              "min_q": round(v["min_q"], 4), "note": note})
            print("  signed doses AUC/p: " + "  ".join(f"{k}:{v['auc']:.2f}/{v['mwu_p']:.0e}" for k, v in signed.items()))
        cross = {}
        for x in TYPES:
            r1, r2 = res_by[(layer, "calib_rel")][x], res_by[(layer, "calib_ceil")][x]
            exp = len(r1["top_neurons"]) * len(r2["top_neurons"]) / 64
            cross[x] = {"neurons_null_pct": round(spearman(r1["_null_pct"], r2["_null_pct"]), 3),
                        "neurons_mass": round(spearman(r1["_mass"], r2["_mass"]), 3),
                        "controls_mass": round(spearman(r1["_ctl_mass"], r2["_ctl_mass"]), 3),
                        "top_overlap": sorted(set(r1["top_neurons"]) & set(r2["top_neurons"])), "overlap_expected": round(exp, 2)}
        report["layers"][str(layer)]["rel_vs_ceil_same_subset"] = cross
    # across layers: each subset type and each single magnitude, per design
    print("\nacross layers (AUC per layer 20,30,40,45,50,57; Stouffer z over the six one-sided p; layers p<0.05)")
    for design in DESIGNS:
        names = [n for n in res_by[(20, design)] if n in res_by[(57, design)]]
        out = {}
        for n in names:
            aucs = [res_by[(L, design)][n]["auc"] for L in LAYERS]
            z, pz = stouffer([res_by[(L, design)][n]["mwu_p"] for L in LAYERS])
            zr, pr = stouffer([res_by[(L, design)][n]["mwu_p_resid"] for L in LAYERS])
            hs = {s: [res_by[(L, design)][n]["halves"][s]["auc"] if res_by[(L, design)][n]["halves"] else None for L in LAYERS] for s in "+-"}
            out[n] = {"auc": aucs, "stouffer_z": z, "stouffer_p": pz,
                      "layers_p_lt_0.05": sum(res_by[(L, design)][n]["mwu_p"] < 0.05 for L in LAYERS),
                      "auc_resid": [res_by[(L, design)][n]["auc_resid"] for L in LAYERS], "stouffer_z_resid": zr,
                      "layers_resid_p_lt_0.05": sum(res_by[(L, design)][n]["mwu_p_resid"] < 0.05 for L in LAYERS),
                      "plus_half_auc": hs["+"], "minus_half_auc": hs["-"],
                      "neurons_median_null_pct": [res_by[(L, design)][n]["neurons_median_null_pct"] for L in LAYERS],
                      "neurons_beating_every_control": [res_by[(L, design)][n]["neurons_beating_every_control"] for L in LAYERS]}
            o = out[n]
            print(f"  {design:10s} {n:34s} AUC " + " ".join(f"{x:.2f}" for x in aucs) + f"  z {z:+5.2f} ({o['layers_p_lt_0.05']}/6)"
                  f"  resid " + " ".join(f"{x:.2f}" for x in o["auc_resid"]) + f"  z {zr:+5.2f} ({o['layers_resid_p_lt_0.05']}/6)")
        report["across_layers"][design] = out
    print("\nrel vs ceil, same neurons, same subset: Spearman of neuron null_pct (controls' mass) [top overlap / expected]")
    for L in LAYERS:
        c = report["layers"][str(L)]["rel_vs_ceil_same_subset"]
        print(f"  L{L}: " + "  ".join(f"{k} {v['neurons_null_pct']:+.2f}({v['controls_mass']:+.2f})[{len(v['top_overlap'])}/{v['overlap_expected']:.1f}]"
                                       for k, v in c.items()))
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    Path(a.table).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}\n-> {a.table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
