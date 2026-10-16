#!/usr/bin/env python3
"""E6_tool_audit_09_analyze_lenses -- offline: lens comparison (J++ vs JLens vs naive) of the last-position gender readout
for all E6 conditions (single BOS). Bootstrap over prompts; woman/man scale positions per lens."""
import json
import numpy as np

R = "./experiments/results/"
d = json.load(open(R + "E6_tool_audit_08_lenses.partial.json"))
BANDS = {"L10-19": (10, 20), "L20-29": (20, 30), "L30-39": (30, 40), "L40-49": (40, 50), "L50-58": (50, 59), "L59": (59, 60)}
rng = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def band(arr, b):
    s, e = BANDS[b]
    return np.asarray(arr, float)[:, s:e].mean(1)  # per prompt


out = {"means": {}, "scale": {}}
for c in d:
    for lens in ("jpp", "jlens", "naive"):
        for k in ("pron", "noun", "neuter_vs_human", "they_vs_human"):
            for b in BANDS:
                out["means"].setdefault(c, {}).setdefault(lens, {}).setdefault(k, {})[b] = boot(band(d[c][lens][k], b))
fams = {"A": ("A_woman", "A_man", ["A_default", "A_bg", "A_raw"]),
        "B": ("B_woman", "B_man", ["B_gemma", "B_atlas", "B_nameless", "B_gemma_human"])}
for fam, (w, m, xs) in fams.items():
    for lens in ("jpp", "jlens", "naive"):
        for k in ("pron", "noun"):
            for x in xs:
                for b in BANDS:
                    W, M, X = band(d[w][lens][k], b), band(d[m][lens][k], b), band(d[x][lens][k], b)
                    pos = []
                    for ii in rng.integers(0, len(X), (2000, len(X))):
                        g = W[ii].mean() - M[ii].mean()
                        pos.append((X[ii].mean() - M[ii].mean()) / g if abs(g) > 1e-9 else np.nan)
                    out["scale"].setdefault(lens, {}).setdefault(k, {}).setdefault(x, {})[b] = [
                        float((X.mean() - M.mean()) / (W.mean() - M.mean())), float(np.nanpercentile(pos, 2.5)),
                        float(np.nanpercentile(pos, 97.5)), float(W.mean() - M.mean())]
json.dump(out, open(R + "E6_tool_audit_09_lenses_summary.json", "w"), indent=1)


def f(v):
    return f"{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]"


for k in ("pron", "neuter_vs_human"):
    print(f"\n## {k} by lens (mean [95% CI] over prompts)")
    for c in d:
        for lens in ("jpp", "jlens", "naive"):
            print(c.ljust(14), lens.ljust(6), "  ".join(f"{b}:{f(out['means'][c][lens][k][b])}" for b in BANDS))
print("\n## woman/man scale (pron)")
for lens in ("jpp", "jlens", "naive"):
    for x, bd in out["scale"][lens]["pron"].items():
        print(lens.ljust(6), x.ljust(14), "  ".join(f"{b}:{f(v)}(gap {v[3]:+.1f})" for b, v in bd.items() if b != "L10-19"))
