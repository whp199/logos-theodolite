#!/usr/bin/env python3
"""E7_binding_07: robustness of binding indices to system-prompt order (self-first k=0,1 vs user-first k=2,3)."""
import json
import numpy as np
RES = "./experiments/results"
Z = [np.load(f"{RES}/E7_binding_factorial_chunk{c}.npz") for c in range(3)]
TYPES = list(Z[0]["types"]); READ = list(Z[0]["readouts"]); CELLN = list(Z[0]["cell_names"])
SC = np.concatenate([z["SC"] for z in Z]); BEH = np.concatenate([z["BEH"] for z in Z])
PID = np.concatenate([z["pid"] for z in Z]); CELL = np.concatenate([z["cell"] for z in Z]); K = np.concatenate([z["k"] for z in Z])
P = sorted(set(PID.tolist()))
rng = np.random.default_rng(5)
GS = {"W": ["WW", "WM", "WU"], "M": ["MW", "MM", "MU"]}; GU = {"W": ["WW", "MW", "UW"], "M": ["WM", "MM", "UM"]}
BANDS = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
out = {}
def per_passage(vals_fn, ks):
    a, b = [], []
    for p in P:
        m = lambda cells: np.mean([vals_fn((PID == p) & (CELL == c) & np.isin(K, ks)) for c in cells])
        a.append((m(GS["W"]) - m(GS["M"])) / 2); b.append((m(GU["W"]) - m(GU["M"])) / 2)
    return np.array(a), np.array(b)
for order, ks in (("self_first", [0, 1]), ("user_first", [2, 3])):
    for t, ref in (("A_self", 1), ("A_user", -1), ("F_I", 1), ("F_Y", -1), ("A_neut", 1)):
        ti = TYPES.index(t); ri = READ.index("jpp_pron")
        for bn, br in BANDS.items():
            a, b = per_passage(lambda m: SC[m][:, ti, list(br), ri].mean(), ks)
            idx = rng.integers(0, len(P), (2000, len(P)))
            A, B = a[idx].mean(1), b[idx].mean(1)
            bi = ref * (A - B) / (np.abs(A) + np.abs(B))
            out[f"{order}|jpp_pron|{t}|{bn}"] = {"a": float(a.mean()), "b": float(b.mean()), "BI": float(ref * (a.mean() - b.mean()) / (abs(a.mean()) + abs(b.mean()))),
                                                  "BI_ci": [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]}
    for k2, t, ref in ((0, "beh_I", 1), (1, "beh_Y", -1)):
        a, b = per_passage(lambda m: BEH[m][:, k2, 2].mean(), ks)
        out[f"{order}|{t}"] = {"a": float(a.mean()), "b": float(b.mean()), "BI": float(ref * (a.mean() - b.mean()) / (abs(a.mean()) + abs(b.mean())))}
json.dump(out, open(f"{RES}/E7_binding_order_check.json", "w"), indent=1)
for k, v in out.items():
    print(k, {kk: (round(vv, 2) if isinstance(vv, float) else [round(x, 2) for x in vv]) for kk, vv in v.items()})
