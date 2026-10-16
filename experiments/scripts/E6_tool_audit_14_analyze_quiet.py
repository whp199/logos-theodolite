#!/usr/bin/env python3
"""E6_tool_audit_14_analyze_quiet -- offline: effect of clamping quiet residual dims on lens gender readouts (from E6_13)."""
import json
import numpy as np
R = "./experiments/results/"
d = json.load(open(R + "E6_tool_audit_13_quiet_readout.partial.json"))
L = d["layers"]; names = np.array(d["names"])
rng = np.random.default_rng(0)
BANDS = {"L30-39": (30, 40), "L40-49": (40, 50), "L50-58": (50, 59)}
out = {}
for kind in ("jpp", "jlens"):
    for ver in ("full", "clamped"):
        A = np.array(d[kind][ver])  # [N, nL]
        for b, (s, e) in BANDS.items():
            cols = [i for i, l in enumerate(L) if s <= l < e]
            v = A[:, cols].mean(1)
            for fam, (w, m, xs) in {"A": ("A_woman", "A_man", ["A_default", "A_bg"]), "B": ("B_woman", "B_man", ["B_gemma", "B_atlas", "B_nameless", "B_gemma_human"])}.items():
                W = v[names == w]; M = v[names == m]
                for x in xs:
                    X = v[names == x]
                    pos = []
                    for ii in rng.integers(0, 12, (2000, 12)):
                        gp = W[ii].mean() - M[ii].mean()
                        pos.append((X[ii].mean() - M[ii].mean()) / gp)
                    out.setdefault(kind, {}).setdefault(ver, {}).setdefault(x, {})[b] = [float((X.mean() - M.mean()) / (W.mean() - M.mean())), float(np.percentile(pos, 2.5)), float(np.percentile(pos, 97.5)), float(X.mean()), float(W.mean() - M.mean())]
        # correlation full vs clamped per prompt-band
    F_ = np.array(d[kind]["full"]); C_ = np.array(d[kind]["clamped"])
    out.setdefault(kind, {})["corr_full_vs_clamped_per_layer"] = [float(np.corrcoef(F_[:, i], C_[:, i])[0, 1]) for i in range(len(L))]
    out[kind]["mean_abs_shift_per_layer"] = [float(np.abs(F_[:, i] - C_[:, i]).mean()) for i in range(len(L))]
out["share"] = d["share"]
json.dump(out, open(R + "E6_tool_audit_14_quiet_summary.json", "w"), indent=1)
for kind in ("jpp", "jlens"):
    print(kind, "corr full vs clamped per layer:", [round(x, 3) for x in out[kind]["corr_full_vs_clamped_per_layer"]])
    print(kind, "mean |shift| per layer:", [round(x, 2) for x in out[kind]["mean_abs_shift_per_layer"]])
    for x in out[kind]["full"]:
        print(kind.ljust(5), x.ljust(14), "  ".join(f"{b}: full {out[kind]['full'][x][b][0]:+.2f} (raw {out[kind]['full'][x][b][3]:+.2f}, gap {out[kind]['full'][x][b][4]:.1f}) / clamped {out[kind]['clamped'][x][b][0]:+.2f} (raw {out[kind]['clamped'][x][b][3]:+.2f}, gap {out[kind]['clamped'][x][b][4]:.1f})" for b in BANDS))
