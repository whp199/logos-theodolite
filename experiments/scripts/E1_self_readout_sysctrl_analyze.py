# E1_self_readout: analysis of identity-free system-prompt control (CPU).
import json, numpy as np
R = "./experiments/results"
Z = np.load(f"{R}/E1_self_readout_sysctrl_means.npz")
O = Z["OUT"].astype(np.float64)  # [q, cond, cls, lens, layer, metric]
conds = list(Z["conds"]); CL = list(Z["classes"]); LN = list(Z["lenses"]); MT = list(Z["metrics"])
main = np.load(f"{R}/E1_self_readout_passage_means.npz")
X = main["X"]; mconds = list(main["conds"]); mcl = list(main["classes"]); mmet = list(main["metrics"])
BANDS = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
rng = np.random.default_rng(3); BOOT = rng.integers(0, 40, (2000, 40))
beh = json.load(open(f"{R}/E1_self_readout_sysctrl_behavior.json"))
res = {"behavior_lse_fm": {c: float(np.mean(v)) for c, v in beh.items()}, "readout": {}}
for ln in ("jpp", "jlens"):
    li = LN.index(ln)
    for mt in ("pron", "noun"):
        mi = MT.index(mt)
        for cls in ("self", "header", "user_all"):
            ki = CL.index(cls)
            for b, r in BANDS.items():
                Y = O[:, :, ki, li][..., list(r), mi].mean(-1)  # [q, cond]
                # woman-man span from the main run (same passages)
                Xm = X[LN.index(ln), :, :, mcl.index(cls), :, mmet.index(mt)][..., list(r)].mean(-1)  # [cond, q]
                span = Xm[[mconds.index("woman_p1"), mconds.index("woman_p2")]].mean(0) - Xm[[mconds.index("man_p1"), mconds.index("man_p2")]].mean(0)
                row = {}
                for c in conds:
                    row[c] = float(Y[:, conds.index(c)].mean())
                for a, bb in [("default", "nameless"), ("sys_irrelevant", "nameless"), ("sys_style", "nameless"), ("default", "sys_irrelevant"), ("default", "sys_style")]:
                    d = Y[:, conds.index(a)] - Y[:, conds.index(bb)]
                    bd = d[BOOT].mean(1); bs = span[BOOT].mean(1)
                    row[f"{a}-{bb}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                        "pct_span": float(100 * d.mean() / span.mean()), "pct_span_ci": [float(np.percentile(100 * bd / bs, 2.5)), float(np.percentile(100 * bd / bs, 97.5))]}
                res["readout"][f"{ln}|{mt}|{cls}|{b}"] = row
json.dump(res, open(f"{R}/E1_self_readout_sysctrl_analysis.json", "w"), indent=1)
print("behavior lse_fm:", {k: round(v, 2) for k, v in res["behavior_lse_fm"].items()})
for key, row in res["readout"].items():
    if "|user_all|" in key:
        continue
    s = f"{key:24s} " + " ".join(f"{c}={row[c]:+.2f}" for c in conds)
    for k in ("default-nameless", "sys_irrelevant-nameless", "sys_style-nameless", "default-sys_irrelevant"):
        s += f" | {k} {row[k]['pct_span']:+.1f}%[{row[k]['pct_span_ci'][0]:+.1f},{row[k]['pct_span_ci'][1]:+.1f}]"
    print(s)
