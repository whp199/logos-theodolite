#!/usr/bin/env python3
"""E7_binding_09: paired bootstrap (over passages) for the explaining-away test (E7_binding_05f)."""
import json, glob
import numpy as np
RES = "./experiments/results"
R = []
for f in sorted(glob.glob(f"{RES}/E7_binding_explainaway_chunk*.json")):
    R += json.load(open(f))["rows"]
pids = sorted(set(r["pid"] for r in R))
rng = np.random.default_rng(9)
def v(pid, S, tgt, it, suf):
    rr = [r["lse4"] for r in R if r["pid"] == pid and r["tgt"] == tgt and r["suf"] == suf and
          (r["int"] == "none" if it == "none" else (r["int"] == it and r["set"] == S))]
    return np.mean(rr)
MEAS = {
  "A_self: leak of self gender into 'you are a' (user unspecified) = (Y[WU]-Y[MU])/2": ("A_self", "WU", "MU", "Y"),
  "A_self: self gender in 'I am a' (user unspecified) = (I[WU]-I[MU])/2": ("A_self", "WU", "MU", "I"),
  "A_self: (Y[WM]-Y[MW])/2 (user-gender effect minus self leak)": ("A_self", "MW", "WM", "Y"),
  "A_user: leak of user gender into 'I am a' (self unspecified) = (I[UW]-I[UM])/2": ("A_user", "UW", "UM", "I"),
  "A_user: user gender in 'you are a' (self unspecified) = (Y[UW]-Y[UM])/2": ("A_user", "UW", "UM", "Y"),
  "A_user: (I[WM]-I[MW])/2 (self effect minus user leak)": ("A_user", "WM", "MW", "I"),
}
out = {}
for name, (S, c1, c2, suf) in MEAS.items():
    res = {}
    base = np.array([(v(p, S, c1, "none", suf) - v(p, S, c2, "none", suf)) / 2 for p in pids])
    for it in ("none", "ref", "rand", "full"):
        x = np.array([(v(p, S, c1, it, suf) - v(p, S, c2, it, suf)) / 2 for p in pids])
        d = x - base
        idx = rng.integers(0, len(pids), (2000, len(pids)))
        res[it] = {"mean": float(x.mean()), "delta_vs_clean": float(d.mean()),
                   "delta_ci": [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))]}
    out[name] = res
    print(name)
    for it, r in res.items():
        print(f"   {it:5s} value={r['mean']:+.3f}  delta vs clean={r['delta_vs_clean']:+.3f} [{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}]")
json.dump(out, open(f"{RES}/E7_binding_explainaway_stats.json", "w"), indent=1)
