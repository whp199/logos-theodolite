# V3_persona_axis_traits_03c (offline): paired deltas (steered - unsteered) per prompt with bootstrap CIs.
import json, numpy as np
R = {}
for t in ("b1", "b2"):
    d = json.load(open(f"./experiments/results/V3_persona_axis_traits_03b_steer_det_{t}.json"))
    R.update(d["res"])
base = R["unsteered"]["rows"]
MET = {"P3": "lse", "P3b": "lse", "P2n": "lse", "P4": "wm", "P5p": "wm", "P5": "lr"}
rng = np.random.default_rng(0)
out = {}
print(f"{'setting':26s}" + "".join(f"{s+':'+m:>22s}" for s, m in MET.items()) + f"{'P5 q':>8s}{'KL':>7s}")
for key in ["trait|-0.04", "trait|-0.02", "trait|+0.02", "trait|+0.04", "persona|-0.04", "persona|-0.02", "persona|+0.02", "persona|+0.04",
            "trait_perp_persona|-0.04", "trait_perp_persona|-0.02", "trait_perp_persona|+0.02", "trait_perp_persona|+0.04",
            "role|-0.04", "role|-0.02", "role|+0.02", "role|+0.04", "ctrl1|-0.04", "ctrl1|-0.02", "ctrl1|+0.02", "ctrl1|+0.04"]:
    rows = R[key]["rows"]; o = {}
    line = f"{key:26s}"
    for s, m in MET.items():
        dd = np.array([r[m] - b[m] for r, b in zip(rows, base) if r["set"] == s])
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        o[s] = [float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        line += f"{dd.mean():+7.1f}[{o[s][1]:+5.1f},{o[s][2]:+5.1f}]"
    q = R[key]["summary"]["P5_q"]; kl = R[key]["summary"]["NEU_kl"]
    o["P5_q"] = q; o["KL"] = kl
    out[key] = o
    print(line + f"{q:8.2f}{kl:7.2f}")
print("unsteered P5 q", R["unsteered"]["summary"]["P5_q"], "P3", R["unsteered"]["summary"]["P3_lse"])
json.dump(out, open("./experiments/results/V3_persona_axis_traits_03c_steer_deltas.json", "w"), indent=1)
