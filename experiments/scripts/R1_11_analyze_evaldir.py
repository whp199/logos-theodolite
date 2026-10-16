#!/usr/bin/env python3
# R1_IT_LENS 11 (local): held-out directional fidelity (from R1_06). Pearson r over positions (pooled within class) between each
# lens readout at layer l and the model's real final readout t_d = <h59,d>/rms(h59), for 128 directions.
# CIs: bootstrap over passages (500 resamples, within class). Group summaries average r over the group's directions.
import json
import numpy as np
import torch

R = "./experiments/results/"
BANDS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]
BN = [f"L{a}-{b}" for a, b in BANDS]
d = torch.load(R + "R1_06_evaldir.pt", weights_only=False)
RD = d["readouts"]
names = torch.load(R + "R1_02_dirs.pt", weights_only=False)["names"]
K = len(names)
GROUPS = {"pron": [0], "noun": [1], "gender4": [0, 1, 2, 3], "ctrl6": list(range(4, 10)), "iso4": list(range(10, 14)),
          "unit16": list(range(14, 30)), "tokpair98": list(range(30, K))}
stats = d["stats"]
rng = np.random.default_rng(1)


def corr(st_list, r):
    n = sum(s["n"] for s in st_list)
    sy = sum(s["sy"] for s in st_list).numpy(); syy = sum(s["syy"] for s in st_list).numpy()
    X = sum(s[r] for s in st_list).numpy()           # [3,60,K]
    sx, sxx, sxy = X[0], X[1], X[2]
    cov = n * sxy - sx * sy[None]
    vx = n * sxx - sx ** 2; vy = n * syy - sy ** 2
    return cov / np.sqrt(np.clip(vx, 1e-300, None) * np.clip(vy[None], 1e-300, None))   # [60,K]


def bandg(x, idx):  # x [60,K] -> [6]
    return np.array([np.nanmean(x[a:b + 1][:, idx]) for a, b in BANDS])


out = {"bands": BN, "readouts": RD}
for cls in ("prose", "asst", "self", "user"):
    keys = [k for k in stats if k.endswith("|" + cls)]
    sts = [stats[k] for k in keys]
    rr = {r: corr(sts, r) for r in RD}
    B = 500
    boots = {r: [] for r in RD}
    for b in range(B):
        sel = rng.integers(0, len(sts), len(sts))
        sl = [sts[i] for i in sel]
        for r in RD:
            boots[r].append(corr(sl, r))
    res = {"n_passages": len(sts), "n_positions": int(sum(s["n"] for s in sts))}
    for g, idx in GROUPS.items():
        for r in RD:
            pt = bandg(rr[r], idx)
            bs = np.stack([bandg(x, idx) for x in boots[r]])
            res.setdefault(g, {})[r] = {"mean": pt.round(3).tolist(), "ci": np.nanpercentile(bs, [2.5, 97.5], 0).T.round(3).tolist()}
        for a, bname in (("it_mix", "jpp_norm"), ("it_mix", "jpp_lin"), ("it_mix", "naive_norm"), ("it_chat", "it_prose"),
                         ("jpp_norm", "naive_norm"), ("it_mix", "jl_norm")):
            pt = bandg(rr[a], idx) - bandg(rr[bname], idx)
            bs = np.stack([bandg(x, idx) - bandg(y, idx) for x, y in zip(boots[a], boots[bname])])
            res[g][f"{a}-{bname}"] = {"mean": pt.round(3).tolist(), "ci": np.nanpercentile(bs, [2.5, 97.5], 0).T.round(3).tolist()}
    # readout-space agreement between lenses (Pearson r across positions), full and loud-coordinate-only parts
    def pair_r(st_list, a, b):
        n = sum(x["n"] for x in st_list)
        A = sum(x[a] for x in st_list).numpy(); Bm = sum(x[b] for x in st_list).numpy()
        sab = sum(x[f"pair:{a}:{b}"] for x in st_list).numpy()
        cov = n * sab - A[0] * Bm[0]
        return cov / np.sqrt(np.clip(n * A[1] - A[0] ** 2, 1e-300, None) * np.clip(n * Bm[1] - Bm[0] ** 2, 1e-300, None))
    for a, b in d.get("pairs", []):
        pr = pair_r(sts, a, b)
        bs = np.stack([bandg(pair_r([sts[i] for i in rng.integers(0, len(sts), len(sts))], a, b), GROUPS["tokpair98"]) for _ in range(200)])
        res.setdefault("pair_r", {})[f"{a}~{b}"] = {g: bandg(pr, idx).round(3).tolist() for g, idx in GROUPS.items()}
        res["pair_r"][f"{a}~{b}"]["tokpair98_ci"] = np.nanpercentile(bs, [2.5, 97.5], 0).T.round(3).tolist()
    out[cls] = res
    print(f"== {cls}: {res['n_passages']} passages, {res['n_positions']} positions")
    for g in ("pron", "gender4", "tokpair98"):
        for r in ("it_mix", "it_chat", "it_prose", "jpp_norm", "jpp_lin", "jl_norm", "naive_norm", "it_mix_loud", "jpp_lin_loud"):
            print(f"  {g:9s} {r:10s}", " ".join(f"{x:+.2f}" for x in res[g][r]["mean"]))
        dd = res[g]["it_mix-jpp_norm"]
        print(f"  {g:9s} IT-J++    ", " ".join(f"{x:+.2f}[{a:+.2f},{b:+.2f}]" for x, (a, b) in zip(dd["mean"], dd["ci"])))
    for k, v in res.get("pair_r", {}).items():
        print(f"  pair r {k:28s} tp", " ".join(f"{x:+.2f}" for x in v["tokpair98"]), "| pron", " ".join(f"{x:+.2f}" for x in v["pron"]))
json.dump(out, open(R + "R1_11_evaldir_analysis.json", "w"), indent=1)
