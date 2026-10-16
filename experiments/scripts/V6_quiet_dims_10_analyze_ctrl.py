# V6_quiet_dims_10_analyze_ctrl (local): gap retention / positions under control masks (06b), self tokens, pron+noun.
import torch, numpy as np, json
R = "./experiments/results/"
D = torch.load(R + "V6_quiet_dims_06b_readout_ctrl.pt")
meta = D["meta"]; LAYS = D["layers"]; VARS = D["vars"]; CONDS = D["conds"]; NP = 24
BANDS = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
labs = np.array([m["lab"] for m in meta]); conds = np.array([m["cond"] for m in meta]); qis = np.array([m["qi"] for m in meta])
ci = {c: i for i, c in enumerate(CONDS)}
rng = np.random.default_rng(1)
out = {}
for kind in ("jpp", "jlens"):
    X = D["SC"][kind].numpy()
    P = np.zeros((len(CONDS), NP) + X.shape[1:])
    for c in CONDS:
        for q in range(NP):
            P[ci[c], q] = X[(conds == c) & (qis == q) & (labs == "self")].mean(0)
    W = (P[ci["woman_p1"]] + P[ci["woman_p2"]]) / 2; M = (P[ci["man_p1"]] + P[ci["man_p2"]]) / 2
    for mi, met in enumerate(("pron", "noun")):
        for b, r in BANDS.items():
            li = [LAYS.index(l) for l in r]
            gf = (W[:, li, 0, mi] - M[:, li, 0, mi]).mean(1)
            rec = {}
            for vi, vn in enumerate(VARS):
                w = W[:, li, vi, mi].mean(1); m = M[:, li, vi, mi].mean(1); gv = w - m
                x = {c: P[ci[c]][:, li, vi, mi].mean(1) for c in ("default", "nameless", "gemma_short")}
                bs = [gv[i].mean() / gf[i].mean() for i in (rng.integers(0, NP, NP) for _ in range(1000))]
                rec[vn] = {"ret": [float(gv.mean() / gf.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                           "pos_def": float((x["default"].mean() - m.mean()) / gv.mean()), "gem_nl": float((x["gemma_short"] - x["nameless"]).mean() / gv.mean()),
                           "def_nl": float((x["default"] - x["nameless"]).mean() / gv.mean()), "raw_def": float(x["default"].mean())}
            out[f"{kind}|{met}|{b}"] = rec
            print(f"{kind} {met} {b} " + " | ".join(f"{vn}: ret {v['ret'][0]:+.2f}[{v['ret'][1]:+.2f},{v['ret'][2]:+.2f}] pos {v['pos_def']:+.2f} gem-nl {v['gem_nl']:+.2f} def-nl {v['def_nl']:+.2f}" for vn, v in rec.items()))
json.dump(out, open(R + "V6_quiet_dims_10_ctrl_summary.json", "w"), indent=0)
