#!/usr/bin/env python3
# R1_IT_LENS 17 (local): identity positions under the full-matrix IT lens (tool convention), from R1_16. Self tokens,
# woman(1)/man(0) persona scale on passage means; bands = FM layers {30,35}, {40,45}, {50,55,58}; passage bootstrap (2000).
import json
import numpy as np
import torch

R = "./experiments/results/"
d = torch.load(R + "R1_16_identity_full.pt", weights_only=False)
meta, S, cols, conds, FM = d["meta"], d["scores"].float().numpy(), d["cols"], d["conds"], d["layers"]
BANDS = {"L30-39": [30, 35], "L40-49": [40, 45], "L50-58": [50, 55, 58]}
qis = sorted({m["qi"] for m in meta})
A = np.full((len(qis), len(conds), len(FM), len(cols)), np.nan)
for m in meta:
    idx = [m["off"] + i for i, lb in enumerate(m["labels"]) if lb == "self"]
    A[qis.index(m["qi"]), conds.index(m["cond"])] = S[idx].mean(0)
ci = {c: i for i, c in enumerate(conds)}
AI = ["default", "nameless", "gemma_short", "gemma_explicit", "f_sarah", "m_james", "n_atlas", "n_quill", "n_unit7"]


def pos(Aq, col, layers):
    li = [FM.index(l) for l in layers]
    M = Aq[:, :, li, cols.index(col)].mean(-1).mean(0)
    w = (M[ci["woman_p1"]] + M[ci["woman_p2"]]) / 2; mn = (M[ci["man_p1"]] + M[ci["man_p2"]]) / 2
    p = {c: (M[ci[c]] - mn) / (w - mn) for c in conds}
    p["neutral3"] = np.mean([p[c] for c in ("n_atlas", "n_quill", "n_unit7")])
    p["mean_ai"] = np.mean([p[c] for c in AI]); p["n_ge_woman"] = int(sum(p[c] >= 1 for c in AI)); p["gap"] = w - mn
    p["default-nameless"] = p["default"] - p["nameless"]
    return p


rng = np.random.default_rng(5)
out = {}
for col in cols:
    for bn, layers in BANDS.items():
        pt = pos(A, col, layers)
        bs = [pos(A[rng.integers(0, len(qis), len(qis))], col, layers) for _ in range(2000)]
        rec = {}
        for k in ("default", "nameless", "neutral3", "gemma_short", "f_sarah", "m_james", "mean_ai", "default-nameless", "gap"):
            arr = np.array([b[k] for b in bs])
            rec[k] = [round(float(pt[k]), 3), round(float(np.percentile(arr, 2.5)), 3), round(float(np.percentile(arr, 97.5)), 3)]
        rec["n_ge_woman"] = pt["n_ge_woman"]
        out.setdefault(col, {})[bn] = rec
        print(f"{col:12s} {bn}: " + "  ".join(f"{k}={v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for k, v in rec.items() if k not in ("n_ge_woman", "gap"))
              + f"  n>=woman={rec['n_ge_woman']}  gap={rec['gap'][0]:.2f}")
json.dump(out, open(R + "R1_17_identity_full_analysis.json", "w"), indent=1)
