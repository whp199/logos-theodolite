#!/usr/bin/env python3
# N1C_08_coherence -- is the "neatness" of J-VINDEX labels evidence about the neuron, or a property of the lens?
# Run: .venv/bin/python experiments/scripts/N1C_08_coherence.py -> experiments/results/N1C_coherence.json
# TIGHTNESS of a 10-token label = mean pairwise cosine of the tokens' input embeddings (tied W_E, mean-centred over the
# vocabulary, L2-normalised) -- a cheap proxy for "the label looks like one concept".
# Conditions per layer: real sample poles (JV, NV, C, Enx), and null directions from N1C_07 (single other neurons,
# 2-neuron sums, 512-neuron Gaussian mixtures, isotropic) through J++ (JV) and the naive lens (NV).
# PRE-STATED PREDICTION: see N1C_07 header. Additionally: real JV tightness > real NV tightness at L10-30.
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/")
SNAP += sorted(os.listdir(SNAP))[0] + "/model.safetensors"
with safe_open(SNAP, "pt") as f:
    WE = f.get_tensor("model.language_model.embed_tokens.weight")
mu = torch.zeros(WE.shape[1])
for s in range(0, WE.shape[0], 32768):
    mu += WE[s:s + 32768].float().sum(0)
mu /= WE.shape[0]

def tight(ids_list):
    out = []
    for ids in ids_list:
        ids = [int(t) for t in ids if int(t) >= 0]
        if len(ids) < 2:
            out.append(np.nan); continue
        E = WE[ids].float() - mu
        E = E / E.norm(dim=-1, keepdim=True)
        G = E @ E.T
        k = len(ids)
        out.append(float((G.sum() - k) / (k * (k - 1))))
    return np.array(out)

Z = np.load(f"{ROOT}/N1C_labels.npz")
A = Z["A"]
lab_enx = Z["lab_Enx"]
N07 = json.load(open(f"{ROOT}/N1C_07_coherence.json"))
res = {"definition": "mean pairwise cosine of mean-centred, normalised input embeddings of the label's top-10 tokens", "per_layer": {}}
rng = np.random.default_rng(3)
def bmean(x):
    x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "n": int(len(x))}
def bdiff(x, y):
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    b = x[rng.integers(0, len(x), (2000, len(x)))].mean(1) - y[rng.integers(0, len(y), (2000, len(y)))].mean(1)
    return {"diff": float(x.mean() - y.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
allv = {}
for l in LAYERS:
    rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
    cond = {"real_JV": tight(Z["top_JV"][rows][:, :10]), "real_NV": tight(Z["top_NV"][rows][:, :10]),
            "real_C": tight(Z["top_C"][rows][:, :10]),
            "real_Enx": tight([A[np.argsort(-lab_enx[r])[:min(10, int(np.isfinite(lab_enx[r]).sum()))]] for r in rows])}
    for nm in ("single", "pair", "mix512", "iso"):
        cond[f"{nm}_JV"] = tight(N07[str(l)][f"{nm}_jpp"])
        cond[f"{nm}_NV"] = tight(N07[str(l)][f"{nm}_naive"])
    d = {k: bmean(v) for k, v in cond.items()}
    d["real_JV-mix512_JV"] = bdiff(cond["real_JV"], cond["mix512_JV"])
    d["real_JV-real_NV"] = bdiff(cond["real_JV"], cond["real_NV"])
    d["real_NV-mix512_NV"] = bdiff(cond["real_NV"], cond["mix512_NV"])
    res["per_layer"][str(l)] = d
    for k, v in cond.items():
        allv.setdefault(k, {}).setdefault("early" if l in EARLY else "late", []).append(v)
res["bands"] = {}
for k, dd in allv.items():
    for band, vs in dd.items():
        res["bands"].setdefault(band, {})[k] = bmean(np.concatenate(vs))
for band in ("early", "late"):
    cat = lambda k: np.concatenate(allv[k][band])
    res["bands"][band]["real_JV-mix512_JV"] = bdiff(cat("real_JV"), cat("mix512_JV"))
    res["bands"][band]["real_JV-real_NV"] = bdiff(cat("real_JV"), cat("real_NV"))
    res["bands"][band]["real_NV-mix512_NV"] = bdiff(cat("real_NV"), cat("mix512_NV"))
json.dump(res, open(f"{ROOT}/N1C_coherence.json", "w"), indent=1)
keys = ["real_JV", "single_JV", "pair_JV", "mix512_JV", "iso_JV", "real_NV", "mix512_NV", "iso_NV", "real_C", "real_Enx"]
print("| cond | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
for k in keys:
    print(f"| {k} | " + " | ".join(f"{res['per_layer'][str(l)][k]['mean']:.3f}" for l in LAYERS) + " |")
for k in ["real_JV-mix512_JV", "real_JV-real_NV", "real_NV-mix512_NV"]:
    print(f"| {k} | " + " | ".join(f"{res['per_layer'][str(l)][k]['diff']:+.3f}" for l in LAYERS) + " |")
print(json.dumps(res["bands"], indent=0)[:3000])
