#!/usr/bin/env python3
# N1V_audit_07_tightness_iso -- verifier CPU check: N1C_08 tightness of real J-VINDEX labels vs isotropic / single / mix512 null directions, per band, with bootstrap CIs.
import sys, os, json, numpy as np, torch
sys.path.insert(0, "./experiments/scripts")
from N1C_common import *
from safetensors import safe_open
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/")
SNAP += sorted(os.listdir(SNAP))[0] + "/model.safetensors"
with safe_open(SNAP, "pt") as f:
    WE = f.get_tensor("model.language_model.embed_tokens.weight")
mu = torch.zeros(WE.shape[1])
for s in range(0, WE.shape[0], 32768): mu += WE[s:s+32768].float().sum(0)
mu /= WE.shape[0]
def tight(L):
    o = []
    for ids in L:
        ids = [int(t) for t in ids]
        E = WE[ids].float() - mu; E = E / E.norm(dim=-1, keepdim=True); G = E @ E.T; k = len(ids)
        o.append(float((G.sum() - k) / (k * (k - 1))))
    return np.array(o)
Z = np.load(f"{ROOT}/N1C_labels.npz"); N07 = json.load(open(f"{ROOT}/N1C_07_coherence.json"))
rng = np.random.default_rng(0)
def bd(x, y):
    b = x[rng.integers(0, len(x), (4000, len(x)))].mean(1) - y[rng.integers(0, len(y), (4000, len(y)))].mean(1)
    return round(float(x.mean() - y.mean()), 4), [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]
for band, Ls in (("early", EARLY), ("late", LATE)):
    rj, ij, sj, mj = [], [], [], []
    for l in Ls:
        rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        rj.append(tight(Z["top_JV"][rows][:, :10])); ij.append(tight(N07[str(l)]["iso_jpp"]))
        sj.append(tight(N07[str(l)]["single_jpp"])); mj.append(tight(N07[str(l)]["mix512_jpp"]))
    rj, ij, sj, mj = map(np.concatenate, (rj, ij, sj, mj))
    print(band, "real-iso", bd(rj, ij), "real-single", bd(rj, sj), "single-mix", bd(sj, mj), "iso-mix", bd(ij, mj))
