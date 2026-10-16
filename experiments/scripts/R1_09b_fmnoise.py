#!/usr/bin/env python3
# R1_IT_LENS 09b (local): noise diagnostic for the probe-estimated full-matrix IT lens. For fit-position states h (held-in for
# the gradients but independent of probe noise): split-half agreement cos(h A_0/N_0, h A_1/N_1) in y-space, and agreement with
# the real final normalised state y59 = fnw*h59 (cosine), vs J++ (fnw * J++ h). Extrapolates the samples needed:
# with signal s and per-sample noise n, E cos(y0,y1) ~ s/(s + n/S_half)  ->  n/s = S_half (1/c - 1).
import glob, sys
import numpy as np
import torch
torch.set_num_threads(12)
R = "./experiments/results/"
sys.path.insert(0, ".")
FM = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
ck = torch.load(R + "R1_03_fit_ckpt.pt", mmap=True, weights_only=False)
A, NA = ck["A"], ck["NA"]
extra = sys.argv[1:] and sys.argv[1] == "withB"
if extra:
    fm = torch.load(R + "R1_09_fm_ckpt.pt", mmap=True, weights_only=False)
v6 = torch.load(R + "V6_quiet_dims_stats.pt", weights_only=False)
fnw = v6["fnw"].float()
Hs = []
for f in sorted(glob.glob(R + "R1_03_fit_job*_*.pt"))[:6]:
    Hs.append(torch.load(f, weights_only=False)["H"].float())
H = torch.cat(Hs, 0)[::2][:300]                      # [n,60,D]
y59 = H[:, 59] * fnw
jpp = torch.load(__import__("os").path.expanduser("~/.cache/huggingface/") + "hub/models--neuronpedia--jacobian-lens/snapshots/e166deb5ad540932100ef803c8c9bcc2a5453694/gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt", mmap=True, weights_only=True)["J"]
print("n states", H.shape[0], "samples per half", NA.tolist(), ("+B " + str(fm["NA"].tolist())) if extra else "")
for j, l in enumerate(FM):
    A0 = A[0, j].float(); A1 = A[1, j].float(); n0, n1 = float(NA[0]), float(NA[1])
    if extra:
        A0 = A0 + fm["A"][0, j].float(); A1 = A1 + fm["A"][1, j].float(); n0 += float(fm["NA"][0]); n1 += float(fm["NA"][1])
    h = H[:, l]
    y0 = h @ (A0 / n0); y1 = h @ (A1 / n1); yb = h @ ((A0 + A1) / (n0 + n1))
    c01 = torch.nn.functional.cosine_similarity(y0, y1, dim=-1).mean().item()
    cb = torch.nn.functional.cosine_similarity(yb, y59, dim=-1).mean().item()
    yj = (h @ jpp[l].float().T) * fnw
    cj = torch.nn.functional.cosine_similarity(yj, y59, dim=-1).mean().item()
    Sh = (n0 + n1) / 2
    ns = Sh * (1 / max(c01, 1e-3) - 1)
    need = {f"S_for_noise{int(p*100)}%": int(ns * (1 - p) / p) for p in (0.2, 0.1)}   # pooled S so that noise/signal energy = p/(1-p)
    print(f"L{l:2d} split-half cos {c01:+.3f}  cos(IT,y59) {cb:+.3f}  cos(J++,y59) {cj:+.3f}  n/s={ns:,.0f}  {need}")
