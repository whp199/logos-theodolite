#!/usr/bin/env python3
# R1_IT_LENS 15 (local): is the IT lens as low-rank as J++?
# (a) exact directional view: the 102 unit-normalised covectors (98 token-pair + 4 isotropic directions) at each layer form a
#     102 x D matrix for each lens (IT gbar, J++ pullback, JLens pullback). Energy in its top-k singular values and the mean
#     |cos| between covectors of different random directions (a low-rank lens maps unrelated directions onto similar covectors).
# (b) full-matrix view at R1_FM_LAYERS, in y-coordinates (L_y = diag(fnw) J): split-half-debiased signal energy of the IT
#     estimate in (i) J++'s top-k output subspace, (ii) its own top-k output subspace (from the other half). Probe noise is
#     isotropic in the output coordinates, so E||N V_k||^2 = (k/D) E||N||^2 is subtracted.
import json
import numpy as np
import torch

torch.set_num_threads(12)
R = "./experiments/results/"
LENSDIR = __import__("os").path.expanduser("~/.cache/huggingface/") + "hub/models--neuronpedia--jacobian-lens/snapshots/e166deb5ad540932100ef803c8c9bcc2a5453694/gemma-4-31b/"
FM = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
BANDS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]
D = 5376
res = {}
# ---------------- (a)
GB = torch.load(R + "R1_04_gbar.pt", weights_only=False)
DD = torch.load(R + "R1_02_dirs.pt", weights_only=False)
idx = list(range(30, 128)) + list(range(10, 14))
mats = {"it": GB["mix"][idx], "jpp": DD["pb_jpp"][idx], "jlens": DD["pb_jlens"][idx]}
KS = [1, 3, 10, 30]
for nm, M in mats.items():
    top = np.zeros((60, len(KS))); mc = np.zeros(60); er = np.zeros(60)
    for l in range(59):
        X = M[:, l].double()
        X = X / X.norm(dim=-1, keepdim=True)
        s = torch.linalg.svdvals(X) ** 2
        c = torch.cumsum(s, 0) / s.sum()
        top[l] = [float(c[k - 1]) for k in KS]
        p = s / s.sum(); er[l] = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
        G = X @ X.T
        mc[l] = float(G.abs()[~torch.eye(len(idx), dtype=bool)].mean())
    res.setdefault("directional", {})[nm] = {
        "topk_energy": {f"k{k}": [round(float(top[a:b + 1, i].mean()), 3) for a, b in BANDS] for i, k in enumerate(KS)},
        "eff_rank": [round(float(er[a:b + 1].mean()), 2) for a, b in BANDS],
        "mean_abs_cos_between_dirs": [round(float(mc[a:b + 1].mean()), 3) for a, b in BANDS]}
print("(a) directional, bands", [f"L{a}-{b}" for a, b in BANDS])
for nm in mats:
    d = res["directional"][nm]
    print(f"  {nm:6s} top1 {d['topk_energy']['k1']}  top10 {d['topk_energy']['k10']}  effrank {d['eff_rank']}  mean|cos| {d['mean_abs_cos_between_dirs']}")

# ---------------- (b)
ck3 = torch.load(R + "R1_03_fit_ckpt.pt", mmap=True, weights_only=False)
ck9 = torch.load(R + "R1_09_fm_ckpt.pt", mmap=True, weights_only=False)
NH = [float(ck3["NA"][h] + ck9["NA"][h]) for h in (0, 1)]
fnw = torch.load(R + "V6_quiet_dims_stats.pt", weights_only=False)["fnw"].float()
Jpp = torch.load(LENSDIR + "jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt", mmap=True, weights_only=True)["J"]
JLn = torch.load(LENSDIR + "jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt", mmap=True, weights_only=True)["J"]
KK = [10, 100, 300]
res["full"] = {"NH": NH, "layers": FM}
for j, l in enumerate(FM):
    L0 = (ck3["A"][0, j] + ck9["A"][0, j]).float() / NH[0]       # [D_in, D_out]  = L_y^T estimate
    L1 = (ck3["A"][1, j] + ck9["A"][1, j]).float() / NH[1]
    sig = float((L0 * L1).sum())
    n0 = float((L0 * L0).sum()) - sig; n1 = float((L1 * L1).sum()) - sig
    rec = {"signal_energy": sig, "noise_energy_h0": n0, "noise_energy_h1": n1, "snr_half": sig / max(n0, 1e-30)}
    for nm, Jm in (("jpp", Jpp), ("jlens", JLn)):
        Jy = (Jm[l].float() * fnw[:, None]).T                     # [D_in, D_out] = (diag(fnw) J)^T
        Jr = Jm[l].float().T
        _, sy, Vy = torch.svd_lowrank(Jy, q=320, niter=4)
        _, sr, _ = torch.svd_lowrank(Jr, q=320, niter=4)
        ty = float((Jy * Jy).sum()); tr = float((Jr * Jr).sum())
        rec[f"{nm}_self_topk_y"] = {k: round(float((sy[:k] ** 2).sum()) / ty, 3) for k in KK}
        rec[f"{nm}_self_topk_raw"] = {k: round(float((sr[:k] ** 2).sum()) / tr, 3) for k in KK}
        if nm == "jpp":
            for k in KK:
                V = Vy[:, :k]
                e = float(((L0 @ V) * (L1 @ V)).sum())
                rec.setdefault("it_in_jpp_topk_out", {})[k] = round(e / sig, 3)
    # IT's own top-k output subspace: from one half, energy measured on the other (noise-debiased), both directions
    fr = {k: [] for k in KK}
    for a, b, nb in ((L0, L1, n1), (L1, L0, n0)):
        _, s_, V_ = torch.svd_lowrank(a, q=320, niter=4)
        for k in KK:
            Vk = V_[:, :k]
            e = float(((b @ Vk) ** 2).sum()) - k / D * nb
            fr[k].append(e / sig)
    rec["it_self_topk_out_debiased"] = {k: round(float(np.mean(v)), 3) for k, v in fr.items()}
    res["full"][f"L{l}"] = rec
    print(f"L{l}: snr(half) {rec['snr_half']:.3f}  J++ top100(y) {rec['jpp_self_topk_y'][100]} raw {rec['jpp_self_topk_raw'][100]} | "
          f"JLens top100(y) {rec['jlens_self_topk_y'][100]} | IT in J++ top-k out {rec['it_in_jpp_topk_out']} | IT own top-k {rec['it_self_topk_out_debiased']}", flush=True)
json.dump(res, open(R + "R1_15_spectrum.json", "w"), indent=1)
