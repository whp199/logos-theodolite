#!/usr/bin/env python3
# R1_IT_LENS 14 (local): E3 fidelity metrics for the full-matrix IT lens vs J++ / JLens / naive / hybrids (from R1_10).
# Per position class: mean KL(final||lens) at tau=1, cross-fitted calibrated KL (tau chosen on even passages, scored on odd and
# vice versa), top-1 agreement, top-10 overlap. 95% CIs by bootstrap over passages (1000 resamples).
import json
import numpy as np
import torch

R = "./experiments/results/"
d = torch.load(R + "R1_10_evalfull.pt", weights_only=False)
meta, KL, T1, T10 = d["meta"], d["KL"].float().numpy(), d["T1"].float().numpy(), d["T10"].float().numpy()
LEN, TAUS, FM = d["lenses"], d["taus"], d["layers"]
ti1 = TAUS.index(1.0)
rng = np.random.default_rng(4)
cls = np.array([m["cls"] for m in meta]); nm = np.array([m["name"] for m in meta])
out = {"layers": FM, "lenses": LEN, "taus": TAUS, "NH": d["NH"]}


def calib_kl(sel):
    """cross-fitted calibrated KL per (layer, lens): choose tau on one passage parity, score on the other. -> [nL, nLens]"""
    names = sorted(set(nm[sel]))
    par = np.array([names.index(x) % 2 for x in nm[sel]])
    K = KL[sel]                                            # [P,nL,nLens,nT]
    tot = np.zeros(K.shape[1:3]); n = 0
    for a in (0, 1):
        fitm = K[par == a].mean(0)                         # [nL,nLens,nT]
        best = fitm.argmin(-1)                             # [nL,nLens]
        ev = K[par != a]
        tot += np.take_along_axis(ev.mean(0), best[..., None], -1)[..., 0] * (par != a).sum(); n += (par != a).sum()
    return tot / n


for c in ("prose", "asst", "self", "user"):
    sel = cls == c
    names = sorted(set(nm[sel]))
    res = {"n_pos": int(sel.sum()), "n_passages": len(names)}
    res["kl"] = KL[sel][:, :, :, ti1].mean(0).round(3).tolist()
    res["kl_cal"] = calib_kl(sel).round(3).tolist()
    res["top1"] = T1[sel].mean(0).round(3).tolist()
    res["top10"] = T10[sel].mean(0).round(3).tolist()
    res["best_tau"] = np.array(TAUS)[KL[sel].mean(0).argmin(-1)].tolist()
    # passage bootstrap for differences vs J++
    pidx = {x: np.where(sel & (nm == x))[0] for x in names}
    B = 1000
    bs_kl, bs_t1 = [], []
    for b in range(B):
        pick = rng.choice(names, len(names))
        ii = np.concatenate([pidx[x] for x in pick])
        bs_kl.append(KL[ii][:, :, :, ti1].mean(0)); bs_t1.append(T1[ii].mean(0))
    bs_kl = np.stack(bs_kl); bs_t1 = np.stack(bs_t1)
    j = LEN.index("jpp")
    res["kl_ci"] = np.percentile(bs_kl, [2.5, 97.5], 0).round(3).tolist()
    res["top1_ci"] = np.percentile(bs_t1, [2.5, 97.5], 0).round(3).tolist()
    res["kl_minus_jpp_ci"] = np.percentile(bs_kl - bs_kl[:, :, j:j + 1], [2.5, 97.5], 0).round(3).tolist()
    res["top1_minus_jpp_ci"] = np.percentile(bs_t1 - bs_t1[:, :, j:j + 1], [2.5, 97.5], 0).round(3).tolist()
    out[c] = res
    print(f"\n== {c}: {res['n_pos']} positions, {len(names)} passages; layers {FM}")
    for k, ln in enumerate(LEN):
        print(f"  KL  {ln:10s}", " ".join(f"{np.array(res['kl'])[i, k]:6.2f}" for i in range(len(FM))))
    for k, ln in enumerate(LEN):
        print(f"  KLc {ln:10s}", " ".join(f"{np.array(res['kl_cal'])[i, k]:6.2f}" for i in range(len(FM))))
    for k, ln in enumerate(LEN):
        print(f"  T1  {ln:10s}", " ".join(f"{np.array(res['top1'])[i, k]:6.3f}" for i in range(len(FM))))
json.dump(out, open(R + "R1_14_full_analysis.json", "w"), indent=1)
