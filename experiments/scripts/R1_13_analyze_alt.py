#!/usr/bin/env python3
# R1_IT_LENS 13 (local): alternative explanations for the IT-vs-J++ quiet-subspace disagreement.
#  (a) method: LRP-linearised same-position Jacobian (frozen norms / attention patterns / GeGLU gate) vs the true one (R1_05a,
#      paired positions). Does linearisation move the IT gradient toward J++ inside q10?
#  (b) definition: total-derivative covectors sum_{p'>=p} J_{p',p}^T d (true and LRP) vs J++ (R1_05b).
# CIs: bootstrap over the 98 token-pair directions where available; key directions reported individually.
import glob, json
import numpy as np
import torch

R = "./experiments/results/"
BANDS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]
BN = [f"L{a}-{b}" for a, b in BANDS]
rng = np.random.default_rng(3)
DD = torch.load(R + "R1_02_dirs.pt", weights_only=False)
names, PBJ, PBL = DD["names"], DD["pb_jpp"], DD["pb_jlens"]
K = len(names)
v6 = torch.load(R + "V6_quiet_dims_stats.pt", weights_only=False)
std = v6["std_all"].float()
MASK = {"all": torch.ones(60, 5376)}
MASK["q10"] = (std <= torch.quantile(std, 0.1, dim=1, keepdim=True)).float()
MASK["loud"] = 1 - MASK["q10"]
GB = torch.load(R + "R1_04_gbar.pt", weights_only=False)


def mcos(a, b, m):
    num = (a * b * m).sum(-1)
    return num / ((a * a * m).sum(-1) * (b * b * m).sum(-1)).clamp_min(1e-300).sqrt()


def band(x):
    return np.stack([x[..., a:b + 1].mean(-1) for a, b in BANDS], -1)


def summ(x, idx):
    xb = band(x)[idx]
    d = {"mean": xb.mean(0).round(3).tolist()}
    if len(idx) >= 4:
        bs = np.stack([xb[rng.integers(0, len(idx), len(idx))].mean(0) for _ in range(1000)])
        d["ci"] = np.percentile(bs, [2.5, 97.5], 0).T.round(3).tolist()
    return d


res = {"bands": BN}
# ---------------- (a) LRP same-position, paired
fs = sorted(glob.glob(R + "R1_05_lrpsame_part*.pt"))
if fs:
    acc = {"true": 0, "lrp": 0}; cnt = 0; pc = []
    for f in fs:
        d = torch.load(f, weights_only=False)
        for m in ("true", "lrp"):
            acc[m] = acc[m] + d["acc"][m]
        cnt = cnt + d["cnt"]; pc.append(d["pcos"].float())
    present = [i for i in range(5) if cnt[i] > 0]
    L = {m: sum(acc[m][i] / cnt[i] for i in present) / len(present) for m in ("true", "lrp")}
    TP = list(range(30, K))
    out = {"n_positions": int(cnt.sum())}
    for mk in ("all", "loud", "q10"):
        for m in ("true", "lrp"):
            c = mcos(L[m], PBJ, MASK[mk]).numpy()
            out[f"cos_{m}_jpp_{mk}"] = {"tokpair98": summ(c, TP), "pron": summ(c, [0])}
            c = mcos(L[m], PBL, MASK[mk]).numpy()
            out[f"cos_{m}_jlens_{mk}"] = {"tokpair98": summ(c, TP), "pron": summ(c, [0])}
        c = mcos(L["lrp"], L["true"], MASK[mk]).numpy()
        out[f"cos_lrp_true_{mk}"] = {"tokpair98": summ(c, TP), "pron": summ(c, [0])}
        c = mcos(L["true"], GB["mix"], MASK[mk]).numpy()
        out[f"cos_subsettrue_fullfit_{mk}"] = {"tokpair98": summ(c, TP), "pron": summ(c, [0])}
    out["single_pos_cos_lrp_true"] = {"tokpair98": summ(torch.cat(pc).mean(0).numpy(), TP)}
    res["lrp_same"] = out
    print("== (a) LRP-linearised vs true same-position VJP,", out["n_positions"], "positions (token pairs)")
    for k in ("cos_true_jpp_all", "cos_lrp_jpp_all", "cos_true_jpp_loud", "cos_lrp_jpp_loud", "cos_true_jpp_q10", "cos_lrp_jpp_q10",
              "cos_lrp_true_all", "cos_lrp_true_loud", "cos_lrp_true_q10", "cos_subsettrue_fullfit_all"):
        print(f"  {k:28s}", " ".join(f"{x:+.2f}" for x in out[k]["tokpair98"]["mean"]), "| pron", " ".join(f"{x:+.2f}" for x in out[k]["pron"]["mean"]))
# ---------------- (b) total derivative
fs = sorted(glob.glob(R + "R1_05_total_part*.pt"))
if fs:
    acc, cnt = {}, {}
    for f in fs:
        d = torch.load(f, weights_only=False)
        KD = d["dirs"]
        for k, v in d["acc"].items():
            acc[k] = acc.get(k, 0) + v; cnt[k] = cnt.get(k, 0) + d["cnt"][k]
    out = {"dirs": [names[k] for k in KD], "counts": cnt}
    regions = sorted({k.split("|")[1] for k in acc})
    for mode in ("true", "lrp"):
        Lm = sum(acc[f"{mode}|{r}"] / cnt[f"{mode}|{r}"] for r in regions) / len(regions)    # [6,60,D]
        for mk in ("all", "loud", "q10"):
            out[f"cos_total{mode}_jpp_{mk}"] = band(mcos(Lm, PBJ[KD], MASK[mk]).numpy()).round(3).tolist()
            out[f"cos_total{mode}_jlens_{mk}"] = band(mcos(Lm, PBL[KD], MASK[mk]).numpy()).round(3).tolist()
            out[f"cos_total{mode}_samepos_{mk}"] = band(mcos(Lm, GB["mix"][KD], MASK[mk]).numpy()).round(3).tolist()
        for r in regions:
            Lr = acc[f"{mode}|{r}"] / cnt[f"{mode}|{r}"]
            out[f"cos_total{mode}_{r}_jpp_q10"] = band(mcos(Lr, PBJ[KD], MASK["q10"]).numpy()).round(3).tolist()
        out[f"norm_ratio_total{mode}_over_samepos"] = band((Lm.norm(dim=-1) / GB["mix"][KD].norm(dim=-1)).numpy()).round(3).tolist()
    for mk in ("all", "loud", "q10"):
        out[f"cos_samepos_jpp_{mk}"] = band(mcos(GB["mix"][KD], PBJ[KD], MASK[mk]).numpy()).round(3).tolist()
    res["total"] = out
    print("== (b) total-derivative covectors; rows = dirs", out["dirs"])
    for k in ("cos_samepos_jpp_all", "cos_totaltrue_jpp_all", "cos_totallrp_jpp_all", "cos_samepos_jpp_q10", "cos_totaltrue_jpp_q10",
              "cos_totallrp_jpp_q10", "cos_samepos_jpp_loud", "cos_totaltrue_jpp_loud", "cos_totaltrue_samepos_all", "norm_ratio_totaltrue_over_samepos"):
        a = np.array(out[k])
        print(f"  {k:34s} mean over dirs:", " ".join(f"{x:+.2f}" for x in a.mean(0)), "| pron", " ".join(f"{x:+.2f}" for x in a[0]))
json.dump(res, open(R + "R1_13_alt_analysis.json", "w"), indent=1)
