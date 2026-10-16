#!/usr/bin/env python3
# N1V_audit_02_metrics -- offline (CPU) metrics for N1V_audit_01_postnorm steering results vs the N1A kA cache.
# Run: .venv/bin/python experiments/scripts/N1V_audit_02_metrics.py -> experiments/results/N1V_audit_02_metrics.json
# For each steered layer: residual-direct steering c_dir (norm-matched to the k=60 residual change) vs static labels
# (J-VINDEX 'jpp', naive) -- full-vocab Spearman and P@10->100, each with the within-layer same-sign other-neuron
# null (23 partners, exhaustive) -- plus split-half reliability, rho(c+, c-), cross-neuron rho; and the correlation of
# the MLP-ablation effect c_abl with the symmetric part of N1A's cached kA vectors.
import json, os
import torch
torch.set_num_threads(max(1, os.cpu_count() - 4))
ROOT = "./experiments/results"
NEU = json.load(open(f"{ROOT}/N1_neurons.json"))
NEURONS = {int(l): [int(i) for i in v] for l, v in NEU["neurons"].items()}


def rk(X):
    X = X.float()
    R = X.argsort(-1).argsort(-1).float()
    R = R - R.mean(-1, keepdim=True)
    return R / R.norm(dim=-1, keepdim=True)


def p10(S, C):
    a = S.topk(10, -1).indices; b = C.topk(100, -1).indices
    A = torch.zeros(S.shape[0], S.shape[1]); A.scatter_(1, a, 1.)
    B = torch.zeros(C.shape[0], C.shape[1]); B.scatter_(1, b, 1.)
    return A @ B.T / 10


def real_null(M, sg, nn):
    mask = (sg[None, :] == sg[:, None]) & (nn[None, :] != nn[:, None])
    real = M.diag()
    null = (M * mask).sum(1) / mask.sum(1)
    return {"real": round(float(real.mean()), 4), "null": round(float(null.mean()), 4),
            "lift": round(float((real - null).mean()), 4), "frac_real_gt_null": round(float((real > null).float().mean()), 3)}


out = {}
for l in [10, 20, 30, 40]:
    fn = f"{ROOT}/N1V_cache/steer_dir_L{l}.pt"
    if not os.path.exists(fn):
        continue
    d = torch.load(fn)
    jobs = d["poles"]
    idx_dir = [k for k, j in enumerate(jobs) if j[0] == "dir"]
    k_abl = [k for k, j in enumerate(jobs) if j[0] == "abl"][0]
    Cd = d["c"][idx_dir].float()                      # [48, V] pole order neuron-major (+,-)
    cA = d["cA"][idx_dir].float(); cB = d["cB"][idx_dir].float()
    c_abl = d["c"][k_abl].float()
    st = torch.load(f"{ROOT}/N1A_cache/static_L{l}.pt")
    cz = torch.load(f"{ROOT}/N1A_cache/causal_kA_L{l}.pt")
    sel = [k for k, m in enumerate(cz["poles"]) if m["set"] == "sample"]
    Ck = cz["c"][sel].float()
    sg = torch.tensor([1, -1] * 24); nn = torch.arange(24).repeat_interleave(2)
    r = {"kl_dir": d["kl"]}
    for kind in ("jpp", "naive"):
        S = torch.stack([s * st[kind][n].float() for n in range(24) for s in (1, -1)])
        RS = rk(S)
        for nm, C in (("dir", Cd), ("kA_cached", Ck)):
            r[f"{kind}->{nm}|rho"] = real_null(RS @ rk(C).T, sg, nn)
            r[f"{kind}->{nm}|p10"] = real_null(p10(S, C), sg, nn)
    for nm, C, a_, b_ in (("dir", Cd, cA, cB),):
        r[f"split_rho_{nm}_median"] = round(float((rk(a_) * rk(b_)).sum(-1).median()), 4)
    for nm, C in (("dir", Cd), ("kA_cached", Ck)):
        R = rk(C)
        cc = R @ R.T
        r[f"rho_c+_c-_{nm}"] = round(float(torch.stack([cc[2 * i, 2 * i + 1] for i in range(24)]).mean()), 4)
        mask = (nn[None, :] != nn[:, None]) & (sg[None, :] == sg[:, None])
        r[f"cross_neuron_rho_{nm}"] = round(float(((cc * mask).sum(1) / mask.sum(1)).mean()), 4)
    # ablation vs symmetric part of the cached kA vectors (and of the direct vectors)
    Ra = rk(c_abl[None])
    for nm, C in (("kA_cached", Ck), ("dir", Cd)):
        sym = (C[0::2] + C[1::2]) / 2                      # [24, V]
        r[f"rho_abl_vs_sym_{nm}_mean_over_neurons"] = round(float((Ra @ rk(sym).T).mean()), 4)
        r[f"rho_abl_vs_layer_mean_{nm}"] = round(float(Ra @ rk(C.mean(0, keepdim=True)).T), 4)
        r[f"rho_abl_vs_each_pole_{nm}_mean"] = round(float((Ra @ rk(C).T).mean()), 4)
    out[str(l)] = r
    print(l, json.dumps(r, indent=0)[:3000], flush=True)
json.dump(out, open(f"{ROOT}/N1V_audit_02_metrics.json", "w"), indent=1)
