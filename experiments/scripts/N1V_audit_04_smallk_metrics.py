#!/usr/bin/env python3
# N1V_audit_04_smallk_metrics -- offline metrics for N1V_audit_03_smallk (k=10 / k=3 steering, linear regime) vs the
# N1A k=60 cache: static label (J-VINDEX 'jpp', naive) vs causal c -- full-vocab Spearman and P@10->100 with the
# same-sign other-neuron null; split-half reliability; rho(c+, c-); cross-neuron rho; rho(c_k, c_k60) per pole.
# Run: .venv/bin/python experiments/scripts/N1V_audit_04_smallk_metrics.py -> experiments/results/N1V_audit_04_smallk_metrics.json
import glob, json, os, re
import torch
torch.set_num_threads(max(1, os.cpu_count() - 4))
ROOT = "./experiments/results"


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


sg = torch.tensor([1, -1] * 24); nn = torch.arange(24).repeat_interleave(2)
MASK = (sg[None, :] == sg[:, None]) & (nn[None, :] != nn[:, None])


def real_null(M):
    real = M.diag(); null = (M * MASK).sum(1) / MASK.sum(1)
    return {"real": round(float(real.mean()), 4), "null": round(float(null.mean()), 4), "lift": round(float((real - null).mean()), 4)}


out = {}
for fn in sorted(glob.glob(f"{ROOT}/N1V_cache/smallk_k*_L*.pt")):
    k, l = map(int, re.findall(r"smallk_k(\d+)_L(\d+)", fn)[0])
    d = torch.load(fn)
    C = d["c"].float(); cA = d["cA"].float(); cB = d["cB"].float()
    st = torch.load(f"{ROOT}/N1A_cache/static_L{l}.pt")
    assert st["neurons"] == d["neurons"]
    cz = torch.load(f"{ROOT}/N1A_cache/causal_kA_L{l}.pt")
    Ck = cz["c"][[i for i, m in enumerate(cz["poles"]) if m["set"] == "sample"]].float()
    r = {"kl_median": float(torch.tensor(d["kl"]).median())}
    for kind in ("jpp", "naive"):
        S = torch.stack([s * st[kind][n].float() for n in range(24) for s in (1, -1)])
        RS = rk(S)
        r[f"{kind}|rho"] = real_null(RS @ rk(C).T)
        r[f"{kind}|p10"] = real_null(p10(S, C))
        r[f"{kind}|p10_k60"] = real_null(p10(S, Ck))
    anti = (C[0::2] - C[1::2]) / 2
    Sj = torch.stack([st["jpp"][n].float() for n in range(24)])
    Mj = rk(Sj) @ rk(anti).T
    r["jpp_vs_antisym|rho"] = {"real": round(float(Mj.diag().mean()), 4),
                               "null": round(float(((Mj.sum(1) - Mj.diag()) / 23).mean()), 4)}
    Pj = p10(Sj, anti)
    r["jpp_vs_antisym|p10"] = {"real": round(float(Pj.diag().mean()), 4),
                               "null": round(float(((Pj.sum(1) - Pj.diag()) / 23).mean()), 4)}
    r["split_rho_median"] = round(float((rk(cA) * rk(cB)).sum(-1).median()), 4)
    R = rk(C); cc = R @ R.T
    r["rho_c+_c-"] = round(float(torch.stack([cc[2 * i, 2 * i + 1] for i in range(24)]).mean()), 4)
    r["cross_neuron_rho"] = round(float(((cc * MASK).sum(1) / MASK.sum(1)).mean()), 4)
    r["rho_c_k_vs_c_k60_same_pole_mean"] = round(float((rk(C) * rk(Ck)).sum(-1).mean()), 4)
    out[f"k{k}_L{l}"] = r
    print(f"k{k} L{l}", json.dumps(r), flush=True)
json.dump(out, open(f"{ROOT}/N1V_audit_04_smallk_metrics.json", "w"), indent=1)
