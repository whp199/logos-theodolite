#!/usr/bin/env python3
# N1V_audit_05_antisym_cpu -- verifier CPU check: static labels vs c+, antisymmetric (c+ - c-)/2 and c+ minus layer-mean of the N1A cache (removes the shared even/generic component). Output: experiments/results/N1V_audit_05_antisym.json
import json, torch, os
torch.set_num_threads(16)
ROOT="./experiments/results"; CACHE=f"{ROOT}/N1A_cache"
NEU=json.load(open(f"{ROOT}/N1_neurons.json")); LAYERS=[int(l) for l in NEU["layers"]]
NEURONS={int(l):[int(i) for i in v] for l,v in NEU["neurons"].items()}
def rk(X):
    R = X.float().argsort(-1).argsort(-1).float()
    R = R - R.mean(-1, keepdim=True); return R / R.norm(dim=-1, keepdim=True)
def p10(S, C):
    a = S.topk(10,-1).indices; b = C.topk(100,-1).indices
    A = torch.zeros(S.shape[0], S.shape[1]); A.scatter_(1,a,1.)
    B = torch.zeros(C.shape[0], C.shape[1]); B.scatter_(1,b,1.)
    return A @ B.T / 10
out={}
for tag in ["kA","kB"]:
  for l in LAYERS:
    st = torch.load(f"{CACHE}/static_L{l}.pt"); cz = torch.load(f"{CACHE}/causal_{tag}_L{l}.pt")
    idx=[i for i,m in enumerate(cz["poles"]) if m["set"]=="sample"]
    C = cz["c"][idx].float()
    cp, cm = C[0::2], C[1::2]           # + and - poles per neuron
    anti = (cp - cm)/2                   # antisymmetric part (shared generic term cancels)
    resid = cp - C.mean(0, keepdim=True) # + pole minus layer-mean of all poles
    r = {}
    for kind in ["jpp","naive"]:
        S = st[kind].float()           # + pole label
        RS = rk(S)
        for name, T in [("c+", cp), ("anti", anti), ("c+_minus_layermean", resid)]:
            RT = rk(T); M = RS @ RT.T
            n = M.shape[0]; off = ~torch.eye(n, dtype=bool)
            real = M.diag().mean().item(); null = (M*off).sum(1).div(n-1).mean().item()
            P = p10(S, T); pr = P.diag().mean().item(); pn = (P*off).sum(1).div(n-1).mean().item()
            r[f"{kind}|{name}"] = dict(rho=round(real,3), rho_null=round(null,3), p10=round(pr,3), p10_null=round(pn,3))
    # norms: generic vs antisym part
    sym = (cp + cm)/2
    r["norm_sym_over_anti"] = round((sym.norm(dim=-1)/anti.norm(dim=-1)).median().item(),3)
    out[f"{tag}_L{l}"]=r
    print(tag, l, r["norm_sym_over_anti"], {k:v for k,v in r.items() if k!="norm_sym_over_anti"}, flush=True)
json.dump(out, open("./experiments/results/N1V_audit_05_antisym.json","w"), indent=1)
