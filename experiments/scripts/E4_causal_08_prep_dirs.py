# E4_causal_08 (CPU, offline): unit directions per layer for the causal tests -> cache/dirs_causal.pt
import torch
C = "./experiments/results/E4_causal_cache"
S1 = torch.load(f"{C}/self.pt"); S2 = torch.load(f"{C}/self2.pt"); T3 = torch.load(f"{C}/third.pt"); PB = torch.load(f"{C}/pullbacks.pt")
def diff(S, cf, cm, key="self_post"):
    it = S["meta"]["items"]
    f = {x["p"]: i for i, x in enumerate(it) if x["cond"] == cf}; m = {x["p"]: i for i, x in enumerate(it) if x["cond"] == cm}
    return (S[key][[f[p] for p in range(24)]].double() - S[key][[m[p] for p in range(24)]].double()).mean(0)
t3 = T3["meta"]["items"]
F_ = {x["i"]: k for k, x in enumerate(t3) if x["cond"] == "F"}; M_ = {x["i"]: k for k, x in enumerate(t3) if x["cond"] == "M"}
def cont(ii):
    return (T3["cont_post"][[F_[i] for i in ii]].double() - T3["cont_post"][[M_[i] for i in ii]].double()).mean(0)
dirs = {
    "u_self": diff(S1, "woman", "man"),
    "u_nn": diff(S2, "woman_nn", "man_nn"),
    "u_aiexp": diff(S2, "AI_female", "AI_male"),
    "u_cont": cont(range(36)), "u_cont_A": cont(range(18)), "u_cont_B": cont(range(18, 36)),
    "pb_jpp": PB["pron_jpp"].double(),
}
g = torch.Generator().manual_seed(1234)
for k in range(3):
    dirs[f"rand{k}"] = torch.randn(60, 5376, generator=g, dtype=torch.float64)
gap = {k: v.norm(dim=-1).float() for k, v in dirs.items()}   # raw diff-of-means norms (gap size along own direction)
dirs = {k: (v / v.norm(dim=-1, keepdim=True)).float() for k, v in dirs.items()}
norms = torch.cat([S1["norms"], S2["norms"]]).mean(0).float()  # mean content-token resid norm per layer
torch.save({"dirs": dirs, "gap": gap, "resid_norm": norms}, f"{C}/dirs_causal.pt")
print({k: round(float(v[30]), 2) for k, v in gap.items()}, "resid norm L30", float(norms[30]))
