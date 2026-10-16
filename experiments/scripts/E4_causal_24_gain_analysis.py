# E4_causal_24 (CPU, offline): why are random / J++-pullback directions destructive while diff-of-means directions are not?
# Hypothesis: RMSNorm weights have extreme per-dim gains (input_layernorm up to ~1248, pre_feedforward_layernorm up to ~860,
# mean |w| ~23-41); natural activation variation avoids high-gain dims, isotropic directions do not.
# Metric: gain-weighted energy  E_g(u) = sum_i (w_i u_i)^2 for unit u (w = input_layernorm weight of layer l+1, i.e. the
# next layer's read), averaged over L25-39; compared with the same metric for the empirical activation-difference vectors.
import json, torch, numpy as np, torch.nn.functional as F
C = "./experiments/results/E4_causal_cache"
W = torch.load(f"{C}/normweights.pt"); D1 = torch.load(f"{C}/dirs_causal.pt")["dirs"]; D2 = torch.load(f"{C}/dirs_ctrl.pt")["dirs"]
S1 = torch.load(f"{C}/self.pt")
dirs = {**{k: D1[k] for k in ("u_self", "u_aiexp", "u_cont", "pb_jpp", "rand0", "rand1")}, **{k: D2[k] for k in ("ctrl_nat0", "ctrl_topic", "ctrl_name")}}
X = S1["self_post"].double()
res = {}
layers = list(range(25, 40))
for nm in ("input_layernorm", "pre_feedforward_layernorm"):
    res[nm] = {}
    for k, v in dirs.items():
        e = [float(((W[nm][l + 1].double() * v[l].double()) ** 2).sum()) for l in layers]
        res[nm][k] = float(np.mean(e))
    # empirical: unit-normalized differences between random pairs of states (natural variation)
    g = torch.Generator().manual_seed(0); emp = []
    for l in layers:
        i = torch.randint(0, X.shape[0], (200,), generator=g); j = torch.randint(0, X.shape[0], (200,), generator=g)
        d = F.normalize(X[i, l] - X[j, l], dim=-1)
        emp.append(float(((W[nm][l + 1].double() * d) ** 2).sum(-1).mean()))
    res[nm]["natural_state_differences"] = float(np.mean(emp))
    w = W[nm][layers].abs()
    res[nm]["weight_stats_L25-39"] = {"max": float(w.max()), "mean": float(w.mean()), "median": float(w.median()), "frac_dims_gt_100": float((w > 100).float().mean())}
for nm in res:
    print(nm, res[nm].pop("weight_stats_L25-39") if False else res[nm]["weight_stats_L25-39"])
    for k, v in res[nm].items():
        if k != "weight_stats_L25-39": print(f"   {k:26s} gain-weighted energy {v:10.1f}")
# KL observed (additive alpha=0.02, L25-39) for reference
json.dump(res, open("./experiments/results/E4_causal_24_gain.json", "w"))
