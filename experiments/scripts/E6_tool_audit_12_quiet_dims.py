# E6_tool_audit_12_quiet_dims -- follow-up to 11: random unit directions of norm 2 (~1% of ||h||) added at L29..53 collapse the
# model while a data-derived direction of norm 10 is benign. Hypothesis: "quiet" residual dimensions (tiny natural variance)
# have enormous downstream gain; J++-pullback directions concentrate on them (regression-fitted lens weights low-variance dims).
# PRE-STATED PREDICTIONS: (a) random directions restricted to quiet dims (per-dim std < 10th pct) collapse the model at norm 2;
# (b) random directions restricted to loud dims (std > median) or variance-weighted random directions do not;
# (c) the tool's steering directions (server, grad) carry far more of their energy on quiet dims than random (10%).
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
C = build_conditions()
LAY = [29, 35, 41, 47, 53]
# per-dim residual std over all non-BOS tokens of 24 prompts (A_default + B_gemma), per layer
prompts = C["A_default"] + C["B_gemma"]
stds = {l: [] for l in LAY}
acc = {l: [] for l in LAY}
for p in prompts:
    ids = lab.ids(p)
    H = lab.forward_resid([ids], layers=LAY)[0]  # [5,T,D]
    for i, l in enumerate(LAY):
        acc[l].append(H[i, 1:])
std = {l: torch.cat(acc[l], 0).std(0).to(lab.dev) for l in LAY}
del acc
fnw = rt.final_norm_weight.float(); WU = rt.lm_head_weight
du = WU[tid1(" she")].float() - WU[tid1(" he")].float()
dirs = {"server": {l: (du / fnw.clamp(min=0.1)) @ rt.jpp_lenses[l].float() for l in LAY},
        "grad": {l: (du * fnw) @ rt.jpp_lenses[l].float() for l in LAY}}
gen_r = torch.Generator().manual_seed(11)
R = {l: torch.randn(5376, generator=gen_r).to(lab.dev) for l in LAY}
q10 = {l: torch.quantile(std[l], 0.10) for l in LAY}
q50 = {l: torch.quantile(std[l], 0.50) for l in LAY}
dirs["random"] = R
dirs["random_quiet"] = {l: R[l] * (std[l] < q10[l]).float() for l in LAY}
dirs["random_loud"] = {l: R[l] * (std[l] > q50[l]).float() for l in LAY}
dirs["random_varweighted"] = {l: R[l] * std[l] for l in LAY}
energy = {}
for name, dd in dirs.items():
    energy[name] = {l: float(((dd[l] / dd[l].norm()).pow(2) * (std[l] < q10[l]).float()).sum()) for l in LAY}
res = {"quiet_energy_share": energy,
       "std_quantiles": {l: [float(torch.quantile(std[l], q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)] + [float(std[l].max())] for l in LAY}}
print("quiet-dim energy share (dims with std<10th pct):", {n: {l: round(v, 3) for l, v in e.items()} for n, e in energy.items()})
print("std quantiles 1/10/50/90/99/max:", {l: [round(x, 3) for x in v] for l, v in res["std_quantiles"].items()})
g = lab.gender
SETS = {"B_gemma": C["B_gemma"], "B_atlas": C["B_atlas"], "B_man": C["B_man"]}


def measure():
    o = {}
    for s, ps in SETS.items():
        lg = lab.next_logits_batch(ps)
        lp = F.log_softmax(lg, -1)
        o[s] = {"lse_pron": float((torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)).mean()),
                "pF": float(torch.logsumexp(lp[:, g["pron_f"]], -1).exp().mean()),
                "entropy": float((-(lp.exp() * lp).sum(-1)).mean()),
                "top1": [rt.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()][:3]}
    return o


res["none"] = measure()
for name in ("random", "random_quiet", "random_loud", "random_varweighted"):
    for sc in (2.0, 10.0):
        lab.clear_hooks(); lab.steer_add(LAY, dirs[name], sc, unit=True, start_pos=1)
        res[f"{name}@{sc:g}"] = measure(); lab.clear_hooks()
# tool directions with quiet dims removed
for name in ("grad", "server"):
    dq = {l: dirs[name][l] * (std[l] >= q10[l]).float() for l in LAY}
    for sc in (2.0, 10.0):
        lab.clear_hooks(); lab.steer_add(LAY, dq, sc, unit=True, start_pos=1)
        res[f"{name}_noquiet@{sc:g}"] = measure(); lab.clear_hooks()
for k, v in res.items():
    if k in ("quiet_energy_share", "std_quantiles"):
        continue
    print(k.ljust(24), {s: (round(x["lse_pron"], 2), round(x["pF"], 3), round(x["entropy"], 2), x["top1"][:2]) for s, x in v.items()})
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_12_quiet_dims.partial.json", "w"))
result = {"ok": True}
