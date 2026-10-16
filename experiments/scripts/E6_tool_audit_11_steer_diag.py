# E6_tool_audit_11_steer_diag -- why does even scale 2 (per layer, ~1% of ||h||) along J-pulled-back directions collapse the model?
# PRE-STATED PREDICTIONS: random unit directions at scale 2-10 are harmless (entropy ~ baseline); the J-pullback directions
# are concentrated on a few residual dimensions with huge downstream gain (top-5 dims carry >50% of the norm), and a
# data-driven direction (mean resid difference B_woman - B_man) at the same scale is benign and shifts gender.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
C = build_conditions()
LAY = [29, 35, 41, 47, 53]
fnw = rt.final_norm_weight.float(); WU = rt.lm_head_weight
du = WU[tid1(" she")].float() - WU[tid1(" he")].float()
inv = 1.0 / fnw.clamp(min=0.1)
dirs = {"server": {l: (du * inv) @ rt.jpp_lenses[l].float() for l in LAY},
        "grad": {l: (du * fnw) @ rt.jpp_lenses[l].float() for l in LAY}}
gen_r = torch.Generator().manual_seed(3)
dirs["random"] = {l: torch.randn(5376, generator=gen_r).to(lab.dev) for l in LAY}
# data-driven: mean resid_post difference (B_woman - B_man) at the slot
idsW = [lab.ids(p) for p in C["B_woman"]]; idsM = [lab.ids(p) for p in C["B_man"]]
HW = lab.forward_resid(idsW, layers=LAY, positions=[[len(x) - 1] for x in idsW])[:, :, 0]
HM = lab.forward_resid(idsM, layers=LAY, positions=[[len(x) - 1] for x in idsM])[:, :, 0]
dirs["meandiff"] = {l: (HW[:, i] - HM[:, i]).mean(0).to(lab.dev) for i, l in enumerate(LAY)}
# residual-coordinate statistics at those layers (B_gemma prompts)
idsG = [lab.ids(p) for p in C["B_gemma"]]
conc = {}
for name, dd in dirs.items():
    conc[name] = {}
    for l in LAY:
        v = dd[l] / dd[l].norm()
        sq = v.pow(2)
        top = torch.topk(sq, 5)
        conc[name][l] = {"top5_share": float(top.values.sum()), "top_dims": top.indices.tolist(),
                         "participation_ratio": float(1.0 / sq.pow(2).sum()),
                         "cos_meandiff": float(F.cosine_similarity(v, dirs["meandiff"][l] / dirs["meandiff"][l].norm(), dim=0))}
print("concentration:", {n: {l: (round(c['top5_share'], 3), round(c['participation_ratio'])) for l, c in d.items()} for n, d in conc.items()})
# typical magnitude of residual coordinates at the top dims (B_gemma slot)
H1 = lab.forward_resid(idsG, layers=LAY, positions=[[len(x) - 1] for x in idsG])[:, :, 0]  # [B,5,D]
dimstats = {}
for i, l in enumerate(LAY):
    td = conc["grad"][l]["top_dims"][:3] + conc["server"][l]["top_dims"][:3]
    dimstats[l] = {int(dd): float(H1[:, i, dd].abs().mean()) for dd in td}
    dimstats[l]["median_abs_coord"] = float(H1[:, i].abs().median())
print("dim magnitudes:", dimstats)
g = lab.gender
SETS = {"B_gemma": C["B_gemma"], "B_atlas": C["B_atlas"], "B_man": C["B_man"]}


def measure():
    o = {}
    for s, prompts in SETS.items():
        lg = lab.next_logits_batch(prompts)
        lp = F.log_softmax(lg, -1)
        o[s] = {"lse_pron": float((torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)).mean()),
                "pF": float(torch.logsumexp(lp[:, g["pron_f"]], -1).exp().mean()),
                "pM": float(torch.logsumexp(lp[:, g["pron_m"]], -1).exp().mean()),
                "entropy": float((-(lp.exp() * lp).sum(-1)).mean()),
                "top1": [rt.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()][:3]}
    return o


res = {"concentration": conc, "dimstats": dimstats, "none": measure()}
for name in ("random", "meandiff", "grad", "server"):
    for sc in (0.5, 2.0, 10.0, 40.0):
        lab.clear_hooks(); lab.steer_add(LAY, dirs[name], sc, unit=True, start_pos=1)
        res[f"{name}@{sc:g}"] = measure(); lab.clear_hooks()
# single-layer steering at the latest layer only (L53) with grad direction
for sc in (0.5, 2.0, 10.0):
    lab.clear_hooks(); lab.steer_add([53], dirs["grad"], sc, unit=True, start_pos=1)
    res[f"grad_L53only@{sc:g}"] = measure(); lab.clear_hooks()
    lab.clear_hooks(); lab.steer_add([29], dirs["grad"], sc, unit=True, start_pos=1)
    res[f"grad_L29only@{sc:g}"] = measure(); lab.clear_hooks()
for k, v in res.items():
    if k in ("concentration", "dimstats"):
        continue
    print(k.ljust(20), {s: (round(x["lse_pron"], 2), round(x["pF"], 3), round(x["pM"], 3), round(x["entropy"], 2), x["top1"][:2]) for s, x in v.items()})
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_11_steer_diag.partial.json", "w"))
result = {"ok": True}
