# E6_tool_audit_10_steer_scan -- follow-up to 06: server defaults (650) and even 65 collapse the model. Find the usable range
# and compare the server's 1/fnw direction with the gradient-correct fnw direction at sane scales; check BOS sensitivity.
# PRE-STATED PREDICTIONS: (a) at scales <= ~10 per layer the text stays coherent; (b) the gradient-correct direction shifts
# she-vs-he log-odds several times more per unit norm than the server direction; (c) skipping BOS (start_pos=1) makes large
# scales less destructive but 650 still breaks the model.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
C = build_conditions()
LAY = [29, 35, 41, 47, 53]
fnw = rt.final_norm_weight.float(); WU = rt.lm_head_weight
du = WU[tid1(" she")].float() - WU[tid1(" he")].float()
inv = 1.0 / fnw.clamp(min=0.1)
dirs = {"server": {l: (du * inv) @ rt.jpp_lenses[l].float() for l in LAY},
        "grad": {l: (du * fnw) @ rt.jpp_lenses[l].float() for l in LAY}}
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
                "top1": [rt.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()][:4]}
    return o


res = {"none": measure()}
for dname in ("server", "grad"):
    for sc in (2.0, 5.0, 10.0, 20.0, 40.0):
        lab.clear_hooks(); lab.steer_add(LAY, dirs[dname], sc, unit=True, start_pos=1)
        res[f"{dname}@{sc:g}_noBOS"] = measure(); lab.clear_hooks()
for dname, sc in (("server", 650.0), ("server", 65.0), ("grad", 10.0), ("grad", 20.0)):
    lab.clear_hooks(); lab.steer_add(LAY, dirs[dname], sc, unit=True, start_pos=0)
    res[f"{dname}@{sc:g}_withBOS"] = measure(); lab.clear_hooks()
lab.clear_hooks(); lab.steer_add(LAY, dirs["server"], 650.0, unit=True, start_pos=1)
res["server@650_noBOS"] = measure(); lab.clear_hooks()
for k, v in res.items():
    print(k.ljust(22), {s: (round(x["lse_pron"], 2), round(x["pF"], 3), round(x["pM"], 3), round(x["entropy"], 2), x["top1"][:2]) for s, x in v.items()})
gens = {}
for name, dname, sc in (("none", None, 0), ("grad@10_noBOS", "grad", 10.0), ("grad@20_noBOS", "grad", 20.0), ("server@20_noBOS", "server", 20.0)):
    lab.clear_hooks()
    if dname:
        lab.steer_add(LAY, dirs[dname], sc, unit=True, start_pos=1)
    gens[name] = [x[0] for x in lab.generate_batch([C["B_gemma"][i] for i in (0, 4, 8)], max_new_tokens=16)]
    lab.clear_hooks()
    print(name, gens[name])
res["generations"] = gens
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_10_steer_scan.partial.json", "w"))
result = {"ok": True}
