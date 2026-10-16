# E6_tool_audit_06_steering -- server.py "laser" steering defaults applied to the gender question.
# Default: SteeringLaserRequest(scale=650, layers=[29,35,41,47,53]), direction ((W_U[t]-W_U[c]) / clamp(fnw,0.1)) @ J_l.
# PRE-STATED PREDICTIONS:
#  S1 scale 650 per layer x 5 layers is 2.6-7.8x the median residual norm -> text degenerates (random direction at 650
#     degrades it too), so any "she" shift at default scale is not interpretable as a gentle probe.
#  S2 The 1/fnw "preconditioning" points away from the gradient-correct direction (fnw * dW_U) @ J: cos < 0.5, and per unit
#     norm it moves the she-vs-he log-odds less than the gradient-correct direction.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
from backend.server import SteeringLaserRequest, _build_steering_spec_from_req
C = build_conditions()
res = {}
req = SteeringLaserRequest(enabled=True, target_concept="she", contrast_concept="he")
spec650 = _build_steering_spec_from_req(rt, req)
LAY = list(req.layers)
res["defaults"] = {"layers": LAY, "scale": req.scale}
fnw = rt.final_norm_weight.float()
WU = rt.lm_head_weight
du = WU[tid1(" she")].float() - WU[tid1(" he")].float()
inv = 1.0 / fnw.clamp(min=0.1)
geo = {}
dirs_server, dirs_grad = {}, {}
for l in LAY:
    J = rt.jpp_lenses[l].float()
    vs = (du * inv) @ J
    vg = (du * fnw) @ J
    dirs_server[l] = vs; dirs_grad[l] = vg
    vsn = vs / vs.norm(); vgn = vg / vg.norm()
    # lens-linear effect on the she-he effective logit direction per unit added norm
    eff_s = float((vsn @ J.T) @ (du * fnw)); eff_g = float((vgn @ J.T) @ (du * fnw))
    geo[l] = {"cos_server_vs_grad": float(vsn @ vgn), "lens_effect_server": eff_s, "lens_effect_grad": eff_g,
              "spec_vec_matches_server": float(F.cosine_similarity(spec650.layers[l].operations[0].vector.float().to(vs.device), vs, dim=0))}
res["geometry"] = geo
print("geometry:", {l: {k: round(v, 3) for k, v in g.items()} for l, g in geo.items()})
g = lab.gender
SETS = {"B_gemma": C["B_gemma"], "B_woman": C["B_woman"], "B_man": C["B_man"], "A_default": C["A_default"]}


def measure():
    o = {}
    for s, prompts in SETS.items():
        lg = lab.next_logits_batch(prompts)
        lp = F.log_softmax(lg, -1)
        o[s] = {"lse_pron": (torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)).cpu().tolist(),
                "pF": torch.logsumexp(lp[:, g["pron_f"]], -1).exp().cpu().tolist(),
                "pM": torch.logsumexp(lp[:, g["pron_m"]], -1).exp().cpu().tolist(),
                "entropy": (-(lp.exp() * lp).sum(-1)).cpu().tolist(),
                "top1": [rt.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()]}
    return o


gen_rnd = torch.Generator().manual_seed(1)
rnd_dirs = {l: torch.randn(5376, generator=gen_rnd).to(lab.dev) for l in LAY}
configs = {"none": None, "server@650": (dirs_server, 650.0), "server@65": (dirs_server, 65.0),
           "grad@650": (dirs_grad, 650.0), "grad@65": (dirs_grad, 65.0), "random@650": (rnd_dirs, 650.0),
           "server@200": (dirs_server, 200.0), "grad@200": (dirs_grad, 200.0)}
beh = {}
for name, cfg in configs.items():
    lab.clear_hooks()
    if cfg is not None:
        lab.steer_add(LAY, cfg[0], cfg[1], unit=True)
    beh[name] = measure()
    lab.clear_hooks()
    print(name, {s: (round(sum(v["lse_pron"]) / 12, 2), round(sum(v["pF"]) / 12, 3), round(sum(v["pM"]) / 12, 3),
                     round(sum(v["entropy"]) / 12, 2), v["top1"][:4]) for s, v in beh[name].items()})
res["behavior"] = beh
# tool path: generation with the server SteeringSpec through rt.run_theodolite_sweep
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec
spec65 = SteeringSpec(layers={l: LayerSteeringSpec(operations=[AddSpec(vector=dirs_server[l].cpu(), scale=65.0, normalize=True)]) for l in LAY},
                      point="resid_post")
gens = {}
for pi in (0, 4, 8):
    p = C["B_gemma"][pi]
    row = {}
    for name, sp in (("none", None), ("server@650", spec650), ("server@65", spec65)):
        bb = await rt.run_theodolite_sweep(p, monument_concepts=("she", "he", "it"), max_new_tokens=16, tracked_concepts=["she"],
                                           steering_spec=sp, prepend_bos=False)
        row[name] = bb.generated_text
    gens[pi] = row
    print(pi, row)
res["tool_generations"] = gens
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_06_steering.partial.json", "w"))
result = {"ok": True}
