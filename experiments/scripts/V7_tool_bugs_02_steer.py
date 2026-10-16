# V7_tool_bugs_02_steer: (c) server steering default (scale 650 at L29/35/41/47/53) vs residual norms; effect on generations.
# Uses backend.server._build_steering_spec_from_req (the server's own code) and rt.eager_model.capture_generation
# (the exact call run_theodolite_sweep makes with a steering_spec).
# PRE-REGISTERED PREDICTIONS
#  S1 650 exceeds the median non-BOS residual norm at every one of the 5 layers (ratio > 1.5) and exceeds the BOS norm.
#  S2 At scale 650 the greedy generations are degenerate (single repeated token / special tokens) for >= 5/6 prompts,
#     for the server direction AND for a random unit direction (i.e. it's the magnitude, not the semantics).
#  S3 The server direction is built with 1/fnw clamped at 0.1: the 3 negative-fnw dims get weight 10 (the max).
#     Prediction: these + other low-fnw dims carry a large share of u_dir energy, and cos(server u_dir, fnw*dW_U) is low.
#  S4 Dose-response: the server direction degrades text at a scale where it has NOT yet shifted the target/contrast
#     log-odds in the intended direction by > 2 nats (no usable window), unlike a quiet-dim-masked direction.
#  Refutation of C8(c): coherent, on-target generations at 650, or 650 << ||h||.
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
from interp_engine import Address
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec
from backend.server import _build_steering_spec_from_req, SteeringLaserRequest
from backend.theodolite_math import unembed_gemma4
LAYERS = [29, 35, 41, 47, 53]
out = {}
fnw = rt.final_norm_weight.float()

# ---- directions
def spec_vectors(spec):
    return {l: torch.as_tensor(ls.operations[0].vector).float() for l, ls in spec.layers.items()}

req_sh = SteeringLaserRequest(enabled=True, target_concept="she", contrast_concept="he")
req_def = SteeringLaserRequest(enabled=True)   # server defaults: six vs eight, 650, L29..53
out["server_defaults"] = {"target": req_def.target_concept, "contrast": req_def.contrast_concept, "layers": req_def.layers,
                          "scale": req_def.scale}
spec_sh = _build_steering_spec_from_req(rt, req_sh)
spec_def = _build_steering_spec_from_req(rt, req_def)
V_sh = spec_vectors(spec_sh); V_def = spec_vectors(spec_def)
out["spec_vector_norms"] = {l: float(v.norm()) for l, v in V_sh.items()}
out["spec_scale"] = float(spec_sh.layers[29].operations[0].scale)

# S3: 1/fnw clamp analysis
inv = 1.0 / fnw.clamp(min=0.1)
u_srv = (rt.lm_head_weight[tid1(" she")].float() - rt.lm_head_weight[tid1(" he")].float()) * inv
u_ok = (rt.lm_head_weight[tid1(" she")].float() - rt.lm_head_weight[tid1(" he")].float()) * fnw
low = fnw < 0.1
lowish = fnw < 1.0
out["inv_fnw"] = {"n_clamped": int(low.sum()), "inv_weight_clamped": float(inv[low].max()) if low.any() else None,
                  "inv_weight_median": float(inv.median()),
                  "u_srv_energy_share_clamped_dims": float(u_srv[low].pow(2).sum() / u_srv.pow(2).sum()),
                  "u_srv_energy_share_fnw_lt1": float(u_srv[lowish].pow(2).sum() / u_srv.pow(2).sum()),
                  "n_fnw_lt1": int(lowish.sum()),
                  "cos_final_space_srv_vs_fnw": float(F.cosine_similarity(u_srv, u_ok, dim=0))}
# logit effect per unit norm in final space (first-order, ignoring norm): <fnw*dW, v>/||v|| after RMSnorm-scale
# pulled-back comparison per layer
cmp = {}
for l in LAYERS:
    J = rt.jpp_lenses[l].float()
    v_srv = u_srv @ J
    v_ok = u_ok @ J
    cmp[l] = {"cos_pullback_srv_vs_fnw": float(F.cosine_similarity(v_srv, v_ok, dim=0)),
              "cos_spec_vs_mine": float(F.cosine_similarity(v_srv.cpu(), V_sh[l], dim=0))}
    del J
out["dir_compare"] = cmp

# ---- prompts
prompts = {"demo_raw": RAW_PROMPTS[0], "nurse_raw": RAW_PROMPTS[3], "gemma_friend_raw": RAW_PROMPTS[1]}
cp = chat_prompts()
prompts.update({"gemma_self_chat": cp[0], "neighbor_chat": cp[3], "doctor_chat": cp[4]})
ids_of = {k: lab.ids(v) for k, v in prompts.items()}
PSET = globals().get("PSET", None)

# ---- residual norms at the 5 layers (non-BOS positions, and BOS)
norms = {l: [] for l in LAYERS}; bosn = {l: [] for l in LAYERS}
for k, ids in ids_of.items():
    caps = await lab.capture(ids, ("resid_post",), LAYERS)
    for i, l in enumerate(LAYERS):
        n = caps["resid_post"][i].float().norm(dim=-1)
        norms[l] += n[1:].tolist(); bosn[l].append(float(n[0]))
out["resid_norms"] = {l: {"median_nonBOS": float(np.median(norms[l])), "p10": float(np.percentile(norms[l], 10)),
                          "p90": float(np.percentile(norms[l], 90)), "BOS": float(np.mean(bosn[l])),
                          "ratio_650_over_median": 650.0 / float(np.median(norms[l]))} for l in LAYERS}
print("norms", {l: round(v["median_nonBOS"], 1) for l, v in out["resid_norms"].items()})

# ---- generations
def mk_spec(V, scale):
    return SteeringSpec(layers={l: LayerSteeringSpec(operations=[AddSpec(vector=V[l], scale=float(scale), normalize=True)])
                                for l in LAYERS}, point="resid_post")

g = torch.Generator().manual_seed(7)
V_rand = {l: torch.randn(5376, generator=g) for l in LAYERS}
# quiet-dim-masked fnw-correct pullback (E6's proposed fix), computed from natural per-dim std on these prompts
stds = {}
for k, ids in list(ids_of.items())[:3]:
    caps = await lab.capture(ids, ("resid_post",), LAYERS)
    for i, l in enumerate(LAYERS):
        stds.setdefault(l, []).append(caps["resid_post"][i, 1:].float())
V_qm = {}
for l in LAYERS:
    sd = torch.cat(stds[l], 0).std(0)
    v = (u_ok @ rt.jpp_lenses[l].float()).cpu()
    v = v * (sd > sd.quantile(0.10)).float()
    V_qm[l] = v
DIRS = {"server_she_he": (V_sh, " she", " he", [0.0, 2.0, 6.5, 20.0, 65.0, 650.0]),
        "server_default_six_eight": (V_def, " six", " eight", [6.5, 65.0, 650.0]),
        "random": (V_rand, " she", " he", [6.5, 65.0, 650.0]),
        "quietmasked_fnw_she_he": (V_qm, " she", " he", [2.0, 6.5, 20.0, 65.0])}
gens = []
t0 = time.time()
for k, ids in ids_of.items():
    if PSET is not None and k not in PSET:
        continue
    for dname, (V, tpos, tneg, SCALES) in DIRS.items():
        for sc in SCALES:
            if sc == 0.0 and dname != "server_she_he":
                continue
            spec = mk_spec(V, sc) if sc > 0 else None
            comp, caps = await rt.eager_model.capture_generation(ids, [Address("resid_post", 59)], max_tokens=12,
                                                                temperature=0.0, steering_spec=spec)
            h = caps[Address("resid_post", 59)][len(ids) - 1].to(lab.dev)
            lg = unembed_gemma4(h, rt.final_norm_weight, rt.lm_head_weight).float()
            lp = F.log_softmax(lg, -1)
            ent = float(-(lp.exp() * lp).sum())
            toks = [int(t) for t in comp.token_ids]
            gens.append({"prompt": k, "dir": dname if sc > 0 else "none", "scale": sc, "text": comp.text,
                         "n_distinct_tok": len(set(toks)), "n_tok": len(toks),
                         "entropy1": ent, "top1": rt.tokenizer.decode([int(lg.argmax())]),
                         "lo_target": float(lg[tid1(tpos)] - lg[tid1(tneg)]),
                         "lo_she_he": float(lg[tid1(" she")] - lg[tid1(" he")]),
                         "lo_six_eight": float(lg[tid1(" six")] - lg[tid1(" eight")]), "_lp": None})
            if sc == 0.0:
                base_lp = lp
            gens[-1]["kl_vs_base"] = float((lp.exp() * (lp - base_lp)).sum()) if sc > 0 else 0.0
            gens[-1].pop("_lp")
    print(f"{k} done {time.time()-t0:.0f}s")
out["gens"] = gens
json.dump(out, open(RES + f"V7_tool_bugs_02_steer_{globals().get('PTAG','all')}.partial.json", "w"))
for r in gens:
    if r["prompt"] in ("demo_raw", "gemma_self_chat"):
        print(r["prompt"], r["dir"], r["scale"], repr(r["text"][:50]), round(r["entropy1"], 2), round(r["lo_target"], 2), r["n_distinct_tok"])
result = {k: v for k, v in out.items() if k != "gens"}
