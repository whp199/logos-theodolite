# E6_tool_audit_05_filaments -- are Module-3 "verified live circuit" filaments causal? + Module 2 pole checks + CKA/BOS.
# PRE-STATED PREDICTIONS:
#  F1 Coupling scores of tool filaments are NOT distinguishable from couplings of random (inactive) source neurons to the
#     same destination neurons (score dominated by |act_dst| and weight norms).
#  F2 Ablating a filament's source neuron (at the target position) changes the destination neuron's gate pre-activation by a
#     tiny amount; Spearman(coupling, measured effect) is weak (|rho| < 0.3); sign agreement near chance.
#  F3 Pole J-VINDEX logits sit near the softcap (25-30) for every pole (normalization artifact), and ranking poles by |act|
#     disagrees substantially with ranking by actual write norm |act|*||W_down*post_w||.
#  F4 Linear CKA changes when BOS (and the duplicate BOS) are excluded.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
from types import SimpleNamespace


def spearmanr(x, y, nperm=2000):
    x = torch.tensor(list(map(float, x))); y = torch.tensor(list(map(float, y)))
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    def r(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    obs = r(rx, ry)
    g = torch.Generator().manual_seed(0)
    cnt = sum(1 for _ in range(nperm) if abs(r(rx, ry[torch.randperm(len(ry), generator=g)])) >= abs(obs))
    return SimpleNamespace(correlation=obs, pvalue=(cnt + 1) / (nperm + 1))
C = build_conditions()
res = {}


class _Stop(Exception):
    pass


def run_with(ids, pos, src=None, dst_specs=(), stop_layer=None):
    """Forward [ids]; optionally zero mlp act of neuron src=(l,n) at position pos; capture gate pre-act and act of
    dst neurons [(l,n),...] at pos. Stops the forward after stop_layer's MLP."""
    store = {}
    hs = []
    try:
        if src is not None:
            ls_, ns_ = src
            def zpre(mod, args):
                a = args[0].clone()
                a[:, pos, ns_] = 0
                return (a,) + tuple(args[1:])
            hs.append(rt.get_mlp_module(ls_).down_proj.register_forward_pre_hook(zpre))
        by_layer = {}
        for (l, n) in dst_specs:
            by_layer.setdefault(l, []).append(n)
        for l, nlist in by_layer.items():
            idx = torch.tensor(nlist, device=lab.dev)
            def gate_hook(mod, args, out, l=l, idx=idx):
                store[("g", l)] = out[0, pos].index_select(0, idx).float().cpu()
            def act_pre(mod, args, l=l, idx=idx):
                store[("a", l)] = args[0][0, pos].index_select(0, idx).float().cpu()
                if stop_layer is not None and l == stop_layer:
                    raise _Stop()
            hs.append(rt.get_mlp_module(l).gate_proj.register_forward_hook(gate_hook))
            hs.append(rt.get_mlp_module(l).down_proj.register_forward_pre_hook(act_pre))
        try:
            lab.hf(input_ids=torch.tensor([ids], device=lab.dev), use_cache=False)
        except _Stop:
            pass
    finally:
        for h in hs:
            h.remove()
    out = {}
    for (l, n) in dst_specs:
        j = by_layer[l].index(n)
        out[(l, n)] = (float(store[("g", l)][j]), float(store[("a", l)][j]))
    return out


fil_res = []
for pi in (0, 3):
    prompt = C["A_default"][pi]
    ids = lab.ids(prompt)
    b = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=0,
                                      tracked_concepts=["she", "he"], prepend_bos=False)
    pos = b.target_pos
    fils = sorted(b.filaments, key=lambda f: -abs(f.coupling_score))
    # stratify: take top-3 filaments per source-layer band so that not all are L0->L6
    chosen, seen = [], {}
    for f in fils:
        band = f.src_layer // 10
        if seen.get(band, 0) < 4:
            chosen.append(f); seen[band] = seen.get(band, 0) + 1
    dst_specs = sorted(set((f.dst_layer, f.dst_neuron) for f in chosen))
    base = run_with(ids, pos, None, dst_specs, stop_layer=None)
    # null couplings: random source neurons at same source layer, same destination
    for f in chosen:
        ls_, ld = f.src_layer, f.dst_layer
        rnd = torch.randint(0, 21504, (200,), generator=torch.Generator().manual_seed(f.src_neuron)).to(lab.dev)
        wd = rt.slice_w_down(ls_, rnd).float() * rt._post_ffn_norm_weights[ls_].float()
        gt = rt.slice_w_gate(ld, torch.tensor([f.dst_neuron], device=lab.dev)).float()[0] * rt._pre_ffn_norm_weights[ld].float()
        null_c = (wd @ gt).abs() * f.dst_abs_act
        pct = float((null_c < abs(f.coupling_score)).float().mean())
        # causal: ablate source neuron at pos
        eff = run_with(ids, pos, (ls_, f.src_neuron), [(ld, f.dst_neuron)], stop_layer=ld)[(ld, f.dst_neuron)]
        b_g, b_a = base[(ld, f.dst_neuron)]
        # random-source ablation control (3 random neurons at same source layer)
        rnd_eff = []
        for rn in rnd[:3].tolist():
            e2 = run_with(ids, pos, (ls_, rn), [(ld, f.dst_neuron)], stop_layer=ld)[(ld, f.dst_neuron)]
            rnd_eff.append(e2[0] - b_g)
        src_act = float(b.bipolar_poles[ls_].signed_activations[f.src_rank])
        fil_res.append({"prompt": pi, "src": [ls_, f.src_neuron, f.src_polarity], "dst": [ld, f.dst_neuron, f.dst_polarity],
                        "coupling": f.coupling_score, "dot": f.dot_product, "dst_abs_act": f.dst_abs_act, "src_act": src_act,
                        "null_pct": pct, "null_c_median": float(null_c.median()),
                        "base_gate": b_g, "base_act": b_a, "d_gate": eff[0] - b_g, "d_act": eff[1] - b_a,
                        "rel_d_gate": (eff[0] - b_g) / (abs(b_g) + 1e-6), "rand_src_d_gate": rnd_eff})
    print(f"prompt {pi}: {len(b.filaments)} filaments, tested {len(chosen)}")
    torch.cuda.empty_cache()
res["filaments"] = fil_res
cs = [r["coupling"] for r in fil_res]
dg = [r["d_gate"] for r in fil_res]
rho = spearmanr(cs, dg)
# prediction: ablating source removes +sigma*W_down*a; predicted sign of d_gate = -sign(dot)*sign(src_act)*sign(polarity)=-sign(dot)
pred_sign = [-1 if r["dot"] > 0 else 1 for r in fil_res]
agree = sum(1 for p, d in zip(pred_sign, dg) if (d > 0) == (p > 0)) / len(dg)
res["filament_summary"] = {"n": len(fil_res), "spearman_coupling_vs_dgate": [float(rho.correlation), float(rho.pvalue)],
                           "sign_agreement": agree,
                           "median_abs_rel_d_gate": float(np.median([abs(r["rel_d_gate"]) for r in fil_res])),
                           "median_null_pct": float(np.median([r["null_pct"] for r in fil_res])),
                           "median_abs_d_gate": float(np.median([abs(x) for x in dg])),
                           "median_abs_rand_src_d_gate": float(np.median([abs(x) for r in fil_res for x in r["rand_src_d_gate"]]))}
print("filament summary:", res["filament_summary"])

# ---------------- Module 2 checks on the last bundle (b)
m2 = {}
for l in (10, 30, 35, 45, 55, 58):
    P = b.bipolar_poles[l]
    v = P.jpp_pole_vectors.float()
    ms_ = v.pow(2).mean(-1)
    lg_hack = unembed_gemma4(v, rt.final_norm_weight, rt.lm_head_weight)              # eps=1e-12 hack
    # unembed_gemma4 passes eps but gemma4_rms_norm overrides with 1e-12 when ms<1e-3 -> emulate the honest eps manually
    xn = v * torch.rsqrt(ms_.unsqueeze(-1) + 1e-6) * rt.final_norm_weight.float()
    lg_honest = torch.tanh((xn.to(torch.bfloat16) @ rt.lm_head_weight.T).float() / 30) * 30
    top_same = [bool(torch.equal(torch.topk(lg_hack[i], 3).indices, torch.topk(lg_honest[i], 3).indices)) for i in range(v.shape[0])]
    # rank by |act| vs by actual write norm
    m2[l] = {"pole_vec_ms": ms_.cpu().tolist()[:5], "max_logit_hack": lg_hack.max(-1).values.cpu().tolist()[:5],
             "max_logit_honest_eps": lg_honest.max(-1).values.cpu().tolist()[:5], "top3_same_frac": sum(top_same) / len(top_same)}
res["m2_eps"] = m2
print("m2 eps:", {l: (round(v_['top3_same_frac'], 2), [round(x, 1) for x in v_['max_logit_hack'][:3]], [round(x, 2) for x in v_['max_logit_honest_eps'][:3]]) for l, v_ in m2.items()})

# |act| ranking vs write-norm ranking (needs full mlp_act at the target)
caps = await lab.capture(ids, points=("mlp_act",), layers=[10, 30, 45, 55, 58])
rank_res = {}
for i, l in enumerate([10, 30, 45, 55, 58]):
    a = caps["mlp_act"][i, pos].float().to(lab.dev)
    top200 = torch.topk(a.abs(), 200).indices
    wn = (rt.slice_w_down(l, top200).float() * rt._post_ffn_norm_weights[l].float()).norm(dim=-1)
    contrib = a[top200].abs() * wn
    rho2 = spearmanr(a[top200].abs().cpu().numpy(), contrib.cpu().numpy())
    top15_act = set(top200[:15].tolist())
    top15_con = set(top200[torch.topk(contrib, 15).indices].tolist())
    rank_res[l] = {"spearman_absact_vs_contrib_top200": float(rho2.correlation), "top15_overlap": len(top15_act & top15_con),
                   "wnorm_cv": float(wn.std() / wn.mean())}
res["m2_rank"] = rank_res
print("m2 rank:", rank_res)

# ---------------- CKA with/without BOS, single vs double BOS
cka_res = []
for pi in range(6):
    prompt = C["A_default"][pi]
    ids1 = lab.ids(prompt); ids2 = [2] + ids1
    for tag, ids_ in (("single", ids1), ("double", ids2)):
        X = lab.forward_resid([ids_])[0]  # [60,T,D] float32 CPU
        full = compute_linear_cka_matrix(X.to(lab.dev))
        nb = 1 if tag == "single" else 2
        nob = compute_linear_cka_matrix(X[:, nb:].to(lab.dev))
        offm = ~torch.eye(60, dtype=torch.bool, device=lab.dev)
        cka_res.append({"prompt": pi, "tok": tag, "offdiag_full": float(full[offm].mean()), "offdiag_noBOS": float(nob[offm].mean()),
                        "max_abs_diff": float((full - nob).abs().max()), "L10_L50_full": float(full[10, 50]), "L10_L50_noBOS": float(nob[10, 50])})
        del X
res["cka"] = cka_res
print("cka:", [(r["prompt"], r["tok"], round(r["offdiag_full"], 3), round(r["offdiag_noBOS"], 3), round(r["max_abs_diff"], 3)) for r in cka_res])
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_05_filaments.partial.json", "w"), default=str)
result = {"filament_summary": res["filament_summary"]}
