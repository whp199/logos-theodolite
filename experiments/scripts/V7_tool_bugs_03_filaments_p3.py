PIDX = 3
# V7_tool_bugs_03_filaments: (d) Are Module-3 filaments causal? Full candidate-matrix ablation at scale.
# Set PIDX (index into FIL_PROMPTS) before exec'ing.
# Method: run the tool's own sweep (rt.run_theodolite_sweep, max_new_tokens=0, single BOS) -> 15 poles/layer and the
# tool's filaments. Then zero-ablate EVERY pole (60 x 15 = 900 sources) at the target position, one source per batch
# row (B=32, 2 clean rows per batch as a numerical-noise floor), and record the change in gate pre-activation g_j,
# up pre-activation u_j and post-GeGLU activation a_j of EVERY pole j in the next 6 layers (the tool's full
# candidate set, 15x15x6 pairs per source layer). Static scores per pair:
#   C_tool  = <sig_i W_down_i (.) w_post_l , W_gate_j (.) w_pre_l'> * |a_j|        (tool formula; checked vs filaments)
#   D_gate  = direct-path prediction of the ablation's Delta g_j: -a_i * s_l * prod_{l<k<l'} s_k * J_post(y_l) W_down_i,
#             then exact pre-FFN RMSNorm Jacobian at mid_l', dotted with W_gate_j  (uses actual a_i, norms, scalars)
#   D_act   = gelu(g+Dg)(u+Du) - gelu(g)u with Dg, Du from the same direct path (uses W_up and actual pre-acts)
# PRE-REGISTERED PREDICTIONS (pooled over >= 6 prompts, >= 150 tool filaments):
#  P1 Spearman(|C_tool|, |Delta g_dst|) over tool filaments <= 0.1 (E6: -0.44); 95% bootstrap CI excludes +0.3.
#  P2 Spearman(|D_gate|, |Delta g|) >= 0.6 and Spearman(|D_act|, |Delta a|) >= 0.6 (direct path dominates single-neuron
#     effects); both CIs exclude the C_tool CI.
#  P3 Among all candidate pairs, the tool's top-6-per-source-layer selection has precision@6 (overlap with the true
#     top-6 by |Delta a|) near chance; D_act selection is much better.
#  Refutation of C8(d) "filament coupling anti-correlated with causal effect": rho(|C_tool|, |effect|) > +0.3.
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
import numpy as _np
cp = chat_prompts()
FIL_PROMPTS = [("raw_demo", RAW_PROMPTS[0], True), ("chat_gemma_self", cp[0], False), ("raw_gemma_friend", RAW_PROMPTS[1], True),
               ("chat_sarah", cp[1], False), ("chat_neighbor", cp[3], False), ("raw_nurse", RAW_PROMPTS[3], True),
               ("chat_doctor", cp[4], False), ("chat_atlas", cp[7], False)]
name, prompt, prepend = FIL_PROMPTS[PIDX]
t0 = time.time()
bd = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=0,
                                   tracked_concepts=["she", "he"], prepend_bos=prepend)
ids = bd.prompt_token_ids
pos = bd.target_pos
assert ids[0] == 2 and ids[1] != 2, ids[:3]
NL, K = 60, 15
pole_idx = torch.stack([bd.bipolar_poles[l].neuron_indices for l in range(NL)]).to(lab.dev)     # [60,15]
pole_act_tool = torch.stack([bd.bipolar_poles[l].signed_activations for l in range(NL)]).float().cpu()  # [60,15]
fils = [(f.src_layer, f.src_rank, f.dst_layer, f.dst_rank, f.coupling_score, f.dot_product) for f in bd.filaments]
del bd
torch.cuda.empty_cache()
mlps = [rt.get_mlp_module(l) for l in range(NL)]
blocks = rt._decoder_layers

# ---------------- clean forward: per-layer y (down_proj out), mid (pre-FFN-norm input), g/u/a for poles
clean = {}
hs = []
try:
    for l in range(NL):
        idx = pole_idx[l]
        def gh(m, a, o, l=l, idx=idx): clean[("g", l)] = o[0, pos].float()
        def uh(m, a, o, l=l, idx=idx): clean[("u", l)] = o[0, pos].float()
        def dpre(m, a, l=l): clean[("a", l)] = a[0][0, pos].float()
        def dout(m, a, o, l=l): clean[("y", l)] = o[0, pos].float()
        def mid(m, a, l=l): clean[("mid", l)] = a[0][0, pos].float()
        hs += [mlps[l].gate_proj.register_forward_hook(gh), mlps[l].up_proj.register_forward_hook(uh),
               mlps[l].down_proj.register_forward_pre_hook(dpre), mlps[l].down_proj.register_forward_hook(dout),
               blocks[l].pre_feedforward_layernorm.register_forward_pre_hook(mid)]
    lab.hf(input_ids=torch.tensor([ids], device=lab.dev), use_cache=False)
finally:
    for h in hs:
        h.remove()
s = torch.tensor([float(blocks[l].layer_scalar.float().reshape(-1)[0]) for l in range(NL)], device=lab.dev)
a_src = torch.stack([clean[("a", l)][pole_idx[l]] for l in range(NL)])   # [60,15] signed acts
check_act = float((a_src.cpu() - pole_act_tool).abs().max())

# ---------------- static scores for all candidate pairs
def rms(v):
    return torch.sqrt(v.pow(2).mean() + 1e-6)

def jnorm(x, w, v):
    """Jacobian of RMSNorm(x)*w applied to v (rows of v: [n, D])."""
    r = rms(x)
    D = x.numel()
    proj = (v @ x) / (D * r * r)            # [n]
    return (v - proj[:, None] * x[None, :]) * w[None, :] / r

gelu = lambda z: F.gelu(z, approximate="tanh")
wpost = {l: rt._post_ffn_norm_weights[l].float() for l in range(NL)}
wpre = {l: rt._pre_ffn_norm_weights[l].float() for l in range(NL)}
Wd = {l: rt.slice_w_down(l, pole_idx[l]).float() for l in range(NL)}     # [15,D]
Wg = {l: rt.slice_w_gate(l, pole_idx[l]).float() for l in range(NL)}
Wu = {l: rt.slice_w_up(l, pole_idx[l]).float() for l in range(NL)}
g_cl = torch.stack([clean[("g", l)][pole_idx[l]] for l in range(NL)])   # [60,15]
u_cl = torch.stack([clean[("u", l)][pole_idx[l]] for l in range(NL)])
pol = torch.sign(a_src); pol[pol == 0] = 1
static = {}   # (l, lp) -> dict of [15,15] tensors
for l in range(NL - 1):
    dpost = s[l] * jnorm(clean[("y", l)], wpost[l], -a_src[l][:, None] * Wd[l])     # [15,D] Delta resid_post_l
    for lp in range(l + 1, min(l + 7, NL)):
        carry = torch.prod(s[l + 1:lp]) if lp > l + 1 else torch.tensor(1.0, device=lab.dev)
        dmid = carry * dpost
        dxh = jnorm(clean[("mid", lp)], wpre[lp], dmid)                               # [15,D]
        dg = dxh @ Wg[lp].T                                                          # [15src,15dst]
        du = dxh @ Wu[lp].T
        g0, u0 = g_cl[lp][None, :], u_cl[lp][None, :]
        dact = gelu(g0 + dg) * (u0 + du) - gelu(g0) * u0
        dot = (pol[l][:, None] * Wd[l] * wpost[l]) @ (Wg[lp] * wpre[lp]).T          # tool dot
        ctool = dot * a_src[lp].abs()[None, :]
        static[(l, lp)] = {"C_tool": ctool, "dot": dot, "D_gate": dg, "D_up": du, "D_act": dact}

# ---------------- batched ablations: one source per row
B = 32
srcs = [(l, k) for l in range(NL - 1) for k in range(K)]
eff_g = {}; eff_a = {}; eff_u = {}
noise_g, noise_a = [], []
for b0 in range(0, len(srcs), B - 2):
    chunk = srcs[b0:b0 + B - 2]
    nb = len(chunk) + 2
    rows_by_layer = {}
    for r, (l, k) in enumerate(chunk):
        rows_by_layer.setdefault(l, []).append((r + 2, int(pole_idx[l, k])))
    meas = {}
    hs = []
    try:
        for l in range(NL):
            idx = pole_idx[l]
            def gh(m, a, o, l=l, idx=idx): meas[("g", l)] = o[:, pos].index_select(-1, idx).float()
            def uh(m, a, o, l=l, idx=idx): meas[("u", l)] = o[:, pos].index_select(-1, idx).float()
            def dpre(m, a, l=l, idx=idx): meas[("a", l)] = a[0][:, pos].index_select(-1, idx).float()
            hs += [mlps[l].gate_proj.register_forward_hook(gh), mlps[l].up_proj.register_forward_hook(uh),
                   mlps[l].down_proj.register_forward_pre_hook(dpre)]
            if l in rows_by_layer:
                rr = torch.tensor([r for r, _ in rows_by_layer[l]], device=lab.dev)
                nn_ = torch.tensor([n for _, n in rows_by_layer[l]], device=lab.dev)
                def zpre(m, a, rr=rr, nn_=nn_):
                    x = a[0].clone()
                    x[rr, pos, nn_] = 0
                    return (x,) + tuple(a[1:])
                hs.append(mlps[l].down_proj.register_forward_pre_hook(zpre))
        lab.hf(input_ids=torch.tensor([ids] * nb, device=lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    for l in range(NL):
        G, U, A = meas[("g", l)], meas[("u", l)], meas[("a", l)]
        g0 = 0.5 * (G[0] + G[1]); a0 = 0.5 * (A[0] + A[1]); u0 = 0.5 * (U[0] + U[1])
        if l > 0:
            noise_g.append((G[0] - G[1]).abs().max().item()); noise_a.append((A[0] - A[1]).abs().max().item())
        for r, (ls_, k) in enumerate(chunk):
            if ls_ < l <= ls_ + 6:
                eff_g[(ls_, k, l)] = (G[r + 2] - g0)
                eff_a[(ls_, k, l)] = (A[r + 2] - a0)
                eff_u[(ls_, k, l)] = (U[r + 2] - u0)
    del meas
print(f"{name}: ablations done {time.time()-t0:.0f}s; act check {check_act:.3g}; noise g {max(noise_g):.3g} a {max(noise_a):.3g}")

# ---------------- assemble per-pair arrays (vectorized per (l, lp) block)
recs = {k: [] for k in ("l", "k", "lp", "j", "C_tool", "dot", "D_gate", "D_up", "D_act", "dg", "du", "da", "a_src", "g0", "u0", "a0", "is_fil", "fil_C")}
filset = {(fl, fr, dl, dr): c for (fl, fr, dl, dr, c, d) in fils}
a_cl_poles = torch.stack([clean[("a", l)][pole_idx[l]] for l in range(NL)]).cpu().numpy()
a_src_np, g_np, u_np = a_src.cpu().numpy(), g_cl.cpu().numpy(), u_cl.cpu().numpy()
KK, JJ = _np.meshgrid(_np.arange(K), _np.arange(K), indexing="ij")
for (l, lp), st in static.items():
    if (l, 0, lp) not in eff_g:
        continue
    EG = torch.stack([eff_g[(l, k, lp)] for k in range(K)]).cpu().numpy()
    EA = torch.stack([eff_a[(l, k, lp)] for k in range(K)]).cpu().numpy()
    EU = torch.stack([eff_u[(l, k, lp)] for k in range(K)]).cpu().numpy()
    n = K * K
    recs["l"].append(_np.full(n, l)); recs["lp"].append(_np.full(n, lp))
    recs["k"].append(KK.ravel()); recs["j"].append(JJ.ravel())
    for nm in ("C_tool", "dot", "D_gate", "D_up", "D_act"):
        recs[nm].append(st[nm].cpu().numpy().ravel())
    recs["dg"].append(EG.ravel()); recs["da"].append(EA.ravel()); recs["du"].append(EU.ravel())
    recs["a_src"].append(_np.repeat(a_src_np[l], K)); recs["g0"].append(_np.tile(g_np[lp], K))
    recs["u0"].append(_np.tile(u_np[lp], K)); recs["a0"].append(_np.tile(a_cl_poles[lp], K))
    fc = _np.array([filset.get((l, int(k), lp, int(j)), _np.nan) for k, j in zip(KK.ravel(), JJ.ravel())])
    recs["is_fil"].append(~_np.isnan(fc)); recs["fil_C"].append(fc)
arr = {k: _np.concatenate(v) for k, v in recs.items()}
_np.savez_compressed(RES + f"V7_tool_bugs_03_fil_{name}.npz", **arr, pole_idx=pole_idx.cpu().numpy(),
                     noise_g=_np.array(noise_g), noise_a=_np.array(noise_a))
isf = arr["is_fil"]
c_err = float(_np.nanmax(_np.abs(arr["C_tool"][isf] - arr["fil_C"][isf]) / (_np.abs(arr["fil_C"][isf]) + 1e-9))) if isf.any() else None
quick = {"name": name, "n_tokens": len(ids), "pos": pos, "n_fil_tool": len(fils), "n_fil_matched": int(isf.sum()),
         "n_pairs": int(len(arr["l"])), "C_tool_vs_tool_relerr_max": c_err, "act_check": check_act,
         "noise_g_max": max(noise_g), "noise_a_max": max(noise_a),
         "rho_fil_Ctool_dg": spearman(_np.abs(arr["C_tool"][isf]), _np.abs(arr["dg"][isf])),
         "rho_fil_Ctool_da": spearman(_np.abs(arr["C_tool"][isf]), _np.abs(arr["da"][isf])),
         "rho_fil_Dgate_dg": spearman(_np.abs(arr["D_gate"][isf]), _np.abs(arr["dg"][isf])),
         "rho_fil_Dact_da": spearman(_np.abs(arr["D_act"][isf]), _np.abs(arr["da"][isf])),
         "rho_all_Ctool_da": spearman(_np.abs(arr["C_tool"]), _np.abs(arr["da"])),
         "rho_all_Dact_da": spearman(_np.abs(arr["D_act"]), _np.abs(arr["da"])),
         "elapsed": time.time() - t0}
print(json.dumps(quick))
json.dump(quick, open(RES + f"V7_tool_bugs_03_fil_{name}.quick.json", "w"))
result = quick
