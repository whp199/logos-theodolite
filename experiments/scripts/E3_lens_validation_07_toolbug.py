# E3_lens_validation (d) layer_scalar delta_ffn bug across all 60 layers + pole-readout checks.
# PRE-STATED PREDICTIONS:
#  P7 cos(tool delta_ffn, true FFN write) is low wherever s is far from 1 (L0-1, L23-41, L59) and near 1 where s~0.97+;
#     the rescale share dominates the tool vector at L0, L1, L59 and is large (>50%) in the L25-40 band.
#  P8 corrected ffn_dominance_ratio will be LOWER than the tool's at layers with small s (the rescale inflates FFN).
#  P9 post_ffn_norm_weight scaling gives the exact per-neuron DIRECTION (RMSNorm of the whole MLP output is a shared
#     positive scalar per position), so pole token rankings are exact; only magnitudes are off. The eps hack never
#     triggers on real residual states.
import json, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fix.py").read())
from backend.theodolite_math import compute_causal_vector_split, gemma4_rms_norm, extract_bipolar_poles, unembed_gemma4
t0 = time.time()
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
prompts = [lab.ids(PROSE[i][1]) for i in (0, 4, 8, 12, 16, 20)] + \
          [lab.ids(lab.chat(chats[i]["q"], assistant=chats[i]["a"])) for i in (0, 3, 6, 12, 15, 18)]
S = torch.stack([lab.layers[l].layer_scalar.float().reshape(-1)[0] for l in range(60)]).to(lab.dev)
POLE_LAYERS = [2, 10, 20, 30, 40, 50, 58]
wpost = {l: rt._post_ffn_norm_weights[l].float() for l in range(60)}
coln = {}
for l in POLE_LAYERS:
    Wd = rt.get_full_w_down(l).float()  # [D, F]
    coln[l] = (Wd * wpost[l].unsqueeze(1)).norm(dim=0)  # ||w_down_i (.) w_post||, [F]
    del Wd
M = {k: [] for k in ["cos_tool_true_ffn", "share_rescale", "ratio_resc_norm", "cos_tool_resc", "dom_tool_J", "dom_fix_J",
                     "dom_tool_raw", "dom_fix_raw", "fix_matches_decompose", "recon_err", "min_ms_post"]}
POLE = {k: {str(l): [] for l in POLE_LAYERS} for k in ["mlp_identity_relerr", "cos_contrib_dir", "inv_rms_y", "top15_jaccard_act_vs_write",
                                                       "frac_ffn_top15_act", "frac_ffn_top15_write", "ms_pole_vec", "rank_equal_eps"]}
for pi, ids in enumerate(prompts):
    caps = await lab.capture(ids, points=("resid_pre", "resid_mid", "resid_post"))
    acts = await lab.capture(ids, points=("mlp_act",), layers=POLE_LAYERS)
    P = list(range(4, len(ids)))
    rows = {k: [] for k in M}
    for l in range(60):
        pre = caps["resid_pre"][l, P].to(lab.dev).float(); mid = caps["resid_mid"][l, P].to(lab.dev).float()
        post = caps["resid_post"][l, P].to(lab.dev).float(); s = S[l]
        tool_ffn = post - mid; true_ffn = post - s * mid; resc = (s - 1) * mid
        tool_attn = mid - pre; true_attn = s * (mid - pre)
        rows["cos_tool_true_ffn"].append(float(F.cosine_similarity(tool_ffn, true_ffn, dim=-1).mean()))
        rows["share_rescale"].append(float(((tool_ffn * resc).sum(-1) / tool_ffn.pow(2).sum(-1)).mean()))
        rows["ratio_resc_norm"].append(float((resc.norm(dim=-1) / tool_ffn.norm(dim=-1)).mean()))
        rows["cos_tool_resc"].append(float(F.cosine_similarity(tool_ffn, resc, dim=-1).mean()))
        J = lab.J[l]
        old = compute_causal_vector_split(caps["resid_pre"][l, P].to(lab.dev), caps["resid_mid"][l, P].to(lab.dev), caps["resid_post"][l, P].to(lab.dev), J)
        new = compute_causal_vector_split_fixed(caps["resid_pre"][l, P].to(lab.dev), caps["resid_mid"][l, P].to(lab.dev), caps["resid_post"][l, P].to(lab.dev), J, layer_scalar=s)
        rows["dom_tool_J"].append(float(old.ffn_dominance_ratio.float().mean())); rows["dom_fix_J"].append(float(new.ffn_dominance_ratio.float().mean()))
        rows["dom_tool_raw"].append(float((tool_ffn.norm(dim=-1) / (tool_ffn.norm(dim=-1) + tool_attn.norm(dim=-1))).mean()))
        rows["dom_fix_raw"].append(float((true_ffn.norm(dim=-1) / (true_ffn.norm(dim=-1) + true_attn.norm(dim=-1))).mean()))
        ref = (true_ffn.to(torch.bfloat16) @ J.T).float()
        rows["fix_matches_decompose"].append(float(F.cosine_similarity(new.delta_ffn_J.float(), ref, dim=-1).mean()))
        rows["recon_err"].append(float(((pre + true_attn + true_ffn + (s - 1) * pre - post).norm(dim=-1) / post.norm(dim=-1)).max()))
        rows["min_ms_post"].append(float(post.pow(2).mean(-1).min()))
        if l in POLE_LAYERS:
            a = acts["mlp_act"][POLE_LAYERS.index(l), P].to(lab.dev).float()  # [P, F]
            Wd = rt.get_full_w_down(l)
            y = (a.to(torch.bfloat16) @ Wd.T).float()  # MLP output before post-norm
            inv_rms = torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
            pred = s * y * inv_rms * wpost[l]
            POLE["mlp_identity_relerr"][str(l)].append(float(((pred - true_ffn).norm(dim=-1) / true_ffn.norm(dim=-1)).mean()))
            POLE["inv_rms_y"][str(l)].append(inv_rms.squeeze(-1).cpu().tolist())
            # per-neuron exact contribution c_i = s * a_i * inv_rms * (w_i (.) w_post); tool pole vector = sign(a_i) * (w_i (.) w_post)
            p_last = -1
            av = a[p_last]; idx_act = torch.topk(av.abs(), 15).indices
            wr = av.abs() * coln[l]; idx_wr = torch.topk(wr, 15).indices
            sa, sw = set(idx_act.tolist()), set(idx_wr.tolist())
            POLE["top15_jaccard_act_vs_write"][str(l)].append(len(sa & sw) / len(sa | sw))
            Wcols_a = Wd[:, idx_act].float().T * wpost[l]; Wcols_w = Wd[:, idx_wr].float().T * wpost[l]
            tf = true_ffn[p_last]
            c_act = (s * inv_rms[p_last] * av[idx_act].unsqueeze(-1) * Wcols_a).sum(0)
            c_wr = (s * inv_rms[p_last] * av[idx_wr].unsqueeze(-1) * Wcols_w).sum(0)
            POLE["frac_ffn_top15_act"][str(l)].append(float((c_act @ tf) / (tf @ tf)))
            POLE["frac_ffn_top15_write"][str(l)].append(float((c_wr @ tf) / (tf @ tf)))
            # direction check: exact contribution vs tool's polarity*(w (.) w_post)
            exact = s * inv_rms[p_last] * av[idx_act].unsqueeze(-1) * Wcols_a
            toolv = torch.sign(av[idx_act]).unsqueeze(-1) * Wcols_a
            POLE["cos_contrib_dir"][str(l)].append(float(F.cosine_similarity(exact, toolv, dim=-1).min()))
            # eps hack: ms of J++-mapped pole vectors and rank equality under eps 1e-6 vs 1e-12
            pv = (toolv.to(torch.bfloat16) @ J.T)
            POLE["ms_pole_vec"][str(l)].append(float(pv.float().pow(2).mean(-1).median()))
            lg_hack = unembed_gemma4(pv, rt.final_norm_weight, rt.lm_head_weight)
            x = pv.float(); ms = x.pow(2).mean(-1, keepdim=True)
            lg_std = torch.tanh(((x * torch.rsqrt(ms + 1e-6) * rt.final_norm_weight.float()).to(torch.bfloat16) @ rt.lm_head_weight.T).float() / 30) * 30
            t_h = torch.topk(lg_hack, 10, -1).indices; t_s = torch.topk(lg_std, 10, -1).indices
            POLE["rank_equal_eps"][str(l)].append(float((t_h == t_s).float().mean()))
            del Wd
    for k in M: M[k].append(rows[k])
    print("prompt", pi, "T", len(ids), round(time.time() - t0, 1), flush=True)
out = {"layer_scalar": S.cpu().tolist(), "per_prompt": M, "pole": POLE, "n_prompts": len(prompts)}
json.dump(out, open(E3_RES + "E3_lens_validation_d_toolbug.json", "w"))
import numpy as _np
for k in ["cos_tool_true_ffn", "share_rescale", "dom_tool_J", "dom_fix_J", "fix_matches_decompose", "recon_err", "min_ms_post"]:
    arr = _np.array(M[k]).mean(0)
    print(k, " ".join(f"{x:.2f}" if abs(x) > 1e-3 else f"{x:.1e}" for x in arr))
for k in POLE:
    if k == "inv_rms_y": continue
    print(k, {l: round(float(_np.mean(v)), 4) for l, v in POLE[k].items()})
result = {"ok": 1}
