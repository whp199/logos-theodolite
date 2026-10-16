# E6_tool_audit helpers: exec(open(...).read()) inside daemon jobs. Pure analysis helpers; no hooks left behind.
import math, json, random, time
import torch
import torch.nn.functional as F
from backend.theodolite_math import (build_monument_frame, project_trajectory_to_3d, compute_linear_cka_matrix,
                                     unembed_gemma4, gram_schmidt_orthonormalize_3d, N_LAYERS)

BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40),
         "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}


def band_means(arr):
    return {b: float(sum(arr[l] for l in rg) / len(rg)) for b, rg in BANDS.items()}


def boot_ci(xs, n=2000, seed=0):
    xs = [float(x) for x in xs]
    rng = random.Random(seed)
    k = len(xs)
    if k == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    ms = []
    for _ in range(n):
        s = [xs[rng.randrange(k)] for _ in range(k)]
        ms.append(sum(s) / k)
    ms.sort()
    return {"mean": sum(xs) / k, "lo": ms[int(0.025 * n)], "hi": ms[int(0.975 * n) - 1], "n": k}


def tid1(s):
    e = rt.tokenizer.encode(s, add_special_tokens=False)
    assert len(e) == 1, (s, e)
    return e[0]


def corrected_split_J(bundle, l):
    """Corrected attn/ffn write at the target position in J++ space from the bundle's own J-space states."""
    cs = bundle.causal_splits[l]
    s = float(rt._decoder_layers[l].layer_scalar.float().mean())
    pre, mid, post = cs.resid_pre_J.float(), cs.resid_mid_J.float(), cs.resid_post_J.float()
    attn = s * (mid - pre)
    ffn = post - s * mid
    return attn, ffn, s


def summarize_bundle(b, frame_tokens=(" she", " he", " it"), keep_cka=False):
    out = {"target_pos": b.target_pos, "n_tokens": len(b.prompt_token_ids), "first_ids": b.prompt_token_ids[:3],
           "target_tok": b.prompt_token_strs[b.target_pos], "generated": b.generated_text,
           "survey_labels": list(b.survey_labels)}
    tr = b.trajectory_3d
    out["azimuth_deg"] = tr.azimuth_deg.float().cpu().tolist()
    out["elevation_deg"] = tr.elevation_deg.float().cpu().tolist()
    out["proj_norm"] = tr.projection_norm.float().cpu().tolist()
    hJn = [float(b.causal_splits[l].resid_post_J.float().norm()) for l in range(60)]
    out["proj_frac_of_hJ"] = [p / (h + 1e-9) for p, h in zip(out["proj_norm"], hJn)]
    out["xyz"] = tr.xyz.float().cpu().tolist()
    # tracked readouts
    out["jpp_tracked_rank"] = {k: [b.jpp_layer_readouts[l].tracked_token_ranks.get(k) for l in range(60)]
                               for k in b.jpp_layer_readouts[0].tracked_token_ranks}
    out["jpp_tracked_logit"] = {k: [b.jpp_layer_readouts[l].tracked_token_logits.get(k) for l in range(60)]
                                for k in b.jpp_layer_readouts[0].tracked_token_logits}
    out["naive_tracked_rank"] = {k: [b.naive_layer_readouts[l].tracked_token_ranks.get(k) for l in range(60)]
                                 for k in b.naive_layer_readouts[0].tracked_token_ranks}
    out["naive_tracked_logit"] = {k: [b.naive_layer_readouts[l].tracked_token_logits.get(k) for l in range(60)]
                                  for k in b.naive_layer_readouts[0].tracked_token_logits}
    out["jpp_top5"] = {l: list(zip(b.jpp_layer_readouts[l].top_token_strs, [round(p, 3) for p in b.jpp_layer_readouts[l].top_probs]))
                       for l in (10, 20, 30, 35, 40, 45, 50, 55, 59)}
    # module 1 tool vs corrected
    tool_dom, corr_dom, cos_ffn, cos_attn = [], [], [], []
    for l in range(60):
        cs = b.causal_splits[l]
        a_c, f_c, s = corrected_split_J(b, l)
        tool_dom.append(float(cs.ffn_dominance_ratio))
        corr_dom.append(float(f_c.norm() / (a_c.norm() + f_c.norm() + 1e-8)))
        cos_ffn.append(float(F.cosine_similarity(cs.delta_ffn_J.float(), f_c, dim=0)))
        cos_attn.append(float(F.cosine_similarity(cs.delta_attn_J.float(), a_c, dim=0)))
    out["m1_tool_ffn_dominance"] = tool_dom
    out["m1_corrected_ffn_dominance"] = corr_dom
    out["m1_cos_tool_vs_true_ffn"] = cos_ffn
    out["m1_cos_tool_vs_true_attn"] = cos_attn
    # poles labels
    out["pole_labels_jpp"] = {f"{l}:{r}": v for (l, r), v in b.pole_top_tokens_jpp.items()}
    out["pole_labels_naive"] = {f"{l}:{r}": v for (l, r), v in b.pole_top_tokens_naive.items()}
    out["pole_neurons"] = {l: b.bipolar_poles[l].neuron_indices.cpu().tolist() for l in range(60)}
    out["pole_acts"] = {l: b.bipolar_poles[l].signed_activations.cpu().tolist() for l in range(60)}
    out["n_filaments"] = len(b.filaments)
    out["filaments_top"] = [dict(src=(f.src_layer, f.src_neuron, f.src_polarity), dst=(f.dst_layer, f.dst_neuron, f.dst_polarity),
                                 c=f.coupling_score, dot=f.dot_product, act=f.dst_abs_act)
                            for f in sorted(b.filaments, key=lambda f: -abs(f.coupling_score))[:12]]
    C = b.cka_matrix.float().cpu()
    off = C[~torch.eye(60, dtype=torch.bool)]
    out["cka_mean_offdiag"] = float(off.mean())
    out["cka_adjacent_mean"] = float(torch.diagonal(C, 1).mean())
    out["cka_L10_vs_L50"] = float(C[10, 50])
    if keep_cka:
        out["cka"] = C.tolist()
    out["elapsed_ms"] = b.elapsed_ms
    return out


GENDER_GROUPS = {"F": [" she", " her", " She", " Her", " herself", " hers"],
                 "M": [" he", " him", " He", " His", " himself", " his"]}
