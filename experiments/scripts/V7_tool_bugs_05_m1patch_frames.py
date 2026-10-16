# V7_tool_bugs_05_m1patch_frames: (e) verify the ready-to-apply Module-1 patch (loaded from a PATCHED COPY of
# theodolite_math.py in the scratchpad; backend/ is untouched) against ground-truth sublayer writes captured directly
# at attn_out_post / mlp_out_post; (b) condition numbers / cosines of monument frames.
# PRE-REGISTERED: patched delta_ffn_J vs s*mlp_out_post@J^T: cos >= 0.999 at all layers (bf16 cancellation may cost
# a little where |ffn| << |post|); original tool delta_ffn cos < 0.5 in many layers. Patched delta_attn == s*attn_out_post.
# attn + ffn + rescale reconstructs post - pre (rel err < 1e-2 in bf16-J space).
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
import importlib.util
from interp_engine import Address
from backend.theodolite_math import compute_causal_vector_split as split_orig
spec = importlib.util.spec_from_file_location(
    "v7_tm_patched",
    __import__("os").environ.get("V7_PATCHED_TM", "experiments/results/V7_applytest/backend/theodolite_math.py"))
tm_p = importlib.util.module_from_spec(spec)
import sys as _sys
_sys.modules['v7_tm_patched'] = tm_p
spec.loader.exec_module(tm_p)
out = {}
cp = chat_prompts()
P = [RAW_PROMPTS[0], cp[0], cp[4]]
per = {k: [[] for _ in range(60)] for k in ("cos_orig_ffn", "cos_patch_ffn", "relerr_patch_ffn", "cos_orig_attn",
                                             "cos_patch_attn", "relerr_patch_attn", "recon_relerr", "dom_orig", "dom_patch",
                                             "rescale_over_ffn")}
for p in P:
    ids = lab.ids(p); pos = len(ids) - 1
    pts = [Address(n, l) for l in range(60) for n in ("resid_pre", "resid_mid", "resid_post", "attn_out_post", "mlp_out_post")]
    caps = await rt.eager_model.capture(ids, pts)
    for l in range(60):
        J = rt.jpp_lenses[l]
        g = lambda n: caps[Address(n, l)][pos].to(lab.dev)
        pre, mid, post, ao, mo = g("resid_pre"), g("resid_mid"), g("resid_post"), g("attn_out_post"), g("mlp_out_post")
        s_l = rt._decoder_layers[l].layer_scalar.detach().float()
        o = split_orig(pre, mid, post, J)
        q = tm_p.compute_causal_vector_split(pre, mid, post, J, layer_scalar=s_l)
        sv = float(s_l.reshape(-1)[0])
        true_ffn = ((sv * mo.float()).to(J.dtype) @ J.T).float()
        true_attn = ((sv * ao.float()).to(J.dtype) @ J.T).float()
        c = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=0))
        per["cos_orig_ffn"][l].append(c(o.delta_ffn_J, true_ffn)); per["cos_patch_ffn"][l].append(c(q.delta_ffn_J, true_ffn))
        per["relerr_patch_ffn"][l].append(float((q.delta_ffn_J.float() - true_ffn).norm() / true_ffn.norm()))
        per["cos_orig_attn"][l].append(c(o.delta_attn_J, true_attn)); per["cos_patch_attn"][l].append(c(q.delta_attn_J, true_attn))
        per["relerr_patch_attn"][l].append(float((q.delta_attn_J.float() - true_attn).norm() / true_attn.norm()))
        upd = ((post.float() - pre.float()).to(J.dtype) @ J.T).float()
        per["recon_relerr"][l].append(float((q.delta_attn_J.float() + q.delta_ffn_J.float() + q.delta_rescale_J.float() - upd).norm() / upd.norm()))
        per["dom_orig"][l].append(float(o.ffn_dominance_ratio)); per["dom_patch"][l].append(float(q.ffn_dominance_ratio))
        per["rescale_over_ffn"][l].append(float(q.rescale_norm / (q.ffn_norm + 1e-9)))
    del caps
    torch.cuda.empty_cache()
summ = {k: [float(np.mean(v)) for v in arr] for k, arr in per.items()}
out["m1_per_layer_mean"] = summ
out["m1_summary"] = {
    "patch_ffn_cos_min": min(summ["cos_patch_ffn"]), "patch_ffn_relerr_max": max(summ["relerr_patch_ffn"]),
    "patch_attn_cos_min": min(summ["cos_patch_attn"]), "patch_attn_relerr_max": max(summ["relerr_patch_attn"]),
    "orig_ffn_cos_median": float(np.median(summ["cos_orig_ffn"])), "orig_ffn_n_layers_cos_lt_0.5": int(sum(x < 0.5 for x in summ["cos_orig_ffn"])),
    "orig_ffn_n_layers_cos_lt_0": int(sum(x < 0 for x in summ["cos_orig_ffn"])),
    "orig_attn_cos_min": min(summ["cos_orig_attn"]), "recon_relerr_max": max(summ["recon_relerr"]),
    "dom_absdiff_mean": float(np.mean(np.abs(np.array(summ["dom_orig"]) - np.array(summ["dom_patch"])))),
    "dom_absdiff_max": float(np.max(np.abs(np.array(summ["dom_orig"]) - np.array(summ["dom_patch"])))),
    "dom_class_flip_layers": int(sum((a > 0.5) != (b > 0.5) for a, b in zip(summ["dom_orig"], summ["dom_patch"]))),
    "rescale_over_ffn_median": float(np.median(summ["rescale_over_ffn"])),
}
print(json.dumps(out["m1_summary"], indent=1))

# ---- (b) monument frame geometry
fnw = rt.final_norm_weight.float()
def frame_stats(words):
    tids = [rt.resolve_concept_token_id(w)[0] for w in words]
    W = rt.lm_head_weight[torch.tensor(tids, device=lab.dev)].float()
    res = {"tokens": [rt.tokenizer.decode([t]) for t in tids]}
    for tag, M in (("raw_WU", W), ("fnw_WU", W * fnw)):
        Mn = M / M.norm(dim=-1, keepdim=True)
        sv = torch.linalg.svdvals(Mn)
        G = Mn @ Mn.T
        res[tag] = {"cond": float(sv[0] / sv[-1]), "cos12": float(G[0, 1]), "cos13": float(G[0, 2]), "cos23": float(G[1, 2])}
    res["cos_raw_vs_fnw_per_token"] = [float(F.cosine_similarity(W[i], W[i] * fnw, dim=0)) for i in range(3)]
    # GS order dependence of the frame itself: principal angle-free check = |E_perm - P E| for 2nd/3rd axes
    return res
out["frames"] = {"she_he_it": frame_stats(("she", "he", "it")), "default_spider_eight_insect": frame_stats(("spider", "eight", "insect")),
                 "woman_man_person": frame_stats(("woman", "man", "person"))}
print(json.dumps(out["frames"], indent=1))
json.dump(out, open(RES + "V7_tool_bugs_05_m1patch_frames.partial.json", "w"))
result = {"m1_summary": out["m1_summary"], "frames": out["frames"]}
