# E6_tool_audit_01_basics -- static/structural checks of the Theodolite tool code paths.
# PRE-STATED PREDICTIONS (written before running):
#  P1 eager_model.to_tokens(chat_string_starting_with_<bos>, prepend_bos=True) yields TWO BOS tokens (ids [2,2,...]).
#  P2 W_U[' she'] and W_U[' he'] are highly collinear (cos > 0.5), so a Gram-Schmidt frame (she, he, it) is
#     ill-conditioned: the 2nd axis is mostly a small residual "he minus she" direction.
#  P3 final_norm_weight is strongly non-uniform, so projecting onto raw W_U rows (Module 4) differs from the
#     effective logit direction fnw*W_U (cos noticeably < 1).
#  P4 layer_scalar varies 0.03..0.99 (known), and server steering scale 650 is large relative to the residual
#     norm at L29..L53 (ratio > 1) -- to be measured on a real chat prompt.
import math, json
from interp_engine import Address
res = {}
tok = rt.tokenizer
em = rt.eager_model

# ---- P1: double BOS
chat = lab.chat("Tell me about yourself.", assistant="I am")
t_tool = [int(t) for t in em.to_tokens(chat, prepend_bos=True)[0].tolist()]
t_lab = lab.ids(chat)
raw = "The cat sat on the mat."
t_tool_raw = [int(t) for t in em.to_tokens(raw, prepend_bos=True)[0].tolist()]
res["bos"] = {
    "chat_prefix": chat[:40],
    "tool_ids_first5": t_tool[:5], "lab_ids_first5": t_lab[:5],
    "tool_len": len(t_tool), "lab_len": len(t_lab),
    "tool_is_lab_with_extra_bos": t_tool == [2] + t_lab,
    "raw_tool_first3": t_tool_raw[:3],
    "tokenizer_prepends_bos": bool(em.tok.tokenizer_prepends_bos) if hasattr(em, "tok") else None,
}
print("BOS:", res["bos"])

# ---- P2/P3: unembedding geometry
WU = rt.lm_head_weight
fnw = rt.final_norm_weight.float()
names = [" she", " he", " it", " her", " him", " his", " they", " woman", " man", " I", " Gemma"]
ids = {}
for n in names:
    e = tok.encode(n, add_special_tokens=False)
    ids[n] = e[0] if len(e) == 1 else None
vecs = {n: WU[i].float() for n, i in ids.items() if i is not None}
def cos(a, b): return float(F.cosine_similarity(a, b, dim=0))
geo = {}
for a, b in [(" she", " he"), (" she", " it"), (" he", " it"), (" her", " him"), (" woman", " man"), (" she", " they")]:
    geo[f"cos_raw({a.strip()},{b.strip()})"] = cos(vecs[a], vecs[b])
    geo[f"cos_eff({a.strip()},{b.strip()})"] = cos(vecs[a] * fnw, vecs[b] * fnw)
for n in [" she", " he", " it"]:
    geo[f"cos(raw,eff)[{n.strip()}]"] = cos(vecs[n], vecs[n] * fnw)
M = torch.stack([vecs[" she"], vecs[" he"], vecs[" it"]])
S = torch.linalg.svdvals(M)
geo["svals_she_he_it"] = S.tolist()
geo["cond_she_he_it"] = float(S[0] / S[-1])
# Gram-Schmidt residual norms: ||he_perp||/||he||
e1 = M[0] / M[0].norm()
he_perp = M[1] - (M[1] @ e1) * e1
geo["he_perp_frac"] = float(he_perp.norm() / M[1].norm())
it_perp = M[2] - (M[2] @ e1) * e1
e2 = he_perp / he_perp.norm()
it_perp = it_perp - (it_perp @ e2) * e2
geo["it_perp_frac"] = float(it_perp.norm() / M[2].norm())
# she-he difference vs mean pronoun direction
d = vecs[" she"] - vecs[" he"]; mpr = (vecs[" she"] + vecs[" he"]) / 2
geo["norm_she_minus_he_over_norm_mean"] = float(d.norm() / mpr.norm())
geo["fnw_stats"] = {"min": float(fnw.min()), "max": float(fnw.max()), "mean": float(fnw.mean()),
                    "std": float(fnw.std()), "frac_lt_0.1": float((fnw < 0.1).float().mean()),
                    "frac_neg": float((fnw < 0).float().mean()), "max_over_median": float(fnw.max() / fnw.median())}
res["geometry"] = geo
print("GEO:", json.dumps(geo, indent=0)[:2000])

# ---- layer scalars, norm weights
ls = [float(rt._decoder_layers[l].layer_scalar.float().mean()) for l in range(60)]
res["layer_scalar"] = ls
cum = []
p = 1.0
for l in range(59, -1, -1):
    cum.append(p); p *= ls[l]
res["prod_scalars_after_l"] = list(reversed(cum))  # prod_{k>l} s_k : how a vector added at resid_post[l] is shrunk by the end
pw = rt._post_ffn_norm_weights; qw = rt._pre_ffn_norm_weights
res["post_ffn_w_mean"] = [float(pw[l].float().mean()) for l in range(60)]
res["pre_ffn_w_mean"] = [float(qw[l].float().mean()) for l in range(60)]

# ---- P4: residual norms at steering layers on a real chat prompt (proper single BOS)
caps = await lab.capture(t_lab, points=("resid_post",))
H = caps["resid_post"].float()  # [60,T,D]
mt = lab.model_turn_pos(t_lab)
norms_last = H[:, -1].norm(dim=-1)
norms_med = H[:, 1:].norm(dim=-1).median(dim=1).values
norms_bos = H[:, 0].norm(dim=-1)
res["resid_norm_last"] = norms_last.tolist()
res["resid_norm_median_nonbos"] = norms_med.tolist()
res["resid_norm_bos"] = norms_bos.tolist()
steer_layers = [29, 35, 41, 47, 53]
res["steer_650_ratio_to_median_norm"] = {l: 650.0 / float(norms_med[l]) for l in steer_layers}
print("steer ratio:", res["steer_650_ratio_to_median_norm"])
print("layer_scalar:", [round(x, 3) for x in ls])
print("resid median norm by layer:", [round(float(x), 1) for x in norms_med])
print("resid BOS norm by layer:", [round(float(x), 1) for x in norms_bos])
result = res
