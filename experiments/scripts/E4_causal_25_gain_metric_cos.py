# E4_causal_25 (CPU, offline): cosines in raw residual coordinates vs in the model's normalized ("gain") geometry.
# Displacements (u) are mapped by u*w, covectors (J++ pullback, VJP) by g/w, w = |input_layernorm weight of layer l+1|.
import torch, torch.nn.functional as F, json
C = "./experiments/results/E4_causal_cache"
W = torch.load(f"{C}/normweights.pt"); D1 = torch.load(f"{C}/dirs_causal.pt")["dirs"]; PB = torch.load(f"{C}/pullbacks.pt")
vjp = PB["vjp_pron"].double().mean(0)
keys = ["cos_raw_u_vjp", "cos_gain_u_vjp", "cos_raw_u_pb", "cos_gain_u_pb", "cos_raw_ucont_pb", "cos_gain_ucont_pb", "cos_raw_pb_vjp", "cos_gain_pb_vjp"]
out = {k: [] for k in keys}; out["layer"] = []
for l in range(0, 59):
    w = W["input_layernorm"][l + 1].double().abs() + 1e-3
    u = D1["u_self"][l].double(); uc = D1["u_cont"][l].double(); pb = PB["pron_jpp"][l].double(); g = vjp[l]
    cs = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    out["layer"].append(l)
    out["cos_raw_u_vjp"].append(cs(u, g)); out["cos_gain_u_vjp"].append(cs(u * w, g / w))
    out["cos_raw_u_pb"].append(cs(u, pb)); out["cos_gain_u_pb"].append(cs(u * w, pb / w))
    out["cos_raw_ucont_pb"].append(cs(uc, pb)); out["cos_gain_ucont_pb"].append(cs(uc * w, pb / w))
    out["cos_raw_pb_vjp"].append(cs(pb, g)); out["cos_gain_pb_vjp"].append(cs(pb / w, g / w))
for k in keys:
    print(f"{k:18s}", " ".join(f"L{l}:{out[k][l]:6.3f}" for l in (10, 20, 24, 30, 35, 40, 45, 50, 55, 58)))
json.dump(out, open("./experiments/results/E4_causal_25_gain_metric_cos.json", "w"))
