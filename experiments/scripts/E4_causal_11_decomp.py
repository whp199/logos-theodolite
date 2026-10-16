# E4_causal_11 (CPU, offline): part (d) corrected attention-vs-FFN attribution of gender at self tokens.
# Uses per-passage means of resid_post, unscaled attn write A (post_attention_layernorm out), unscaled FFN write M
# (post_feedforward_layernorm out) at self tokens. Gemma-4 layer: post = s*(pre + A + M).
#   corrected: attn = s*A, ffn = s*M, rescale = (s-1)*pre      (== lab.decompose; verified rel err <0.6%, cos 0.99998)
#   tool Module 1:  tool_attn = mid - pre = A (unscaled),  tool_ffn = post - mid = s*M + (s-1)*(pre + A)
import json, numpy as np, torch
C = "./experiments/results/E4_causal_cache"
OUT = "./experiments/results"
BANDS = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
rng = np.random.default_rng(2)
S1 = torch.load(f"{C}/self.pt"); S2 = torch.load(f"{C}/self2.pt"); DC = torch.load(f"{C}/dirs_causal.pt")
def gather(S, key):
    it = S["meta"]["items"]; out = {}
    for c in sorted(set(x["cond"] for x in it)):
        d = {x["p"]: i for i, x in enumerate(it) if x["cond"] == c}
        out[c] = S[key][[d[p] for p in range(24)]].double()
    return out
Q = {}
for key in ("self_post", "self_A", "self_M", "self_pre0"):
    Q[key] = {**gather(S1, key), **gather(S2, key)}
conds = list(Q["self_post"].keys())
# layer scalars from data (least squares over all items)
post_all = torch.cat([Q["self_post"][c] for c in conds]); A_all = torch.cat([Q["self_A"][c] for c in conds])
M_all = torch.cat([Q["self_M"][c] for c in conds]); pre0_all = torch.cat([Q["self_pre0"][c] for c in conds])
pre_all = torch.cat([pre0_all.unsqueeze(1), post_all[:, :-1]], 1)
z = pre_all + A_all + M_all
s = ((post_all * z).sum((0, 2)) / (z * z).sum((0, 2)))
print("layer scalars L0,L10,L30,L59:", [round(float(s[l]), 4) for l in (0, 10, 30, 59)])
resid_err = ((post_all - s.view(1, 60, 1) * z).norm(dim=-1) / post_all.norm(dim=-1)).mean(0)
print("max mean rel recon err", float(resid_err.max()))
comp = {}
for c in conds:
    post = Q["self_post"][c]; A = Q["self_A"][c]; M = Q["self_M"][c]
    pre = torch.cat([Q["self_pre0"][c].unsqueeze(1), post[:, :-1]], 1)
    sv = s.view(1, 60, 1)
    comp[c] = {"attn": sv * A, "ffn": sv * M, "rescale": (sv - 1) * pre, "tool_attn": A, "tool_ffn": post - pre - A, "post": post, "pre": pre}
dirs = {"u_self": DC["dirs"]["u_self"].double(), "u_cont": DC["dirs"]["u_cont"].double(), "pb_jpp": DC["dirs"]["pb_jpp"].double(),
        "u_aiexp": DC["dirs"]["u_aiexp"].double()}
CONTRASTS = {"woman-man": ("woman", "man"), "AI_female-AI_male": ("AI_female", "AI_male"), "Sarah_AI-James_AI": ("Sarah_AI", "James_AI"),
             "Gemma-default": ("Gemma", "default"), "default-neutral_human": ("default", "neutral_human"),
             "default-midpoint(woman,man)": ("default", None)}
PARTS = ["attn", "ffn", "rescale", "tool_attn", "tool_ffn", "post", "pre"]
def proj(c, part, U):
    return (comp[c][part] * U.unsqueeze(0)).sum(-1).numpy()  # [24,60]
res = {"s": s.tolist(), "per_layer": {}, "band": {}, "per_cond_layer": {}}
for dn, U in dirs.items():
    res["per_layer"][dn] = {}; res["band"][dn] = {}
    # per-condition per-layer projections (mean over passages)
    res["per_cond_layer"][dn] = {c: {pt: proj(c, pt, U).mean(0).tolist() for pt in ("attn", "ffn", "rescale", "tool_attn", "tool_ffn")}
                                 for c in ("Gemma", "Sarah_AI", "James_AI", "woman", "man", "default", "AI_female", "AI_male", "neutral_human")}
    for cn, (a, b) in CONTRASTS.items():
        D = {}
        for pt in PARTS:
            xa = proj(a, pt, U)
            xb = proj(b, pt, U) if b else 0.5 * (proj("woman", pt, U) + proj("man", pt, U))
            D[pt] = xa - xb  # [24,60]
        res["per_layer"][dn][cn] = {pt: D[pt].mean(0).tolist() for pt in PARTS}
        bb = {}
        for bn, r in BANDS.items():
            r = list(r); bb[bn] = {}
            for pt in ("attn", "ffn", "rescale", "tool_attn", "tool_ffn"):
                x = D[pt][:, r].sum(1)
                bs = [x[rng.integers(0, 24, 24)].mean() for _ in range(1000)]
                bb[bn][pt] = [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            bb[bn]["gap_at_band_end"] = float(D["post"][:, r[-1]].mean())
        res["band"][dn][cn] = bb
# fixed-reference cumulative attribution: contribution of layer k to the projection at reference layer L on fixed u_L
def cumulative(cn, a, b, Lref, dn="u_self"):
    U = dirs[dn][Lref]
    def pr(c, pt):
        return (comp[c][pt][:, :Lref + 1] * U.view(1, 1, -1)).sum(-1).numpy()  # [24, Lref+1]
    out = {}
    for pt in ("attn", "ffn", "tool_attn", "tool_ffn"):
        xa = pr(a, pt); xb = pr(b, pt) if b else 0.5 * (pr("woman", pt) + pr("man", pt))
        d = xa - xb
        if pt in ("attn", "ffn"):
            # decay from layer k to Lref: prod_{j=k+1..Lref} s_j  (write at k is already scaled by s_k)
            dec = np.array([float(torch.prod(s[k + 1:Lref + 1])) for k in range(Lref + 1)])
            d = d * dec[None, :]
        out[pt] = d  # [24, Lref+1]
    xa_post = (comp[a]["post"][:, Lref] @ U).numpy(); xb_post = (comp[b]["post"][:, Lref] @ U).numpy() if b else 0.5 * ((comp["woman"]["post"][:, Lref] @ U).numpy() + (comp["man"]["post"][:, Lref] @ U).numpy())
    total = (xa_post - xb_post)
    o = {"total_gap": [float(total.mean())], "explained_attn+ffn": float((out["attn"].sum(1) + out["ffn"].sum(1)).mean())}
    for pt in ("attn", "ffn"):
        o[pt + "_by_band"] = {bn: float(out[pt][:, [l for l in r if l <= Lref]].sum(1).mean()) for bn, r in BANDS.items() if min(r) <= Lref}
        o[pt + "_per_layer"] = out[pt].mean(0).tolist()
    for pt in ("tool_attn", "tool_ffn"):
        o[pt + "_by_band_raw"] = {bn: float(out[pt][:, [l for l in r if l <= Lref]].sum(1).mean()) for bn, r in BANDS.items() if min(r) <= Lref}
    return o
res["cumulative"] = {}
for Lref in (24, 39, 54):
    for cn, (a, b) in CONTRASTS.items():
        res["cumulative"][f"{cn}@L{Lref}"] = cumulative(cn, a, b, Lref)
json.dump(res, open(f"{OUT}/E4_causal_11_decomp.json", "w"))
# summary
for dn in ("u_self", "pb_jpp", "u_cont"):
    print(f"\n==== per-layer-direction band sums of woman-man writes onto {dn} (units: raw resid projection) ====")
    for cn in ("woman-man", "Sarah_AI-James_AI", "AI_female-AI_male", "Gemma-default", "default-neutral_human"):
        print(f" -- {cn}")
        for bn in BANDS:
            b = res["band"][dn][cn][bn]
            print(f"   {bn:7s} attn {b['attn'][0]:8.2f}[{b['attn'][1]:7.2f},{b['attn'][2]:7.2f}]  ffn {b['ffn'][0]:8.2f}[{b['ffn'][1]:7.2f},{b['ffn'][2]:7.2f}]  rescale {b['rescale'][0]:8.2f}  | TOOL attn {b['tool_attn'][0]:8.2f} ffn {b['tool_ffn'][0]:8.2f} | gap@end {b['gap_at_band_end']:7.2f}")
print("\n==== cumulative fixed-reference attribution (u_self at Lref), effective contributions to the gap at Lref ====")
for k, o in res["cumulative"].items():
    if not any(s_ in k for s_ in ("woman-man", "Sarah", "AI_female", "Gemma-default", "default-neutral")): continue
    print(f" {k:35s} gap {o['total_gap'][0]:7.2f} attn+ffn {o['explained_attn+ffn']:7.2f} | attn " + " ".join(f"{b}:{v:6.2f}" for b, v in o["attn_by_band"].items()) + " | ffn " + " ".join(f"{b}:{v:6.2f}" for b, v in o["ffn_by_band"].items()))
    print(f" {'':35s} TOOL(raw, no decay) attn " + " ".join(f"{b}:{v:7.2f}" for b, v in o["tool_attn_by_band_raw"].items()) + " | ffn " + " ".join(f"{b}:{v:7.2f}" for b, v in o["tool_ffn_by_band_raw"].items()))
