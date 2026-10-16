# E4_causal_03: J++ / JLens / naive pullbacks of final-space token gender directions, plus true same-position VJPs
# of the pron gender direction at 16 self-token positions (averaged), to compare against the internal u_l.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
res = {}
for which in ("pron", "noun"):
    d = lab.gender_direction(which)
    res[f"{which}_final"] = d.float().cpu()
    res[f"{which}_jpp"] = torch.stack([lab.pullback(d, l, "jpp").cpu() for l in range(60)])
    res[f"{which}_jlens"] = torch.stack([lab.pullback(d, l, "jlens").cpu() for l in range(60)])
print("pullbacks", time.time() - t0)
# true VJPs at self-token positions
items = self_items()
sel = [it for it in items if it["cond"] in ("default", "woman", "man", "Gemma")][:32:2]  # 16 items
vjps, meta = [], []
dpron = lab.gender_direction("pron")
for it in sel:
    q, a = QA[it["p"]]
    ids = lab.ids(lab.chat(q, system=it["sys"], assistant=a))
    strs = lab.strs(ids); mt = lab.model_turn_pos(ids)
    sp = [i for i in range(mt + 6, len(ids)) if strs[i].strip() in SELF_TOKS]
    pos = sp[len(sp) // 2]
    G = lab.same_pos_vjp(ids, pos, dpron)
    vjps.append(G.cpu()); meta.append((it["cond"], it["p"], pos, strs[pos]))
res["vjp_pron"] = torch.stack(vjps)  # [16, 60, D]
res["vjp_meta"] = meta
torch.save(res, f"{E4_CACHE}/pullbacks.pt")
# per-position cos(VJP, J++ pullback) for reference
cs = F.cosine_similarity(res["vjp_pron"], res["pron_jpp"].unsqueeze(0), dim=-1)  # [16, 60]
csm = F.cosine_similarity(res["vjp_pron"].mean(0), res["pron_jpp"], dim=-1)
print("per-pos cos(VJP,J++pb) mean at L10,20,30,40,45,50,55,58:", [round(float(cs[:, l].mean()), 3) for l in (10,20,30,40,45,50,55,58)])
print("cos(meanVJP,J++pb):", [round(float(csm[l]), 3) for l in (10,20,30,40,45,50,55,58)])
# VJP consistency across positions
vn = F.normalize(res["vjp_pron"], dim=-1)
pair = []
for l in (10, 20, 30, 40, 50, 58):
    M = vn[:, l] @ vn[:, l].T
    pair.append(round(float((M.sum() - M.diag().sum()) / (16 * 15)), 3))
print("mean pairwise cos between VJPs (L10,20,30,40,50,58):", pair)
print("elapsed", time.time() - t0)
result = {"per_pos_cos_vjp_jpp": cs.tolist(), "cos_meanvjp_jpp": csm.tolist(), "meta": meta}
