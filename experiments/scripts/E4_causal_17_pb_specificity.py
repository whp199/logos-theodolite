# E4_causal_17: specificity control for the surprisingly strong effect of pair-matched clamping along the J++ pullback of the
# pronoun gender direction (removed ~90% of P1 gap at L10-24 and L25-39 in E4_causal_09b).
# Directions (unit, per layer): J++ pullbacks of gender pron/noun directions, JLens pullback of pron, J++ pullbacks of
# CONTROL token-pair directions (cat-dog, yes-no, big-small, the-a), u_self, u_self orthogonalized to pb_jpp_pron, rand0.
# Pair-matched clamp (position-wise F/M midpoint of clean coefficients); clean coefs recomputed with the SAME batches.
# PRE-STATED PREDICTION: if pb_jpp_pron is a genuine gender readout direction, control pullbacks remove <10% of the gap.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); D0 = DC["dirs"]
def pb_pair(a, b, kind="jpp"):
    d = (lab.WU[lab.tid(a)].float() - lab.WU[lab.tid(b)].float()) * lab.fnw.float()
    return torch.stack([lab.pullback(d, l, kind).float().cpu() for l in range(60)])
DIRS = {"pb_jpp_pron": D0["pb_jpp"], "pb_jpp_noun": torch.stack([lab.pullback(lab.gender_direction("noun"), l, "jpp").float().cpu() for l in range(60)]),
        "pb_jlens_pron": torch.stack([lab.pullback(lab.gender_direction("pron"), l, "jlens").float().cpu() for l in range(60)]),
        "pb_jpp_cat_dog": pb_pair(" cat", " dog"), "pb_jpp_yes_no": pb_pair(" yes", " no"), "pb_jpp_big_small": pb_pair(" big", " small"),
        "pb_jpp_the_a": pb_pair(" the", " a"), "u_self": D0["u_self"], "rand0": D0["rand0"]}
us = D0["u_self"]; pbn = F.normalize(D0["pb_jpp"], dim=-1)
DIRS["u_self_perp_pb"] = us - (us * pbn).sum(-1, keepdim=True) * pbn
DIRS = {k: F.normalize(v.float(), dim=-1) for k, v in DIRS.items()}
DN = list(DIRS)
print("cos(pb_jpp_pron, controls) L20:", {k: round(float((DIRS[k][20] * DIRS["pb_jpp_pron"][20]).sum()), 3) for k in DN})
BLOCKS = {"L10-24": list(range(10, 25)), "L25-39": list(range(25, 40))}
P = [p for p in causal_prompts(lab) if p["set"] in ("P1", "P2")]
for j, p in enumerate(P):
    q = j + 1 if p["cond"] == "F" else j - 1
    p["partner"] = q if len(P[q]["ids"]) == len(p["ids"]) else None
N = len(P); BS = 24
batches = [list(range(b, min(b + BS, N))) for b in range(0, N, BS)]
Ust = torch.stack([DIRS[d] for d in DN], 1).to(lab.dev)
coef = {}; base = []
lab.clear_hooks()
for bi in batches:
    rec = {}
    def red(kind, l, h):
        rec[l] = (h.float() @ Ust[l].T).cpu()
    base.append(batched_hook_forward(lab, [P[j]["ids"] for j in bi], red).cpu())
    for k, j in enumerate(bi):
        T = len(P[j]["ids"]); coef[j] = torch.stack([rec[l][k, :T] for l in range(60)], 0)
base = torch.cat(base)
out = {"prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P], "base": metrics_from_logits(lab, base.to(lab.dev), P), "conds": {}}
cur = {}
def install(dname, layers):
    for l in layers:
        u = DIRS[dname][l].to(lab.dev, torch.float32)
        def fn(h, u=u, l=l):
            hf = h.float(); c = hf @ u
            tgt = cur[l].to(hf.device)
            delta = torch.where(torch.isnan(tgt), torch.zeros_like(c), tgt - c)
            return (hf + delta.unsqueeze(-1) * u).to(h.dtype)
        lab.add_resid_hook(l, fn)
for dname in DN:
    di = DN.index(dname)
    for bname, layers in BLOCKS.items():
        LG = []
        for bi in batches:
            T = max(len(P[j]["ids"]) for j in bi)
            cur.clear()
            for l in layers:
                t = torch.full((len(bi), T), float("nan"))
                for k, j in enumerate(bi):
                    q = P[j]["partner"]
                    if q is None: continue
                    Tj = len(P[j]["ids"]); t[k, 1:Tj] = 0.5 * (coef[j][l, 1:Tj, di] + coef[q][l, 1:Tj, di])
                cur[l] = t
            lab.clear_hooks(); install(dname, layers)
            try:
                LG.append(batched_hook_forward(lab, [P[j]["ids"] for j in bi]).cpu())
            finally:
                lab.clear_hooks()
        LG = torch.cat(LG)
        out["conds"][f"{dname}|{bname}"] = {"idx": list(range(N)), "m": metrics_from_logits(lab, LG.to(lab.dev), P)}
        print(dname, bname, round(time.time() - t0, 1))
json.dump(out, open(f"{E4_DIR}/E4_causal_17_pb_specificity_data.json", "w"))
result = {"ok": True}
