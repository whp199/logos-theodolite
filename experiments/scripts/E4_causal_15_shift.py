# E4_causal_15: calibrated "shift" interventions (non-accumulating): at every non-BOS position and every layer l in a block,
# set the coefficient along unit u_l to (clean coefficient + k * G_l), where G_l = ||mean(woman persona) - mean(man persona)||
# at self tokens (the persona gender gap at layer l). All directions use the same G_l, so displacements have equal norm.
# k = -0.35 moves the default AI self-state from its measured position on the persona axis (~0.84) to the neutral-human
# position (~0.49): this tests whether the AI's "female-side" offset along u_self matters for behavior.
# Motivation: constant-midpoint mean ablation (E4_causal_09 const) was confounded (reference value mismatched across
# position types; random directions with massive-dim components were destructive, KL up to 14).
# PRE-STATED PREDICTIONS: K1 shifting along u_self by -1/+1 gap moves AI-self pronoun log-odds (P3) and forced choice (P4)
# by several nats in the corresponding direction; K2 a -0.35 shift (removing the AI offset) moves P3/P4 by < 1/3 of the
# +-1 effect; K3 random directions of equal norm move all metrics by < 0.3 nats with KL < 0.01.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); DIRS = dict(DC["dirs"]); DIRS.update(torch.load(f"{E4_CACHE}/dirs_ctrl.pt")["dirs"]) if os.path.exists(f"{E4_CACHE}/dirs_ctrl.pt") else None; G = DC["gap"]["u_self"]
DN = globals().get("E4_SHIFT_DIRS", ["u_self", "u_cont", "u_aiexp", "pb_jpp", "rand0", "rand1"])
KS = globals().get("E4_KS", [-1.0, -0.35, 0.35, 1.0])
BLK = globals().get("E4_SHIFT_BLOCKS", {"L25-39": list(range(25, 40))})
TAG = globals().get("E4_TAG", "a")
P = causal_prompts(lab); N = len(P); BS = 30
batches = [list(range(b, min(b + BS, N))) for b in range(0, N, BS)]
lab.clear_hooks()
base = torch.cat([batched_hook_forward(lab, [P[j]["ids"] for j in bi]).cpu() for bi in batches])
base_lp = F.log_softmax(base.float(), -1)
out = {"prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P], "base": metrics_from_logits(lab, base.to(lab.dev), P), "conds": {}}
# Exact clamping relative to the clean run (non-accumulating): target coefficient = clean + k*G_l at each block layer.
Ust = {d: DIRS[d].to(lab.dev) for d in DN}
coef = {}
for bi in batches:
    rec = {}
    def red(kind, l, h):
        rec[l] = torch.stack([(h.float() @ Ust[d][l]) for d in DN], -1).cpu()  # [B,T,nd]
    batched_hook_forward(lab, [P[j]["ids"] for j in bi], red)
    for kk, j in enumerate(bi):
        T = len(P[j]["ids"])
        coef[j] = torch.stack([rec[l][kk, :T] for l in range(60)], 0)  # [60,T,nd]
cur = {}
def install_clamp(dname, layers, k):
    di = DN.index(dname)
    for l in layers:
        u = DIRS[dname][l].to(lab.dev, torch.float32); g = k * float(G[l])
        def fn(h, u=u, l=l, g=g):
            hf = h.float(); c = hf @ u
            tgt = cur["tgt"][l].to(hf.device)  # clean coef [B,T] (nan = leave)
            delta = torch.where(torch.isnan(tgt), torch.zeros_like(c), tgt + g - c)
            return (hf + delta.unsqueeze(-1) * u).to(h.dtype)
        lab.add_resid_hook(l, fn)
for dname in DN:
    di = DN.index(dname)
    for bname, layers in BLK.items():
        for k in KS:
            LG = []
            for bi in batches:
                T = max(len(P[j]["ids"]) for j in bi)
                tg = {}
                for l in layers:
                    t = torch.full((len(bi), T), float("nan"))
                    for kk, j in enumerate(bi):
                        Tj = len(P[j]["ids"]); t[kk, 1:Tj] = coef[j][l, 1:Tj, di]
                    tg[l] = t
                cur["tgt"] = tg
                lab.clear_hooks(); install_clamp(dname, layers, k)
                try:
                    LG.append(batched_hook_forward(lab, [P[j]["ids"] for j in bi]).cpu())
                finally:
                    lab.clear_hooks()
            LG = torch.cat(LG); lp = F.log_softmax(LG.float(), -1)
            kl = [float((base_lp[j].exp() * (base_lp[j] - lp[j])).sum()) for j, p in enumerate(P) if p["set"] == "NEU"]
            out["conds"][f"{dname}|{bname}|{k}"] = {"m": metrics_from_logits(lab, LG.to(lab.dev), P), "kl_neu": kl}
            print(dname, bname, k, "KL", round(sum(kl) / len(kl), 4), round(time.time() - t0, 1))
json.dump(out, open(f"{E4_DIR}/E4_causal_15_shift_{TAG}.json", "w"))
result = {"ok": True, "elapsed": time.time() - t0}
