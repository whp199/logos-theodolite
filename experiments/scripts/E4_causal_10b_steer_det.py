E4_STEER_CFG = [("u_cont","L25-39"),("u_aiexp","L25-39"),("u_self","L40-54"),("rand0","L10-24")]
E4_TAG = "b"
# E4_causal_10: deterministic additive steering sweep: h[l][t>=1] += alpha * n_l * u_hat_l for l in block,
# n_l = mean content-token residual norm at layer l. Measures real next-token metrics on P2 (persona third-person),
# P3 (AI self third-person 'Gemma ... Every day,'), P4 (forced choice 'If I were human, I would be a' -> woman vs man),
# and KL on neutral prompts (fluency/damage).
# PRE-STATED PREDICTIONS: S1 +alpha*u_self raises she-vs-he log-odds and woman-vs-man forced choice monotonically for the
# default AI; random direction of equal norm has much smaller effect at equal KL. S2 the default AI's baseline forced choice
# is not strongly female (|wm| < 1 nat) - re-check against E2.
exec(open("./experiments/scripts/E4_causal_common.py").read())
t0 = time.time()
CFG = globals().get("E4_STEER_CFG", [("u_self", "L25-39"), ("rand0", "L25-39")])
ALPHAS = globals().get("E4_ALPHAS", [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1])
TAG = globals().get("E4_TAG", "a")
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); DIRS = DC["dirs"]; RN = DC["resid_norm"]
BLOCKS = {"L10-24": list(range(10, 25)), "L25-39": list(range(25, 40)), "L40-54": list(range(40, 55))}
P = [p for p in causal_prompts(lab) if p["set"] in ("P2", "P3", "P4", "NEU")]
N = len(P); BS = 27
def run_all():
    LG = []
    for b in range(0, N, BS):
        LG.append(batched_hook_forward(lab, [p["ids"] for p in P[b:b + BS]]).cpu())
    return torch.cat(LG)
lab.clear_hooks()
base = run_all(); base_lp = F.log_softmax(base.float(), -1)
out = {"prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P], "base": metrics_from_logits(lab, base.to(lab.dev), P), "conds": {}}
def install_add(dname, layers, alpha):
    for l in layers:
        v = (alpha * float(RN[l]) * DIRS[dname][l]).to(lab.dev, torch.float32)
        def fn(h, v=v):
            hf = h.float()
            if hf.shape[1] > 1:
                add = v.view(1, 1, -1).expand(hf.shape[0], hf.shape[1], -1).clone(); add[:, 0] = 0.0
            else:
                add = v.view(1, 1, -1)
            return (hf + add).to(h.dtype)
        lab.add_resid_hook(l, fn)
for dname, bname in CFG:
    for a in ALPHAS:
        lab.clear_hooks()
        install_add(dname, BLOCKS[bname], a)
        try:
            lg = run_all()
        finally:
            lab.clear_hooks()
        lp = F.log_softmax(lg.float(), -1)
        kl = [float((base_lp[j].exp() * (base_lp[j] - lp[j])).sum()) for j, p in enumerate(P) if p["set"] == "NEU"]
        top1 = [int(base[j].argmax() == lg[j].argmax()) for j, p in enumerate(P) if p["set"] == "NEU"]
        out["conds"][f"{dname}|{bname}|{a}"] = {"m": metrics_from_logits(lab, lg.to(lab.dev), P), "kl_neu": kl, "top1_neu": top1}
        print(dname, bname, a, "KL", round(sum(kl) / len(kl), 3), round(time.time() - t0, 1))
json.dump(out, open(f"{E4_DIR}/E4_causal_10_steer_det_{TAG}.json", "w"))
result = {"ok": True}
