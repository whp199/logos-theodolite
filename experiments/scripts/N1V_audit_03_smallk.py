# N1V_audit_03_smallk -- verifier daemon job: does the N1A late-layer label validity (and the L20-30 null) survive at
# a steering magnitude in the LINEAR regime of the post-FFN RMSNorm? N1V_audit_01 showed that at the N1A k=60 the
# residual change is ~1.0-1.6x the norm of the whole MLP output at L20-57 (saturation 0.52-0.82, cos with the MLP-
# ablation direction 0.22-0.52). Here: identical N1A protocol (pre-hook on down_proj adding s*k*max|act| at the last
# position; base slot in the same forward), but k=10 (ratio ||dW||/||y|| ~0.13-0.31, saturation >= 0.965) at
# L20-55 and k=3 at L50/L55.
# PRE-STATED PREDICTIONS: (S1) at L40-55 J-VINDEX/naive P@10 vs causal at k=10 is still far above null (>= 0.25),
# i.e. late validity is not an artefact of the huge k; rho(c+,c-) more negative than at k=60. (S2) at L20-30 static
# labels stay at null even in the linear regime (P@10 <= 0.01), and c+/c- stop being positively correlated.
exec(open("./experiments/scripts/N1A_common.py").read())
import time
t0 = time.time()
OUTD = f"{N1A_ROOT}/results/N1V_cache"
os.makedirs(OUTD, exist_ok=True)
NCTX = 36
SLOTS = 8
RUNS = globals().get("N1V_RUNS", [(10, 30), (10, 40), (10, 50), (10, 20), (10, 45), (10, 55), (3, 50), (3, 55)])
ctx = n1a_contexts()
ids = [lab.ids(t) for _, t in ctx]
inp, att, last = n1a_pad(ids)
valid = att.clone().bool(); valid[:, 0] = False
ar = torch.arange(NCTX, device=lab.dev)
LAYS = sorted(set(l for _, l in RUNS))
cap, hooks = {}, []
try:
    for l in LAYS:
        nidx = torch.tensor(N1A_NEURONS[l], device=lab.dev)
        def pre(mod, args, l=l, nidx=nidx):
            a = args[0][..., nidx].abs() * valid.unsqueeze(-1)
            cap[("max", l)] = a.reshape(-1, a.shape[-1]).max(0).values.float()
        hooks.append(lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        def prevh(mod, args, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            cap[("prev", l)] = h.detach().clone()
        hooks.append(lab.layers[l - 1].register_forward_hook(prevh))
    with torch.no_grad():
        n1a_forward_last(inp, att, last)
finally:
    for h in hooks:
        h.remove()
res = {"runs": {}}


@torch.no_grad()
def run(l, k):
    mx = cap[("max", l)]
    jobs = [(ni, s) for ni in range(24) for s in (1, -1)]
    prev = cap[("prev", l)]
    outs = []
    for c0 in range(0, len(jobs), SLOTS - 1):
        chunk = jobs[c0:c0 + SLOTS - 1]
        S = len(chunk) + 1
        rows = torch.cat([ar + si * NCTX for si in range(1, S)])
        lasts = last.repeat(S - 1)
        neus = torch.cat([torch.full((NCTX,), N1A_NEURONS[l][ni], device=lab.dev, dtype=torch.long) for ni, s in chunk])
        ds = torch.cat([torch.full((NCTX,), float(s * k * mx[ni]), device=lab.dev) for ni, s in chunk])
        def pre(mod, args):
            x = args[0]
            x.index_put_((rows, lasts, neus), ds.to(x.dtype), accumulate=True)
            return (x,) + tuple(args[1:])
        h = lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
        try:
            with N1ARunFrom(l, prev.repeat(S, 1, 1)):
                hl = n1a_forward_last(inp.repeat(S, 1), att.repeat(S, 1), last.repeat(S))
        finally:
            h.remove()
        lsm = F.log_softmax(n1a_logits_fp32(hl), -1).view(S, NCTX, -1)
        base = lsm[0]
        for si in range(1, S):
            diff = lsm[si] - base
            klc = (lsm[si].exp() * diff).sum(-1)
            outs.append({"c": diff.mean(0).cpu(), "cA": diff[0::2].mean(0).cpu(), "cB": diff[1::2].mean(0).cpu(),
                         "kl": float(klc.mean())})
        del lsm, base, hl
    torch.save({"poles": jobs, "neurons": N1A_NEURONS[l], "k": k, "c": torch.stack([o["c"] for o in outs]),
                "cA": torch.stack([o["cA"] for o in outs]).half(), "cB": torch.stack([o["cB"] for o in outs]).half(),
                "kl": [o["kl"] for o in outs]}, f"{OUTD}/smallk_k{k}_L{l}.pt")
    kl = torch.tensor([o["kl"] for o in outs])
    return {"kl_median": float(kl.median()), "kl_q10": float(kl.quantile(0.1)), "kl_q90": float(kl.quantile(0.9))}


for k, l in RUNS:
    if time.time() - t0 > 190:
        res["runs"][f"k{k}_L{l}"] = "skipped (time)"
        continue
    res["runs"][f"k{k}_L{l}"] = run(l, k)
    print(f"k{k} L{l}", res["runs"][f"k{k}_L{l}"], f"{time.time()-t0:.1f}s", flush=True)
    torch.cuda.empty_cache()
cap.clear()
torch.cuda.empty_cache()
res["elapsed"] = time.time() - t0
result = res
