# N1A_02_causal_core -- CAUSAL ground truth for output-side pole labels (exec'd by small wrapper jobs that set
#   N1A_MODE ('calib' | 'main'), N1A_RUN_LAYERS (list), N1A_KS (dict tag->k), N1A_GENDER (bool)).
# Intervention: forward_pre_hook on layers[l].mlp.down_proj adds delta = sign*k*max|mlp_act[l,i]| (max over the
# 36 contexts' non-BOS positions) to mlp_act[l, last, i] at the LAST position only. Layers < l are skipped by
# replaying the cached resid_post[l-1] (exact: nothing upstream changes). The unsteered baseline is a slot IN THE
# SAME FORWARD (batch-shape-dependent bf16 GEMM noise was measured at KL up to 0.014 between batch sizes, while
# rows inside one forward are bit-identical), so steered-minus-base differences contain only the intervention.
# c_p = mean over 36 contexts of (log_softmax_steered - log_softmax_base) over the full vocab (fp32 logits).
# PRE-STATED PREDICTIONS (calibration): KL grows ~ quadratically in k for small k; per-layer KL varies by >10x
# at fixed k (late layers larger); k ~ 3-10 needed for median KL ~0.05-0.3.
# CALIBRATION RESULT (N1A_02_calib): median KL at k=30 only 0.001-0.03 per layer; KL ~ k^2 between k=10 and 30;
# a KL floor ~1e-4..1e-3 that does not shrink with k at L10 (bf16 rounding chaos) and split-half rho ~0 at L10-L30
# even when KL is 0.02-0.18. Chosen global kA=60 (expected overall median KL ~0.07), kB=3*kA=180.
# PRE-STATED PREDICTIONS (main): c_+ ~ -c_- (rho(c_+, c_-) < -0.8) at the small k; split-half reliability of c_p
# (even vs odd contexts) is high (rho > 0.7) for most poles, i.e. a pole's causal vocab effect is context-general.
exec(open("./experiments/scripts/N1A_common.py").read())
t_start = time.time()
NCTX = 36
P_SLOTS = 7  # steered slots per forward (+1 base slot)


def n1a_ensure_cache():
    if all(k in STORE for k in ("N1A_ids", "N1A_prev", "N1A_maxabs")):
        return
    ctx = n1a_contexts()
    ids = [lab.ids(t) for _, t in ctx]
    inp, att, last = n1a_pad(ids)
    valid = att.clone().bool(); valid[:, 0] = False
    acts, prevs, hooks = {}, {}, []
    try:
        for l in N1A_LAYERS:
            nidx = torch.arange(21504, device=lab.dev)
            def pre(mod, args, l=l):
                a = args[0].abs() * valid.unsqueeze(-1)
                acts[l] = a.reshape(-1, a.shape[-1]).max(0).values.float().cpu()
            hooks.append(lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
            def post(mod, args, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                prevs[l] = h.detach().cpu()
            hooks.append(lab.layers[l - 1].register_forward_hook(post))
        n1a_forward_last(inp, att, last)
    finally:
        for h in hooks:
            h.remove()
    STORE["N1A_ids"] = ids
    STORE["N1A_prev"] = prevs
    STORE["N1A_maxabs_all"] = acts  # full-width per layer
    STORE["N1A_maxabs"] = {l: acts[l][torch.tensor(N1A_NEURONS[l])] for l in N1A_LAYERS}
    print("rebuilt cache", flush=True)


def maxabs_of(l, n):
    if "N1A_maxabs_all" not in STORE:
        # compute full-width max|act| for layer set (needed for gender neurons outside the sample)
        ids = STORE["N1A_ids"]
        inp, att, last = n1a_pad(ids)
        valid = att.clone().bool(); valid[:, 0] = False
        acts, hooks = {}, []
        try:
            for ll in N1A_LAYERS:
                def pre(mod, args, ll=ll):
                    a = args[0].abs() * valid.unsqueeze(-1)
                    acts[ll] = a.reshape(-1, a.shape[-1]).max(0).values.float().cpu()
                hooks.append(lab.layers[ll].mlp.down_proj.register_forward_pre_hook(pre))
            n1a_forward_last(inp, att, last)
        finally:
            for h in hooks:
                h.remove()
        STORE["N1A_maxabs_all"] = acts
    return float(STORE["N1A_maxabs_all"][l][n])


def spearman_gpu(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


@torch.no_grad()
def run_layer_jobs(l, jobs):
    """jobs: list of (neuron, delta). Returns list of dicts with c (fp32 CPU [V]) and KL stats."""
    ids = STORE["N1A_ids"]
    inp, att, last = n1a_pad(ids)
    prev = STORE["N1A_prev"][l].to(lab.dev)
    out = []
    for c0 in range(0, len(jobs), P_SLOTS):
        chunk = jobs[c0:c0 + P_SLOTS]
        S = len(chunk) + 1
        rows, lasts, neus, ds = [], [], [], []
        for s, (n, d) in enumerate(chunk, start=1):
            rows.append(torch.arange(NCTX, device=lab.dev) + s * NCTX)
            lasts.append(last)
            neus.append(torch.full((NCTX,), n, device=lab.dev, dtype=torch.long))
            ds.append(torch.full((NCTX,), float(d), device=lab.dev))
        rows, lasts, neus, ds = map(lambda x: torch.cat(x), (rows, lasts, neus, ds))

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
        for s, (n, d) in enumerate(chunk, start=1):
            diff = lsm[s] - base                           # [36, V]
            klc = (lsm[s].exp() * diff).sum(-1)            # KL(steered || base) per context
            c = diff.mean(0)
            cA, cB = diff[0::2].mean(0), diff[1::2].mean(0)
            out.append({"n": n, "delta": float(d), "c": c.cpu(), "kl_mean": float(klc.mean()),
                        "kl_median": float(klc.median()), "kl_ctx": [round(float(x), 5) for x in klc.tolist()],
                        "split_rho": spearman_gpu(cA, cB),
                        "base_top1_p_change": float((lsm[s].exp().max(-1).values - base.exp().max(-1).values).abs().mean())})
        del lsm, base, hl
    del prev
    return out


n1a_ensure_cache()
res = {"mode": N1A_MODE, "layers": N1A_RUN_LAYERS, "ks": N1A_KS, "per_layer": {}}
if N1A_MODE == "calib":
    for l in N1A_RUN_LAYERS:
        neus = N1A_NEURONS[l][:4]
        jobs, meta = [], []
        for tag, k in N1A_KS.items():
            for n in neus:
                m = maxabs_of(l, n)
                for s in (1, -1):
                    jobs.append((n, s * k * m)); meta.append((tag, n, s))
        r = run_layer_jobs(l, jobs)
        res["per_layer"][l] = [{"k": tag, "n": n, "s": s, "delta": x["delta"], "kl_mean": x["kl_mean"],
                                "kl_median": x["kl_median"], "split_rho": x["split_rho"]}
                               for (tag, n, s), x in zip(meta, r)]
        summ = {tag: float(torch.tensor([x["kl_mean"] for (t, n, s), x in zip(meta, r) if t == tag]).median())
                for tag in N1A_KS}
        print(f"L{l} median KL by k: {summ}  ({time.time() - t_start:.1f}s)", flush=True)
else:
    gender = torch.load(f"{N1A_CACHE}/gender_static.pt")["poles"] if N1A_GENDER else []
    for l in N1A_RUN_LAYERS:
        plist = [(n, s, "sample") for n in N1A_NEURONS[l] for s in (1, -1)]
        if N1A_GENDER:
            plist += [(g["neuron"], s, "gender") for g in gender if g["layer"] == l for s in (1, -1)]
        # bf16 'chaos floor': a tiny perturbation (k=0.01) of the first sampled neuron, both signs
        plist += [(N1A_NEURONS[l][0], s, "floor") for s in (1, -1)]
        for tag, k in N1A_KS.items():
            jobs = [(n, s * (0.01 if st == "floor" else k) * maxabs_of(l, n)) for n, s, st in plist]
            r = run_layer_jobs(l, jobs)
            C = torch.stack([x["c"] for x in r], 0)  # [P, V] fp32
            meta = [{"n": n, "s": s, "set": st, "delta": x["delta"], "kl_mean": x["kl_mean"], "kl_median": x["kl_median"],
                     "kl_ctx": x["kl_ctx"], "split_rho": x["split_rho"]} for (n, s, st), x in zip(plist, r)]
            torch.save({"poles": meta, "c": C}, f"{N1A_CACHE}/causal_{tag}_L{l}.pt")
            key = f"N1A_causal_{tag}"
            STORE.setdefault(key, {})[l] = C.half()
            kls = torch.tensor([m["kl_mean"] for m in meta if m["set"] == "sample"])
            sr = torch.tensor([m["split_rho"] for m in meta if m["set"] == "sample"])
            fl = [m for m in meta if m["set"] == "floor"]
            res["per_layer"][f"{tag}_L{l}"] = {"kl_median": float(kls.median()), "kl_q10": float(kls.quantile(0.1)),
                                                 "kl_q90": float(kls.quantile(0.9)), "split_rho_median": float(sr.median()),
                                                 "floor_kl": [m["kl_mean"] for m in fl], "floor_split_rho": [m["split_rho"] for m in fl]}
            print(f"L{l} {tag}: KL med {kls.median():.4f} [q10 {kls.quantile(0.1):.4f}, q90 {kls.quantile(0.9):.4f}] "
                  f"split-rho med {sr.median():.3f}  ({time.time() - t_start:.1f}s)", flush=True)
            torch.cuda.empty_cache()
res["elapsed"] = time.time() - t_start
result = res
