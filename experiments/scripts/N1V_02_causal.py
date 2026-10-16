# N1V_02_causal -- (daemon job, resumable, time-budgeted) causal steering ground truth for the FRESH sample.
# Units = (layer, condition); each writes N1V_cache/causal_{cond}_L{l}.pt and is skipped if present.
# Conditions (delta = s * k * max|mlp_act[l,i]| over the 32 fresh contexts, non-BOS positions):
#   A8k60 : add delta to mlp_act[l, p, i] at the LAST 8 positions p (PRIMARY; N1A used the last position only, k=60)
#   A1k60 : same at the last position only (N1A's style, on fresh neurons + fresh contexts -> attributes differences)
#   A8k10 : as A8k60 with k=10 (moderate magnitude)
#   L8k10 : "frozen-norm linear write": add delta * (w_post * W_down[:,i]) / rms(y_p) to the OUTPUT of the post-FFN
#           RMSNorm at the last 8 positions (y_p = unsteered down_proj output). Same first-order write as A8, but the
#           RMSNorm cannot saturate, and the rest of the FFN output is NOT squashed.
#   L8k60 : same with k=60.
#   floor : A8 with k=0.01 on the first neuron (bf16 chaos floor), stored with A8k60.
# Measurement: unsteered base slot in the SAME forward; c_last = mean over contexts of logprob_steered - logprob_base at
# the last position (full vocab, fp32 logits); c8 = same averaged over the 8 window positions; split-half rho (even vs
# odd contexts) of c_last; KL(steered||base) at the last position.
# PRE-STATED PREDICTIONS (verifier, written before any causal result on the fresh sample):
#  V2a A8k60 replicates N1A qualitatively: P@10(JV->C) >> null at L40-57, ~null at L10-30; JV ~ NV.
#  V2b under A8k60 the positive c+/c- correlation at L10-30 (N1A: +0.15..+0.28) is reproduced, but under L8 (no RMSNorm
#      saturation) c+ ~ -c- (rho < -0.3) at L10-30, i.e. the "generic shared response" is partly an intervention artifact.
#  V2c A1 causality check: logprob diffs at window positions before the last are exactly 0.
exec(open("./experiments/scripts/N1V_common.py").read())
exec(open("./experiments/scripts/N1V_lib.py").read())
t0 = time.time()
TB = float(globals().get("N1V_TBUDGET", 170))
ORDER = globals().get("N1V_ORDER", ["A8k60", "L8k10", "A1k60", "A8k10", "L8k60"])
LAYS = globals().get("N1V_RUN_LAYERS", N1V_LAYERS)
SLOTS = int(globals().get("N1V_SLOTS", 9))  # steered slots per forward (+1 base)
CONDS = {"A8k60": ("A", 8, 60.0), "A1k60": ("A", 1, 60.0), "A8k10": ("A", 8, 10.0),
         "L8k10": ("L", 8, 10.0), "L8k60": ("L", 8, 60.0)}
NEU = n1v_load_sample()
if "N1V_prev" not in STORE:
    raise RuntimeError("run N1V_01_static_cache first (STORE cache missing)")
ids = STORE["N1V_ids"]
NC = len(ids)
inp0, att0, last0 = v_pad(ids)
win0 = torch.stack([last0 - k for k in range(NWIN - 1, -1, -1)], 1)  # [NC, 8]
res = {"done": [], "log": []}


@torch.no_grad()
def run_unit(l, cond):
    mode, npos, k = CONDS[cond]
    neus = NEU[l]
    maxabs = STORE["N1V_maxabs"][l]
    plist = [(n, s, "sample", s * k * float(maxabs[j])) for j, n in enumerate(neus) for s in (1, -1)]
    if cond == "A8k60":
        plist += [(neus[0], s, "floor", s * 0.01 * float(maxabs[0])) for s in (1, -1)]
    prev = STORE["N1V_prev"][l].to(lab.dev)
    wpost = rt._post_ffn_norm_weights[l].to(lab.dev).float()
    Wd = {n: rt.slice_w_down(l, torch.tensor([n], device=lab.dev))[0].float() * wpost for n in set(neus)}
    out = []
    c0 = 0
    slots = SLOTS
    while c0 < len(plist):
        chunk = plist[c0:c0 + slots]
        S = len(chunk) + 1
        inp, att = inp0.repeat(S, 1), att0.repeat(S, 1)
        rows_l, pos_l, neu_l, d_l, vec_rows = [], [], [], [], []
        for si, (n, s, st, d) in enumerate(chunk, start=1):
            r = (torch.arange(NC, device=lab.dev) + si * NC).unsqueeze(1).expand(-1, npos)
            p = win0[:, NWIN - npos:]
            rows_l.append(r.reshape(-1)); pos_l.append(p.reshape(-1))
            neu_l.append(torch.full((NC * npos,), n, device=lab.dev, dtype=torch.long))
            d_l.append(torch.full((NC * npos,), float(d), device=lab.dev))
            vec_rows.append(Wd[n].unsqueeze(0).expand(NC * npos, -1) * float(d))
        rows, poss, neu_t, dd = (torch.cat(x) for x in (rows_l, pos_l, neu_l, d_l))
        hk = []
        if mode == "A":
            def pre(mod, args):
                x = args[0]
                x.index_put_((rows, poss, neu_t), dd.to(x.dtype), accumulate=True)
                return (x,) + tuple(args[1:])
            hk.append(lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        else:
            V = torch.cat(vec_rows, 0)  # [N, D] = d * w_post*W_down[:,i]
            normmod = lab.layers[l].post_feedforward_layernorm
            eps = float(getattr(normmod, "eps", 1e-6))
            def post(mod, args, o):
                x = args[0][rows, poss].float()
                rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
                o2 = o.clone()
                o2[rows, poss] = (o[rows, poss].float() + V / rms).to(o.dtype)
                return o2
            hk.append(normmod.register_forward_hook(post))
        try:
            with VRunFrom(l, prev.repeat(S, 1, 1)):
                hs = v_LM()(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
        except torch.OutOfMemoryError:
            for h in hk:
                h.remove()
            hk = []
            torch.cuda.empty_cache()
            if slots <= 2:
                raise
            slots = max(2, slots // 2)
            print(f"OOM -> slots {slots}", flush=True)
            continue
        finally:
            for h in hk:
                h.remove()
        bi = torch.arange(NC, device=lab.dev).unsqueeze(1).expand(-1, NWIN)
        base = F.log_softmax(v_logits_fp32(hs[bi, win0].reshape(-1, hs.shape[-1])), -1).view(NC, NWIN, -1)
        for si, (n, s, st, d) in enumerate(chunk, start=1):
            lsm = F.log_softmax(v_logits_fp32(hs[bi + si * NC, win0].reshape(-1, hs.shape[-1])), -1).view(NC, NWIN, -1)
            diff = lsm - base
            del lsm
            dl = diff[:, -1]
            kl = ((dl + base[:, -1]).exp() * dl).sum(-1)
            cA, cB = dl[0::2].mean(0), dl[1::2].mean(0)
            rec = {"n": n, "s": s, "set": st, "delta": d, "c_last": dl.mean(0).cpu(), "c8": diff.mean((0, 1)).half().cpu(),
                   "split_rho": v_spearman_gpu(cA, cB), "kl_ctx": kl.cpu(), "kl_mean": float(kl.mean()),
                   "pre_window_maxabs": float(diff[:, :-1].abs().max()) if npos == 1 else None}
            out.append(rec)
            del diff, dl
        del hs, base
        c0 += len(chunk)
    del prev
    return out


for cond in ORDER:
    for l in LAYS:
        fn = f"{N1V_CACHE}/causal_{cond}_L{l}.pt"
        if os.path.exists(fn):
            continue
        if time.time() - t0 > TB:
            break
        t1 = time.time()
        r = run_unit(l, cond)
        meta = [{k: v for k, v in x.items() if k not in ("c_last", "c8", "kl_ctx")} for x in r]
        torch.save({"poles": meta, "c_last": torch.stack([x["c_last"] for x in r]), "c8": torch.stack([x["c8"] for x in r]),
                    "kl_ctx": torch.stack([x["kl_ctx"] for x in r])}, fn)
        sm = [m for m in meta if m["set"] == "sample"]
        kls = torch.tensor([m["kl_mean"] for m in sm]); srs = torch.tensor([m["split_rho"] for m in sm])
        fl = [m for m in meta if m["set"] == "floor"]
        res["log"].append({"cond": cond, "l": l, "s": round(time.time() - t1, 1), "kl_med": float(kls.median()),
                           "split_rho_med": float(srs.median()), "floor_kl": [m["kl_mean"] for m in fl],
                           "floor_rho": [m["split_rho"] for m in fl],
                           "prewin": max([m["pre_window_maxabs"] or 0 for m in sm])})
        res["done"].append(f"{cond}_L{l}")
        print(res["log"][-1], flush=True)
        torch.cuda.empty_cache()
res["elapsed"] = time.time() - t0
res["remaining"] = [f"{c}_L{l}" for c in ORDER for l in LAYS if not os.path.exists(f"{N1V_CACHE}/causal_{c}_L{l}.pt")]
result = res
