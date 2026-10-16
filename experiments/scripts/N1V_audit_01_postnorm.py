# N1V_audit_01_postnorm -- verifier (code audit) daemon job. Tests whether the N1A causal ground truth at L10-30 is
# confounded by the post-feedforward RMSNorm that sits between down_proj and the residual add in Gemma-4:
#   resid += post_ffn_norm(down_proj(mlp_act))  ->  adding delta = s*k*max|act| to mlp_act[l,last,i] (N1A protocol,
#   k=60/180) changes y = down_proj output by delta*W_down[:,i]; if ||delta*W_down[:,i]|| ~ ||y||, the norm RESCALES the
#   whole MLP output, so the residual change contains -post_norm(y) (an ablation of the rest of the MLP output at that
#   position) that is shared by every neuron and both signs.
# PRE-STATED PREDICTIONS (written before running):
#  V1 At k=60, ||delta*W_down_i|| / ||y_last|| is >~1 at L10-30 (saturation: ||Delta|| / ||linear prediction|| << 1 and
#     cos(Delta, -post_norm(y)) substantially > 0), and smaller at L40-57.
#  V2 The 'ablation' intervention (post_norm(y_last) := 0) gives a c that correlates with the symmetric part
#     (c+ + c-)/2 of the cached N1A kA vectors at L10-30.
#  V3 Residual-direct steering along s*w_post*W_down_i (bypassing the norm), with norm matched per context to the k=60
#     residual change, gives c+ ~ -c- (rho(c+,c-) < 0) at L20-30 instead of N1A's +0.27/+0.31; whether static labels
#     then predict c at L20-30 is open (computed offline by N1V_audit_02_metrics.py).
# Also checks: rt.slice_w_down == down_proj applied to one-hot inputs; rt._post_ffn_norm_weights == module weight.
exec(open("./experiments/scripts/N1A_common.py").read())
import time
t0 = time.time()
OUTD = f"{N1A_ROOT}/results/N1V_cache"
os.makedirs(OUTD, exist_ok=True)
NCTX = 36
LAY = [10, 20, 30, 40, 45, 50, 55, 57]
STEER = [int(x) for x in globals().get("N1V_STEER", [20, 30, 40, 10])]
KS = [1, 3, 10, 30, 60, 180]
SLOTS = 8  # incl. base slot
res = {"diag": {}, "checks": {}, "steer": {}}
ctx = n1a_contexts()
ids = [lab.ids(t) for _, t in ctx]
inp, att, last = n1a_pad(ids)
valid = att.clone().bool(); valid[:, 0] = False
ar = torch.arange(NCTX, device=lab.dev)
cap, hooks = {}, []
try:
    for l in LAY:
        lay = lab.layers[l]
        nidx = torch.tensor(N1A_NEURONS[l], device=lab.dev)
        def pre(mod, args, l=l, nidx=nidx):
            a = args[0][..., nidx].abs() * valid.unsqueeze(-1)
            cap[("max", l)] = a.reshape(-1, a.shape[-1]).max(0).values.float()
            cap[("actlast", l)] = args[0][ar, last][:, nidx].float()
        def fwd(mod, args, out, l=l):
            cap[("y", l)] = out[ar, last].float()
        def pn(mod, args, out, l=l):
            cap[("pn", l)] = out[ar, last].float()
        hooks.append(lay.mlp.down_proj.register_forward_pre_hook(pre))
        hooks.append(lay.mlp.down_proj.register_forward_hook(fwd))
        hooks.append(lay.post_feedforward_layernorm.register_forward_hook(pn))
        if l in STEER:
            def prevh(mod, args, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                cap[("prev", l)] = h.detach().clone()
            hooks.append(lab.layers[l - 1].register_forward_hook(prevh))
        def resh(mod, args, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            cap[("resid", l)] = h[ar, last].float()
        hooks.append(lay.register_forward_hook(resh))
    with torch.no_grad():
        n1a_forward_last(inp, att, last)
finally:
    for h in hooks:
        h.remove()
print(f"base fwd {time.time()-t0:.1f}s", flush=True)

alpha = {}  # (l) -> [48 poles, 36] residual-change norm at k=60 (pole order: neuron-major, +,-)
with torch.no_grad():
    for l in LAY:
        lay = lab.layers[l]
        nidx = torch.tensor(N1A_NEURONS[l], device=lab.dev)
        Wd = rt.slice_w_down(l, nidx).float()                       # [24, D]
        pnm = lay.post_feedforward_layernorm
        wpost = pnm.weight.float()
        eps = float(pnm.eps)
        if l == LAY[0] or l == 40:
            # W_down slice vs module on one-hot inputs; post_ffn weight vs rt copy
            oh = torch.zeros(3, 21504, device=lab.dev, dtype=torch.bfloat16)
            for r in range(3):
                oh[r, int(N1A_NEURONS[l][r])] = 1.0
            mo = lay.mlp.down_proj(oh.unsqueeze(0))[0].float()
            res["checks"][f"wdown_onehot_rel_err_L{l}"] = float(((mo - Wd[:3]).norm(dim=-1) / Wd[:3].norm(dim=-1)).max())
            res["checks"][f"wpost_vs_rt_maxabs_L{l}"] = float((wpost - rt._post_ffn_norm_weights[l].to(lab.dev).float()).abs().max())
        def pnorm(v):
            return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * wpost
        y = cap[("y", l)]                                           # [36, D]
        pny = pnorm(y)
        res["checks"][f"pnorm_recon_rel_err_L{l}"] = float(((pny - cap[("pn", l)]).norm(dim=-1) / cap[("pn", l)].norm(dim=-1)).max())
        mx = cap[("max", l)]                                        # [24]
        rms_y = y.pow(2).mean(-1, keepdim=True).add(eps).sqrt()     # [36,1]
        d0 = {"y_norm_med": float(y.norm(dim=-1).median()), "pn_norm_med": float(pny.norm(dim=-1).median()),
              "resid_norm_med": float(cap[("resid", l)].norm(dim=-1).median()),
              "maxact_med": float(mx.median()),
              "act_last_over_max_med": float((cap[("actlast", l)].abs() / mx).median())}
        for k in KS:
            agg = {"ratio_d_y": [], "sat": [], "cos_dir": [], "cos_abl": [], "rel_change": []}
            al = []
            for s in (1, -1):
                d = s * k * mx[:, None] * Wd                             # [24, D]
                Dl = pnorm(y[None] + d[:, None]) - pny[None]             # [24,36,D]
                lin = (wpost * d)[:, None] / rms_y[None]                 # [24,36,D]
                dirv = (s * wpost * Wd)[:, None].expand_as(Dl)
                agg["ratio_d_y"].append((d.norm(dim=-1)[:, None] / y.norm(dim=-1)[None]).flatten())
                agg["sat"].append((Dl.norm(dim=-1) / lin.norm(dim=-1)).flatten())
                agg["cos_dir"].append(F.cosine_similarity(Dl, dirv, dim=-1).flatten())
                agg["cos_abl"].append(F.cosine_similarity(Dl, -pny[None].expand_as(Dl), dim=-1).flatten())
                agg["rel_change"].append((Dl.norm(dim=-1) / pny.norm(dim=-1)[None]).flatten())
                al.append(Dl.norm(dim=-1))                               # [24,36]
            d0[f"k{k}"] = {kk: {"median": float(torch.cat(v).median()), "q10": float(torch.cat(v).quantile(0.1)),
                                "q90": float(torch.cat(v).quantile(0.9))} for kk, v in agg.items()}
            if k == 60:
                alpha[l] = torch.stack([al[0], al[1]], 1).reshape(48, NCTX)  # neuron-major, (+,-)
        res["diag"][str(l)] = d0
        print(l, {k: (round(d0[f'k{k}']['ratio_d_y']['median'], 3), round(d0[f'k{k}']['sat']['median'], 3),
                      round(d0[f'k{k}']['cos_abl']['median'], 3)) for k in KS}, flush=True)
        del Wd
STORE["N1V_alpha60"] = {l: v.cpu() for l, v in alpha.items()}
result = res


@torch.no_grad()
def steer_layer(l):
    """slot 0 base; slot 1 ablate post_norm(y_last); slots 2.. residual-direct steering for 48 poles."""
    lay = lab.layers[l]
    nidx = torch.tensor(N1A_NEURONS[l], device=lab.dev)
    Wd = rt.slice_w_down(l, nidx).float()
    wpost = lay.post_feedforward_layernorm.weight.float()
    U = wpost * Wd
    U = U / U.norm(dim=-1, keepdim=True)                              # [24, D] unit static write directions
    poles = [(ni, s) for ni in range(24) for s in (1, -1)]
    jobs = [("abl", None, None)] + [("dir", ni, s) for ni, s in poles]
    prev = cap[("prev", l)]
    outs = []
    for c0 in range(0, len(jobs), SLOTS - 1):
        chunk = jobs[c0:c0 + SLOTS - 1]
        S = len(chunk) + 1
        rows, lasts, addv, abl_rows = [], [], [], []
        for si, (kind, ni, s) in enumerate(chunk, start=1):
            r = ar + si * NCTX
            if kind == "abl":
                abl_rows.append(r)
            else:
                pi = 2 * ni + (0 if s == 1 else 1)
                rows.append(r); lasts.append(last)
                addv.append(s * alpha[l][pi][:, None] * U[ni][None])    # [36, D]
        rows_t = torch.cat(rows) if rows else None
        lasts_t = torch.cat(lasts) if lasts else None
        addv_t = torch.cat(addv) if addv else None
        abl_t = torch.cat(abl_rows) if abl_rows else None
        def hk(mod, args, out):
            out = out.clone()
            if rows_t is not None:
                out.index_put_((rows_t, lasts_t), addv_t.to(out.dtype), accumulate=True)
            if abl_t is not None:
                out[abl_t, last] = 0
            return out
        h = lay.post_feedforward_layernorm.register_forward_hook(hk)
        try:
            with N1ARunFrom(l, prev.repeat(S, 1, 1)):
                hl = n1a_forward_last(inp.repeat(S, 1), att.repeat(S, 1), last.repeat(S))
        finally:
            h.remove()
        lsm = F.log_softmax(n1a_logits_fp32(hl), -1).view(S, NCTX, -1)
        base = lsm[0]
        for si, job in enumerate(chunk, start=1):
            diff = lsm[si] - base
            klc = (lsm[si].exp() * diff).sum(-1)
            outs.append({"job": job, "c": diff.mean(0).cpu(), "cA": diff[0::2].mean(0).cpu(), "cB": diff[1::2].mean(0).cpu(),
                         "kl_mean": float(klc.mean()), "kl_median": float(klc.median())})
        del lsm, base, hl
    torch.save({"poles": [o["job"] for o in outs], "c": torch.stack([o["c"] for o in outs]),
                "cA": torch.stack([o["cA"] for o in outs]).half(), "cB": torch.stack([o["cB"] for o in outs]).half(),
                "kl": [o["kl_mean"] for o in outs]}, f"{OUTD}/steer_dir_L{l}.pt")
    kl_dir = torch.tensor([o["kl_mean"] for o in outs if o["job"][0] == "dir"])
    return {"kl_abl": outs[0]["kl_mean"], "kl_dir_median": float(kl_dir.median()), "kl_dir_q90": float(kl_dir.quantile(0.9))}


for l in STEER:
    if time.time() - t0 > 170:
        res["steer"][str(l)] = "skipped (time)"
        continue
    res["steer"][str(l)] = steer_layer(l)
    print("steer", l, res["steer"][str(l)], f"{time.time()-t0:.1f}s", flush=True)
    torch.cuda.empty_cache()
cap.clear()
torch.cuda.empty_cache()
res["elapsed"] = time.time() - t0
result = res
