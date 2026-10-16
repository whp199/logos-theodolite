# N1V_01_static_cache -- (daemon job) fresh-sample static labels + steering cache + pipeline sanity checks.
# Fresh sample = N1V_neurons.json (seed 99, 8 neurons x 8 layers, disjoint from N1). Contexts = N1V_common (32, new).
# Static labels per neuron (sign +; '-' pole = negation), all tool-exact (gemma4_rms_norm -> bf16 -> fp32 GEMM -> softcap):
#   JV  = unembed((W_down[:,i]*w_post) @ J++[l].T)   (tool J-VINDEX)      JL = same via JLens      NV = unembed(W_down[:,i]*w_post)
#   G   = unembed(W_gate[i] @ J++[l].T)               (tool gate_trigger_tokens)
#   Gemb= W_U @ (W_gate[i]*w_pre)  (direct embedding readout, tied W_E)    Uemb = W_U @ (W_up[i]*w_pre)
# Null directions for the label-tightness test: 32 single other neurons, 32 Gaussian 512-neuron mixtures, 32 isotropic.
# Cache: resid_post[l-1] for the 32 contexts, max|mlp_act| (non-BOS) of each fresh neuron, ||down_proj out|| at the last 8
# positions (to quantify post-FFN-RMSNorm saturation under k*max|act| steering).
# PRE-STATED PREDICTIONS (verifier):
#  V1a run-from-layer-l replay reproduces the full forward's final hidden state (max abs diff < 1e-2, bf16).
#  V1b SATURATION: a steering delta of 60*max|act| on one neuron is LARGER than the whole FFN output at that position
#      (median ratio r60 = 60*max|act|*||W_down[:,i]|| / ||y|| > 1) at most layers, so the post-FFN RMSNorm caps the
#      write and, in effect, replaces the FFN output with the neuron direction (a confound shared by all neurons).
#  V1c no MEMIT overlays are active on the sampled layers.
exec(open("./experiments/scripts/N1V_common.py").read())
exec(open("./experiments/scripts/N1V_lib.py").read())
t0 = time.time()
NEU = n1v_load_sample()
res = {"checks": {}, "sat": {}, "mass": {}}
hooks = []
try:
    with torch.no_grad():
        WU = lab.WU
        # ---------------------------------------------------------------- static labels
        g = torch.Generator(device=lab.dev).manual_seed(99099)
        gc = torch.Generator().manual_seed(99099)
        old = json.load(open(f"{N1V_RES}/N1_neurons.json"))["neurons"]
        tight_ids = {}
        for l in N1V_LAYERS:
            idx = torch.tensor(NEU[l], device=lab.dev)
            mlp = lab.layers[l].mlp
            res["checks"][f"overlay_L{l}"] = bool(getattr(mlp.down_proj, "_memit_col_overlays", None)) or \
                bool(getattr(mlp.gate_proj, "_memit_row_overlays", None))
            J = lab.J[l]
            wpost = rt._post_ffn_norm_weights[l].to(lab.dev).float()
            wpre = rt._pre_ffn_norm_weights[l].to(lab.dev).float()
            wd = rt.slice_w_down(l, idx)
            v = (wd.float() * wpost).to(J.dtype)
            JLm = lab.JL[l].to(lab.dev, torch.bfloat16)
            st = {"neurons": NEU[l],
                  "JV": v_unembed_tool(v @ J.T).cpu(),
                  "JL": v_unembed_tool(v @ JLm.T).cpu(),
                  "NV": v_unembed_tool(v).cpu(),
                  "wd_norm": wd.float().norm(dim=-1).cpu(),
                  "wpost_wd_norm": (wd.float() * wpost).norm(dim=-1).cpu()}
            wg = rt.slice_w_gate(l, idx)
            wu = rt.slice_w_up(l, idx)
            st["G"] = v_unembed_tool(wg.to(J.dtype) @ J.T).cpu()
            st["Gemb"] = torch.cat([(wg.float() * wpre) @ WU[s:s + 8192].float().T for s in range(0, WU.shape[0], 8192)], -1).cpu()
            st["Uemb"] = torch.cat([(wu.float() * wpre) @ WU[s:s + 8192].float().T for s in range(0, WU.shape[0], 8192)], -1).cpu()
            p = F.softmax(st["JV"].float(), -1)
            res["mass"][f"JV_top10_L{l}"] = float(p.topk(10, -1).values.sum(-1).median())
            torch.save(st, f"{N1V_CACHE}/static_L{l}.pt")
            # null directions (tightness test)
            excl = set(NEU[l]) | set(int(i) for i in old[str(l)])
            pool = [i for i in torch.randperm(D_FF_, generator=gc).tolist() if i not in excl]
            Wm = rt.slice_w_down(l, torch.tensor(pool[:512], device=lab.dev)).float() * wpost
            vmix = torch.randn(32, 512, device=lab.dev, generator=g) @ Wm
            vsing = rt.slice_w_down(l, torch.tensor(pool[512:544], device=lab.dev)).float() * wpost
            viso = torch.randn(32, Wm.shape[1], device=lab.dev, generator=g)
            rec = {}
            for nm, vv in (("mix512", vmix), ("single", vsing), ("iso", viso)):
                vv = vv.to(J.dtype)
                rec[nm + "_jpp"] = v_unembed_tool(vv @ J.T).topk(10, -1).indices.cpu().tolist()
                rec[nm + "_naive"] = v_unembed_tool(vv).topk(10, -1).indices.cpu().tolist()
            tight_ids[str(l)] = rec
            del Wm, JLm
            torch.cuda.empty_cache()
            print(f"static L{l} {time.time() - t0:.1f}s  JV top: {[lab.tok.decode([int(t)]) for t in st['JV'][0].topk(5).indices]}", flush=True)
        json.dump(tight_ids, open(f"{N1V_CACHE}/null_dirs_top10.json", "w"))

        # ---------------------------------------------------------------- steering cache + saturation stats
        ctx = n1v_contexts_text(lambda u, a: lab.chat(u, assistant=a))
        ids = [lab.ids(t) for _, t in ctx]
        inp, att, last = v_pad(ids)
        valid = att.clone().bool(); valid[:, 0] = False
        B = len(ids)
        win = torch.stack([last - k for k in range(NWIN - 1, -1, -1)], 1)  # [B, 8] last-8 positions
        bidx = torch.arange(B, device=lab.dev).unsqueeze(1).expand(-1, NWIN)
        maxabs, ynorm, prevs, actwin = {}, {}, {}, {}
        for l in N1V_LAYERS:
            idx = torch.tensor(NEU[l], device=lab.dev)
            def pre(mod, args, l=l, idx=idx):
                a = args[0][..., idx].float()
                maxabs[l] = (a.abs() * valid.unsqueeze(-1)).reshape(-1, len(idx)).max(0).values.cpu()
                actwin[l] = a[bidx, win].cpu()  # [B, 8, 8neu]
            def post_dp(mod, args, out, l=l):
                ynorm[l] = out[bidx, win].float().norm(dim=-1).cpu()  # [B, 8]
            def post_prev(mod, args, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                prevs[l] = h.detach().cpu()
            hooks.append(lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
            hooks.append(lab.layers[l].mlp.down_proj.register_forward_hook(post_dp))
            hooks.append(lab.layers[l - 1].register_forward_hook(post_prev))
        out = v_LM()(input_ids=inp, attention_mask=att, use_cache=False)
        for h in hooks:
            h.remove()
        hooks = []
        hs_full = out.last_hidden_state[torch.arange(B, device=lab.dev), last].clone()
        STORE["N1V_ids"] = ids
        STORE["N1V_prev"] = prevs
        STORE["N1V_maxabs"] = maxabs
        torch.save({"ids": ids, "maxabs": maxabs, "ynorm": ynorm, "actwin": actwin, "kinds": [k for k, _ in ctx]},
                   f"{N1V_CACHE}/ctx_cache.pt")
        for l in N1V_LAYERS:
            st = torch.load(f"{N1V_CACHE}/static_L{l}.pt")
            wn = st["wd_norm"]  # [8]
            nat = (maxabs[l] * wn).unsqueeze(0).unsqueeze(0) / ynorm[l].unsqueeze(-1)  # [B,8pos,8neu] natural max write / ||y||
            res["sat"][f"L{l}"] = {"nat_ratio_median": float(nat.median()),
                                   "r10_median": float((10 * nat).median()), "r60_median": float((60 * nat).median()),
                                   "r60_q10": float((60 * nat).quantile(0.1)), "r60_q90": float((60 * nat).quantile(0.9)),
                                   "frac_r60_gt1": float((60 * nat > 1).float().mean()),
                                   "frac_r10_gt1": float((10 * nat > 1).float().mean()),
                                   "maxabs": [round(float(x), 3) for x in maxabs[l]],
                                   "ynorm_median": float(ynorm[l].median())}
            print(f"sat L{l}: {res['sat'][f'L{l}']}", flush=True)

        # ---------------------------------------------------------------- sanity checks
        # (a) last_hidden_state is final-normed: compare our fp32 logits with HF logits at the last position
        hf_lg = lab.hf(input_ids=inp[:4], attention_mask=att[:4], use_cache=False).logits
        hf_last = hf_lg[torch.arange(4, device=lab.dev), last[:4]].float()
        mine = v_logits_fp32(hs_full[:4])
        res["checks"]["hf_vs_mine_logit_maxabs"] = float((hf_last - mine).abs().max())
        res["checks"]["hf_vs_mine_argmax_agree"] = bool((hf_last.argmax(-1) == mine.argmax(-1)).all())
        del hf_lg
        # (b) run-from-layer-l replay exactness
        for l in (10, 40, 57):
            with VRunFrom(l, prevs[l].to(lab.dev)):
                o2 = v_LM()(input_ids=inp, attention_mask=att, use_cache=False)
            h2 = o2.last_hidden_state[torch.arange(B, device=lab.dev), last]
            res["checks"][f"runfrom_L{l}_maxabs"] = float((h2.float() - hs_full.float()).abs().max())
            del o2
        print("checks", res["checks"], flush=True)
finally:
    for h in hooks:
        h.remove()
res["elapsed"] = time.time() - t0
result = res
