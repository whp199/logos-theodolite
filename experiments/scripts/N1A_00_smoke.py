# N1A_00_smoke -- infrastructure checks for N1A (no science claims).
# Checks: GPU headroom; layer identity; baseline forward timing; fp32 vs HF bf16 logits; exactness of the
# run-from-layer-l skip trick (should reproduce full-forward last-position logits up to bf16 batch noise);
# captures per-neuron max|mlp_act| over the 36 contexts (excl. BOS) and caches resid_post[l-1] in STORE.
exec(open("./experiments/scripts/N1A_common.py").read())
res = {}
torch.cuda.empty_cache()
free, tot = torch.cuda.mem_get_info()
res["mem_free_gb"] = free / 1e9
res["torch_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
res["torch_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
print("mem", res)
LM = n1a_LM()
res["layers_identity"] = bool(lab.layers[0] is LM.layers[0]) and bool(lab.layers[30] is LM.layers[30])
print("identity", res["layers_identity"], type(lab.layers[0]))

ctx = n1a_contexts()
ids = [lab.ids(t) for _, t in ctx]
res["ctx_lens"] = [len(x) for x in ids]
print("lens", res["ctx_lens"])
inp, att, last = n1a_pad(ids)
T = inp.shape[1]
valid = att.clone().bool()
valid[:, 0] = False   # exclude BOS position from max-activation statistics

acts, prevs, hooks = {}, {}, []
try:
    for l in N1A_LAYERS:
        nidx = torch.tensor(N1A_NEURONS[l], device=lab.dev)
        def pre(mod, args, l=l, nidx=nidx):
            acts[l] = args[0].index_select(-1, nidx).float().cpu()
        hooks.append(lab.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        def post(mod, args, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            prevs[l] = h.detach().cpu()
        hooks.append(lab.layers[l - 1].register_forward_hook(post))
    torch.cuda.synchronize(); t0 = time.time()
    hb = n1a_forward_last(inp, att, last)
    torch.cuda.synchronize(); res["t_full_forward_s"] = time.time() - t0
finally:
    for h in hooks:
        h.remove()
print("full fwd s", res["t_full_forward_s"], "peak", torch.cuda.max_memory_allocated() / 1e9)
base = n1a_logits_fp32(hb)
ref = lab.next_logits_batch(ids[:4])
res["fp32_vs_hf_bf16_maxabs"] = float((base[:4] - ref).abs().max())
res["fp32_vs_hf_argmax_agree"] = float((base[:4].argmax(-1) == ref.argmax(-1)).float().mean())
print("fp32 vs hf", res["fp32_vs_hf_bf16_maxabs"], res["fp32_vs_hf_argmax_agree"])
res["base_top1"] = [lab.tok.decode([int(t)]) for t in base.argmax(-1).tolist()]
print("top1", res["base_top1"])

# max |act| per neuron over valid positions
vm = valid.cpu().unsqueeze(-1)
maxabs, posfrac, lastact = {}, {}, {}
for l in N1A_LAYERS:
    a = acts[l]  # [B, T, 24]
    aa = (a.abs() * vm).reshape(-1, a.shape[-1])
    m = aa.max(0).values
    maxabs[l] = m
    strong = (a.abs() > 0.5 * m) & vm
    posfrac[l] = (((a > 0) & strong).sum((0, 1)).float() / strong.sum((0, 1)).clamp(min=1)).tolist()
    lastact[l] = a[torch.arange(a.shape[0]), last.cpu()].tolist()
res["maxabs"] = {l: v.tolist() for l, v in maxabs.items()}
res["strong_event_pos_frac"] = posfrac
print("maxabs median per layer", {l: float(v.median()) for l, v in maxabs.items()})

STORE["N1A_ids"] = ids
STORE["N1A_prev"] = prevs
STORE["N1A_maxabs"] = maxabs
STORE["N1A_base_logits_full"] = base.cpu()
STORE["N1A_lastact"] = lastact

# skip-trick exactness at l=30 with 36 rows and with 72 rows (2 copies)
lp_base = F.log_softmax(base, -1)
def kl(lp_s, lp_b):
    return (lp_s.exp() * (lp_s - lp_b)).sum(-1)
for l in (10, 30, 57):
    cp = prevs[l].to(lab.dev)
    torch.cuda.synchronize(); t0 = time.time()
    with N1ARunFrom(l, cp):
        h1 = n1a_forward_last(inp, att, last)
    torch.cuda.synchronize(); t1 = time.time() - t0
    lg1 = n1a_logits_fp32(h1)
    inp2, att2, last2 = inp.repeat(2, 1), att.repeat(2, 1), last.repeat(2)
    torch.cuda.synchronize(); t0 = time.time()
    with N1ARunFrom(l, cp.repeat(2, 1, 1)):
        h2 = n1a_forward_last(inp2, att2, last2)
    torch.cuda.synchronize(); t2 = time.time() - t0
    lg2 = n1a_logits_fp32(h2)
    res[f"skip_l{l}"] = {
        "t36_s": t1, "t72_s": t2,
        "maxabs_logit_diff_36": float((lg1 - base).abs().max()),
        "kl36_median": float(kl(F.log_softmax(lg1, -1), lp_base).median()),
        "kl36_max": float(kl(F.log_softmax(lg1, -1), lp_base).max()),
        "kl72_vs_base_max": float(kl(F.log_softmax(lg2[:36], -1), lp_base).max()),
        "kl72_copy2_vs_copy1_max": float(kl(F.log_softmax(lg2[36:], -1), F.log_softmax(lg2[:36], -1)).max()),
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(l, res[f"skip_l{l}"])
    del cp
torch.cuda.empty_cache()
result = res
