# E3_lens_validation helper: batched, multi-direction same-position VJPs (exec'd inside daemon jobs).
# Each batch item b is truncated to pos_b+1 tokens and right-padded; the objective for direction k is
# sum_b <h_59[b,pos_b], d_k>; items are independent, and causal masking means padding after pos_b cannot
# affect h[b,pos_b]. Gradient read at h_l[b,pos_b] = same-position block of the Jacobian (as labkit.same_pos_vjp).
import torch as _t

def e3_vjp_batch(lab, seqs, poss, dirs):
    """seqs: list of token-id lists; poss: list of positions; dirs: [K, D] final-space directions.
    Returns [K, B, 60, D] float32 CPU."""
    B = len(seqs)
    seqs = [list(s)[: p + 1] for s, p in zip(seqs, poss)]
    T = max(len(s) for s in seqs)
    inp = _t.zeros(B, T, dtype=_t.long); att = _t.zeros(B, T, dtype=_t.long)
    for b, s in enumerate(seqs):
        inp[b, :len(s)] = _t.tensor(s); att[b, :len(s)] = 1
    outs = {}
    hs = []
    def mk(l):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if l == 0:
                h = h.detach().requires_grad_(True)
            outs[l] = h
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for l in range(60):
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    pidx = _t.tensor(poss, device=lab.dev); bidx = _t.arange(B, device=lab.dev)
    res = []
    try:
        with _t.enable_grad(), lab.grad_linears():
            lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
            H59 = outs[59][bidx, pidx].float()
            dirs = dirs.to(lab.dev).float()
            K = dirs.shape[0]
            for k in range(K):
                obj = (H59 @ dirs[k]).sum()
                grads = _t.autograd.grad(obj, [outs[l] for l in range(60)], retain_graph=(k < K - 1))
                res.append(_t.stack([g[bidx, pidx].float() for g in grads], 1).cpu())
    finally:
        for h in hs:
            h.remove()
        outs.clear()
    return _t.stack(res, 0)
