# R1_IT_LENS helper (exec'd inside daemon jobs): exact same-position VJPs for K final-space probes in ONE backward pass.
# Trick: h_59[pos] depends on h_l[pos] only through position pos's own residual chain (earlier positions' states are
# constants w.r.t. h_l[pos] by causality). So we (1) run the prefix ids[:pos] without grad into a KV cache, then
# (2) run K copies of token ids[pos], all with position id `pos`, under a custom 4D mask (each copy sees the prefix and
# itself only), with grad through the recompute-in-backward W4A16 linears. Copy k gets objective <h_59[copy k], R_k>,
# so one backward returns J_l(pos)^T R_k for all k and all 60 layers. Checked against lab.same_pos_vjp in R1_00.
import torch as _t
from transformers import DynamicCache as _DC

def _r1_lm(lab):
    return lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model

def r1_prefix_cache(lab, ids, pos):
    lm = _r1_lm(lab)
    cache = _DC(config=lm.config)
    with _t.no_grad():
        lm(input_ids=_t.tensor([list(ids)[:pos]], device=lab.dev), past_key_values=cache, use_cache=True)
    return cache

def r1_vjp_copies(lab, ids, pos, R, cache=None):
    """R: [K, D] final-space probes. Returns G [K, 60, D] float32 on GPU: row (k, l) = d<h_59[pos], R_k>/d resid_post[l][pos]."""
    assert pos >= 1 and pos < 1000, pos
    lm = _r1_lm(lab)
    dev = lab.dev
    K = R.shape[0]
    if cache is None:
        cache = r1_prefix_cache(lab, ids, pos)
    T = cache.get_seq_length()
    assert T == pos, (T, pos)
    dt = lab.WU.dtype
    m = _t.full((K, T + K), _t.finfo(dt).min, dtype=dt, device=dev)
    m[:, :T] = 0
    ar = _t.arange(K, device=dev)
    m[ar, T + ar] = 0
    m = m[None, None]
    masks = {"full_attention": m, "sliding_attention": m}
    pid = _t.full((1, K), pos, dtype=_t.long, device=dev)
    inp = _t.full((1, K), int(ids[pos]), dtype=_t.long, device=dev)
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
    try:
        with _t.enable_grad(), lab.grad_linears():
            lm(input_ids=inp, attention_mask=masks, position_ids=pid, past_key_values=cache, use_cache=True)
            obj = (outs[59][0].float() * R.to(dev).float()).sum()
            grads = _t.autograd.grad(obj, [outs[l] for l in range(60)])
        G = _t.stack([g[0].float() for g in grads], 1)
        H = _t.stack([outs[l][0, 0].detach().float() for l in range(60)], 0)  # [60, D] states at pos (copy 0)
    finally:
        for h in hs:
            h.remove()
        outs.clear()
    return G, H

import time as _time
def r1_vjp_robust(lab, ids, pos, R, min_chunk=16, waits=6):
    """r1_vjp_copies with OOM fallback: split the probe batch in halves (down to min_chunk), then wait and retry.
    (A non-lab GPU process intermittently takes 0.7-2.7 GB.)"""
    K = R.shape[0]
    try:
        return r1_vjp_copies(lab, ids, pos, R)
    except _t.OutOfMemoryError:
        _t.cuda.empty_cache()
    if K > min_chunk:
        h = K // 2
        G1, H = r1_vjp_robust(lab, ids, pos, R[:h], min_chunk, waits)
        G1 = G1.cpu(); _t.cuda.empty_cache()
        G2, _ = r1_vjp_robust(lab, ids, pos, R[h:], min_chunk, waits)
        return _t.cat([G1.to(G2.device), G2], 0), H
    for a in range(waits):
        _time.sleep(10)
        try:
            return r1_vjp_copies(lab, ids, pos, R)
        except _t.OutOfMemoryError:
            _t.cuda.empty_cache()
    raise RuntimeError("persistent OOM")

# ---- LRP-style "linearised" Jacobian (a guess at what J++'s lrp_mode='rlens' does; the fitting code is not available):
# RMSNorm normalisers frozen (detached 1/rms), attention patterns frozen (q and k detached; values still carry gradient),
# GeGLU gate frozen (act(gate) detached; gradient only through up_proj). With these, each layer is linear in its input
# at fixed gates/patterns/norm scales, which is the usual LRP/"relevance" linearisation.
from contextlib import contextmanager as _cm

def _r1_frozen_norm_hook(mod, args, out):
    x = args[0]
    xf = x.float()
    inv = _t.pow(xf.pow(2).mean(-1, keepdim=True) + mod.eps, -0.5).detach()
    y = xf * inv
    if getattr(mod, "with_scale", True):
        y = y * mod.weight.float()
    return y.type_as(out)

def _r1_detach_hook(mod, args, out):
    return out.detach()

@_cm
def r1_lrp_patches(lab):
    hs = []
    for layer in lab.layers:
        for nm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            hs.append(getattr(layer, nm).register_forward_hook(_r1_frozen_norm_hook))
        att = layer.self_attn
        for nm in ("q_norm", "k_norm"):
            if getattr(att, nm, None) is not None:
                hs.append(getattr(att, nm).register_forward_hook(_r1_detach_hook))
        if getattr(att, "v_norm", None) is not None:
            hs.append(att.v_norm.register_forward_hook(_r1_frozen_norm_hook))
        hs.append(layer.mlp.gate_proj.register_forward_hook(_r1_detach_hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()

def r1_total_vjp(lab, ids, dirs, lrp=False):
    """Total-derivative covectors: grad of sum_p <h_59[p], d_k> w.r.t. resid_post[l][p] for ALL p (includes cross-position
    terms d h_59[p'] / d h_l[p], p' > p). Returns [K, 60, T, D] float32 CPU."""
    lm = _r1_lm(lab)
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
    res = []
    try:
        ctx = r1_lrp_patches(lab) if lrp else _nullctx()
        with _t.enable_grad(), lab.grad_linears(), ctx:
            lm(input_ids=_t.tensor([list(ids)], device=lab.dev), use_cache=False)
            H59 = outs[59][0].float()
            K = dirs.shape[0]
            for k in range(K):
                obj = (H59[1:] @ dirs[k].to(lab.dev).float()).sum()
                gr = _t.autograd.grad(obj, [outs[l] for l in range(60)], retain_graph=(k < K - 1))
                res.append(_t.stack([g[0].float() for g in gr], 0).cpu())
    finally:
        for h in hs:
            h.remove()
        outs.clear()
    return _t.stack(res, 0)

@_cm
def _nullctx():
    yield

def r1_vjp_multi(lab, ids, poss, R):
    """Like r1_vjp_copies, but copy k sits at position poss[k] (several positions of ONE sequence per pass).
    Cache = ids[:max(poss)]; copy k attends to cache columns [0, poss[k]) and to itself. Returns G [K,60,D] (GPU)."""
    lm = _r1_lm(lab)
    dev = lab.dev
    K = R.shape[0]
    assert len(poss) == K and min(poss) >= 1 and max(poss) < 1000
    T = max(poss)
    cache = r1_prefix_cache(lab, ids, T)
    dt = lab.WU.dtype
    pt = _t.tensor(list(poss), device=dev)
    cols = _t.arange(T + K, device=dev)
    allow = (cols[None, :] < pt[:, None]) & (cols[None, :] < T)
    allow[_t.arange(K, device=dev), T + _t.arange(K, device=dev)] = True
    m = _t.where(allow, _t.zeros((), dtype=dt, device=dev), _t.full((), _t.finfo(dt).min, dtype=dt, device=dev))[None, None]
    masks = {"full_attention": m, "sliding_attention": m}
    pid = pt[None, :].long()
    inp = _t.tensor([[int(ids[p]) for p in poss]], device=dev)
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
    try:
        with _t.enable_grad(), lab.grad_linears():
            lm(input_ids=inp, attention_mask=masks, position_ids=pid, past_key_values=cache, use_cache=True)
            obj = (outs[59][0].float() * R.to(dev).float()).sum()
            grads = _t.autograd.grad(obj, [outs[l] for l in range(60)])
        G = _t.stack([g[0].float() for g in grads], 1)
    finally:
        for h in hs:
            h.remove()
        outs.clear()
    return G
