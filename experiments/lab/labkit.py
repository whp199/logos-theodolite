"""labkit: experiment primitives over a loaded TheodoliteRuntime (Gemma-4-31B W4A16 + J++/JLens).

Conventions
- Layers are 0-based (0..59). resid_post[l] = residual stream leaving layer l.
- Lens readout at layer l:  hJ = h @ J[l].T, logits = 30*tanh(RMSNorm_final(hJ) @ W_U.T / 30).
  kind='jpp' (J++ lens, GPU), 'jlens' (Jacobian lens, CPU->GPU per layer), 'naive' (logit lens, J=I).
- Logit differences between two tokens at one position are log-odds under the lens distribution.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from interp_engine import Address

from backend.theodolite_math import D_MODEL, FINAL_LOGIT_SOFTCAP, N_LAYERS, RMS_NORM_EPS

BOS_ID = 2

# Paired gendered token sets (index-aligned female/male pairs) -- single tokens in Gemma-4 vocab.
PRON_PAIRS = [(" she", " he"), (" her", " him"), (" hers", " his"), (" herself", " himself"),
              (" She", " He"), (" Her", " His")]
NOUN_PAIRS = [(" woman", " man"), (" girl", " boy"), (" female", " male"), (" lady", " guy"),
              (" mother", " father"), (" queen", " king"), (" sister", " brother"),
              (" daughter", " son"), (" wife", " husband"), (" women", " men"), (" feminine", " masculine")]
NEUTER = [" it", " its", " itself", " It", " Its"]
PLURAL_THEY = [" they", " them", " their", " themselves", " They"]
ANIMACY_MACHINE = [" machine", " computer", " robot", " AI", " program", " software", " model", " algorithm"]
ANIMACY_HUMAN = [" person", " human", " people", " someone", " individual", " man", " woman", " friend"]


class Lab:
    def __init__(self, rt):
        self.rt = rt
        self.tok = rt.tokenizer
        self.em = rt.eager_model
        self.hf = rt.hf_model
        self.dev = rt.device
        self.WU = rt.lm_head_weight            # [V, D] bf16
        self.fnw = rt.final_norm_weight        # [D]
        self.J = rt.jpp_lenses                 # l -> [D, D] bf16 (GPU)
        self.JL = rt.jlens_lenses              # l -> [D, D] bf16 (CPU)
        self.layers = rt._decoder_layers
        self._hooks: List = []
        self.gender = self._build_gender_ids()

    # ------------------------------------------------------------------ tokenization
    def tid(self, s: str) -> int:
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token: {ids}")
        return ids[0]

    def _build_gender_ids(self):
        g = {}
        g["pron_f"] = [self.tid(f) for f, m in PRON_PAIRS]
        g["pron_m"] = [self.tid(m) for f, m in PRON_PAIRS]
        g["noun_f"] = [self.tid(f) for f, m in NOUN_PAIRS]
        g["noun_m"] = [self.tid(m) for f, m in NOUN_PAIRS]
        g["neuter"] = [self.tid(t) for t in NEUTER]
        g["they"] = [self.tid(t) for t in PLURAL_THEY]
        g["machine"] = [self.tid(t) for t in ANIMACY_MACHINE]
        g["human"] = [self.tid(t) for t in ANIMACY_HUMAN]
        return g

    def ids(self, text: str) -> List[int]:
        """Token ids. Chat strings from `chat()` already start with <bos>; raw text gets BOS prepended."""
        if text.startswith("<bos>"):
            return self.tok.encode(text, add_special_tokens=False)
        return [BOS_ID] + self.tok.encode(text, add_special_tokens=False)

    def chat(self, user: str, system: Optional[str] = None, assistant: str = "",
             history: Optional[List[dict]] = None) -> str:
        """Gemma-4 chat prompt (thinking disabled -> empty thought channel), optional assistant prefix."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user})
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return s + assistant

    def strs(self, ids: Sequence[int]) -> List[str]:
        return [self.tok.decode([i]) for i in ids]

    def find(self, ids: Sequence[int], token: str, after: int = 0, strip: bool = True) -> List[int]:
        """Positions p >= after whose decoded token equals `token` (whitespace-stripped by default)."""
        out = []
        for p in range(after, len(ids)):
            s = self.tok.decode([ids[p]])
            if (s.strip() == token.strip()) if strip else (s == token):
                out.append(p)
        return out

    def model_turn_pos(self, ids: Sequence[int]) -> int:
        """Index of the 'model' token of the LAST '<|turn>model' header."""
        strs = self.strs(ids)
        cand = [p for p in range(1, len(ids)) if strs[p - 1] == "<|turn>" and strs[p].startswith("model")]
        if not cand:
            raise ValueError("no model turn")
        return cand[-1]

    def span_positions(self, ids: Sequence[int], substring: str, after: int = 0) -> List[int]:
        """Token positions covering the LAST occurrence of `substring` in the decoded text after `after`."""
        offs, text = [], ""
        for p in range(len(ids)):
            offs.append(len(text))
            text += self.tok.decode([ids[p]])
        start_char = offs[after] if after < len(offs) else len(text)
        k = text.rfind(substring, start_char)
        if k < 0:
            raise ValueError(f"{substring!r} not found")
        e = k + len(substring)
        return [p for p in range(len(ids)) if offs[p] < e and (offs[p + 1] if p + 1 < len(offs) else len(text)) > k]

    # ------------------------------------------------------------------ capture
    async def capture(self, ids: Sequence[int], points: Sequence[str] = ("resid_post",),
                      layers: Optional[Iterable[int]] = None) -> Dict[str, torch.Tensor]:
        """{point: [L, T, D] bf16 CPU} for each point over all (or given) layers."""
        layers = list(range(N_LAYERS)) if layers is None else list(layers)
        addrs = [Address(pt, l) for pt in points for l in layers]
        caps = await self.em.capture(list(ids), addrs)
        return {pt: torch.stack([caps[Address(pt, l)] for l in layers], 0) for pt in points}

    @torch.no_grad()
    def forward_resid(self, batch_ids: List[List[int]], layers: Optional[Iterable[int]] = None,
                      positions: Optional[List[List[int]]] = None) -> torch.Tensor:
        """Fast batched capture of resid_post via decoder-layer forward hooks (right-padded batch).
        Returns [B, L, P, D] float32 CPU for the requested positions (same count P per item),
        or [B, L, T, D] if positions is None (requires equal lengths)."""
        layers = list(range(N_LAYERS)) if layers is None else list(layers)
        B = len(batch_ids)
        T = max(len(x) for x in batch_ids)
        inp = torch.zeros(B, T, dtype=torch.long)
        att = torch.zeros(B, T, dtype=torch.long)
        for b, x in enumerate(batch_ids):
            inp[b, : len(x)] = torch.tensor(x)
            att[b, : len(x)] = 1
        store: Dict[int, torch.Tensor] = {}
        hs = []
        for l in layers:
            def mk(l):
                def hook(mod, args, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[l] = h.detach()
                return hook
            hs.append(self.layers[l].register_forward_hook(mk(l)))
        try:
            self.hf(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        if positions is None:
            out = torch.stack([store[l] for l in layers], 1)  # [B, L, T, D]
            return out.float().cpu()
        res = []
        for b in range(B):
            pos = torch.tensor(positions[b], device=self.dev)
            res.append(torch.stack([store[l][b].index_select(0, pos) for l in layers], 0))
        return torch.stack(res, 0).float().cpu()

    # ------------------------------------------------------------------ lenses
    def lens(self, h: torch.Tensor, l: int, kind: str = "jpp") -> torch.Tensor:
        """Map layer-l state(s) h [..., D] into final-layer coordinates (bf16 on GPU)."""
        h = h.to(self.dev, torch.bfloat16)
        if kind == "naive" or l == N_LAYERS - 1:
            return h
        if kind == "jpp":
            return h @ self.J[l].T
        if kind == "jlens":
            return h @ self.JL[l].to(self.dev, non_blocking=True).T
        raise ValueError(kind)

    def normed(self, hJ: torch.Tensor) -> torch.Tensor:
        # Same eps rule as backend.theodolite_math.gemma4_rms_norm: tiny direction vectors (ms < 1e-3, e.g. single
        # neuron writes pushed through J) use a 1e-12 floor, otherwise eps would flatten their logits.
        x = hJ.float()
        ms = x.pow(2).mean(-1, keepdim=True)
        eps = torch.where(ms < 1e-3, torch.full_like(ms, 1e-12), torch.full_like(ms, RMS_NORM_EPS))
        return x * torch.rsqrt(ms + eps) * self.fnw.float()

    def tok_logits(self, hJ: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
        """Exact softcapped logits for a token subset: [..., len(token_ids)] float32."""
        W = self.WU[torch.tensor(list(token_ids), device=self.dev)].float()
        z = self.normed(hJ) @ W.T
        return torch.tanh(z / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP

    def full_logits(self, hJ: torch.Tensor) -> torch.Tensor:
        z = self.normed(hJ).to(self.WU.dtype) @ self.WU.T
        return torch.tanh(z.float() / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP

    def topk(self, hJ: torch.Tensor, k: int = 10) -> List[Tuple[str, float]]:
        lg = self.full_logits(hJ.reshape(-1, D_MODEL)[0])
        p = F.softmax(lg, -1)
        v, i = torch.topk(p, k)
        return [(self.tok.decode([int(t)]), round(float(x), 4)) for x, t in zip(v.tolist(), i.tolist())]

    def ranks(self, hJ: torch.Tensor, token_ids: Sequence[int]) -> List[int]:
        lg = self.full_logits(hJ.reshape(-1, D_MODEL)[0])
        return [int((lg > lg[t]).sum().item()) + 1 for t in token_ids]

    # ------------------------------------------------------------------ gender readouts
    def gender_scores(self, hJ: torch.Tensor) -> Dict[str, torch.Tensor]:
        """For final-space states hJ [..., D]: per-position scores (float32, shape [...]).
        pron = mean_i (logit f_i - logit m_i) over pronoun pairs; noun likewise;
        lse_pron = logsumexp(F prons) - logsumexp(M prons); neuter_vs_human = lse(it set) - lse(he+she sets)."""
        g = self.gender
        allids = g["pron_f"] + g["pron_m"] + g["noun_f"] + g["noun_m"] + g["neuter"] + g["they"]
        L = self.tok_logits(hJ, allids)
        n1, n2, n3, n4, n5 = (len(g["pron_f"]), len(g["pron_m"]), len(g["noun_f"]), len(g["noun_m"]), len(g["neuter"]))
        o = 0
        pf = L[..., o:o + n1]; o += n1
        pm = L[..., o:o + n2]; o += n2
        nf = L[..., o:o + n3]; o += n3
        nm = L[..., o:o + n4]; o += n4
        ne = L[..., o:o + n5]; o += n5
        th = L[..., o:]
        hum = torch.cat([pf, pm], -1)
        return {
            "pron": (pf - pm).mean(-1),
            "noun": (nf - nm).mean(-1),
            "lse_pron": torch.logsumexp(pf, -1) - torch.logsumexp(pm, -1),
            "she_he": pf[..., 0] - pm[..., 0],
            "neuter_vs_human": torch.logsumexp(ne, -1) - torch.logsumexp(hum, -1),
            "they_vs_human": torch.logsumexp(th, -1) - torch.logsumexp(hum, -1),
        }

    def gender_direction(self, which: str = "pron") -> torch.Tensor:
        """Final-space effective direction (incl. final-norm weight): mean(fnw*W_U[F]) - mean(fnw*W_U[M])."""
        g = self.gender
        f = self.WU[torch.tensor(g[f"{which}_f"], device=self.dev)].float().mean(0)
        m = self.WU[torch.tensor(g[f"{which}_m"], device=self.dev)].float().mean(0)
        return (f - m) * self.fnw.float()

    def pullback(self, d_final: torch.Tensor, l: int, kind: str = "jpp") -> torch.Tensor:
        """Direction in raw layer-l residual space whose dot with h equals <h J^T, d>: d @ J[l]."""
        if kind == "naive" or l == N_LAYERS - 1:
            return d_final.float()
        Jm = self.J[l] if kind == "jpp" else self.JL[l].to(self.dev)
        return (d_final.to(self.dev, torch.bfloat16) @ Jm).float()

    def readout_layers(self, states: torch.Tensor, kind: str = "jpp", layers: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        """states [L, P, D] (resid_post at the given layers, default all 60) -> {score: [L, P]}."""
        layers = list(range(states.shape[0])) if layers is None else list(layers)
        acc: Dict[str, List[torch.Tensor]] = {}
        for i, l in enumerate(layers):
            sc = self.gender_scores(self.lens(states[i], l, kind))
            for k, v in sc.items():
                acc.setdefault(k, []).append(v.float().cpu())
        return {k: torch.stack(v, 0) for k, v in acc.items()}

    # ------------------------------------------------------------------ hooks / interventions
    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def add_resid_hook(self, l: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        """fn(h [B,T,D]) -> new h, applied to decoder layer l's output (resid_post[l])."""
        def hook(mod, args, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        self._hooks.append(self.layers[l].register_forward_hook(hook))

    def ablate_direction(self, layers: Iterable[int], dirs: Dict[int, torch.Tensor], mean_value: Optional[Dict[int, float]] = None):
        """Project out unit direction dirs[l] from resid_post[l] (optionally set the coefficient to mean_value[l])."""
        for l in layers:
            u = dirs[l].to(self.dev, torch.float32)
            u = u / u.norm()
            mv = 0.0 if mean_value is None else float(mean_value[l])
            def fn(h, u=u, mv=mv):
                hf = h.float()
                c = hf @ u
                return (hf + (mv - c).unsqueeze(-1) * u).to(h.dtype)
            self.add_resid_hook(l, fn)

    def steer_add(self, layers: Iterable[int], dirs: Dict[int, torch.Tensor], alpha: float,
                  unit: bool = True, rel_norm: bool = False, start_pos: int = 0):
        """Add alpha * dir to resid_post[l] at positions >= start_pos (in the current forward's frame).
        rel_norm=True scales alpha by each position's residual norm."""
        for l in layers:
            d = dirs[l].to(self.dev, torch.float32)
            if unit:
                d = d / d.norm()
            def fn(h, d=d):
                hf = h.float()
                if rel_norm:
                    add = alpha * hf.norm(dim=-1, keepdim=True) * d
                else:
                    add = alpha * d
                if start_pos > 0 and hf.shape[1] > 1:
                    mask = torch.zeros(hf.shape[1], 1, device=hf.device)
                    mask[start_pos:] = 1.0
                    add = add * mask
                return (hf + add).to(h.dtype)
            self.add_resid_hook(l, fn)

    # ------------------------------------------------------------------ generation / behavior
    @torch.no_grad()
    def generate(self, prompt: Union[str, List[int]], max_new_tokens: int = 64, n: int = 1,
                 temperature: float = 0.0, top_p: float = 0.95, top_k: int = 64, seed: int = 0) -> List[str]:
        """Batched sampling via HF generate (respects active resid hooks). temperature=0 -> greedy."""
        ids = self.ids(prompt) if isinstance(prompt, str) else list(prompt)
        inp = torch.tensor([ids], device=self.dev)
        torch.manual_seed(seed)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=0)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, num_return_sequences=n)
        else:
            kw.update(do_sample=False)
        out = self.hf.generate(input_ids=inp, attention_mask=torch.ones_like(inp), **kw)
        return [self.tok.decode(o[len(ids):], skip_special_tokens=True) for o in out]

    @torch.no_grad()
    def next_logits(self, prompt: Union[str, List[int]]) -> torch.Tensor:
        """Actual model next-token logits at the last position (float32 [V])."""
        ids = self.ids(prompt) if isinstance(prompt, str) else list(prompt)
        out = self.hf(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
        return out.logits[0, -1].float()

    def next_gender(self, prompt: Union[str, List[int]]) -> Dict[str, float]:
        """Behavioral next-token gender log-odds from the real model output."""
        lg = self.next_logits(prompt)
        g = self.gender
        lp = F.log_softmax(lg, -1)
        P = lambda ids: float(torch.logsumexp(lp[torch.tensor(ids, device=self.dev)], 0).exp())
        return {"p_f_pron": P(g["pron_f"]), "p_m_pron": P(g["pron_m"]), "p_neuter": P(g["neuter"]),
                "p_they": P(g["they"]),
                "lse_pron": float(torch.logsumexp(lg[g["pron_f"]], 0) - torch.logsumexp(lg[g["pron_m"]], 0))}


def stats(xs: Sequence[float]) -> Dict[str, float]:
    import statistics as st
    xs = [float(x) for x in xs]
    n = len(xs)
    m = sum(xs) / n
    sd = st.stdev(xs) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "se": sd / math.sqrt(n) if n > 1 else 0.0, "n": n}


# ---------------------------------------------------------------------------- extensions
def _decompose(self, caps: Dict[str, torch.Tensor], l: int):
    """Corrected layer-l update decomposition (Gemma-4 multiplies each layer output by layer_scalar s):
        resid_post = s*resid_pre + s*attn_write + s*ffn_write
    Returns dict of [T, D] float32 GPU tensors: attn (= s*(mid-pre)), ffn (= post - s*mid), rescale (= (s-1)*pre).
    caps must contain resid_pre/resid_mid/resid_post as [L, T, D] (all 60 layers)."""
    s = self.layers[l].layer_scalar.float().to(self.dev)
    pre = caps["resid_pre"][l].to(self.dev).float()
    mid = caps["resid_mid"][l].to(self.dev).float()
    post = caps["resid_post"][l].to(self.dev).float()
    return {"attn": s * (mid - pre), "ffn": post - s * mid, "rescale": (s - 1) * pre, "s": float(s.mean())}


@torch.no_grad()
def _generate_batch(self, prompts: List[Union[str, List[int]]], max_new_tokens: int = 64,
                    temperature: float = 0.0, top_p: float = 0.95, top_k: int = 64, seed: int = 0,
                    n_per_prompt: int = 1) -> List[List[str]]:
    """Left-padded batched generation over several prompts. Returns [len(prompts)][n_per_prompt] strings."""
    all_ids = [self.ids(p) if isinstance(p, str) else list(p) for p in prompts]
    T = max(len(x) for x in all_ids)
    inp = torch.zeros(len(all_ids), T, dtype=torch.long)
    att = torch.zeros(len(all_ids), T, dtype=torch.long)
    for b, x in enumerate(all_ids):
        inp[b, T - len(x):] = torch.tensor(x)
        att[b, T - len(x):] = 1
    torch.manual_seed(seed)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=0)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, num_return_sequences=n_per_prompt)
    else:
        kw.update(do_sample=False)
        n_per_prompt = 1
    out = self.hf.generate(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), **kw)
    texts = [self.tok.decode(o[T:], skip_special_tokens=True) for o in out]
    return [texts[i * n_per_prompt:(i + 1) * n_per_prompt] for i in range(len(all_ids))]


@torch.no_grad()
def _next_logits_batch(self, prompts: List[Union[str, List[int]]]) -> torch.Tensor:
    """Real next-token logits at each prompt's last position, right-padded batch -> [B, V] float32."""
    all_ids = [self.ids(p) if isinstance(p, str) else list(p) for p in prompts]
    T = max(len(x) for x in all_ids)
    inp = torch.zeros(len(all_ids), T, dtype=torch.long)
    att = torch.zeros(len(all_ids), T, dtype=torch.long)
    for b, x in enumerate(all_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    out = self.hf(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), use_cache=False)
    last = torch.tensor([len(x) - 1 for x in all_ids], device=self.dev)
    return out.logits[torch.arange(len(all_ids), device=self.dev), last].float()


Lab.decompose = _decompose
Lab.generate_batch = _generate_batch
Lab.next_logits_batch = _next_logits_batch


# ---------------------------------------------------------------------------- autograd (true Jacobian VJPs)
from contextlib import contextmanager


class _W4A16LinearFn(torch.autograd.Function):
    """y = x @ deq(W).T with W re-dequantized in backward (never stores the 62GB bf16 weights)."""

    @staticmethod
    def forward(ctx, x, packed, scale):
        from backend.engine_runtime import dequantize_w4a16_full
        w = dequantize_w4a16_full(packed, scale)
        ctx.save_for_backward(packed, scale)
        return F.linear(x.to(w.dtype), w)

    @staticmethod
    def backward(ctx, g):
        from backend.engine_runtime import dequantize_w4a16_full
        packed, scale = ctx.saved_tensors
        w = dequantize_w4a16_full(packed, scale)
        return g.to(w.dtype) @ w, None, None


@contextmanager
def _grad_linears(self):
    """Temporarily swap every W4A16 linear to the recompute-in-backward autograd function."""
    import types
    saved = []
    for _n, mod in self.hf.named_modules():
        if hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
            saved.append((mod, mod.forward))
            def fwd(m, x):
                return _W4A16LinearFn.apply(x, m.weight_packed, m.weight_scale)
            mod.forward = types.MethodType(fwd, mod)
    try:
        yield
    finally:
        for mod, f in saved:
            mod.forward = f


def _same_pos_vjp(self, ids: Sequence[int], pos: int, d_final: torch.Tensor,
                  target: str = "resid59") -> torch.Tensor:
    """True same-position VJP: g_l = (d h_59[pos] / d h_l[pos])^T d_final for l = 0..58, plus l=59 -> d itself.
    target='resid59' differentiates <h_59[pos], d> (the quantity the J-lens linearly approximates).
    Returns [60, D] float32 (row l = gradient wrt resid_post[l][pos]).
    Note: gradients at h_l[pos] here include paths through attention at LATER positions? No: we backprop only
    from position `pos`, and causal attention means h_59[pos] depends on h_l[p<=pos]; we read the gradient at
    h_l[pos] only, i.e. the same-position block of the full Jacobian."""
    outs: Dict[int, torch.Tensor] = {}
    hs = []
    def mk(l):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if l == 0:
                h = h.detach().requires_grad_(True)
            h.retain_grad()
            outs[l] = h
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for l in range(N_LAYERS):
        hs.append(self.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.enable_grad(), _grad_linears(self):
            inp = torch.tensor([list(ids)[: pos + 1]], device=self.dev)
            self.hf.model.language_model(input_ids=inp, use_cache=False) if hasattr(self.hf.model, "language_model") \
                else self.hf.model(input_ids=inp, use_cache=False)
            obj = (outs[N_LAYERS - 1][0, pos].float() * d_final.to(self.dev).float()).sum()
            obj.backward()
            G = torch.stack([outs[l].grad[0, pos].float() for l in range(N_LAYERS)], 0)
    finally:
        for h in hs:
            h.remove()
    return G.detach()


Lab.grad_linears = _grad_linears
Lab.same_pos_vjp = _same_pos_vjp
