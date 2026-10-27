"""The atlas's live operations (`backend/atlas_ops.py`): push a neuron and read what the model writes, read where it fires,
read the model's own odds under a push, and read the wiring between neurons and attention heads from the weights.

After the neuron explorer of lyramakesmusic/neuron-explorer (`explorer/runtime.py`), rebuilt for Gemma 4. Its model is
Qwen3.5-4B, whose feed-forward output joins the stream as it is; Gemma 4 puts a norm around each block and scales each
layer's output (transformers `Gemma4TextDecoderLayer`):

    mid  = pre + post_attention_layernorm(attn(input_layernorm(pre)))
    post = s * (mid + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(mid))))      s = layer_scalar

with every norm `x / rms(x) * w` (the unit offset is already in w), the feed-forward block GeGLU,
`a = gelu_tanh(W_gate x) * (W_up x)`, `mlp = W_down a`, and attention with q_norm, k_norm and an unweighted v_norm,
scaling 1, sliding-window layers and full layers whose values are the normed keys (`attention_k_eq_v`). So:

- A push (`Push`) has three modes. "add" adds `dose * p50 * u` to the stream leaving the layer at every word-piece,
  u = unit(w_post (.) W_down[:, n]) the neuron's write direction as the stream receives it and p50 the median length of
  that stream (`stream_p50`): the reference's "add" in its units of the residual's median norm, placed after the norm and
  the scalar, which a push on the activation itself would pass through (the norm squashes a large one: erratum E47).
  "mul" multiplies the neuron's own activation (0 silences it, 2 doubles it): exact, through the model's own norm.
  "act" adds `dose * scale` to the activation, scale the neuron's typical largest |a| (`activation_extremes`): the
  causal check's activation write at every word-piece. A whole layer's feed-forward block takes "mul" only.
- Every push, and every head scaled (`HeadScale`), leaves the start mark alone unless told not to (`spare_bos`): the
  nudge does the same (V7), since the first position is the attention sink every later one reads.
- What a neuron or a head writes into the stream is read through the norm that follows it, which is linear once its
  input's rms is fixed: a head's part of the attention output, `w_pa (.) W_O[:, h] y_h / rms(o)`, and a neuron's part of
  the feed-forward output, `w_post (.) W_down[:, n] a / rms(y)`, add up exactly to what the norm writes.

Everything here is pure torch on the text model (`causal_check.language_model`), batch-first, under `torch.no_grad`;
the caller holds the GPU lock. Layers count from 0, as every field the server sends. scripts/check_atlas_ops.py checks
each function against brute force on a tiny random Gemma 4 on the CPU.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.causal_check import language_model, logits_fp32, model_device

PUSH_MODES = ("add", "mul", "act")
# The three voices of the push chat: pushed up, as it is, pushed down (the reference's rows 0, 1, 2).
THREE_VOICES = (1, 0, -1)
MAX_PUSHES = 32
MAX_HEAD_SCALES = 64


# ------------------------------------------------------------------------------------------------- the model's shape


def dense_weight(linear: nn.Module) -> torch.Tensor:
    """A linear layer's weight [out, in] as the model uses it: dequantized from W4A16 where the checkpoint packs it
    (`engine_runtime.dequantize_w4a16_full`), else the plain weight."""
    if hasattr(linear, "weight_packed") and hasattr(linear, "weight_scale"):
        from backend.engine_runtime import dequantize_w4a16_full

        return dequantize_w4a16_full(linear.weight_packed, linear.weight_scale)
    return linear.weight.detach()


def ffn_width(lm: nn.Module, layer: int = 0) -> int:
    down = lm.layers[int(layer)].mlp.down_proj
    n = getattr(down, "in_features", None)
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        return int(n)
    return 8 * int(down.weight_packed.shape[1])


def hidden_size(lm: nn.Module) -> int:
    return int(lm.norm.weight.shape[0])


@dataclass
class AttnShape:
    layer: int
    kind: str  # "sliding" | "full"
    n_heads: int
    n_kv: int
    head_dim: int
    values_from_keys: bool  # attention_k_eq_v on a full layer: the values are the normed keys
    window: Optional[int]

    @property
    def group(self) -> int:
        return self.n_heads // self.n_kv

    def kv_of(self, head: int) -> int:
        return int(head) // self.group


def attn_shape(lm: nn.Module, layer: int) -> AttnShape:
    a = lm.layers[int(layer)].self_attn
    hd = int(a.head_dim)
    n_out = getattr(a.o_proj, "in_features", None) or 8 * int(a.o_proj.weight_packed.shape[1])
    n_k = getattr(a.k_proj, "out_features", None) or int(a.k_proj.weight_packed.shape[0])
    return AttnShape(
        layer=int(layer),
        kind="sliding" if getattr(a, "is_sliding", False) else "full",
        n_heads=int(n_out) // hd,
        n_kv=int(n_k) // hd,
        head_dim=hd,
        values_from_keys=getattr(a, "v_proj", None) is None,
        window=int(a.sliding_window) if getattr(a, "sliding_window", None) else None,
    )


def layer_scalars(lm: nn.Module) -> torch.Tensor:
    """[L] float32: each layer's output scalar s_l (1 where a layer has none)."""
    out = []
    for layer in lm.layers:
        s = getattr(layer, "layer_scalar", None)
        out.append(float(s.reshape(-1)[0]) if s is not None else 1.0)
    return torch.tensor(out, dtype=torch.float32)


def softcap_of(lm_or_hf: nn.Module) -> Optional[float]:
    cfg = getattr(lm_or_hf, "config", None)
    cfg = getattr(cfg, "text_config", None) or cfg
    cap = getattr(cfg, "final_logit_softcapping", None)
    return float(cap) if cap else None


class Weights:
    """The weights the wiring and the pushes read, as the model now uses them: through the runtime's accessors when one
    is given (W4A16 dequantized on the GPU, the vault's edits included), else from the plain Linear modules (the CPU
    stand-in). Norm weights already hold Gemma's unit offset (`Gemma4RMSNorm` multiplies by w)."""

    def __init__(self, lm: nn.Module, runtime: Any = None) -> None:
        self.lm = lm
        self.runtime = runtime
        self.n_layers = len(lm.layers)
        self.d_model = hidden_size(lm)
        self.d_ff = ffn_width(lm, 0)
        self.device = model_device(lm)
        self.scalars = layer_scalars(lm)

    def norm(self, layer: int, which: str) -> torch.Tensor:
        """`which` is input | post_attention | pre_feedforward | post_feedforward; float32 [D]."""
        return getattr(self.lm.layers[int(layer)], f"{which}_layernorm").weight.detach().float()

    def eps(self, layer: int, which: str) -> float:
        return float(getattr(self.lm.layers[int(layer)], f"{which}_layernorm").eps)

    def w_down(self, layer: int) -> torch.Tensor:
        """[D, N]"""
        if self.runtime is not None:
            return self.runtime.get_full_w_down(int(layer))
        return self.lm.layers[int(layer)].mlp.down_proj.weight.detach()

    def w_gate(self, layer: int) -> torch.Tensor:
        """[N, D]"""
        if self.runtime is not None:
            return self.runtime.get_full_w_gate(int(layer))
        return self.lm.layers[int(layer)].mlp.gate_proj.weight.detach()

    def w_up(self, layer: int) -> torch.Tensor:
        """[N, D]"""
        if self.runtime is not None:
            return self.runtime.get_full_w_up(int(layer))
        return self.lm.layers[int(layer)].mlp.up_proj.weight.detach()

    def down_cols(self, layer: int, neurons: Sequence[int]) -> torch.Tensor:
        """[K, D] float32: W_down[:, n] for each n (vault edits included)."""
        idx = torch.tensor([int(n) for n in neurons], dtype=torch.long, device=self.device)
        if self.runtime is not None:
            return self.runtime.slice_w_down(int(layer), idx).float()
        return self.lm.layers[int(layer)].mlp.down_proj.weight.detach()[:, idx].T.float()

    def gate_rows(self, layer: int, neurons: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor([int(n) for n in neurons], dtype=torch.long, device=self.device)
        if self.runtime is not None:
            return self.runtime.slice_w_gate(int(layer), idx).float()
        return self.lm.layers[int(layer)].mlp.gate_proj.weight.detach()[idx].float()

    def up_rows(self, layer: int, neurons: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor([int(n) for n in neurons], dtype=torch.long, device=self.device)
        if self.runtime is not None:
            return self.runtime.slice_w_up(int(layer), idx).float()
        return self.lm.layers[int(layer)].mlp.up_proj.weight.detach()[idx].float()

    def write_dirs(self, layer: int, neurons: Sequence[int]) -> torch.Tensor:
        """[K, D] float32: each neuron's write direction as the stream receives it, w_post (.) W_down[:, n], unit length
        (a silenced neuron's zero column stays zero)."""
        v = self.down_cols(layer, neurons) * self.norm(layer, "post_feedforward").to(self.device)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)

    def attn_weight(self, layer: int, which: str) -> torch.Tensor:
        """q | k | v | o projection weight of the layer's attention, dense [out, in] (v is k on a layer whose values are
        its keys)."""
        a = self.lm.layers[int(layer)].self_attn
        if which == "v" and getattr(a, "v_proj", None) is None:
            which = "k"
        return dense_weight(getattr(a, f"{which}_proj"))


# ---------------------------------------------------------------------------------------------- pushes and head scales


@dataclass
class Push:
    layer: int
    neuron: Optional[int]  # None: the whole feed-forward block of the layer ("mul" only)
    mode: str = "add"
    dose: float = 0.8


@dataclass
class HeadScale:
    layer: int
    head: Optional[int]  # None: every head of the layer
    scale: float = 0.0  # 0 silences it


def check_pushes(lm: nn.Module, pushes: Sequence[Push], heads: Sequence[HeadScale] = ()) -> None:
    """Refuse, in plain words, what cannot be pushed (layers and neurons and heads counted from 0)."""
    L = len(lm.layers)
    if len(pushes) > MAX_PUSHES:
        raise ValueError(f"At most {MAX_PUSHES} pushes go into one reading, not {len(pushes)}.")
    if len(heads) > MAX_HEAD_SCALES:
        raise ValueError(f"At most {MAX_HEAD_SCALES} heads can be scaled in one reading, not {len(heads)}.")
    for p in pushes:
        if not 0 <= int(p.layer) < L:
            raise ValueError(f"Layer {p.layer} does not exist: layers count from 0 to {L - 1}.")
        if p.mode not in PUSH_MODES:
            raise ValueError(f'A push\'s mode is "add", "mul" or "act", not "{p.mode}".')
        if not math.isfinite(float(p.dose)):
            raise ValueError("A push's dose must be a finite number.")
        if p.neuron is None:
            if p.mode != "mul":
                raise ValueError('A whole layer\'s feed-forward block takes only mode "mul" (dose 0 silences it).')
        elif not 0 <= int(p.neuron) < ffn_width(lm, p.layer):
            raise ValueError(f"Neuron {p.neuron} does not exist: layer {p.layer} has {ffn_width(lm, p.layer)} (0 to {ffn_width(lm, p.layer) - 1}).")
    for h in heads:
        if not 0 <= int(h.layer) < L:
            raise ValueError(f"Layer {h.layer} does not exist: layers count from 0 to {L - 1}.")
        n = attn_shape(lm, h.layer).n_heads
        if h.head is not None and not 0 <= int(h.head) < n:
            raise ValueError(f"Layer {h.layer} has {n} attention heads (0 to {n - 1}), not head {h.head}.")
        if not math.isfinite(float(h.scale)):
            raise ValueError("A head's scale must be a finite number.")


@dataclass
class Interventions:
    """Pushes and head scales resolved to per-row values for a batch of `rows` rows:
    - adds[l]: [B, D] float32 added to the stream leaving layer l;
    - acts[l]: [(neuron, [B] float32 added to its activation)];
    - muls[l]: [(neuron or None, [B] float32 factors)];
    - heads[l]: [(head or None, [B] float32 factors on the head's output)]."""

    rows: int
    adds: Dict[int, torch.Tensor] = field(default_factory=dict)
    acts: Dict[int, List[Tuple[int, torch.Tensor]]] = field(default_factory=dict)
    muls: Dict[int, List[Tuple[Optional[int], torch.Tensor]]] = field(default_factory=dict)
    heads: Dict[int, List[Tuple[Optional[int], torch.Tensor]]] = field(default_factory=dict)

    def empty(self) -> bool:
        return not (self.adds or self.acts or self.muls or self.heads)


def resolve(
    weights: Weights,
    pushes: Sequence[Push],
    heads: Sequence[HeadScale],
    signs: Sequence[int],
    p50: Callable[[int], float],
    act_scale: Callable[[int, int], float],
) -> Interventions:
    """Each push and head scale as per-row values, one row per entry of `signs`: +1 pushed as asked, 0 left alone, -1
    pushed the other way. "add": sign * dose * p50(l) along the neuron's write direction; "act": sign * dose *
    act_scale(l, n) on its activation; "mul": the factor dose where the sign is +1, 1 where it is 0, and -dose where it is
    -1 (the reference's down row, x -dose); a head's factor is its scale wherever the sign is not 0."""
    dev = weights.device
    B = len(signs)
    sg = torch.tensor([float(s) for s in signs], dtype=torch.float32, device=dev)
    out = Interventions(rows=B)
    for p in pushes:
        l, d = int(p.layer), float(p.dose)
        if p.mode == "add":
            u = weights.write_dirs(l, [int(p.neuron)])[0]  # [D]
            delta = (sg * d * float(p50(l)))[:, None] * u[None, :]
            out.adds[l] = out.adds[l] + delta if l in out.adds else delta
        elif p.mode == "act":
            out.acts.setdefault(l, []).append((int(p.neuron), sg * d * float(act_scale(l, int(p.neuron)))))
        else:
            f = torch.where(sg > 0, torch.full_like(sg, d), torch.where(sg < 0, torch.full_like(sg, -d), torch.ones_like(sg)))
            out.muls.setdefault(l, []).append((None if p.neuron is None else int(p.neuron), f))
    for h in heads:
        f = torch.where(sg != 0, torch.full_like(sg, float(h.scale)), torch.ones_like(sg))
        out.heads.setdefault(int(h.layer), []).append((None if h.head is None else int(h.head), f))
    return out


class _Keep:
    """Which positions the interventions act on in the forward under way: [B, T] bool, or None for all of them."""

    def __init__(self) -> None:
        self.mask: Optional[torch.Tensor] = None


@contextmanager
def installed(
    lm: nn.Module,
    iv: Optional[Interventions],
    keep: _Keep,
    reads: Sequence[Tuple[int, int]] = (),
    seen: Optional[Dict[Tuple[int, int], List[torch.Tensor]]] = None,
) -> Iterator[None]:
    """Install the interventions (and reads of single neurons' activations) as hooks for the length of the block. A read
    is taken before any push at its own layer, so only what comes before the feed-forward block (earlier layers, and the
    layer's own attention) can move it; each forward appends [B, T] float32 to `seen[(layer, neuron)]`."""
    hooks = []
    keep_of = lambda x: keep.mask if keep.mask is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)  # noqa: E731
    try:
        for l, n in reads:

            def read(_m: nn.Module, args: Tuple[Any, ...], l: int = l, n: int = n) -> None:
                assert seen is not None
                seen.setdefault((l, n), []).append(args[0][:, :, n].detach().float().cpu())

            hooks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(read, prepend=True))
        if iv is not None:
            for l in sorted(set(iv.acts) | set(iv.muls)):
                acts, muls = iv.acts.get(l, []), iv.muls.get(l, [])

                def push_act(_m: nn.Module, args: Tuple[Any, ...], acts=acts, muls=muls) -> Tuple[Any, ...]:
                    x = args[0].clone()
                    k = keep_of(x)
                    for n, f in muls:
                        ff = torch.where(k, f[:, None].to(x.device), torch.ones((), device=x.device)).to(x.dtype)
                        if n is None:
                            x = x * ff[:, :, None]
                        else:
                            x[:, :, n] = x[:, :, n] * ff
                    for n, v in acts:
                        vv = torch.where(k, v[:, None].to(x.device), torch.zeros((), device=x.device)).to(x.dtype)
                        x[:, :, n] = x[:, :, n] + vv
                    return (x,) + tuple(args[1:])

                hooks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(push_act))
            for l, delta in iv.adds.items():

                def push_add(_m: nn.Module, _args: Tuple[Any, ...], out: Any, delta: torch.Tensor = delta) -> Any:
                    h = out[0] if isinstance(out, tuple) else out
                    k = keep_of(h).to(h.dtype)
                    h2 = h + (k[:, :, None] * delta[:, None, :].to(device=h.device, dtype=h.dtype))
                    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

                hooks.append(lm.layers[l].register_forward_hook(push_add))
            for l, scales in iv.heads.items():
                hd = attn_shape(lm, l).head_dim

                def scale_heads(_m: nn.Module, args: Tuple[Any, ...], scales=scales, hd: int = hd) -> Tuple[Any, ...]:
                    x = args[0].clone()
                    k = keep_of(x)
                    v = x.view(*x.shape[:-1], x.shape[-1] // hd, hd)
                    for h, f in scales:
                        ff = torch.where(k, f[:, None].to(x.device), torch.ones((), device=x.device)).to(x.dtype)
                        if h is None:
                            v.mul_(ff[:, :, None, None])
                        else:
                            v[:, :, h, :] = v[:, :, h, :] * ff[:, :, None]
                    return (x,) + tuple(args[1:])

                hooks.append(lm.layers[l].self_attn.o_proj.register_forward_pre_hook(scale_heads))
        yield
    finally:
        for h in hooks:
            h.remove()


def keep_mask(ids: torch.Tensor, attention: Optional[torch.Tensor], bos_id: Optional[int], spare_bos: bool) -> torch.Tensor:
    """[B, T] bool: the positions a push acts on in a prefill of (left-padded) `ids`: every real word-piece but, with
    `spare_bos`, each row's first one when it is the start mark."""
    real = attention.bool() if attention is not None else torch.ones_like(ids, dtype=torch.bool)
    if not spare_bos or bos_id is None:
        return real
    first = real & (real.long().cumsum(-1) == 1)
    return real & ~(first & (ids == int(bos_id)))


# ------------------------------------------------------------------------------------------------------------ writing


@dataclass
class Step:
    row: int
    token_id: int
    logprob: float  # the written piece's own log-probability (float32 softmax of the logits it was chosen from)
    top_ids: List[int]
    top_lps: List[float]
    end: bool  # an end mark: the row stops after it


def _model_logits(lm: nn.Module, W_U: torch.Tensor, h: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    """The model's own logits for final-normed states h [B, D]: the lm_head in the model's dtype, then the softcap, as
    `Gemma4ForCausalLM.forward` computes them (the instrument's writing chooses from these)."""
    logits = F.linear(h.to(W_U.dtype), W_U.to(h.device))
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


@torch.no_grad()
def write_rows(
    lm: nn.Module,
    W_U: torch.Tensor,
    rows_ids: Sequence[Sequence[int]],
    iv: Optional[Interventions],
    *,
    max_new: int,
    stop_ids: Sequence[int],
    bos_id: Optional[int],
    spare_bos: bool = True,
    softcap: Optional[float] = 30.0,
    top_k: int = 0,
    cancel: Optional[Callable[[], bool]] = None,
) -> Iterator[Step]:
    """Write greedily from several prompts at once, each row under its own pushes (`iv`, one value per row), left-padded
    in one batch with a cache: the next piece of every row unfinished, in order, step after step. A row stops after the
    first of `stop_ids` it writes; the batch stops at `max_new` or when `cancel()` turns true. The start mark of each row
    is spared by the pushes (`spare_bos`); the pieces written are never the start mark. With `top_k`, each step carries
    the k likeliest pieces and their log-probabilities."""
    dev = model_device(lm)
    B = len(rows_ids)
    if iv is not None and iv.rows != B:
        raise ValueError(f"{iv.rows} rows of pushes for {B} prompts.")
    T = max(len(r) for r in rows_ids)
    pad = 0
    ids = torch.tensor([[pad] * (T - len(r)) + [int(t) for t in r] for r in rows_ids], dtype=torch.long, device=dev)
    mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows_ids], dtype=torch.long, device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    stop = {int(t) for t in stop_ids}
    done = [False] * B
    keep = _Keep()
    keep.mask = keep_mask(ids, mask, bos_id, spare_bos)
    with installed(lm, iv, keep):
        out = lm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state[:, -1]
        del out
        for _ in range(int(max_new)):
            if cancel is not None and cancel():
                return
            logits = _model_logits(lm, W_U, h, softcap)
            nxt = logits.argmax(-1)
            lsm = F.log_softmax(logits.float(), dim=-1)
            if top_k:
                tv, ti = torch.topk(lsm, k=int(top_k), dim=-1)
            for r in range(B):
                if done[r]:
                    continue
                t = int(nxt[r])
                end = t in stop
                yield Step(
                    row=r,
                    token_id=t,
                    logprob=float(lsm[r, t]),
                    top_ids=[int(x) for x in ti[r].tolist()] if top_k else [],
                    top_lps=[round(float(x), 4) for x in tv[r].tolist()] if top_k else [],
                    end=end,
                )
                if end:
                    done[r] = True
            if all(done):
                return
            keep.mask = None  # a written piece is never the start mark
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
            pos = pos[:, -1:] + 1
            out = lm(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
            del out


class TextDeltas:
    """Each row's written pieces turned into text as they come, holding back a piece that ends inside a character (a
    split multi-byte character decodes to U+FFFD until it is whole), as the reference's steer stream does."""

    def __init__(self, tokenizer: Any, rows: int) -> None:
        self.tok = tokenizer
        self.ids: List[List[int]] = [[] for _ in range(rows)]
        self.sent = [""] * rows

    def add(self, row: int, token_id: int, end: bool, skip_special: bool = True) -> str:
        if not end:
            self.ids[row].append(int(token_id))
        text = self.tok.decode(self.ids[row], skip_special_tokens=skip_special)
        if text.endswith("�") and not end:
            return ""
        delta = text[len(self.sent[row]) :] if text.startswith(self.sent[row]) else text
        self.sent[row] = text
        return delta


# ----------------------------------------------------------------------------------------------------------- reading


@torch.no_grad()
def read_neuron(
    lm: nn.Module,
    ids: Sequence[int],
    layer: int,
    neuron: int,
    iv: Optional[Interventions] = None,
    *,
    bos_id: Optional[int] = None,
    spare_bos: bool = True,
) -> List[float]:
    """The neuron's activation at every word-piece of `ids` (one row), under the interventions given (one row)."""
    dev = model_device(lm)
    x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=dev)
    keep = _Keep()
    keep.mask = keep_mask(x, None, bos_id, spare_bos)
    seen: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    with installed(lm, iv, keep, reads=[(int(layer), int(neuron))], seen=seen):
        lm(input_ids=x, use_cache=False)
    return [float(v) for v in seen[(int(layer), int(neuron))][0][0].tolist()]


@torch.no_grad()
def next_logprobs(
    lm: nn.Module,
    W_U: torch.Tensor,
    ids: Sequence[int],
    iv: Optional[Interventions] = None,
    *,
    bos_id: Optional[int] = None,
    spare_bos: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The next piece's log-probabilities after `ids` ([V] float32 on the CPU, from fp32 logits with Gemma 4's softcap
    of 30, the study's measure: `causal_check.logits_fp32`), and with interventions the same under them. The two are rows
    of one forward (`iv` must hold two rows, the first left alone: `resolve(..., signs=(0, 1))`), so they differ by the
    interventions alone (rows of one forward are bit-identical, where different batch shapes were not: N1A)."""
    dev = model_device(lm)
    R = 1 if iv is None or iv.empty() else 2
    if R == 2 and iv.rows != 2:
        raise ValueError("next_logprobs takes the base row and the intervened row: resolve with signs (0, 1).")
    x = torch.tensor([[int(t) for t in ids]] * R, dtype=torch.long, device=dev)
    keep = _Keep()
    keep.mask = keep_mask(x, None, bos_id, spare_bos)
    with installed(lm, iv if R == 2 else None, keep):
        out = lm(input_ids=x, use_cache=False)
    h = out.last_hidden_state[:, -1]
    lsm = F.log_softmax(logits_fp32(h, W_U), dim=-1).cpu()
    return lsm[0], (lsm[1] if R == 2 else None)


@torch.no_grad()
def all_activations(lm: nn.Module, ids: Sequence[int], dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Every neuron's activation at every word-piece: [L, T, N] on the CPU (float16 by default: 60 x 21,504 values a
    word-piece on Gemma-4-31B is 2.6 MB at float16)."""
    dev = model_device(lm)
    x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=dev)
    acts: Dict[int, torch.Tensor] = {}
    hooks = []
    try:
        for l, layer in enumerate(lm.layers):

            def pre(_m: nn.Module, args: Tuple[Any, ...], l: int = l) -> None:
                acts[l] = args[0][0].detach().to("cpu", dtype=dtype)

            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        lm(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([acts[l] for l in range(len(lm.layers))])


@dataclass
class Extremes:
    """Each neuron's extremes over the word-pieces read (every one but the start mark of each text): `pos` its largest
    activation, `neg` its most negative, and the four sign regimes of GeGLU, gelu_tanh(g) * u: `pp` (g > 0, u > 0) and
    `nn` (g < 0, u < 0, the double negative) for the positive side, `pn` (g > 0, u < 0) and `np` (g < 0, u > 0) for the
    negative; all [L, N] float32, 0 where never reached. `tokens` counts the word-pieces read; `resid_norms[l]` the length
    of the stream leaving layer l at each of them."""

    pos: torch.Tensor
    neg: torch.Tensor
    pp: torch.Tensor
    nn: torch.Tensor
    pn: torch.Tensor
    np: torch.Tensor
    tokens: int = 0
    resid_norms: List[List[float]] = field(default_factory=list)


@torch.no_grad()
def activation_extremes(lm: nn.Module, texts_ids: Sequence[Sequence[int]], *, keep_norms: bool = True) -> Extremes:
    """`Extremes` over `texts_ids`, read one text at a time (positions from 1: the start mark is the attention sink,
    with outsized activations, as the reference's neuron_stats leaves it out)."""
    L, N = len(lm.layers), ffn_width(lm, 0)
    z = lambda: torch.zeros(L, N, dtype=torch.float32)  # noqa: E731
    ex = Extremes(pos=z(), neg=z(), pp=z(), nn=z(), pn=z(), np=z(), resid_norms=[[] for _ in range(L)])
    dev = model_device(lm)
    gate: Dict[int, torch.Tensor] = {}
    up: Dict[int, torch.Tensor] = {}
    hooks = []
    try:
        for l, layer in enumerate(lm.layers):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(lambda _m, _a, o, l=l: gate.__setitem__(l, o)))
            hooks.append(layer.mlp.up_proj.register_forward_hook(lambda _m, _a, o, l=l: up.__setitem__(l, o)))

            def pre(_m: nn.Module, args: Tuple[Any, ...], l: int = l) -> None:
                a = args[0][0, 1:].float()
                g, u = gate[l][0, 1:].float(), up[l][0, 1:].float()
                if a.shape[0] == 0:
                    return
                ninf = torch.tensor(float("-inf"), device=a.device)
                pinf = torch.tensor(float("inf"), device=a.device)
                gp, upos = g > 0, u > 0
                ex.pos[l] = torch.maximum(ex.pos[l], a.amax(0).clamp(min=0).cpu())
                ex.neg[l] = torch.minimum(ex.neg[l], a.amin(0).clamp(max=0).cpu())
                ex.pp[l] = torch.maximum(ex.pp[l], torch.where(gp & upos, a, ninf).amax(0).clamp(min=0).cpu())
                ex.nn[l] = torch.maximum(ex.nn[l], torch.where(~gp & ~upos, a, ninf).amax(0).clamp(min=0).cpu())
                ex.pn[l] = torch.minimum(ex.pn[l], torch.where(gp & ~upos, a, pinf).amin(0).clamp(max=0).cpu())
                ex.np[l] = torch.minimum(ex.np[l], torch.where(~gp & upos, a, pinf).amin(0).clamp(max=0).cpu())

            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            if keep_norms:

                def post(_m: nn.Module, _a: Tuple[Any, ...], out: Any, l: int = l) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    ex.resid_norms[l].extend(h[0, 1:].float().norm(dim=-1).cpu().tolist())

                hooks.append(layer.register_forward_hook(post))
        for ids in texts_ids:
            x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=dev)
            lm(input_ids=x, use_cache=False)
            ex.tokens += max(0, len(ids) - 1)
            gate.clear()
            up.clear()
    finally:
        for h in hooks:
            h.remove()
    return ex


def stream_p50(ex: Extremes) -> List[float]:
    """The median length of the stream leaving each layer over the word-pieces `ex` read (the add push's unit)."""
    return [float(np.median(v)) if v else float("nan") for v in ex.resid_norms]


@torch.no_grad()
def forward_from_layer(
    lm: nn.Module, start: int, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor], position_ids: torch.Tensor
) -> torch.Tensor:
    """The text model's forward from layer `start` on: `hidden` [B, T, D] is the stream entering that layer (leaving
    `start` - 1), and layers start.. and the final norm run as `Gemma4TextModel.forward` runs them without a cache (its
    masks for both kinds of layer, its rotary embeddings per kind, the shared key-value states). Returns the final-normed
    states [B, T, D]. For a model with per-layer inputs or layers sharing keys and values (not Gemma-4-31B) it refuses:
    those need the earlier layers. Hooks on the layers run as in a whole forward."""
    from collections import UserDict

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cfg = lm.config
    if getattr(cfg, "hidden_size_per_layer_input", 0) or getattr(cfg, "num_kv_shared_layers", 0):
        raise ValueError("forward_from_layer needs a model without per-layer inputs or shared key-value layers.")
    kw = {"config": cfg, "inputs_embeds": hidden, "attention_mask": attention_mask, "past_key_values": None,
          "position_ids": position_ids}
    masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
    pe = {t: lm.rotary_emb(hidden, position_ids, t) for t in set(cfg.layer_types)}
    shared: Any = UserDict()
    h = hidden
    for i in range(int(start), int(cfg.num_hidden_layers)):
        t = cfg.layer_types[i]
        h = lm.layers[i](h, None, shared_kv_states=shared, position_embeddings=pe[t], attention_mask=masks[t],
                         position_ids=position_ids, past_key_values=None)
    return lm.norm(h)


# ------------------------------------------------------------------------------------------------- display units


def token_units(tokenizer: Any, ids: Sequence[int]) -> List[Tuple[str, List[int]]]:
    """Word-pieces grouped into what can be shown: the pieces of one character split over several (an emoji, a CJK
    character in byte pieces) join the piece that completes it, and emoji joiners, variation selectors and skin tones
    join the unit before them (the reference's `token_units`). [(text, [indices])]."""
    units: List[Tuple[str, List[int]]] = []
    pending: List[int] = []
    prev = ""
    for i in range(len(ids)):
        full = tokenizer.decode([int(t) for t in ids[: i + 1]])
        pending.append(i)
        if full.endswith("�") and i < len(ids) - 1:
            continue
        piece = full[len(prev) :] if full.startswith(prev) else tokenizer.decode([int(ids[j]) for j in pending])
        units.append((piece, pending))
        prev, pending = full, []
    glue = {"‍", "️", "︎", "⃣"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
    merged: List[Tuple[str, List[int]]] = []
    for text, idx in units:
        if merged and text and (text[0] in glue or merged[-1][0].endswith("‍")):
            merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
        else:
            merged.append((text, idx))
    return merged


# --------------------------------------------------------------------------------------------------------- attention


@dataclass
class HeadCapture:
    """One text read with every attention head kept, per layer l (each [B=1] dropped; float32 on the CPU):
    q[l] [H, T, hd] and k[l] [KV, T, hd] after their norms and the rotary embedding, v[l] [KV, T, hd] after v_norm,
    y[l] [H, T, hd] each head's output (o_proj's input), o_rms[l] [T] the rms the post-attention norm divides by,
    writes[l] [T, H] each head's write into the stream leaving the layer, s * |w_pa (.) W_O[:, h] y_h| / rms(o);
    with `streams`, mid[l] [T, D] the stream entering the feed-forward block, y_mlp[l] [T, D] the feed-forward output
    before its norm, and embed [T, D] the scaled embedding (what `attention_drive` reads)."""

    ids: List[int]
    q: Dict[int, torch.Tensor] = field(default_factory=dict)
    k: Dict[int, torch.Tensor] = field(default_factory=dict)
    v: Dict[int, torch.Tensor] = field(default_factory=dict)
    y: Dict[int, torch.Tensor] = field(default_factory=dict)
    o_rms: Dict[int, torch.Tensor] = field(default_factory=dict)
    writes: Dict[int, torch.Tensor] = field(default_factory=dict)
    mid: Dict[int, torch.Tensor] = field(default_factory=dict)
    y_mlp: Dict[int, torch.Tensor] = field(default_factory=dict)
    embed: Optional[torch.Tensor] = None
    acts: Optional[torch.Tensor] = None  # [L, T, N] float16, when asked for

    @property
    def T(self) -> int:
        return len(self.ids)


@torch.no_grad()
def capture_heads(
    lm: nn.Module, ids: Sequence[int], weights: Weights, *, streams: bool = False, with_acts: bool = False
) -> HeadCapture:
    """Read `ids` once and keep what the head readings need (`HeadCapture`), from hooks on each layer's attention (its
    rotary cos/sin), q_norm, k_norm, v_norm and o_proj (input and output); with `streams`, the pre- and post-feed-forward
    norms' inputs and the embedding; with `with_acts`, every neuron's activation too."""
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

    dev = model_device(lm)
    x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=dev)
    cap = HeadCapture(ids=[int(t) for t in ids])
    rope: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    qn: Dict[int, torch.Tensor] = {}
    kn: Dict[int, torch.Tensor] = {}
    acts: Dict[int, torch.Tensor] = {}
    s = weights.scalars
    hooks = []
    try:
        if streams:
            hooks.append(lm.embed_tokens.register_forward_hook(lambda _m, _a, o: setattr(cap, "embed", o[0].detach().float().cpu())))
        for l, layer in enumerate(lm.layers):
            a = layer.self_attn

            def attn_pre(_m: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], l: int = l) -> None:
                cos, sin = kwargs["position_embeddings"]
                rope[l] = (cos.detach(), sin.detach())

            hooks.append(a.register_forward_pre_hook(attn_pre, with_kwargs=True))
            hooks.append(a.q_norm.register_forward_hook(lambda _m, _a, o, l=l: qn.__setitem__(l, o.detach())))
            if hasattr(a, "k_norm"):
                hooks.append(a.k_norm.register_forward_hook(lambda _m, _a, o, l=l: kn.__setitem__(l, o.detach())))
                hooks.append(
                    a.v_norm.register_forward_hook(
                        lambda _m, _a, o, l=l: cap.v.__setitem__(l, o[0].detach().float().transpose(0, 1).cpu())
                    )
                )

            def o_pre(_m: nn.Module, args: Tuple[Any, ...], l: int = l, hd: int = int(a.head_dim)) -> None:
                y = args[0][0].detach().float()
                cap.y[l] = y.view(y.shape[0], -1, hd).transpose(0, 1).cpu()

            def o_post(_m: nn.Module, _args: Tuple[Any, ...], out: torch.Tensor, l: int = l) -> None:
                o = out[0].detach().float()
                cap.o_rms[l] = (o.pow(2).mean(-1) + weights.eps(l, "post_attention")).sqrt().cpu()

            hooks.append(a.o_proj.register_forward_pre_hook(o_pre))
            hooks.append(a.o_proj.register_forward_hook(o_post))
            if streams:
                hooks.append(
                    layer.pre_feedforward_layernorm.register_forward_hook(
                        lambda _m, args, _o, l=l: cap.mid.__setitem__(l, args[0][0].detach().float().cpu())
                    )
                )
                hooks.append(
                    layer.post_feedforward_layernorm.register_forward_hook(
                        lambda _m, args, _o, l=l: cap.y_mlp.__setitem__(l, args[0][0].detach().float().cpu())
                    )
                )
            if with_acts:
                hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        lambda _m, args, l=l: acts.__setitem__(l, args[0][0].detach().to("cpu", dtype=torch.float16))
                    )
                )
        lm(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    for l in range(len(lm.layers)):
        cos, sin = rope[l]
        q = apply_rotary_pos_emb(qn[l], cos, sin, unsqueeze_dim=2)  # [1, T, H, hd]
        cap.q[l] = q[0].float().transpose(0, 1).cpu()
        if l in kn:
            k = apply_rotary_pos_emb(kn[l], cos, sin, unsqueeze_dim=2)
            cap.k[l] = k[0].float().transpose(0, 1).cpu()
        # each head's write into the stream leaving the layer, |w_pa (.) W_O[:, h] y_h|, from the head's Gram matrix
        # W_O[:, h]^T diag(w_pa^2) W_O[:, h] [hd, hd], so no [T, H, D] tensor is formed
        y = cap.y[l]  # [H, T, hd]
        H, T, hd = y.shape
        W_O = weights.attn_weight(l, "o")  # [D, H*hd], on the device
        Mw = (W_O.float() * weights.norm(l, "post_attention").to(W_O.device)[:, None]).view(-1, H, hd)
        G = torch.einsum("dhi,dhj->hij", Mw, Mw).cpu()  # [H, hd, hd]
        del Mw
        q2 = torch.einsum("hti,hij,htj->th", y, G, y).clamp(min=0)
        cap.writes[l] = float(s[l]) * q2.sqrt() / cap.o_rms[l][:, None]
    if with_acts:
        cap.acts = torch.stack([acts[l] for l in range(len(lm.layers))])
    return cap


def head_pattern(cap: HeadCapture, shape: AttnShape, head: int) -> torch.Tensor:
    """One head's attention weights on the captured text, [T, T] (row: the reading word-piece; column: the one read; 0
    above the diagonal and outside the sliding window), from its normed, rotated queries and keys at the module's own
    scaling (1 on Gemma 4) and causal mask, keeping keys j with i - j < window on a sliding layer (transformers'
    sliding mask)."""
    l = shape.layer
    q = cap.q[l][int(head)]  # [T, hd]
    k = cap.k[l][shape.kv_of(head)]
    s = q @ k.T
    T = s.shape[0]
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = j <= i
    if shape.window:
        allowed &= (i - j) < shape.window
    s = s.masked_fill(~allowed, float("-inf"))
    return torch.softmax(s, dim=-1)


def gelu_tanh_grad(x: float) -> float:
    """d/dx of gelu_tanh(x) = x/2 (1 + tanh(c (x + 0.044715 x^3))), c = sqrt(2/pi), in closed form (the drive is read
    under no_grad)."""
    c = math.sqrt(2.0 / math.pi)
    t = math.tanh(c * (x + 0.044715 * x**3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * x * x)


def layer_patterns(cap: HeadCapture, shape: AttnShape) -> torch.Tensor:
    """`head_pattern` for every head of the layer: [H, T, T]."""
    return torch.stack([head_pattern(cap, shape, h) for h in range(shape.n_heads)])


# The reference's eight texts for where each head looks (explorer/server.py HEAD_TEXTS): a pronoun story, code, an
# objection, a recipe, Chinese, dialogue, a schedule and science.
HEAD_TEXTS: Tuple[str, ...] = (
    "Alice called Bob because she wanted to borrow his car. He said yes, and she drove it to the lake.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n    return out + a + b",
    "The committee, however, rejected the proposal; members said it was expensive, rushed, and unclear about who would pay.",
    "Ingredients: 2 cups flour, 1 tsp salt, 3 eggs, and a pinch of sugar. Mix the flour and salt, then add the eggs.",
    "我昨天去了北京，但是天气很冷，所以我们在酒店里待了一整天。",
    "\"Where are you going?\" asked the old man. \"To the market,\" said Mara, \"to sell the goat my brother raised.\"",
    "- Monday: 9:00 standup, 14:30 review with Sam\n- Tuesday: dentist at 11\n- Friday: release 2.4.1, then retro",
    "The mitochondria produce ATP through oxidative phosphorylation, which requires oxygen; without it, cells fall back on glycolysis.",
)


@torch.no_grad()
def head_profiles(lm: nn.Module, weights: Weights, tokenizer: Any, texts_ids: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Where every head looks over the texts, by display unit, for reads from the fourth unit on (the reference's
    `_head_profiles`): `dist`, the mean share of weight on the unit itself, the one before, two to five back, further
    back, and the first unit (the start mark, the attention sink); `src`, by unit text met at least twice, the weight it
    receives against what a uniform read would give it (the ten it draws most beyond chance, as [text, ratio]); `pairs`,
    the strongest single reads more than one unit back and past the first unit, [weight, text index, reading unit, unit
    read]. Gemma 4's attention is a softmax, so no weight is negative (the reference's share of negative weight, for its
    linear-attention heads, has nothing to measure here). Keys "layer/head"."""
    shapes = [attn_shape(lm, l) for l in range(len(lm.layers))]
    acc: Dict[str, Dict[str, Any]] = {}
    tokens: List[List[str]] = []
    for t, ids in enumerate(texts_ids):
        cap = capture_heads(lm, ids, weights)
        units = token_units(tokenizer, ids)
        toks = [s for s, _ in units]
        tokens.append(toks)
        U = len(units)
        if U < 5:
            continue
        last = [idx[-1] for _, idx in units]
        qi = np.arange(3, U)
        j = np.arange(U)
        back = qi[:, None] - j[None]
        for l, sh in enumerate(shapes):
            A = layer_patterns(cap, sh).numpy()  # [H, T, T]
            A = np.stack([A[:, :, idx].sum(-1) for _, idx in units], -1)[:, last]  # [H, U, U]
            tot = np.abs(A).sum(-1, keepdims=True).clip(1e-12)
            R = (A / tot)[:, 3:]  # [H, U - 3, U]
            dist = np.stack([(R * (back == 0)).sum(-1), (R * (back == 1)).sum(-1),
                             (R * ((back >= 2) & (back <= 5) & (j[None] > 0))).sum(-1),
                             (R * ((back > 5) & (j[None] > 0))).sum(-1), R[..., 0]], -1).sum(1)  # [H, 5]
            mask = (back >= 1) & (j[None] > 0)
            recv = (R * mask).sum(1)  # [H, U]
            expv = (mask / (qi[:, None] + 1)).sum(0)  # [U]
            for h in range(sh.n_heads):
                a = acc.setdefault(f"{l}/{h}", {"dist": np.zeros(5), "n": 0, "src": {}, "pairs": []})
                a["dist"] += dist[h]
                a["n"] += len(qi)
                for uu in range(1, U - 1):
                    key = toks[uu].strip() or toks[uu]
                    s = a["src"].setdefault(key, [0.0, 0.0, 0])
                    s[0] += float(recv[h, uu])
                    s[1] += float(expv[uu])
                    s[2] += 1
                far = np.where((back > 1) & (j[None] > 0), R[h], 0)
                for r, c in zip(*np.unravel_index(np.argsort(-np.abs(far), None)[:6], far.shape)):
                    a["pairs"].append([round(float(far[r, c]), 3), t, int(qi[r]), int(c)])
    heads: Dict[str, Any] = {}
    for key, a in acc.items():
        src = sorted(((s, r / e, r - e) for s, (r, e, n) in a["src"].items() if n >= 2 and r > e), key=lambda x: -x[2])[:10]
        seen, pairs = set(), []
        for w_, t, qu, ku in sorted(a["pairs"], key=lambda p: -abs(p[0])):
            if (t, qu, ku) in seen:
                continue
            seen.add((t, qu, ku))
            pairs.append([w_, t, qu, ku])
            if len(pairs) == 8:
                break
        heads[key] = {"dist": [round(float(x) / max(a["n"], 1), 3) for x in a["dist"]],
                      "src": [[s, round(float(ratio), 1)] for s, ratio, _ in src], "pairs": pairs}
    return {"tokens": tokens, "heads": heads}


@torch.no_grad()
def attention_drive(cap: HeadCapture, weights: Weights, shapes: Sequence[AttnShape], layer: int, neuron: int, pos: int, top: int = 8) -> Dict[str, Any]:
    """How the stream drives neuron (layer, neuron) at word-piece `pos` of the captured text, part by part.

    The activation a = gelu_tanh(g) u, g = W_gate[n] . x, u = W_up[n] . x, x = w_pre (.) mid / rms(mid), is linearised in
    the stream entering the feed-forward block (mid) with the norm's rms held at its value there (as the reference
    holds it; with the rms free the parts would add up to nothing, the norm being scale-free):
        J = (gelu'(g) u W_gate[n] + gelu(g) W_up[n]) (.) w_pre / rms(mid),    total = J . mid = gelu'(g) u g + gelu(g) u.
    mid is a sum of parts: the embedding, each earlier layer's attention and feed-forward writes, and this layer's
    attention, each carried by the layer scalars it passed (s_k for each layer k from where it was written to l - 1). A
    head's part is its write through the post-attention norm, s-carried, `w_pa (.) W_O[:, h] y_h / rms(o)`; it splits
    exactly over the word-pieces the head read, y_h = sum_j A[h, pos, j] v_j. Returns act, total, every head's
    contribution J . part [(layer, head, c)], each feed-forward block's [(layer, c)], the embedding's, the share they
    leave unexplained (`rest`, float error only), and for the `top` heads by |c| their split over source pieces."""
    import torch.nn.functional as Fn

    l, n, p = int(layer), int(neuron), int(pos)
    s = weights.scalars
    mid = cap.mid[l][p].double()
    w_pre = weights.norm(l, "pre_feedforward").cpu().double()
    eps = weights.eps(l, "pre_feedforward")
    rms = float((mid.pow(2).mean() + eps).sqrt())
    xh = w_pre * mid / rms
    wg = weights.gate_rows(l, [n])[0].cpu().double()
    wu = weights.up_rows(l, [n])[0].cpu().double()
    g, u = float(wg @ xh), float(wu @ xh)
    gelu_g = float(Fn.gelu(torch.tensor(g, dtype=torch.float64), approximate="tanh"))
    dgelu = gelu_tanh_grad(g)
    J = (dgelu * u * wg + gelu_g * wu) * w_pre / rms  # [D]
    total = float(J @ mid)

    def carried(src: int) -> float:
        """The product of the layer scalars a part written at layer `src` passes before it reaches mid_l: s_src (its
        own layer's output scalar) and every s_k after it up to l - 1; 1 for this layer's own attention, which joins mid
        before the layer's scalar."""
        return float(torch.prod(s[src:l].double())) if src < l else 1.0

    heads: List[Tuple[int, int, float]] = []
    t_by_layer: Dict[int, torch.Tensor] = {}
    for ll in range(l + 1):
        sh = shapes[ll]
        W_O = weights.attn_weight(ll, "o").cpu().double()  # [D, H*hd]
        w_pa = weights.norm(ll, "post_attention").cpu().double()
        c_ll = carried(ll) / float(cap.o_rms[ll][p])
        t = (W_O.T @ (J * w_pa)).view(sh.n_heads, sh.head_dim) * c_ll  # [H, hd]
        t_by_layer[ll] = t
        y = cap.y[ll][:, p].double()  # [H, hd]
        heads += [(ll, h, float(v)) for h, v in enumerate((t * y).sum(-1).tolist())]
    mlps: List[Tuple[int, float]] = []
    for ll in range(l):
        w_post = weights.norm(ll, "post_feedforward").cpu().double()
        ym = cap.y_mlp[ll][p].double()
        r = float((ym.pow(2).mean() + weights.eps(ll, "post_feedforward")).sqrt())
        mlps.append((ll, float(carried(ll) * (J @ (w_post * ym / r)))))
    embed = float(carried(0) * (J @ cap.embed[p].double())) if cap.embed is not None else 0.0
    explained = sum(c for _, _, c in heads) + sum(c for _, c in mlps) + embed
    src: Dict[Tuple[int, int], List[float]] = {}
    for ll, h, _ in sorted(heads, key=lambda r: -abs(r[2]))[: int(top)]:
        sh = shapes[ll]
        A = head_pattern(cap, sh, h)[p, : p + 1].double()  # [p + 1]
        v = cap.v[ll][sh.kv_of(h), : p + 1].double()  # [p + 1, hd]
        src[(ll, h)] = (A * (v @ t_by_layer[ll][h])).tolist()
    return {
        "act": gelu_g * u,
        "gate": g,
        "up": u,
        "total": total,
        "heads": heads,
        "mlps": mlps,
        "embed": embed,
        "rest": total - explained,
        "src": src,
    }


# ------------------------------------------------------------------------------------------------- wiring by weights


@torch.no_grad()
def neuron_wiring(weights: Weights, layer: int, neuron: int, k: int = 24) -> Dict[str, List[Dict[str, Any]]]:
    """Direct-path links between one neuron and the neurons of every other layer, from the weights alone, as the
    reference's `connections` reads them, with Gemma 4's norms in place: a writer A at layer i writes
    v_A = w_post_i (.) W_down_i[:, A] (its direction through the post-feed-forward norm); a reader B at layer j reads
    it through the pre-feed-forward norm, g = W_gate_j[B] . (w_pre_j (.) v_A) and u = W_up_j[B] . (w_pre_j (.) v_A).
    P = g u (both of the reader's projections must read the writer for it to move the reader); |P| z-scored within the
    layer pair (so the norms' rms, the layer scalars between them and any single scale shared by the pair drop out);
    sign = sign(P), the sign of the reader's activation the writer's direction gives it, either way the writer fires.
    `down`: readers in later layers; `up`: writers in earlier layers. Each the k strongest by z, {layer, n, z, sign}."""
    l, n = int(layer), int(neuron)
    dev = weights.device
    v = (weights.down_cols(l, [n])[0] * weights.norm(l, "post_feedforward").to(dev)).float()  # [D]
    w_pre_l = weights.norm(l, "pre_feedforward").to(dev)
    rg = (weights.gate_rows(l, [n])[0] * w_pre_l).float()
    ru = (weights.up_rows(l, [n])[0] * w_pre_l).float()
    out: Dict[str, List[Dict[str, Any]]] = {"down": [], "up": []}
    for j in range(weights.n_layers):
        if j == l:
            continue
        # the norm weights are folded into the one vector, so each matrix is read once, in its own dtype
        if j > l:
            Wg, Wu = weights.w_gate(j), weights.w_up(j)  # [N, D]
            x = (v * weights.norm(j, "pre_feedforward").to(dev)).to(Wg.dtype)
            g, u = (Wg @ x).float(), (Wu @ x.to(Wu.dtype)).float()
            side = "down"
        else:
            Wd = weights.w_down(j)  # [D, N]
            w_post_j = weights.norm(j, "post_feedforward").to(dev)
            g = ((rg * w_post_j).to(Wd.dtype) @ Wd).float()
            u = ((ru * w_post_j).to(Wd.dtype) @ Wd).float()
            side = "up"
        prod = g * u
        mag = prod.abs()
        z = (mag - mag.mean()) / (mag.std() + 1e-12)
        top = torch.topk(z, k=min(int(k), z.numel()))
        out[side] += [
            {"layer": j, "n": int(i), "z": round(float(zz), 2), "sign": 1 if float(prod[int(i)]) > 0 else -1}
            for zz, i in zip(top.values.tolist(), top.indices.tolist())
        ]
    for side in out:
        out[side] = sorted(out[side], key=lambda c: -c["z"])[: int(k)]
    return out


class HeadSpaces:
    """Each attention head's read spaces (its query, key and value rows, after the input norm's weight) and write space
    (its output columns, after the post-attention norm's weight), read through Gram matrices, so a vector's share of a
    space is ||P d|| = sqrt((W d)^T (W W^T)^-1 (W d)) without a basis per head. Nothing is kept between calls: a basis of
    every head's spaces on Gemma-4-31B would take gigabytes, and one layer's Gram matrices take milliseconds to remake
    from its weights (float32 products, float64 factors)."""

    def __init__(self, weights: Weights, shapes: Sequence[AttnShape]) -> None:
        self.w = weights
        self.shapes = shapes

    @staticmethod
    def _chol(G: torch.Tensor) -> torch.Tensor:
        G = G.double().cpu()
        G += torch.eye(G.shape[-1], dtype=torch.float64) * (1e-9 * G.diagonal(dim1=-2, dim2=-1).mean())
        return torch.linalg.cholesky(G)

    def read_rows(self, layer: int, role: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """(rows [units, hd, D] float32 on the device, Cholesky factors [units, hd, hd] float64 on the CPU) of the q, k or
        v read space: units are the query heads for q, the key-value heads for k and v."""
        sh = self.shapes[layer]
        W = self.w.attn_weight(layer, role).float() * self.w.norm(layer, "input").to(self.w.device)[None, :]
        units = sh.n_heads if role == "q" else sh.n_kv
        R = W.view(units, sh.head_dim, -1)
        return R, self._chol(R @ R.transpose(1, 2))

    def write_cols(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(M [H, D, hd] float32 on the device, Cholesky factors of M^T M [H, hd, hd] float64 on the CPU)."""
        sh = self.shapes[layer]
        W_O = self.w.attn_weight(layer, "o").float()  # [D, H*hd]
        M = (W_O * self.w.norm(layer, "post_attention").to(self.w.device)[:, None]).view(-1, sh.n_heads, sh.head_dim).permute(1, 0, 2)
        return M, self._chol(M.transpose(1, 2) @ M)

    @staticmethod
    def _share(proj: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
        """proj [U, hd, K] (W d for K vectors) -> ||P d|| [U, K] (float64, CPU)."""
        c = proj.double().cpu()
        sol = torch.cholesky_solve(c, chol)
        return (c * sol).sum(1).clamp(min=0).sqrt()

    def read_share(self, layer: int, role: str, d: torch.Tensor) -> torch.Tensor:
        """The share of each vector in d [D, K] that falls in each unit's read space: [U, K]."""
        R, chol = self.read_rows(layer, role)
        proj = torch.einsum("uhd,dk->uhk", R, d.float().to(R.device))
        return self._share(proj, chol) / d.double().norm(dim=0).clamp(min=1e-30).cpu()[None, :]

    def write_share(self, layer: int, r: torch.Tensor) -> torch.Tensor:
        """The share of each row vector in r [K, D] that falls in each head's write space: [H, K]."""
        M, chol = self.write_cols(layer)
        proj = torch.einsum("hdj,kd->hjk", M, r.float().to(M.device))
        return self._share(proj, chol) / r.double().norm(dim=1).clamp(min=1e-30).cpu()[None, :]


def _z(v: torch.Tensor) -> torch.Tensor:
    v = v.double()
    return (v - v.mean()) / (v.std() + 1e-12) if v.numel() > 1 else torch.zeros_like(v)


@torch.no_grad()
def neuron_head_roles(spaces: HeadSpaces, layer: int, neuron: int, k: int = 8) -> Dict[str, Any]:
    """The attention heads a neuron meets, from the weights, by role (the reference's `neuron_head_roles`): q, k and v
    are later heads whose query, key or value read space holds a large share of the neuron's write (w_post (.)
    W_down[:, n]): it shapes what they look for from its word-piece, makes its word-piece findable, or is what they
    carry on; o are heads at its layer or before whose write space holds a large share of its gate row times the same
    for its up row (after the pre-feed-forward norm's weight): heads whose output it reads. Each share is taken against a
    random direction's (sqrt(hd / D)), z-scored over every candidate head of the role; heads sharing one key or value
    space score alike and are listed once with all their numbers. Returns {role: [{layer, head, heads, z}]} (top k) and
    `z_by_layer` {layer: [z per head]} (v for later layers, o for earlier ones, as the map lights them)."""
    w, shapes = spaces.w, spaces.shapes
    l, n = int(layer), int(neuron)
    dev = w.device
    d = (w.down_cols(l, [n])[0] * w.norm(l, "post_feedforward").to(dev))[:, None]  # [D, 1]
    w_pre = w.norm(l, "pre_feedforward").to(dev)
    rows = torch.stack([w.gate_rows(l, [n])[0] * w_pre, w.up_rows(l, [n])[0] * w_pre])  # [2, D]
    D = w.d_model
    roles: Dict[str, List[Tuple[float, int, List[int]]]] = {"q": [], "k": [], "v": [], "o": []}
    for ll in range(w.n_layers):
        sh = shapes[ll]
        base = math.sqrt(sh.head_dim / D)
        if ll > l:
            for role in ("q", "k", "v"):
                share = spaces.read_share(ll, role, d)[:, 0] / base  # [U]
                for u_idx, val in enumerate(share.tolist()):
                    hs = [u_idx] if role == "q" else list(range(u_idx * sh.group, (u_idx + 1) * sh.group))
                    roles[role].append((val, ll, hs))
        else:
            share = spaces.write_share(ll, rows)  # [H, 2]
            val = (share[:, 0] / base) * (share[:, 1] / base)
            for h, x in enumerate(val.tolist()):
                roles["o"].append((x, ll, [h]))
    out: Dict[str, Any] = {}
    z_by_layer: Dict[int, List[Optional[float]]] = {}
    for role, cand in roles.items():
        if len(cand) < 2:
            out[role] = []
            continue
        z = _z(torch.tensor([c[0] for c in cand]))
        ranked = sorted(zip(z.tolist(), cand), key=lambda r: -r[0])
        out[role] = [{"layer": ll, "head": hs[0], "heads": hs, "z": round(zz, 2)} for zz, (_, ll, hs) in ranked[: int(k)]]
        if role in ("v", "o"):
            for zz, (_, ll, hs) in zip(z.tolist(), cand):
                row = z_by_layer.setdefault(ll, [None] * shapes[ll].n_heads)
                for h in hs:
                    row[h] = round(zz, 2)
    out["z_by_layer"] = {str(ll): row for ll, row in sorted(z_by_layer.items())}
    return out


@torch.no_grad()
def head_wiring(spaces: HeadSpaces, layer: int, head: int, k: int = 40) -> Dict[str, List[Dict[str, Any]]]:
    """What one head reads and writes, from the weights (the reference's `head_links`): q, k, v are the earlier
    neurons whose write lands in its query, key or value read space, as the share of the write the space picks up; o are
    the neurons at its layer or later whose gate and up rows both read its write space, the two shares multiplied. Shares
    make the long writers that line up with every head sink. Each list is the top k by that share, z-scored over every
    candidate neuron of the role: [{layer, n, z}]."""
    w, shapes = spaces.w, spaces.shapes
    l, h = int(layer), int(head)
    sh = shapes[l]
    dev = w.device
    unit = {"q": h, "k": sh.kv_of(h), "v": sh.kv_of(h)}
    scores: Dict[str, List[torch.Tensor]] = {"q": [], "k": [], "v": [], "o": []}
    reads = {}
    for role in ("q", "k", "v"):
        R, chol = spaces.read_rows(l, role)
        reads[role] = (R[unit[role]].clone(), chol[unit[role] : unit[role] + 1])
        del R
    for i in range(l):
        D_i = w.w_down(i).float() * w.norm(i, "post_feedforward").to(dev)[:, None]  # [D, N]
        norms = D_i.double().norm(dim=0).clamp(min=1e-30).cpu()
        for role in ("q", "k", "v"):
            R_u, chol_u = reads[role]
            share = HeadSpaces._share((R_u @ D_i)[None], chol_u)[0] / norms
            scores[role].append(share)
        del D_i
    M, chol_o = spaces.write_cols(l)
    for j in range(l, w.n_layers):
        w_pre = w.norm(j, "pre_feedforward").to(dev)
        G = w.w_gate(j).float() * w_pre[None, :]
        U = w.w_up(j).float() * w_pre[None, :]
        sg = HeadSpaces._share((M[h].T @ G.T)[None], chol_o[h : h + 1])[0] / G.double().norm(dim=1).clamp(min=1e-30).cpu()
        su = HeadSpaces._share((M[h].T @ U.T)[None], chol_o[h : h + 1])[0] / U.double().norm(dim=1).clamp(min=1e-30).cpu()
        scores["o"].append(sg * su)
    out: Dict[str, List[Dict[str, Any]]] = {}
    N = w.d_ff
    for role, parts in scores.items():
        if not parts:
            out[role] = []
            continue
        flat = torch.cat(parts)
        z = _z(flat)
        top = torch.topk(z, k=min(int(k), z.numel()))
        l0 = 0 if role != "o" else l
        out[role] = [{"layer": l0 + int(i) // N, "n": int(i) % N, "z": round(float(zz), 2)} for zz, i in zip(top.values.tolist(), top.indices.tolist())]
    return out


# ---------------------------------------------------------------------------------------------------- the steer lens

# The reference's four contexts (sweep/steer_lens.py): two chat turns and two plain openings.
STEER_LENS_CHATS = ("Who are you?", "Write whatever you like.")
STEER_LENS_PLAIN = ("I think the most important thing to remember is", "Today we are going to talk about")
STEER_LENS_DOSE = 0.8


@torch.no_grad()
def steer_lens(
    lm: nn.Module,
    W_U: torch.Tensor,
    contexts: Sequence[Sequence[int]],
    weights: Weights,
    layer: int,
    neuron: int,
    p50: float,
    *,
    dose: float = STEER_LENS_DOSE,
    k: int = 12,
    bos_id: Optional[int] = None,
    spare_bos: bool = True,
) -> Dict[str, Any]:
    """What pushing the neuron does to the next piece (the reference's steer lens): ±dose of the stream's median length
    along its write direction ("add") at every piece of each context but the start mark, one forward with an unpushed
    row per context; the mean change of log-probability over the contexts, fp32 softcapped, its k biggest rises for each
    sign, and KL(pushed || base) averaged over the contexts. Rows are slot-major, right-padded, each read at its own
    last piece."""
    from backend.causal_check import pad_right

    dev = model_device(lm)
    C = len(contexts)
    inp, att, last, _ = pad_right(contexts, dev)
    inp3, att3, last3 = inp.repeat(3, 1), att.repeat(3, 1), last.repeat(3)
    u = weights.write_dirs(int(layer), [int(neuron)])[0]
    sgn = torch.tensor([0.0] * C + [1.0] * C + [-1.0] * C, device=dev)
    iv = Interventions(rows=3 * C, adds={int(layer): (sgn * float(dose) * float(p50))[:, None] * u[None, :]})
    keep = _Keep()
    keep.mask = keep_mask(inp3, att3, bos_id, spare_bos)
    with installed(lm, iv, keep):
        out = lm(input_ids=inp3, attention_mask=att3, use_cache=False)
    h = out.last_hidden_state[torch.arange(3 * C, device=dev), last3]
    lsm = F.log_softmax(logits_fp32(h, W_U), dim=-1).view(3, C, -1)
    base = lsm[0]
    res: Dict[str, Any] = {"dose": float(dose), "contexts": C}
    for slot, key in ((1, "up"), (2, "down")):
        diff = (lsm[slot] - base).mean(0)
        kl = (lsm[slot].exp() * (lsm[slot] - base)).sum(-1).mean()
        top = torch.topk(diff, k=int(k))
        res[key] = [int(t) for t in top.indices.tolist()]
        res[f"{key}_dlogprob"] = [round(float(x), 4) for x in top.values.tolist()]
        res[f"kl_{key}"] = round(float(kl), 4)
    return res
