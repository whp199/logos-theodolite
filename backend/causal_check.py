"""The causal spot-check (`backend/causal_check.py`): turn one neuron up at one word-piece and read the model's own odds.

A neuron's label is a guess read off its wiring through the lens. The study tested such guesses causally (N1A,
experiments/scripts/N1A_02_causal_core.py:3-8, 79-120): a forward pre-hook on layers[l].mlp.down_proj adds
delta = s * k * max|a_i| to mlp_act[l, p, i] at one position p (max|a_i| over the 36 texts below, every word-piece after
the start mark), the untouched copy of the same reading runs as another row of the SAME forward (rows of one forward are
bit-identical, while different batch shapes were not: KL up to 0.014), and c = log_softmax(pushed) - log_softmax(base)
over the whole vocabulary, from fp32 softcapped logits (N1A_common.py:88-97: bf16 logits would swamp small effects). A
label passed when its first ten words sat among the hundred that c raised most (P@10, N1_REPORT.md:84-86).

The study's second look found 60x pushes saturate the post-feed-forward norm at L20-50 and advised k ~ 10 or a
frozen-norm write (erratum E47, REPORT.md:910-916; N1V_robustness.md:49-55, 221-225), so the default here is k = 10. The
frozen-norm write (N1V, experiments/scripts/N1V_02_causal.py:7-9, 73-83) adds delta * (w_post (.) W_down[:, i]) / rms(y_p)
to the norm's output instead, where y_p is the layer's feed-forward output at p: the same first-order change, which
cannot squash the norm (but a large one still disturbs layers 20-30, N1V_robustness.md:57-60). Each check reports how
straight its push stayed, in closed form from the base y (N1V_audit_01_postnorm.py:90-104).

Two ways to read (`CAUSAL_CONTEXTS`):
- "given": one text, given as exact ids, pushed at its last word-piece. Nothing before that word-piece can change (N1V,
  V2c), so the text before it runs once, batch 1, and only the last word-piece is run for every row, in one forward,
  over that shared prefix (`run_check`): base, +k, -k and, by default, the bf16 floor +-0.01 (N1A_02:147-150).
- "study": the study's 36 texts, right-padded in one forward with every row (slot-major, as N1A_02:91), each pushed at
  its own last word-piece; c is the mean over the texts, with the split-half agreement of even and odd texts.
If the card runs out of memory, the pushed rows are split over more forwards, halving down to the base and one pushed
row, each forward with its own base row (N1V_02:87-96); `forwards` and `base_rows_identical` say what happened.

Pure torch, importable on the CPU: scripts/check_causal_check.py runs every function here on a tiny random Gemma 4; the
scientist tool `causal_spot_check` (backend/mcp_scientist.py) runs them on the real model. Layers count from 0.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.theodolite_math import FINAL_LOGIT_SOFTCAP

CAUSAL_CHECK_VERSION = 1
CAUSAL_WRITES = ("activation", "frozen_norm")
CAUSAL_CONTEXTS = ("given", "study")
SCALE_REFS = ("study", "this_text")
K_DEFAULT = 10.0  # E47: the linear regime (N1V_audit_03 re-ran the N1A protocol at k = 10)
K_MAX = 200.0  # room for N1A's kB = 180
K_FLOOR = 0.01  # N1A's bf16 floor (N1A_02_causal_core.py:147-150)
TOP_N_MAX = 50
TRACK_MAX = 16
LABEL_K = 10  # P@10: the label's first ten ...
CAUSAL_REF_K = 100  # ... among the hundred word-pieces c raised most (N1_REPORT.md:84-86)
FROZEN_NORM_NO_LINEARITY = "a frozen-norm write cannot squash the norm; see write_over_ffn and rho(c+, c-)"

# The study's 36 texts, copied verbatim from experiments/scripts/N1A_common.py:19-60 (_CHAT, _PROSE, _CODE, in that
# order): 12 chats, each a user turn and the begun answer (lab.chat(user, assistant=prefix): the chat template, thinking
# off, then the prefix), 12 passages of prose and 12 of code, read as bare text after the start mark.
# scripts/check_causal_check.py compares them with that file.
STUDY_TEXTS: Tuple[Tuple[str, ...], ...] = (
    ("chat", "What's a good way to learn to cook?", "Start with a few simple dishes you enjoy eating, like"),
    ("chat", "Can you explain how vaccines work?", "Vaccines train your immune system by exposing it to"),
    ("chat", "Write a short poem about the sea.", "The waves roll in beneath a"),
    ("chat", "How do I fix a flat bicycle tire?", "First, remove the wheel from the frame and"),
    ("chat", "What is the capital of Australia?", "The capital of Australia is"),
    ("chat", "I'm feeling stressed about my exams.", "That's completely understandable. One thing that often helps is"),
    ("chat", "Summarize the plot of Romeo and Juliet.", "Romeo and Juliet is a tragedy about two young"),
    ("chat", "What are the benefits of exercise?", "Regular exercise improves your"),
    ("chat", "Translate 'good morning' into French.", "In French, 'good morning' is"),
    ("chat", "Give me a tip for public speaking.", "One of the most effective tips is to"),
    ("chat", "Why is the sky blue?", "The sky appears blue because of a phenomenon called"),
    ("chat", "Recommend a book for a long flight.", "If you enjoy thrillers, I'd recommend"),
    ("prose", "The old lighthouse keeper climbed the spiral stairs every evening, carrying a lantern and a"),
    ("prose", "In 1969, astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the"),
    ("prose", "The recipe calls for two cups of flour, a pinch of salt, and"),
    ("prose", "After the storm passed, the villagers gathered in the square to"),
    ("prose", "Photosynthesis is the process by which plants convert sunlight into"),
    ("prose", "She opened the letter slowly, afraid of what it might"),
    ("prose", "The stock market fell sharply on Monday after the central bank announced"),
    ("prose", "Mount Everest, the tallest mountain in the world, is located in the"),
    ("prose", "He had never seen so many stars; the night sky over the desert was"),
    ("prose", "The committee will meet next Thursday to discuss the proposed"),
    ("prose", "Dolphins are highly intelligent marine mammals known for their"),
    ("prose", "The train was delayed by nearly an hour, so we decided to"),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
    ("code", "import numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np."),
    ("code", "for (let i = 0; i < arr.length; i++) {\n    console."),
    ("code", "SELECT name, email FROM users WHERE age > 30 ORDER BY"),
    ("code", "<div class=\"container\">\n  <h1>Welcome</h1>\n  <p>"),
    ("code", "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items."),
    ("code", "#include <stdio.h>\n\nint main() {\n    printf(\""),
    ("code", "with open('data.txt', 'r') as f:\n    lines = f."),
    ("code", "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n    res."),
    ("code", "git commit -m \"Fix bug in"),
    ("code", "def is_prime(n):\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return"),
    ("code", "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(df."),
)
STUDY_TEXTS_SOURCE = "experiments/scripts/N1A_common.py:19-60"


def language_model(hf: nn.Module) -> nn.Module:
    """The text model under a Gemma 4 checkpoint (its decoder layers, final norm, no lm_head), as N1A_common.n1a_LM."""
    m = hf.model
    return m.language_model if hasattr(m, "language_model") else m


def model_device(lm: nn.Module) -> torch.device:
    return lm.embed_tokens.weight.device


def pad_right(
    ids_list: Sequence[Sequence[int]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded ids [B, T] (pad 0), their attention mask, each row's last position, and `valid`: the mask with
    position 0, the start mark, cleared (N1A_common.py:71-80, N1A_02:28)."""
    B, T = len(ids_list), max(len(x) for x in ids_list)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(ids_list):
        inp[b, : len(x)] = torch.tensor([int(t) for t in x], dtype=torch.long)
        att[b, : len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in ids_list], dtype=torch.long)
    valid = att.clone().bool()
    valid[:, 0] = False
    return inp.to(device), att.to(device), last.to(device), valid.to(device)


@torch.no_grad()
def logits_fp32(h: torch.Tensor, W_U: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Softcapped logits with fp32 output from final-normed states h [B, D] (N1A_common.py:88-97)."""
    hf = h.float()
    out = torch.empty(h.shape[0], W_U.shape[0], device=h.device, dtype=torch.float32)
    for s in range(0, W_U.shape[0], chunk):
        out[:, s : s + chunk] = hf @ W_U[s : s + chunk].to(h.device).float().T
    return torch.tanh(out / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


@torch.no_grad()
def study_max_abs(lm: nn.Module, inp: torch.Tensor, att: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """max |mlp_act[l, ., i]| over the `valid` positions (no start mark, no padding), for every layer and neuron, in one
    forward with a pre-hook on every down_proj (N1A_02:31-41): [n_layers, d_ff] float32, on the CPU."""
    acts: Dict[int, torch.Tensor] = {}
    hooks = []
    try:
        for l, layer in enumerate(lm.layers):

            def pre(_mod: nn.Module, args: Tuple[Any, ...], l: int = l) -> None:
                a = args[0].abs() * valid.unsqueeze(-1).to(args[0].dtype)
                acts[l] = a.reshape(-1, a.shape[-1]).max(0).values.float().cpu()

            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        lm(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([acts[l] for l in range(len(lm.layers))])


@torch.no_grad()
def text_max_abs(lm: nn.Module, layer: int, neuron: int, ids: Sequence[int]) -> float:
    """max |mlp_act[layer, p, neuron]| over one text's word-pieces after the start mark (p >= 1), in one forward."""
    seen: Dict[str, float] = {}

    def pre(_mod: nn.Module, args: Tuple[Any, ...]) -> None:
        seen["max"] = float(args[0][0, 1:, int(neuron)].abs().max().float())

    h = lm.layers[int(layer)].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        inp = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=model_device(lm))
        lm(input_ids=inp, use_cache=False)
    finally:
        h.remove()
    return seen["max"]


@contextmanager
def _frozen_prefix(cache: Any) -> Iterator[None]:
    """Let one forward read a batch-1 prefix cache from every row without growing it: each cache layer's update returns
    its stored keys and values broadcast to the rows, with the new ones after them (what DynamicLayer and
    DynamicSlidingWindowLayer return: the stored window, then the new piece), and stores nothing. So the prefix is
    never copied once per row, and the same cache serves a later forward after an out-of-memory retry. The attention
    masks were built from the prefix's own lengths before any layer ran, so they are the ones a growing cache gives."""
    patched = []
    try:
        for layer in cache.layers:

            def update(key_states: torch.Tensor, value_states: torch.Tensor, *_a: Any, _l: Any = layer, **_k: Any):
                r = key_states.shape[0]
                pk = _l.keys.expand(r, *_l.keys.shape[1:])
                pv = _l.values.expand(r, *_l.values.shape[1:])
                return torch.cat([pk, key_states], dim=-2), torch.cat([pv, value_states], dim=-2)

            layer.update = update  # an instance attribute, shadowing the class's update
            patched.append(layer)
        yield
    finally:
        for layer in patched:
            try:
                del layer.update
            except AttributeError:
                pass


@dataclass
class CheckRun:
    """What `run_check` read, per context c: `base` [C, V] (the first forward's base rows, fp32 log-softmax),
    `steered[s]` and `bases[s]` [C, V] for pushed slot s (bases[s] is the base row of the forward that slot ran in),
    the neuron's own activation `act` [C] and the layer's feed-forward output `y` [C, D] (float32) at the pushed
    position in the base rows, how many forwards it took, the slots in each, and whether every forward's base rows
    were bit-identical to the first's."""

    base: torch.Tensor
    steered: List[torch.Tensor]
    bases: List[torch.Tensor]
    act: torch.Tensor
    y: torch.Tensor
    forwards: int
    slots_per_forward: List[int] = field(default_factory=list)
    base_rows_identical: bool = True


@torch.no_grad()
def run_check(
    lm: nn.Module,
    layer: int,
    neuron: int,
    deltas: Sequence[float],
    write: str,
    w_down: torch.Tensor,
    w_post: torch.Tensor,
    eps: float,
    W_U: torch.Tensor,
    *,
    token_ids: Optional[Sequence[int]] = None,
    texts_ids: Optional[Sequence[Sequence[int]]] = None,
    kv_cache: bool = True,
    max_slots: Optional[int] = None,
) -> CheckRun:
    """Push neuron `neuron` of layer `layer` by each of `deltas` at the read position, beside an unpushed base row.

    Exactly one of `token_ids` (given mode: one text, read at its last word-piece) and `texts_ids` (study mode: each
    text read at its own last word-piece, right-padded) is given. `write` is "activation" (mlp_act[p, i] += delta, a
    pre-hook on down_proj) or "frozen_norm" (the post-feed-forward norm's output at p gains delta * w_post (.) w_down /
    rms(y_p), a forward hook on that norm, with rms over its input in fp32 and the norm's own eps); `w_down` is the
    neuron's write column as the model uses it (vault edits included) and `w_post` the norm's weight. Rows are
    slot-major (row = slot * C + context); slot 0 is the base. Given mode with `kv_cache` runs the text before the read
    position once (batch 1) and then only the read position for every row, over that shared prefix; without it, the
    whole text for every row (scripts/check_causal_check.py checks the two agree). `max_slots` caps the pushed slots
    per forward (all of them by default); on running out of memory the cap halves, down to one, and each forward keeps
    its own base row. The write hooks go first among the module's hooks, so any other hook sees the written value;
    every hook is removed whatever happens."""
    if (token_ids is None) == (texts_ids is None):
        raise ValueError("run_check needs exactly one of token_ids (given) and texts_ids (study).")
    if write not in CAUSAL_WRITES:
        raise ValueError(f"Unknown write {write!r}; expected one of {CAUSAL_WRITES}")
    if not deltas:
        raise ValueError("run_check needs at least one push.")
    dev = model_device(lm)
    l, i = int(layer), int(neuron)
    mlp_down = lm.layers[l].mlp.down_proj
    post_norm = lm.layers[l].post_feedforward_layernorm
    u = (w_post.to(dev).float() * w_down.to(dev).float()).reshape(-1)  # [D], the frozen-norm write's direction

    prefix_cache = None
    if token_ids is not None:
        ids = [int(t) for t in token_ids]
        if len(ids) < 2:
            raise ValueError("run_check pushes the last word-piece, after the start mark: give at least two ids.")
        C = 1
        if kv_cache:
            out = lm(input_ids=torch.tensor([ids[:-1]], dtype=torch.long, device=dev), use_cache=True)
            prefix_cache = out.past_key_values
            del out
            inp0 = torch.tensor([[ids[-1]]], dtype=torch.long, device=dev)
            pos0 = torch.zeros(1, dtype=torch.long, device=dev)
        else:
            inp0 = torch.tensor([ids], dtype=torch.long, device=dev)
            pos0 = torch.tensor([len(ids) - 1], dtype=torch.long, device=dev)
        att0 = None
    else:
        assert texts_ids is not None
        inp0, att0, pos0, _valid = pad_right(texts_ids, dev)
        C = inp0.shape[0]

    def one_forward(slots: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        R = 1 + len(slots)
        inp = inp0.repeat(R, 1)
        att = att0.repeat(R, 1) if att0 is not None else None
        pos = pos0.repeat(R)
        base_rows = torch.arange(C, device=dev)
        rows = torch.cat([torch.arange(C, device=dev) + (j + 1) * C for j in range(len(slots))])
        ds = torch.cat([torch.full((C,), float(deltas[s]), device=dev) for s in slots])
        poss = pos[rows]
        seen: Dict[str, torch.Tensor] = {}
        hooks = []

        def read_act(_mod: nn.Module, args: Tuple[Any, ...]) -> None:
            seen["act"] = args[0][base_rows, pos[:C], i].detach().float().clone()

        def read_y(_mod: nn.Module, _args: Tuple[Any, ...], out: torch.Tensor) -> None:
            seen["y"] = out[base_rows, pos[:C]].detach().float().clone()

        def push_act(_mod: nn.Module, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            x = args[0]
            x.index_put_((rows, poss, torch.full_like(rows, i)), ds.to(x.dtype), accumulate=True)
            return (x,) + tuple(args[1:])

        def push_norm(_mod: nn.Module, args: Tuple[Any, ...], out: torch.Tensor) -> torch.Tensor:
            x = args[0][rows, poss].float()
            rms = (x.pow(2).mean(-1, keepdim=True) + float(eps)).sqrt()
            o2 = out.clone()
            o2[rows, poss] = (out[rows, poss].float() + ds[:, None] * u[None] / rms).to(out.dtype)
            return o2

        try:
            if write == "activation":
                hooks.append(mlp_down.register_forward_pre_hook(push_act, prepend=True))
            else:
                hooks.append(post_norm.register_forward_hook(push_norm, prepend=True))
            hooks.append(mlp_down.register_forward_pre_hook(read_act))
            hooks.append(mlp_down.register_forward_hook(read_y))
            if prefix_cache is not None:
                with _frozen_prefix(prefix_cache):
                    out = lm(input_ids=inp, past_key_values=prefix_cache, use_cache=True)
            else:
                out = lm(input_ids=inp, attention_mask=att, use_cache=False)
            hs = out.last_hidden_state
            h = hs[torch.arange(hs.shape[0], device=dev), pos]
            del out, hs
        finally:
            for hk in hooks:
                hk.remove()
        lsm = F.log_softmax(logits_fp32(h, W_U), dim=-1).view(R, C, -1)
        return lsm, seen["act"], seen["y"]

    pending = list(range(len(deltas)))
    per = len(deltas) if max_slots is None else max(1, min(int(max_slots), len(deltas)))
    steered: List[Optional[torch.Tensor]] = [None] * len(deltas)
    bases: List[Optional[torch.Tensor]] = [None] * len(deltas)
    first: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    identical = True
    sizes: List[int] = []
    while pending:
        chunk = pending[:per]
        try:
            lsm, act, y = one_forward(chunk)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if per <= 1:
                raise
            per = max(1, per // 2)
            continue
        if first is None:
            first = (lsm[0], act, y)
        elif not torch.equal(lsm[0], first[0]):
            identical = False
        for j, s in enumerate(chunk):
            steered[s] = lsm[j + 1]
            bases[s] = lsm[0]
        sizes.append(len(chunk))
        pending = pending[len(chunk) :]
    assert first is not None
    return CheckRun(
        base=first[0],
        steered=[t for t in steered if t is not None],
        bases=[t for t in bases if t is not None],
        act=first[1],
        y=first[2],
        forwards=len(sizes),
        slots_per_forward=sizes,
        base_rows_identical=identical,
    )


@torch.no_grad()
def linear_regime(
    y: torch.Tensor, w_down: torch.Tensor, w_post: torch.Tensor, delta: float, eps: float
) -> Dict[str, torch.Tensor]:
    """How straight a push of `delta` along the neuron's column stays through the post-feed-forward norm, in closed
    form from the base feed-forward output y [C, D] at the pushed position, exactly as N1V_audit_01_postnorm.py:90-104:
    - write_over_ffn = ||delta w_down|| / ||y|| (its ratio_d_y);
    - linearity = ||pn(y + delta w_down) - pn(y)|| / ||w_post (.) delta w_down / rms(y)|| (its sat: 1 is a straight
      push, less is squashed), where pn(v) = v / rms(v) * w_post with rms over the norm's eps;
    - ablation_cos = cos(pn(y + delta w_down) - pn(y), -pn(y)) (its cos_abl: how far the change is the rest of the
      layer's output taken away). Each is [C] float32."""
    y = y.float()
    w = w_down.to(y.device).float().reshape(-1)
    wp = w_post.to(y.device).float().reshape(-1)

    def pnorm(v: torch.Tensor) -> torch.Tensor:
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + float(eps)) * wp

    d = float(delta) * w
    pny = pnorm(y)
    change = pnorm(y + d[None]) - pny
    rms_y = y.pow(2).mean(-1, keepdim=True).add(float(eps)).sqrt()
    lin = (wp * d)[None] / rms_y
    return {
        "write_over_ffn": d.norm() / y.norm(dim=-1),
        "linearity": change.norm(dim=-1) / lin.norm(dim=-1),
        "ablation_cos": F.cosine_similarity(change, -pny, dim=-1),
        "frozen_ablation_cos": F.cosine_similarity(lin, -pny, dim=-1),
    }


def kl(lsm_s: torch.Tensor, lsm_b: torch.Tensor) -> torch.Tensor:
    """KL(pushed || base) per row, from log-softmaxes (N1A_02:111)."""
    return (lsm_s.exp() * (lsm_s - lsm_b)).sum(-1)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman's rho of two [V] vectors by argsort ranks (N1A_02:73-76; ties broken by position)."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def top_ids(c: torch.Tensor, k: int) -> List[int]:
    return [int(t) for t in torch.topk(c, k=min(int(k), c.numel())).indices.tolist()]


def p_at(label_ids: Sequence[int], c: torch.Tensor, k_label: int = LABEL_K, k_ref: int = CAUSAL_REF_K) -> float:
    """P@10: the share of the label's first `k_label` ids among the `k_ref` ids c raised most (N1_REPORT.md:84-86,
    whole vocabulary; N1V_audit_04_smallk_metrics.py:19-23 takes C.topk(100), by signed rise)."""
    lab = [int(t) for t in list(label_ids)[:k_label]]
    if not lab:
        return 0.0
    ref = set(top_ids(c, k_ref))
    return sum(1 for t in lab if t in ref) / len(lab)


def movers(c: torch.Tensor, top_n: int) -> Tuple[List[int], List[int]]:
    """The ids whose log-probability c raised most (rises, highest first) and lowered most (falls, lowest first)."""
    return top_ids(c, top_n), top_ids(-c, top_n)


@dataclass
class FaceEffect:
    """One push's effect over the contexts: c [V] (mean change of log-probability), the mean base and pushed
    log-probabilities [V], KL per context [C], and the split-half rho of even and odd contexts (None for one)."""

    c: torch.Tensor
    base_lp: torch.Tensor
    steered_lp: torch.Tensor
    kl_by_context: torch.Tensor
    split_half_rho: Optional[float]


def face_effect(steered: torch.Tensor, base: torch.Tensor) -> FaceEffect:
    """`steered` and `base` [C, V] log-softmaxes of one pushed slot and its own forward's base rows."""
    diff = steered - base
    split = spearman(diff[0::2].mean(0), diff[1::2].mean(0)) if diff.shape[0] >= 2 else None
    return FaceEffect(
        c=diff.mean(0),
        base_lp=base.mean(0),
        steered_lp=steered.mean(0),
        kl_by_context=kl(steered, base),
        split_half_rho=split,
    )


def summary(values: torch.Tensor) -> Tuple[float, float]:
    """(median, tenth percentile) of a [C] tensor."""
    v = values.float().cpu()
    return float(v.median()), float(torch.quantile(v, 0.1))


def finite_or_none(x: Optional[float]) -> Optional[float]:
    return None if x is None or not math.isfinite(float(x)) else float(x)
