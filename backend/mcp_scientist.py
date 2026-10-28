"""FastMCP Autonomous Circuit Scientist (`backend/mcp_scientist.py`).

Exposes four core mechanistic interpretability tools via FastMCP:
1. `inspect_pole(layer, neuron_idx, polarity)`
2. `trace_downstream_readers(layer, neuron_idx, prompt)`
3. `test_motif_hypothesis(layer, neuron_idx, pos_prompts, neg_prompts)`: whether the neuron tells two sets of
   sentences apart, set against chance (`selectivity`, `motif_statistics`: Welch's t, a permutation p, at least four
   sentences a side to judge)
4. `causal_spot_check(layer, neuron_idx, contexts, token_ids | prompt, k, write)`: the label tested causally (the
   study's N1A protocol at k = 10, `backend/causal_check.py`)
Plus `query_larql_vault(lql_query)` for declarative VINDEX3 queries and `.patch` edits (each one neuron's down_proj
column, rank one, with no covariance solve: named after MEMIT in the code, not MEMIT). It takes no voice: the server's
routes can hear an edit before and after (`LarqlRequest.voice`) because they hold the GPU lock around it, and this tool
does not take that lock.

The first three scientist tools decode words through the J++ lens, and each reply names it under "lens"
(`TheodoliteRuntime.lens_card`: what it was fitted on, what it reads, its source, and a `caution` on how far to trust
its words), so an agent calling them over MCP gets the same record as the HTTP and WebSocket replies, which pass these
replies on unchanged. `query_larql_vault` does the same for the vault statements that read words or write an edit
through that lens (NEAREST, DESCRIBE POLE, UPDATE, INSERT: `LarqlVaultEngine.execute_lql`, which also answers
POST /api/larql and the WebSocket "larql" action).

Every neuron label they send is also marked by the study's measure, under "label_check" (`backend.label_gate`: how
many word-pieces the lens and the naked eye share in their first ten, fp32, and whether the instrument's pages hold the
label back from their main reading, with the reasons): each face of `inspect_pole`, each reader of
`trace_downstream_readers` (`dst_label_check`, beside its naked-eye words `dst_naive_tokens`), the pole of
`test_motif_hypothesis` (`pole_label_check`), and each NEAREST row and DESCRIBE POLE face of the vault. The rule they
follow travels with them under "label_gate" (`LABEL_GATE_CARD`), just before "lens". The words themselves are sent as
before, and every old key is kept.

`causal_spot_check` reads its effect from the model's own odds, with no lens; only the label it compares the effect
with goes through the J++ lens (the same first ten ids each face's `label_check` counts), so its reply names the lens
too, last, with `label_gate` before it. It answers POST /api/scientist/causal_check and the WebSocket "causal_check"
action, which pass its reply on unchanged.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastmcp import FastMCP
from interp_engine import Address

from backend import causal_check as cc
from backend.engine_runtime import MAX_SWEEP_POSITIONS, TheodoliteRuntime
from backend.label_gate import LABEL_GATE_CARD, label_check, label_top_ids, writes_nothing
from backend.larql_vault import LarqlVaultEngine
from backend.runtime_factory import make_runtime
from backend.theodolite_math import (
    D_FF,
    N_LAYERS,
    THREAD_SCORES,
    VOCAB_SIZE,
    direct_path_reader_effects,
    direct_path_source_write,
    extract_bipolar_poles,
    unembed_gemma4,
)

mcp = FastMCP("Logos Theodolite Circuit Scientist")

_SHARED_RUNTIME: Optional[TheodoliteRuntime] = None
_SHARED_VAULT: Optional[LarqlVaultEngine] = None


def bind_runtime(runtime: TheodoliteRuntime, vault: Optional[LarqlVaultEngine] = None) -> None:
    """Bind an already-loaded `TheodoliteRuntime` and `LarqlVaultEngine` (e.g. from `server.py`)."""
    global _SHARED_RUNTIME, _SHARED_VAULT
    _SHARED_RUNTIME = runtime
    _SHARED_VAULT = vault if vault is not None else LarqlVaultEngine(runtime)


def get_runtime_and_vault() -> tuple[TheodoliteRuntime, LarqlVaultEngine]:
    """Lazily obtain or initialize the singleton runtime and `LarqlVaultEngine`. Run alone (no server bound one), the
    runtime is `runtime_factory.make_runtime()`'s, as the server's: the W4A16 checkpoint on cuda:0 unless
    THEODOLITE_GGUF or THEODOLITE_DEVICE say otherwise (ARCHITECTURE §5)."""
    global _SHARED_RUNTIME, _SHARED_VAULT
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = make_runtime(load_jlens=False).load()
    if _SHARED_VAULT is None:
        _SHARED_VAULT = LarqlVaultEngine(_SHARED_RUNTIME)
    return _SHARED_RUNTIME, _SHARED_VAULT


@mcp.tool()
def inspect_pole(
    layer: int,
    neuron_idx: int,
    polarity: float = 1.0,
) -> Dict[str, Any]:
    """Inspect a Bipolar GeGLU neuron pole `(layer, neuron_idx, polarity)` in Gemma-4-31B.

    Decodes both its North Pole (`+1`) and South Pole (`-1`) write vectors through the
    J-VINDEX (`(polarity * W_down[l, i]) @ J_pp[l].T`) and contrasts them against the naive
    Logit Lens (`W_U`), plus decodes the upstream `W_gate[l, i]` trigger tokens.

    `lens` names the lens the J-VINDEX words went through; read its `caution` before trusting them. Each face's
    `label_check` counts the word-pieces its two readings share in their first ten; its `held_back` marks a label the
    instrument's pages leave out of their main reading, for the reasons given (`label_gate`, the rule, comes just before
    `lens`, so the keys come in the order of the WebSocket's DESCRIBE POLE reply).
    """
    rt, vault = get_runtime_and_vault()
    reply = vault.inspect_bipolar_slot(
        layer_idx=int(layer),
        neuron_idx=int(neuron_idx),
        polarity=float(polarity),
        top_k_tokens=8,
    )
    return {**reply, "label_gate": LABEL_GATE_CARD, "lens": rt.lens_card()}


@mcp.tool()
async def trace_downstream_readers(
    layer: int,
    neuron_idx: int,
    prompt: str,
    polarity: Optional[float] = None,
    max_layer_hop: int = 6,
    top_k: int = 10,
    score: str = "coupling",
    chat: bool = False,
    system: Optional[str] = None,
    assistant_prefix: str = "",
    token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Trace verified Live Downstream Reader poles for `(layer, neuron_idx)` on `prompt`.

    Computes the Spillover-Filtered coupling score between `(layer, neuron_idx, sigma_i)` and
    active poles `(l', j, sigma_j)` at downstream layers `l' in layer+1 .. layer+max_layer_hop`:
        C_{(l, i) -> (l', j)} = dot(sigma_i * W_down[l, i], W_gate[l', j]) * abs(mlp_act[l', j])
    and the direct-path linear estimates of what the source adds to each reader's gate pre-activation
    (`direct_gate`) and activation (`direct_act`), from the source's real activation, the layer scalars and
    the exact RMSNorm Jacobians (REPORT §3.4 bug 4: rho 0.87 against ablation, where C reaches 0.41).
    `score="direct"` ranks the readers by |direct_gate|; `"coupling"` (default) by |C|.

    `chat=True` wraps `prompt` in the chat template (optional `system` turn, thinking off) and appends
    `assistant_prefix`; otherwise `prompt` is read as given (an earlier sighting's echoed `prompt` is exact).

    `token_ids` (1 to MAX_SWEEP_POSITIONS word-piece ids) are read exactly as given instead, with no composing and no
    tokenizing, at the last of them: a sighting's own ids up to its read position, so the trace reads the stream that
    sighting read (re-tokenizing a prompt and its answer as text need not give the same ids). `prompt`, `chat`,
    `system` and `assistant_prefix` are then not used; the reply's `prompt` is those ids decoded, for reading only.
    The trace always reads the model without any nudge (`steered: false`), on whatever text it is given.

    `lens` names the lens each reader's `dst_j_vindex_tokens` went through; read its `caution` before trusting them.
    Each reader returned also carries the naked eye's words (`dst_naive_tokens`, four, as many as through the lens)
    and `dst_label_check`, which marks its label by the study's measure (`label_gate`, the rule, comes before `lens`).
    """
    if score not in THREAD_SCORES:
        raise ValueError(f"Unknown score {score!r}; expected one of {THREAD_SCORES}")
    runtime, vault = get_runtime_and_vault()
    assert runtime.eager_model is not None

    l_src = max(0, min(int(layer), N_LAYERS - 2))
    n_src = max(0, min(int(neuron_idx), D_FF - 1))
    l_end = min(l_src + max(1, int(max_layer_hop)), N_LAYERS - 1)

    if token_ids is not None:
        if isinstance(token_ids, (str, bytes)) or not 1 <= len(token_ids) <= MAX_SWEEP_POSITIONS:
            raise ValueError(f"token_ids needs 1 to {MAX_SWEEP_POSITIONS} word-piece ids.")
        for t in token_ids:
            if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < VOCAB_SIZE:
                raise ValueError(
                    f"token_ids holds {t!r}, which is not a word-piece id: ids are whole numbers from 0 to {VOCAB_SIZE - 1}."
                )
        prompt_ids = [int(t) for t in token_ids]
        text = runtime.tokenizer.decode(prompt_ids)
    else:
        text = runtime.compose_prompt(prompt, chat=chat, system=system, assistant_prefix=assistant_prefix)
        prompt_ids = runtime.tokenize_prompt(text)
    target_pos = len(prompt_ids) - 1

    points = [Address("mlp_act", l) for l in range(l_src, l_end + 1)]
    points += [Address("mlp_out", l_src)] + [Address("resid_mid", l) for l in range(l_src + 1, l_end + 1)]
    # the last word-piece only (row 0 is target_pos): the rest of the sequence is never read
    caps = runtime.capture_rows(prompt_ids, read=points, n_read=1)

    src_mlp_act = caps[Address("mlp_act", l_src)][0].to(runtime.device)
    src_signed_act = float(src_mlp_act[n_src].item())
    sigma_i = (
        (1.0 if float(polarity) >= 0 else -1.0)
        if polarity is not None
        else (1.0 if src_signed_act >= 0 else -1.0)
    )

    # Fetch source pole write vector: sigma_i * W_down[l_src, n_src]
    src_idx_t = torch.tensor([n_src], device=runtime.device, dtype=torch.long)
    w_down_raw = runtime.slice_w_down(l_src, src_idx_t)
    w_down_src = w_down_raw[0].float()
    if l_src in runtime._post_ffn_norm_weights:
        w_down_src = w_down_src * runtime._post_ffn_norm_weights[l_src].to(runtime.device).float()
    signed_writer = sigma_i * w_down_src  # [5376]

    # Direct path: the source's real activation a_i (its sign included; `polarity` only steers C)
    layer_scalars = [float(runtime._decoder_layers[l].layer_scalar.detach().float().reshape(-1)[0]) for l in range(N_LAYERS)]
    d_post = direct_path_source_write(
        src_mlp_act[n_src : n_src + 1],
        w_down_raw,
        caps[Address("mlp_out", l_src)][0].to(runtime.device).float(),
        runtime._post_ffn_norm_weights[l_src],
        layer_scalars[l_src],
    )  # [1, 5376]

    candidates: List[Dict[str, Any]] = []
    # where each candidate's pole came from (its layer and its rank there), and each layer's pole vectors (through the
    # lens, and to the naked eye; not their logits, which are let go layer by layer), for the readers returned
    found: List[Any] = []
    layer_vecs: Dict[int, Any] = {}
    for l_dst in range(l_src + 1, l_end + 1):
        dst_mlp_act = caps[Address("mlp_act", l_dst)][0].to(runtime.device)
        poles_dst = extract_bipolar_poles(
            mlp_act_l=dst_mlp_act,
            w_down_slice_fn=runtime.slice_w_down,
            J_l=runtime.jpp_lenses[l_dst],
            layer_idx=l_dst,
            k=25,
            final_norm_weight=runtime.final_norm_weight,
            lm_head_weight=runtime.lm_head_weight,
            post_ffn_norm_weight=runtime._post_ffn_norm_weights.get(l_dst),
            compute_logits=True,
            compute_naive_logits=False,
        )
        layer_vecs[l_dst] = (poles_dst.jpp_pole_vectors, poles_dst.signed_w_down)
        w_gate_dst = runtime.slice_w_gate(l_dst, poles_dst.neuron_indices).float()
        w_up_dst = runtime.slice_w_up(l_dst, poles_dst.neuron_indices).float()
        if l_dst in runtime._pre_ffn_norm_weights:
            w_pre = runtime._pre_ffn_norm_weights[l_dst].to(runtime.device).float()
            w_gate_dst = w_gate_dst * w_pre
            w_up_dst = w_up_dst * w_pre

        dots = w_gate_dst @ signed_writer  # [25]
        couplings = dots * poles_dst.abs_activations.float()
        carry = 1.0
        for k_l in range(l_src + 1, l_dst):
            carry *= layer_scalars[k_l]
        d_gate, d_act = direct_path_reader_effects(
            carry * d_post,
            caps[Address("resid_mid", l_dst)][0].to(runtime.device).float(),
            w_gate_dst,
            w_up_dst,
        )  # [1, 25] each

        for r_idx in range(poles_dst.neuron_indices.numel()):
            c_score = float(couplings[r_idx].item())
            d_val = float(dots[r_idx].item())
            dg = float(d_gate[0, r_idx].item())
            da = float(d_act[0, r_idx].item())
            ranked = dg if score == "direct" else c_score
            dst_n = int(poles_dst.neuron_indices[r_idx].item())
            dst_pol = float(poles_dst.polarities[r_idx].item())
            dst_act = float(poles_dst.signed_activations[r_idx].item())
            top_v, top_i = torch.topk(poles_dst.pole_logits[r_idx], k=4)
            jpp_tokens = [
                {"token": runtime.tokenizer.decode([int(tid)]), "logit": round(float(val), 2)}
                for val, tid in zip(top_v.cpu().tolist(), top_i.cpu().tolist())
            ]
            candidates.append(
                {
                    "dst_layer": l_dst,
                    "dst_neuron_idx": dst_n,
                    "dst_polarity": "+" if dst_pol >= 0 else "-",
                    "dst_signed_activation": round(dst_act, 4),
                    "dot_product": round(d_val, 5),
                    "coupling_score": round(c_score, 5),
                    "abs_coupling": round(abs(c_score), 5),
                    "direct_gate": round(dg, 6),
                    "direct_act": round(da, 6),
                    "score": round(ranked, 6),
                    "abs_score": round(abs(ranked), 6),
                    "dst_j_vindex_tokens": jpp_tokens,
                }
            )
            found.append((candidates[-1], l_dst, r_idx))

    candidates.sort(key=lambda item: item["abs_score"], reverse=True)
    # the readers returned (only those) are also read by the naked eye, and their labels checked (label_gate.py)
    where = {id(c): (l, r) for c, l, r in found}
    lens_id = (runtime.lens_card() or {}).get("id")
    edited = {(p.layer_idx, p.neuron_idx) for p in vault.edits_in_force()}
    fnw, lm = runtime.final_norm_weight, runtime.lm_head_weight
    for cand in candidates[: int(top_k)]:
        l_dst, r_idx = where[id(cand)]
        lens_vec = layer_vecs[l_dst][0][r_idx : r_idx + 1]
        eye_vec = layer_vecs[l_dst][1][r_idx : r_idx + 1]
        eye_logits = unembed_gemma4(eye_vec, fnw, lm)
        nv_top, ni_top = torch.topk(eye_logits[0], k=4)
        cand["dst_naive_tokens"] = [
            {"token": runtime.tokenizer.decode([int(tid)]), "logit": round(float(val), 2)}
            for val, tid in zip(nv_top.cpu().tolist(), ni_top.cpu().tolist())
        ]
        cand["dst_label_check"] = label_check(
            l_dst,
            label_top_ids(lens_vec, fnw, lm, unembed_gemma4(lens_vec, fnw, lm))[0],
            label_top_ids(eye_vec, fnw, lm, eye_logits)[0],
            lens_id,
            edited=(l_dst, cand["dst_neuron_idx"]) in edited,
            silenced=writes_nothing(eye_vec)[0],
        )
    return {
        "prompt": text,
        "chat": bool(chat),  # as asked; with token_ids the template was not applied here (the ids may hold one)
        "token_ids_given": token_ids is not None,
        "read_pos": target_pos,
        # read without any nudge, also when the sighting it reads for was nudged
        "steered": False,
        "score_kind": score,
        "max_layer_hop": l_end - l_src,
        "top_k": int(top_k),
        "source_pole": {
            "layer": l_src,
            "neuron_idx": n_src,
            "polarity": "+" if sigma_i >= 0 else "-",
            "signed_activation_on_prompt": round(src_signed_act, 4),
        },
        "top_downstream_readers": candidates[: int(top_k)],
        "label_gate": LABEL_GATE_CARD,
        "lens": runtime.lens_card(),
    }


# test_motif_hypothesis's test against chance (`motif_statistics`, its reply's `selectivity`)
MOTIF_MIN_N = 4  # sentences a side before it judges: at three against three no split reaches p < 0.05 (the least is 0.1)
MOTIF_ALPHA = 0.05
MOTIF_DRAWS = 10_000  # every split of the sentences when there are at most this many, else this many shuffles
MOTIF_SEED = 0
MOTIF_CHUNK = 2048  # splits weighed at once


def _welch(x: np.ndarray, side: np.ndarray, floor: float) -> tuple:
    """Welch's t and its degrees of freedom, and the difference of means, for each split of `x` (`side`: [splits,
    len(x)], True for a + sentence). A difference or a side's spread within `floor` (rounding) counts as none, and a
    difference with no spread on either side is an infinite t."""
    n1 = side.sum(1)
    n2 = side.shape[1] - n1
    m1 = (side * x).sum(1) / n1
    m2 = (~side * x).sum(1) / n2
    v1 = (side * (x - m1[:, None]) ** 2).sum(1) / (n1 - 1)
    v2 = (~side * (x - m2[:, None]) ** 2).sum(1) / (n2 - 1)
    v1[v1 <= floor**2] = 0.0
    v2[v2 <= floor**2] = 0.0
    diff = m1 - m2
    diff[np.abs(diff) <= floor] = 0.0
    se2 = v1 / n1 + v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se2 > 0, diff / np.sqrt(se2), np.where(diff == 0, 0.0, np.copysign(np.inf, diff)))
        df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, df, diff


def _p_words(p: float) -> str:
    """A p to two significant figures, or to more where two would round it up to MOTIF_ALPHA (0.0496 is not 0.05)."""
    for k in (2, 3, 4, 6):
        s = f"{p:.{k}g}"
        if not p < MOTIF_ALPHA <= float(s):
            break
    return s


def _sides(n: int, n1: int, picks: Any) -> np.ndarray:
    """Splits as [splits, n] masks, True at each split's + sentences (`picks`: their indices, one row a split)."""
    picks = np.asarray(picks, dtype=np.int64).reshape(-1, n1)
    side = np.zeros((len(picks), n), dtype=bool)
    side[np.arange(len(picks))[:, None], picks] = True
    return side


def motif_statistics(pos: List[float], neg: List[float], draws: int = MOTIF_DRAWS, seed: int = MOTIF_SEED) -> Dict[str, Any]:
    """Whether a neuron's signed activations on the + sentences (`pos`, where it should fire) and the - sentences
    (`neg`, where it should stay quiet) differ by more than chance makes: test_motif_hypothesis's `selectivity`.

    It gives the n, mean and sample standard deviation of each side, Cohen's d over the pooled sample variance, and
    Welch's t with its degrees of freedom. Its p is two-sided, for |t|, from splitting the sentences between the sides
    afresh: exact over every split when there are at most `draws` (the share of splits whose |t| is at least the
    observed one, the observed split among them), else from `draws` random splits seeded with `seed`, (1 + as
    large) / (1 + draws), so never 0. Under the null that a sentence's side does not matter to the neuron the split is
    arbitrary; Welch's t keeps the test fair when the sides vary unequally. A difference with no spread on either side
    has an infinite t (sent as null) and is weighed as such. Fewer than two a side leave t and p null.

    `verdict`: CANNOT_TEST when a reading is not a finite number (its means, spreads, d, t and p are then null);
    TOO_FEW_TO_JUDGE below MOTIF_MIN_N sentences a side, whatever the numbers; else NOT_SHOWN unless p < MOTIF_ALPHA;
    else SELECTIVE_NORTH_POLE (+) when the + sentences read higher and their mean is above 0, SELECTIVE_SOUTH_POLE (-)
    when they read lower and their mean is below 0 (`face`), and DIFFERS_NOT_AS_GUESSED when the sides differ the other
    way: the quiet sentences fire a face more. `words` says it with its p and n, and `p_text` is that p as written
    there (`_p_words`, from the p unrounded), for a page to show as it is. One test is one neuron's: p's from several
    want correcting together (the study used Benjamini-Hochberg, E6 §2.3). An empty side is refused in plain words
    (ValueError)."""
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(neg, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        raise ValueError("The test needs sentences on both sides.")
    bad = int(np.count_nonzero(~np.isfinite(a)) + np.count_nonzero(~np.isfinite(b)))  # readings not finite numbers
    m1, m2 = (float(a.mean()), float(b.mean())) if not bad else (None, None)
    sd1 = float(a.std(ddof=1)) if n1 > 1 and not bad else None
    sd2 = float(b.std(ddof=1)) if n2 > 1 and not bad else None
    d = t = df = p = method = splits = None
    if n1 > 1 and n2 > 1 and not bad:
        x = np.concatenate([a, b])
        n = len(x)
        floor = 4 * n * float(np.finfo(np.float64).eps) * float(np.abs(x).max())
        t_obs, df_obs, _ = _welch(x, _sides(n, n1, [range(n1)]), floor)
        t, df = float(t_obs[0]), float(df_obs[0])
        pooled = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
        d = (m1 - m2) / math.sqrt(pooled) if pooled > floor**2 else None
        at_least = abs(t) * (1 - 1e-9)  # the observed split's own |t|, and its mirror's, within rounding
        total = math.comb(n, n1)
        count = 0
        if total <= draws:
            method, splits = "exact", total
            every = itertools.combinations(range(n), n1)
            for chunk in iter(lambda: list(itertools.islice(every, MOTIF_CHUNK)), []):
                count += int((np.abs(_welch(x, _sides(n, n1, chunk), floor)[0]) >= at_least).sum())
            p = count / total
        else:
            method, splits = "permutation", int(draws)
            rng = np.random.default_rng(seed)
            for start in range(0, draws, MOTIF_CHUNK):
                k = min(MOTIF_CHUNK, draws - start)
                picks = np.argsort(rng.random((k, n)), axis=1)[:, :n1]
                count += int((np.abs(_welch(x, _sides(n, n1, picks), floor)[0]) >= at_least).sum())
            p = (1 + count) / (1 + draws)
    n_words = f"{n1} a side" if n1 == n2 else f"{n1} and {n2}"
    face = None
    if bad:
        verdict = "CANNOT_TEST"
        words = f"cannot test: {bad} of the {n1 + n2} readings {'is' if bad == 1 else 'are'} not a finite number, n = {n_words}"
    elif n1 < MOTIF_MIN_N or n2 < MOTIF_MIN_N:
        verdict = "TOO_FEW_TO_JUDGE"
        words = f"too few to judge: n = {n_words}, and the test needs at least {MOTIF_MIN_N} a side"
    elif p is None or not p < MOTIF_ALPHA:
        verdict = "NOT_SHOWN"
        words = f"no difference shown: {'' if p is None else f'p = {_p_words(p)}, '}n = {n_words}"
    elif m1 > m2 and m1 > 0:
        verdict, face = "SELECTIVE_NORTH_POLE (+)", "+"
        words = f"selective on its + face, p = {_p_words(p)}, n = {n_words}"
    elif m1 < m2 and m1 < 0:
        verdict, face = "SELECTIVE_SOUTH_POLE (-)", "-"
        words = f"selective on its - face, p = {_p_words(p)}, n = {n_words}"
    else:
        verdict = "DIFFERS_NOT_AS_GUESSED"
        words = (f"the sides differ, p = {_p_words(p)}, n = {n_words}, but not as guessed: the quiet sentences fire its "
                 f"{'-' if m1 > m2 else '+'} face more")
    return {
        "read_at": "last word-piece",
        "n_pos": n1,
        "n_neg": n2,
        "mean_pos": _sig(m1),
        "mean_neg": _sig(m2),
        "sd_pos": _sig(sd1),
        "sd_neg": _sig(sd2),
        "cohens_d": _sig(d),
        "welch_t": _sig(t),
        "welch_df": _sig(df),
        "p": _sig(p),
        "p_text": None if p is None else _p_words(p),
        "p_method": method,
        "splits": splits,
        "seed": int(seed) if method == "permutation" else None,
        "min_n": MOTIF_MIN_N,
        "alpha": MOTIF_ALPHA,
        "verdict": verdict,
        "face": face,
        "words": words,
    }


@mcp.tool()
async def test_motif_hypothesis(
    layer: int,
    neuron_idx: int,
    pos_prompts: List[str],
    neg_prompts: List[str],
    chat: bool = False,
    system: Optional[str] = None,
) -> Dict[str, Any]:
    """Test whether Bipolar GeGLU neuron `(layer, neuron_idx)` selectively encodes a hypothesized
    motif on `pos_prompts` vs. `neg_prompts`.

    Captures `Address("mlp_act", layer)` across all positive and negative prompts, evaluates
    both last-token and max-over-sequence signed activations, computes Cohen's d effect size,
    and determines whether the motif is bound to the North (`+`) or South (`-`) pole.

    `chat=True` wraps each prompt as the user turn of the chat template (optional `system` turn, thinking
    off); the last token is then the end of the model-turn header.

    `verdict` and `cohens_d` are kept as first built, for old callers: Cohen's d over population variances (plus
    1e-8), with a verdict at |d| >= 0.8 for any number of sentences and no test against chance, so that one sentence
    a side reads as "CONFIRMED" (d near 10^4). Quote `selectivity` instead (`motif_statistics`): on the last word-piece's
    unrounded readings, the n each side, sample variances, Welch's t, a two-sided p for it by splitting the sentences
    afresh (exact over every split up to MOTIF_DRAWS of them, else MOTIF_DRAWS shuffles seeded MOTIF_SEED), and its own
    verdict and `words`, which judge nothing below MOTIF_MIN_N sentences a side, nor when a reading is not a finite
    number (CANNOT_TEST: the reply still comes, with the keys as first built). The peak is reported but not tested.

    Refusals are plain-word ValueErrors (an empty side, a layer or a neuron that does not exist), all before the model
    is run; an empty side was once read as one reading of 0.0, and an out-of-range layer moved to the nearest.

    `lens` names the lens the `pole_j_vindex_signature` words went through; read its `caution` before trusting them.
    `pole_label_check` marks that pole's label by the study's measure (`label_gate`, the rule, comes before `lens`).
    """
    runtime, vault = get_runtime_and_vault()
    assert runtime.eager_model is not None
    lm = cc.language_model(runtime.hf_model)
    n_layers = len(lm.layers)

    # plain-word refusals, all before the model is run
    if not _whole(layer) or not 0 <= layer < n_layers:
        raise ValueError(
            f"layer is {layer!r}: layers count from 0 to {n_layers - 1} here (the pages count them 1 to {n_layers})."
        )
    d_ff = _ffn_width(lm.layers[layer].mlp.down_proj)
    if not _whole(neuron_idx) or not 0 <= neuron_idx < d_ff:
        raise ValueError(f"neuron_idx is {neuron_idx!r}: each layer's neurons count from 0 to {d_ff - 1}.")
    for what, prompts in (("pos_prompts", pos_prompts), ("neg_prompts", neg_prompts)):
        if isinstance(prompts, (str, bytes)) or not prompts:
            raise ValueError(
                f"{what} is empty: the test reads the neuron on sentences where it should fire (pos_prompts) and where "
                f"it should stay quiet (neg_prompts), one or more on each side, and judges from {MOTIF_MIN_N} a side."
            )

    l_idx, n_idx = int(layer), int(neuron_idx)
    addr = Address("mlp_act", l_idx)

    async def _eval_prompts(prompts: List[str]) -> tuple:
        """Each prompt's record, and its last word-piece's reading unrounded."""
        records: List[Dict[str, Any]] = []
        lasts: List[float] = []
        for p_text in prompts:
            p_ids = runtime.tokenize_prompt(runtime.compose_prompt(p_text, chat=chat, system=system))
            caps = await runtime.eager_model.capture(p_ids, [addr])
            seq_act = caps[addr][:, n_idx].float()  # [seq_len]
            last_act = float(seq_act[-1].item())
            peak_pos = int(seq_act.abs().argmax().item())
            peak_act = float(seq_act[peak_pos].item())
            peak_tok = runtime.tokenizer.decode([p_ids[peak_pos]])
            lasts.append(last_act)
            records.append(
                {
                    "prompt": p_text,
                    "last_token_signed_act": round(last_act, 4),
                    "peak_signed_act": round(peak_act, 4),
                    "peak_token": peak_tok,
                    "polarity": "+" if last_act >= 0 else "-",
                }
            )
        return records, lasts

    pos_records, pos_last = await _eval_prompts(pos_prompts)
    neg_records, neg_last = await _eval_prompts(neg_prompts)

    # as first built, for old callers: the rounded readings, population variances, a threshold on d at any n
    pos_vals = [r["last_token_signed_act"] for r in pos_records]
    neg_vals = [r["last_token_signed_act"] for r in neg_records]

    mean_pos = sum(pos_vals) / len(pos_vals)
    mean_neg = sum(neg_vals) / len(neg_vals)
    var_pos = sum((x - mean_pos) ** 2 for x in pos_vals) / max(len(pos_vals), 1)
    var_neg = sum((x - mean_neg) ** 2 for x in neg_vals) / max(len(neg_vals), 1)
    pooled_std = math.sqrt(0.5 * (var_pos + var_neg) + 1e-8)
    cohens_d = (mean_pos - mean_neg) / pooled_std

    if cohens_d >= 0.8 and mean_pos > 0:
        verdict = "CONFIRMED_NORTH_POLE (+)"
    elif cohens_d <= -0.8 and mean_pos < 0:
        verdict = "CONFIRMED_SOUTH_POLE (-)"
    elif abs(cohens_d) >= 0.5:
        verdict = "WEAK_SELECTIVITY"
    else:
        verdict = "REJECTED_OR_INCONCLUSIVE"

    pole_info = vault.inspect_bipolar_slot(l_idx, n_idx, polarity=1.0 if mean_pos >= 0 else -1.0, top_k_tokens=5)

    return {
        "layer": l_idx,
        "neuron_idx": n_idx,
        "chat": bool(chat),
        "verdict": verdict,
        "cohens_d": round(cohens_d, 4),
        "mean_pos_signed_act": round(mean_pos, 4),
        "mean_neg_signed_act": round(mean_neg, 4),
        "selectivity": motif_statistics(pos_last, neg_last),
        "pos_prompt_results": pos_records,
        "neg_prompt_results": neg_records,
        "pole_j_vindex_signature": pole_info["active_pole"]["j_vindex_tokens"],
        "pole_label_check": pole_info["active_pole"]["label_check"],
        "label_gate": LABEL_GATE_CARD,
        "lens": runtime.lens_card(),
    }


def _whole(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _known(what: str, value: Any, known: tuple) -> None:
    if value not in known:
        choices = " or ".join(f'"{k}"' for k in known)
        shown = str(value) if len(str(value)) <= 40 else str(value)[:40] + "…"
        raise ValueError(f'{what} is "{shown}", which this instrument does not know: it must be {choices}.')


def _check_ids(ids: Any, what: str, vocab: int) -> List[int]:
    for t in ids:
        if not _whole(t) or not 0 <= t < vocab:
            raise ValueError(f"{what} holds {t!r}, which is not a word-piece id: ids are whole numbers from 0 to {vocab - 1}.")
    return [int(t) for t in ids]


def _ffn_width(down_proj: Any) -> int:
    """The feed-forward width a down_proj reads (21,504 on Gemma-4-31B): a plain Linear's in_features, else the packed
    W4A16 weight's eight nibbles per int32."""
    n = getattr(down_proj, "in_features", None)
    if _whole(n) and n > 0:
        return int(n)
    return 8 * int(down_proj.weight_packed.shape[1])


def _study_ids(runtime: TheodoliteRuntime) -> List[List[int]]:
    """The study's 36 texts as the model reads them: each chat through the chat template (thinking off) with its begun
    answer after the model-turn header, as lab.chat(user, assistant=prefix) composes it; prose and code as bare text
    after the start mark (`causal_check.STUDY_TEXTS`; scripts/check_causal_check.py compares the ids with labkit's)."""
    cached = getattr(runtime, "_causal_study_ids", None)
    if cached is not None:
        return cached
    out: List[List[int]] = []
    for t in cc.STUDY_TEXTS:
        text = runtime.compose_prompt(t[1], chat=True, assistant_prefix=t[2]) if t[0] == "chat" else t[1]
        out.append(runtime.tokenize_prompt(text))
    setattr(runtime, "_causal_study_ids", out)
    return out


def _study_max_abs(runtime: TheodoliteRuntime, vault: LarqlVaultEngine, lm: Any) -> tuple:
    """max |a| of every neuron over the study's 36 texts, [n_layers, d_ff], for the model as it now stands: cached on the
    runtime per set of vault edits in force (their patch ids; the empty tuple is the clean model, reused after a
    ROLLBACK). Returns (table, whether it was cached)."""
    cache: Optional[Dict[tuple, torch.Tensor]] = getattr(runtime, "_causal_study_maxabs", None)
    if cache is None:
        cache = {}
        setattr(runtime, "_causal_study_maxabs", cache)
    key = tuple(int(p.patch_id) for p in vault.edits_in_force())
    if key in cache:
        return cache[key], True
    inp, att, _last, valid = cc.pad_right(_study_ids(runtime), cc.model_device(lm))
    table = cc.study_max_abs(lm, inp, att, valid)
    for old in [k for k in cache if k and k != key][: max(0, len(cache) - 3)]:
        del cache[old]  # keep the clean model's table and a few edited ones
    cache[key] = table
    return table, False


def _sig(x: Optional[float], n: int = 6) -> Optional[float]:
    """A float to `n` significant figures (KL at k = 10 is often ~1e-3), None kept, not-finite sent as None."""
    if x is None or not math.isfinite(float(x)):
        return None
    return float(f"{float(x):.{n}g}")


@mcp.tool()
async def causal_spot_check(
    layer: int,
    neuron_idx: int,
    polarity: Optional[float] = None,
    contexts: str = "given",
    token_ids: Optional[List[int]] = None,
    prompt: Optional[str] = None,
    chat: bool = False,
    system: Optional[str] = None,
    assistant_prefix: str = "",
    k: float = cc.K_DEFAULT,
    write: str = "activation",
    scale_ref: str = "study",
    floor: bool = True,
    top_n: int = 10,
    track_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Test a neuron's label causally: turn neuron `(layer, neuron_idx)` up by k times its strongest firing, on each face
    (+ and -), and list the word-pieces whose log-probability the model's own odds raised and lowered most, beside an
    untouched copy of the same reading in the same forward (the study's N1A protocol, at the k = 10 its second look
    advised, erratum E47; `backend/causal_check.py`).

    `contexts="given"` (default) reads one text at its last word-piece: `token_ids` (2 to MAX_SWEEP_POSITIONS exact ids,
    a sighting's ids up to its read position; position 0, the start mark, is never pushed) or else `prompt`, composed as
    `trace_downstream_readers` composes it (`chat`, `system`, `assistant_prefix`). `contexts="study"` reads the study's
    36 texts (12 chats, 12 passages of prose, 12 of code), each at its own last word-piece, and averages over them; only
    this mode compares with the study's numbers. `write` is "activation" (mlp_act += delta at the read position) or
    "frozen_norm" (the same first-order change added after the post-feed-forward norm, which cannot squash it; its
    `linearity` is then null). `scale_ref` sizes the push by the neuron's strongest firing on the study's texts
    ("study", default; cached per set of vault edits in force) or on the given text ("this_text", given mode only).
    `floor` (default on) also runs the same push at k = 0.01, the bf16 rounding floor the study measured beside every
    push. `top_n` (1-50) words rise and fall per face; `track_ids` (at most 16) are followed on both faces whatever
    their rank. `polarity` names the face the reply puts first (`polarity`): None takes the face the neuron fired with
    at the read position in given mode, else "+"; both faces are always measured.

    The effect is read from the model's own odds, without any nudge (`steered: false`) and with any vault edits in
    force (`vault_edits`), so it does not depend on the lens. Only the label it is compared with goes through the J++
    lens: each face's `label_overlap` counts how many of its label's first ten ids (the lens's and the naked eye's, the
    same ids `inspect_pole`'s label_check counts) are among the hundred ids this push raised most (P@10, whole
    vocabulary), and carries that face's `label_check`; `label_gate` and `lens` come last. The study's bar (three of
    ten, over 36 texts) was set on pushes at k = 60; one word-piece is a smaller test than 36 texts. A push
    shows what the neuron does when turned up, not what it does when it fires on its own. Refusals are plain-word
    ValueErrors, and running out of GPU memory comes back as one.
    """
    t0 = time.perf_counter()
    runtime, vault = get_runtime_and_vault()
    assert runtime.hf_model is not None, "Call runtime.load() first."
    lm = cc.language_model(runtime.hf_model)
    n_layers = len(lm.layers)
    vocab = int(runtime.lm_head_weight.shape[0])

    # plain-word refusals, all before the GPU is touched
    if not _whole(layer) or not 0 <= layer < n_layers:
        raise ValueError(
            f"layer is {layer!r}: layers count from 0 to {n_layers - 1} here (the pages count them 1 to {n_layers})."
        )
    d_ff = _ffn_width(lm.layers[layer].mlp.down_proj)
    if not _whole(neuron_idx) or not 0 <= neuron_idx < d_ff:
        raise ValueError(f"neuron_idx is {neuron_idx!r}: each layer's neurons count from 0 to {d_ff - 1}.")
    if polarity is not None and not _number(polarity):
        raise ValueError(f"polarity is {polarity!r}: it must be a number (+1 or -1 names the face), or left out.")
    _known("contexts", contexts, cc.CAUSAL_CONTEXTS)
    _known("write", write, cc.CAUSAL_WRITES)
    _known("scale_ref", scale_ref, cc.SCALE_REFS)
    if not isinstance(floor, bool) or not isinstance(chat, bool):
        raise ValueError("floor and chat are true or false.")
    if not _number(k) or not 0 < float(k) <= cc.K_MAX:
        raise ValueError(
            f"k is {k!r}: the test turns the neuron up by k times its strongest firing, and k must be more than 0 and "
            f"at most {cc.K_MAX:g} (the study tested at 3, 10, 60 and 180)."
        )
    k = float(k)
    if floor and k <= cc.K_FLOOR:
        raise ValueError(
            f"k is {k:g}: with the floor on, the push must be larger than the floor's own k = {cc.K_FLOOR:g}, the "
            "rounding-noise push it is compared with. Push harder, or turn the floor off (floor false)."
        )
    if not _whole(top_n) or not 1 <= top_n <= cc.TOP_N_MAX:
        raise ValueError(f"top_n is {top_n!r}: the lists of words that rose and fell hold 1 to {cc.TOP_N_MAX} each.")
    if track_ids is not None and not isinstance(track_ids, (list, tuple)):
        raise ValueError("track_ids is a list of word-piece ids.")
    if track_ids is not None and len(track_ids) > cc.TRACK_MAX:
        raise ValueError(f"track_ids holds {len(track_ids)} ids: at most {cc.TRACK_MAX} word-pieces can be followed.")
    tracked = _check_ids(track_ids or [], "track_ids", vocab)

    ids: Optional[List[int]] = None
    text: Optional[str] = None
    if contexts == "given":
        if token_ids is not None:
            if not isinstance(token_ids, (list, tuple)):
                raise ValueError("token_ids is a list of word-piece ids.")
            if not 2 <= len(token_ids) <= MAX_SWEEP_POSITIONS:
                n = len(token_ids)
                raise ValueError(
                    f"token_ids holds {n} word-piece ids: the test pushes the last of them, after the start mark, so "
                    f"it needs 2 to {MAX_SWEEP_POSITIONS}."
                )
            ids = _check_ids(token_ids, "token_ids", vocab)
            text = runtime.tokenizer.decode(ids)
        elif prompt is not None:
            text = runtime.compose_prompt(str(prompt), chat=chat, system=system, assistant_prefix=assistant_prefix)
            ids = runtime.tokenize_prompt(text)
            if not 2 <= len(ids) <= MAX_SWEEP_POSITIONS:
                raise ValueError(
                    f"The prompt is {len(ids)} word-pieces with its start mark: the test pushes the last of them, so it "
                    f"needs 2 to {MAX_SWEEP_POSITIONS}."
                )
        else:
            raise ValueError(
                "A test at one word-piece needs the text it is read on: give token_ids (a sighting's ids up to its read "
                "position) or a prompt."
            )
    else:
        if token_ids is not None or prompt is not None:
            raise ValueError(
                "A test on the study's 36 texts reads those texts only: leave out token_ids and prompt (they are for a "
                'test at one word-piece, contexts "given").'
            )
        if scale_ref == "this_text":
            raise ValueError(
                'scale_ref "this_text" sizes the push by the given text, and a test on the study\'s texts has none: use '
                'scale_ref "study".'
            )

    l, i = int(layer), int(neuron_idx)
    edits = vault.edits_in_force()
    read_pos_by_context: Optional[List[int]] = None
    decode = lambda t: runtime.tokenizer.decode([int(t)], clean_up_tokenization_spaces=False)  # noqa: E731
    try:
        # 1. the push's size: k times the neuron's strongest firing on the reference texts
        t_scale = time.perf_counter()
        scale_cached = False
        if scale_ref == "study":
            table, scale_cached = _study_max_abs(runtime, vault, lm)
            max_abs = float(table[l, i])
        else:
            assert ids is not None
            max_abs = cc.text_max_abs(lm, l, i, ids)
        scale_ms = (time.perf_counter() - t_scale) * 1000.0
        if not max_abs > 0.0:
            where = "the study's 36 texts" if scale_ref == "study" else "this text"
            raise ValueError(
                f"Neuron {i} of layer {l + 1} (as the pages count; index {l}) never fired on {where} (its strongest "
                "firing is 0), so there is nothing to size the test by."
            )
        deltas = [k * max_abs, -k * max_abs]
        if floor:
            deltas += [cc.K_FLOOR * max_abs, -cc.K_FLOOR * max_abs]

        # 2. the push and its untouched copy, in one forward (halving on out of memory)
        idx_t = torch.tensor([i], device=runtime.device, dtype=torch.long)
        w_down = runtime.slice_w_down(l, idx_t)[0]  # the column as the model now uses it (vault edits included)
        post_norm = lm.layers[l].post_feedforward_layernorm
        w_post, eps = post_norm.weight.detach(), float(getattr(post_norm, "eps", 1e-6))
        t_check = time.perf_counter()
        study_ids = _study_ids(runtime) if contexts == "study" else None
        run = cc.run_check(
            lm,
            l,
            i,
            deltas,
            write,
            w_down,
            w_post,
            eps,
            runtime.lm_head_weight,
            token_ids=ids,
            texts_ids=study_ids,
        )
        check_ms = (time.perf_counter() - t_check) * 1000.0

        # 3. the labels each face is compared with: the same first ten ids as inspect_pole's label_check
        J = runtime.jpp_lenses[l]
        w_scaled = w_down.float()
        if l in runtime._post_ffn_norm_weights:
            w_scaled = w_scaled * runtime._post_ffn_norm_weights[l].to(w_scaled.device).float()
        w_scaled = w_scaled.to(J.dtype)
        fnw, lmw = runtime.final_norm_weight, runtime.lm_head_weight
        lens_id = (runtime.lens_card() or {}).get("id")
        edited = (l, i) in {(p.layer_idx, p.neuron_idx) for p in edits}

        effects = [cc.face_effect(s, b) for s, b in zip(run.steered, run.bases)]
        acts = [float(a) for a in run.act.cpu().tolist()]
        sign_here = (1.0 if acts[0] >= 0 else -1.0) if contexts == "given" else 1.0
        opened = "+" if (float(polarity) if polarity is not None else sign_here) >= 0 else "-"

        def word(t: int, eff: cc.FaceEffect) -> Dict[str, Any]:
            return {
                "token_id": int(t),
                "token": decode(t),
                "dlogprob": _sig(float(eff.c[t])),
                "base_logprob": _sig(float(eff.base_lp[t])),
                "steered_logprob": _sig(float(eff.steered_lp[t])),
            }

        def linear(delta: float) -> Dict[str, Any]:
            lr = cc.linear_regime(run.y, w_down, w_post, delta, eps)
            wof = lr["write_over_ffn"]
            if write == "activation":
                lin, abl = lr["linearity"], lr["ablation_cos"]
            else:
                lin, abl = None, lr["frozen_ablation_cos"]  # the change written is the straight one itself
            out: Dict[str, Any] = {
                "write_over_ffn": _sig(cc.summary(wof)[0], 4),
                "linearity": _sig(cc.summary(lin)[0], 4) if lin is not None else None,
                "ablation_cos": _sig(cc.summary(abl)[0], 4),
                "linearity_q10": _sig(cc.summary(lin)[1], 4) if lin is not None else None,
                "by_context": {
                    "write_over_ffn": [_sig(x, 4) for x in wof.tolist()],
                    "linearity": [_sig(x, 4) for x in lin.tolist()] if lin is not None else None,
                    "ablation_cos": [_sig(x, 4) for x in abl.tolist()],
                },
            }
            if lin is None:
                out["linearity_reason"] = cc.FROZEN_NORM_NO_LINEARITY
            return out

        def stats(eff: cc.FaceEffect) -> Dict[str, Any]:
            kl_ctx = eff.kl_by_context.float().cpu()
            return {
                "kl_mean": _sig(float(kl_ctx.mean())),
                "kl_median": _sig(float(kl_ctx.median())),
                "kl_by_context": [_sig(x) for x in kl_ctx.tolist()],
                "split_half_rho": _sig(eff.split_half_rho, 4),
            }

        faces: Dict[str, Any] = {}
        for j, (face, s) in enumerate((("+", 1.0), ("-", -1.0))):
            eff = effects[j]
            rises, falls = cc.movers(eff.c, int(top_n))
            vec_raw = (s * w_scaled).unsqueeze(0)
            vec_jpp = vec_raw @ J.T
            lens_ids = label_top_ids(vec_jpp, fnw, lmw, unembed_gemma4(vec_jpp, fnw, lmw))[0]
            eye_ids = label_top_ids(vec_raw, fnw, lmw, unembed_gemma4(vec_raw, fnw, lmw))[0]
            check = label_check(l, lens_ids, eye_ids, lens_id, edited=edited, silenced=writes_nothing(vec_raw)[0])
            silenced = "writes_nothing" in check["reasons"]
            top100 = set(cc.top_ids(eff.c, cc.CAUSAL_REF_K))

            def overlap(label: List[int]) -> Dict[str, Any]:
                # a silenced neuron (a DELETE in force) writes nothing: every word ties, so it has no label to test
                return {
                    "p10": None if silenced else round(cc.p_at(label, eff.c), 4),
                    "label_ids": [int(t) for t in label],
                    "tokens": [decode(t) for t in label],
                    "in_top100": [int(t) in top100 for t in label],
                }

            st = stats(eff)
            fl = effects[2 + j] if floor else None
            faces[face] = {
                "delta": _sig(deltas[j]),
                **{key: st[key] for key in ("kl_mean", "kl_median", "kl_by_context")},
                "kl_over_floor": (
                    _sig(float(eff.kl_by_context.mean()) / float(fl.kl_by_context.mean()), 4)
                    if fl is not None and float(fl.kl_by_context.mean()) > 0
                    else None
                ),
                "split_half_rho": st["split_half_rho"],
                "rises": [word(t, eff) for t in rises],
                "falls": [word(t, eff) for t in falls],
                "tracked": [word(t, eff) for t in tracked],
                # how many of the hundred word-pieces this push raised most the floor's push of this sign raised too
                "floor_top100_overlap": (
                    len(top100 & set(cc.top_ids(fl.c, cc.CAUSAL_REF_K))) if fl is not None else None
                ),
                "linear": linear(deltas[j]),
                "label_overlap": {
                    "lens": overlap(lens_ids),
                    "naked_eye": overlap(eye_ids),
                    "label_k": cc.LABEL_K,
                    "causal_k": cc.CAUSAL_REF_K,
                    "label_check": check,
                },
            }
        c_plus, c_minus = effects[0].c, effects[1].c
        neg_top100 = set(cc.top_ids(-c_minus, cc.CAUSAL_REF_K))
        floor_reply = None
        if floor:
            floor_reply = {}
            for j, face in ((2, "+"), (3, "-")):
                st = stats(effects[j])
                floor_reply[face] = {"delta": _sig(deltas[j]), **st}
        base_top = None
        if contexts == "given":
            b = run.base[0]
            tv, ti = torch.topk(b, k=5)
            base_top = [
                {"token_id": int(t), "token": decode(t), "logprob": _sig(float(v))}
                for v, t in zip(tv.cpu().tolist(), ti.cpu().tolist())
            ]
        if study_ids is not None:
            read_pos_by_context = [len(x) - 1 for x in study_ids]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        from backend.cpu_memory import is_oom  # the card's, or the CPU allocator's

        if not is_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        where = "GPU" if isinstance(exc, torch.cuda.OutOfMemoryError) else "CPU"
        raise ValueError(
            f"The {where} ran out of memory during this test, even with its pushed rows split over separate forwards. "
            "Try a shorter text, or test it on the study's 36 texts."
        ) from exc

    given = contexts == "given"
    return {
        "layer": l,
        "neuron_idx": i,
        "polarity": opened,
        "contexts": contexts,
        "n_contexts": 1 if given else len(cc.STUDY_TEXTS),
        "prompt": text,
        "chat": bool(chat),  # as asked; with token_ids the template was not applied here
        "token_ids_given": token_ids is not None,
        "read_pos": (len(ids) - 1) if given and ids is not None else None,
        "read_pos_by_context": None if given else read_pos_by_context,
        "read_token_id": int(ids[-1]) if given and ids is not None else None,
        "read_token": decode(ids[-1]) if given and ids is not None else None,
        "study_texts": None if given else cc.STUDY_TEXTS_SOURCE,
        # read without any nudge, also when the sighting it reads for was nudged
        "steered": False,
        "write": write,
        "k": k,
        "k_floor": cc.K_FLOOR if floor else None,
        "scale_ref": scale_ref,
        "max_abs_act": _sig(max_abs),
        "max_abs_act_over": (
            "the study's 36 texts, every word-piece after the start mark"
            if scale_ref == "study"
            else "this text, every word-piece after the start mark"
        ),
        "max_abs_act_cached": scale_cached,
        "act_here": _sig(acts[0]) if given else None,
        "act_here_by_context": None if given else [_sig(a) for a in acts],
        "faces": faces,
        "antipodal": {
            "rho": _sig(cc.spearman(c_plus, c_minus), 4),
            # of the ten word-pieces the + push raised most, how many are among the hundred the - push lowered most
            "top10_in_neg_top100": len(set(cc.top_ids(c_plus, cc.LABEL_K)) & neg_top100),
        },
        "floor": floor_reply,
        "base_top": base_top,
        "forwards": run.forwards,
        "slots_per_forward": run.slots_per_forward,
        "base_rows_identical": run.base_rows_identical,
        "vault_edits": [int(p.patch_id) for p in edits],
        "timing_ms": {
            "scale_ms": round(scale_ms, 1),
            "check_ms": round(check_ms, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        },
        "label_gate": LABEL_GATE_CARD,
        "lens": runtime.lens_card(),
    }


@mcp.tool()
def query_larql_vault(lql_query: str) -> Dict[str, Any]:
    """Execute a LARQL / VINDEX3 query or declarative `.patch` edit statement. An edit (`UPDATE`, `INSERT`, `DELETE`)
    moves one column of one layer's down_proj, rank one, with nothing solved for: the statements are named after
    MEMIT, but it is not MEMIT. This tool takes no voice (the model's answer before and after an edit): only POST
    /api/larql and the WebSocket "larql" action, which hold the GPU lock around the edit, offer one.

    The statements that read words through the J++ lens (`SELECT * FROM EDGES|POLES NEAREST`, `DESCRIBE POLE`) or
    write an edit along a direction pulled back through it (`UPDATE`, `INSERT`) also return `lens`, naming that lens;
    read its `caution` before trusting the words. `DESCRIBE`, `DELETE` and `ROLLBACK` touch no lens and return none.
    The two that read neuron labels mark each one under `label_check` (each NEAREST row, each DESCRIBE POLE face) and
    return the rule under `label_gate`, just before `lens`.
    """
    _rt, vault = get_runtime_and_vault()
    return vault.execute_lql(lql_query)


if __name__ == "__main__":
    mcp.run()
