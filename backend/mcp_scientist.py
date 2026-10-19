"""FastMCP Autonomous Circuit Scientist (`backend/mcp_scientist.py`).

Exposes three core mechanistic interpretability tools via FastMCP:
1. `inspect_pole(layer, neuron_idx, polarity)`
2. `trace_downstream_readers(layer, neuron_idx, prompt)`
3. `test_motif_hypothesis(layer, neuron_idx, pos_prompts, neg_prompts)`
Plus `query_larql_vault(lql_query)` for declarative VINDEX3 & `.patch` weight surgery.

The three scientist tools decode words through the J++ lens, and each reply names it under "lens"
(`TheodoliteRuntime.lens_card`: what it was fitted on, what it reads, its source, and a `caution` on how far to trust
its words), so an agent calling them over MCP gets the same record as the HTTP and WebSocket replies, which pass these
replies on unchanged. `query_larql_vault` does the same for the vault statements that read words or write an edit
through that lens (NEAREST, DESCRIBE POLE, UPDATE, INSERT: `LarqlVaultEngine.execute_lql`, which also answers
POST /api/larql and the WebSocket "larql" action).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from fastmcp import FastMCP
from interp_engine import Address

from backend.engine_runtime import MAX_SWEEP_POSITIONS, TheodoliteRuntime
from backend.larql_vault import LarqlVaultEngine
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
    """Lazily obtain or initialize the singleton `TheodoliteRuntime` and `LarqlVaultEngine`."""
    global _SHARED_RUNTIME, _SHARED_VAULT
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = TheodoliteRuntime(load_jlens=False).load()
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

    `lens` names the lens the J-VINDEX words went through; read its `caution` before trusting them.
    """
    rt, vault = get_runtime_and_vault()
    reply = vault.inspect_bipolar_slot(
        layer_idx=int(layer),
        neuron_idx=int(neuron_idx),
        polarity=float(polarity),
        top_k_tokens=8,
    )
    return {**reply, "lens": rt.lens_card()}


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
    """
    if score not in THREAD_SCORES:
        raise ValueError(f"Unknown score {score!r}; expected one of {THREAD_SCORES}")
    runtime, _vault = get_runtime_and_vault()
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
    caps = await runtime.eager_model.capture(prompt_ids, points)

    src_mlp_act = caps[Address("mlp_act", l_src)].to(runtime.device)[target_pos]
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
        caps[Address("mlp_out", l_src)].to(runtime.device)[target_pos].float(),
        runtime._post_ffn_norm_weights[l_src],
        layer_scalars[l_src],
    )  # [1, 5376]

    candidates: List[Dict[str, Any]] = []
    for l_dst in range(l_src + 1, l_end + 1):
        dst_mlp_act = caps[Address("mlp_act", l_dst)].to(runtime.device)[target_pos]
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
            caps[Address("resid_mid", l_dst)].to(runtime.device)[target_pos].float(),
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

    candidates.sort(key=lambda item: item["abs_score"], reverse=True)
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
        "lens": runtime.lens_card(),
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

    `lens` names the lens the `pole_j_vindex_signature` words went through; read its `caution` before trusting them.
    """
    runtime, vault = get_runtime_and_vault()
    assert runtime.eager_model is not None

    l_idx = max(0, min(int(layer), N_LAYERS - 1))
    n_idx = max(0, min(int(neuron_idx), D_FF - 1))
    addr = Address("mlp_act", l_idx)

    async def _eval_prompts(prompts: List[str]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for p_text in prompts:
            p_ids = runtime.tokenize_prompt(runtime.compose_prompt(p_text, chat=chat, system=system))
            caps = await runtime.eager_model.capture(p_ids, [addr])
            seq_act = caps[addr][:, n_idx].float()  # [seq_len]
            last_act = float(seq_act[-1].item())
            peak_pos = int(seq_act.abs().argmax().item())
            peak_act = float(seq_act[peak_pos].item())
            peak_tok = runtime.tokenizer.decode([p_ids[peak_pos]])
            records.append(
                {
                    "prompt": p_text,
                    "last_token_signed_act": round(last_act, 4),
                    "peak_signed_act": round(peak_act, 4),
                    "peak_token": peak_tok,
                    "polarity": "+" if last_act >= 0 else "-",
                }
            )
        return records

    pos_records = await _eval_prompts(pos_prompts)
    neg_records = await _eval_prompts(neg_prompts)

    pos_vals = [r["last_token_signed_act"] for r in pos_records] or [0.0]
    neg_vals = [r["last_token_signed_act"] for r in neg_records] or [0.0]

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
        "pos_prompt_results": pos_records,
        "neg_prompt_results": neg_records,
        "pole_j_vindex_signature": pole_info["active_pole"]["j_vindex_tokens"],
        "lens": runtime.lens_card(),
    }


@mcp.tool()
def query_larql_vault(lql_query: str) -> Dict[str, Any]:
    """Execute a LARQL / VINDEX3 query or declarative `.patch` weight surgery statement.

    The statements that read words through the J++ lens (`SELECT * FROM EDGES|POLES NEAREST`, `DESCRIBE POLE`) or
    write an edit along a direction pulled back through it (`UPDATE`, `INSERT`) also return `lens`, naming that lens;
    read its `caution` before trusting the words. `DESCRIBE`, `DELETE` and `ROLLBACK` touch no lens and return none.
    """
    _rt, vault = get_runtime_and_vault()
    return vault.execute_lql(lql_query)


if __name__ == "__main__":
    mcp.run()
