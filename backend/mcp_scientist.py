"""FastMCP Autonomous Circuit Scientist (`backend/mcp_scientist.py`).

Exposes three core mechanistic interpretability tools via FastMCP:
1. `inspect_pole(layer, neuron_idx, polarity)`
2. `trace_downstream_readers(layer, neuron_idx, prompt)`
3. `test_motif_hypothesis(layer, neuron_idx, pos_prompts, neg_prompts)`
Plus `query_larql_vault(lql_query)` for declarative VINDEX3 & `.patch` weight surgery.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from fastmcp import FastMCP
from interp_engine import Address

from backend.engine_runtime import TheodoliteRuntime
from backend.larql_vault import LarqlVaultEngine
from backend.theodolite_math import (
    D_FF,
    N_LAYERS,
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
    """
    _rt, vault = get_runtime_and_vault()
    return vault.inspect_bipolar_slot(
        layer_idx=int(layer),
        neuron_idx=int(neuron_idx),
        polarity=float(polarity),
        top_k_tokens=8,
    )


@mcp.tool()
async def trace_downstream_readers(
    layer: int,
    neuron_idx: int,
    prompt: str,
    polarity: Optional[float] = None,
    max_layer_hop: int = 6,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Trace verified Live Downstream Reader poles for `(layer, neuron_idx)` on `prompt`.

    Computes the Spillover-Filtered coupling score between `(layer, neuron_idx, sigma_i)` and
    active poles `(l', j, sigma_j)` at downstream layers `l' in layer+1 .. layer+max_layer_hop`:
        C_{(l, i) -> (l', j)} = dot(sigma_i * W_down[l, i], W_gate[l', j]) * abs(mlp_act[l', j])
    """
    runtime, _vault = get_runtime_and_vault()
    assert runtime.eager_model is not None

    l_src = max(0, min(int(layer), N_LAYERS - 2))
    n_src = max(0, min(int(neuron_idx), D_FF - 1))
    l_end = min(l_src + int(max_layer_hop), N_LAYERS - 1)

    token_tensor = runtime.eager_model.to_tokens(prompt, prepend_bos=True)
    prompt_ids = [int(t) for t in token_tensor[0].tolist()]
    target_pos = len(prompt_ids) - 1

    points = [Address("mlp_act", l) for l in range(l_src, l_end + 1)]
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
    w_down_src = runtime.slice_w_down(l_src, src_idx_t)[0].float()
    if l_src in runtime._post_ffn_norm_weights:
        w_down_src = w_down_src * runtime._post_ffn_norm_weights[l_src].to(runtime.device).float()
    signed_writer = sigma_i * w_down_src  # [5376]

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
        if l_dst in runtime._pre_ffn_norm_weights:
            w_gate_dst = w_gate_dst * runtime._pre_ffn_norm_weights[l_dst].to(runtime.device).float()

        dots = w_gate_dst @ signed_writer  # [25]
        couplings = dots * poles_dst.abs_activations.float()

        for r_idx in range(poles_dst.neuron_indices.numel()):
            c_score = float(couplings[r_idx].item())
            d_val = float(dots[r_idx].item())
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
                    "dst_j_vindex_tokens": jpp_tokens,
                }
            )

    candidates.sort(key=lambda item: item["abs_coupling"], reverse=True)
    return {
        "prompt": prompt,
        "source_pole": {
            "layer": l_src,
            "neuron_idx": n_src,
            "polarity": "+" if sigma_i >= 0 else "-",
            "signed_activation_on_prompt": round(src_signed_act, 4),
        },
        "top_downstream_readers": candidates[: int(top_k)],
    }


@mcp.tool()
async def test_motif_hypothesis(
    layer: int,
    neuron_idx: int,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> Dict[str, Any]:
    """Test whether Bipolar GeGLU neuron `(layer, neuron_idx)` selectively encodes a hypothesized
    motif on `pos_prompts` vs. `neg_prompts`.

    Captures `Address("mlp_act", layer)` across all positive and negative prompts, evaluates
    both last-token and max-over-sequence signed activations, computes Cohen's d effect size,
    and determines whether the motif is bound to the North (`+`) or South (`-`) pole.
    """
    runtime, vault = get_runtime_and_vault()
    assert runtime.eager_model is not None

    l_idx = max(0, min(int(layer), N_LAYERS - 1))
    n_idx = max(0, min(int(neuron_idx), D_FF - 1))
    addr = Address("mlp_act", l_idx)

    async def _eval_prompts(prompts: List[str]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for p_text in prompts:
            tok_t = runtime.eager_model.to_tokens(p_text, prepend_bos=True)
            p_ids = [int(t) for t in tok_t[0].tolist()]
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
        "verdict": verdict,
        "cohens_d": round(cohens_d, 4),
        "mean_pos_signed_act": round(mean_pos, 4),
        "mean_neg_signed_act": round(mean_neg, 4),
        "pos_prompt_results": pos_records,
        "neg_prompt_results": neg_records,
        "pole_j_vindex_signature": pole_info["active_pole"]["j_vindex_tokens"],
    }


@mcp.tool()
def query_larql_vault(lql_query: str) -> Dict[str, Any]:
    """Execute a LARQL / VINDEX3 query or declarative `.patch` weight surgery statement."""
    _rt, vault = get_runtime_and_vault()
    return vault.execute_lql(lql_query)


if __name__ == "__main__":
    mcp.run()
