"""Four mathematical modules for Logos Theodolite (Gemma-4-31B).

Implements in clean, vectorized PyTorch:
1. Attention Router vs. FFN Vault Causal Vector Split in J++-Space
2. Bipolar GeGLU Neuron Poles ("Already Clean" Native Basis) & J-VINDEX Fusion
3. Live Downstream Reader Circuit Filaments (Spillover Filter)
4. Projective Theodolite Triangulation (Modified Gram-Schmidt Survey Frame)
Plus Gemma 4 Exact Unembedding (RMSNorm + LM Head + 30.0 Softcapping) and 60x60 Linear CKA.

Based on (numbers refer to REFERENCES.md): Jacobian lens / J-space [2] and J++ lens [4] readouts; the logit lens [8];
neuron "poles" and the "already clean" framing (Cammarata 2026, Goodfire) [9]; the residual-stream decomposition and
weight-based (virtual-weight) neuron coupling of Elhage et al. 2021 [14]; GeGLU [15]; linear CKA (Kornblith et al. 2019) [13].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 5376
N_LAYERS = 60
D_FF = 21504
VOCAB_SIZE = 262144
FINAL_LOGIT_SOFTCAP = 30.0
RMS_NORM_EPS = 1e-6

# 1-based global full-attention layers in Gemma-4-31B: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
GLOBAL_ATTN_LAYERS_1BASED: Tuple[int, ...] = (6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
GLOBAL_ATTN_LAYERS_0BASED: Tuple[int, ...] = tuple(l - 1 for l in GLOBAL_ATTN_LAYERS_1BASED)


def validate_and_cast_lens(
    j_raw: torch.Tensor,
    layer_idx: int,
    target_dtype: torch.dtype = torch.bfloat16,
    device: Optional[ Union[str, torch.device] ] = None,
) -> torch.Tensor:
    """Validate a stored [5376, 5376] fp16 Jacobian / J++ lens matrix and cast to bfloat16/fp32.

    CRITICAL PRECISION FIX:
    The stored maps in `neuronpedia/jacobian-lens` are fp16 (~1.1e-3 relative error on
    Gemma 4 31B due to small layer values). On load, immediately validate for non-finite
    values and cast J[l] to bfloat16 (or fp32 for reductions).
    """
    if j_raw.shape != (D_MODEL, D_MODEL):
        raise ValueError(
            f"Layer {layer_idx} lens shape mismatch: expected ({D_MODEL}, {D_MODEL}), got {tuple(j_raw.shape)}"
        )
    finite_mask = torch.isfinite(j_raw)
    if not bool(finite_mask.all()):
        n_bad = int((~finite_mask).sum().item())
        raise ValueError(f"Layer {layer_idx} lens contains {n_bad} non-finite (NaN/Inf) entries!")
    out = j_raw.to(dtype=target_dtype)
    if device is not None:
        out = out.to(device=device, non_blocking=True)
    return out


def gemma4_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = RMS_NORM_EPS,
) -> torch.Tensor:
    """Apply Gemma 4 RMSNorm in fp32 and cast back to norm_weight's dtype.

    Note: In HuggingFace `transformers` (`Gemma4RMSNorm`), `norm.weight` already stores
    the `(1 + w)` unit-offset scale factor (initialized to ones).
    For small-magnitude direction vectors (such as single W_down columns projected through
    mid-layer J++ lenses where mean(x^2) ~ 1e-7 < 1e-6), uses 1e-12 floor so eps does not
    artificially damp the direction's logits.
    """
    x_fp32 = x.float()
    ms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    effective_eps = torch.where(ms < 1e-3, torch.full_like(ms, 1e-12), torch.full_like(ms, eps))
    inv_rms = torch.rsqrt(ms + effective_eps)
    normed = x_fp32 * inv_rms * norm_weight.float()
    return normed.to(dtype=norm_weight.dtype)


@torch.no_grad()
def unembed_gemma4(
    h: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    softcap: Optional[float] = FINAL_LOGIT_SOFTCAP,
    eps: float = RMS_NORM_EPS,
) -> torch.Tensor:
    """Neuronpedia & Gemma-4-31B exact unembed contract.

    Computes:
        u_norm = FinalRMSNorm(h)
        logits = u_norm @ W_U.T
        logits = 30.0 * tanh(logits / 30.0)
    """
    h_norm = gemma4_rms_norm(h, final_norm_weight, eps=eps)
    logits = F.linear(h_norm.to(dtype=lm_head_weight.dtype), lm_head_weight)
    if softcap is not None and softcap > 0.0:
        logits = torch.tanh(logits.float() / softcap) * softcap
    return logits


@torch.no_grad()
def jpp_readout_logits(
    h: torch.Tensor,
    J_l: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    softcap: Optional[float] = FINAL_LOGIT_SOFTCAP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the J++ Lens at layer l followed by Gemma 4 unembed:
        h_J = h @ J[l].T
        logits = unembed(h_J)
    Returns (logits, h_J).
    """
    h_J = h.to(dtype=J_l.dtype) @ J_l.T
    logits = unembed_gemma4(h_J, final_norm_weight, lm_head_weight, softcap=softcap)
    return logits, h_J


# ============================================================================
# MODULE 1: Attention Router vs. FFN Vault Causal Vector Split
# ============================================================================


@dataclass
class CausalSplitResult:
    """Decomposed Attention Router vs. FFN Vault vectors in J++-Space."""

    delta_attn_J: torch.Tensor       # [..., 5376] in J++-Space
    delta_ffn_J: torch.Tensor        # [..., 5376] in J++-Space
    resid_pre_J: torch.Tensor        # [..., 5376] in J++-Space
    resid_mid_J: torch.Tensor        # [..., 5376] in J++-Space
    resid_post_J: torch.Tensor       # [..., 5376] in J++-Space
    attn_norm: torch.Tensor          # [...] L2 norm of delta_attn_J
    ffn_norm: torch.Tensor           # [...] L2 norm of delta_ffn_J
    ffn_dominance_ratio: torch.Tensor  # [...] ||delta_ffn_J|| / (||delta_attn_J|| + ||delta_ffn_J||)
    # Gemma-4 layer_scalar shrinkage of the incoming residual, (s - 1) * resid_pre, in J++-Space.
    # delta_attn_J + delta_ffn_J + delta_rescale_J == (resid_post - resid_pre) @ J^T exactly.
    delta_rescale_J: Optional[torch.Tensor] = None
    rescale_norm: Optional[torch.Tensor] = None   # [...] L2 norm of delta_rescale_J
    layer_scalar: float = 1.0


@torch.no_grad()
def compute_causal_vector_split(
    resid_pre: torch.Tensor,
    resid_mid: torch.Tensor,
    resid_post: torch.Tensor,
    J_l: torch.Tensor,
    layer_scalar: Union[float, torch.Tensor] = 1.0,
) -> CausalSplitResult:
    """Module 1: Decompose the layer-l residual update in J++-Space into causal vectors.

    Gemma-4 decoder layers (transformers `Gemma4TextDecoderLayer.forward`) end with
    `hidden_states *= self.layer_scalar`, i.e.
        resid_mid  = resid_pre + post_attention_layernorm(attn(input_layernorm(resid_pre)))     (unscaled)
        resid_post = s * (resid_mid + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(resid_mid))))
    with s = layer_scalar in [0.03, 0.99] on Gemma-4-31B. The exact additive split of the update is
        resid_post - resid_pre = s*(mid - pre)        [attention write, as it lands in resid_post]
                               + (post - s*mid)       [FFN write = s * post_ffn_norm(mlp(...))]
                               + (s - 1)*pre          [layer_scalar rescale of the incoming residual]
    and each term is mapped to J++-Space with J_pp[l]:
        delta_attn_J    = s*(resid_mid - resid_pre) @ J_pp[l].T     (cyan trajectory)
        delta_ffn_J     = (resid_post - s*resid_mid) @ J_pp[l].T    (gold trajectory)
        delta_rescale_J = (s - 1)*resid_pre @ J_pp[l].T             (reported separately)
    The previous version (equivalent to s = 1) used delta_ffn = post - mid = s*ffn + (s-1)*mid, which folds the
    rescale of the whole incoming residual AND of the attention write into the "FFN vault" vector, and reported the
    attention write 1/s too large. Pass layer_scalar=1.0 only for architectures without a per-layer output scalar.
    """
    dtype = J_l.dtype
    JT = J_l.T

    pre = resid_pre.float()
    mid = resid_mid.float()
    post = resid_post.float()
    s = torch.as_tensor(layer_scalar, dtype=torch.float32, device=pre.device).reshape(-1)[0]

    delta_attn_raw = (s * (mid - pre)).to(dtype=dtype)
    delta_ffn_raw = (post - s * mid).to(dtype=dtype)
    delta_rescale_raw = ((s - 1.0) * pre).to(dtype=dtype)

    delta_attn_J = delta_attn_raw @ JT
    delta_ffn_J = delta_ffn_raw @ JT
    delta_rescale_J = delta_rescale_raw @ JT

    resid_pre_J = resid_pre.to(dtype=dtype) @ JT
    resid_mid_J = resid_mid.to(dtype=dtype) @ JT
    resid_post_J = resid_post.to(dtype=dtype) @ JT

    attn_norm = torch.linalg.vector_norm(delta_attn_J.float(), dim=-1)
    ffn_norm = torch.linalg.vector_norm(delta_ffn_J.float(), dim=-1)
    rescale_norm = torch.linalg.vector_norm(delta_rescale_J.float(), dim=-1)
    ffn_dominance_ratio = ffn_norm / (attn_norm + ffn_norm + 1e-8)

    return CausalSplitResult(
        delta_attn_J=delta_attn_J,
        delta_ffn_J=delta_ffn_J,
        resid_pre_J=resid_pre_J,
        resid_mid_J=resid_mid_J,
        resid_post_J=resid_post_J,
        attn_norm=attn_norm,
        ffn_norm=ffn_norm,
        ffn_dominance_ratio=ffn_dominance_ratio,
        delta_rescale_J=delta_rescale_J,
        rescale_norm=rescale_norm,
        layer_scalar=float(s),
    )


# ============================================================================
# MODULE 2: Bipolar GeGLU Neuron Poles ("Already Clean" Native Basis)
# ============================================================================


@dataclass
class BipolarPolesResult:
    """Top-k active Bipolar GeGLU poles at layer l with J-VINDEX vs. Naive readouts."""

    layer_idx: int
    neuron_indices: torch.Tensor     # [k] int64 in 0..21503
    abs_activations: torch.Tensor    # [k] float32 |mlp_act[l, idx]|
    signed_activations: torch.Tensor # [k] float32 mlp_act[l, idx]
    polarities: torch.Tensor         # [k] float32 in {-1.0, +1.0}
    signed_w_down: torch.Tensor      # [k, 5376] bfloat16/float32 polarity * W_down[l, i]
    jpp_pole_vectors: torch.Tensor   # [k, 5376] bfloat16/float32 (polarity * W_down[l, i]) @ J_pp[l].T
    pole_logits: Optional[torch.Tensor] = None       # [k, vocab_size] J-VINDEX logits
    naive_pole_logits: Optional[torch.Tensor] = None # [k, vocab_size] Naive Logit Lens logits
    # [k] ||W_down[l, i] (.) w_post||: the length of each pole's write column as the stream receives it
    write_norms: Optional[torch.Tensor] = None
    ranked_by: str = "activation"                    # "activation" (|a|) or "write" (|a| * ||w_down (.) w_post||)


POLE_RANKINGS: Tuple[str, ...] = ("activation", "write")


@torch.no_grad()
def extract_bipolar_poles(
    mlp_act_l: torch.Tensor,
    w_down_slice_fn,
    J_l: torch.Tensor,
    layer_idx: int,
    k: int = 15,
    final_norm_weight: Optional[torch.Tensor] = None,
    lm_head_weight: Optional[torch.Tensor] = None,
    post_ffn_norm_weight: Optional[torch.Tensor] = None,
    compute_logits: bool = True,
    compute_naive_logits: bool = False,
    write_col_norms: Optional[torch.Tensor] = None,
) -> BipolarPolesResult:
    """Module 2: Extract top-k Bipolar GeGLU neuron poles and project through J++ Lens.

    Executes:
        vals, idx = torch.topk(mlp_act[l].abs(), k=15, dim=-1)
        polarity = torch.sign(mlp_act[l].gather(-1, idx))
        pole_logits(l, i, polarity) = unembed((polarity * W_down[l, i]) @ J_pp[l].T)

    With `write_col_norms` (`[21504]`, `||W_down[l, i] (.) w_post||` for every neuron) the poles are ranked by
    write size instead, |a_i| * ||W_down[l, i] (.) w_post|| (REPORT §3.4 bug 8; the `wr` ranking of
    experiments/scripts/E3_lens_validation_07_toolbug.py). Within one layer this orders the neurons as their
    linearised writes s * |a_i| * ||W_down[l, i] (.) w_post|| / rms(y) do, since s and rms(y) are shared.

    Args:
        mlp_act_l: 1D tensor of shape [21504] for the target token position.
        w_down_slice_fn: Callable `fn(layer_idx, neuron_indices) -> Tensor[k, 5376]` returning
            the down-projection vectors `W_down[l, idx]` in [k, 5376].
        J_l: Layer l J++ matrix of shape [5376, 5376].
        layer_idx: 0-based layer index in 0..59.
        k: Number of top active poles to extract (default 15).
    """
    if mlp_act_l.dim() != 1 or mlp_act_l.shape[0] != D_FF:
        raise ValueError(f"Expected 1D mlp_act_l of shape ({D_FF},), got {tuple(mlp_act_l.shape)}")

    if write_col_norms is not None:
        score = mlp_act_l.float().abs() * write_col_norms.to(device=mlp_act_l.device, dtype=torch.float32)
        _, idx = torch.topk(score, k=k, dim=-1)
        vals = mlp_act_l.abs().gather(-1, idx)
    else:
        vals, idx = torch.topk(mlp_act_l.abs(), k=k, dim=-1)
    signed_act = mlp_act_l.gather(-1, idx)
    polarity = torch.sign(signed_act)
    # Guard against exact zero activation having sign 0
    polarity = torch.where(polarity == 0, torch.ones_like(polarity), polarity)

    # Fetch only the k active columns of W_down[l] -> shape [k, 5376]
    w_down_k = w_down_slice_fn(layer_idx, idx).to(device=J_l.device, dtype=J_l.dtype)
    if post_ffn_norm_weight is not None:
        w_down_k = (w_down_k.float() * post_ffn_norm_weight.to(J_l.device).float()).to(dtype=J_l.dtype)
    write_norms = torch.linalg.vector_norm(w_down_k.float(), dim=-1)

    signed_w_down = polarity.to(dtype=J_l.dtype).unsqueeze(-1) * w_down_k  # [k, 5376]
    jpp_pole_vectors = signed_w_down @ J_l.T  # [k, 5376]

    pole_logits = None
    naive_pole_logits = None
    if compute_logits and final_norm_weight is not None and lm_head_weight is not None:
        pole_logits = unembed_gemma4(jpp_pole_vectors, final_norm_weight, lm_head_weight)
    if compute_naive_logits and final_norm_weight is not None and lm_head_weight is not None:
        naive_pole_logits = unembed_gemma4(signed_w_down, final_norm_weight, lm_head_weight)

    return BipolarPolesResult(
        layer_idx=layer_idx,
        neuron_indices=idx,
        abs_activations=vals.float(),
        signed_activations=signed_act.float(),
        polarities=polarity.float(),
        signed_w_down=signed_w_down,
        jpp_pole_vectors=jpp_pole_vectors,
        pole_logits=pole_logits,
        naive_pole_logits=naive_pole_logits,
        write_norms=write_norms,
        ranked_by="write" if write_col_norms is not None else "activation",
    )


@torch.no_grad()
def pole_write_shares(
    poles: BipolarPolesResult,
    mlp_out_t: torch.Tensor,
    ffn_write_t: torch.Tensor,
    layer_scalar: Union[float, torch.Tensor],
    eps: float = RMS_NORM_EPS,
) -> Tuple[torch.Tensor, float]:
    """Each listed pole's write size, and the share of the layer's feed-forward write they explain.

    The FFN write that lands in resid_post is f = s * w_post (.) y / rms(y) with y = W_down a (`mlp_out`).
    Neuron i contributes c_i = s * a_i * (W_down[l, i] (.) w_post) / rms(y), of length
        write_size_i = s * |a_i| * ||W_down[l, i] (.) w_post|| / rms(y),
    and the listed poles explain <sum_i c_i, f> / <f, f> of the write (`frac_ffn_top15_*` in
    experiments/scripts/E3_lens_validation_07_toolbug.py, which found 10-41% for the top 15 by |a| at L20-58).

    Args:
        poles: the layer's poles (`signed_w_down` already carries w_post).
        mlp_out_t: `[5376]` y at the target position (the MLP output before the post-FFN norm).
        ffn_write_t: `[5376]` f = resid_post - s * resid_mid at the target position.
    Returns:
        (`write_sizes` `[k]` float32, `explained_frac` float)
    """
    s = float(torch.as_tensor(layer_scalar).float().reshape(-1)[0])
    y = mlp_out_t.float()
    inv_rms = float(torch.rsqrt(y.pow(2).mean() + eps))
    a_abs = poles.abs_activations.float().to(y.device)
    norms = (
        poles.write_norms.float().to(y.device)
        if poles.write_norms is not None
        else torch.linalg.vector_norm(poles.signed_w_down.float(), dim=-1).to(y.device)
    )
    write_sizes = s * a_abs * norms * inv_rms
    # sigma_i * |a_i| = a_i, and signed_w_down = sigma_i * (W_down[l, i] (.) w_post)
    contrib = (s * inv_rms) * (a_abs.unsqueeze(-1) * poles.signed_w_down.float().to(y.device)).sum(0)
    f = ffn_write_t.float().to(y.device)
    explained = float((contrib @ f) / ((f @ f) + 1e-12))
    return write_sizes, explained


# ============================================================================
# MODULE 3: Live Downstream Reader Circuit Filaments (Spillover Filter)
# ============================================================================


@dataclass
class CircuitFilament:
    """Verified live functional circuit edge between a writer pole and downstream reader pole."""

    src_layer: int
    src_rank: int
    src_neuron: int
    src_polarity: float
    dst_layer: int
    dst_rank: int
    dst_neuron: int
    dst_polarity: float
    coupling_score: float
    dot_product: float
    dst_abs_act: float
    # Direct-path estimates (None when the clean forward's y / mid were not supplied): what the source adds,
    # through the residual stream alone, to the reader's gate pre-activation g_j and to its activation a_j.
    direct_gate: Optional[float] = None
    direct_act: Optional[float] = None
    # The score the filament was selected by: coupling_score ("coupling") or direct_gate ("direct").
    score: Optional[float] = None
    score_kind: str = "coupling"


THREAD_SCORES: Tuple[str, ...] = ("coupling", "direct")


@dataclass
class DirectPathInputs:
    """What the direct-path thread score needs from the clean forward at the target position."""

    w_down_slice_fn: object                  # fn(l, idx) -> raw W_down[l, idx]  [k, 5376]
    w_up_slice_fn: object                    # fn(l, idx) -> raw W_up[l, idx]    [k, 5376]
    mlp_out: Dict[int, torch.Tensor]         # l -> y_l = W_down a_l, the MLP output before the post-FFN norm  [5376]
    resid_mid: Dict[int, torch.Tensor]       # l -> mid_l, the input of the pre-FFN norm  [5376]
    layer_scalars: torch.Tensor              # [60] Gemma-4 layer_scalar s_l
    post_ffn_norm_weights: Dict[int, torch.Tensor]
    pre_ffn_norm_weights: Dict[int, torch.Tensor]


def _rms_jacobian_unweighted(x: torch.Tensor, v: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """d(x / rms(x)) applied to the rows of `v` (`[n, D]`), with rms(x) = sqrt(mean(x^2) + eps).

    The exact Jacobian `(I - x x^T / (D rms^2)) / rms`; multiply by the norm weight afterwards.
    """
    x = x.float()
    r2 = x.pow(2).mean() + eps
    proj = (v @ x) / (x.numel() * r2)  # [n]
    return (v - proj.unsqueeze(-1) * x.unsqueeze(0)) * torch.rsqrt(r2)


@torch.no_grad()
def direct_path_source_write(
    signed_act: torch.Tensor,
    w_down_raw: torch.Tensor,
    mlp_out: torch.Tensor,
    post_ffn_norm_weight: torch.Tensor,
    layer_scalar: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Linearised change of resid_post_l due to each source neuron: s_l * J_postnorm(y_l) (a_i W_down[l, i]).

    `signed_act` `[k]`, `w_down_raw` `[k, 5376]` (no norm weight); returns `[k, 5376]` float32. The negative of
    this is the first-order effect of zero-ablating the neuron (V7_tool_bugs_03_filaments.py, `dpost`).
    """
    s = torch.as_tensor(layer_scalar, dtype=torch.float32).reshape(-1)[0].to(mlp_out.device)
    v = signed_act.float().to(mlp_out.device).unsqueeze(-1) * w_down_raw.float().to(mlp_out.device)
    w = post_ffn_norm_weight.float().to(mlp_out.device)
    return s * _rms_jacobian_unweighted(mlp_out, v) * w.unsqueeze(0)


@torch.no_grad()
def direct_path_reader_effects(
    d_mid: torch.Tensor,
    resid_mid: torch.Tensor,
    w_gate_pre: torch.Tensor,
    w_up_pre: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The direct-path effect of resid_mid changes `d_mid` (`[k, 5376]`) on readers at layer l'.

    `w_gate_pre` / `w_up_pre` are the readers' W_gate / W_up rows times the pre-FFN norm weight (`[m, 5376]`).
    Through the exact pre-FFN RMSNorm Jacobian at mid_l':
        D_gate = J_prenorm(mid_l') d_mid . W_gate_j                       (change of the gate pre-activation g_j)
        D_act  = gelu(g_j) u_j - gelu(g_j - D_gate)(u_j - D_up)          (change of a_j the source accounts for)
    with g_j, u_j the clean pre-activations recomputed from mid_l'. Returns (`D_gate`, `D_act`), each `[k, m]`.
    Sign convention: what the source ADDS (the ablation effect of V7_tool_bugs_03_filaments.py with the sign
    flipped); V7 found rho(|D_gate|, |measured dg|) = 0.87 and rho(|D_act|, |measured da|) = 0.84.
    """
    mid = resid_mid.float()
    wg = w_gate_pre.float().to(mid.device)
    wu = w_up_pre.float().to(mid.device)
    dxh = _rms_jacobian_unweighted(mid, d_mid.float().to(mid.device))  # [k, D]; the weight is in wg / wu
    d_gate = dxh @ wg.T
    d_up = dxh @ wu.T
    xh = mid * torch.rsqrt(mid.pow(2).mean() + RMS_NORM_EPS)
    g0 = (xh @ wg.T).unsqueeze(0)
    u0 = (xh @ wu.T).unsqueeze(0)
    d_act = F.gelu(g0, approximate="tanh") * u0 - F.gelu(g0 - d_gate, approximate="tanh") * (u0 - d_up)
    return d_gate, d_act


@torch.no_grad()
def compute_downstream_filaments(
    poles_by_layer: Dict[int, BipolarPolesResult],
    w_gate_slice_fn,
    max_layer_hop: int = 6,
    top_k_per_source_layer: int = 8,
    min_abs_coupling: float = 0.0,
    pre_ffn_norm_weights: Optional[Dict[int, torch.Tensor]] = None,
    score: str = "coupling",
    direct: Optional[DirectPathInputs] = None,
) -> List[CircuitFilament]:
    """Module 3: Compute Live Downstream Reader coupling scores across layers:
        C_{(l, i) -> (l', j)} = dot(sigma_i * W_down[l, i], W_gate[l', j]) * abs(mlp_act[l', j])

    Distinguishes true functional circuit edges (where a downstream neuron's gate actually
    reads the upstream pole's write vector AND fires on the current prompt) from unread
    orthogonal spillover.

    With `direct` supplied, every candidate also gets the direct-path linear estimates `direct_gate` and
    `direct_act` (see `direct_path_reader_effects`): they use the source's real activation, the layer scalars
    s_l * prod_{l<k<l'} s_k and both exact RMSNorm Jacobians, and V7 measured rho = 0.87 against ablation where C
    reaches 0.41 (REPORT §3.4 bug 4). `score="direct"` selects and ranks the filaments by |direct_gate| (with
    `min_abs_coupling` then applied to it); `score="coupling"` keeps the original ranking by |C|.
    """
    if score not in THREAD_SCORES:
        raise ValueError(f"Unknown thread score {score!r}; expected one of {THREAD_SCORES}")
    if score == "direct" and direct is None:
        raise ValueError("score='direct' needs the clean forward's DirectPathInputs")
    sorted_layers = sorted(poles_by_layer.keys())
    if not sorted_layers:
        return []

    # Pre-fetch W_gate rows [k, 5376] for every layer's active poles
    gate_cache: Dict[int, torch.Tensor] = {}
    up_cache: Dict[int, torch.Tensor] = {}
    for l in sorted_layers:
        p = poles_by_layer[l]
        w_gate = w_gate_slice_fn(l, p.neuron_indices).float()
        if pre_ffn_norm_weights is not None and l in pre_ffn_norm_weights:
            w_gate = w_gate * pre_ffn_norm_weights[l].to(w_gate.device).float()
        gate_cache[l] = w_gate
        if direct is not None:
            w_up = direct.w_up_slice_fn(l, p.neuron_indices).float()
            if l in direct.pre_ffn_norm_weights:
                w_up = w_up * direct.pre_ffn_norm_weights[l].to(w_up.device).float()
            up_cache[l] = w_up

    filaments: List[CircuitFilament] = []

    for l in sorted_layers:
        src_poles = poles_by_layer[l]
        # signed_w_down: [k_src, 5376] = sigma_i * W_down[l, i]
        src_writers = src_poles.signed_w_down.float()
        layer_candidates: List[CircuitFilament] = []

        d_post = None
        if direct is not None and l in direct.mlp_out and l in direct.post_ffn_norm_weights:
            d_post = direct_path_source_write(
                src_poles.signed_activations,
                direct.w_down_slice_fn(l, src_poles.neuron_indices),
                direct.mlp_out[l].to(src_writers.device),
                direct.post_ffn_norm_weights[l],
                direct.layer_scalars[l],
            )

        for l_dst in range(l + 1, min(l + max_layer_hop + 1, N_LAYERS)):
            if l_dst not in poles_by_layer:
                continue
            dst_poles = poles_by_layer[l_dst]
            dst_gates = gate_cache[l_dst].to(src_writers.device)  # [k_dst, 5376]
            dst_abs_act = dst_poles.abs_activations.to(src_writers.device).float()  # [k_dst]

            # Pairwise dot products: [k_src, k_dst]
            dots = src_writers @ dst_gates.T
            # Coupling score: C_{(l, i) -> (l', j)} = dot(sigma_i * W_down[l, i], W_gate[l', j]) * |mlp_act[l', j]|
            couplings = dots * dst_abs_act.unsqueeze(0)

            d_gate = d_act = None
            if d_post is not None and l_dst in direct.resid_mid:
                ls = direct.layer_scalars.float()
                carry = torch.prod(ls[l + 1 : l_dst]) if l_dst > l + 1 else torch.tensor(1.0)
                d_gate, d_act = direct_path_reader_effects(
                    carry.to(d_post.device) * d_post,
                    direct.resid_mid[l_dst].to(src_writers.device),
                    dst_gates,
                    up_cache[l_dst].to(src_writers.device),
                )
            rank_mat = d_gate if score == "direct" else couplings

            # Take top couplings from this (l, l_dst) pair
            flat_abs = rank_mat.abs().reshape(-1)
            n_take = min(top_k_per_source_layer, flat_abs.numel())
            top_vals, top_flat_idx = torch.topk(flat_abs, k=n_take)

            k_dst = dst_gates.shape[0]
            src_idx_cpu = src_poles.neuron_indices.cpu().tolist()
            src_pol_cpu = src_poles.polarities.cpu().tolist()
            dst_idx_cpu = dst_poles.neuron_indices.cpu().tolist()
            dst_pol_cpu = dst_poles.polarities.cpu().tolist()
            dst_act_cpu = dst_abs_act.cpu().tolist()

            for rank_idx in range(n_take):
                abs_c = float(top_vals[rank_idx].item())
                if abs_c <= min_abs_coupling:
                    continue
                flat_i = int(top_flat_idx[rank_idx].item())
                s_rank = flat_i // k_dst
                d_rank = flat_i % k_dst
                c_val = float(couplings[s_rank, d_rank].item())
                d_val = float(dots[s_rank, d_rank].item())
                dg_val = float(d_gate[s_rank, d_rank].item()) if d_gate is not None else None
                da_val = float(d_act[s_rank, d_rank].item()) if d_act is not None else None
                layer_candidates.append(
                    CircuitFilament(
                        src_layer=l,
                        src_rank=s_rank,
                        src_neuron=int(src_idx_cpu[s_rank]),
                        src_polarity=float(src_pol_cpu[s_rank]),
                        dst_layer=l_dst,
                        dst_rank=d_rank,
                        dst_neuron=int(dst_idx_cpu[d_rank]),
                        dst_polarity=float(dst_pol_cpu[d_rank]),
                        coupling_score=c_val,
                        dot_product=d_val,
                        dst_abs_act=float(dst_act_cpu[d_rank]),
                        direct_gate=dg_val,
                        direct_act=da_val,
                        score=dg_val if score == "direct" else c_val,
                        score_kind=score,
                    )
                )

        # Retain top_k_per_source_layer strongest filaments originating from layer l
        layer_candidates.sort(key=lambda f: abs(f.score if f.score is not None else f.coupling_score), reverse=True)
        filaments.extend(layer_candidates[:top_k_per_source_layer])

    return filaments


# ============================================================================
# MODULE 4: Projective Theodolite Triangulation (Gram-Schmidt Survey Frame)
# ============================================================================


@torch.no_grad()
def gram_schmidt_orthonormalize_3d(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Modified Gram-Schmidt orthonormalization in fp32 for 3 vectors of shape [..., 5376].

    Supports both single 3D frames (`v_i` of shape `[5376]`) and batched per-layer 3D frames
    (`v_i` of shape `[L, 5376]`).
    Returns orthonormal basis `E` of shape `[..., 3, 5376]` satisfying `E @ E.T == I_3`.
    """
    u1 = v1.float()
    e1 = u1 / (torch.linalg.vector_norm(u1, dim=-1, keepdim=True) + eps)

    u2 = v2.float()
    u2 = u2 - (u2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.linalg.vector_norm(u2, dim=-1, keepdim=True) + eps)

    u3 = v3.float()
    u3 = u3 - (u3 * e1).sum(dim=-1, keepdim=True) * e1
    u3 = u3 - (u3 * e2).sum(dim=-1, keepdim=True) * e2
    e3 = u3 / (torch.linalg.vector_norm(u3, dim=-1, keepdim=True) + eps)

    return torch.stack([e1, e2, e3], dim=-2)


@torch.no_grad()
def lowdin_orthonormalize_3d(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric (Löwdin 1950) orthonormalization in fp32 of 3 vectors of shape [..., 5376].

    Each row is first scaled to unit length (so every word weighs the same), then
        E = (V V^T)^(-1/2) V,
    the orthonormal frame closest to V in the Frobenius norm. Unlike Gram-Schmidt it treats the three
    vectors alike, so the frame is the same set of axes for every order of the inputs (permuting the
    inputs permutes the rows of E and nothing else). This is the fix V7 proposed for bug 3 and checked
    to 1.2e-7 (`loewdin` in experiments/scripts/V7_tool_bugs_01_bos_raw.py, which also unit-scales the rows).
    Returns `E` of shape `[..., 3, 5376]` satisfying `E @ E.T == I_3`.
    """
    V = torch.stack([v1.float(), v2.float(), v3.float()], dim=-2)  # [..., 3, D]
    V = V / (torch.linalg.vector_norm(V, dim=-1, keepdim=True) + eps)
    G = V @ V.transpose(-1, -2)  # [..., 3, 3]
    ev, U = torch.linalg.eigh(G.double())
    inv_sqrt = (U * ev.clamp_min(1e-12).rsqrt().unsqueeze(-2)) @ U.transpose(-1, -2)
    return (inv_sqrt.to(V.dtype) @ V).contiguous()


FRAME_SQUARINGS: Tuple[str, ...] = ("gram_schmidt", "lowdin")


def orthonormalize_frame_3d(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    squaring: str = "gram_schmidt",
) -> torch.Tensor:
    """Square three axes into an orthonormal frame: `"gram_schmidt"` (in order, as the instrument was
    built) or `"lowdin"` (all at once, order-free)."""
    if squaring == "gram_schmidt":
        return gram_schmidt_orthonormalize_3d(v1, v2, v3)
    if squaring == "lowdin":
        return lowdin_orthonormalize_3d(v1, v2, v3)
    raise ValueError(f"Unknown frame squaring {squaring!r}; expected one of {FRAME_SQUARINGS}")


@torch.no_grad()
def build_monument_frame(
    token_ids: Tuple[int, int, int],
    lm_head_weight: torch.Tensor,
    jpp_lenses: Optional[Dict[int, torch.Tensor]] = None,
    pullback_per_layer: bool = True,
    squaring: str = "gram_schmidt",
    final_norm_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Monument Mode Survey Frame:
    User picks 3 vocabulary tokens/concepts (t_1, t_2, t_3); pull their unembedding vectors
    back to layer l via J_pp[l] (`v_m[l] = W_U[t_m] @ J_pp[l]`) and orthonormalize via
    Gram-Schmidt (or Löwdin with `squaring="lowdin"`).

    With `final_norm_weight` given, the rows are the effective directions the model reads just before
    speaking, `fnw * W_U[t_m]` (logit_t = <h / rms(h), fnw * W_U[t]>), instead of the raw rows.

    Returns:
        - `global_basis_J`: `[3, 5376]` orthonormal basis in J++-Space (`W_U[t_m]` orthonormalized)
        - `layer_pullback_basis`: `[60, 3, 5376]` per-layer pulled-back orthonormal basis, in the raw
          residual coordinates of layer l (`<h_l, u J_l> = <h_l J_l^T, u>`), or the global basis expanded
          if `pullback_per_layer=False` or `jpp_lenses` is None.
    """
    t1, t2, t3 = token_ids
    u1 = lm_head_weight[t1].float()
    u2 = lm_head_weight[t2].float()
    u3 = lm_head_weight[t3].float()
    if final_norm_weight is not None:
        w = final_norm_weight.to(device=u1.device, dtype=torch.float32)
        u1, u2, u3 = u1 * w, u2 * w, u3 * w

    global_basis_J = orthonormalize_frame_3d(u1, u2, u3, squaring)  # [3, 5376]

    if not pullback_per_layer or jpp_lenses is None:
        return global_basis_J, global_basis_J.unsqueeze(0).expand(N_LAYERS, 3, D_MODEL).contiguous()

    layer_bases: List[torch.Tensor] = []
    for l in range(N_LAYERS):
        J_l = jpp_lenses[l].float()
        # Pull unembedding vector back to layer l via J_pp[l]: v_m[l] = u_m @ J_pp[l]
        v1_l = u1.to(J_l.device) @ J_l
        v2_l = u2.to(J_l.device) @ J_l
        v3_l = u3.to(J_l.device) @ J_l
        e_l = orthonormalize_frame_3d(v1_l, v2_l, v3_l, squaring)  # [3, 5376]
        layer_bases.append(e_l)
        del J_l

    return global_basis_J, torch.stack(layer_bases, dim=0)  # [60, 3, 5376]


@torch.no_grad()
def frame_word_geometry(
    rows: torch.Tensor,
    final_norm_weight: torch.Tensor,
) -> Dict[str, object]:
    """Angles and squareness of the three landmark rows, raw and final-norm weighted.

    The definitions are those of `frame_stats` in experiments/scripts/V7_tool_bugs_05_m1patch_frames.py:
    rows scaled to unit length; `cond` = sigma_max / sigma_min of the three unit rows; `cos` the pairwise
    cosines (keys "01", "02", "12"); `self_cos[m]` = cos(W_U[t_m], fnw * W_U[t_m]).
    """
    W = rows.float()
    w = final_norm_weight.to(device=W.device, dtype=torch.float32)
    out: Dict[str, object] = {}
    for tag, M in (("raw", W), ("fnw", W * w)):
        Mn = M / torch.linalg.vector_norm(M, dim=-1, keepdim=True)
        sv = torch.linalg.svdvals(Mn)
        G = Mn @ Mn.T
        out[tag] = {
            "cond": float(sv[0] / sv[-1]),
            "cos": {"01": float(G[0, 1]), "02": float(G[0, 2]), "12": float(G[1, 2])},
        }
    out["self_cos"] = [float(F.cosine_similarity(W[i], W[i] * w, dim=0)) for i in range(W.shape[0])]
    return out


@torch.no_grad()
def frame_condition_numbers(vectors: torch.Tensor) -> torch.Tensor:
    """sigma_max / sigma_min of unit-scaled frame vectors, `[..., 3, D]` -> `[...]`."""
    Vn = vectors.float() / torch.linalg.vector_norm(vectors.float(), dim=-1, keepdim=True)
    sv = torch.linalg.svdvals(Vn)
    return sv[..., 0] / sv[..., -1]


# ============================================================================
# One word against another: contrast readings (V7 bug 3's "contrastive axes", E3 (c)'s background)
# ============================================================================

# What a contrast may be read against: "controls", the matched controls sent with the sweep (a difference of two
# log-odds, E3 recommendation 1); "study_mean", E3's average state subtracted before the lens, as E3 centred it.
CONTRAST_BACKGROUNDS: Tuple[str, ...] = ("controls", "study_mean")


def _contrast_rows(
    toward_ids: Sequence[int], away_ids: Sequence[int], lm_head_weight: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """`[2n, D]` float32: the unembedding rows of the n toward words, then the n away words."""
    if not toward_ids or len(toward_ids) != len(away_ids):
        raise ValueError(
            f"A contrast reads its words in pairs: {len(toward_ids)} toward and {len(away_ids)} away will not pair."
        )
    ids = torch.tensor([int(t) for t in toward_ids] + [int(t) for t in away_ids], device=lm_head_weight.device)
    return lm_head_weight.index_select(0, ids).to(device=device, dtype=torch.float32)


@torch.no_grad()
def contrast_of_state(
    state: torch.Tensor,
    toward_ids: Sequence[int],
    away_ids: Sequence[int],
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    softcap: Optional[float] = FINAL_LOGIT_SOFTCAP,
) -> torch.Tensor:
    """The mean over the pairs of the capped logit of toward[i] less that of away[i], for final-space state(s)
    `[..., D]` (h_l J_l^T through the lens, h_l itself by the naked eye): `[...]` float32.

    Computed in fp32 on the pair rows only, with the unembed's own norm (the 1e-12 floor below mean x^2 = 1e-3, as
    `gemma4_rms_norm`) and softcap, so for one pair it is the log-odds of toward against away under that
    distribution, and for the six pronoun pairs it is labkit's `gender_scores(...)["pron"]` (E3, V7)."""
    rows = _contrast_rows(toward_ids, away_ids, lm_head_weight, state.device)
    x = state.float()
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    eps = torch.where(ms < 1e-3, torch.full_like(ms, 1e-12), torch.full_like(ms, RMS_NORM_EPS))
    z = (x * torch.rsqrt(ms + eps) * final_norm_weight.to(device=x.device, dtype=torch.float32)) @ rows.T
    if softcap is not None and softcap > 0.0:
        z = torch.tanh(z / softcap) * softcap
    n = len(toward_ids)
    return (z[..., :n] - z[..., n:]).mean(dim=-1)


@torch.no_grad()
def contrast_direction(
    toward_ids: Sequence[int],
    away_ids: Sequence[int],
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """`[D]` float32: d = fnw * mean_i (W_U[toward_i] - W_U[away_i]), the direction a contrast reads before the
    softcap (labkit's `gender_direction`)."""
    rows = _contrast_rows(toward_ids, away_ids, lm_head_weight, lm_head_weight.device)
    n = len(toward_ids)
    return (rows[:n] - rows[n:]).mean(dim=0) * final_norm_weight.to(device=rows.device, dtype=torch.float32)


@torch.no_grad()
def contrast_mean_part(
    mean_state: torch.Tensor,
    state: torch.Tensor,
    toward_ids: Sequence[int],
    away_ids: Sequence[int],
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """The average state's share of a state's contrast, linear and softcap-free: <mu J, d> / rms(h J), with
    `mean_state` = mu J and `state` = h J (or mu and h by the naked eye) `[..., D]`, and d as `contrast_direction`
    (E3_lens_validation_09_background.py). With the deviation's share <(h - mu) J, d> / rms(h J) it sums to the raw
    state's linear score <h J, d> / rms(h J). Not a log-odds the model gives."""
    d = contrast_direction(toward_ids, away_ids, final_norm_weight, lm_head_weight).to(state.device)
    rms = state.float().pow(2).mean(dim=-1).sqrt()
    return (mean_state.float().to(state.device) @ d) / rms


@torch.no_grad()
def build_pole_motif_frame(
    layer_idx: int,
    neuron_idx: int,
    polarity: float,
    w_gate_vec: torch.Tensor,
    w_down_vec: torch.Tensor,
    J_l: torch.Tensor,
    downstream_gate_vec: torch.Tensor,
    squaring: str = "gram_schmidt",
) -> torch.Tensor:
    """Pole-Locked Motif Mode Survey Frame:
    User clicks a firing neuron pole (l, i, sigma_i):
      - Axis X = upstream gate W_gate[l, i]
      - Axis Y = J-corrected write (sigma_i * W_down[l, i]) @ J_pp[l].T
      - Axis Z = dominant downstream reader gate W_gate[l+1, j*]
    Orthonormalized in fp32 via Gram-Schmidt (or Löwdin with `squaring="lowdin"`).
    Returns `basis` of shape `[3, 5376]`.
    """
    dev = J_l.device
    v_x = w_gate_vec.to(device=dev, dtype=torch.float32)
    signed_down = float(polarity) * w_down_vec.to(device=dev, dtype=J_l.dtype)
    v_y = (signed_down @ J_l.T).float()
    v_z = downstream_gate_vec.to(device=dev, dtype=torch.float32)
    return orthonormalize_frame_3d(v_x, v_y, v_z, squaring)


@dataclass
class TheodoliteTrajectory3D:
    """3D Projective Theodolite coordinates across all 60 layers."""

    xyz: torch.Tensor              # [60, 3] (x_l, y_l, z_l)
    azimuth_rad: torch.Tensor      # [60] theta = atan2(y_l, x_l) in [-pi, pi]
    elevation_rad: torch.Tensor    # [60] phi = asin(z_l / ||proj||) in [-pi/2, pi/2]
    azimuth_deg: torch.Tensor      # [60] theta in degrees [-180, 180]
    elevation_deg: torch.Tensor    # [60] phi in degrees [-90, 90]
    residual_norm: torch.Tensor    # [60] ||h_l||_2 full 5376-D L2 norm
    projection_norm: torch.Tensor  # [60] ||(x_l, y_l, z_l)||_2 in 3D subspace
    delta_attn_xyz: torch.Tensor   # [60, 3] Attention router delta in 3D frame
    delta_ffn_xyz: torch.Tensor    # [60, 3] FFN vault delta in 3D frame
    # [60] ||(x, y, z)|| / ||state||: the share of the projected state the frame catches (state = h_l J_l^T,
    # or the raw h_l for a per-layer pulled-back frame; less the centre when one is subtracted)
    proj_frac: Optional[torch.Tensor] = None


@torch.no_grad()
def project_trajectory_to_3d(
    resid_post_layers: torch.Tensor,
    resid_post_J_layers: torch.Tensor,
    delta_attn_J_layers: torch.Tensor,
    delta_ffn_J_layers: torch.Tensor,
    basis: torch.Tensor,
    use_jpp_states: bool = True,
    eps: float = 1e-8,
    centre_states: Optional[torch.Tensor] = None,
    delta_attn_raw_layers: Optional[torch.Tensor] = None,
    delta_ffn_raw_layers: Optional[torch.Tensor] = None,
) -> TheodoliteTrajectory3D:
    """Module 4: For every layer l in 0..59, compute exact 3D coordinates (x_l, y_l, z_l),
    Azimuth theta = atan2(y_l, x_l), Elevation phi = asin(z_l / ||proj||), and residual
    norm ||h_l||_2.

    Args:
        resid_post_layers: `[60, 5376]` raw residual stream exiting each layer l.
        resid_post_J_layers: `[60, 5376]` J++-aligned residual stream `h_l @ J_pp[l].T`.
        delta_attn_J_layers: `[60, 5376]` Attention router delta in J++-Space.
        delta_ffn_J_layers: `[60, 5376]` FFN vault delta in J++-Space.
        basis: Either `[3, 5376]` global orthonormal basis OR `[60, 3, 5376]` per-layer
            pulled-back orthonormal basis.
        use_jpp_states: If True (or if basis is `[3, 5376]`), project `resid_post_J_layers`
            onto `basis`. If False and `basis` is `[60, 3, 5376]`, project raw `resid_post_layers`
            onto the per-layer pulled-back basis.
        centre_states: optional `[60, 5376]` centre (e.g. a matched control's state), in the same space as
            the projected states, subtracted from them before projection (REPORT §3.5). The deltas are
            differences and are not centred.
        delta_attn_raw_layers / delta_ffn_raw_layers: optional `[60, 5376]` raw-space deltas, projected instead
            of the J++-space ones when the raw states are (a per-layer pulled-back frame lives in raw space).
    """
    dev = resid_post_layers.device
    h_raw = resid_post_layers.float()
    h_J = resid_post_J_layers.float()
    d_attn = delta_attn_J_layers.float()
    d_ffn = delta_ffn_J_layers.float()
    B = basis.to(device=dev, dtype=torch.float32)

    target_states = h_J if use_jpp_states else h_raw
    if not use_jpp_states and delta_attn_raw_layers is not None and delta_ffn_raw_layers is not None:
        d_attn = delta_attn_raw_layers.float()
        d_ffn = delta_ffn_raw_layers.float()
    if centre_states is not None:
        target_states = target_states - centre_states.to(device=dev, dtype=torch.float32)

    if B.dim() == 2:
        # B is [3, 5376]
        xyz = target_states @ B.T           # [60, 3]
        delta_attn_xyz = d_attn @ B.T       # [60, 3]
        delta_ffn_xyz = d_ffn @ B.T         # [60, 3]
    elif B.dim() == 3:
        # B is [60, 3, 5376]
        xyz = torch.einsum("ld,lkd->lk", target_states, B)
        delta_attn_xyz = torch.einsum("ld,lkd->lk", d_attn, B)
        delta_ffn_xyz = torch.einsum("ld,lkd->lk", d_ffn, B)
    else:
        raise ValueError(f"Invalid basis shape: {tuple(B.shape)}")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    proj_norm = torch.linalg.vector_norm(xyz, dim=-1)  # [60]
    resid_norm = torch.linalg.vector_norm(h_raw, dim=-1)  # [60]
    proj_frac = proj_norm / (torch.linalg.vector_norm(target_states, dim=-1) + eps)  # [60]

    azimuth_rad = torch.atan2(y, x)
    sin_phi = torch.clamp(z / (proj_norm + eps), -1.0, 1.0)
    elevation_rad = torch.asin(sin_phi)

    rad2deg = 180.0 / torch.pi
    azimuth_deg = azimuth_rad * rad2deg
    elevation_deg = elevation_rad * rad2deg

    return TheodoliteTrajectory3D(
        xyz=xyz,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        residual_norm=resid_norm,
        projection_norm=proj_norm,
        delta_attn_xyz=delta_attn_xyz,
        delta_ffn_xyz=delta_ffn_xyz,
        proj_frac=proj_frac,
    )


@torch.no_grad()
def compute_linear_cka_matrix(
    layer_seq_states: torch.Tensor,
    eps: float = 1e-8,
    *,
    overwrite: bool = False,
) -> torch.Tensor:
    """Compute the [60, 60] Linear CKA similarity matrix across all 60 layers (Kornblith et al. 2019).

    Args:
        layer_seq_states: Tensor of shape `[60, n_tokens, 5376]` (or `[60, 5376]` if single token,
            in which case centered feature cosine similarity is returned).
        overwrite: a float32 `layer_seq_states` may be centred in place (the caller reads it no more), saving a
            copy of it. The Gram matrices are centred and scaled in place either way: the same element-wise
            operations in the same order as `G - row means - column means + mean`, so the result is the same,
            bit for bit, with less memory ([60, n, n] once, not three times).
    Returns:
        `cka`: Tensor of shape `[60, 60]` in float32 with values in `[0, 1]` and diagonal `1.0`.
    """
    X = layer_seq_states.float()
    if X.dim() == 2 or X.shape[1] < 2:
        # Single token position: [60, 5376] -> feature-centered cosine similarity squared
        V = X.view(X.shape[0], -1)
        V = V - V.mean(dim=-1, keepdim=True)
        V_norm = V / (torch.linalg.vector_norm(V, dim=-1, keepdim=True) + eps)
        sim = V_norm @ V_norm.T
        return torch.clamp(sim.pow(2), 0.0, 1.0)

    # X is [L, N, D] where L=60, N=n_tokens, D=5376
    # Center across token dimension N
    token_mean = X.mean(dim=1, keepdim=True)
    # in place when X is this function's own float32 copy, or the caller lets it overwrite theirs
    X_c = X.sub_(token_mean) if overwrite or X is not layer_seq_states else X - token_mean  # [L, N, D]
    # Gram matrices G_l = X_c[l] @ X_c[l].T of shape [L, N, N]
    G = torch.bmm(X_c, X_c.transpose(1, 2))  # [L, N, N]
    del X_c, X
    # Center Gram matrices: H G H (all three means of the uncentred G, then subtracted and added in that order)
    row_mean = G.mean(dim=-1, keepdim=True)
    col_mean = G.mean(dim=-2, keepdim=True)
    all_mean = G.mean(dim=(-2, -1), keepdim=True)
    G.sub_(row_mean).sub_(col_mean).add_(all_mean)
    flat_G = G.reshape(G.shape[0], -1)  # [L, N*N]
    norms = torch.linalg.vector_norm(flat_G, dim=-1, keepdim=True) + eps
    unit_G = flat_G.div_(norms)
    cka = unit_G @ unit_G.T
    return torch.clamp(cka, 0.0, 1.0)
