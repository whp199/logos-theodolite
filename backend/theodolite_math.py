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
from typing import Dict, List, Optional, Tuple, Union

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


@torch.no_grad()
def compute_causal_vector_split(
    resid_pre: torch.Tensor,
    resid_mid: torch.Tensor,
    resid_post: torch.Tensor,
    J_l: torch.Tensor,
) -> CausalSplitResult:
    """Module 1: Decompose the layer-l residual update in J++-Space into two causal vectors:
        delta_attn_J = (resid_mid[l] - resid_pre[l]) @ J_pp[l].T
        delta_ffn_J  = (resid_post[l] - resid_mid[l]) @ J_pp[l].T
    Isolates whether a concept emerging at layer l was routed from context by Attention
    (cyan trajectory) or written from parametric memory by the FFN vault (gold trajectory).
    """
    dtype = J_l.dtype
    JT = J_l.T

    delta_attn_raw = (resid_mid.float() - resid_pre.float()).to(dtype=dtype)
    delta_ffn_raw = (resid_post.float() - resid_mid.float()).to(dtype=dtype)

    delta_attn_J = delta_attn_raw @ JT
    delta_ffn_J = delta_ffn_raw @ JT

    resid_pre_J = resid_pre.to(dtype=dtype) @ JT
    resid_mid_J = resid_mid.to(dtype=dtype) @ JT
    resid_post_J = resid_post.to(dtype=dtype) @ JT

    attn_norm = torch.linalg.vector_norm(delta_attn_J.float(), dim=-1)
    ffn_norm = torch.linalg.vector_norm(delta_ffn_J.float(), dim=-1)
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
) -> BipolarPolesResult:
    """Module 2: Extract top-k Bipolar GeGLU neuron poles and project through J++ Lens.

    Executes:
        vals, idx = torch.topk(mlp_act[l].abs(), k=15, dim=-1)
        polarity = torch.sign(mlp_act[l].gather(-1, idx))
        pole_logits(l, i, polarity) = unembed((polarity * W_down[l, i]) @ J_pp[l].T)

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

    vals, idx = torch.topk(mlp_act_l.abs(), k=k, dim=-1)
    signed_act = mlp_act_l.gather(-1, idx)
    polarity = torch.sign(signed_act)
    # Guard against exact zero activation having sign 0
    polarity = torch.where(polarity == 0, torch.ones_like(polarity), polarity)

    # Fetch only the k active columns of W_down[l] -> shape [k, 5376]
    w_down_k = w_down_slice_fn(layer_idx, idx).to(device=J_l.device, dtype=J_l.dtype)
    if post_ffn_norm_weight is not None:
        w_down_k = (w_down_k.float() * post_ffn_norm_weight.to(J_l.device).float()).to(dtype=J_l.dtype)

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
    )


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


@torch.no_grad()
def compute_downstream_filaments(
    poles_by_layer: Dict[int, BipolarPolesResult],
    w_gate_slice_fn,
    max_layer_hop: int = 6,
    top_k_per_source_layer: int = 8,
    min_abs_coupling: float = 0.0,
    pre_ffn_norm_weights: Optional[Dict[int, torch.Tensor]] = None,
) -> List[CircuitFilament]:
    """Module 3: Compute Live Downstream Reader coupling scores across layers:
        C_{(l, i) -> (l', j)} = dot(sigma_i * W_down[l, i], W_gate[l', j]) * abs(mlp_act[l', j])

    Distinguishes true functional circuit edges (where a downstream neuron's gate actually
    reads the upstream pole's write vector AND fires on the current prompt) from unread
    orthogonal spillover.
    """
    sorted_layers = sorted(poles_by_layer.keys())
    if not sorted_layers:
        return []

    # Pre-fetch W_gate rows [k, 5376] for every layer's active poles
    gate_cache: Dict[int, torch.Tensor] = {}
    for l in sorted_layers:
        p = poles_by_layer[l]
        w_gate = w_gate_slice_fn(l, p.neuron_indices).float()
        if pre_ffn_norm_weights is not None and l in pre_ffn_norm_weights:
            w_gate = w_gate * pre_ffn_norm_weights[l].to(w_gate.device).float()
        gate_cache[l] = w_gate

    filaments: List[CircuitFilament] = []

    for l in sorted_layers:
        src_poles = poles_by_layer[l]
        # signed_w_down: [k_src, 5376] = sigma_i * W_down[l, i]
        src_writers = src_poles.signed_w_down.float()
        layer_candidates: List[CircuitFilament] = []

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

            # Take top couplings from this (l, l_dst) pair
            flat_abs = couplings.abs().view(-1)
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
                    )
                )

        # Retain top_k_per_source_layer strongest filaments originating from layer l
        layer_candidates.sort(key=lambda f: abs(f.coupling_score), reverse=True)
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
def build_monument_frame(
    token_ids: Tuple[int, int, int],
    lm_head_weight: torch.Tensor,
    jpp_lenses: Optional[Dict[int, torch.Tensor]] = None,
    pullback_per_layer: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Monument Mode Survey Frame:
    User picks 3 vocabulary tokens/concepts (t_1, t_2, t_3); pull their unembedding vectors
    back to layer l via J_pp[l] (`v_m[l] = W_U[t_m] @ J_pp[l]`) and orthonormalize via
    Gram-Schmidt.

    Returns:
        - `global_basis_J`: `[3, 5376]` orthonormal basis in J++-Space (`W_U[t_m]` orthonormalized)
        - `layer_pullback_basis`: `[60, 3, 5376]` per-layer pulled-back orthonormal basis
          (or `[3, 5376]` expanded if `pullback_per_layer=False` or `jpp_lenses` is None).
    """
    t1, t2, t3 = token_ids
    u1 = lm_head_weight[t1].float()
    u2 = lm_head_weight[t2].float()
    u3 = lm_head_weight[t3].float()

    global_basis_J = gram_schmidt_orthonormalize_3d(u1, u2, u3)  # [3, 5376]

    if not pullback_per_layer or jpp_lenses is None:
        return global_basis_J, global_basis_J.unsqueeze(0).expand(N_LAYERS, 3, D_MODEL).contiguous()

    layer_bases: List[torch.Tensor] = []
    for l in range(N_LAYERS):
        J_l = jpp_lenses[l].float()
        # Pull unembedding vector back to layer l via J_pp[l]: v_m[l] = u_m @ J_pp[l]
        v1_l = u1.to(J_l.device) @ J_l
        v2_l = u2.to(J_l.device) @ J_l
        v3_l = u3.to(J_l.device) @ J_l
        e_l = gram_schmidt_orthonormalize_3d(v1_l, v2_l, v3_l)  # [3, 5376]
        layer_bases.append(e_l)

    return global_basis_J, torch.stack(layer_bases, dim=0)  # [60, 3, 5376]


@torch.no_grad()
def build_pole_motif_frame(
    layer_idx: int,
    neuron_idx: int,
    polarity: float,
    w_gate_vec: torch.Tensor,
    w_down_vec: torch.Tensor,
    J_l: torch.Tensor,
    downstream_gate_vec: torch.Tensor,
) -> torch.Tensor:
    """Pole-Locked Motif Mode Survey Frame:
    User clicks a firing neuron pole (l, i, sigma_i):
      - Axis X = upstream gate W_gate[l, i]
      - Axis Y = J-corrected write (sigma_i * W_down[l, i]) @ J_pp[l].T
      - Axis Z = dominant downstream reader gate W_gate[l+1, j*]
    Orthonormalized in fp32 via Gram-Schmidt.
    Returns `basis` of shape `[3, 5376]`.
    """
    dev = J_l.device
    v_x = w_gate_vec.to(device=dev, dtype=torch.float32)
    signed_down = float(polarity) * w_down_vec.to(device=dev, dtype=J_l.dtype)
    v_y = (signed_down @ J_l.T).float()
    v_z = downstream_gate_vec.to(device=dev, dtype=torch.float32)
    return gram_schmidt_orthonormalize_3d(v_x, v_y, v_z)


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


@torch.no_grad()
def project_trajectory_to_3d(
    resid_post_layers: torch.Tensor,
    resid_post_J_layers: torch.Tensor,
    delta_attn_J_layers: torch.Tensor,
    delta_ffn_J_layers: torch.Tensor,
    basis: torch.Tensor,
    use_jpp_states: bool = True,
    eps: float = 1e-8,
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
    """
    dev = resid_post_layers.device
    h_raw = resid_post_layers.float()
    h_J = resid_post_J_layers.float()
    d_attn = delta_attn_J_layers.float()
    d_ffn = delta_ffn_J_layers.float()
    B = basis.to(device=dev, dtype=torch.float32)

    target_states = h_J if use_jpp_states else h_raw

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
    )


@torch.no_grad()
def compute_linear_cka_matrix(
    layer_seq_states: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the [60, 60] Linear CKA similarity matrix across all 60 layers (Kornblith et al. 2019).

    Args:
        layer_seq_states: Tensor of shape `[60, n_tokens, 5376]` (or `[60, 5376]` if single token,
            in which case centered feature cosine similarity is returned).
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
    X_c = X - X.mean(dim=1, keepdim=True)  # [L, N, D]
    # Gram matrices G_l = X_c[l] @ X_c[l].T of shape [L, N, N]
    G = torch.bmm(X_c, X_c.transpose(1, 2))  # [L, N, N]
    # Center Gram matrices: H G H
    G = G - G.mean(dim=-1, keepdim=True) - G.mean(dim=-2, keepdim=True) + G.mean(dim=(-2, -1), keepdim=True)
    flat_G = G.reshape(G.shape[0], -1)  # [L, N*N]
    norms = torch.linalg.vector_norm(flat_G, dim=-1, keepdim=True) + eps
    unit_G = flat_G / norms
    cka = unit_G @ unit_G.T
    return torch.clamp(cka, 0.0, 1.0)
