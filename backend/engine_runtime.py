"""Execution & Hook Runtime (`decoderesearch/interp-engine`) + Pre-Fitted Coordinate Lenses
(`neuronpedia/jacobian-lens`) + On-the-Fly W4A16 GPU Dequantizer for Gemma-4-31B.

Preserves the exact linear algebra of Gemma-4-31B (`d_model=5376`, `L=60`, `d_ff=21504`,
`vocab_size=262144`, `final_logit_softcapping=30.0`) inside a single 32GB GPU (< 26GB peak VRAM).

Based on (numbers refer to REFERENCES.md): interp-engine [36]; the Jacobian lens [2,3] and J++ lens [4] files hosted by
Neuronpedia [6]; the Gemma 4 QAT W4A16 checkpoint [1] loaded via compressed-tensors [39].
"""

from __future__ import annotations

import gc
import logging
import re
import secrets
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from interp_engine import Address, EagerModel
from interp_engine.sampling import SamplingSettings, apply_presence_penalty
from interp_engine.steer import generate_stream as ie_generate_stream
from interp_engine.steer import steer as ie_steer
from interp_engine.steer import steering_spec_to_eager_specs
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, OrthogonalDecompSpec, SteeringSpec
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from backend.theodolite_math import (
    FRAME_SQUARINGS,
    POLE_RANKINGS,
    THREAD_SCORES,
    BipolarPolesResult,
    CausalSplitResult,
    CircuitFilament,
    D_FF,
    D_MODEL,
    DirectPathInputs,
    FINAL_LOGIT_SOFTCAP,
    GLOBAL_ATTN_LAYERS_0BASED,
    GLOBAL_ATTN_LAYERS_1BASED,
    N_LAYERS,
    TheodoliteTrajectory3D,
    VOCAB_SIZE,
    build_monument_frame,
    build_pole_motif_frame,
    compute_causal_vector_split,
    compute_downstream_filaments,
    compute_linear_cka_matrix,
    extract_bipolar_poles,
    frame_word_geometry,
    jpp_readout_logits,
    pole_write_shares,
    project_trajectory_to_3d,
    unembed_gemma4,
    validate_and_cast_lens,
)

logger = logging.getLogger(__name__)

DEFAULT_W4A16_MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
CANONICAL_MODEL_ID = "google/gemma-4-31b-it"
LENS_REPO_ID = "neuronpedia/jacobian-lens"
JPP_LENS_FILENAME = "gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt"
JLENS_FILENAME = "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt"
# The lens every reading goes through, as GET /api/health, every sweep, every scientist tool reply (MCP, HTTP and
# WebSocket) and every vault statement that reads words or writes an edit through it (`LarqlVaultEngine.execute_lql`)
# name it ("lens", from `TheodoliteRuntime.lens_card`). The shipped J++ file was fitted by its authors on the
# BASE model in bf16 and is applied here to the instruct W4A16 checkpoint (REPORT §0, §3.4 bug 9; R1). Its own
# "provenance" block (source, method, lrp_mode, source_config.hf_model_name, n_prompts, max_seq_len) names the model
# and the recipe but neither the dtype nor the corpus. The facts below fill those gaps, and only for a file whose
# provenance says it is that kind of file (hf_model_name google/gemma-4-31B, method jpp); any other file is described
# by its own provenance alone, with None for what it does not record.
JPP_FITTED_MODEL = "google/gemma-4-31B"
JPP_LENS_FALLBACK: Dict[str, Any] = {
    "id": "jpp-base",
    "name": "J++",
    "fitted_on": f"{JPP_FITTED_MODEL} (base, bf16)",
    "method": "J++, R-lens LRP backward pass",
    # the corpus is the file's Salesforce-wikitext folder (WikiText-103, REPORT §0); the counts are its provenance's
    "fit_data": "64 WikiText passages of 128 word-pieces",
    # Layers count from 0, as everywhere in this API (the pages count from 1). The bands are the study's: condition
    # means track behaviour at r 0.88 at L20-29, and at L10-19 at 0.92 in E1 but 0.31 in E2 (E1 table, E2 §6); no
    # average-Jacobian lens is stable below L30 (E3, R1); contrasts at L30-49 and the L50-58 pronoun flag are erratum
    # E49; the she-lean is erratum E2; neuron labels were sampled at L10/20/30/40/45/50/55/57 (N1).
    "caution": (
        "Fitted on the base model, not on the model it reads: compare its readings between matched sightings; do not "
        "take its words as this model's own. Layers count from 0 here (the pages count from 1). The order of "
        "conditions it gives was checked against the model's behaviour from layer 20; at layers 20-29 trust only that "
        "order; below layer 20 the study's two checks disagreed, so treat even comparisons there as unconfirmed. Use "
        "layers 30-49 for condition contrasts. At layers 50-58 this lens reads 'she' at or above a woman persona for "
        "6-7 of the 9 AI personas the study tried: flag pronoun readouts there unless a lens fitted on the instruct "
        "model agrees. Neuron-pole word labels predicted causal effects only at the sampled layers 40-57; they were at "
        "chance at 10-30, and 31-39 were not sampled."
    ),
}
_LENS_METHOD_NAMES = {"jpp": "J++"}
# The name sent for a file whose provenance records no method. The pages print "the {name} lens", so it reads "the
# unnamed lens"; it is never None, so a card that is present but incomplete is not mistaken for no card at all.
UNNAMED_LENS = "unnamed"
_LRP_MODE_NAMES = {"rlens": "R-lens LRP backward pass"}
_PROVENANCE_KEYS = ("source", "method", "lrp_mode", "hf_model_name", "n_prompts", "max_seq_len")
# For a lens file other than the shipped one: the study's checks were made on that one and do not carry over.
_OTHER_LENS_CAUTION = (
    "Until it is checked against this model's behaviour, compare its readings between matched sightings and do not "
    "take its words as this model's own; the instrument's checks were made with another lens file."
)


def _fitted_kind(hf_model_name: str) -> Optional[str]:
    """"base" or "instruct" where the model's name says so (the base model is known by name); None otherwise."""
    if hf_model_name == JPP_FITTED_MODEL:
        return "base"
    if re.search(r"-it\b|instruct", hf_model_name, re.IGNORECASE):
        return "instruct"
    return None

# The 181 residual dims that are deep-quiet at every layer L42-57 (V6; REPORT §2.2), a [5376] bool mask.
QUIET_DIMS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "results" / "V6_quiet_dims_pers181.npy"
# A fixed reference set for each layer's quietest tenth of dims (the q10 set of E6/V6, which holds 99-100% of a J++
# gender pullback's energy at L35-58, where the 181 hold 56% at L30 and 80% at L40; V6 energy shares). Quietness is a property of the network, not of the text
# (V6: prose/chat deep-set Jaccard 0.89-1.00), so one small mixed set serves every prompt. The texts are V7's
# (V7_tool_bugs_common.py): three bare prompts, the ones V7's fluent quiet-masked nudge was measured on, and three chats.
QUIET_REFERENCE_RAW: Tuple[str, ...] = (
    "The number of legs on the animal that spins webs is",
    "When the nurse finished the shift,",
    "My friend Gemma is a software engineer. Yesterday",
)
QUIET_REFERENCE_CHAT: Tuple[Tuple[str, str], ...] = (
    ("What is the capital of France?", "The capital of France is"),
    ("Give me a fun fact about octopuses.", "Octopuses have"),
    ("Tell me a story about a king.", "Once upon a time, a king"),
)
QUIET_QUANTILE = 0.10


def _provenance_text(v: Any) -> Optional[str]:
    """A provenance field as text; None when absent or blank (so a blank field counts as not recorded)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _provenance_count(v: Any) -> Optional[int]:
    """A provenance count as a plain int (a file may hold a one-element tensor); None unless a positive whole number."""
    try:
        n = v.item() if hasattr(v, "item") else v
        if isinstance(n, bool) or float(n) != int(n):
            return None
        n = int(n)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    return n if n > 0 else None


def _read_lens_provenance(payload: Any) -> Dict[str, Any]:
    """The fields of a lens file's "provenance" block that name what it was fitted on and how, flattened, as text and
    plain ints (they go out as JSON in every lens card); {} if none. A field that is absent, blank or not a usable
    value is left out, as not recorded."""
    if not isinstance(payload, dict):
        return {}
    prov = payload.get("provenance")
    prov = prov if isinstance(prov, dict) else {}
    cfg = prov.get("source_config")
    cfg = cfg if isinstance(cfg, dict) else {}
    out = {
        "source": _provenance_text(prov.get("source")),
        "method": _provenance_text(prov.get("method")),
        "lrp_mode": _provenance_text(prov.get("lrp_mode")) or _provenance_text(cfg.get("lrp_mode")),
        "hf_model_name": _provenance_text(cfg.get("hf_model_name")),
        "n_prompts": (
            _provenance_count(payload.get("n_prompts")) or _provenance_count(cfg.get("num_prompts_trained_on"))
        ),
        "max_seq_len": _provenance_count(cfg.get("max_seq_len")),
    }
    return {k: v for k, v in out.items() if v is not None}


_SHIFTS_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_nibble_shifts(device: torch.device) -> torch.Tensor:
    shifts = _SHIFTS_CACHE.get(device)
    if shifts is None:
        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=device, dtype=torch.int32)
        _SHIFTS_CACHE[device] = shifts
    return shifts


@torch.no_grad()
def dequantize_w4a16_full(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Bit-exact GPU W4A16 group-32 symmetric unpacker matching `compressed-tensors` pack-quantized.

    Args:
        packed: int32 tensor of shape `[out_features, in_features // 8]`
        scale: bfloat16 tensor of shape `[out_features, in_features // 32]`
    Returns:
        `weight`: bfloat16 tensor of shape `[out_features, in_features]`
    """
    rows = packed.shape[0]
    cols = packed.shape[1] * 8
    shifts = _get_nibble_shifts(packed.device)
    w_q = (((packed.unsqueeze(-1) >> shifts) & 0xF) - 8).to(dtype=scale.dtype)
    return (w_q.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, cols)


@torch.no_grad()
def dequantize_w4a16_rows(
    packed: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """O(K) row-slice W4A16 dequantizer (e.g. for `W_gate[l, idx]` or `W_up[l, idx]`).

    Returns tensor of shape `[K, in_features]`.
    """
    idx = row_indices.to(device=packed.device, dtype=torch.long)
    sub_packed = packed.index_select(0, idx)  # [K, C // 8]
    sub_scale = scale.index_select(0, idx)    # [K, C // 32]
    return dequantize_w4a16_full(sub_packed, sub_scale)


@torch.no_grad()
def dequantize_w4a16_cols(
    packed: torch.Tensor,
    scale: torch.Tensor,
    col_indices: torch.Tensor,
) -> torch.Tensor:
    """O(K) column-slice W4A16 dequantizer (e.g. for `W_down[l][:, idx]`).

    Returns tensor of shape `[K, out_features]` (i.e. `[K, 5376]` where each row k is the
    5376-D residual write vector of neuron `col_indices[k]`).
    """
    idx = col_indices.to(device=packed.device, dtype=torch.long)
    word_col = idx // 8
    bit_shift = ((idx % 8) * 4).to(dtype=torch.int32)
    scale_col = idx // 32

    # packed[:, word_col] has shape [out_features, K]
    words = packed.index_select(1, word_col)
    scales = scale.index_select(1, scale_col)
    w_q = (((words >> bit_shift.unsqueeze(0)) & 0xF) - 8).to(dtype=scale.dtype)
    w_cols = (w_q * scales).T.contiguous()  # [K, out_features]
    return w_cols


def _make_w4a16_linear_forward(module: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an on-the-fly W4A16 forward method with live LARQL `.patch` MEMIT overlay support."""

    def _w4a16_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_w4a16_full(self.weight_packed, self.weight_scale)
        # Apply any active LARQL .patch MEMIT column overlays (on down_proj: w[:, col_idx] = new_vec)
        col_overlays: Optional[Dict[int, torch.Tensor]] = getattr(self, "_memit_col_overlays", None)
        if col_overlays:
            w = w.clone()
            for col_idx, new_col in col_overlays.items():
                w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
        # Apply any active LARQL .patch MEMIT row overlays (on gate_proj / up_proj: w[row_idx, :] = new_vec)
        row_overlays: Optional[Dict[int, torch.Tensor]] = getattr(self, "_memit_row_overlays", None)
        if row_overlays:
            if not col_overlays:
                w = w.clone()
            for row_idx, new_row in row_overlays.items():
                w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)

        bias = getattr(self, "bias", None)
        return F.linear(x.to(dtype=w.dtype), w, bias)

    return types.MethodType(_w4a16_forward, module)


def patch_compressed_tensors_for_low_vram(hf_model: nn.Module) -> int:
    """Prevent `compressed-tensors` from unpacking 4-bit weights to 62GB bf16 at load/first-forward,
    drop unused vision towers on multimodal Gemma-4, and install on-the-fly W4A16 GPU linear kernels.

    Returns the number of quantized linear layers patched.
    """
    # 1. Remove compressed-tensors' automatic full-model decompression pre-hook
    hook = getattr(hf_model, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()
        delattr(hf_model, "ct_decompress_hook")

    # 2. Drop unused vision tower & vision embeddings to save ~1.15 GB of GPU VRAM
    inner_model = getattr(hf_model, "model", None)
    if inner_model is not None:
        for unused_attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
            if hasattr(inner_model, unused_attr) and getattr(inner_model, unused_attr) is not None:
                setattr(inner_model, unused_attr, None)

    # 3. Bind on-the-fly W4A16 GPU forward to every quantized Linear module
    patched_count = 0
    for _name, mod in hf_model.named_modules():
        if hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
            mod._memit_col_overlays = {}
            mod._memit_row_overlays = {}
            mod.forward = _make_w4a16_linear_forward(mod)
            patched_count += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return patched_count



# Word-pieces whose rank and logit every sweep tracks layer by layer (the default two-hop prompt's words).
DEFAULT_TRACKED_CONCEPTS: Tuple[str, ...] = ("spider", "arachnid", "eight", "8", "insect", "six", "6")

@dataclass
class SweepLayerTokenReadout:
    """Top-k vocabulary predictions at a layer under J++ or Naive Logit Lens."""

    layer_idx: int
    top_token_ids: List[int]
    top_token_strs: List[str]
    top_probs: List[float]
    top_logits: List[float]
    tracked_token_probs: Dict[str, float] = field(default_factory=dict)
    tracked_token_logits: Dict[str, float] = field(default_factory=dict)
    tracked_token_ranks: Dict[str, int] = field(default_factory=dict)


# The most word-pieces one sweep may hold, prompt and answer together (the server refuses more, in plain words): a
# capture holds all 300 of its addresses at every position on the GPU, about 5.2 MB per position
# (60 x (4 x 5376 + 21504) x 2 bytes), before it moves to the CPU, and the writing's first forward builds full-length
# logits (0.5 MB per position) besides. Measured on this RTX 5090 (32.6 GB) with the desktop holding about 4.5 GB
# (2026-10-18): 621 positions peaked at 31.5 GB and about 750 ran out of memory, so 512 leaves about 1.5 GB spare.
# A sweep that still runs out is refused in plain words and the socket keeps answering (server._stream_sweep).
MAX_SWEEP_POSITIONS = 512


@dataclass
class SamplerSettings:
    """How an answer is written. The defaults are the study's greedy decoding: its likeliest word-piece each time.

    Every knob goes to interp_engine as the value it means. A knob left None there takes the checkpoint's
    recommendation (temperature 1, top-k 64, top-p 0.95), so "off" is top_k 0 and top_p 1.0, never None. The presence
    penalty is subtracted from the logit of every word-piece already written (not the prompt's) before the temperature,
    on the greedy path too. A temperature of 0 is greedy, and a seed does nothing there."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return float(self.temperature) <= 0.0


@dataclass
class WrittenStep:
    """One word-piece of an answer and the odds it was chosen at (the wire's STEP).

    The model's numbers are its own odds at that step, from the raw logits in float32 (temperature 1, no filter, no
    penalty). The sampler's are the chance the sampler gave it, replayed op for op on the same logits
    (`sampler_chances`); None when greedy. A written piece's logits are the writing's own step logits; a given
    piece's are the recapture's layer-60 readout at its read position."""

    index: int
    token_id: int
    token: str
    model_prob: float
    model_logprob: float
    model_rank: int  # 1 + how many word-pieces had a higher logit
    n_tied: int  # how many others had exactly its logit (greedy takes the lowest id of a tie)
    chosen_logit: float
    top: List[Tuple[int, str, float]]  # the model's five likeliest: (id, word-piece, prob)
    sampler_prob: Optional[float]
    sampler_support: Optional[int]  # how many word-pieces the sampler could have drawn
    end_mark: bool  # one of the end marks that stop the writing
    source: str  # "written" (by this sweep) | "given" (by the request)

    def as_wire(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "token_id": self.token_id,
            "token": self.token,
            "model_prob": self.model_prob,
            "model_logprob": self.model_logprob,
            "model_rank": self.model_rank,
            "n_tied": self.n_tied,
            "chosen_logit": self.chosen_logit,
            "top": [{"token_id": i, "token": s, "prob": p} for i, s, p in self.top],
            "sampler_prob": self.sampler_prob,
            "sampler_support": self.sampler_support,
            "end_mark": self.end_mark,
            "source": self.source,
        }


@torch.no_grad()
def sampler_draw_distribution(
    logits: torch.Tensor, settings: SamplingSettings, generated: Sequence[int]
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """The distribution interp_engine's sampler (`steer._sample_next`, private) draws the next word-piece from,
    replayed op for op in the logits' own dtype: the presence penalty on the raw logits, then / temperature, the
    top-k mask (ties at the k-th value kept), softmax, then top-p (sorted, `cum - p > top_p` masked, renormalised).

    Returns (probs, None) when it draws in word-piece order, or (sorted_probs, sorted_idx) when top-p is on (it then
    draws in sorted order, so `torch.multinomial` on these, after the same `torch.manual_seed`, picks what it picks:
    scripts/check_sampler_mirror.py). None when greedy (temperature 0: the argmax of the penalised logits).
    `settings` is the resolved form (`EagerModel.sampling_settings`), whose top_k / top_p are None when off."""
    logits = apply_presence_penalty(logits, generated, settings.presence_penalty)
    if settings.temperature <= 0:
        return None
    logits = logits / settings.temperature
    if settings.top_k:
        kth = torch.topk(logits, settings.top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    if settings.top_p and settings.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > settings.top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        return sorted_probs, sorted_idx
    return probs, None


@torch.no_grad()
def sampler_chances(
    logits: torch.Tensor, settings: SamplingSettings, generated: Sequence[int]
) -> Optional[torch.Tensor]:
    """`[vocab]`: each word-piece's chance under the sampler (`sampler_draw_distribution`, scattered back to
    word-piece order); None when greedy."""
    drawn = sampler_draw_distribution(logits, settings, generated)
    if drawn is None:
        return None
    probs, order = drawn
    return probs if order is None else torch.zeros_like(probs).scatter_(-1, order, probs)


@torch.no_grad()
def step_numbers(
    logits: torch.Tensor,
    chosen: int,
    settings: SamplingSettings,
    generated_before: Sequence[int],
    top_n: int = 5,
) -> Dict[str, Any]:
    """The numbers of a `WrittenStep`, from one step's `[vocab]` logits: the model's own odds in float32 (prob and
    logprob of the chosen id, its rank, its exact ties, its logit, the top `top_n`) and the sampler's chance for it."""
    lf = logits.float()
    probs = F.softmax(lf, dim=-1)
    logprobs = F.log_softmax(lf, dim=-1)
    c = int(chosen)
    c_logit = lf[c]
    top_p, top_i = torch.topk(probs, k=min(int(top_n), probs.numel()))
    q = sampler_chances(logits, settings, generated_before)
    return {
        "model_prob": float(probs[c].item()),
        "model_logprob": float(logprobs[c].item()),
        "model_rank": int((lf > c_logit).sum().item()) + 1,
        "n_tied": int((lf == c_logit).sum().item()) - 1,
        "chosen_logit": float(c_logit.item()),
        "top": [(int(i), float(p)) for i, p in zip(top_i.cpu().tolist(), top_p.cpu().tolist())],
        "sampler_prob": float(q[c].item()) if q is not None else None,
        "sampler_support": int((q > 0).sum().item()) if q is not None else None,
    }


@dataclass
class TheodoliteSweepBundle:
    """Complete result of a 60-layer Theodolite optical & circuit sweep."""

    prompt: str
    prompt_token_ids: List[int]
    prompt_token_strs: List[str]
    target_pos: int
    generated_text: str
    generated_token_ids: List[int]
    trajectory_3d: TheodoliteTrajectory3D
    causal_splits: Dict[int, CausalSplitResult]
    bipolar_poles: Dict[int, BipolarPolesResult]
    filaments: List[CircuitFilament]
    cka_matrix: torch.Tensor  # [60, 60] float32
    jpp_layer_readouts: Dict[int, SweepLayerTokenReadout]
    naive_layer_readouts: Dict[int, SweepLayerTokenReadout]
    pole_top_tokens_jpp: Dict[Tuple[int, int], List[Tuple[str, float]]]
    pole_top_tokens_naive: Dict[Tuple[int, int], List[Tuple[str, float]]]
    survey_mode: str
    survey_labels: Tuple[str, str, str]
    peak_vram_gb: float
    elapsed_ms: float
    # The survey frame actually used: [3, 5376] or [60, 3, 5376], and whether it lives in J++ space ("J",
    # read against h_l J_l^T) or in each layer's raw residual coordinates ("raw", a per-layer pullback).
    survey_basis: Optional[torch.Tensor] = None
    survey_basis_space: str = "J"
    frame_info: Dict[str, Any] = field(default_factory=dict)
    # Per layer: each listed pole's write size s*|a|*||w_down (.) w_post||/rms(y), and the share of the FFN write
    # the listed poles explain (None when mlp_out was not captured).
    pole_write_sizes: Dict[int, List[float]] = field(default_factory=dict)
    poles_ffn_explained: Dict[int, Optional[float]] = field(default_factory=dict)
    pole_ranking: str = "activation"
    thread_score: str = "coupling"
    # Which word-piece of the answer this sighting belongs to (all defaulted, so a caller that reads only the fields
    # above is untouched). Token index k reads at target_pos = P - 1 + k: the stream that has just read
    # `read_token` (the prompt's last piece for k = 0, else answer[k - 1]) and whose layer 60 chose `chosen`
    # (answer[k]; None when nothing was written).
    token_index: int = 0
    read_token_id: Optional[int] = None
    read_token: Optional[str] = None
    chosen: Optional[WrittenStep] = None
    # At layer 60 (J = I): the readout's top word-piece, whether it is the chosen one, and the readout's logit for the
    # chosen one less the writing's own (the reading forward and the writing forward agree to bf16 rounding). For a
    # given piece the logit and the diff are None: its odds are this readout's own, with no writing to compare with.
    recapture_check: Optional[Dict[str, Any]] = None
    kinship_n_states: Optional[int] = None  # the word-pieces kinship (CKA) is taken over: target_pos + 1
    # "none" | "control_prompt_end" (k = 0: the controls' own last piece) | "control_reading_same_answer" (k >= 1)
    centre_kind: str = "none"
    # The pole-motif frame's dominant reader {layer, neuron, from_token_index}: the position its frame was built at.
    frame_reader: Optional[Dict[str, Any]] = None
    # Per layer, per tracked word: True where its J++ rank, logit and prob were taken from the next position (the
    # space-then-digit rule, `rank_lookahead`), not this one.
    tracked_from_next: Dict[int, Dict[str, bool]] = field(default_factory=dict)
    rank_lookahead: bool = True
    analyse_ms: float = 0.0

    @property
    def read_pos(self) -> int:
        """The read position, P - 1 + token_index (the same as `target_pos`)."""
        return self.target_pos


@dataclass
class SweepCapture:
    """What one sweep wrote, and the recapture that reads it: prompt + answer[:-1] in one forward, inside the same
    steering block as the writing. Row P - 1 + k holds the stream that read prompt + answer[:k] and chose answer[k]
    (`analyse_position(k)`); the last piece is chosen but never read. `caps` are CPU tensors."""

    prompt: str
    prompt_token_ids: List[int]
    prompt_token_strs: List[str]
    answer_token_ids: List[int]
    answer_token_strs: List[str]  # each id decoded alone, clean_up_tokenization_spaces=False
    answer_text: str
    steps: List[WrittenStep]
    caps: Dict[Address, torch.Tensor]
    # The sampler as resolved and applied (`SamplingSettings.as_dict`); top_k / top_p None when off, or when greedy
    # (unused there).
    settings: Dict[str, Any]
    greedy: bool
    seed: Optional[int]
    seed_drawn: bool
    stop_reason: str  # "end_mark" | "length" | "stopped" | "given" | "none"
    stop_token_ids: List[int]
    answer_source: str  # "written" | "given"
    max_new_tokens: int
    generate_ms: float
    capture_ms: float


@dataclass
class SweepShared:
    """What every sighting of one sweep shares (`prepare_sweep`): the tracked words, the survey frame (a pole-motif
    frame is built at the first position analysed and kept), and the matched controls' states."""

    tracked_map: Dict[str, int]
    survey_mode: str
    locked_pole: Optional[Tuple[int, int, float]]
    frame_squaring: str
    basis: Optional[torch.Tensor]
    basis_space: str
    frame_info: Dict[str, Any]
    survey_labels: Optional[Tuple[str, str, str]]
    control_ids: List[List[int]]
    # The controls' mean state at their own last piece, in the frame's space (the centre at token 0)
    centre_k0: Optional[torch.Tensor]
    centre_answer: bool
    # [n - 1, 60, 5376] float32 CPU: the controls reading this very answer, rows k - 1 (filled on first need)
    answer_centre_raw: Optional[torch.Tensor] = None
    prepare_ms: float = 0.0


class TheodoliteRuntime:
    """Manages the `interp-engine` EagerModel, `neuronpedia/jacobian-lens` J++ & JLens maps,
    and fast W4A16 weight accessors for Gemma-4-31B.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_W4A16_MODEL_ID,
        lens_repo_id: str = LENS_REPO_ID,
        device: str = "cuda:0",
        load_jlens: bool = True,
    ) -> None:
        self.model_id = model_id
        self.lens_repo_id = lens_repo_id
        self.device = torch.device(device)
        self.load_jlens = load_jlens

        self.eager_model: Optional[EagerModel] = None
        self.hf_model: Optional[nn.Module] = None
        self.tokenizer: Any = None

        self.jpp_lenses: Dict[int, torch.Tensor] = {}
        self.jlens_lenses: Dict[int, torch.Tensor] = {}
        # the J++ file's own "provenance" block (plus its top-level n_prompts), as loaded; {} until then
        self.jpp_provenance: Dict[str, Any] = {}

        self._decoder_layers: Optional[nn.ModuleList] = None
        self._final_norm_weight: Optional[torch.Tensor] = None
        self._lm_head_weight: Optional[torch.Tensor] = None
        self._pre_ffn_norm_weights: Dict[int, torch.Tensor] = {}
        self._post_ffn_norm_weights: Dict[int, torch.Tensor] = {}
        # ||W_down[l][:, i] (.) w_post|| for every neuron, from the unpatched weights (LARQL overlays applied on use)
        self._write_norm_cache: Dict[int, torch.Tensor] = {}
        self._quiet_mask: Optional[torch.Tensor] = None
        self._quiet_layer_masks: Optional[torch.Tensor] = None  # [60, 5376] bool, CPU

    def load(self) -> "TheodoliteRuntime":
        """Load the W4A16 Gemma-4-31B model into `interp-engine` and load/validate the J++ lenses."""
        t0 = time.perf_counter()
        logger.info("Loading Gemma-4-31B W4A16 model from %s on %s...", self.model_id, self.device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False, local_files_only=True)
            hf_model, loading_info = Gemma4ForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype="auto",
                device_map=str(self.device),
                trust_remote_code=False,
                output_loading_info=True,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
            hf_model, loading_info = Gemma4ForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype="auto",
                device_map=str(self.device),
                trust_remote_code=False,
                output_loading_info=True,
            )
        setattr(hf_model, "ie_missing_keys", frozenset(loading_info.get("missing_keys", ()) or ()))

        n_patched = patch_compressed_tensors_for_low_vram(hf_model)
        logger.info("Patched %d W4A16 compressed-tensors linear modules for on-the-fly GPU execution.", n_patched)

        self.eager_model = EagerModel(
            self.model_id,
            device=str(self.device),
            dtype="bfloat16",
            hf_model=hf_model,
            tokenizer=tokenizer,
        )
        self.hf_model = hf_model
        self.tokenizer = tokenizer

        # Cache layer references and norm weights
        self._decoder_layers = self.eager_model.arch.decoder_layers
        self._final_norm_weight = self.eager_model.arch.final_norm.weight.detach()
        self._lm_head_weight = self.eager_model.arch.lm_head.weight.detach()

        for l in range(N_LAYERS):
            block = self._decoder_layers[l]
            if hasattr(block, "pre_feedforward_layernorm"):
                self._pre_ffn_norm_weights[l] = block.pre_feedforward_layernorm.weight.detach()
            if hasattr(block, "post_feedforward_layernorm"):
                self._post_ffn_norm_weights[l] = block.post_feedforward_layernorm.weight.detach()

        # Load & validate Pre-Fitted Coordinate Lenses (J++ and optionally JLens)
        self._load_coordinate_lenses()

        # Register J++ lenses on interp_engine's lens stream as well
        self.eager_model._lens_jacobians["jpp"] = self.jpp_lenses
        if self.jlens_lenses:
            self.eager_model._lens_jacobians["jlens"] = self.jlens_lenses

        vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0.0
        logger.info(
            "TheodoliteRuntime ready in %.2fs | Peak VRAM: %.2f GB (limit: 30.0 GB)",
            time.perf_counter() - t0,
            vram_gb,
        )
        return self

    def _load_coordinate_lenses(self) -> None:
        """Load J++ (`/jpp/`) and Jacobian (`/jlens/`) lenses from Hugging Face cache/hub,
        validate every layer matrix for non-finite values, and cast fp16 -> bfloat16.
        """
        try:
            jpp_path = hf_hub_download(repo_id=self.lens_repo_id, filename=JPP_LENS_FILENAME, local_files_only=True)
        except Exception:
            jpp_path = hf_hub_download(repo_id=self.lens_repo_id, filename=JPP_LENS_FILENAME)
        logger.info("Loading J++ Lens from %s...", jpp_path)
        jpp_payload = torch.load(jpp_path, map_location="cpu", weights_only=True)
        raw_jpp = jpp_payload["J"] if isinstance(jpp_payload, dict) and "J" in jpp_payload else jpp_payload
        self.jpp_provenance = _read_lens_provenance(jpp_payload)

        for l in range(N_LAYERS - 1):
            self.jpp_lenses[l] = validate_and_cast_lens(
                raw_jpp[l],
                layer_idx=l,
                target_dtype=torch.bfloat16,
                device=self.device,
            )
        # Final layer 59 is the target unembedding frame (Identity)
        self.jpp_lenses[N_LAYERS - 1] = torch.eye(D_MODEL, dtype=torch.bfloat16, device=self.device)
        del raw_jpp, jpp_payload
        logger.info("J++ lens: %s", self.lens_card())
        gc.collect()

        if self.load_jlens:
            try:
                try:
                    jlens_path = hf_hub_download(
                        repo_id=self.lens_repo_id, filename=JLENS_FILENAME, local_files_only=True
                    )
                except Exception:
                    jlens_path = hf_hub_download(repo_id=self.lens_repo_id, filename=JLENS_FILENAME)
                logger.info("Loading Jacobian Lens (JLens) from %s...", jlens_path)
                jlens_payload = torch.load(jlens_path, map_location="cpu", weights_only=True)
                raw_jlens = (
                    jlens_payload["J"] if isinstance(jlens_payload, dict) and "J" in jlens_payload else jlens_payload
                )
                # Keep JLens validated in bfloat16 on CPU (moved to GPU on demand) to conserve ~3.4GB VRAM
                for l in range(N_LAYERS - 1):
                    self.jlens_lenses[l] = validate_and_cast_lens(
                        raw_jlens[l],
                        layer_idx=l,
                        target_dtype=torch.bfloat16,
                        device="cpu",
                    )
                self.jlens_lenses[N_LAYERS - 1] = torch.eye(D_MODEL, dtype=torch.bfloat16, device="cpu")
                del raw_jlens, jlens_payload
                gc.collect()
            except Exception as exc:
                logger.warning("Optional JLens load skipped: %s", exc)

    def lens_card(self) -> Optional[Dict[str, Any]]:
        """The lens every reading goes through, named for the API: what it was fitted on, what it is applied to, how,
        on what, where the file came from, and how far to trust its words (`caution`, always sent). For the shipped
        kind of file (provenance: hf_model_name google/gemma-4-31B, method jpp) the provenance is completed from
        `JPP_LENS_FALLBACK`; any other file is described by its provenance alone, None where it records nothing,
        except `name`, which is `UNNAMED_LENS` when the file records no method. `provenance` repeats the file's own
        fields (None where absent). None before `load()`. scripts/check_lens_card.py checks every branch on the CPU."""
        if not self.jpp_lenses:
            return None
        p = self.jpp_provenance
        hf, method, lrp = p.get("hf_model_name"), p.get("method"), p.get("lrp_mode")
        n, seq = p.get("n_prompts"), p.get("max_seq_len")
        how = None
        if method:
            how = _LENS_METHOD_NAMES.get(method, method)
            if lrp:
                how = f"{how}, {_LRP_MODE_NAMES.get(lrp, f'{lrp} LRP backward pass')}"
        card: Dict[str, Any] = {
            "id": None,
            "name": _LENS_METHOD_NAMES.get(method, method) if method else UNNAMED_LENS,
            "fitted_on": None,
            "applied_to": self.model_id,
            "method": how,
            "fit_data": None,
            "source": p.get("source"),
            # the file the server loaded (for the shipped lens, Neuronpedia's copy of the source above)
            "file": f"{self.lens_repo_id}:{JPP_LENS_FILENAME}",
            "caution": _OTHER_LENS_CAUTION,
            "provenance": {k: p.get(k) for k in _PROVENANCE_KEYS},
        }
        if hf == JPP_FITTED_MODEL and method == "jpp":
            fb = JPP_LENS_FALLBACK
            card.update(
                id=fb["id"],
                name=fb["name"],
                fitted_on=fb["fitted_on"],
                method=how or fb["method"],
                fit_data=(
                    f"{n} WikiText passages of {seq} word-pieces" if n and seq
                    else f"{n} WikiText passages" if n
                    else fb["fit_data"]
                ),
                caution=fb["caution"],
            )
            return card
        if hf:
            kind = _fitted_kind(hf)
            card["fitted_on"] = f"{hf} ({kind})" if kind else hf
            card["id"] = f"{method}-{kind}" if method and kind else method
            card["caution"] = f"Fitted on {hf}, read here on {self.model_id}. {_OTHER_LENS_CAUTION}"
        else:
            card["id"] = method
            card["caution"] = f"The file does not say which model it was fitted on. {_OTHER_LENS_CAUTION}"
        if n:
            card["fit_data"] = f"{n} prompts of up to {seq} word-pieces" if seq else f"{n} prompts"
        return card

    @property
    def final_norm_weight(self) -> torch.Tensor:
        assert self._final_norm_weight is not None
        return self._final_norm_weight

    @property
    def lm_head_weight(self) -> torch.Tensor:
        assert self._lm_head_weight is not None
        return self._lm_head_weight

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        assert self._decoder_layers is not None
        return self._decoder_layers[layer_idx].mlp

    @torch.no_grad()
    def slice_w_down(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """Return `W_down[l, idx]` of shape `[K, 5376]` (columns `idx` of `down_proj.weight`),
        incorporating any active LARQL `.patch` MEMIT overlays.
        """
        down_proj = self.get_mlp_module(layer_idx).down_proj
        if hasattr(down_proj, "weight_packed"):
            cols = dequantize_w4a16_cols(down_proj.weight_packed, down_proj.weight_scale, neuron_indices)
        else:
            cols = down_proj.weight[:, neuron_indices.to(down_proj.weight.device)].T.contiguous()

        overlays: Optional[Dict[int, torch.Tensor]] = getattr(down_proj, "_memit_col_overlays", None)
        if overlays:
            cols = cols.clone()
            idx_list = neuron_indices.cpu().tolist()
            for k_pos, n_id in enumerate(idx_list):
                if int(n_id) in overlays:
                    cols[k_pos] = overlays[int(n_id)].to(device=cols.device, dtype=cols.dtype)
        return cols

    @torch.no_grad()
    def slice_w_gate(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """Return `W_gate[l, idx]` of shape `[K, 5376]` (rows `idx` of `gate_proj.weight`)."""
        gate_proj = self.get_mlp_module(layer_idx).gate_proj
        if hasattr(gate_proj, "weight_packed"):
            rows = dequantize_w4a16_rows(gate_proj.weight_packed, gate_proj.weight_scale, neuron_indices)
        else:
            rows = gate_proj.weight.index_select(0, neuron_indices.to(gate_proj.weight.device)).contiguous()

        overlays: Optional[Dict[int, torch.Tensor]] = getattr(gate_proj, "_memit_row_overlays", None)
        if overlays:
            rows = rows.clone()
            idx_list = neuron_indices.cpu().tolist()
            for k_pos, n_id in enumerate(idx_list):
                if int(n_id) in overlays:
                    rows[k_pos] = overlays[int(n_id)].to(device=rows.device, dtype=rows.dtype)
        return rows

    @torch.no_grad()
    def slice_w_up(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """Return `W_up[l, idx]` of shape `[K, 5376]` (rows `idx` of `up_proj.weight`)."""
        up_proj = self.get_mlp_module(layer_idx).up_proj
        if hasattr(up_proj, "weight_packed"):
            rows = dequantize_w4a16_rows(up_proj.weight_packed, up_proj.weight_scale, neuron_indices)
        else:
            rows = up_proj.weight.index_select(0, neuron_indices.to(up_proj.weight.device)).contiguous()
        return rows

    @torch.no_grad()
    def get_full_w_down(self, layer_idx: int) -> torch.Tensor:
        """Return full `[5376, 21504]` `down_proj` matrix for layer `layer_idx`."""
        down_proj = self.get_mlp_module(layer_idx).down_proj
        w = (
            dequantize_w4a16_full(down_proj.weight_packed, down_proj.weight_scale)
            if hasattr(down_proj, "weight_packed")
            else down_proj.weight.detach()
        )
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(down_proj, "_memit_col_overlays", None)
        if overlays:
            w = w.clone()
            for col_idx, new_col in overlays.items():
                w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
        return w

    @torch.no_grad()
    def get_full_w_gate(self, layer_idx: int) -> torch.Tensor:
        """Return full `[21504, 5376]` `gate_proj` matrix for layer `layer_idx`."""
        gate_proj = self.get_mlp_module(layer_idx).gate_proj
        w = (
            dequantize_w4a16_full(gate_proj.weight_packed, gate_proj.weight_scale)
            if hasattr(gate_proj, "weight_packed")
            else gate_proj.weight.detach()
        )
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(gate_proj, "_memit_row_overlays", None)
        if overlays:
            w = w.clone()
            for row_idx, new_row in overlays.items():
                w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)
        return w

    @torch.no_grad()
    def get_full_w_up(self, layer_idx: int) -> torch.Tensor:
        """Return full `[21504, 5376]` `up_proj` matrix for layer `layer_idx`."""
        up_proj = self.get_mlp_module(layer_idx).up_proj
        return (
            dequantize_w4a16_full(up_proj.weight_packed, up_proj.weight_scale)
            if hasattr(up_proj, "weight_packed")
            else up_proj.weight.detach()
        )

    def resolve_concept_token_id(self, concept_or_token: Union[str, int]) -> Tuple[int, str]:
        """Resolve a string concept (e.g., `'spider'`, `'eight'`, `'web'`) or token ID to a single vocabulary token ID."""
        if isinstance(concept_or_token, int):
            tok_str = self.tokenizer.decode([concept_or_token])
            return concept_or_token, tok_str
        text = str(concept_or_token)
        # Prefer space-prefixed word token if it encodes to a single token
        for candidate in (f" {text.strip()}", text.strip(), text):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0]), self.tokenizer.decode([ids[0]])
        ids = self.tokenizer.encode(text.strip(), add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not tokenize concept {concept_or_token!r}")
        return int(ids[0]), self.tokenizer.decode([ids[0]])

    def compose_prompt(
        self,
        text: str,
        *,
        chat: bool = False,
        system: Optional[str] = None,
        assistant_prefix: str = "",
        thinking: bool = False,
    ) -> str:
        """The exact text the model reads.

        `chat=True` wraps `text` as the user turn of the tokenizer's chat template, with an optional system turn
        before it, and appends the begun answer `assistant_prefix` after the model-turn header, as
        experiments/lab/labkit.py `Lab.chat()` does. Thinking is off (`enable_thinking=False`, the template's
        default, which leaves an empty thought channel); `thinking=True` turns it on instead. The result starts
        with "<bos>", so the single-BOS rule in `tokenize_prompt` adds no second one.

        `chat=False` sends `text` as bare text, with the begun answer, when given, on a new line after it, and
        `system` unused (a bare prompt has no system turn).
        """
        if not chat:
            prefix = assistant_prefix.strip()
            return f"{text}\n{prefix}" if prefix else text
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": text})
        templated = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking)
        )
        return templated + assistant_prefix

    def tokenize_prompt(self, prompt: str, prepend_bos: bool = True) -> List[int]:
        """Token ids with exactly one BOS: chat-template strings already begin with "<bos>" (prepending another
        gives ids [2, 2, ...]); raw text still needs one (V7 bug 6; `V7_bos_fix.patch`)."""
        assert self.eager_model is not None, "Call runtime.load() first."
        bos_str = getattr(self.tokenizer, "bos_token", None) or "<bos>"
        has_bos = prompt.startswith(bos_str)
        token_tensor = self.eager_model.to_tokens(prompt, prepend_bos=bool(prepend_bos and not has_bos))
        return [int(t) for t in token_tensor[0].tolist()]

    @property
    def quiet_dims_mask(self) -> torch.Tensor:
        """`[5376]` bool on the runtime device: the 181 persistently quiet residual dims (V6_quiet_dims_pers181.npy)."""
        if self._quiet_mask is None:
            mask = np.load(QUIET_DIMS_PATH)
            if mask.shape != (D_MODEL,) or mask.dtype != np.bool_:
                raise ValueError(f"{QUIET_DIMS_PATH} is not a [{D_MODEL}] bool mask: {mask.shape} {mask.dtype}")
            self._quiet_mask = torch.from_numpy(mask).to(self.device)
        return self._quiet_mask

    async def ensure_quiet_masks(self) -> torch.Tensor:
        """`[60, 5376]` bool (CPU): per layer, the 181 persistent quiet dims together with the layer's quietest tenth
        by std over the non-BOS positions of the fixed reference set (`QUIET_REFERENCE_*`). Measured once, unsteered.

        Why both: masking only the 181 leaves 20-44% of a J++ pullback's energy at L30-40 in other quiet,
        high-gain dims (V6 energy shares), and in a live check a fnw she/he nudge masked that way still collapsed at
        0.005 x ||h||; with the quietest tenth masked too it read fluently at 0.005-0.01 and broke at 0.02.
        """
        if self._quiet_layer_masks is not None:
            return self._quiet_layer_masks
        assert self.eager_model is not None, "Call runtime.load() first."
        texts = list(QUIET_REFERENCE_RAW) + [
            self.compose_prompt(q, chat=True, assistant_prefix=a) for q, a in QUIET_REFERENCE_CHAT
        ]
        acc: List[List[torch.Tensor]] = [[] for _ in range(N_LAYERS)]
        for text in texts:
            ids = self.tokenize_prompt(text)
            caps = await self.eager_model.capture(ids, [Address("resid_post", l) for l in range(N_LAYERS)])
            for l in range(N_LAYERS):
                acc[l].append(caps[Address("resid_post", l)][1:].float())
            del caps
        pers = self.quiet_dims_mask.cpu()
        masks = torch.zeros(N_LAYERS, D_MODEL, dtype=torch.bool)
        for l in range(N_LAYERS):
            sd = torch.cat(acc[l], dim=0).std(dim=0)
            masks[l] = (sd <= sd.quantile(QUIET_QUANTILE)) | pers
        self._quiet_layer_masks = masks
        return masks

    def layer_quiet_mask(self, layer_idx: int) -> torch.Tensor:
        """`[5376]` bool (CPU): the dims a quiet-masked nudge leaves alone at `layer_idx` (see `ensure_quiet_masks`)."""
        if self._quiet_layer_masks is None:
            raise RuntimeError("Quiet masks not measured yet: await runtime.ensure_quiet_masks() first.")
        return self._quiet_layer_masks[int(layer_idx)]

    @torch.no_grad()
    def write_col_norms(self, layer_idx: int, chunk: int = 2048) -> torch.Tensor:
        """`[21504]` float32: ||W_down[l][:, i] (.) w_post|| for every neuron i of layer l (write-size ranking).

        Computed once per layer from the packed weights in column chunks, then cached; columns replaced by an
        active LARQL `.patch` overlay are recomputed from the overlay on every call.
        """
        down_proj = self.get_mlp_module(layer_idx).down_proj
        w_post = self._post_ffn_norm_weights.get(layer_idx)
        w_post_f = w_post.to(self.device).float() if w_post is not None else None
        base = self._write_norm_cache.get(layer_idx)
        if base is None:
            parts: List[torch.Tensor] = []
            for c0 in range(0, D_FF, chunk):
                idx = torch.arange(c0, min(c0 + chunk, D_FF), device=self.device)
                if hasattr(down_proj, "weight_packed"):
                    cols = dequantize_w4a16_cols(down_proj.weight_packed, down_proj.weight_scale, idx).float()
                else:
                    cols = down_proj.weight[:, idx].T.float()
                if w_post_f is not None:
                    cols = cols * w_post_f
                parts.append(torch.linalg.vector_norm(cols, dim=-1))
                del cols
            base = torch.cat(parts)
            self._write_norm_cache[layer_idx] = base
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(down_proj, "_memit_col_overlays", None)
        if overlays:
            base = base.clone()
            for col_idx, new_col in overlays.items():
                v = new_col.to(device=self.device).float()
                base[int(col_idx)] = torch.linalg.vector_norm(v * w_post_f if w_post_f is not None else v)
        return base

    def decode_top_tokens(
        self,
        logits_1d: torch.Tensor,
        top_k: int = 5,
        tracked_token_ids: Optional[Dict[str, int]] = None,
        layer_idx: int = 0,
    ) -> SweepLayerTokenReadout:
        """Decode top-k probabilities/logits and optionally track specific bridge/target token IDs."""
        probs = F.softmax(logits_1d.float(), dim=-1)
        top_probs_t, top_ids_t = torch.topk(probs, k=top_k, dim=-1)
        top_ids = [int(i) for i in top_ids_t.cpu().tolist()]
        top_probs = [float(p) for p in top_probs_t.cpu().tolist()]
        top_logits = [float(logits_1d[i].item()) for i in top_ids]
        top_strs = [self.tokenizer.decode([i]) for i in top_ids]

        tracked_probs: Dict[str, float] = {}
        tracked_logits: Dict[str, float] = {}
        tracked_ranks: Dict[str, int] = {}
        if tracked_token_ids:
            for label, tid in tracked_token_ids.items():
                t_logit = float(logits_1d[tid].item())
                t_prob = float(probs[tid].item())
                # Compute exact 1-based rank in vocabulary
                rank = int((logits_1d > logits_1d[tid]).sum().item()) + 1
                tracked_probs[label] = t_prob
                tracked_logits[label] = t_logit
                tracked_ranks[label] = rank

        return SweepLayerTokenReadout(
            layer_idx=layer_idx,
            top_token_ids=top_ids,
            top_token_strs=top_strs,
            top_probs=top_probs,
            top_logits=top_logits,
            tracked_token_probs=tracked_probs,
            tracked_token_logits=tracked_logits,
            tracked_token_ranks=tracked_ranks,
        )

    def build_steering_spec(
        self,
        layer_idx: int,
        direction_vector: torch.Tensor,
        scale: float,
        normalize: bool = True,
        point: str = "resid_post",
    ) -> SteeringSpec:
        """Construct an `interp-engine` `SteeringSpec` (`LayerSteeringSpec` + `AddSpec`) at `layer_idx`."""
        return SteeringSpec(
            layers={
                int(layer_idx): LayerSteeringSpec(
                    operations=[
                        AddSpec(
                            vector=direction_vector.detach().float().cpu(),
                            scale=float(scale),
                            normalize=bool(normalize),
                        )
                    ]
                )
            },
            point=point,
        )

    def end_of_turn_ids(self) -> List[int]:
        """The word-pieces that end the model's turn: the checkpoint's generation_config.eos_token_id as a list
        (1 <eos>, 106 <turn|>, 50 <|tool_response> on this checkpoint), with the tokenizer's eos added if missing.
        interp_engine's own loop stops only at the tokenizer's eos, so a chat answer runs past its <turn|> without
        these."""
        gen_cfg = getattr(self.hf_model, "generation_config", None)
        raw = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        ids: List[int] = [raw] if isinstance(raw, int) else [int(t) for t in (raw or [])]
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(int(eos))
        return list(dict.fromkeys(int(t) for t in ids))

    @staticmethod
    def _sweep_points() -> List[Address]:
        """The 300 addresses every sweep captures: resid_pre, resid_mid, mlp_act, mlp_out, resid_post at each layer."""
        points: List[Address] = []
        for l in range(N_LAYERS):
            points.append(Address("resid_pre", l))
            points.append(Address("resid_mid", l))
            points.append(Address("mlp_act", l))
            points.append(Address("mlp_out", l))
            points.append(Address("resid_post", l))
        return points

    def _steering_context(
        self,
        prompt_token_ids: List[int],
        steering_spec: Optional[SteeringSpec],
        steering_position_mask: Optional[Sequence[int]],
    ):
        """The one steering block the writing and its recapture both run inside: exactly the block each of the old
        two branches opened. With a position mask, `steer()` on the spec itself (it converts it with
        `steering_spec_to_eager_specs`); without one, what `EagerModel.capture_generation` opened through
        `_maybe_steer`: the spec converted to eager specs and `steer()` on that list, or no block at all when there
        is nothing to steer. Either way the block is opened over the prompt, and the recapture declares its forward
        to start at position 0 (`forward_from`), so a mask lands on the same rows in both forwards."""
        if steering_spec is None:
            return nullcontext()
        if steering_position_mask:
            return ie_steer(
                self.eager_model,
                steering_spec,
                prompt_token_ids=torch.tensor([prompt_token_ids]),
                position_mask=[int(p) for p in steering_position_mask],
            )
        specs = steering_spec_to_eager_specs(steering_spec)
        if not specs:
            return nullcontext()
        return ie_steer(self.eager_model, specs, prompt_token_ids=torch.tensor([prompt_token_ids]))

    def _written_step(
        self,
        logits: torch.Tensor,
        chosen: int,
        settings: SamplingSettings,
        generated_before: Sequence[int],
        index: int,
        source: str,
        end_mark: bool,
    ) -> WrittenStep:
        """A `WrittenStep` from one step's `[vocab]` logits (`step_numbers`), with its word-pieces decoded alone."""
        nums = step_numbers(logits, chosen, settings, generated_before)
        decode = lambda i: self.tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)  # noqa: E731
        step = WrittenStep(
            index=int(index),
            token_id=int(chosen),
            token=decode(chosen),
            model_prob=nums["model_prob"],
            model_logprob=nums["model_logprob"],
            model_rank=nums["model_rank"],
            n_tied=nums["n_tied"],
            chosen_logit=nums["chosen_logit"],
            top=[(i, decode(i), p) for i, p in nums["top"]],
            sampler_prob=nums["sampler_prob"],
            sampler_support=nums["sampler_support"],
            end_mark=bool(end_mark),
            source=source,
        )
        if step.sampler_prob is not None and source == "written" and not step.sampler_prob > 0.0:
            # the mirror disagrees with the sampler that drew it (scripts/check_sampler_mirror.py checks it on the CPU)
            logger.warning("Sampler mirror gave the written word-piece %d at step %d no chance", step.token_id, index)
        return step

    async def capture_for_sweep(
        self,
        prompt: str,
        *,
        steering_spec: Optional[SteeringSpec] = None,
        steering_position_mask: Optional[Sequence[int]] = None,
        max_new_tokens: int = 6,
        sampler: Optional[SamplerSettings] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        answer_token_ids: Optional[Sequence[int]] = None,
        prepend_bos: bool = True,
        on_written: Optional[Callable[[int], Awaitable[bool]]] = None,
    ) -> SweepCapture:
        """Write the answer, then read prompt + answer[:-1] back in one forward, both inside one steering block.

        `sampler` (default: greedy) is passed to interp_engine's `generate_stream` with every knob explicit. With
        `stop_token_ids`, the writing stops after the first of them it writes (`end_of_turn_ids` gives the
        checkpoint's); None keeps interp_engine's own rule (its tokenizer's eos only), as the instrument always had.
        With `answer_token_ids`, nothing is written: those ids are the answer, read back teacher-forced, and their odds
        are taken from the recapture's layer-60 readout. `on_written(i)` is awaited after each written piece; False
        stops the writing ("stopped"). When the temperature is above 0 and no seed is given, one is drawn
        (`seed_drawn`) so the draw can be repeated on this server."""
        assert self.eager_model is not None, "Call runtime.load() first."
        sampler = sampler if sampler is not None else SamplerSettings()
        temperature = float(sampler.temperature)
        greedy = temperature <= 0.0
        top_k = int(sampler.top_k or 0)
        top_p = float(sampler.top_p) if sampler.top_p is not None else 1.0
        penalty = float(sampler.presence_penalty or 0.0)
        resolved = self.eager_model.sampling_settings(
            temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=penalty
        )
        settings: Dict[str, Any] = dict(resolved.as_dict())
        if greedy:
            settings["top_k"] = None
            settings["top_p"] = None
        given = answer_token_ids is not None
        writing = not given and int(max_new_tokens) > 0
        seed: Optional[int] = None if greedy else sampler.seed
        seed_drawn = False
        if writing and not greedy and seed is None:
            seed = secrets.randbelow(2**31)
            seed_drawn = True

        eos = getattr(self.tokenizer, "eos_token_id", None)
        stop_set = {int(t) for t in stop_token_ids} if stop_token_ids is not None else set()
        if eos is not None:
            # interp_engine's loop always stops at its eos too (stop_at_eos=True)
            stop_set.add(int(eos))
        stop_list = (
            list(dict.fromkeys([int(t) for t in stop_token_ids] + ([int(eos)] if eos is not None else [])))
            if stop_token_ids is not None
            else ([int(eos)] if eos is not None else [])
        )

        prompt_token_ids = self.tokenize_prompt(prompt, prepend_bos=prepend_bos)
        prompt_token_strs = [self.tokenizer.decode([tid]) for tid in prompt_token_ids]
        points = self._sweep_points()

        answer: List[int] = []
        answer_strs: List[str] = []
        steps: List[WrittenStep] = []
        stop_reason = "given" if given else "none"
        t0 = time.perf_counter()
        with self._steering_context(prompt_token_ids, steering_spec, steering_position_mask):
            if writing:
                stop_reason = "length"
                gen = ie_generate_stream(
                    self.eager_model,
                    torch.tensor([prompt_token_ids], device=self.eager_model.device),
                    max_tokens=int(max_new_tokens),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=penalty,
                    stop_at_eos=True,
                    n_logprobs=0,
                    seed=seed,
                )
                try:
                    for i, st in enumerate(gen):
                        tid = int(st.token_id)
                        end = tid in stop_set
                        # the step's numbers now, then drop the step: its logits are a view of the prefill's
                        # [1, P, 262144] output and would keep it alive through the recapture
                        steps.append(self._written_step(st.logits, tid, resolved, answer, i, "written", end))
                        answer.append(tid)
                        answer_strs.append(st.token_str)
                        del st
                        keep = True if on_written is None else bool(await on_written(i))
                        if end:
                            stop_reason = "end_mark"
                            break
                        if not keep and i + 1 < int(max_new_tokens):
                            stop_reason = "stopped"
                            break
                finally:
                    gen.close()  # frees the loop's KV cache before the recapture
            elif given:
                answer = [int(t) for t in answer_token_ids]  # type: ignore[union-attr]
                answer_strs = [self.tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in answer]
            t1 = time.perf_counter()
            # The last written piece is never fed back, so it has no state: rows P - 1 .. P + n - 2 are the n
            # places a piece was chosen (as `EagerModel.capture_generation` recaptures).
            caps = await self.eager_model.capture(prompt_token_ids + answer[: max(len(answer) - 1, 0)], points)
            t2 = time.perf_counter()

        if given:
            P = len(prompt_token_ids)
            r_post_59 = caps[Address("resid_post", N_LAYERS - 1)]
            for j, tid in enumerate(answer):
                logits = unembed_gemma4(
                    r_post_59[P - 1 + j].to(self.device), self.final_norm_weight, self.lm_head_weight
                )
                steps.append(self._written_step(logits, tid, resolved, answer[:j], j, "given", tid in stop_set))

        return SweepCapture(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_token_strs=prompt_token_strs,
            answer_token_ids=answer,
            answer_token_strs=answer_strs,
            answer_text="".join(answer_strs),
            steps=steps,
            caps=caps,
            settings=settings,
            greedy=greedy,
            seed=seed,
            seed_drawn=seed_drawn,
            stop_reason=stop_reason,
            stop_token_ids=stop_list,
            answer_source="given" if given else "written",
            max_new_tokens=int(max_new_tokens),
            generate_ms=(t1 - t0) * 1000.0,
            capture_ms=(t2 - t1) * 1000.0,
        )

    def _centre_in_frame_space(self, centre_raw: torch.Tensor, basis_space: str) -> torch.Tensor:
        """`[60, 5376]` raw centre states, moved into the space the frame reads (raw, or h_l J_l^T)."""
        if basis_space == "raw":
            return centre_raw
        return torch.stack(
            [(centre_raw[l].to(self.jpp_lenses[l].dtype) @ self.jpp_lenses[l].T).float() for l in range(N_LAYERS)],
            dim=0,
        )

    async def prepare_sweep(
        self,
        cap: SweepCapture,
        *,
        monument_concepts: Tuple[Union[str, int], Union[str, int], Union[str, int]],
        survey_mode: str = "monument",
        locked_pole: Optional[Tuple[int, int, float]] = None,
        tracked_concepts: Optional[Sequence[str]] = DEFAULT_TRACKED_CONCEPTS,
        frame_squaring: str = "gram_schmidt",
        frame_per_layer: bool = False,
        frame_fnw: bool = False,
        centre_prompts: Optional[Sequence[str]] = None,
        centre_token_ids: Optional[Sequence[Sequence[int]]] = None,
        centre_answer: bool = False,
        prepend_bos: bool = True,
        frame_override: Optional[Tuple[torch.Tensor, str, Dict[str, Any], Tuple[str, str, str]]] = None,
        centre_override: Optional[torch.Tensor] = None,
    ) -> SweepShared:
        """What every sighting of `cap` shares: the tracked words' ids, the survey frame and the matched controls.

        A monument frame is built here (it reads no position). A pole-motif frame (`survey_mode="pole_motif"` with a
        `locked_pole`) needs a position's poles and threads, so `analyse_position` builds it at the first position it
        analyses and keeps it for the rest. `frame_override` = (basis, basis_space, frame_info, survey_labels) and
        `centre_override` (a `[60, 5376]` centre in that space) reuse another sweep's frame and centre (the thinking
        comparison is read in the main sweep's).

        Controls: the texts `centre_prompts` (tokenized as the prompt is) and the exact ids `centre_token_ids`. Each
        is captured alone, unsteered, and the mean of their states at their own last piece centres token 0, as the
        instrument always did. With `centre_answer`, token k >= 1 is centred on the controls reading this very
        answer: control_ids + answer[:-1], row len(control_ids) - 1 + k ("control_reading_same_answer"), captured on
        first need. Without it, a position past token 0 is not centred: a control's state at its own last piece is
        never subtracted from a state read mid-answer."""
        assert self.eager_model is not None, "Call runtime.load() first."
        if frame_squaring not in FRAME_SQUARINGS:
            raise ValueError(f"Unknown frame_squaring {frame_squaring!r}; expected one of {FRAME_SQUARINGS}")
        t_start = time.perf_counter()

        # Resolve tracked concepts to token IDs
        tracked_map: Dict[str, int] = {}
        if tracked_concepts:
            for c_str in tracked_concepts:
                try:
                    tid, _ = self.resolve_concept_token_id(c_str)
                    tracked_map[c_str] = tid
                except ValueError:
                    pass

        # Module 4's survey frame (Gram-Schmidt or Löwdin); the pole-motif one waits for its first position
        basis: Optional[torch.Tensor] = None
        basis_space = "J"
        survey_labels: Optional[Tuple[str, str, str]] = None
        frame_info: Dict[str, Any] = {"squaring": frame_squaring, "per_layer": False, "fnw_weighted": False}
        if frame_override is not None:
            basis, basis_space, override_info, survey_labels = frame_override
            frame_info = dict(override_info)
        elif not (survey_mode == "pole_motif" and locked_pole is not None):
            t1_id, t1_str = self.resolve_concept_token_id(monument_concepts[0])
            t2_id, t2_str = self.resolve_concept_token_id(monument_concepts[1])
            t3_id, t3_str = self.resolve_concept_token_id(monument_concepts[2])
            global_basis_J, layer_pullback_basis = build_monument_frame(
                token_ids=(t1_id, t2_id, t3_id),
                lm_head_weight=self.lm_head_weight,
                jpp_lenses=self.jpp_lenses,
                pullback_per_layer=bool(frame_per_layer),
                squaring=frame_squaring,
                final_norm_weight=self.final_norm_weight if frame_fnw else None,
            )
            if frame_per_layer:
                basis = layer_pullback_basis
                basis_space = "raw"
            else:
                basis = global_basis_J
            survey_labels = (t1_str.strip() or str(monument_concepts[0]), t2_str.strip() or str(monument_concepts[1]), t3_str.strip() or str(monument_concepts[2]))
            rows = self.lm_head_weight[torch.tensor([t1_id, t2_id, t3_id], device=self.lm_head_weight.device)]
            frame_info.update(
                {
                    "per_layer": bool(frame_per_layer),
                    "fnw_weighted": bool(frame_fnw),
                    "token_ids": [int(t1_id), int(t2_id), int(t3_id)],
                    "tokens": [t1_str, t2_str, t3_str],
                    "geometry": frame_word_geometry(rows, self.final_norm_weight),
                }
            )
        if frame_override is None:
            frame_info["state"] = "h_raw" if basis_space == "raw" else "h_J"

        # Matched controls: their mean state at their own last position, in the frame's space (REPORT §3.5)
        control_ids: List[List[int]] = [
            self.tokenize_prompt(c_text, prepend_bos=prepend_bos) for c_text in (centre_prompts or []) if c_text and c_text.strip()
        ]
        for ids in centre_token_ids or []:
            ids = [int(t) for t in ids]
            if not ids:
                raise ValueError("A matched control given as word-piece ids has none: each needs at least one.")
            control_ids.append(ids)
        centre_k0 = centre_override
        if centre_override is None and control_ids:
            centre_raw = torch.zeros(N_LAYERS, D_MODEL, dtype=torch.float32, device=self.device)
            for c_ids in control_ids:
                c_caps = await self.eager_model.capture(c_ids, [Address("resid_post", l) for l in range(N_LAYERS)])
                for l in range(N_LAYERS):
                    centre_raw[l] += c_caps[Address("resid_post", l)][len(c_ids) - 1].to(self.device).float()
                del c_caps
            centre_raw /= len(control_ids)
            centre_k0 = self._centre_in_frame_space(centre_raw, basis_space)

        return SweepShared(
            tracked_map=tracked_map,
            survey_mode=survey_mode,
            locked_pole=locked_pole,
            frame_squaring=frame_squaring,
            basis=basis,
            basis_space=basis_space,
            frame_info=frame_info,
            survey_labels=survey_labels,
            control_ids=control_ids,
            centre_k0=centre_k0,
            centre_answer=bool(centre_answer),
            prepare_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    async def _answer_centre(self, cap: SweepCapture, shared: SweepShared, token_index: int) -> torch.Tensor:
        """The centre for token k >= 1: the controls reading this very answer (control_ids + answer[:-1], unsteered),
        their mean state at row len(control_ids) - 1 + k, in the frame's space. One forward per control, on first
        need, kept in `shared`."""
        assert self.eager_model is not None, "Call runtime.load() first."
        n = len(cap.answer_token_ids)
        if shared.answer_centre_raw is None:
            acc = torch.zeros(max(n - 1, 0), N_LAYERS, D_MODEL, dtype=torch.float32)
            tail = cap.answer_token_ids[: max(n - 1, 0)]
            for c_ids in shared.control_ids:
                c_caps = await self.eager_model.capture(
                    c_ids + tail, [Address("resid_post", l) for l in range(N_LAYERS)]
                )
                for l in range(N_LAYERS):
                    acc[:, l] += c_caps[Address("resid_post", l)][len(c_ids) : len(c_ids) + n - 1].float()
                del c_caps
            shared.answer_centre_raw = acc / len(shared.control_ids)
        centre_raw = shared.answer_centre_raw[int(token_index) - 1].to(self.device)
        return self._centre_in_frame_space(centre_raw, shared.basis_space)

    async def analyse_position(
        self,
        cap: SweepCapture,
        shared: SweepShared,
        token_index: int,
        *,
        top_k_poles: int = 15,
        decode_all_pole_tokens: bool = False,
        pole_ranking: str = "activation",
        thread_score: str = "coupling",
        rank_lookahead: bool = True,
    ) -> TheodoliteSweepBundle:
        """One sighting: all four modules at read position P - 1 + token_index, the stream that has just read the
        prompt's last piece (token 0) or answer[k - 1], and whose layer 60 chose answer[k].

        `rank_lookahead` is the space-then-digit rule: when the piece chosen here decodes blank (Gemma writes a digit
        after a lone space) and the next position was read, each tracked word's J++ rank, logit and prob is taken
        from the next position where it ranks better there (recorded in `tracked_from_next`). At token 0 this is the
        rule the instrument always applied."""
        assert self.eager_model is not None, "Call runtime.load() first."
        if pole_ranking not in POLE_RANKINGS:
            raise ValueError(f"Unknown pole_ranking {pole_ranking!r}; expected one of {POLE_RANKINGS}")
        if thread_score not in THREAD_SCORES:
            raise ValueError(f"Unknown thread_score {thread_score!r}; expected one of {THREAD_SCORES}")
        t_start = time.perf_counter()

        caps = cap.caps
        k = int(token_index)
        n = len(cap.answer_token_ids)
        if not 0 <= k < max(n, 1):
            raise ValueError(
                f"There is no word-piece {k} to sight: the answer has {n} word-piece(s), numbered from 0."
            )
        P = len(cap.prompt_token_ids)
        target_pos = P - 1 + k  # the read position
        n_rows = int(caps[Address("resid_post", 0)].shape[0])
        read_token_id = int((cap.prompt_token_ids + cap.answer_token_ids)[target_pos])
        chosen = cap.steps[k] if k < len(cap.steps) else None
        tracked_map = shared.tracked_map
        lookahead = bool(
            rank_lookahead
            and k < n
            and self.tokenizer.decode([cap.answer_token_ids[k]]).strip() == ""
            and n_rows > target_pos + 1
        )

        # 4. Execute Module 1 (Attn vs FFN Causal Split), J++ Readouts, and Module 2 (Bipolar Poles)
        causal_splits: Dict[int, CausalSplitResult] = {}
        bipolar_poles: Dict[int, BipolarPolesResult] = {}
        jpp_layer_readouts: Dict[int, SweepLayerTokenReadout] = {}
        naive_layer_readouts: Dict[int, SweepLayerTokenReadout] = {}
        pole_top_tokens_jpp: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
        pole_top_tokens_naive: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
        tracked_from_next: Dict[int, Dict[str, bool]] = {}

        resid_post_target_list: List[torch.Tensor] = []
        resid_post_J_target_list: List[torch.Tensor] = []
        delta_attn_J_target_list: List[torch.Tensor] = []
        delta_ffn_J_target_list: List[torch.Tensor] = []
        delta_attn_raw_list: List[torch.Tensor] = []
        delta_ffn_raw_list: List[torch.Tensor] = []
        seq_states_list: List[torch.Tensor] = []
        mlp_out_target: Dict[int, torch.Tensor] = {}
        resid_mid_target: Dict[int, torch.Tensor] = {}
        pole_write_sizes: Dict[int, List[float]] = {}
        poles_ffn_explained: Dict[int, Optional[float]] = {}
        layer_scalars = torch.zeros(N_LAYERS, dtype=torch.float32)
        readout_60: Optional[torch.Tensor] = None

        for l in range(N_LAYERS):
            J_l = self.jpp_lenses[l]
            # rows are picked on the CPU and only they move to the GPU (the same values the whole sequence held)
            r_post_cap = caps[Address("resid_post", l)]
            r_pre_t = caps[Address("resid_pre", l)][target_pos].to(self.device)
            r_mid_t = caps[Address("resid_mid", l)][target_pos].to(self.device)
            r_post_t = r_post_cap[target_pos].to(self.device)
            mlp_act_t = caps[Address("mlp_act", l)][target_pos].to(self.device)
            mlp_out_cap = caps.get(Address("mlp_out", l))

            seq_states_list.append(r_post_cap[: target_pos + 1].to(self.device).float())

            # Module 1: Causal Vector Split (Gemma-4 multiplies every layer output by layer_scalar)
            s_l = self._decoder_layers[l].layer_scalar.detach().float()
            split_l = compute_causal_vector_split(r_pre_t, r_mid_t, r_post_t, J_l, layer_scalar=s_l)
            causal_splits[l] = split_l
            s_val = float(s_l.reshape(-1)[0])
            layer_scalars[l] = s_val
            ffn_write_raw = r_post_t.float() - s_val * r_mid_t.float()
            delta_attn_raw_list.append(s_val * (r_mid_t.float() - r_pre_t.float()))
            delta_ffn_raw_list.append(ffn_write_raw)
            resid_mid_target[l] = r_mid_t.float()
            if mlp_out_cap is not None:
                mlp_out_target[l] = mlp_out_cap[target_pos].to(self.device).float()

            resid_post_target_list.append(r_post_t)
            resid_post_J_target_list.append(split_l.resid_post_J)
            delta_attn_J_target_list.append(split_l.delta_attn_J)
            delta_ffn_J_target_list.append(split_l.delta_ffn_J)

            # Layer l J++ Readout & Naive Logit Lens Readout
            jpp_logits = unembed_gemma4(split_l.resid_post_J, self.final_norm_weight, self.lm_head_weight)
            if l == N_LAYERS - 1:
                readout_60 = jpp_logits
            ro_jpp = self.decode_top_tokens(
                jpp_logits, top_k=5, tracked_token_ids=tracked_map, layer_idx=l
            )
            from_next = {lbl: False for lbl in ro_jpp.tracked_token_ranks}
            # In SentencePiece (Gemma-4), digits ('8', '6') are emitted after a standalone space token (' ').
            # When the model emits ' ' at target_pos, evaluate digit readout at target_pos + 1 as well.
            if lookahead:
                r_post_next_J = r_post_cap[target_pos + 1].to(self.device).to(dtype=J_l.dtype) @ J_l.T
                jpp_logits_next = unembed_gemma4(r_post_next_J, self.final_norm_weight, self.lm_head_weight)
                ro_next = self.decode_top_tokens(
                    jpp_logits_next, top_k=5, tracked_token_ids=tracked_map, layer_idx=l
                )
                for lbl in ro_jpp.tracked_token_ranks:
                    if ro_next.tracked_token_ranks[lbl] < ro_jpp.tracked_token_ranks[lbl]:
                        ro_jpp.tracked_token_ranks[lbl] = ro_next.tracked_token_ranks[lbl]
                        ro_jpp.tracked_token_logits[lbl] = ro_next.tracked_token_logits[lbl]
                        ro_jpp.tracked_token_probs[lbl] = ro_next.tracked_token_probs[lbl]
                        from_next[lbl] = True
            jpp_layer_readouts[l] = ro_jpp
            tracked_from_next[l] = from_next

            naive_logits = unembed_gemma4(r_post_t, self.final_norm_weight, self.lm_head_weight)
            naive_layer_readouts[l] = self.decode_top_tokens(
                naive_logits, top_k=5, tracked_token_ids=tracked_map, layer_idx=l
            )

            # Module 2: Bipolar GeGLU Neuron Poles
            compute_pole_logits_here = bool(
                decode_all_pole_tokens or (l in GLOBAL_ATTN_LAYERS_0BASED) or (l >= N_LAYERS - 6)
            )
            poles_l = extract_bipolar_poles(
                mlp_act_l=mlp_act_t,
                w_down_slice_fn=self.slice_w_down,
                J_l=J_l,
                layer_idx=l,
                k=top_k_poles,
                final_norm_weight=self.final_norm_weight,
                lm_head_weight=self.lm_head_weight,
                post_ffn_norm_weight=self._post_ffn_norm_weights.get(l),
                compute_logits=compute_pole_logits_here,
                compute_naive_logits=compute_pole_logits_here,
                write_col_norms=self.write_col_norms(l) if pole_ranking == "write" else None,
            )
            bipolar_poles[l] = poles_l
            if l in mlp_out_target:
                w_sizes, explained = pole_write_shares(poles_l, mlp_out_target[l], ffn_write_raw, s_val)
                pole_write_sizes[l] = [float(v) for v in w_sizes.cpu().tolist()]
                poles_ffn_explained[l] = explained
            else:
                poles_ffn_explained[l] = None

            if poles_l.pole_logits is not None:
                n_decode_ranks = min(5 if not decode_all_pole_tokens else top_k_poles, top_k_poles)
                for r_idx in range(n_decode_ranks):
                    p_logits = poles_l.pole_logits[r_idx]
                    top_v, top_i = torch.topk(p_logits, k=3)
                    pole_top_tokens_jpp[(l, r_idx)] = [
                        (self.tokenizer.decode([int(tid)]), float(val))
                        for val, tid in zip(top_v.cpu().tolist(), top_i.cpu().tolist())
                    ]
                    if poles_l.naive_pole_logits is not None:
                        np_logits = poles_l.naive_pole_logits[r_idx]
                        ntop_v, ntop_i = torch.topk(np_logits, k=3)
                        pole_top_tokens_naive[(l, r_idx)] = [
                            (self.tokenizer.decode([int(tid)]), float(val))
                            for val, tid in zip(ntop_v.cpu().tolist(), ntop_i.cpu().tolist())
                        ]

        # 5. Execute Module 3: Live Downstream Reader Circuit Filaments (C, plus the direct-path estimates)
        direct_inputs = (
            DirectPathInputs(
                w_down_slice_fn=self.slice_w_down,
                w_up_slice_fn=self.slice_w_up,
                mlp_out=mlp_out_target,
                resid_mid=resid_mid_target,
                layer_scalars=layer_scalars,
                post_ffn_norm_weights=self._post_ffn_norm_weights,
                pre_ffn_norm_weights=self._pre_ffn_norm_weights,
            )
            if len(mlp_out_target) == N_LAYERS
            else None
        )
        filaments = compute_downstream_filaments(
            poles_by_layer=bipolar_poles,
            w_gate_slice_fn=self.slice_w_gate,
            max_layer_hop=6,
            top_k_per_source_layer=6,
            min_abs_coupling=0.05 if thread_score == "coupling" else 0.0,
            pre_ffn_norm_weights=self._pre_ffn_norm_weights,
            score=thread_score,
            direct=direct_inputs,
        )

        # 6. Execute Module 4: Projective Theodolite Triangulation (survey frame: Gram-Schmidt or Löwdin). A pole-motif
        #    frame's third axis is the locked pole's dominant reader at this position: built once, at the first
        #    position analysed, and kept, so every word-piece of the answer is read in one frame.
        if shared.basis is None:
            assert shared.locked_pole is not None
            p_layer, p_neuron, p_pol = shared.locked_pole
            n_idx_t = torch.tensor([p_neuron], device=self.device, dtype=torch.long)
            w_gate_v = self.slice_w_gate(p_layer, n_idx_t)[0]
            w_down_v = self.slice_w_down(p_layer, n_idx_t)[0]
            next_l = min(p_layer + 1, N_LAYERS - 1)
            dom_dst_neuron = int(bipolar_poles[next_l].neuron_indices[0].item())
            for fil in filaments:
                if fil.src_layer == p_layer and fil.src_neuron == p_neuron:
                    next_l = fil.dst_layer
                    dom_dst_neuron = fil.dst_neuron
                    break
            dst_gate_v = self.slice_w_gate(
                next_l, torch.tensor([dom_dst_neuron], device=self.device, dtype=torch.long)
            )[0]
            shared.basis = build_pole_motif_frame(
                layer_idx=p_layer,
                neuron_idx=p_neuron,
                polarity=p_pol,
                w_gate_vec=w_gate_v,
                w_down_vec=w_down_v,
                J_l=self.jpp_lenses[p_layer],
                downstream_gate_vec=dst_gate_v,
                squaring=shared.frame_squaring,
            )
            pol_char = "+" if p_pol >= 0 else "-"
            shared.survey_labels = (
                f"Gate L{p_layer}:N{p_neuron}",
                f"Write L{p_layer}:N{p_neuron}({pol_char})@J++",
                f"Reader L{next_l}:N{dom_dst_neuron}",
            )
            shared.frame_info["reader"] = {"layer": int(next_l), "neuron": int(dom_dst_neuron), "from_token_index": k}
        basis = shared.basis
        basis_space = shared.basis_space
        survey_labels = shared.survey_labels
        assert survey_labels is not None

        resid_post_layers_t = torch.stack(resid_post_target_list, dim=0)        # [60, 5376]
        resid_post_J_layers_t = torch.stack(resid_post_J_target_list, dim=0)    # [60, 5376]
        delta_attn_J_layers_t = torch.stack(delta_attn_J_target_list, dim=0)    # [60, 5376]
        delta_ffn_J_layers_t = torch.stack(delta_ffn_J_target_list, dim=0)      # [60, 5376]

        # Matched-control centre, in the frame's space: at token 0 the controls' own last piece; past it, the
        # controls reading this very answer when asked for (`centre_answer`), else none
        centre_states: Optional[torch.Tensor] = None
        centre_kind = "none"
        if k == 0:
            centre_states = shared.centre_k0
            centre_kind = "control_prompt_end" if centre_states is not None else "none"
        elif shared.centre_answer and shared.control_ids:
            centre_states = await self._answer_centre(cap, shared, k)
            centre_kind = "control_reading_same_answer"
        frame_info = dict(shared.frame_info)
        frame_info["centred"] = centre_states is not None
        frame_info["n_centre_prompts"] = len(shared.control_ids)
        frame_info["centre_kind"] = centre_kind

        trajectory_3d = project_trajectory_to_3d(
            resid_post_layers=resid_post_layers_t,
            resid_post_J_layers=resid_post_J_layers_t,
            delta_attn_J_layers=delta_attn_J_layers_t,
            delta_ffn_J_layers=delta_ffn_J_layers_t,
            basis=basis,
            use_jpp_states=(basis_space == "J"),
            centre_states=centre_states,
            delta_attn_raw_layers=torch.stack(delta_attn_raw_list, dim=0),
            delta_ffn_raw_layers=torch.stack(delta_ffn_raw_list, dim=0),
        )

        # 7. Compute 60x60 Linear CKA Similarity Matrix (over the word-pieces read up to here)
        seq_states_t = torch.stack(seq_states_list, dim=0)  # [60, n_tokens, 5376]
        cka_matrix = compute_linear_cka_matrix(seq_states_t)

        # The reading forward against the writing one, at layer 60 (J = I): the readout's top piece and its logit for
        # the chosen piece, less the logit the writing chose it at. A given piece was never written here: its odds
        # are this very readout's (capture_for_sweep), so a diff would be 0 by construction and is sent as None.
        recapture_check: Optional[Dict[str, Any]] = None
        if chosen is not None and readout_60 is not None:
            top1_id = int(torch.argmax(readout_60).item())
            written = chosen.source == "written"
            chosen_logit_recapture = float(readout_60[chosen.token_id].item()) if written else None
            recapture_check = {
                "top1_id": top1_id,
                "top1_token": self.tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                "top1_is_chosen": top1_id == chosen.token_id,
                "chosen_logit_recapture": chosen_logit_recapture,
                "chosen_logit_diff": (
                    chosen_logit_recapture - chosen.chosen_logit if chosen_logit_recapture is not None else None
                ),
            }

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0.0
        )
        analyse_ms = (time.perf_counter() - t_start) * 1000.0

        return TheodoliteSweepBundle(
            prompt=cap.prompt,
            prompt_token_ids=cap.prompt_token_ids,
            prompt_token_strs=cap.prompt_token_strs,
            target_pos=target_pos,
            generated_text=cap.answer_text,
            generated_token_ids=cap.answer_token_ids,
            trajectory_3d=trajectory_3d,
            causal_splits=causal_splits,
            bipolar_poles=bipolar_poles,
            filaments=filaments,
            cka_matrix=cka_matrix,
            jpp_layer_readouts=jpp_layer_readouts,
            naive_layer_readouts=naive_layer_readouts,
            pole_top_tokens_jpp=pole_top_tokens_jpp,
            pole_top_tokens_naive=pole_top_tokens_naive,
            survey_mode=shared.survey_mode,
            survey_labels=survey_labels,
            peak_vram_gb=peak_vram_gb,
            elapsed_ms=cap.generate_ms + cap.capture_ms + shared.prepare_ms + analyse_ms,
            survey_basis=basis,
            survey_basis_space=basis_space,
            frame_info=frame_info,
            pole_write_sizes=pole_write_sizes,
            poles_ffn_explained=poles_ffn_explained,
            pole_ranking=pole_ranking,
            thread_score=thread_score,
            token_index=k,
            read_token_id=read_token_id,
            read_token=self.tokenizer.decode([read_token_id], clean_up_tokenization_spaces=False),
            chosen=chosen,
            recapture_check=recapture_check,
            kinship_n_states=target_pos + 1,
            centre_kind=centre_kind,
            frame_reader=shared.frame_info.get("reader"),
            tracked_from_next=tracked_from_next,
            rank_lookahead=bool(rank_lookahead),
            analyse_ms=analyse_ms,
        )

    async def run_theodolite_sweep(
        self,
        prompt: str,
        *,
        monument_concepts: Tuple[Union[str, int], Union[str, int], Union[str, int]] = ("spider", "eight", "web"),
        survey_mode: str = "monument",
        locked_pole: Optional[Tuple[int, int, float]] = None,
        steering_spec: Optional[SteeringSpec] = None,
        max_new_tokens: int = 6,
        top_k_poles: int = 15,
        tracked_concepts: Optional[Sequence[str]] = DEFAULT_TRACKED_CONCEPTS,
        decode_all_pole_tokens: bool = False,
        prepend_bos: bool = True,
        steering_position_mask: Optional[Sequence[int]] = None,
        pole_ranking: str = "activation",
        thread_score: str = "coupling",
        frame_squaring: str = "gram_schmidt",
        frame_per_layer: bool = False,
        frame_fnw: bool = False,
        centre_prompts: Optional[Sequence[str]] = None,
        sampler: Optional[SamplerSettings] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> TheodoliteSweepBundle:
        """Run a full 60-layer Theodolite capture & generation sweep via `interp-engine`, read at the prompt's last
        word-piece (token 0): `capture_for_sweep` + `prepare_sweep` + `analyse_position(0)`.

        Captures standardized hook addresses across all 60 layers:
          - `Address("resid_pre", l)`
          - `Address("resid_mid", l)`
          - `Address("mlp_act", l)`
          - `Address("mlp_out", l)` (y = W_down a, before the post-FFN norm; for write sizes and direct paths)
          - `Address("resid_post", l)`
        and executes all 4 mathematical modules in `backend/theodolite_math.py`.

        Options (each default reproduces the instrument as first built):
          steering_position_mask: prompt positions the steering leaves alone (e.g. [0], the start mark).
          pole_ranking: "activation" (|a|) or "write" (|a| * ||w_down (.) w_post||).
          thread_score: "coupling" (C) or "direct" (the direct-path estimate) for selecting the filaments.
          frame_squaring: "gram_schmidt" (in order) or "lowdin" (all at once) for the survey frame.
          frame_per_layer: square the word frame afresh at each layer from W_U rows pulled back by J_l, and read
              the raw h_l in it (word frames only).
          frame_fnw: build the word frame from the final-norm weighted rows fnw * W_U (word frames only).
          centre_prompts: exact texts of matched controls; the mean of their states at their last position is
              subtracted from every layer's state before it is projected (REPORT §3.5). Captured unsteered.
          sampler: how the continuation is written (default: greedy, as always; see `SamplerSettings`).
          stop_token_ids: stop writing after any of these (e.g. `end_of_turn_ids()`); None keeps interp_engine's own
              rule (its tokenizer's eos only).
        """
        assert self.eager_model is not None, "Call runtime.load() first."
        if pole_ranking not in POLE_RANKINGS:
            raise ValueError(f"Unknown pole_ranking {pole_ranking!r}; expected one of {POLE_RANKINGS}")
        if thread_score not in THREAD_SCORES:
            raise ValueError(f"Unknown thread_score {thread_score!r}; expected one of {THREAD_SCORES}")
        if frame_squaring not in FRAME_SQUARINGS:
            raise ValueError(f"Unknown frame_squaring {frame_squaring!r}; expected one of {FRAME_SQUARINGS}")
        t_start = time.perf_counter()
        cap = await self.capture_for_sweep(
            prompt,
            steering_spec=steering_spec,
            steering_position_mask=steering_position_mask,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            stop_token_ids=stop_token_ids,
            prepend_bos=prepend_bos,
        )
        shared = await self.prepare_sweep(
            cap,
            monument_concepts=monument_concepts,
            survey_mode=survey_mode,
            locked_pole=locked_pole,
            tracked_concepts=tracked_concepts,
            frame_squaring=frame_squaring,
            frame_per_layer=frame_per_layer,
            frame_fnw=frame_fnw,
            centre_prompts=centre_prompts,
            prepend_bos=prepend_bos,
        )
        bundle = await self.analyse_position(
            cap,
            shared,
            0,
            top_k_poles=top_k_poles,
            decode_all_pole_tokens=decode_all_pole_tokens,
            pole_ranking=pole_ranking,
            thread_score=thread_score,
        )
        bundle.elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return bundle

    async def iter_theodolite_sweeps(
        self,
        prompt: str,
        *,
        token_indices: Optional[Sequence[int]] = None,
        max_sighted: int = 64,
        monument_concepts: Tuple[Union[str, int], Union[str, int], Union[str, int]] = ("spider", "eight", "web"),
        survey_mode: str = "monument",
        locked_pole: Optional[Tuple[int, int, float]] = None,
        steering_spec: Optional[SteeringSpec] = None,
        max_new_tokens: int = 6,
        top_k_poles: int = 15,
        tracked_concepts: Optional[Sequence[str]] = DEFAULT_TRACKED_CONCEPTS,
        decode_all_pole_tokens: bool = False,
        prepend_bos: bool = True,
        steering_position_mask: Optional[Sequence[int]] = None,
        pole_ranking: str = "activation",
        thread_score: str = "coupling",
        frame_squaring: str = "gram_schmidt",
        frame_per_layer: bool = False,
        frame_fnw: bool = False,
        centre_prompts: Optional[Sequence[str]] = None,
        centre_token_ids: Optional[Sequence[Sequence[int]]] = None,
        centre_answer: bool = False,
        sampler: Optional[SamplerSettings] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        answer_token_ids: Optional[Sequence[int]] = None,
        rank_lookahead: bool = True,
    ) -> AsyncIterator[TheodoliteSweepBundle]:
        """For scripts: write once, then yield a sighting for each of `token_indices` (default: every written
        word-piece, the first `max_sighted`; token 0 alone when nothing was written), all read in one frame."""
        cap = await self.capture_for_sweep(
            prompt,
            steering_spec=steering_spec,
            steering_position_mask=steering_position_mask,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            stop_token_ids=stop_token_ids,
            answer_token_ids=answer_token_ids,
            prepend_bos=prepend_bos,
        )
        shared = await self.prepare_sweep(
            cap,
            monument_concepts=monument_concepts,
            survey_mode=survey_mode,
            locked_pole=locked_pole,
            tracked_concepts=tracked_concepts,
            frame_squaring=frame_squaring,
            frame_per_layer=frame_per_layer,
            frame_fnw=frame_fnw,
            centre_prompts=centre_prompts,
            centre_token_ids=centre_token_ids,
            centre_answer=centre_answer,
            prepend_bos=prepend_bos,
        )
        n = len(cap.answer_token_ids)
        order = list(token_indices) if token_indices is not None else list(range(max(1, min(n, int(max_sighted)))))
        for k in order:
            yield await self.analyse_position(
                cap,
                shared,
                int(k),
                top_k_poles=top_k_poles,
                decode_all_pole_tokens=decode_all_pole_tokens,
                pole_ranking=pole_ranking,
                thread_score=thread_score,
                rank_lookahead=rank_lookahead,
            )
