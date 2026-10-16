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
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from interp_engine import Address, EagerModel
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, OrthogonalDecompSpec, SteeringSpec
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from backend.theodolite_math import (
    BipolarPolesResult,
    CausalSplitResult,
    CircuitFilament,
    D_FF,
    D_MODEL,
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
    jpp_readout_logits,
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

        self._decoder_layers: Optional[nn.ModuleList] = None
        self._final_norm_weight: Optional[torch.Tensor] = None
        self._lm_head_weight: Optional[torch.Tensor] = None
        self._pre_ffn_norm_weights: Dict[int, torch.Tensor] = {}
        self._post_ffn_norm_weights: Dict[int, torch.Tensor] = {}

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
        tracked_concepts: Optional[Sequence[str]] = ("spider", "arachnid", "eight", "8", "insect", "six", "6"),
        decode_all_pole_tokens: bool = False,
        prepend_bos: bool = True,
    ) -> TheodoliteSweepBundle:
        """Run a full 60-layer Theodolite capture & generation sweep via `interp-engine`.

        Captures standardized hook addresses across all 60 layers:
          - `Address("resid_pre", l)`
          - `Address("resid_mid", l)`
          - `Address("mlp_act", l)`
          - `Address("resid_post", l)`
        and executes all 4 mathematical modules in `backend/theodolite_math.py`.
        """
        assert self.eager_model is not None, "Call runtime.load() first."
        t_start = time.perf_counter()

        # 1. Tokenize prompt. Chat-template strings already begin with "<bos>"; prepending another one gives
        #    ids [2, 2, ...] (double BOS). Raw text still needs exactly one BOS (prepend_bos=False on raw text
        #    yields NO BOS and destroys Gemma's output), so only skip the prepend when a BOS is already present.
        bos_str = getattr(self.tokenizer, "bos_token", None) or "<bos>"
        has_bos = prompt.startswith(bos_str)
        token_tensor = self.eager_model.to_tokens(prompt, prepend_bos=bool(prepend_bos and not has_bos))
        prompt_token_ids = [int(t) for t in token_tensor[0].tolist()]
        prompt_token_strs = [self.tokenizer.decode([tid]) for tid in prompt_token_ids]
        target_pos = len(prompt_token_ids) - 1

        # 2. Build all 240 standardized interp-engine hook addresses across L=60 layers
        points: List[Address] = []
        for l in range(N_LAYERS):
            points.append(Address("resid_pre", l))
            points.append(Address("resid_mid", l))
            points.append(Address("mlp_act", l))
            points.append(Address("resid_post", l))

        # 3. Capture activations over prompt (and generate continuation if max_new_tokens > 0)
        if max_new_tokens > 0:
            completion, caps = await self.eager_model.capture_generation(
                prompt_token_ids,
                points,
                max_tokens=max_new_tokens,
                temperature=0.0,
                steering_spec=steering_spec,
            )
            generated_text = completion.text
            generated_token_ids = [int(t) for t in completion.token_ids]
        else:
            generated_text = ""
            generated_token_ids = []
            caps = await self.eager_model.capture(
                prompt_token_ids,
                points,
                steering_spec=steering_spec,
            )

        has_leading_space_prefix = bool(
            len(generated_token_ids) >= 2
            and self.tokenizer.decode([generated_token_ids[0]]).strip() == ""
        )

        # Resolve tracked concepts to token IDs
        tracked_map: Dict[str, int] = {}
        if tracked_concepts:
            for c_str in tracked_concepts:
                try:
                    tid, _ = self.resolve_concept_token_id(c_str)
                    tracked_map[c_str] = tid
                except ValueError:
                    pass

        # 4. Execute Module 1 (Attn vs FFN Causal Split), J++ Readouts, and Module 2 (Bipolar Poles)
        causal_splits: Dict[int, CausalSplitResult] = {}
        bipolar_poles: Dict[int, BipolarPolesResult] = {}
        jpp_layer_readouts: Dict[int, SweepLayerTokenReadout] = {}
        naive_layer_readouts: Dict[int, SweepLayerTokenReadout] = {}
        pole_top_tokens_jpp: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
        pole_top_tokens_naive: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}

        resid_post_target_list: List[torch.Tensor] = []
        resid_post_J_target_list: List[torch.Tensor] = []
        delta_attn_J_target_list: List[torch.Tensor] = []
        delta_ffn_J_target_list: List[torch.Tensor] = []
        seq_states_list: List[torch.Tensor] = []

        for l in range(N_LAYERS):
            J_l = self.jpp_lenses[l]
            r_pre_seq = caps[Address("resid_pre", l)].to(self.device)
            r_mid_seq = caps[Address("resid_mid", l)].to(self.device)
            r_post_seq = caps[Address("resid_post", l)].to(self.device)
            mlp_act_seq = caps[Address("mlp_act", l)].to(self.device)

            seq_states_list.append(r_post_seq[: target_pos + 1].float())

            r_pre_t = r_pre_seq[target_pos]
            r_mid_t = r_mid_seq[target_pos]
            r_post_t = r_post_seq[target_pos]
            mlp_act_t = mlp_act_seq[target_pos]

            # Module 1: Causal Vector Split (Gemma-4 multiplies every layer output by layer_scalar)
            s_l = self._decoder_layers[l].layer_scalar.detach().float()
            split_l = compute_causal_vector_split(r_pre_t, r_mid_t, r_post_t, J_l, layer_scalar=s_l)
            causal_splits[l] = split_l

            resid_post_target_list.append(r_post_t)
            resid_post_J_target_list.append(split_l.resid_post_J)
            delta_attn_J_target_list.append(split_l.delta_attn_J)
            delta_ffn_J_target_list.append(split_l.delta_ffn_J)

            # Layer l J++ Readout & Naive Logit Lens Readout
            jpp_logits = unembed_gemma4(split_l.resid_post_J, self.final_norm_weight, self.lm_head_weight)
            ro_jpp = self.decode_top_tokens(
                jpp_logits, top_k=5, tracked_token_ids=tracked_map, layer_idx=l
            )
            # In SentencePiece (Gemma-4), digits ('8', '6') are emitted after a standalone space token (' ').
            # When the model emits ' ' at target_pos, evaluate digit readout at target_pos + 1 as well.
            if has_leading_space_prefix and r_post_seq.shape[0] > target_pos + 1:
                r_post_next_J = r_post_seq[target_pos + 1].to(dtype=J_l.dtype) @ J_l.T
                jpp_logits_next = unembed_gemma4(r_post_next_J, self.final_norm_weight, self.lm_head_weight)
                ro_next = self.decode_top_tokens(
                    jpp_logits_next, top_k=5, tracked_token_ids=tracked_map, layer_idx=l
                )
                for lbl in ro_jpp.tracked_token_ranks:
                    if ro_next.tracked_token_ranks[lbl] < ro_jpp.tracked_token_ranks[lbl]:
                        ro_jpp.tracked_token_ranks[lbl] = ro_next.tracked_token_ranks[lbl]
                        ro_jpp.tracked_token_logits[lbl] = ro_next.tracked_token_logits[lbl]
                        ro_jpp.tracked_token_probs[lbl] = ro_next.tracked_token_probs[lbl]
            jpp_layer_readouts[l] = ro_jpp

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
            )
            bipolar_poles[l] = poles_l

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

        # 5. Execute Module 3: Live Downstream Reader Circuit Filaments
        filaments = compute_downstream_filaments(
            poles_by_layer=bipolar_poles,
            w_gate_slice_fn=self.slice_w_gate,
            max_layer_hop=6,
            top_k_per_source_layer=6,
            min_abs_coupling=0.05,
            pre_ffn_norm_weights=self._pre_ffn_norm_weights,
        )

        # 6. Execute Module 4: Projective Theodolite Triangulation (Gram-Schmidt Survey Frame)
        if survey_mode == "pole_motif" and locked_pole is not None:
            p_layer, p_neuron, p_pol = locked_pole
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
            basis = build_pole_motif_frame(
                layer_idx=p_layer,
                neuron_idx=p_neuron,
                polarity=p_pol,
                w_gate_vec=w_gate_v,
                w_down_vec=w_down_v,
                J_l=self.jpp_lenses[p_layer],
                downstream_gate_vec=dst_gate_v,
            )
            pol_char = "+" if p_pol >= 0 else "-"
            survey_labels = (
                f"Gate L{p_layer}:N{p_neuron}",
                f"Write L{p_layer}:N{p_neuron}({pol_char})@J++",
                f"Reader L{next_l}:N{dom_dst_neuron}",
            )
        else:
            t1_id, t1_str = self.resolve_concept_token_id(monument_concepts[0])
            t2_id, t2_str = self.resolve_concept_token_id(monument_concepts[1])
            t3_id, t3_str = self.resolve_concept_token_id(monument_concepts[2])
            global_basis_J, _layer_pullback_basis = build_monument_frame(
                token_ids=(t1_id, t2_id, t3_id),
                lm_head_weight=self.lm_head_weight,
                jpp_lenses=self.jpp_lenses,
                pullback_per_layer=False,
            )
            basis = global_basis_J
            survey_labels = (t1_str.strip() or str(monument_concepts[0]), t2_str.strip() or str(monument_concepts[1]), t3_str.strip() or str(monument_concepts[2]))

        resid_post_layers_t = torch.stack(resid_post_target_list, dim=0)        # [60, 5376]
        resid_post_J_layers_t = torch.stack(resid_post_J_target_list, dim=0)    # [60, 5376]
        delta_attn_J_layers_t = torch.stack(delta_attn_J_target_list, dim=0)    # [60, 5376]
        delta_ffn_J_layers_t = torch.stack(delta_ffn_J_target_list, dim=0)      # [60, 5376]

        trajectory_3d = project_trajectory_to_3d(
            resid_post_layers=resid_post_layers_t,
            resid_post_J_layers=resid_post_J_layers_t,
            delta_attn_J_layers=delta_attn_J_layers_t,
            delta_ffn_J_layers=delta_ffn_J_layers_t,
            basis=basis,
            use_jpp_states=True,
        )

        # 7. Compute 60x60 Linear CKA Similarity Matrix
        seq_states_t = torch.stack(seq_states_list, dim=0)  # [60, n_tokens, 5376]
        cka_matrix = compute_linear_cka_matrix(seq_states_t)

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0.0
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return TheodoliteSweepBundle(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_token_strs=prompt_token_strs,
            target_pos=target_pos,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            trajectory_3d=trajectory_3d,
            causal_splits=causal_splits,
            bipolar_poles=bipolar_poles,
            filaments=filaments,
            cka_matrix=cka_matrix,
            jpp_layer_readouts=jpp_layer_readouts,
            naive_layer_readouts=naive_layer_readouts,
            pole_top_tokens_jpp=pole_top_tokens_jpp,
            pole_top_tokens_naive=pole_top_tokens_naive,
            survey_mode=survey_mode,
            survey_labels=survey_labels,
            peak_vram_gb=peak_vram_gb,
            elapsed_ms=elapsed_ms,
        )
