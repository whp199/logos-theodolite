"""FastAPI + Binary Float32 WebSocket Server (`backend/server.py`) for Logos Theodolite.

Streams packed Binary Float32 frames (`ArrayBuffer`) containing:
- 60-layer 3D Projective Theodolite trajectory `(x_l, y_l, z_l)` + Azimuth `theta`, Elevation `phi`, `||h_l||_2`
- Attention Router (`delta_attn_J`, cyan) vs. FFN Vault (`delta_ffn_J`, gold) 3D vectors & norms
- 60x60 Linear CKA similarity matrix
- 60 x 15 Bipolar GeGLU dipole needles (`+` North / `-` South) oriented in the 3D Survey Frame
- Verified Downstream Reader Circuit Filaments (`C_{(l,i)->(l',j)}`)
- Optional `<|think|>` vs. silent comparison trajectory

Paired with a JSON metadata frame for token strings, J-VINDEX vs. Naive Logit Lens pole labels,
live `SteeringSpec` laser interventions, LARQL `.patch` weight surgery, and FastMCP scientist endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, NormScaledAddSpec, SteeringSpec
from pydantic import BaseModel, Field, ValidationError

from backend.engine_runtime import DEFAULT_TRACKED_CONCEPTS, TheodoliteRuntime, TheodoliteSweepBundle
from backend.larql_vault import LarqlVaultEngine
from backend.mcp_scientist import (
    bind_runtime,
    inspect_pole,
    test_motif_hypothesis,
    trace_downstream_readers,
)
from backend.theodolite_math import (
    D_FF,
    D_MODEL,
    GLOBAL_ATTN_LAYERS_0BASED,
    GLOBAL_ATTN_LAYERS_1BASED,
    N_LAYERS,
    VOCAB_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("theodolite.server")

MAGIC_THEO = 0x5448454F  # "THEO"
WIRE_VERSION = 1

_RUNTIME: Optional[TheodoliteRuntime] = None
_VAULT: Optional[LarqlVaultEngine] = None
_GPU_LOCK = asyncio.Lock()


def get_app_state() -> Tuple[TheodoliteRuntime, LarqlVaultEngine]:
    if _RUNTIME is None or _VAULT is None:
        raise HTTPException(status_code=503, detail="TheodoliteRuntime is still initializing.")
    return _RUNTIME, _VAULT


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _RUNTIME, _VAULT
    logger.info("Initializing TheodoliteRuntime (Gemma-4-31B W4A16 + J++ Lens)...")
    _RUNTIME = TheodoliteRuntime(load_jlens=False).load()
    _VAULT = LarqlVaultEngine(_RUNTIME)
    bind_runtime(_RUNTIME, _VAULT)
    logger.info("TheodoliteRuntime & LarqlVaultEngine bound and ready.")
    yield


app = FastAPI(
    title="Logos Theodolite — Gemma-4-31B 3D Projective Geometry & Circuit Ecology Instrument",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SteeringLaserRequest(BaseModel):
    enabled: bool = False
    target_concept: str = "six"
    contrast_concept: str = "eight"
    layers: List[int] = Field(default_factory=lambda: [29, 35, 41, 47, 53])
    scale: float = 650.0
    pole_layer: Optional[int] = None
    pole_neuron: Optional[int] = None
    pole_polarity: float = 1.0
    # The V7 fixes for bug 1 (REPORT §3.4), each off by default so an old request means what it meant:
    # "legacy": u = (W_U[t+] - W_U[t-]) / fnw.clamp(min=0.1), pulled back by J_l (the original recipe);
    # "fnw":    u = fnw * (W_U[t+] - W_U[t-]), pulled back by J_l, with no inverse and no clamp.
    direction: str = "legacy"
    # zero the quiet residual dims of the vector added at each layer: the 181 persistently quiet dims
    # (V6_quiet_dims_pers181.npy) and the layer's quietest tenth (TheodoliteRuntime.ensure_quiet_masks)
    mask_quiet: bool = False
    # leave the start mark (position 0, <bos>) unsteered: a position mask
    spare_bos: bool = False
    # when set, push strength * ||h|| at every position and layer (NormScaledAddSpec) instead of the absolute
    # length `scale`; V7 found ~0.01 fluent and ~0.024 the edge of collapse
    strength: Optional[float] = None


class SweepRequest(BaseModel):
    # Default two-hop prompt is the example from Gurnee et al. 2026 (REFERENCES.md [2]).
    prompt: str = "The number of legs on the animal that spins webs is"
    monuments: List[str] = Field(default_factory=lambda: ["spider", "eight", "insect"])
    survey_mode: str = "monument"  # "monument" | "pole_motif"
    locked_pole: Optional[List[float]] = None  # [layer, neuron_idx, polarity]
    compare_think: bool = False
    max_new_tokens: int = 6
    steering: Optional[SteeringLaserRequest] = None
    # Chat template (bug 5): wrap `prompt` as the user turn, after an optional system turn, thinking off, and
    # append `assistant_prefix`, the begun answer (labkit.chat). Off: `prompt` is sent as bare text.
    chat: bool = False
    system: Optional[str] = None
    assistant_prefix: str = ""
    # Survey frame (bug 3): "gram_schmidt" (in order, as built) | "lowdin" (all at once, order-free); a frame
    # pulled back by each layer's own lens; the final-norm weighted rows fnw * W_U; and matched controls whose
    # mean state is subtracted before projecting (exact texts, e.g. an earlier sighting's echoed `prompt`).
    frame_squaring: str = "gram_schmidt"
    frame_per_layer: bool = False
    frame_fnw: bool = False
    # at most 16: each is a forward pass under the GPU lock
    centre_prompts: List[str] = Field(default_factory=list, max_length=16)
    # Poles (bug 8): "activation" ranks by |a| | "write" by |a| * ||w_down (.) w_post||.
    pole_ranking: str = "activation"
    # Threads (bug 4): "coupling" selects by C | "direct" by the direct-path estimate of the gate change.
    thread_score: str = "coupling"


class LarqlRequest(BaseModel):
    query: str = "DESCRIBE"


class PoleInspectRequest(BaseModel):
    layer: int = 35
    neuron_idx: int = 0
    polarity: float = 1.0


class TraceReadersRequest(BaseModel):
    layer: int = 35
    neuron_idx: int = 0
    prompt: str = "The number of legs on the animal that spins webs is"
    polarity: Optional[float] = None
    max_layer_hop: int = 6
    top_k: int = 10
    score: str = "coupling"  # "coupling" | "direct"
    chat: bool = False
    system: Optional[str] = None
    assistant_prefix: str = ""


class MotifHypothesisRequest(BaseModel):
    layer: int = 35
    neuron_idx: int = 0
    pos_prompts: List[str]
    neg_prompts: List[str]
    chat: bool = False
    system: Optional[str] = None


STEERING_DIRECTIONS = ("legacy", "fnw")


def _steering_op(vector: torch.Tensor, req: SteeringLaserRequest):
    """The op that adds `vector` (unit-scaled): an absolute length `scale`, or `strength` * ||h|| per position."""
    if req.strength is not None:
        return NormScaledAddSpec(vector=vector, strength=float(req.strength), max_fraction=1.0, normalize=True)
    return AddSpec(vector=vector, scale=float(req.scale), normalize=True)


def _build_steering_spec_from_req(
    runtime: TheodoliteRuntime,
    req: Optional[SteeringLaserRequest],
) -> Optional[SteeringSpec]:
    if req is None or not req.enabled:
        return None
    if req.direction not in STEERING_DIRECTIONS:
        raise ValueError(f"Unknown steering direction {req.direction!r}; expected one of {STEERING_DIRECTIONS}")

    def keep(l_idx: int) -> Optional[torch.Tensor]:
        # needs `await runtime.ensure_quiet_masks()` first (done by _execute_sweep_request)
        return (~runtime.layer_quiet_mask(l_idx)).float() if req.mask_quiet else None

    steer_layers: Dict[int, LayerSteeringSpec] = {}
    if req.pole_layer is not None and req.pole_neuron is not None:
        # Steer directly along a clicked Bipolar GeGLU pole's write vector!
        l_idx = max(0, min(int(req.pole_layer), N_LAYERS - 1))
        n_idx = max(0, min(int(req.pole_neuron), D_FF - 1))
        idx_t = torch.tensor([n_idx], device=runtime.device, dtype=torch.long)
        w_down = runtime.slice_w_down(l_idx, idx_t)[0].float()
        if l_idx in runtime._post_ffn_norm_weights:
            w_down = w_down * runtime._post_ffn_norm_weights[l_idx].to(runtime.device).float()
        pol = 1.0 if float(req.pole_polarity) >= 0 else -1.0
        v_steer = (pol * w_down).detach().cpu()
        if req.mask_quiet:
            v_steer = v_steer * keep(l_idx)
        steer_layers[l_idx] = LayerSteeringSpec(operations=[_steering_op(v_steer, req)])
        return SteeringSpec(layers=steer_layers, point="resid_post")

    tid_pos, _ = runtime.resolve_concept_token_id(req.target_concept)
    tid_neg = runtime.resolve_concept_token_id(req.contrast_concept)[0] if req.contrast_concept.strip() else None
    if req.direction == "fnw":
        # The direction the model reads before speaking: logit_t = <h / rms(h), fnw * W_U[t]> (V7 bug 1 fix)
        w = runtime.final_norm_weight.float()
    else:
        # Concept contrast laser steering pulled back via J_pp[l] with inverse RMSNorm preconditioning
        w = 1.0 / runtime.final_norm_weight.float().clamp(min=0.1)
    u_pos = runtime.lm_head_weight[tid_pos].float() * w
    if tid_neg is not None:
        u_neg = runtime.lm_head_weight[tid_neg].float() * w
        u_dir = u_pos - u_neg
    else:
        u_dir = u_pos

    for l_raw in req.layers:
        l_idx = max(0, min(int(l_raw), N_LAYERS - 1))
        J_l = runtime.jpp_lenses[l_idx].float()
        v_steer = (u_dir.to(J_l.device) @ J_l).detach().cpu()
        if req.mask_quiet:
            v_steer = v_steer * keep(l_idx)
        steer_layers[l_idx] = LayerSteeringSpec(operations=[_steering_op(v_steer, req)])
    return SteeringSpec(layers=steer_layers, point="resid_post")


def _steering_position_mask(req: Optional[SteeringLaserRequest]) -> Optional[List[int]]:
    return [0] if req is not None and req.enabled and req.spare_bos else None


def _describe_steering(
    runtime: TheodoliteRuntime, req: Optional[SteeringLaserRequest], spec: Optional[SteeringSpec]
) -> Optional[Dict[str, Any]]:
    """What was pushed, for the metadata: the recipe, the layers, and how many dims were masked at each."""
    if req is None or spec is None:
        return None
    by_neuron = req.pole_layer is not None and req.pole_neuron is not None
    return {
        "n_masked_by_layer": (
            {str(l): int(runtime.layer_quiet_mask(l).sum()) for l in sorted(spec.layers.keys())}
            if req.mask_quiet
            else None
        ),
        "mode": "neuron" if by_neuron else "words",
        "direction": "neuron_write" if by_neuron else req.direction,
        "mask_quiet": bool(req.mask_quiet),
        "spare_bos": bool(req.spare_bos),
        "strength": float(req.strength) if req.strength is not None else None,
        "scale": None if req.strength is not None else float(req.scale),
        "scale_kind": "share_of_norm" if req.strength is not None else "absolute",
        "layers": sorted(int(l) for l in spec.layers.keys()),
    }


@torch.no_grad()
def pack_binary_and_json_frames(
    runtime: TheodoliteRuntime,
    vault: LarqlVaultEngine,
    sweep: TheodoliteSweepBundle,
    basis_3d: Optional[torch.Tensor] = None,
    think_sweep: Optional[TheodoliteSweepBundle] = None,
    is_steered: bool = False,
    request_echo: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Pack a `TheodoliteSweepBundle` into a deterministic Binary Float32 frame + JSON metadata.

    `basis_3d` defaults to the frame the sweep used (`sweep.survey_basis`); a per-layer frame (`[60, 3, 5376]`,
    raw residual coordinates) projects each pole's raw write vector instead of its J++ image.
    """
    if basis_3d is None:
        basis_3d = sweep.survey_basis
    per_layer_raw = basis_3d is not None and basis_3d.dim() == 3
    n_layers = N_LAYERS
    n_poles = 15
    filaments = sweep.filaments
    n_filaments = len(filaments)

    flags = 0
    if think_sweep is not None:
        flags |= 1
    if is_steered:
        flags |= 2

    header_bytes = struct.pack(
        "<6I",
        MAGIC_THEO,
        WIRE_VERSION,
        n_layers,
        n_poles,
        n_filaments,
        flags,
    )

    # 1. traj_xyz: [60, 3]
    traj_xyz = sweep.trajectory_3d.xyz.detach().cpu().numpy().astype(np.float32)

    # 2. traj_angles_norm: [60, 3] -> (azimuth_deg, elevation_deg, residual_norm)
    traj_angles_norm = np.stack(
        [
            sweep.trajectory_3d.azimuth_deg.detach().cpu().numpy(),
            sweep.trajectory_3d.elevation_deg.detach().cpu().numpy(),
            sweep.trajectory_3d.residual_norm.detach().cpu().numpy(),
        ],
        axis=-1,
    ).astype(np.float32)

    # 3. delta_attn_xyz: [60, 4] -> (dx, dy, dz, attn_norm)
    attn_norms = np.array([float(sweep.causal_splits[l].attn_norm.item()) for l in range(n_layers)], dtype=np.float32)
    delta_attn_xyz = np.concatenate(
        [
            sweep.trajectory_3d.delta_attn_xyz.detach().cpu().numpy().astype(np.float32),
            attn_norms[:, None],
        ],
        axis=-1,
    )

    # 4. delta_ffn_xyz: [60, 4] -> (dx, dy, dz, ffn_norm)
    ffn_norms = np.array([float(sweep.causal_splits[l].ffn_norm.item()) for l in range(n_layers)], dtype=np.float32)
    delta_ffn_xyz = np.concatenate(
        [
            sweep.trajectory_3d.delta_ffn_xyz.detach().cpu().numpy().astype(np.float32),
            ffn_norms[:, None],
        ],
        axis=-1,
    )

    # 5. cka_matrix: [60, 60]
    cka_np = sweep.cka_matrix.detach().cpu().numpy().astype(np.float32)

    # 6. poles_data: [60, 15, 6] -> (neuron_idx, polarity, abs_act, dir_x, dir_y, dir_z)
    B_dev = basis_3d.to(device=runtime.device, dtype=torch.float32)  # [3, 5376] or [60, 3, 5376]
    poles_np = np.zeros((n_layers, n_poles, 6), dtype=np.float32)
    poles_labels: Dict[str, List[Dict[str, Any]]] = {}

    for l in range(n_layers):
        p_l = sweep.bipolar_poles[l]
        # Project pole J++ vectors [15, 5376] onto 3D Survey Frame [3, 5376] -> [15, 3]
        if per_layer_raw:
            dirs_3d = p_l.signed_w_down.float() @ B_dev[l].T
        else:
            dirs_3d = p_l.jpp_pole_vectors.float() @ B_dev.T
        dirs_norm = torch.linalg.vector_norm(dirs_3d, dim=-1, keepdim=True) + 1e-8
        unit_dirs = (dirs_3d / dirs_norm).detach().cpu().numpy()

        n_idx_np = p_l.neuron_indices.detach().cpu().numpy().astype(np.float32)
        pol_np = p_l.polarities.detach().cpu().numpy().astype(np.float32)
        act_np = p_l.abs_activations.detach().cpu().numpy().astype(np.float32)

        poles_np[l, :, 0] = n_idx_np
        poles_np[l, :, 1] = pol_np
        poles_np[l, :, 2] = act_np
        poles_np[l, :, 3:6] = unit_dirs

        layer_pole_meta: List[Dict[str, Any]] = []
        w_sizes = sweep.pole_write_sizes.get(l)
        for r_idx in range(min(5, n_poles)):
            j_toks = sweep.pole_top_tokens_jpp.get((l, r_idx), [])
            n_toks = sweep.pole_top_tokens_naive.get((l, r_idx), [])
            layer_pole_meta.append(
                {
                    "rank": r_idx,
                    "neuron_idx": int(n_idx_np[r_idx]),
                    "polarity": "+" if pol_np[r_idx] >= 0 else "-",
                    "abs_act": round(float(act_np[r_idx]), 3),
                    "write_size": round(float(w_sizes[r_idx]), 4) if w_sizes else None,
                    "j_vindex_tokens": [{"token": t, "logit": round(v, 2)} for t, v in j_toks],
                    "naive_tokens": [{"token": t, "logit": round(v, 2)} for t, v in n_toks],
                }
            )
        poles_labels[str(l)] = layer_pole_meta

    # 7. filaments_data: [M, 7]
    filaments_np = np.zeros((n_filaments, 7), dtype=np.float32)
    filaments_meta: List[Dict[str, Any]] = []
    for f_i, fil in enumerate(filaments):
        filaments_np[f_i, 0] = float(fil.src_layer)
        filaments_np[f_i, 1] = float(fil.src_rank)
        filaments_np[f_i, 2] = float(fil.dst_layer)
        filaments_np[f_i, 3] = float(fil.dst_rank)
        filaments_np[f_i, 4] = float(fil.score if fil.score is not None else fil.coupling_score)
        filaments_np[f_i, 5] = float(fil.src_polarity)
        filaments_np[f_i, 6] = float(fil.dst_polarity)
        if f_i < 60:
            filaments_meta.append(
                {
                    "src_layer": fil.src_layer,
                    "src_rank": fil.src_rank,
                    "src_neuron": fil.src_neuron,
                    "src_polarity": "+" if fil.src_polarity >= 0 else "-",
                    "dst_layer": fil.dst_layer,
                    "dst_rank": fil.dst_rank,
                    "dst_neuron": fil.dst_neuron,
                    "dst_polarity": "+" if fil.dst_polarity >= 0 else "-",
                    "coupling_score": round(fil.coupling_score, 4),
                    "dot_product": round(fil.dot_product, 4),
                    "direct_gate": round(fil.direct_gate, 5) if fil.direct_gate is not None else None,
                    "direct_act": round(fil.direct_act, 5) if fil.direct_act is not None else None,
                    "score": round(fil.score, 5) if fil.score is not None else round(fil.coupling_score, 4),
                    "score_kind": fil.score_kind,
                }
            )

    payload_arrays = [
        traj_xyz.ravel(),
        traj_angles_norm.ravel(),
        delta_attn_xyz.ravel(),
        delta_ffn_xyz.ravel(),
        cka_np.ravel(),
        poles_np.ravel(),
        filaments_np.ravel(),
    ]
    if think_sweep is not None:
        think_xyz = think_sweep.trajectory_3d.xyz.detach().cpu().numpy().astype(np.float32)
        payload_arrays.append(think_xyz.ravel())

    float_payload = np.concatenate(payload_arrays).astype(np.float32).tobytes()
    binary_frame = header_bytes + float_payload

    # Build layer plumes JSON metadata
    layer_plumes: List[Dict[str, Any]] = []
    for l in range(n_layers):
        ro = sweep.jpp_layer_readouts[l]
        nro = sweep.naive_layer_readouts[l]
        split_l = sweep.causal_splits[l]
        layer_plumes.append(
            {
                "layer": l,
                "layer_1based": l + 1,
                "is_global_attention": (l in GLOBAL_ATTN_LAYERS_0BASED),
                "jpp_top_tokens": [
                    {"token": t, "prob": round(p, 4), "logit": round(lg, 2)}
                    for t, p, lg in zip(ro.top_token_strs, ro.top_probs, ro.top_logits)
                ],
                "naive_top_token": nro.top_token_strs[0] if nro.top_token_strs else "",
                "tracked_ranks": ro.tracked_token_ranks,
                "tracked_logits": {k: round(v, 2) for k, v in ro.tracked_token_logits.items()},
                "attn_norm": round(float(split_l.attn_norm.item()), 3),
                "ffn_norm": round(float(split_l.ffn_norm.item()), 3),
                "ffn_dominance": round(float(split_l.ffn_dominance_ratio.item()), 4),
                "rescale_norm": (
                    round(float(split_l.rescale_norm.item()), 3) if split_l.rescale_norm is not None else None
                ),
                "layer_scalar": round(float(split_l.layer_scalar), 4),
                "azimuth_deg": round(float(sweep.trajectory_3d.azimuth_deg[l].item()), 2),
                "elevation_deg": round(float(sweep.trajectory_3d.elevation_deg[l].item()), 2),
                "residual_norm": round(float(sweep.trajectory_3d.residual_norm[l].item()), 2),
                "proj_frac": (
                    round(float(sweep.trajectory_3d.proj_frac[l].item()), 5)
                    if sweep.trajectory_3d.proj_frac is not None
                    else None
                ),
                "pole_write_sizes": (
                    [round(v, 4) for v in sweep.pole_write_sizes[l]] if l in sweep.pole_write_sizes else None
                ),
                "poles_ffn_explained": (
                    round(sweep.poles_ffn_explained[l], 4) if sweep.poles_ffn_explained.get(l) is not None else None
                ),
            }
        )

    json_frame: Dict[str, Any] = {
        **(request_echo or {}),
        "type": "theodolite_metadata",
        "prompt": sweep.prompt,
        "prompt_token_strs": sweep.prompt_token_strs,
        "generated_text": sweep.generated_text,
        "think_generated_text": think_sweep.generated_text if think_sweep is not None else None,
        "survey_mode": sweep.survey_mode,
        "survey_labels": list(sweep.survey_labels),
        "elapsed_ms": round(sweep.elapsed_ms, 1),
        "peak_vram_gb": round(sweep.peak_vram_gb, 2),
        "global_layers_0based": list(GLOBAL_ATTN_LAYERS_0BASED),
        "global_layers_1based": list(GLOBAL_ATTN_LAYERS_1BASED),
        "n_filaments": n_filaments,
        "is_steered": is_steered,
        "prompt_token_ids": list(sweep.prompt_token_ids),
        "pole_ranking": sweep.pole_ranking,
        "thread_score": sweep.thread_score,
        "frame": sweep.frame_info,
        # which lens produced every reading below: what it was fitted on, what it reads, how, its source, and a caution
        "lens": runtime.lens_card(),
        "layer_plumes": layer_plumes,
        "poles_labels": poles_labels,
        "filaments_meta": filaments_meta,
        "larql_status": {
            "active_patches_count": len(vault.active_patches),
            "active_patches": [
                {
                    "patch_id": p.patch_id,
                    "op_type": p.op_type,
                    "layer_idx": p.layer_idx,
                    "neuron_idx": p.neuron_idx,
                    "target_concept": p.target_concept,
                    "statement": p.statement,
                }
                for p in vault.active_patches
            ],
        },
    }
    return binary_frame, json_frame


async def _execute_sweep_request(
    runtime: TheodoliteRuntime,
    vault: LarqlVaultEngine,
    req: SweepRequest,
) -> Tuple[bytes, Dict[str, Any]]:
    monuments = (
        req.monuments[0] if len(req.monuments) > 0 else "spider",
        req.monuments[1] if len(req.monuments) > 1 else "eight",
        req.monuments[2] if len(req.monuments) > 2 else "insect",
    )
    locked_pole = None
    if req.survey_mode == "pole_motif" and req.locked_pole and len(req.locked_pole) >= 3:
        locked_pole = (int(req.locked_pole[0]), int(req.locked_pole[1]), float(req.locked_pole[2]))

    if req.steering is not None and req.steering.enabled and req.steering.mask_quiet:
        await runtime.ensure_quiet_masks()
    steering_spec = _build_steering_spec_from_req(runtime, req.steering)
    position_mask = _steering_position_mask(req.steering) if steering_spec is not None else None
    text = runtime.compose_prompt(
        req.prompt, chat=req.chat, system=req.system, assistant_prefix=req.assistant_prefix
    )
    # Track the landmarks too, so the pages can say how far they were; the default landmarks are already tracked,
    # so a request with the defaults gets exactly the tracked words it always got.
    tracked = tuple(DEFAULT_TRACKED_CONCEPTS) + tuple(
        m for m in dict.fromkeys(str(x) for x in monuments) if m and m not in DEFAULT_TRACKED_CONCEPTS
    )
    options = dict(
        monument_concepts=monuments,
        tracked_concepts=tracked,
        survey_mode=req.survey_mode,
        locked_pole=locked_pole,
        steering_spec=steering_spec,
        steering_position_mask=position_mask,
        top_k_poles=15,
        pole_ranking=req.pole_ranking,
        thread_score=req.thread_score,
        frame_squaring=req.frame_squaring,
        frame_per_layer=req.frame_per_layer,
        frame_fnw=req.frame_fnw,
        centre_prompts=req.centre_prompts,
    )

    sweep = await runtime.run_theodolite_sweep(text, max_new_tokens=req.max_new_tokens, **options)

    think_sweep = None
    if req.compare_think:
        think_prompt = (
            runtime.compose_prompt(
                req.prompt, chat=True, system=req.system, assistant_prefix=req.assistant_prefix, thinking=True
            )
            if req.chat
            else f"<|think|>\n{text}"
        )
        think_sweep = await runtime.run_theodolite_sweep(
            think_prompt, max_new_tokens=max(req.max_new_tokens, 10), **options
        )

    request_echo = {
        "chat": bool(req.chat),
        "system": req.system if (req.chat and req.system) else None,
        "user_prompt": req.prompt,
        "assistant_prefix": req.assistant_prefix if req.assistant_prefix else None,
        "steering_applied": _describe_steering(runtime, req.steering, steering_spec),
    }
    return pack_binary_and_json_frames(
        runtime=runtime,
        vault=vault,
        sweep=sweep,
        think_sweep=think_sweep,
        is_steered=(steering_spec is not None),
        request_echo=request_echo,
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    from fastapi.responses import Response

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<circle cx="32" cy="32" r="26" stroke="#00e5ff" stroke-width="5" fill="#030712"/>'
        '<circle cx="32" cy="32" r="6" fill="#fbbf24"/></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    runtime, vault = get_app_state()
    vram_alloc = torch.cuda.memory_allocated(runtime.device) / (1024**3) if torch.cuda.is_available() else 0.0
    vram_peak = torch.cuda.max_memory_allocated(runtime.device) / (1024**3) if torch.cuda.is_available() else 0.0
    return {
        "status": "ok",
        "model_id": runtime.model_id,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "vocab_size": VOCAB_SIZE,
        "global_layers_1based": list(GLOBAL_ATTN_LAYERS_1BASED),
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_peak_gb": round(vram_peak, 2),
        "larql_cli": vault.cli_versions(),
        "lens": runtime.lens_card(),
    }


@app.post("/api/larql")
async def api_larql(req: LarqlRequest) -> Dict[str, Any]:
    _runtime, vault = get_app_state()
    async with _GPU_LOCK:
        try:
            return vault.execute_lql(req.query)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scientist/inspect_pole")
async def api_inspect_pole(req: PoleInspectRequest) -> Dict[str, Any]:
    # the scientist tools name their lens themselves ("lens"); these routes pass their replies on unchanged
    _runtime, _vault = get_app_state()
    async with _GPU_LOCK:
        return inspect_pole(layer=req.layer, neuron_idx=req.neuron_idx, polarity=req.polarity)


@app.post("/api/scientist/trace_readers")
async def api_trace_readers(req: TraceReadersRequest) -> Dict[str, Any]:
    _runtime, _vault = get_app_state()
    async with _GPU_LOCK:
        try:
            return await trace_downstream_readers(
                layer=req.layer,
                neuron_idx=req.neuron_idx,
                prompt=req.prompt,
                polarity=req.polarity,
                max_layer_hop=req.max_layer_hop,
                top_k=req.top_k,
                score=req.score,
                chat=req.chat,
                system=req.system,
                assistant_prefix=req.assistant_prefix,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scientist/test_hypothesis")
async def api_test_hypothesis(req: MotifHypothesisRequest) -> Dict[str, Any]:
    _runtime, _vault = get_app_state()
    async with _GPU_LOCK:
        return await test_motif_hypothesis(
            layer=req.layer,
            neuron_idx=req.neuron_idx,
            pos_prompts=req.pos_prompts,
            neg_prompts=req.neg_prompts,
            chat=req.chat,
            system=req.system,
        )


@app.websocket("/ws/theodolite")
async def ws_theodolite(websocket: WebSocket) -> None:
    """Real-time Binary Float32 + JSON WebSocket stream for the 3D Theodolite frontend."""
    await websocket.accept()
    runtime, vault = get_app_state()
    logger.info("WebSocket client connected to /ws/theodolite")

    try:
        while True:
            raw_msg = await websocket.receive_json()
            action = raw_msg.get("action", "sweep")

            try:
                if action == "sweep":
                    sweep_req = SweepRequest(**raw_msg.get("payload", raw_msg))
                    async with _GPU_LOCK:
                        binary_frame, json_frame = await _execute_sweep_request(runtime, vault, sweep_req)
                    await websocket.send_bytes(binary_frame)
                    await websocket.send_json(json_frame)

                elif action == "larql":
                    query = raw_msg.get("query", "DESCRIBE")
                    async with _GPU_LOCK:
                        try:
                            res = vault.execute_lql(query)
                            await websocket.send_json({"type": "larql_result", "ok": True, "data": res})
                        except Exception as exc:
                            await websocket.send_json({"type": "larql_result", "ok": False, "error": str(exc)})

                elif action == "inspect_pole":
                    l_idx = int(raw_msg.get("layer", 35))
                    n_idx = int(raw_msg.get("neuron_idx", 0))
                    pol = float(raw_msg.get("polarity", 1.0))
                    async with _GPU_LOCK:
                        res = inspect_pole(layer=l_idx, neuron_idx=n_idx, polarity=pol)
                    await websocket.send_json({"type": "pole_inspection", "data": res})

                elif action == "trace_readers":
                    tr = TraceReadersRequest(**{k: v for k, v in raw_msg.items() if k != "action"})
                    async with _GPU_LOCK:
                        res = await trace_downstream_readers(
                            layer=tr.layer,
                            neuron_idx=tr.neuron_idx,
                            prompt=tr.prompt,
                            polarity=tr.polarity,
                            max_layer_hop=tr.max_layer_hop,
                            top_k=tr.top_k,
                            score=tr.score,
                            chat=tr.chat,
                            system=tr.system,
                            assistant_prefix=tr.assistant_prefix,
                        )
                    await websocket.send_json({"type": "downstream_readers", "data": res})
            except (ValidationError, ValueError) as exc:
                # A malformed request (a bad option value, say) fails alone and the socket stays open; the reply
                # names the action and its neuron so the client can fail just that request.
                logger.warning("WebSocket %s request refused: %s", action, exc)
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(exc),
                        "action": action,
                        "recoverable": True,
                        "layer": raw_msg.get("layer"),
                        "neuron_idx": raw_msg.get("neuron_idx"),
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/theodolite")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")

