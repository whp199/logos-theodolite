"""FastAPI + Binary Float32 WebSocket Server (`backend/server.py`) for Logos Theodolite.

Streams packed Binary Float32 frames (`ArrayBuffer`) containing:
- 60-layer 3D Projective Theodolite trajectory `(x_l, y_l, z_l)` + Azimuth `theta`, Elevation `phi`, `||h_l||_2`
- Attention Router (`delta_attn_J`, cyan) vs. FFN Vault (`delta_ffn_J`, gold) 3D vectors & norms
- 60x60 Linear CKA similarity matrix
- 60 x 15 Bipolar GeGLU dipole needles (`+` North / `-` South) oriented in the 3D Survey Frame
- Verified Downstream Reader Circuit Filaments (`C_{(l,i)->(l',j)}`)
- Optional thinking vs. silent comparison trajectory: for a chat, thinking switched on by the chat template's own switch
  (`enable_thinking`); for bare text, the old `<|think|>` set before the text unless `think_bare` is "none", which
  leaves it out (`think_protocol` and `think_left_out` say which)

Paired with a JSON metadata frame for token strings, J-VINDEX vs. Naive Logit Lens pole labels,
live `SteeringSpec` laser interventions, LARQL `.patch` weight surgery, and FastMCP scientist endpoints.

A sweep request with a `request_id` is answered as a stream: progress while the answer is written, the answer once,
one (binary, JSON) pair per sighted word-piece, then an end; `cancel_sweep` stops it (ARCHITECTURE.md §4.1-4.3).
Without one it gets the single pair it always got. Saved questions and system prompts: backend/saved_prompts.py.

Every route, the pages and the socket answer only the instrument's own pages, served from this machine, and scripts run
on it (backend/local_only.py, around the whole app); CORS allows only pages served from this machine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, NormScaledAddSpec, SteeringSpec
from pydantic import BaseModel, Field, StrictInt, ValidationError

from backend import atlas as atlas_routes
from backend import scientist_chat
from backend import w4a16_fast
from backend import causal_check as cc
from backend import cpu_memory
from backend.local_only import LocalOnly
from backend.engine_runtime import (
    DEFAULT_TRACKED_CONCEPTS,
    MAX_SWEEP_POSITIONS,
    SamplerSettings,
    SweepCapture,
    SweepShared,
    TheodoliteRuntime,
    TheodoliteSweepBundle,
    study_mean_problem,
)
from backend.label_gate import LABEL_GATE_CARD
from backend.larql_vault import VOICE_MAX_NEW_TOKENS_DEFAULT, VOICE_MAX_NEW_TOKENS_LIMIT, LarqlVaultEngine
from backend.mcp_scientist import (
    bind_runtime,
    causal_spot_check,
    inspect_pole,
    test_motif_hypothesis,
    trace_downstream_readers,
)
from backend.runtime_factory import make_runtime, weights_card
from backend.saved_prompts import router as saved_prompts_router
from backend.theodolite_math import (
    CONTRAST_BACKGROUNDS,
    D_FF,
    D_MODEL,
    FRAME_SQUARINGS,
    GLOBAL_ATTN_LAYERS_0BASED,
    GLOBAL_ATTN_LAYERS_1BASED,
    N_LAYERS,
    POLE_RANKINGS,
    THREAD_SCORES,
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
    # how the 4-bit weights are unpacked ($THEODOLITE_UNPACKER, backend/w4a16_fast.py: each gives the same bits, tested
    # before anything is read where install tests it: compiled, resident, and lut on CUDA; the reference unless set),
    # its name checked before the model loads, which takes minutes on the 31B
    unpacker = os.environ.get(w4a16_fast.UNPACKER_ENV) or "reference"
    if unpacker not in w4a16_fast.MODES:
        raise w4a16_fast.Refused(f"${w4a16_fast.UNPACKER_ENV} names the unpackers {', '.join(w4a16_fast.MODES)}, not "
                                 f"{unpacker!r}; nothing was loaded.")
    logger.info("Initializing TheodoliteRuntime (Gemma-4-31B W4A16 + J++ Lens)...")
    _RUNTIME = make_runtime(load_jlens=False).load()  # THEODOLITE_GGUF / THEODOLITE_DEVICE (backend/runtime_factory.py)
    try:
        w4a16_fast.install(_RUNTIME, unpacker, log=logger.info)
    except w4a16_fast.Refused as exc:  # a self-test failed: said in plain words, as census_run.py and atlas_build.py do
        raise SystemExit(str(exc)) from exc
    _VAULT = LarqlVaultEngine(_RUNTIME)
    bind_runtime(_RUNTIME, _VAULT)
    logger.info("TheodoliteRuntime & LarqlVaultEngine bound and ready.")
    yield


app = FastAPI(
    title="Logos Theodolite — Gemma-4-31B 3D Projective Geometry & Circuit Ecology Instrument",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS lets a page read a reply back only when it was served from this machine (any port: the server, vite's dev and
# preview servers). CORS does not stop a request being sent, and a browser applies none to a WebSocket, so LocalOnly,
# added last and so run first, refuses a request or a socket from anywhere else before CORS or any route sees it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LocalOnly)


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


class SamplerRequest(BaseModel):
    # How the answer is written; the defaults are greedy, as the instrument always wrote and the study generated.
    # Every knob is sent as the value it means: top_k 0 and top_p 1 are off (interp_engine reads a missing knob as
    # the checkpoint's recommendation, temperature 1, top-k 64, top-p 0.95, so none is ever left missing).
    temperature: float = Field(0.0, ge=0.0, le=2.0)  # 0 = greedy (argmax)
    top_k: int = Field(0, ge=0, le=VOCAB_SIZE)  # 0 = off
    top_p: float = Field(1.0, gt=0.0, le=1.0)  # 1 = off
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)  # subtracted from every piece already WRITTEN (not the prompt)
    seed: Optional[int] = Field(None, ge=0, le=2**53 - 1)  # None: drawn by the server when sampling, and echoed


# a typed word, as the landmarks take one: capped in length only (each is resolved, or refused, by the runtime)
ContrastWord = Annotated[str, Field(max_length=64)]


class ContrastRequest(BaseModel):
    # One word against another: the mean over the pairs of logit(toward[i]) - logit(away[i]) at every layer, through
    # the lens and by the naked eye. Words pair by position, as the study's pronoun pairs do (labkit.PRON_PAIRS); each
    # is read as the landmarks read theirs (" word", then the word), and must be one word-piece in one of those forms.
    toward: List[ContrastWord] = Field(..., min_length=1, max_length=12)
    away: List[ContrastWord] = Field(..., min_length=1, max_length=12)
    label: Optional[str] = Field(None, max_length=64)


class SweepRequest(BaseModel):
    # Default two-hop prompt is the example from Gurnee et al. 2026 (REFERENCES.md [2]).
    prompt: str = Field("The number of legs on the animal that spins webs is", max_length=32000)
    monuments: List[str] = Field(default_factory=lambda: ["spider", "eight", "insect"])
    survey_mode: str = "monument"  # "monument" | "pole_motif"
    locked_pole: Optional[List[float]] = None  # [layer, neuron_idx, polarity]
    compare_think: bool = False
    # Thinking on bare text (E6 §3, item 11): "prefix" sets "<|think|>\n" before the text, as first built, which is not
    # Gemma 4's switch (the template's: <|think|> at the head of a system turn, the model's turn begun with no empty
    # thought); "none" leaves the thinking comparison out for bare text, and the pair says so (think_left_out). A chat
    # always uses the template's switch. The first pair names the one used (think_protocol; null on later pairs, as the
    # other think_* keys are).
    think_bare: str = Field("prefix", max_length=64)
    max_new_tokens: int = Field(6, ge=0, le=256)
    steering: Optional[SteeringLaserRequest] = None
    # Chat template (bug 5): wrap `prompt` as the user turn, after an optional system turn, thinking off, and
    # append `assistant_prefix`, the begun answer (labkit.chat). Off: `prompt` is sent as bare text.
    chat: bool = False
    system: Optional[str] = Field(None, max_length=32000)
    assistant_prefix: str = Field("", max_length=32000)
    # A sighting of every word-piece (all optional; the defaults answer as the instrument always did). With a
    # request_id the reply is a stream (progress, the answer written, one pair per sighted word-piece, an end);
    # without one, exactly one pair, at the first index of the sight order (sight_tokens[0] when given, else token 0).
    request_id: Optional[str] = Field(None, max_length=64)
    sampler: SamplerRequest = Field(default_factory=SamplerRequest)
    # stop on the checkpoint's end marks (generation_config.eos_token_id: 1 <eos>, 106 <turn|>, 50 <|tool_response>)
    stop_at_end_of_turn: bool = True
    sight_every_token: bool = False  # one pair per written word-piece, the first max_sighted
    sight_tokens: Optional[List[StrictInt]] = Field(None, max_length=64)  # these token indices, in this order
    max_sighted: int = Field(64, ge=1, le=64)
    # a given answer: nothing is written; these ids are read back teacher-forced (a retake of the same answer)
    answer_token_ids: Optional[List[StrictInt]] = Field(None, max_length=256)
    # the space-then-digit rank merge (the instrument's old rule); a sighting of every word-piece sends False
    rank_lookahead: bool = True
    # matched controls as exact ids (beside centre_prompts' texts), and whether a word-piece past token 0 is centred
    # on the controls reading this very answer (else it is not centred at all)
    centre_token_ids: List[List[StrictInt]] = Field(default_factory=list, max_length=16)
    centre_answer: bool = False
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
    # One word against another (V7 bug 3's contrastive axes; E3 (c)): at most four contrasts, each read at every layer
    # of every sighting, and read against the backgrounds named here: "controls" (the matched controls sent: the
    # sighting's score less theirs) | "study_mean" (E3's average state subtracted first, as E3 centred it). None by
    # default: the sweep is read exactly as before, and each pair's "contrasts" is [].
    contrasts: List[ContrastRequest] = Field(default_factory=list, max_length=4)
    contrast_backgrounds: List[Annotated[str, Field(max_length=64)]] = Field(default_factory=list, max_length=2)


class VoiceRequest(BaseModel):
    # The question an edit is heard by: just before it and again just after, the model writes its answer the likeliest
    # way (LarqlVaultEngine._heard). Sent as compose_prompt sends a sweep's: as a chat (the default) or bare text.
    prompt: str = Field(..., min_length=1, max_length=32000)
    chat: bool = True
    system: Optional[str] = Field(None, max_length=32000)
    assistant_prefix: str = Field("", max_length=32000)
    max_new_tokens: int = Field(VOICE_MAX_NEW_TOKENS_DEFAULT, ge=1, le=VOICE_MAX_NEW_TOKENS_LIMIT)


class LarqlRequest(BaseModel):
    query: str = "DESCRIBE"
    # optional: hear an edit (DELETE, UPDATE, INSERT) before and after; every other statement ignores it
    voice: Optional[VoiceRequest] = None


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
    # exact word-piece ids to read (a sighting's ids up to its read position): used as they are, read at the last;
    # prompt, chat, system and assistant_prefix are then not used
    token_ids: Optional[List[StrictInt]] = Field(None, min_length=1, max_length=MAX_SWEEP_POSITIONS)


class CausalCheckRequest(BaseModel):
    # A new request, so its defaults are the mended ones (k = 10, erratum E47). Types only: every bound is checked by
    # the tool itself (`mcp_scientist.causal_spot_check`), in plain words, so a refusal reads the same over the
    # WebSocket, HTTP and MCP; strings only get a length cap.
    layer: StrictInt
    neuron_idx: StrictInt
    polarity: Optional[float] = None
    contexts: str = Field("given", max_length=64)  # "given" | "study"
    # given mode: exact word-piece ids (a sighting's ids up to its read position), pushed at the last; else `prompt`
    token_ids: Optional[List[StrictInt]] = None
    prompt: Optional[str] = Field(None, max_length=32000)
    chat: bool = False
    system: Optional[str] = Field(None, max_length=32000)
    assistant_prefix: str = Field("", max_length=32000)
    k: float = cc.K_DEFAULT
    write: str = Field("activation", max_length=64)  # "activation" | "frozen_norm"
    scale_ref: str = Field("study", max_length=64)  # "study" | "this_text"
    floor: bool = True
    top_n: StrictInt = 10
    track_ids: Optional[List[StrictInt]] = None


class MotifHypothesisRequest(BaseModel):
    layer: int = 35
    neuron_idx: int = 0
    pos_prompts: List[str]
    neg_prompts: List[str]
    chat: bool = False
    system: Optional[str] = None


STEERING_DIRECTIONS = ("legacy", "fnw")
# how a sweep with compare_think treats bare text: the old imitation, as first built, or no thinking comparison
THINK_BARE = ("prefix", "none")


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
        # needs `await runtime.ensure_quiet_masks()` first (done by _stream_sweep)
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
    stream_info: Optional[Dict[str, Any]] = None,
    flags_extra: int = 0,
) -> Tuple[bytes, Dict[str, Any]]:
    """Pack a `TheodoliteSweepBundle` into a deterministic Binary Float32 frame + JSON metadata.

    `basis_3d` defaults to the frame the sweep used (`sweep.survey_basis`); a per-layer frame (`[60, 3, 5376]`,
    raw residual coordinates) projects each pole's raw write vector instead of its J++ image. `stream_info` (which
    word-piece this pair sights, the answer, how it was written) is added to the JSON; `flags_extra` is OR-ed into the
    header's flags (4: one of several pairs of one request).
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
    flags |= int(flags_extra)

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
                    # the label's check by the study's measure (first ten a side; backend/label_gate.py); None where
                    # no words were read at this layer
                    "label_check": sweep.pole_label_checks.get((l, r_idx)),
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
                # the same words by id, so a page can match them to the word-piece written without comparing strings
                "jpp_top_ids": [int(i) for i in ro.top_token_ids],
                "naive_top_token": nro.top_token_strs[0] if nro.top_token_strs else "",
                "naive_top_id": int(nro.top_token_ids[0]) if nro.top_token_ids else None,
                "tracked_ranks": ro.tracked_token_ranks,
                "tracked_logits": {k: round(v, 2) for k, v in ro.tracked_token_logits.items()},
                # True where a tracked word's rank and logit were taken from the next position (rank_lookahead)
                "tracked_from_next": {
                    lbl: bool(sweep.tracked_from_next.get(l, {}).get(lbl, False)) for lbl in ro.tracked_token_ranks
                },
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
        # one word against another: one entry per contrast asked for (TheodoliteRuntime._contrast_readings), else []
        "contrasts": list(getattr(sweep, "contrasts", None) or []),
        # the rule each poles_labels entry's label_check follows: which labels the pages hold back, and why
        "label_gate": LABEL_GATE_CARD,
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
    if stream_info:
        json_frame.update({k: v for k, v in stream_info.items() if k != "type"})
    return binary_frame, json_frame


def sight_order(
    n_written: int,
    sight_tokens: Optional[Sequence[int]],
    sight_every_token: bool,
    max_sighted: int,
) -> Tuple[List[int], List[int]]:
    """Which token indices a sweep sights, in order, and the asked-for ones it dropped.

    `sight_tokens` given: those, without repeats, in the order asked, at most `max_sighted`, leaving out (and
    reporting) any outside 0 .. max(n_written, 1) - 1; `sight_every_token`: 0 .. min(n_written, max_sighted) - 1;
    else [0], the one pair the instrument always sent. Never empty: [0] when nothing asked for can be sighted."""
    n_positions = max(int(n_written), 1)
    if sight_tokens is not None:
        order: List[int] = []
        dropped: List[int] = []
        seen: set = set()
        for k in sight_tokens:
            k = int(k)
            if k in seen:
                continue
            seen.add(k)
            if not 0 <= k < n_positions or len(order) >= max_sighted:
                dropped.append(k)
            else:
                order.append(k)
        return (order or [0]), dropped
    if sight_every_token:
        return (list(range(min(int(n_written), int(max_sighted)))) or [0]), []
    return [0], []


def check_sweep_ids(req: SweepRequest) -> None:
    """Refuse, in plain words, word-piece ids no tokenizer could give (pydantic has already checked they are whole
    numbers and how many there are)."""

    def check(ids: Sequence[int], what: str) -> None:
        for t in ids:
            if not 0 <= int(t) < VOCAB_SIZE:
                raise ValueError(f"{what} holds {t}, which is not a word-piece id: ids run from 0 to {VOCAB_SIZE - 1}.")

    check(req.answer_token_ids or [], "The given answer")
    for j, ids in enumerate(req.centre_token_ids):
        if not ids:
            raise ValueError(f"Matched control {j + 1} has no word-pieces: each control needs at least one.")
        if len(ids) > MAX_SWEEP_POSITIONS:
            raise ValueError(
                f"Matched control {j + 1} has {len(ids)} word-pieces, more than the {MAX_SWEEP_POSITIONS} one capture "
                "can hold."
            )
        check(ids, f"Matched control {j + 1}")


def check_sweep_options(req: SweepRequest) -> None:
    """Refuse, in plain words and before anything is written, a reading option the analysis does not know (the
    runtime checks them too, but only once the answer has been written)."""
    for what, value, known in (
        ("pole_ranking", req.pole_ranking, POLE_RANKINGS),
        ("thread_score", req.thread_score, THREAD_SCORES),
        ("frame_squaring", req.frame_squaring, FRAME_SQUARINGS),
        ("think_bare", req.think_bare, THINK_BARE),
    ):
        if value not in known:
            choices = " or ".join(f'"{k}"' for k in known)
            shown = value if len(value) <= 40 else value[:40] + "…"
            raise ValueError(f'{what} is "{shown}", which this instrument does not know: it must be {choices}.')
    for value in req.contrast_backgrounds:
        if value not in CONTRAST_BACKGROUNDS:
            choices = " or ".join(f'"{k}"' for k in CONTRAST_BACKGROUNDS)
            shown = value if len(value) <= 40 else value[:40] + "…"
            raise ValueError(
                f'contrast_backgrounds holds "{shown}", which this instrument does not know: each must be {choices}.'
            )
    if "controls" in req.contrast_backgrounds and not (
        any(t and t.strip() for t in req.centre_prompts) or req.centre_token_ids
    ):
        raise ValueError(
            'contrast_backgrounds asks for "controls", but no matched control was sent: give centre_prompts or '
            'centre_token_ids, or leave "controls" out.'
        )
    if "study_mean" in req.contrast_backgrounds:
        problem = study_mean_problem()
        if problem is not None:
            raise ValueError(problem)
    for j, c in enumerate(req.contrasts):
        if len(c.toward) != len(c.away):
            raise ValueError(
                f"Contrast {j + 1} has {len(c.toward)} word(s) toward and {len(c.away)} away: its words are read in "
                "pairs, the first toward against the first away, so it needs as many of each."
            )
        if any(not w.strip() for w in c.toward + c.away):
            raise ValueError(f"Contrast {j + 1} has a blank word: each word must be typed.")


def check_sweep_positions(n_prompt: int, n_answer: int, what: str = "The prompt") -> None:
    """Refuse a sweep longer than `MAX_SWEEP_POSITIONS` word-pieces, prompt and answer together, in plain words."""
    total = int(n_prompt) + int(n_answer)
    if total > MAX_SWEEP_POSITIONS:
        raise ValueError(
            f"{what} is {n_prompt} word-pieces and the answer may run to {n_answer} more: {total} in all, more than "
            f"the {MAX_SWEEP_POSITIONS} one sweep can hold (the writing and the kinship of layers take more of the "
            "GPU's memory with every word-piece). Shorten the question, the system prompt or the answer."
        )


def _plain_validation(exc: ValidationError, lead: str = "The sweep was refused.") -> str:
    """A pydantic refusal as one line: where, and what was wrong."""
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ())) or "request"
        parts.append(f"{where}: {err.get('msg', 'not accepted')}")
    return f"{lead} " + "; ".join(parts)


def _plain_error(exc: BaseException) -> str:
    if cpu_memory.is_oom(exc):  # the card's, or the CPU allocator's (backend/cpu_memory.py)
        return f"{cpu_memory.plain(exc, 'this sweep')} Try a shorter question or answer, or sight fewer word-pieces."
    if isinstance(exc, ValueError):
        return str(exc)
    return f"The sweep failed: {type(exc).__name__}: {exc}"


def _answer_wire(cap: SweepCapture) -> Dict[str, Any]:
    """The answer as sent (ANSWER): its word-pieces, how the writing stopped, and every piece's odds."""
    return {
        "token_ids": list(cap.answer_token_ids),
        "token_strs": list(cap.answer_token_strs),
        "text": cap.answer_text,
        "n_written": len(cap.answer_token_ids),
        "stop_reason": cap.stop_reason,
        "stop_token_ids": list(cap.stop_token_ids),
        "source": cap.answer_source,
        "steps": [s.as_wire() for s in cap.steps],
    }


def _generation_wire(cap: SweepCapture, req: SweepRequest, capped: bool) -> Dict[str, Any]:
    """How the answer was written (GENERATION): the sampler as resolved and applied, and the seed it used."""
    return {
        "greedy": cap.greedy,
        "settings": dict(cap.settings),
        "seed": cap.seed,
        "seed_drawn": cap.seed_drawn,
        "max_new_tokens": int(req.max_new_tokens),
        "sight_every_token": bool(req.sight_every_token),
        "max_sighted": int(req.max_sighted),
        "capped": bool(capped),
        "stop_at_end_of_turn": bool(req.stop_at_end_of_turn),
    }


@dataclass
class _Tally:
    """How far one request's stream got, for its error and its sweep_end."""

    n_sent: int = 0
    n_to_sight: int = 0
    token_index: Optional[int] = None
    cancelled: bool = False


class _Outbox:
    """Every send on one socket goes through here, so a pair's binary frame and its JSON leave back to back: a reply
    the receiver sends (a cancelled request's end) waits until a pair is out."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.lock = asyncio.Lock()

    async def json(self, obj: Dict[str, Any]) -> None:
        async with self.lock:
            await self.websocket.send_json(obj)

    async def pair(self, binary: bytes, meta: Dict[str, Any]) -> None:
        async with self.lock:
            await self.websocket.send_bytes(binary)
            await self.websocket.send_json(meta)


class _SocketState:
    """What one /ws/theodolite connection knows about its sweep requests (request ids are the client's own, unique
    per socket): the one running and whether it was asked to stop, those still queued, those cancelled while queued,
    and whether the client has gone."""

    def __init__(self) -> None:
        self.running: Optional[str] = None
        self.cancel_running = False
        self.closed = False
        self.queued: Dict[str, int] = {}
        self.dropped: Dict[str, int] = {}

    def cancel_requested(self, request_id: Optional[str]) -> bool:
        return self.closed or (request_id is not None and self.running == request_id and self.cancel_running)

    def enqueue(self, request_id: str) -> None:
        self.queued[request_id] = self.queued.get(request_id, 0) + 1

    def drop_queued(self, request_id: str) -> bool:
        """Mark a queued request cancelled; False when none with this id is waiting."""
        if self.queued.get(request_id, 0) > self.dropped.get(request_id, 0):
            self.dropped[request_id] = self.dropped.get(request_id, 0) + 1
            return True
        return False

    def dequeue(self, request_id: str) -> bool:
        """Take a request off the queue; True when it was cancelled while waiting (its end has been sent)."""
        n = self.queued.get(request_id, 0) - 1
        if n > 0:
            self.queued[request_id] = n
        else:
            self.queued.pop(request_id, None)
        if self.dropped.get(request_id, 0) > 0:
            self.dropped[request_id] -= 1
            if not self.dropped[request_id]:
                del self.dropped[request_id]
            return True
        return False


async def _let_receiver_run() -> None:
    """A few turns of the event loop for the socket's receiver task, so a cancel_sweep that has arrived is read before
    the next GPU step (the GPU work itself never yields)."""
    for _ in range(3):
        await asyncio.sleep(0)


def _sweep_options(req: SweepRequest) -> Tuple[Tuple[str, str, str], Optional[Tuple[int, int, float]], Tuple[str, ...]]:
    """The landmarks, the locked pole and the tracked words a request asks for."""
    monuments = (
        req.monuments[0] if len(req.monuments) > 0 else "spider",
        req.monuments[1] if len(req.monuments) > 1 else "eight",
        req.monuments[2] if len(req.monuments) > 2 else "insect",
    )
    locked_pole = None
    if req.survey_mode == "pole_motif" and req.locked_pole and len(req.locked_pole) >= 3:
        locked_pole = (int(req.locked_pole[0]), int(req.locked_pole[1]), float(req.locked_pole[2]))
    # Track the landmarks too, so the pages can say how far they were; the default landmarks are already tracked,
    # so a request with the defaults gets exactly the tracked words it always got.
    tracked = tuple(DEFAULT_TRACKED_CONCEPTS) + tuple(
        m for m in dict.fromkeys(str(x) for x in monuments) if m and m not in DEFAULT_TRACKED_CONCEPTS
    )
    return monuments, locked_pole, tracked


def _contrast_options(req: SweepRequest) -> Dict[str, Any]:
    """The contrasts a request asks for, as `prepare_sweep` takes them; nothing at all without any, so a sweep
    without contrasts calls the runtime exactly as before (the thinking comparison is read without them)."""
    if not req.contrasts:
        return {}
    return {
        "contrasts": [c.model_dump() for c in req.contrasts],
        "contrast_backgrounds": tuple(dict.fromkeys(req.contrast_backgrounds)),
    }


async def _stream_sweep(
    out: _Outbox,
    conn: _SocketState,
    runtime: TheodoliteRuntime,
    vault: LarqlVaultEngine,
    req: SweepRequest,
    tally: _Tally,
) -> None:
    """Write the answer once, then send one (binary, JSON) pair per sighted word-piece, in sight order.

    Holds `_GPU_LOCK` from the writing to the last pair (and the thinking comparison), so no vault edit can land
    between two positions of one answer. With a request_id: progress after each written piece, the answer once it is
    written and read back (before the frame is built, so a later failure cannot lose it), then the pairs; a cancel
    stops the writing, then only the first index is sighted, and between pairs the stream ends after the current one.
    The thinking comparison (on the first pair) can be stopped too: a stop asked before it begins skips it
    (`think_skipped`), one asked while it is written cuts it short (`think_stop_reason` "stopped"), and either way
    the request ends `cancelled`. Without a request_id: one pair, nothing else."""
    rid = req.request_id
    stream = rid is not None
    t_request = time.perf_counter()
    check_sweep_ids(req)
    check_sweep_options(req)
    monuments, locked_pole, tracked = _sweep_options(req)
    text = runtime.compose_prompt(
        req.prompt, chat=req.chat, system=req.system, assistant_prefix=req.assistant_prefix
    )
    check_sweep_positions(
        len(runtime.tokenize_prompt(text)), max(int(req.max_new_tokens), len(req.answer_token_ids or []))
    )
    think_prompt: Optional[str] = None
    # how the thinking comparison was switched on: "template" (a chat: enable_thinking, <|think|> at the head of a
    # system turn the template opens if there is none, and the model's turn begun with no empty thought) or
    # "bare_prefix" (bare text with "<|think|>\n" before it: the imitation, not Gemma 4's switch); None without one
    think_protocol: Optional[str] = None
    # "bare_text" when compare_think asked for a comparison that bare text cannot have (think_bare "none")
    think_left_out: Optional[str] = None
    # the thinking comparison writes its own start, as long as the answer (at least 10, at most 64 word-pieces)
    think_len = max(min(int(req.max_new_tokens), 64), 10)
    if req.compare_think and req.chat:
        think_prompt = runtime.compose_prompt(
            req.prompt, chat=True, system=req.system, assistant_prefix=req.assistant_prefix, thinking=True
        )
        think_protocol = "template"
    elif req.compare_think and req.think_bare == "prefix":
        think_prompt = f"<|think|>\n{text}"
        think_protocol = "bare_prefix"
    elif req.compare_think:
        think_left_out = "bare_text"
    if think_prompt is not None:
        check_sweep_positions(len(runtime.tokenize_prompt(think_prompt)), think_len, "The thinking prompt")
    # on the CPU, what the sweep will hold against MemAvailable, refused in plain words before it is written
    cpu_memory.ensure(
        runtime,
        lambda: cpu_memory.sweep_need(
            runtime, len(runtime.tokenize_prompt(text)) + max(int(req.max_new_tokens), len(req.answer_token_ids or [])),
            thinking=think_prompt is not None,
        ),
        "This sweep",
        "shorten the question, the system prompt or the answer",
    )
    sampler = SamplerSettings(
        temperature=float(req.sampler.temperature),
        top_k=int(req.sampler.top_k),
        top_p=float(req.sampler.top_p),
        presence_penalty=float(req.sampler.presence_penalty),
        seed=req.sampler.seed,
    )
    stop_ids = runtime.end_of_turn_ids() if req.stop_at_end_of_turn else None

    def wants_stop() -> bool:
        return stream and conn.cancel_requested(rid)

    async def on_written(i: int) -> bool:
        await out.json(
            {"type": "theodolite_progress", "request_id": rid, "phase": "writing", "done": i + 1, "of": int(req.max_new_tokens)}
        )
        await _let_receiver_run()
        return not wants_stop()

    async def on_thought(_i: int) -> bool:
        await _let_receiver_run()
        return not wants_stop()

    async with _GPU_LOCK:
        if req.steering is not None and req.steering.enabled and req.steering.mask_quiet:
            await runtime.ensure_quiet_masks()
        steering_spec = _build_steering_spec_from_req(runtime, req.steering)
        position_mask = _steering_position_mask(req.steering) if steering_spec is not None else None

        cap = await runtime.capture_for_sweep(
            text,
            steering_spec=steering_spec,
            steering_position_mask=position_mask,
            max_new_tokens=int(req.max_new_tokens),
            sampler=sampler,
            stop_token_ids=stop_ids,
            answer_token_ids=req.answer_token_ids,
            on_written=on_written if stream else None,
        )
        n = len(cap.answer_token_ids)
        order, dropped = sight_order(n, req.sight_tokens, req.sight_every_token, req.max_sighted)
        capped = bool(req.sight_every_token and req.sight_tokens is None and n > len(order))
        if not stream:
            order = order[:1]
        elif wants_stop():
            # stopped while writing: sight only the first index of the order
            order = order[:1]
            tally.cancelled = True
        m = len(order)
        tally.n_to_sight = m
        answer = _answer_wire(cap)
        generation = _generation_wire(cap, req, capped)
        if stream:
            # sent before the frame is built: a failure from here on cannot lose an answer already written
            await out.json(
                {
                    "type": "theodolite_written",
                    "request_id": rid,
                    "answer": answer,
                    "generation": generation,
                    "sight_order": order,
                    "n_to_sight": m,
                    "prompt_token_ids": list(cap.prompt_token_ids),
                    "prompt_len": len(cap.prompt_token_ids),
                }
            )
            await out.json({"type": "theodolite_progress", "request_id": rid, "phase": "sighting", "done": 0, "of": m})

        shared: SweepShared = await runtime.prepare_sweep(
            cap,
            monument_concepts=monuments,
            survey_mode=req.survey_mode,
            locked_pole=locked_pole,
            tracked_concepts=tracked,
            frame_squaring=req.frame_squaring,
            frame_per_layer=req.frame_per_layer,
            frame_fnw=req.frame_fnw,
            centre_prompts=req.centre_prompts,
            centre_token_ids=req.centre_token_ids,
            centre_answer=req.centre_answer,
            **_contrast_options(req),
        )
        # the neurons a vault edit is in force on, for the label checks (no edit can land while the lock is held)
        edited = {(p.layer_idx, p.neuron_idx) for p in vault.edits_in_force()}
        request_echo = {
            "chat": bool(req.chat),
            "system": req.system if (req.chat and req.system) else None,
            "user_prompt": req.prompt,
            "assistant_prefix": req.assistant_prefix if req.assistant_prefix else None,
            "steering_applied": _describe_steering(runtime, req.steering, steering_spec),
        }

        for idx, k in enumerate(order):
            if idx > 0 and stream:
                await _let_receiver_run()
                if wants_stop():
                    tally.cancelled = True
                    break
            tally.token_index = k
            # the last pair is sent: let its bundles go before the next is read (each holds about 0.5 GB of pole
            # logits on the GPU)
            bundle = think_bundle = None  # type: ignore[assignment]
            bundle = await runtime.analyse_position(
                cap,
                shared,
                k,
                top_k_poles=15,
                pole_ranking=req.pole_ranking,
                thread_score=req.thread_score,
                rank_lookahead=req.rank_lookahead,
                edited_neurons=edited,
            )
            think_cap: Optional[SweepCapture] = None
            think_bundle: Optional[TheodoliteSweepBundle] = None
            think_skipped = False
            if idx == 0 and think_prompt is not None and stream:
                await _let_receiver_run()
                if wants_stop():
                    # a stop already asked (while the answer was written, or while this first position was read):
                    # the comparison is not begun, and the pair says so
                    think_skipped = True
                    tally.cancelled = True
            if idx == 0 and think_prompt is not None and not think_skipped:
                # written with the same sampler and seed, read in this sweep's frame and centre, at its token 0 only;
                # a stop asked while it is written cuts it short (its stop_reason "stopped", sent on the pair)
                think_cap = await runtime.capture_for_sweep(
                    think_prompt,
                    steering_spec=steering_spec,
                    steering_position_mask=position_mask,
                    max_new_tokens=think_len,
                    sampler=SamplerSettings(
                        temperature=sampler.temperature,
                        top_k=sampler.top_k,
                        top_p=sampler.top_p,
                        presence_penalty=sampler.presence_penalty,
                        seed=cap.seed,
                    ),
                    stop_token_ids=stop_ids,
                    on_written=on_thought if stream else None,
                )
                if think_cap.stop_reason == "stopped" or wants_stop():
                    # also with one sighting, where no later check would see the stop
                    tally.cancelled = True
                assert shared.basis is not None and shared.survey_labels is not None
                think_shared = await runtime.prepare_sweep(
                    think_cap,
                    monument_concepts=monuments,
                    survey_mode=req.survey_mode,
                    locked_pole=locked_pole,
                    tracked_concepts=tracked,
                    frame_squaring=req.frame_squaring,
                    frame_per_layer=req.frame_per_layer,
                    frame_fnw=req.frame_fnw,
                    frame_override=(shared.basis, shared.basis_space, dict(shared.frame_info), shared.survey_labels),
                    centre_override=shared.centre_k0,
                )
                think_bundle = await runtime.analyse_position(
                    think_cap,
                    think_shared,
                    0,
                    top_k_poles=15,
                    pole_ranking=req.pole_ranking,
                    thread_score=req.thread_score,
                    rank_lookahead=req.rank_lookahead,
                    edited_neurons=edited,
                )
            stop_after = False
            if stream and idx + 1 < m:
                await _let_receiver_run()
                stop_after = wants_stop()
            stream_info = {
                "request_id": rid,
                "token_index": bundle.token_index,
                "sight_index": idx,
                "n_to_sight": m,
                "last": idx + 1 == m or stop_after,
                "read_pos": bundle.target_pos,
                # the model's sliding window (text_config.sliding_window, 1024): from read_pos 1024 on, its 50
                # sliding-window layers no longer look back to the start of the text (the lens card's caution)
                "sliding_window": runtime.sliding_window,
                "read_token": bundle.read_token,
                "read_token_id": bundle.read_token_id,
                "prompt_len": len(cap.prompt_token_ids),
                "chosen": bundle.chosen.as_wire() if bundle.chosen is not None else None,
                "answer": answer,
                "generation": generation,
                "recapture_check": bundle.recapture_check,
                "kinship_n_states": bundle.kinship_n_states,
                "rank_lookahead": bool(req.rank_lookahead),
                "sight_dropped": list(dropped),
                "timing": {
                    "generate_ms": round(cap.generate_ms, 1),
                    "capture_ms": round(cap.capture_ms, 1),
                    "analyse_ms": round(bundle.analyse_ms, 1),
                    "total_ms": round((time.perf_counter() - t_request) * 1000.0, 1),
                },
                "think_seed": think_cap.seed if think_cap is not None else None,
                # how the thinking answer's writing ended ("end_mark" | "length" | "stopped"); null without one
                "think_stop_reason": think_cap.stop_reason if think_cap is not None else None,
                # true on the first pair when a stop asked before the comparison began left it unwritten
                "think_skipped": think_skipped,
                "think_read_token": (
                    runtime.tokenizer.decode([think_cap.prompt_token_ids[-1]], clean_up_tokenization_spaces=False)
                    if think_cap is not None
                    else None
                ),
                # how the thinking answer was switched on ("template" | "bare_prefix"), beside the other think_* keys:
                # null without one (later pairs, skipped, or not asked)
                "think_protocol": think_protocol if think_cap is not None else None,
                # "bare_text" on the first pair when the comparison was asked for and left out (think_bare "none")
                "think_left_out": think_left_out if idx == 0 else None,
            }
            binary_frame, json_frame = pack_binary_and_json_frames(
                runtime=runtime,
                vault=vault,
                sweep=bundle,
                think_sweep=think_bundle,
                is_steered=(steering_spec is not None),
                request_echo=request_echo,
                stream_info=stream_info,
                flags_extra=4 if m > 1 else 0,
            )
            await out.pair(binary_frame, json_frame)
            tally.n_sent += 1
            if stop_after:
                tally.cancelled = True
                break


async def _run_sweep(
    out: _Outbox,
    conn: _SocketState,
    runtime: TheodoliteRuntime,
    vault: LarqlVaultEngine,
    req: SweepRequest,
) -> None:
    """One sweep request. Without a request_id it is answered as the instrument always answered: one pair, and a
    failure goes to the handler, which refuses it as before. With one, any failure is reported on this request alone
    (an error, then its sweep_end) and the socket stays open; the sweep_end always comes last."""
    tally = _Tally()
    if req.request_id is None:
        try:
            await _stream_sweep(out, conn, runtime, vault, req, tally)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except RuntimeError as exc:
            if cpu_memory.is_cpu_oom(exc):  # refused in plain words, the socket kept, as a ValueError is
                raise ValueError(_plain_error(exc)) from exc
            raise
        return
    rid = req.request_id
    error: Optional[str] = None
    try:
        await _stream_sweep(out, conn, runtime, vault, req, tally)
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        if conn.closed:
            logger.info("Sweep %s stopped: the client has gone (%s)", rid, exc)
            raise WebSocketDisconnect(code=1001) from exc
        if isinstance(exc, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        error = _plain_error(exc)
        if isinstance(exc, ValueError):
            logger.warning("WebSocket sweep %s refused: %s", rid, exc)
        else:
            logger.exception("WebSocket sweep %s failed after %d pair(s)", rid, tally.n_sent)
        await out.json(
            {
                "type": "error",
                "action": "sweep",
                "request_id": rid,
                "recoverable": True,
                "message": error,
                "n_sent": tally.n_sent,
                "token_index": tally.token_index,
            }
        )
    await out.json(
        {
            "type": "theodolite_sweep_end",
            "request_id": rid,
            "n_sent": tally.n_sent,
            "n_to_sight": tally.n_to_sight,
            "cancelled": tally.cancelled,
            "error": error,
        }
    )


def _raw_request_id(raw_msg: Any) -> Optional[str]:
    """A sweep message's request_id as sent, before any validation (so a refusal can name it)."""
    if not isinstance(raw_msg, dict):
        return None
    payload = raw_msg.get("payload", raw_msg)
    rid = payload.get("request_id") if isinstance(payload, dict) else None
    return rid if isinstance(rid, str) else None


async def _handle_sweep(
    out: _Outbox,
    conn: _SocketState,
    runtime: TheodoliteRuntime,
    vault: LarqlVaultEngine,
    raw_msg: Dict[str, Any],
) -> None:
    rid = _raw_request_id(raw_msg)
    if rid is not None and conn.dequeue(rid):
        return  # cancelled while it waited: its sweep_end has been sent
    try:
        req = SweepRequest(**raw_msg.get("payload", raw_msg))
    except ValidationError as exc:
        if rid is None:
            raise
        message = _plain_validation(exc)
        logger.warning("WebSocket sweep %s refused: %s", rid, message)
        await out.json(
            {
                "type": "error",
                "action": "sweep",
                "request_id": rid,
                "recoverable": True,
                "message": message,
                "n_sent": 0,
                "token_index": None,
            }
        )
        await out.json(
            {"type": "theodolite_sweep_end", "request_id": rid, "n_sent": 0, "n_to_sight": 0, "cancelled": False, "error": message}
        )
        return
    if req.request_id is None:
        await _run_sweep(out, conn, runtime, vault, req)
        return
    conn.running, conn.cancel_running = req.request_id, False
    try:
        await _run_sweep(out, conn, runtime, vault, req)
    finally:
        conn.running, conn.cancel_running = None, False


@dataclass
class _Ended:
    """Put on the queue when the receiver stops: the client left, or sent a frame that is not JSON."""

    exc: BaseException


async def _receive_into(websocket: WebSocket, queue: asyncio.Queue, conn: _SocketState, out: _Outbox) -> None:
    """Read the socket into `queue`, handling cancel_sweep at once: a running request's cancel flag is set (it stops
    at its next check), a queued one is dropped and its sweep_end sent now."""
    try:
        while True:
            msg = await websocket.receive_json()
            if isinstance(msg, dict) and msg.get("action") == "cancel_sweep":
                rid = msg.get("request_id")
                if not isinstance(rid, str):
                    continue
                if conn.running == rid:
                    conn.cancel_running = True
                elif conn.drop_queued(rid):
                    await out.json(
                        {"type": "theodolite_sweep_end", "request_id": rid, "n_sent": 0, "n_to_sight": 0, "cancelled": True, "error": None}
                    )
                continue
            if isinstance(msg, dict) and msg.get("action", "sweep") == "sweep":
                rid = _raw_request_id(msg)
                if rid is not None:
                    conn.enqueue(rid)
            await queue.put(msg)
    except WebSocketDisconnect as exc:
        conn.closed = True
        await queue.put(_Ended(exc))
    except Exception as exc:
        await queue.put(_Ended(exc))


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
    on_gpu = torch.cuda.is_available() and runtime.device.type == "cuda"  # a CPU runtime has no card to report
    vram_alloc = torch.cuda.memory_allocated(runtime.device) / (1024**3) if on_gpu else 0.0
    vram_peak = torch.cuda.max_memory_allocated(runtime.device) / (1024**3) if on_gpu else 0.0
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
        # the CUDA allocator's settings as torch reports them ("" for its defaults); the sweep limit's memory margin
        # was measured with expandable segments on (engine_runtime, the note above `import torch`)
        "allocator": runtime.allocator_settings,
        # how the 4-bit weights are unpacked (backend/w4a16_fast.py; $THEODOLITE_UNPACKER): every unpacker the same bits
        "unpacker": w4a16_fast.describe(runtime),
        # the model's sliding attention window in word-pieces (text_config.sliding_window), as every sighting sends it
        "sliding_window": runtime.sliding_window,
        # where the model runs, and what its weights are (runtime_factory.weights_card: a GGUF's file provenance, the
        # W4A16 checkpoint's repo id and the snapshot the cache holds), as every index built on it records them
        "device": str(runtime.device),
        "weights": weights_card(runtime),
        "larql_cli": vault.cli_versions(),
        # the causal spot-check (backend/causal_check.py): the WebSocket action "causal_check" and POST
        # /api/scientist/causal_check exist on this server (an older one ignores the action without a word)
        "causal_check": {
            "version": cc.CAUSAL_CHECK_VERSION,
            "writes": list(cc.CAUSAL_WRITES),
            "contexts": list(cc.CAUSAL_CONTEXTS),
            "scale_refs": list(cc.SCALE_REFS),
            "k_default": cc.K_DEFAULT,
            "k_max": cc.K_MAX,
            "k_floor": cc.K_FLOOR,
            "n_study_texts": len(cc.STUDY_TEXTS),
        },
        # one word against another (a sweep's `contrasts`): what a request may hold, and why the "study_mean"
        # background cannot be read here, in plain words (null when it can: E3's average state is the file pinned)
        "contrasts": {
            "backgrounds": list(CONTRAST_BACKGROUNDS),
            "max_contrasts": 4,
            "max_words": 12,
            "study_mean_problem": study_mean_problem(),
        },
        # the atlas (backend/atlas.py, the page at #/atlas): that its routes exist on this server, which of its
        # indexes are on disk (by file, none loaded) and where the field notes are kept
        "atlas": atlas_routes.health_card(runtime),
        # the rule the label_check on every neuron label follows (backend/label_gate.py)
        "label_gate": LABEL_GATE_CARD,
        "lens": runtime.lens_card(),
    }


def _execute_lql(vault: LarqlVaultEngine, query: str, voice: Optional[VoiceRequest]) -> Dict[str, Any]:
    """The statement, heard when a voice is given (the caller holds `_GPU_LOCK` around the voice before, the edit and
    the voice after); without one, called exactly as before."""
    if voice is None:
        return vault.execute_lql(query)
    return vault.execute_lql(query, voice=voice.model_dump())


async def _execute_lql_off_loop(vault: LarqlVaultEngine, query: str, voice: Optional[VoiceRequest]) -> Dict[str, Any]:
    """`_execute_lql` in a worker thread, so the event loop keeps answering (health, other sockets, a socket's own
    cancel and keepalive) while a heard edit writes its two answers, which takes seconds (`write_voice` is synchronous:
    one default heard edit took 12.6 s on the GPU). The caller still holds `_GPU_LOCK` around it, so no sweep, scientist
    route or other statement runs on the GPU meanwhile."""
    return await asyncio.to_thread(_execute_lql, vault, query, voice)


@app.post("/api/larql")
async def api_larql(req: LarqlRequest) -> Dict[str, Any]:
    _runtime, vault = get_app_state()
    async with _GPU_LOCK:
        try:
            return await _execute_lql_off_loop(vault, req.query, req.voice)
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
                token_ids=req.token_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _causal_check(req: CausalCheckRequest) -> Dict[str, Any]:
    """The causal spot-check under the GPU lock, passed on unchanged (the tool refuses in plain words, ValueError, and
    turns running out of GPU memory into one)."""
    async with _GPU_LOCK:
        return await causal_spot_check(
            layer=req.layer,
            neuron_idx=req.neuron_idx,
            polarity=req.polarity,
            contexts=req.contexts,
            token_ids=req.token_ids,
            prompt=req.prompt,
            chat=req.chat,
            system=req.system,
            assistant_prefix=req.assistant_prefix,
            k=req.k,
            write=req.write,
            scale_ref=req.scale_ref,
            floor=req.floor,
            top_n=req.top_n,
            track_ids=req.track_ids,
        )


@app.post("/api/scientist/causal_check")
async def api_causal_check(req: CausalCheckRequest) -> Dict[str, Any]:
    _runtime, _vault = get_app_state()
    try:
        return await _causal_check(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scientist/test_hypothesis")
async def api_test_hypothesis(req: MotifHypothesisRequest) -> Dict[str, Any]:
    _runtime, _vault = get_app_state()
    try:
        async with _GPU_LOCK:
            return await test_motif_hypothesis(
                layer=req.layer,
                neuron_idx=req.neuron_idx,
                pos_prompts=req.pos_prompts,
                neg_prompts=req.neg_prompts,
                chat=req.chat,
                system=req.system,
            )
    except ValueError as exc:  # its refusals, in plain words: an empty side, a layer or neuron that does not exist
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/theodolite")
async def ws_theodolite(websocket: WebSocket) -> None:
    """Real-time Binary Float32 + JSON WebSocket stream for the 3D Theodolite frontend.

    A receiver task reads the socket into a queue, so a `cancel_sweep` is handled at once, even mid-sweep; every
    other action is answered in the order it came. Every send goes through one outbox, so a pair's binary frame and
    its JSON leave back to back. A handshake from anywhere but this machine never reaches here: LocalOnly closes it
    (1008) before it is accepted."""
    await websocket.accept()
    runtime, vault = get_app_state()
    logger.info("WebSocket client connected to /ws/theodolite")
    out = _Outbox(websocket)
    conn = _SocketState()
    queue: asyncio.Queue = asyncio.Queue()
    receiver = asyncio.create_task(_receive_into(websocket, queue, conn, out))

    try:
        while True:
            raw_msg = await queue.get()
            if isinstance(raw_msg, _Ended):
                raise raw_msg.exc
            action = raw_msg.get("action", "sweep")

            try:
                if action == "sweep":
                    await _handle_sweep(out, conn, runtime, vault, raw_msg)

                elif action == "larql":
                    query = raw_msg.get("query", "DESCRIBE")
                    try:
                        # the voice alone is checked here; the statement goes to the vault as it always did
                        voice = LarqlRequest(voice=raw_msg.get("voice")).voice
                    except ValidationError as exc:
                        why = _plain_validation(exc, "The voice was refused, and no edit was made.")
                        await out.json({"type": "larql_result", "ok": False, "error": why})
                        continue
                    async with _GPU_LOCK:
                        try:
                            res = await _execute_lql_off_loop(vault, query, voice)
                            await out.json({"type": "larql_result", "ok": True, "data": res})
                        except Exception as exc:
                            await out.json({"type": "larql_result", "ok": False, "error": str(exc)})

                elif action == "inspect_pole":
                    l_idx = int(raw_msg.get("layer", 35))
                    n_idx = int(raw_msg.get("neuron_idx", 0))
                    pol = float(raw_msg.get("polarity", 1.0))
                    async with _GPU_LOCK:
                        res = inspect_pole(layer=l_idx, neuron_idx=n_idx, polarity=pol)
                    await out.json({"type": "pole_inspection", "data": res})

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
                            token_ids=tr.token_ids,
                        )
                    await out.json({"type": "downstream_readers", "data": res})

                elif action == "causal_check":
                    try:
                        cc_req = CausalCheckRequest(**{k: v for k, v in raw_msg.items() if k != "action"})
                    except ValidationError as exc:
                        raise ValueError(_plain_validation(exc, "The test was refused.")) from exc
                    res = await _causal_check(cc_req)
                    await out.json({"type": "causal_check", "data": res})
            except (ValidationError, ValueError) as exc:
                # A malformed request (a bad option value, say) fails alone and the socket stays open; the reply
                # names the action and its neuron so the client can fail just that request.
                logger.warning("WebSocket %s request refused: %s", action, exc)
                await out.json(
                    {
                        "type": "error",
                        "message": str(exc),
                        "action": action,
                        "recoverable": True,
                        "layer": raw_msg.get("layer"),
                        "neuron_idx": raw_msg.get("neuron_idx"),
                        # which of a neuron's tests failed (causal_check: "given" | "study"); null for other actions
                        "contexts": raw_msg.get("contexts"),
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
    finally:
        conn.closed = True
        receiver.cancel()


# Saved questions and system prompts (backend/saved_prompts.py) and the atlas (backend/atlas.py, under the one GPU
# lock): declared before the static mount below, which would otherwise answer every path under "/".
app.include_router(saved_prompts_router)
atlas_routes.bind(get_app_state, _GPU_LOCK)
app.include_router(atlas_routes.router)
# the atlas's scientist chat (backend/scientist_chat.py): an OpenAI-compatible model's tool calls come back through
# this app's own routes
scientist_chat.bind(app)
app.include_router(scientist_chat.router)

_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")

