"""The atlas's routes (`backend/atlas.py`): every neuron of the model laid out, wired, pushed and read, over HTTP.

The page at `#/atlas` (frontend/src/screens/Atlas.tsx) and its scientist chat (backend/scientist_chat.py) read the model
through these routes. They give the instrument the explorer of lyramakesmusic/neuron-explorer (explorer/server.py,
explorer/runtime.py), built for Qwen3.5-4B, on Gemma-4-31B, with the instrument's own rules:
- every route that runs the model holds the server's GPU lock (`bind`), and does its work in a worker thread, so the
  server keeps answering its health route and its sockets while the GPU works; a push chat holds the lock for as long
  as it writes, and stops when its page goes away;
- every neuron label read through the J++ lens or by the naked eye is sent with its `check` (backend/label_gate.py
  `label_check`), and a pooled summary pools only labels the gate lets through; the words are still sent, the pages
  keep held-back ones in the apparatus;
- layers count from 0 in every field (page layer = layer + 1), as everywhere in the server; the scientist's tools turn
  them into the pages' count (backend/scientist_mcp.py);
- a refusal is answered in plain words (400), and a request from a page served anywhere but this machine is refused
  (403, `local_pages_only`), since these routes run the model and, through the scientist chat, a shell.

What is read once for every neuron comes from the atlas's indexes (backend/atlas_index.py, built by
scripts/atlas_build.py); a route that needs one that is not built says so (`ready: false`, or the live reading it falls
back to, named in `source`). The push's unit (the stream's median length leaving each layer) and the activation scale
come from the stats index when it is built, else from the study's 36 texts, read once per set of vault edits in force
(`scale_source` says which). The routes are in ARCHITECTURE.md §4.7.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import anyio
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, StrictInt

from backend import atlas_ops as ao
from backend import census as cz
from backend import field_notes as fn
from backend import sampler_request as sr
from backend.atlas_index import FILES, LENS_K, LINK_MODELS, SITE_SCORE_RERUN, AtlasIndex, match_ids, scale_scores
from backend.causal_check import language_model
from backend.label_gate import LABEL_GATE_CARD
from backend.saved_prompts import LOCAL_HOSTS, _hostname

logger = logging.getLogger("theodolite.atlas")

ATLAS_VERSION = 1
MAX_TEXT_POSITIONS = 2048  # a text read once, its activations of one neuron kept (as MAX_SWEEP_POSITIONS)
MAX_RUN_POSITIONS = 256  # a text whose every activation is kept for the run view: 60 x 21,504 x 2 bytes a piece
MAX_HEAD_POSITIONS = 512  # a text whose attention is kept head by head (about 2.5 GB on the CPU at 512 on the 31B)
MAX_PUSH_NEW = 512
# whole-text steer samples: 3 rows each in one batch (on the 31B a step costs about the same for 1 to 64 rows; each row
# keeps about 0.8 MB a word-piece of context, so 24 rows of 200 are about 3.8 GB, but 24 of 512 written would be about
# 10 GB: inferred from the census pilot's 3.1 GiB for 32 rows, not measured). Several samples are written only while
# their rows hold at most what the push chat's three voices may (3 x 2,048 word-pieces, prompt and answer, about 4.9 GB);
# one sample is taken as before
MAX_STEER_SAMPLES = 8
MAX_STEER_ROW_PIECES = 3 * MAX_TEXT_POSITIONS
MAX_COMPARE_NEURONS = 300
MAX_COMPARE_TEXTS = 24
CACHE_SIZE = 256


# -------------------------------------------------------------------------------------------------- who may ask


def local_pages_only(request: Request) -> None:
    """Refuse a request from a page served anywhere but this machine (403). These routes run the model and, through the
    scientist chat, a shell, so beside the saved prompts' rule (the Host must name this machine, and so must the
    Origin when there is one: `saved_prompts.same_machine`) a request a browser marks as sent from another site is
    refused even without an Origin (a page's <img> or <script> sends none), unless it is a plain navigation (a link
    opened from elsewhere reads nothing back), as the reference's `local_only` does. The peer itself must be this machine
    too, whatever the headers say, should the server ever listen beyond loopback. Which account on this machine asks
    is not this rule's: the server's guard asks that too, on Linux (backend/local_only.py `other_account`), before any
    route; what the scientist's processes may do is bounded by their reach (backend/scientist_chat.py)."""
    peer = request.client.host if request.client else None
    try:
        ip = ipaddress.ip_address(peer) if peer else None
    except ValueError:
        ip = None  # not an address (an in-process test client names itself)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip is not None and not ip.is_loopback:
        raise HTTPException(status_code=403, detail="The atlas answers only the instrument's own pages, served from this machine.")
    host_header = request.headers.get("host")
    origin = request.headers.get("origin")
    site = request.headers.get("sec-fetch-site")
    refused = (
        (host_header is not None and _hostname("//" + host_header) not in LOCAL_HOSTS)
        or (origin is not None and _hostname(origin) not in LOCAL_HOSTS)
        or (site in ("cross-site", "same-site") and not (request.method == "GET" and request.headers.get("sec-fetch-mode") == "navigate"))
    )
    if refused:
        raise HTTPException(status_code=403, detail="The atlas answers only the instrument's own pages, served from this machine.")


router = APIRouter(prefix="/api/atlas", dependencies=[Depends(local_pages_only)])

_get_state: Optional[Callable[[], Tuple[Any, Any]]] = None
_gpu_lock: Optional[asyncio.Lock] = None
_atlas: Optional["Atlas"] = None


def bind(get_state: Callable[[], Tuple[Any, Any]], gpu_lock: asyncio.Lock) -> None:
    """Called by backend/server.py: how to reach the runtime and the vault, and the one GPU lock every route shares."""
    global _get_state, _gpu_lock
    _get_state, _gpu_lock = get_state, gpu_lock


def health_card(rt: Any) -> Dict[str, Any]:
    """What /api/health says of the atlas: that its routes exist here, which indexes are on disk (by file: none is
    loaded for it; GET /arch says whether each fits this model), which layers have a census of this model, and where
    the field notes, the folio's field book and the scientist's proposals are kept."""
    root = AtlasIndex().root
    return {"version": ATLAS_VERSION, "routes": router.prefix, "index_folder": str(root),
            "indexes": {name: (root / f).exists() for name, f in FILES.items()},
            "census_layers": sorted(cz.layer_runs(model=getattr(rt, "model_id", None))), "field_notes": str(fn.field_notes_path()),
            "field_book": str(fn.field_book_path()), "field_proposals": str(fn.field_proposals_path())}


def atlas() -> "Atlas":
    global _atlas
    if _get_state is None:
        raise HTTPException(status_code=503, detail="The atlas is not bound to a server.")
    rt, vault = _get_state()
    if _atlas is None or _atlas.rt is not rt or _atlas.vault is not vault:
        _atlas = Atlas(rt, vault)
    return _atlas


async def on_gpu(fn_: Callable[..., Any], *args: Any) -> Any:
    """`fn_(*args)` in a worker thread under the GPU lock, its ValueError answered as a plain-word 400."""
    assert _gpu_lock is not None
    async with _gpu_lock:
        try:
            return await asyncio.to_thread(fn_, *args)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=400, detail="The GPU ran out of memory for this reading; try a shorter text.") from exc
        except RuntimeError as exc:
            from backend.cpu_memory import is_cpu_oom, plain

            if not is_cpu_oom(exc):
                raise
            raise HTTPException(status_code=400, detail=f"{plain(exc, 'this reading')} Try a shorter text.") from exc


def off_gpu(fn_: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn_(*args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class LRU(OrderedDict):
    def __init__(self, size: int = CACHE_SIZE) -> None:
        super().__init__()
        self.size = size

    def put(self, key: Any, value: Any) -> Any:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)
        return value


# ------------------------------------------------------------------------------------------------------- the state


class Atlas:
    """What the routes share for one runtime: the model's shape, the indexes, the vocabulary's strings, the push's unit
    and the activation scale (per set of vault edits in force), the last text run, and small caches."""

    def __init__(self, rt: Any, vault: Any) -> None:
        self.rt, self.vault = rt, vault
        self.lm = language_model(rt.hf_model)
        self.w = ao.Weights(self.lm, rt)
        self.L, self.N, self.D = self.w.n_layers, self.w.d_ff, self.w.d_model
        self.shapes = [ao.attn_shape(self.lm, l) for l in range(self.L)]
        self.spaces = ao.HeadSpaces(self.w, self.shapes)
        from backend.runtime_factory import weights_card  # an index built from other weights is refused, not read

        self.index = AtlasIndex(model=getattr(rt, "model_id", None), shape=(self.L, self.N), weights=weights_card(rt))
        self.W_U = rt.lm_head_weight
        self._vocab: Optional[List[str]] = None
        self._vocab_lock = threading.Lock()
        self._scales: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.cache: Dict[str, LRU] = {k: LRU() for k in ("wiring", "roles", "head", "steer")}
        self.run: Dict[str, Any] = {}
        self.caps: LRU = LRU(2)  # head captures: each can take gigabytes on the 31B
        self._profiles: Optional[Dict[str, Any]] = None

    # -- the vault's edits decide what is stale
    def edit_key(self) -> Tuple[int, ...]:
        return tuple(int(p.patch_id) for p in self.vault.edits_in_force())

    def edited(self) -> set:
        return {(p.layer_idx, p.neuron_idx) for p in self.vault.edits_in_force()}

    # -- words
    def vocab(self) -> List[str]:
        """Every word-piece decoded alone, once (cached in memory and as the index folder's vocab.json)."""
        with self._vocab_lock:
            if self._vocab is None:
                tok = self.rt.tokenizer
                V = int(self.W_U.shape[0])
                path = self.index.root / "vocab.json"
                key = f"{getattr(self.rt, 'model_id', '')}:{V}"
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if data.get("key") == key and len(data.get("vocab", [])) == V:
                        self._vocab = data["vocab"]
                except (OSError, ValueError):
                    pass
                if self._vocab is None:
                    self._vocab = [tok.decode([i]) for i in range(V)]
                    if not getattr(self.rt, "atlas_stand_in", False):
                        try:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps({"key": key, "vocab": self._vocab}, ensure_ascii=False), encoding="utf-8")
                        except OSError:
                            pass
            return self._vocab

    def words(self, ids: Sequence[int], k: int = 6) -> List[str]:
        v = self.vocab()
        return [v[int(t)] if 0 <= int(t) < len(v) else "" for t in list(ids)[:k]]

    def face(self, layer: int, n: int, sign: str, k: int = 6, edited: Optional[set] = None) -> Dict[str, Any]:
        """A face's words from the indexes: through the lens, by the naked eye (each with the label gate's check of the
        two), and the steer lens's rises when it is built (a measured effect, which the gate does not cover)."""
        out: Dict[str, Any] = {"lens": None, "eye": None, "check": None, "steer": None}
        is_edited = (layer, n) in (edited if edited is not None else self.edited())
        lens_ids = self.index.face_ids(layer, n, sign, "lens", k)
        if lens_ids is not None:
            out["lens"] = self.words(lens_ids, k)
            out["eye"] = self.words(self.index.face_ids(layer, n, sign, "eye", k) or [], k)
            out["check"] = self.index.face_check(layer, n, sign, edited=is_edited)
        st = self.index.get("steer_lens")
        if st is not None:
            out["steer"] = self.words(st["up" if sign == "+" else "down"][layer, n, :k], k)
        if is_edited and (lens_ids is not None or st is not None):
            out["stale"] = "read before the vault edit in force on this neuron; GET /neuron/lens reads it as it now is"
        return out

    # -- the push's unit and the activation scale
    def scales(self) -> Dict[str, Any]:
        """{"p50": [L], "pos": [L, N], "neg": [L, N], "source": {...}}: from the stats index when built, else read over
        the study's 36 texts for the model as it now stands (per set of vault edits in force). Call under the GPU lock
        (the fallback runs the model)."""
        s = self.index.get("act_scale")
        if s is not None:
            return {"p50": [float(x) for x in s["resid_q"][:, 1]], "pos": s["pos"], "neg": s["neg"],
                    "mlp_rms": [float(x) for x in s["mlp_rms_q"][:, 1]], "mid_rms": [float(x) for x in s["mid_rms_q"][:, 1]],
                    "regimes": {k: s[k] for k in ("pp", "nn", "pn", "np")},
                    "source": {"kind": "index", "corpus": s["meta"].get("corpus"), "tokens": int(s["tokens"]),
                               "built_at": s["meta"].get("built_at")}}
        key = self.edit_key()
        if key not in self._scales:
            ids = study_ids(self.rt)
            ex = ao.activation_extremes(self.lm, ids)
            self._scales = {k2: v for k2, v in self._scales.items() if not k2}  # keep the clean model's
            self._scales[key] = {"p50": ao.stream_p50(ex), "pos": ex.pos.numpy(), "neg": ex.neg.numpy(), **ao.norm_scales(ex),
                                 "regimes": {k: getattr(ex, k).numpy() for k in ("pp", "nn", "pn", "np")},
                                 "source": {"kind": "study_texts", "corpus": "the study's 36 texts (provisional)",
                                            "tokens": ex.tokens, "edits_in_force": list(key)}}
        return self._scales[key]

    def link_scale(self) -> torch.Tensor:
        """`atlas_ops.link_scale` for the model as it now stands (the scales above; call under the GPU lock)."""
        sc = self.scales()
        return ao.link_scale(self.w.scalars, sc["mlp_rms"], sc["mid_rms"])

    def act_range(self, layer: int, n: int, sc: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        sc = sc or self._scales_if_ready()
        if sc is None:
            return None
        out = {"act_max": round(float(sc["pos"][layer, n]), 4), "act_min": round(float(sc["neg"][layer, n]), 4),
               "over_tokens": sc["source"]["tokens"], "source": sc["source"]["corpus"]}
        if sc.get("regimes") is not None:
            # each GeGLU sign regime's extreme: (+gate, +up), (-gate, -up) on the positive side; (+, -), (-, +) negative
            out["regimes"] = {k: round(float(v[layer, n]), 4) for k, v in sc["regimes"].items()}
        return out

    def _scales_if_ready(self) -> Optional[Dict[str, Any]]:
        s = self.index.get("act_scale")
        if s is not None:
            return self.scales()
        return self._scales.get(self.edit_key())

    # -- text
    def ids_of(self, text: str, chat: bool, prefix: str = "", system: Optional[str] = None, limit: int = MAX_TEXT_POSITIONS) -> List[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("Give a text to read.")
        if chat:
            ids = self.rt.tokenize_prompt(self.rt.compose_prompt(text, chat=True, system=system, assistant_prefix=prefix))
        else:
            ids = self.rt.tokenize_prompt(text + prefix)
        if len(ids) > limit:
            raise ValueError(f"The text is {len(ids)} word-pieces; this reading takes at most {limit}. Shorten it.")
        return ids

    def chat_ids(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> List[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        text = self.rt.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.rt.tokenize_prompt(text)

    def units(self, ids: Sequence[int]) -> List[Tuple[str, List[int]]]:
        return ao.token_units(self.rt.tokenizer, ids)

    # -- interventions
    def parse_pushes(self, pushes: Sequence["PushIn"], heads: Sequence["HeadIn"] = ()) -> Tuple[List[ao.Push], List[ao.HeadScale]]:
        """Pushes and head scales from a request, checked (a ValueError in plain words); nothing runs the model."""
        ps: List[ao.Push] = []
        hs: List[ao.HeadScale] = []
        for p in pushes:
            if p.head is not None:
                hs.append(ao.HeadScale(int(p.layer), None if p.head == "all" else _head_index(p.head), float(p.dose if p.dose is not None else 0.0)))
                continue
            if p.all:
                ps.append(ao.Push(int(p.layer), None, "mul", float(p.dose if p.dose is not None else 0.0)))
            else:
                if p.n is None:
                    raise ValueError("A push names its neuron (n), the whole layer's block (all: true), or a head (head).")
                mode = p.mode or "add"
                dose = p.dose if p.dose is not None else (2.0 if mode == "mul" else 0.8)
                ps.append(ao.Push(int(p.layer), int(p.n), mode, float(dose)))
        for h in heads:
            hs.append(ao.HeadScale(int(h.layer), None if h.head in ("all", None) else _head_index(h.head), float(h.dose)))
        ao.check_pushes(self.lm, ps, hs)
        return ps, hs

    def interventions(self, ps: Sequence[ao.Push], hs: Sequence[ao.HeadScale], signs: Sequence[int]) -> Tuple[Optional[ao.Interventions], List[Dict[str, Any]]]:
        """Checked pushes and head scales resolved to per-row values (an "add" or "act" push needs the scales, which can
        run the model: call under the GPU lock); and their echo."""
        if not ps and not hs:
            return None, []
        sc = self.scales() if any(p.mode in ("add", "act") for p in ps) else None
        iv = ao.resolve(
            self.w, ps, hs, signs,
            p50=lambda l: sc["p50"][l] if sc else 1.0,
            act_scale=lambda l, n: max(abs(float(sc["pos"][l, n])), abs(float(sc["neg"][l, n])), 1e-6) if sc else 1.0,
        )
        echo = [{"layer": p.layer, "n": p.neuron, "mode": p.mode, "dose": p.dose} for p in ps]
        echo += [{"layer": h.layer, "head": "all" if h.head is None else h.head, "scale": h.scale} for h in hs]
        return iv, echo


def _head_index(h: Any) -> int:
    if isinstance(h, bool) or not isinstance(h, (int, str)) or (isinstance(h, str) and not h.isdigit()):
        raise ValueError(f'A head is a number counting from 0, or "all", not {h!r}.')
    return int(h)


def study_ids(rt: Any) -> List[List[int]]:
    """The study's 36 texts as the model reads them (as the causal check reads them: mcp_scientist._study_ids)."""
    from backend.mcp_scientist import _study_ids

    return _study_ids(rt)


# --------------------------------------------------------------------------------------------------- request bodies


class PushIn(BaseModel):
    layer: StrictInt
    n: Optional[StrictInt] = None
    all: bool = False  # the whole layer's feed-forward block (mode "mul")
    mode: Optional[str] = Field(None, max_length=8)  # "add" | "mul" | "act"
    dose: Optional[float] = None
    head: Optional[Union[StrictInt, str]] = None  # a head scaled instead (dose = its factor; "all" = every head)


class HeadIn(BaseModel):
    layer: StrictInt
    head: Optional[Union[StrictInt, str]] = None
    dose: float = 0.0


class Turn(BaseModel):
    user: str = Field(..., max_length=32000)
    replies: List[str] = Field(..., min_length=3, max_length=3)


class PushChatRequest(BaseModel):
    layer: Optional[StrictInt] = None
    n: Optional[StrictInt] = None
    neurons: List[PushIn] = Field(default_factory=list, max_length=ao.MAX_PUSHES)
    prompt: str = Field(..., min_length=1, max_length=32000)
    history: List[Turn] = Field(default_factory=list, max_length=32)
    system: Optional[str] = Field(None, max_length=32000)
    dose: float = 0.8
    mode: str = Field("add", max_length=8)
    max_new: int = Field(160, ge=1, le=MAX_PUSH_NEW)
    spare_bos: bool = True
    # how the voices are written: greedy unless a temperature above 0 is given (backend/sampler_request.py, read in
    # plain words there, a sampler that is not an object among them); the last event says how they were written
    sampler: Any = None


class ReadRequest(BaseModel):
    layer: StrictInt
    n: StrictInt
    text: Optional[str] = Field(None, min_length=1, max_length=64000)
    ids: Optional[List[StrictInt]] = Field(None, max_length=65536)  # word-piece ids read exactly (at most 2,048: `_read`)
    chat: bool = False
    neurons: List[PushIn] = Field(default_factory=list, max_length=ao.MAX_PUSHES)
    heads: List[HeadIn] = Field(default_factory=list, max_length=ao.MAX_HEAD_SCALES)
    spare_bos: bool = True


class LogprobsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=64000)
    chat: bool = True
    prefix: str = Field("", max_length=32000)
    k: int = Field(15, ge=1, le=100)
    targets: List[str] = Field(default_factory=list, max_length=32)
    neurons: List[PushIn] = Field(default_factory=list, max_length=ao.MAX_PUSHES)
    heads: List[HeadIn] = Field(default_factory=list, max_length=ao.MAX_HEAD_SCALES)
    spare_bos: bool = True


class TextIn(BaseModel):
    label: Optional[str] = Field(None, max_length=80)
    text: str = Field(..., min_length=1, max_length=64000)


class CompareRequest(BaseModel):
    neurons: List[PushIn] = Field(..., min_length=1, max_length=MAX_COMPARE_NEURONS)
    texts: List[Union[str, TextIn]] = Field(..., min_length=1, max_length=MAX_COMPARE_TEXTS)
    chat: bool = False


class FiresRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=64000)
    chat: bool = False
    token: Optional[str] = Field(None, max_length=200)
    at: Optional[StrictInt] = None
    limit: int = Field(30, ge=1, le=300)
    layers: Optional[List[StrictInt]] = Field(None, min_length=2, max_length=2)


class RunRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=64000)
    chat: bool = False


class HeadTextRequest(BaseModel):
    layer: StrictInt
    head: StrictInt
    text: Optional[str] = Field(None, max_length=64000)
    chat: bool = False
    token: Optional[str] = Field(None, max_length=200)
    at: Optional[StrictInt] = None


class DriveRequest(BaseModel):
    layer: StrictInt
    n: StrictInt
    text: Optional[str] = Field(None, min_length=1, max_length=64000)
    ids: Optional[List[StrictInt]] = Field(None, max_length=65536)  # word-piece ids read exactly (at most 512: `_drive`)
    chat: bool = False
    token: Optional[str] = Field(None, max_length=200)
    at: Optional[StrictInt] = None
    top: int = Field(8, ge=1, le=32)


class SteerRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    layer: Optional[StrictInt] = None
    n: Optional[StrictInt] = None
    neurons: List[PushIn] = Field(default_factory=list, max_length=ao.MAX_PUSHES)
    dose: float = 0.8
    mode: str = Field("add", max_length=8)
    max_new: int = Field(150, ge=1, le=MAX_PUSH_NEW)
    sampler: Any = None  # as the push chat's
    samples: StrictInt = 1  # 1 to MAX_STEER_SAMPLES, sampled only: sample s drawn with seed + s, 3 rows each in one batch


class NoteRequest(BaseModel):
    kind: Optional[str] = None
    layer: Optional[StrictInt] = None
    index: Optional[StrictInt] = None
    name: Optional[str] = None
    note: Optional[str] = None
    by: Optional[str] = None


class FieldBookRequest(BaseModel):
    layer: Optional[StrictInt] = None
    n: Optional[StrictInt] = None
    face: Optional[str] = None
    # checked by field_notes.check_face, so a wrong value is answered in its words, and a field left out is kept
    jobs: Optional[Any] = None
    pattern: Optional[Any] = None
    checked: Optional[Any] = None
    by: Optional[str] = None


class ProposalRequest(BaseModel):
    kind: Optional[str] = None
    layer: Optional[StrictInt] = None
    index: Optional[StrictInt] = None
    name: Optional[str] = None
    note: Optional[str] = None
    chat: Optional[str] = None  # its limits checked by `field_notes.propose`, so a refusal says what was wrong in plain words
    part: Optional[str] = None  # keep or discard only the name, or only the note (both when left out)


def _check_neuron(a: Atlas, layer: int, n: int) -> None:
    if not 0 <= int(layer) < a.L:
        raise HTTPException(status_code=400, detail=f"Layer {layer} does not exist: layers count from 0 to {a.L - 1} here (page layers 1 to {a.L}).")
    if not 0 <= int(n) < a.N:
        raise HTTPException(status_code=400, detail=f"Neuron {n} does not exist: each layer has {a.N} (0 to {a.N - 1}).")


def _check_head(a: Atlas, layer: int, head: int) -> None:
    if not 0 <= int(layer) < a.L:
        raise HTTPException(status_code=400, detail=f"Layer {layer} does not exist: layers count from 0 to {a.L - 1}.")
    n = a.shapes[int(layer)].n_heads
    if not 0 <= int(head) < n:
        raise HTTPException(status_code=400, detail=f"Layer {layer} has {n} attention heads (0 to {n - 1}), not head {head}.")


def _find_unit(units: List[Tuple[str, List[int]]], token: Optional[str]) -> Optional[int]:
    """The first display unit holding `token` (exactly, then ignoring case and spaces); None when none does."""
    if not token:
        return None
    for match in (lambda u: token in u, lambda u: token.strip().lower() in u.lower()):
        for k, (u, _) in enumerate(units):
            if match(u):
                return k
    return None


# ------------------------------------------------------------------------------------------------------------ routes


def links_card(a: Atlas) -> Dict[str, Any]:
    """How the atlas reads a link's sign (atlas_ops.LINK_MODEL from rest, LINK_MODEL_USE in use), whether links can be
    read in use here (`why` not), and what the measured check found (`measured`, from wiring_check.npz, built by
    scripts/atlas_build.py wiring_check; None until then): per layer at the last word-piece and over the prompt, the
    shares whose measured change had each sign (the sign in use scored where each reader sits, and at the passages'
    operating point), Spearman's rho against |P| f^2 and the first-order size, the null's, and each layer's cost; and,
    for a check measured before the readers' own gate and up values were recorded, `site_score`, which says its sign in
    use was scored at the passages' operating point and needs measuring again (atlas_index.SITE_SCORE_RERUN)."""
    why = a.index.use_refused()
    wc = a.index.get("wiring_check")
    measured = None
    if wc is not None:
        m = wc["meta"]
        measured = {k: m.get(k) for k in ("built_at", "writers", "dose", "dose_unit", "seed", "questions", "sign_disagreement")}
        measured["layers"] = {l: {**s, "cost": (m.get("cost") or {}).get(l)} for l, s in (m.get("summary") or {}).items()}
        if any("rho_site" not in ((s.get("last") or {}).get("linked") or {}) for s in (m.get("summary") or {}).values()):
            measured["site_score"] = SITE_SCORE_RERUN
    return {"models": {"rest": ao.LINK_MODEL, "use": ao.LINK_MODEL_USE}, "use": {"ready": why is None, **({"why": why} if why else {})},
            "measured": measured}


@router.get("/arch")
async def arch() -> Dict[str, Any]:
    """The model's shape, which indexes are built, which layers have a census, how links are read and what their
    measured check found (`links`), and the cards every reply follows."""
    a = atlas()
    sc = await asyncio.to_thread(a._scales_if_ready)
    status = await asyncio.to_thread(a.index.status)
    links = await asyncio.to_thread(links_card, a)
    refused: Dict[int, str] = {}
    await asyncio.to_thread(cz.layer_runs, None, a.rt.model_id, refused)
    return {
        "model": getattr(a.rt, "model_id", None),
        "stand_in": bool(getattr(a.rt, "atlas_stand_in", False)),
        "n_layers": a.L,
        "d_model": a.D,
        "d_ff": a.N,
        "vocab_size": int(a.W_U.shape[0]),
        "layers": [{"kind": s.kind, "n_heads": s.n_heads, "n_kv": s.n_kv, "head_dim": s.head_dim,
                    "values_from_keys": s.values_from_keys, "window": s.window} for s in a.shapes],
        "layer_scalars": [round(float(x), 5) for x in a.w.scalars.tolist()],
        "indexes": status,
        "census": {str(l): p.name for l, p in sorted(cz.layer_runs(model=a.rt.model_id).items())},
        "census_refused": {str(l): why for l, why in sorted(refused.items())},
        "push": {"modes": list(ao.PUSH_MODES), "spare_bos": True, "units": {
            "add": "a share of the stream's median length leaving the layer, along the neuron's write direction",
            "mul": "a factor on the neuron's own activation (0 silences it)",
            "act": "a multiple of the neuron's largest |activation|, added to it"},
            "scale_source": sc["source"] if sc else None,
            "sampler": {"fields": sr.FIELDS, "default": "greedy", "draw": f"{sr.SHARED_DRAW}: each step one exponential "
                        "race over the vocabulary, its noise from a CPU generator seeded seed + s for sample s, so the voices "
                        "part only where the push moves the odds", "samples": f"POST /steer: 1 to {MAX_STEER_SAMPLES}, three "
                        f"rows each in one batch, several only while their rows hold at most {MAX_STEER_ROW_PIECES:,} "
                        "word-pieces in all (prompt and answer); the push chat writes one", "not_offered": sorted(sr.NOT_OFFERED)}},
        "limits": {"text": MAX_TEXT_POSITIONS, "run": MAX_RUN_POSITIONS, "heads": MAX_HEAD_POSITIONS, "push_new": MAX_PUSH_NEW},
        "links": links,
        "label_gate": LABEL_GATE_CARD,
        "lens": a.rt.lens_card(),
    }


# -- field notes


@router.get("/notes")
async def get_notes() -> Dict[str, Any]:
    data = await fn.all_notes()
    return {**data, "file": str(fn.field_notes_path()), "warning": fn.take_warning()}


@router.post("/notes")
async def post_note(body: NoteRequest) -> Dict[str, Any]:
    a = atlas()
    kind, name, note, by = off_gpu(fn.check_note, body.kind, body.name, body.note, body.by)
    if body.layer is None or body.index is None:
        raise HTTPException(status_code=400, detail="A note names its layer and its index (the neuron's, or the head's).")
    if kind == "neuron":
        _check_neuron(a, body.layer, body.index)
    else:
        _check_head(a, body.layer, body.index)
    try:
        entry, _data = await fn.write_note(kind, int(body.layer), int(body.index), name, note, by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"kind": kind, "key": fn.note_key(body.layer, body.index), "entry": entry, "file": str(fn.field_notes_path()),
            "warning": fn.take_warning()}


@router.get("/fieldbook")
async def get_field_book(layer: Optional[int] = None, n: Optional[int] = None) -> Dict[str, Any]:
    """The folio's field book (backend/field_notes.py): one neuron's two faces (`faces`, each None when nothing is
    said of it) with its entry in the notes (`notes`: the name the folio gives it, and the atlas's note), given its
    `layer` and `n`; or, given neither, the whole book."""
    if (layer is None) != (n is None):
        raise HTTPException(status_code=400, detail="Give a neuron's layer and n, or neither for the whole field book.")
    book = await fn.all_book()
    files = {"file": str(fn.field_book_path()), "notes_file": str(fn.field_notes_path())}
    if layer is None or n is None:
        return {**book, **files, "warning": fn.take_warning("book")}
    _check_neuron(atlas(), layer, n)
    key = fn.note_key(layer, n)
    faces = book["neurons"].get(key) or {}
    notes = await fn.all_notes()
    return {"layer": layer, "n": n, "faces": {s: faces.get(s) for s in fn.FACES}, "notes": notes["neurons"].get(key), **files,
            "warning": fn.take_warning("book"), "notes_warning": fn.take_warning()}


@router.post("/fieldbook")
async def post_field_book(body: FieldBookRequest) -> Dict[str, Any]:
    """Keep one face of a neuron's field book: `{layer, n, face, jobs?, pattern?, checked?, by}`, a field left out kept
    as it was (`checked` may give some of its four ticks only); a face left with nothing said is forgotten. The name goes
    to POST /notes, as the atlas's."""
    a = atlas()
    if body.layer is None or body.n is None:
        raise HTTPException(status_code=400, detail="A face of the field book names its neuron: its layer and n.")
    _check_neuron(a, body.layer, body.n)
    change = off_gpu(fn.check_face, body.face, body.jobs, body.pattern, body.checked, body.by, body.model_fields_set)
    try:
        entry = await fn.write_face(int(body.layer), int(body.n), change)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"layer": body.layer, "n": body.n, "face": change["face"], "entry": entry, "file": str(fn.field_book_path()),
            "warning": fn.take_warning("book")}


def _proposal_unit(a: Atlas, body: ProposalRequest) -> Tuple[str, int, int]:
    """(kind, layer, index) of a proposal's request, checked against the model."""
    if body.kind not in fn.KINDS:
        raise HTTPException(status_code=400, detail='kind must be "neuron" or "head".')
    if body.layer is None or body.index is None:
        raise HTTPException(status_code=400, detail="A proposal names its layer and its index (the neuron's, or the head's).")
    if body.kind == "neuron":
        _check_neuron(a, body.layer, body.index)
    else:
        _check_head(a, body.layer, body.index)
    return body.kind, int(body.layer), int(body.index)


@router.get("/proposals")
async def get_proposals() -> Dict[str, Any]:
    """The names and notes the scientist has proposed, waiting for the person to keep or discard them; nothing else
    reads them (backend/field_notes.py)."""
    data = await fn.all_proposals()
    return {**data, "file": str(fn.field_proposals_path()), "warning": fn.take_warning("proposals")}


@router.post("/proposals")
async def post_proposal(body: ProposalRequest) -> Dict[str, Any]:
    """A name or note the scientist proposes (its tools' `set_name` and `write_note`): `{kind, layer, index, name?, note?,
    chat}`, checked as a note is; it waits for the person, and the notes are not changed. `kept` is the entry the notes
    keep now, which the atlas and later chats go on reading until the person keeps the proposal."""
    a = atlas()
    kind, name, note, _by = off_gpu(fn.check_note, body.kind, body.name, body.note, "scientist")
    kind, layer, index = _proposal_unit(a, body)
    try:
        proposal = await fn.propose(kind, layer, index, name, note, body.chat or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    kept = (await fn.all_notes())[fn.KINDS[kind]].get(fn.note_key(layer, index))
    return {"kind": kind, "key": fn.note_key(layer, index), "proposal": proposal, "kept": kept, "file": str(fn.field_proposals_path()),
            "warning": fn.take_warning("proposals")}


async def _decide(body: ProposalRequest, keep: bool) -> Dict[str, Any]:
    a = atlas()
    kind, layer, index = _proposal_unit(a, body)
    if body.part not in (None, "name", "note"):
        raise HTTPException(status_code=400, detail='part is "name" or "note", or left out for both.')
    try:
        r = await fn.decide(kind, layer, index, keep, body.part)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if r is None:
        what = f"{'the ' + body.part + ' of ' if body.part else ''}{kind} {index} at layer {layer}"
        raise HTTPException(status_code=404, detail=f"No proposal waits for {what}: it may have been kept or discarded already.")
    return {"kind": kind, "key": fn.note_key(layer, index), "kept" if keep else "discarded": r["parts"], "entry": r["entry"],
            "proposal": r["proposal"], "file": str(fn.field_proposals_path()), "warning": fn.take_warning("proposals")}


@router.post("/proposals/keep")
async def keep_proposal(body: ProposalRequest) -> Dict[str, Any]:
    """The person keeps a proposal (`{kind, layer, index, part?}`): its name or note, or both, written to the notes as
    the scientist's (`name_by`, `note_by` "scientist"), and removed from the proposals. 404 when none waits."""
    return await _decide(body, True)


@router.post("/proposals/discard")
async def discard_proposal(body: ProposalRequest) -> Dict[str, Any]:
    """The person discards a proposal (`{kind, layer, index, part?}`); the notes are not changed. 404 when none waits."""
    return await _decide(body, False)


# -- the census


_census_cache: LRU = LRU(2)
_census_lock = threading.Lock()
_runs_cache: Dict[Any, Tuple[float, Dict[int, Any]]] = {}
_census_words: Dict[int, Tuple[Any, Dict[int, List[str]]]] = {}


def census_runs(model: Optional[str]) -> Dict[int, Any]:
    """`census.layer_runs` for the model read here, looked up again at most every five seconds (every wiring row asks)."""
    import time

    key = (str(cz.census_dir()), model)
    hit = _runs_cache.get(key)
    if hit and time.monotonic() - hit[0] < 5.0:
        return hit[1]
    runs = cz.layer_runs(model=model)
    _runs_cache[key] = (time.monotonic(), runs)
    return runs


def census_words(layer: int, model: Optional[str]) -> Dict[int, List[str]]:
    """{neuron: its first four census words} for a layer whose census is ranked, from the ranking file alone (the
    reference's `ranked_words`): what the atlas shows beside a linked neuron, as a description, so only for neurons
    whose words came out stronger than every random control's (the map's gilt; about N / (C + 1) of them by chance
    alone). The rest keep their census words in the census list and their details."""
    path = census_runs(model).get(int(layer))
    if path is None:
        return {}
    up = cz.units_path_of(path)
    try:
        key = (str(up), up.stat().st_mtime)
    except FileNotFoundError:
        return {}
    hit = _census_words.get(int(layer))
    if hit and hit[0] == key:
        return hit[1]
    words = {int(u["neuron"]): [w for w, _, _ in u["words"][:4]] for u in cz.read_rows(up)
             if u["kind"] == "neuron" and u.get("null_pct") == 1.0}
    _census_words[int(layer)] = (key, words)
    return words


def census_layer(layer: int, model: Optional[str]) -> Optional[Dict[str, Any]]:
    """A layer's census run (of the model read here), its rows slimmed to what the page shows and its ranking (made
    beside it once, when missing or older than the run). The last two layers opened are kept (a whole layer's answers
    take a few hundred megabytes even slimmed)."""
    path = cz.layer_runs(model=model).get(int(layer))
    if path is None:
        return None
    units_path = cz.units_path_of(path)
    key = (str(path), path.stat().st_mtime, units_path.stat().st_mtime if units_path.exists() else 0.0)
    hit = _census_cache.get(int(layer))
    if hit and hit["key"] == key:
        _census_cache.move_to_end(int(layer))
        return hit
    with _census_lock:
        if not (units_path.exists() and units_path.stat().st_mtime >= path.stat().st_mtime):
            cz.write_units(path)
            key = (str(path), path.stat().st_mtime, units_path.stat().st_mtime)
        units = list(cz.read_rows(units_path))
        slim = ("kind", "neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen")
        rows = [{k: r.get(k) for k in slim} for r in cz.read_rows(path, tolerate_tail=True)]
        by_unit: Dict[Tuple[str, int], List[int]] = {}
        for i, r in enumerate(rows):
            by_unit.setdefault((r["kind"], int(r["neuron"])), []).append(i)
        meta = json.loads(cz.meta_path_of(path).read_text(encoding="utf-8"))
        n_neu = sum(1 for u in units if u["kind"] == "neuron")
        n_ctl = sum(1 for u in units if u["kind"] == "random")
        null = {"neurons": n_neu, "controls": n_ctl, "above_all": sum(1 for u in units if u["kind"] == "neuron" and u.get("null_pct") == 1.0),
                "above_all_by_chance": cz.chance_above_all(n_neu, n_ctl),
                "q05": sum(1 for u in units if u.get("q") is not None and u["q"] <= 0.05)}
        d = {"key": key, "path": path, "rows": rows, "units": units, "by_unit": by_unit, "meta": meta, "null": null}
        _census_cache.put(int(layer), d)
        return d


def census_card(d: Dict[str, Any]) -> Dict[str, Any]:
    """What every census reply says of its run: where its doses and its push unit come from, the unit a dose is counted
    in (`dose_unit`: "stream", a share of the stream's median length, or "ceiling", a multiple of each neuron's write
    ceiling; a run that records none was taken in the stream's), how loudly the next layer reads its controls against
    its neurons, what its null can and cannot single out, the card it was written on (`device`; None before runs kept
    it, or where its parts name more than one or one unknown), whether its parts were written on more than one runtime
    class (`mixed_runtime`: card, library versions or code, or a part begun before runs recorded theirs; each part's
    readings valid, their bits not alike), for shards joined with some missing, what is missing (`partial`),
    for the second stage of a two-stage census, which screen's strongest it retests (`stage`: its p and q are taken over
    those neurons alone, against fresh controls), how its ranking read its answers (`reading`, as its ranking file
    records it, census.write_units: the largest push counted, the mean a word is set against, whose unpushed words are
    left out), its doses and whether they are calibrated on this model at the run's layer (`doses_calibrated`,
    census.doses_calibrated; `doses_calibrated_layers`, the first and last layer its design is calibrated at, None for
    a design calibrated at none), and, where scripts/verify_atlas_writer.py has measured one for the
    run's model, the writer's floor (`writer_floor`, census.writer_floor: with no push, after how many word-pieces an
    answer written in batches like this run's parts from the same answer written alone; None where none is on disk)."""
    m = d["meta"]
    design = m.get("design_name")
    devices = {(c or {}).get("device") for c in (m.get("mixed_runtime") or [m.get("runtime")])}
    span = cz.DOSES_CALIBRATED.get(str(design))
    # where its doses come from in the words its meta recorded, save for a design whose calibration was recorded since
    # (census.DOSES_CALIBRATED), which speaks with the words that say where it is calibrated
    doses_from = cz.DOSES_FROM.get(design) if span else (m.get("doses_from") or cz.DOSES_FROM.get(design))
    return {"run": d["path"].name, "design": design, "doses_from": doses_from,
            "dose_unit": m.get("dose_unit", "stream"), "p50_source": m.get("p50_source"), "read_gain": m.get("read_gain"), "null": d["null"],
            "device": devices.pop() if len(devices) == 1 else None, "mixed_runtime": bool(m.get("mixed_runtime")), "partial": m.get("partial"),
            "stage": m.get("stage"), "writer_floor": cz.writer_floor(m),
            # how its ranking read its answers (the reading its ranking file records; one written before rankings
            # recorded it was ranked by census.READING_DEFAULT; with no ranking yet, its design's, census.reading_of),
            # its doses, and whether they are calibrated on this model at the run's layer (census.doses_calibrated)
            "reading": next((u["reading"] for u in d.get("units") or [] if u.get("reading")), None)
                       or (dict(cz.READING_DEFAULT) if d.get("units") else cz.reading_of(m.get("design_name"))),
            "doses": sorted({float(x[1]) for x in m.get("design") or []}),
            "doses_calibrated": cz.doses_calibrated(design, m.get("layer")),
            "doses_calibrated_layers": list(span) if span else None}


@router.get("/census/units")
async def census_units(layer: int) -> List[Dict[str, Any]]:
    d = await asyncio.to_thread(census_layer, layer, atlas().rt.model_id)
    if d is None:
        return []
    notes = (await fn.all_notes())["neurons"]
    return [{"kind": u["kind"], "n": u["neuron"], "z": u.get("mass", u["z"]), "null_pct": u["null_pct"], "q": u.get("q"),
             "words": [w for w, _, _ in u["words"][:5]], "name": (notes.get(fn.note_key(layer, u["neuron"])) or {}).get("name", "") if u["kind"] == "neuron" else ""}
            for u in d["units"]]


@router.get("/census/card")
async def census_card_route(layer: int) -> Dict[str, Any]:
    """The layer's census card (`census_card`); {} when the layer has none."""
    d = await asyncio.to_thread(census_layer, layer, atlas().rt.model_id)
    return census_card(d) if d is not None else {}


@router.get("/census/unit")
async def census_unit(layer: int, n: int, kind: str = "neuron") -> Dict[str, Any]:
    d = await asyncio.to_thread(census_layer, layer, atlas().rt.model_id)
    notes = (await fn.all_notes())["neurons"]
    out: Dict[str, Any] = {"layer": layer, "n": n, "kind": kind, "name": (notes.get(fn.note_key(layer, n)) or {}).get("name", "")}
    if d is None:
        return out
    rec = next((u for u in d["units"] if u["neuron"] == n and u["kind"] == kind), None)
    rows = [d["rows"][i] for i in d["by_unit"].get((kind, n), [])]
    keep = ("prompt", "dose", "rel_push", "text", "finish", "n_gen")
    neurons = [u for u in d["units"] if u["kind"] == "neuron"]
    out.update({
        "run": d["path"].name, "prompts": d["meta"].get("prompts", cz.PROMPTS), "design": d["meta"].get("design_name"),
        "card": census_card(d),
        "rollouts": [{k: r[k] for k in keep} for r in rows],
        "baselines": [{k: r[k] for k in keep} for r in d["rows"] if r["kind"] == "baseline"],
        "stats": rec, "rank": next((i + 1 for i, u in enumerate(neurons) if u["neuron"] == n), None) if rec else None,
        "n_ranked": len(neurons),
    })
    return out


@router.get("/census/rollout")
async def census_rollout(layer: int, i: int) -> Dict[str, Any]:
    d = await asyncio.to_thread(census_layer, layer, atlas().rt.model_id)
    if d is None or not 0 <= i < len(d["rows"]):
        raise HTTPException(status_code=404, detail="There is no such answer in this layer's census.")
    r = d["rows"][i]
    return {k: r[k] for k in ("kind", "neuron", "prompt", "dose", "rel_push", "text")}


@router.get("/census/spread")
async def census_spread(layer: int) -> Any:
    path = cz.layer_runs(model=atlas().rt.model_id).get(int(layer))
    if path is None:
        return {}
    sp = cz.spread_path_of(path)
    return Response(sp.read_bytes(), media_type="application/json") if sp.exists() else JSONResponse({})


# -- a neuron's words, wiring and heads


def _lens_reply(a: Atlas, layer: int, n: int) -> Dict[str, Any]:
    r = a.vault.inspect_bipolar_slot(layer_idx=layer, neuron_idx=n, polarity=1.0, top_k_tokens=LENS_K)
    sc = a._scales_if_ready()
    out = {
        "layer": layer, "n": n,
        "faces": {
            "+": {"lens": [t["token"] for t in r["north_pole"]["j_vindex_tokens"]], "eye": [t["token"] for t in r["north_pole"]["naive_tokens"]],
                  "check": r["north_pole"]["label_check"]},
            "-": {"lens": [t["token"] for t in r["south_pole"]["j_vindex_tokens"]], "eye": [t["token"] for t in r["south_pole"]["naive_tokens"]],
                  "check": r["south_pole"]["label_check"]},
        },
        "gate_trigger": [t["token"] for t in r["gate_trigger_tokens"]],
        "steer": None,
        "act_range": a.act_range(layer, n, sc) if sc else None,
        "edited": (layer, n) in a.edited(),
        "ceiling": ceiling(a, layer, n, sc),
    }
    st = a.index.get("steer_lens")
    if st is not None and not out["edited"]:
        out["steer"] = {"source": "index", "up": a.words(st["up"][layer, n], 12), "down": a.words(st["down"][layer, n], 12),
                        "kl_up": round(float(st["up_kl"][layer, n]), 3), "kl_down": round(float(st["down_kl"][layer, n]), 3),
                        "dose": float(st["dose"])}
    else:
        out["steer"] = a.cache["steer"].get((a.edit_key(), layer, n))
        if st is not None and out["steer"] is None:
            # the index's was read before the vault edit in force on this neuron: none until it is read again
            out["steer_stale"] = "the steer-lens index was read before the vault edit in force on this neuron: read it again now"
    return out


STALE_WEB = "the wiring index was read before the vault edit in force on this neuron: its links and web are as they were"


def ceiling(a: Atlas, layer: int, n: int, sc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The most the neuron can ever write to the stream leaving its layer (`Weights.write_ceiling`), and, with the push's
    unit known, as a share of the stream's median length (`rel`): an add push of dose d is d / rel times it."""
    c = float(a.w.write_ceiling(layer, [n])[0])
    p50 = float(sc["p50"][layer]) if sc else None
    return {"write": round(c, 5), "p50": round(p50, 5) if p50 else None, "rel": round(c / p50, 5) if p50 else None,
            "source": sc["source"]["corpus"] if sc else None}


@router.get("/neuron/lens")
async def neuron_lens(layer: int, n: int) -> Dict[str, Any]:
    """Both faces' words through the lens and by the naked eye, read live from the weights (vault edits included), each
    face with its label check; the steer lens from its index (or as read live by POST /neuron/steer_lens); the
    activation range."""
    a = atlas()
    _check_neuron(a, layer, n)
    reply = await on_gpu(_lens_reply, a, layer, n)
    return {**reply, "label_gate": LABEL_GATE_CARD, "lens": a.rt.lens_card()}


def _steer_lens_live(a: Atlas, layer: int, n: int) -> Dict[str, Any]:
    sc = a.scales()
    ctx = [a.rt.tokenize_prompt(a.rt.compose_prompt(c, chat=True)) for c in ao.STEER_LENS_CHATS]
    ctx += [a.rt.tokenize_prompt(p) for p in ao.STEER_LENS_PLAIN]
    r = ao.steer_lens(a.lm, a.W_U, ctx, a.w, layer, n, sc["p50"][layer], k=12, bos_id=a.rt.tokenizer.bos_token_id)
    out = {"source": "live", "up": a.words(r["up"], 12), "down": a.words(r["down"], 12), "kl_up": r["kl_up"],
           "kl_down": r["kl_down"], "dose": r["dose"], "scale_source": sc["source"]}
    a.cache["steer"].put((a.edit_key(), layer, n), out)
    return out


@router.post("/neuron/steer_lens")
async def neuron_steer_lens(layer: int, n: int) -> Dict[str, Any]:
    """The steer lens of one neuron, read live: ±0.8 of the stream's median length along its write direction at every
    piece of four contexts but the start mark, the next piece's mean change of log-probability (fp32, softcapped) and
    KL (atlas_ops.steer_lens)."""
    a = atlas()
    _check_neuron(a, layer, n)
    return await on_gpu(_steer_lens_live, a, layer, n)


def _decorate(a: Atlas, rows: List[Dict[str, Any]], notes: Dict[str, Any], edited: set, side: str) -> List[Dict[str, Any]]:
    """Each linked neuron read on the face the link model gives it (atlas_ops.LINK_MODEL): a reader ("down") on the
    face the link pushes it toward, a writer ("up") on the face that opens the open neuron's gate; that face's words
    (with their check), its census words and its name."""
    out = []
    for c in rows:
        face = ("+" if c["sign"] > 0 else "-") if side == "down" else ("+" if c["gate"] > 0 else "-")
        out.append({**c, "face": face, "words": a.face(c["layer"], c["n"], face, 5, edited),
                    "census": census_words(c["layer"], a.rt.model_id).get(c["n"], []),
                    "name": (notes.get(fn.note_key(c["layer"], c["n"])) or {}).get("name", ""),
                    "name_by": (notes.get(fn.note_key(c["layer"], c["n"])) or {}).get("name_by")})
    return out


def _wiring(a: Atlas, layer: int, n: int, k: int) -> Dict[str, Any]:
    """The links each way with their g and u (where the index kept them; read live always)."""
    got = a.index.links(layer, n, k, gu=True)
    if got is not None and (layer, n) not in a.edited():
        return {"source": "index", **got}
    key = (a.edit_key(), layer, n, k)
    hit = a.cache["wiring"].get(key)
    if hit is None:
        hit = a.cache["wiring"].put(key, ao.neuron_wiring(a.w, layer, n, a.link_scale(), k=k, gu=True))
    return {"source": "live", **hit}


def _model(model: str) -> str:
    if model not in LINK_MODELS:
        raise HTTPException(status_code=400, detail=f'model is "rest" (the default: signs read from rest) or "use" (read at the '
                                                    f"reader's operating point), not {model!r}.")
    return model


USE_STALE = ("the readers' operating points come from the passages index, read before the vault edits in force at earlier "
             "layers, which can move where a reader sits")


def _use(a: Atlas, res: Dict[str, Any], layer: int, n: int) -> Dict[str, Any]:
    """Each link's `use_sign` (AtlasIndex.use_signs: per face of its reader, where the writer fired on its `gate` face
    pushes it at its operating point there), put on the links of `res` in place, and what the reply says of it."""
    why = a.index.use_refused(live=res["source"] == "live")
    for side in ("down", "up"):
        signs = a.index.use_signs(res[side], side, layer, n) if why is None else [None] * len(res[side])
        for c, s in zip(res[side], signs):
            c["use_sign"] = s
    out: Dict[str, Any] = {"ready": why is None, "note": ao.LINK_MODEL_USE}
    if why:
        out["why"] = why
    else:
        m = a.index.get("passages")["meta"]
        out["operating_points"] = {"corpus": m.get("corpus"), "tokens": m.get("tokens"), "K": m.get("K"), "built_at": m.get("built_at")}
        readers = [c["layer"] for c in res["down"]] + ([layer] if res["up"] else [])
        if any(l_e < l_r for l_e, _ in a.edited() for l_r in readers):
            out["stale"] = USE_STALE
    return out


@router.get("/neuron/wiring")
async def neuron_wiring(layer: int, n: int, k: int = 24, model: str = "rest") -> Dict[str, Any]:
    """The neuron's strongest weight links each way (atlas_ops.neuron_wiring), by strength across layers: readers in
    later layers ("down") and writers in earlier ones ("up"), each with w, z, sign, gate, the face it is read on
    (`face`, `_decorate`), that face's words with their check, and its name. From the wiring index when built and the
    neuron is not edited, else read live from the weights (`source`). A neuron that writes nothing (silenced in the
    vault) has no readers. With `model` "use" each link also carries `use_sign` (atlas_ops.LINK_MODEL_USE), per face of
    its reader, {"+", "-"}: the face the writer fired on its `gate` face pushes the reader toward at its operating
    point there (comparable with `sign`; null for a face with no stored event), the link's g and u from the index or
    read live, the operating points always from the passages index; `use` says whether they could be read (`why` when
    not, every `use_sign` then null), from which passages, and `stale` after a vault edit at an earlier layer."""
    a = atlas()
    _check_neuron(a, layer, n)
    _model(model)
    k = max(1, min(int(k), 64))
    if await asyncio.to_thread(a.index.links, layer, n, k) is not None and (layer, n) not in a.edited():
        res = await asyncio.to_thread(_wiring, a, layer, n, k)
    else:
        res = await on_gpu(_wiring, a, layer, n, k)
    res = {"source": res["source"], **{side: [dict(c) for c in res[side]] for side in ("down", "up")}}  # the cache's own stay as they are
    use = await asyncio.to_thread(_use, a, res, layer, n) if model == "use" else None
    for side in ("down", "up"):
        for c in res[side]:
            c.pop("g", None)
            c.pop("u", None)
    notes = (await fn.all_notes())["neurons"]
    edited = a.edited()
    down, up = await asyncio.to_thread(lambda: (_decorate(a, res["down"], notes, edited, "down"), _decorate(a, res["up"], notes, edited, "up")))
    return {"layer": layer, "n": n, "source": res["source"], "down": down, "up": up,
            "note": "weight-only direct-path links, read " + ao.LINK_MODEL + " Leads to check, not circuits.",
            **({"model": "use", "use": use} if use is not None else {})}


def _roles(a: Atlas, layer: int, n: int) -> Dict[str, Any]:
    key = (a.edit_key(), layer, n)
    hit = a.cache["roles"].get(key)
    if hit is None:
        hit = a.cache["roles"].put(key, ao.neuron_head_roles(a.spaces, layer, n, k=8))
    return hit


@router.get("/neuron/heads")
async def neuron_heads(layer: int, n: int) -> Dict[str, Any]:
    """The attention heads the neuron meets, by role, from the weights (atlas_ops.neuron_head_roles): q, k, v are later
    heads reading its write; o are heads at its layer or before whose write it reads; z per head over every head of
    the role; `z_by_layer` lights the map."""
    a = atlas()
    _check_neuron(a, layer, n)
    r = await on_gpu(_roles, a, layer, n)
    return {"layer": layer, "n": n, "roles": {k: r[k] for k in ("q", "k", "v", "o")}, "z_by_layer": r["z_by_layer"]}


# -- a neuron's stored passages

REGIMES = {"pp": "gate > 0, up > 0", "nn": "gate < 0, up < 0 (the double negative)", "pn": "gate > 0, up < 0",
           "np": "gate < 0, up > 0"}
PASSAGES_UNBUILT = ("the passages index is not built: scripts/atlas_build.py passages builds it (one forward a window of a "
                    "corpus, as the stats index; give it the same corpus)")


def regime(g: float, u: float) -> str:
    """GeGLU's sign regime at an event, as `atlas_ops.Extremes` counts them (a gate or up of 0 counts as negative)."""
    return ("p" if g > 0 else "n") + ("p" if u > 0 else "n")


def passage_marks(tok: Any) -> Dict[int, Tuple[str, str]]:
    """The word-pieces a passage is never shown across: {id: (what it marks, "begins" | "ends")}: the start mark (a
    window's first piece: the model read nothing before it; one further in begins another document, as a chat composed
    with its own start mark does, and `cut_passage` calls it "document"), the end mark (between two documents) and the
    two turn marks of a chat (`<|turn>` opens a turn, `<turn|>` closes it), where the tokenizer has them."""
    out: Dict[int, Tuple[str, str]] = {}
    for t, what, side in (("<|turn>", "turn", "begins"), ("<turn|>", "turn", "ends")):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            out[i] = (what, side)
    if tok.eos_token_id is not None:
        out[int(tok.eos_token_id)] = ("document", "ends")
    if tok.bos_token_id is not None:
        out[int(tok.bos_token_id)] = ("start", "begins")
    return out


def cut_passage(ids: Sequence[int], at: int, before: int, after: int, marks: Dict[int, Tuple[str, str]]) -> Tuple[int, int, Optional[str], Optional[str]]:
    """The span [s, e) of a window shown around its word-piece `at`: at most `before` pieces before it and `after` after,
    never across a mark (`passage_marks`, itself left out); a peak on a mark that begins something shows nothing before
    it, on one that ends something nothing after. Returns (s, e, why it stops before, why after): the mark's kind
    ("start" only for the window's own first piece; a start mark further in is a "document" boundary), "end" for the
    window's own end, None where the count ran out first."""
    kind = lambda i, m: "document" if m[0] == "start" and i > 0 else m[0]  # noqa: E731
    s, e, why_b, why_a = at, at + 1, None, None
    own = marks.get(int(ids[at]))
    if own and own[1] == "begins":
        why_b = kind(at, own)
    else:
        while at - s < before and s > 0:
            m = marks.get(int(ids[s - 1]))
            if m:
                why_b = kind(s - 1, m)
                break
            s -= 1
    if own and own[1] == "ends":
        why_a = own[0]
    else:
        while e - at - 1 < after:
            if e >= len(ids):
                why_a = "end"
                break
            m = marks.get(int(ids[e]))
            if m:
                why_a = kind(e, m)
                break
            e += 1
    return s, e, why_b, why_a


def whole_characters(tok: Any, ids: Sequence[int], s: int, e: int, marks: Dict[int, Tuple[str, str]], reach: int = 3) -> Tuple[int, int]:
    """A span [s, e) widened by up to `reach` word-pieces each way where it would begin or end inside a character split
    over several byte pieces (an emoji, a CJK character), never across a mark."""
    text = lambda: tok.decode([int(t) for t in ids[s:e]])  # noqa: E731
    for _ in range(reach):
        if e < len(ids) and int(ids[e]) not in marks and text().endswith("\ufffd"):
            e += 1
    for _ in range(reach):
        if s > 1 and int(ids[s - 1]) not in marks and text().startswith("\ufffd"):
            s -= 1
    return s, e


def _passages(a: Atlas, layer: int, n: int, face: str, k: int, before: int, after: int) -> Dict[str, Any]:
    head: Dict[str, Any] = {"layer": layer, "n": n, "face": face}
    got = a.index.passages(layer, n, face, k)
    if got is None:
        why = a.index.refused("passages")
        return {**head, "ready": False, "note": PASSAGES_UNBUILT if not why else f"the passages index is not read: {why}"}
    P = a.index.get("passages")
    meta = P["meta"]
    tokens, unigram = int(meta.get("tokens") or 0), P["unigram"]
    vocab = a.vocab()
    tok = a.rt.tokenizer
    marks = passage_marks(tok)
    stats = a.index.get("act_scale")
    ext = float(stats["pos" if face == "+" else "neg"][layer, n]) if stats is not None else 0.0
    rel_to = None
    if ext:
        # the build takes whatever corpus it is given: say so when the stats index was read over another
        sm, st = stats["meta"].get("corpus"), int(stats["tokens"])
        rel_to = (f"its extreme in the stats index ({sm})" if (sm, st) == (meta.get("corpus"), tokens)
                  else f"its extreme in the stats index, read over another corpus ({sm}, {st:,} word-pieces)")
    elif got["top"]:
        ext, rel_to = got["top"][0]["act"], ("its strongest passage here (no stats index)" if stats is None
                                             else "its strongest passage here (the stats index never saw this face fire)")
    word = lambda t: vocab[int(t)] if 0 <= int(t) < len(vocab) else ""  # noqa: E731
    out, reads, follows = [], {}, {}
    for kind, slots in (("top", got["top"]), ("sample", sorted(got["sample"], key=lambda x: -abs(x["act"])))):
        for x in slots:
            ids = a.index.window_ids(x["window"]) or []
            at = x["at"]
            s, e, why_b, why_a = cut_passage(ids, at, before, after, marks)
            s, e = whole_characters(tok, ids, s, e, marks)
            units = [[text, (at - s) in idx] for text, idx in ao.token_units(tok, ids[s:e])]
            nxt = ids[at + 1] if at + 1 < len(ids) else None
            if kind == "top":
                reads[ids[at]] = reads.get(ids[at], 0) + 1
                if nxt is not None:
                    follows[nxt] = follows.get(nxt, 0) + 1
            out.append({"kind": kind, "act": round(x["act"], 4), "rel": round(x["act"] / ext, 3) if ext else None,
                        "regime": regime(x["g"], x["u"]), "g": round(x["g"], 4), "u": round(x["u"], 4), "window": x["window"], "at": at,
                        "units": units, "span": [s, e], "next": word(nxt) if nxt is not None else None, "cut": [why_b, why_a],
                        "ids": ids[:e]})

    def lifted(counts: Dict[int, int]) -> List[Dict[str, Any]]:
        used = sum(counts.values())
        rows = [{"tok": word(t), "id": int(t), "n": c,
                 "lift": round((c / used) / (int(unigram[int(t)]) / tokens), 1) if used and tokens and int(unigram[int(t)]) else None}
                for t, c in counts.items()]
        return sorted(rows, key=lambda r: (-r["n"], -(r["lift"] or 0)))[:8]

    k_used = len(got["top"])
    res = {**head, "ready": True, "source": "index", "corpus": meta.get("corpus"), "tokens": tokens, "windows": meta.get("windows"),
           "built_at": meta.get("built_at"), "k": k_used, "K": meta.get("K"), "per_window": meta.get("per_window"),
           "sampled": meta.get("sample") or 0, "sample_threshold": meta.get("sample_threshold"),
           "extreme": round(ext, 4) if ext else None, "rel_to": rel_to, "passages": out,
           "reads": lifted(reads), "follows": lifted(follows),
           "words_note": f"from {k_used} passages of a {tokens:,}-word-piece corpus; not validated as a label",
           "regimes": REGIMES,
           "read_again": "POST /reads with {layer, n, ids}: a passage's ids are its window from the start mark to the "
                         "passage's end, earlier documents included, as the model read it"}
    edited = a.edited()
    if (layer, n) in edited:
        res["stale"] = ("read before the vault edit in force on this neuron: its own edit changes what it writes, not where it "
                        "fires; read a passage again (POST /reads with its ids) to see it as it now is")
    elif any(l < layer for l, _ in edited):
        res["stale"] = ("read before the vault edits in force at earlier layers, which can move where it fires: read a passage "
                        "again (POST /reads with its ids) to see it as it now is")
    return res


@router.get("/neuron/passages")
async def neuron_passages(layer: int, n: int, face: str = "+", k: int = 16, before: int = 24, after: int = 6) -> Dict[str, Any]:
    """One face's stored passages (the passages index, built by scripts/atlas_build.py passages; no GPU): its k strongest
    events, at most one a window, then those drawn at random above half its extreme, each with its activation, its
    share of the face's extreme (`rel`), GeGLU's regime there (from the gate and up values, `g`, `u`), its window and
    position, and its display units ([text, is_peak], the window's word-pieces `span` [s, e)) cut at the window's start
    or end, a boundary between documents (an end mark, or a start mark inside the window) or a turn mark, else where
    `before` or `after` (each at most 128) runs out (`cut`: why each side stops; a character split over byte pieces is
    kept whole); `ids`, the window up to the span's end, for POST /reads. `reads` and `follows`: the word-pieces at the
    peaks and just after them, counted over the k with their lift over the corpus's base rate, which are leads
    (`words_note`), not labels the gate has checked. `rel_to`: what `rel` is a share of (the stats index's extreme,
    naming its corpus, or saying it is another corpus's). `ready: false`, saying which build makes it, without the
    index; `stale` when a vault edit in force could have changed them."""
    a = atlas()
    _check_neuron(a, layer, n)
    face = {"+": "+", " ": "+", "pos": "+", "-": "-", "neg": "-"}.get(face, face)  # a bare "+" in a URL arrives as a space
    if face not in ("+", "-"):
        raise HTTPException(status_code=400, detail=f'A face is "+" or "-" (in a URL "+" is written %2B), not {face!r}.')
    if int(k) < 1 or int(before) < 0 or int(after) < 0:
        raise HTTPException(status_code=400, detail="k counts passages (at least 1); before and after count word-pieces (0 or more).")
    # at most 128 word-pieces each side: a span is split into display units a prefix at a time (`token_units`)
    return await asyncio.to_thread(_passages, a, layer, n, face, min(int(k), 64), min(int(before), 128), min(int(after), 128))


@router.get("/hover")
async def hover(layer: int, n: int) -> Dict[str, Any]:
    """What the map's tooltip shows, from the indexes alone (no GPU): both faces' words with their checks, the steer
    lens, its sixteen strongest links each way and its name."""
    a = atlas()
    _check_neuron(a, layer, n)
    edited = a.edited()
    notes = (await fn.all_notes())["neurons"]

    def work() -> Dict[str, Any]:
        return {"layer": layer, "n": n, "faces": {s: a.face(layer, n, s, 6, edited) for s in ("+", "-")},
                "links": a.index.links(layer, n, 16), "name": (notes.get(fn.note_key(layer, n)) or {}).get("name", ""),
                **({"links_stale": STALE_WEB} if (layer, n) in edited else {}),
                "ready": a.index.get("lens") is not None}

    return await asyncio.to_thread(work)


@router.get("/search")
async def search(q: str, limit: int = 300) -> Dict[str, Any]:
    """Neurons one of whose faces carries a word-piece holding `q` (ignoring case) among its first twelve, through the
    lens, by the naked eye, or in its steer lens (backend/atlas_index.AtlasIndex.search). Each row names the reading it
    matched in (`src`), the face's words, and the label check of the face. A match through the lens or the naked eye
    is a label like any other, so a face the gate holds back goes to `held` (for the apparatus), not `results` (the
    main reading and the map's marks); `held_back` counts them."""
    a = atlas()
    notes = (await fn.all_notes())["neurons"]
    edited = a.edited()

    def work() -> Dict[str, Any]:
        if a.index.get("lens") is None and a.index.get("steer_lens") is None:
            return {"ready": False, "results": [], "held": [], "held_back": 0, "tokens": 0, "total": 0}
        match = match_ids(a.vocab(), q)
        if not len(match):
            return {"ready": True, "results": [], "held": [], "held_back": 0, "tokens": 0, "total": 0}
        lim = max(1, min(int(limit), 2000))
        rows, total = a.index.search(match, 10**9)
        # the limit applies to each list after the gate has split them, so held-back faces do not crowd out the rest
        res, held, n_held = [], [], 0
        for r in rows:
            if r["src"] != "steer":
                chk = a.index.face_check(r["layer"], r["n"], r["sign"], edited=(r["layer"], r["n"]) in edited)
                if chk is None or chk.get("held_back"):
                    n_held += 1
                    if len(held) >= lim:
                        continue
                    held.append(r)
                    continue
            if len(res) < lim:
                res.append(r)
        dec = lambda r: {**r, "words": a.face(r["layer"], r["n"], r["sign"], 6, edited),  # noqa: E731
                         "census": census_words(r["layer"], a.rt.model_id).get(r["n"]),
                         "name": (notes.get(fn.note_key(r["layer"], r["n"])) or {}).get("name", "")}
        return {"ready": True, "results": [dec(r) for r in res], "held": [dec(r) for r in held], "held_back": n_held,
                "tokens": int(len(match)), "total": total}

    return await asyncio.to_thread(work)


@router.get("/web")
async def web(layer: int, n: int, first: int = 5, k: int = 2, zmin: float = 15.0, cap: int = 20000, hops: int = 0) -> Response:
    """The neuron's cone for the map, from the wiring index: int16 rows of six (src layer, src neuron, dst layer, dst
    neuron, round(10 z) * sign, hop; hop > 0 downstream, < 0 upstream); X-Web-Total counts them. Empty when the index
    is not built."""
    a = atlas()
    _check_neuron(a, layer, n)
    rows = await asyncio.to_thread(a.index.cone, layer, n, max(1, int(k)), float(zmin), max(1, min(int(cap), 40000)),
                                   max(1, int(first)), int(hops) or None)
    arr = np.array([(*r[:4], int(round(r[4] * 10)), r[5]) for r in rows], np.int16).reshape(-1, 6)
    headers = {"X-Web-Total": str(len(arr))}
    if (layer, n) in a.edited():
        headers["X-Web-Stale"] = "1"  # the wiring index was read before the vault edit in force on this neuron
    return Response(arr.tobytes(), media_type="application/octet-stream", headers=headers)


@router.get("/web/summary")
async def web_summary(layer: int, n: int, first: int = 5, k: int = 2, top: int = 24, hops: int = 0, src: str = "auto",
                      model: str = "rest") -> Dict[str, Any]:
    """The cone's words pooled (AtlasIndex.web_summary), followed from rest, or with `model` "use" at each reader's
    operating point (not ready without the passages index, saying why)."""
    a = atlas()
    _check_neuron(a, layer, n)
    _model(model)
    if src not in ("auto", "steer", "lens"):
        raise HTTPException(status_code=400, detail='src is "auto", "steer" or "lens".')
    vocab = await asyncio.to_thread(a.vocab)
    res = await asyncio.to_thread(a.index.web_summary, layer, n, vocab, max(1, int(first)), max(1, int(k)),
                                  max(1, min(int(top), 64)), int(hops) or None, src, model)
    if (layer, n) in a.edited() and res.get("ready"):
        res["stale"] = STALE_WEB
    if model == "use" and res.get("ready") and any(l_e < a.L - 1 for l_e, _ in a.edited()):
        res["use_stale"] = USE_STALE
    return res


@router.get("/layout")
async def layout() -> Dict[str, Any]:
    """Each layer's sheet from the layout index: `cell` [L, N] int16, each neuron's cell on a side x side grid (row-major),
    neighbours writing alike (base64, little-endian); `available: false` when not built (the page lays neurons out by
    index)."""
    lay = await asyncio.to_thread(atlas().index.layout)
    if lay is None:
        return {"available": False}
    cell = lay["cell"]
    return {"available": True, "shape": list(cell.shape), "side": lay["side"], "method": lay["meta"].get("method"),
            "cell": base64.b64encode(cell.astype("<i2").tobytes()).decode()}


# -- live: the push chat, reads, odds


@router.post("/push")
async def push_chat(req: PushChatRequest, request: Request) -> StreamingResponse:
    """The three voices (the reference's /api/steer): the model writes its answer to `prompt` three times in one batch,
    pushed up, as it is, and pushed down, each row replaying its own earlier replies (`history`; the user's turns are
    shared): greedily, or with the instrument's sampler (`sampler`), one draw a step shared by the three voices. One
    neuron (layer, n, dose, mode) or several (`neurons`, each pushed as asked in the first row and the other way in the
    last). Server-sent events: {"row", "t"} for each piece of text as it comes, then {"row": null, "done": true,
    "sampling"} (or "error"), `sampling` "greedy" or the sampler with its seed. The GPU lock is held while it writes;
    closing the page stops it."""
    a = atlas()
    ps, hs = off_gpu(a.parse_pushes, _push_items(req))
    smp, sampling = off_gpu(_sampler, req)
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def work() -> None:
        try:
            iv, _echo = a.interventions(ps, hs, ao.THREE_VOICES)
            rows = []
            for r in range(3):
                msgs: List[Dict[str, str]] = []
                for turn in req.history:
                    msgs += [{"role": "user", "content": turn.user}, {"role": "assistant", "content": turn.replies[r]}]
                rows.append(a.chat_ids(msgs + [{"role": "user", "content": req.prompt}], req.system))
            if max(map(len, rows)) + req.max_new > MAX_TEXT_POSITIONS:
                raise ValueError(f"The conversation is too long to continue here ({max(map(len, rows))} word-pieces); clear it.")
            td = ao.TextDeltas(a.rt.tokenizer, 3)
            for st in ao.write_rows(a.lm, a.W_U, rows, iv, max_new=req.max_new, stop_ids=a.rt.end_of_turn_ids(),
                                    bos_id=a.rt.tokenizer.bos_token_id, spare_bos=req.spare_bos,
                                    softcap=ao.softcap_of(a.rt.hf_model), cancel=cancel.is_set, sampler=smp):
                t = td.add(st.row, st.token_id, st.end)
                if t:
                    loop.call_soon_threadsafe(q.put_nowait, {"row": st.row, "t": t})
            loop.call_soon_threadsafe(q.put_nowait, {"row": None, "done": True, "sampling": sampling})
        except Exception as exc:  # surfaced beside the chat, in plain words where it can be
            msg = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
            loop.call_soon_threadsafe(q.put_nowait, {"row": None, "error": msg})

    async def events():
        assert _gpu_lock is not None
        async with _gpu_lock:
            th = threading.Thread(target=work, daemon=True)
            th.start()
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            cancel.set()
                            break
                        yield ": waiting\n\n"
                        continue
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev["row"] is None:
                        break
                    if await request.is_disconnected():
                        cancel.set()
                        break
            finally:
                cancel.set()
                # the writer stops at its next word-piece; the lock is held until it has (a page closed mid-answer
                # cancels this task, so the wait is shielded)
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(th.join)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def _push_items(req: Union[PushChatRequest, "SteerRequest"]) -> List[PushIn]:
    """The neurons a push chat (or a whole-text steer) pushes: its `neurons`, else the one neuron (layer, n), each with
    the request's mode and dose where it names none."""
    if req.neurons:
        items = [p.model_copy() for p in req.neurons]
    elif req.layer is None or req.n is None:
        raise HTTPException(status_code=400, detail="Name the neuron to push (layer and n), or give `neurons`.")
    else:
        items = [PushIn(layer=req.layer, n=req.n)]
    for p in items:
        if p.mode is None:
            p.mode = req.mode
        if p.dose is None:
            p.dose = req.dose
    return items


def _sampler(req: Union[PushChatRequest, "SteerRequest"]) -> Tuple[Any, Union[str, Dict[str, Any]]]:
    """The sampler a push chat (one sample) or a whole-text steer (`samples`) asks for, as write_rows takes it, and how
    its reply says the voices were written (backend/sampler_request.py); refused in plain words."""
    k = getattr(req, "samples", 1)
    if isinstance(k, bool) or not 1 <= int(k) <= MAX_STEER_SAMPLES:
        raise ValueError(f"A steer writes 1 to {MAX_STEER_SAMPLES} samples (three rows each, in one batch), not {k}.")
    return sr.for_writing(sr.sampler_of(req.sampler), int(k))


def _steer_whole(a: Atlas, req: SteerRequest, ps: List[ao.Push], hs: List[ao.HeadScale], smp: Any,
                 sampling: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    k = int(req.samples)
    ids = a.chat_ids([{"role": "user", "content": req.prompt}])
    n = len(ids) + int(req.max_new)
    if k > 1 and 3 * k * n > MAX_STEER_ROW_PIECES:
        most = MAX_STEER_ROW_PIECES // (3 * n)
        raise ValueError(f"{k} samples are {3 * k} rows of up to {n:,} word-pieces (the prompt's {len(ids):,} and {req.max_new} "
                         f"written), {3 * k * n:,} in all; several samples are written only while their rows hold at most "
                         f"{MAX_STEER_ROW_PIECES:,}. Ask for " + (f"at most {most} samples" if most > 1 else "one sample")
                         + " at this length, or a shorter answer (max_new).")
    iv, echo = a.interventions(ps, hs, ao.THREE_VOICES * k)
    rows = [ids] * (3 * k)
    td = ao.TextDeltas(a.rt.tokenizer, 3 * k)
    texts = [""] * (3 * k)
    for st in ao.write_rows(a.lm, a.W_U, rows, iv, max_new=req.max_new, stop_ids=a.rt.end_of_turn_ids(),
                            bos_id=a.rt.tokenizer.bos_token_id, softcap=ao.softcap_of(a.rt.hf_model), sampler=smp,
                            samples_of=[r // 3 for r in range(3 * k)] if smp is not None else None):
        texts[st.row] += td.add(st.row, st.token_id, st.end)
    voices: Dict[str, Any] = {"up": texts[0::3], "baseline": texts[1::3], "down": texts[2::3]}
    if k == 1:  # one sample: the whole texts, as before
        voices = {key: v[0] for key, v in voices.items()}
    return {"prompt": req.prompt, "pushed": echo, **voices, "sampling": sampling}


@router.post("/steer")
async def steer(req: SteerRequest) -> Dict[str, Any]:
    """The three voices as whole texts (the scientist's `steer`): pushed up, as it is, pushed down; greedily, or with
    the instrument's sampler, `samples` of them (each voice a list when more than one, sample s drawn with seed + s, and
    several only while their rows hold at most MAX_STEER_ROW_PIECES word-pieces), the three voices of a sample sharing
    each draw. `sampling` says how they were written."""
    a = atlas()
    ps, hs = off_gpu(a.parse_pushes, _push_items(req))
    smp, sampling = off_gpu(_sampler, req)
    return await on_gpu(_steer_whole, a, req, ps, hs, smp, sampling)


def read_ids(a: Atlas, ids: Sequence[int]) -> List[int]:
    """Word-piece ids to read exactly as given (a stored passage's, which decoded text need not tokenise back to),
    checked in plain words."""
    if not ids:
        raise ValueError("Give the word-piece ids to read.")
    if len(ids) > MAX_TEXT_POSITIONS:
        raise ValueError(f"The ids are {len(ids)} word-pieces; this reading takes at most {MAX_TEXT_POSITIONS:,}. Shorten them.")
    V = int(a.W_U.shape[0])
    bad = [int(t) for t in ids if not 0 <= int(t) < V]
    if bad:
        raise ValueError(f"Word-piece {bad[0]} does not exist: the vocabulary has {V:,} (0 to {V - 1}).")
    return [int(t) for t in ids]


def _read(a: Atlas, req: ReadRequest) -> Dict[str, Any]:
    _check_neuron(a, req.layer, req.n)
    if (req.text is None) == (req.ids is None):
        raise ValueError("Give a text to read (text) or its word-piece ids (ids), one of the two.")
    ids = a.ids_of(req.text, req.chat) if req.text is not None else read_ids(a, req.ids)
    iv, echo = a.interventions(*a.parse_pushes(req.neurons, req.heads), (1,))
    acts = ao.read_neuron(a.lm, ids, req.layer, req.n, iv, bos_id=a.rt.tokenizer.bos_token_id, spare_bos=req.spare_bos)
    units = []
    for text, idx in a.units(ids):
        vals = [acts[i] for i in idx]
        units.append({"text": text, "act": max(vals, key=abs), "pieces": [round(v, 4) for v in vals], "pos": idx[-1]})
    return {"layer": req.layer, "n": req.n, "units": units, "act_range": a.act_range(req.layer, req.n), "pushed": echo}


@router.post("/reads")
async def reads(req: ReadRequest) -> Dict[str, Any]:
    """The neuron's activation at every word-piece of a text (the reference's "reads you" and the scientist's
    `activations`), by display unit (the strongest piece of each), optionally with other neurons pushed and heads scaled
    in the same pass, to test what drives it: only what comes before its feed-forward block can move it. `ids` in place
    of `text` reads those word-pieces exactly as given (no start mark added, `chat` unused): a stored passage's ids
    read it again as its window was read."""
    a = atlas()
    return await on_gpu(_read, a, req)


def _logprobs(a: Atlas, req: LogprobsRequest) -> Dict[str, Any]:
    ids = a.ids_of(req.text, req.chat, prefix=req.prefix)
    iv, echo = a.interventions(*a.parse_pushes(req.neurons, req.heads), (0, 1))
    base, alt = ao.next_logprobs(a.lm, a.W_U, ids, iv, bos_id=a.rt.tokenizer.bos_token_id, spare_bos=req.spare_bos)
    tok = a.rt.tokenizer
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    targets = []
    for t in req.targets:
        tid = tok.encode(t, add_special_tokens=False)
        if tid:
            targets.append((t, int(tid[0]), len(tid) > 1))

    def view(lp: torch.Tensor) -> Dict[str, Any]:
        v, i = lp.topk(req.k)
        return {"top": [[dec(t), round(float(x), 3)] for x, t in zip(v, i)],
                "targets": [{"text": t, "token": dec(tid), "logprob": round(float(lp[tid]), 3), **({"first_piece_only": True} if multi else {})}
                            for t, tid, multi in targets]}

    if alt is None:
        return {"read_at": len(ids) - 1, **view(base)}
    d = alt.exp() - base.exp()  # by probability gained or lost, so the far tail's large log ratios stay out
    up, down = d.topk(12), (-d).topk(12)
    row = lambda i: [dec(i), round(float(base[i]), 2), round(float(alt[i]), 2)]  # noqa: E731
    return {"read_at": len(ids) - 1, "baseline": view(base), "intervened": view(alt),
            "kl": round(float((base.exp() * (base - alt)).sum()), 5),
            "rose": [row(i) for i in up.indices], "fell": [row(i) for i in down.indices],
            "rows_are": "[word-piece, baseline log-probability, intervened log-probability]", "pushed": echo}


@router.post("/logprobs")
async def logprobs(req: LogprobsRequest) -> Dict[str, Any]:
    """The next word-piece's log-probabilities after a text (a chat turn by default, the next piece opening the reply;
    `prefix` begins it), fp32 softcapped; with pushes or heads scaled, the base and the intervened rows of one forward,
    the KL from base to intervened and the pieces that gained and lost the most probability."""
    return await on_gpu(_logprobs, atlas(), req)


def _compare(a: Atlas, req: CompareRequest) -> Dict[str, Any]:
    neurons = []
    for p in req.neurons:
        if p.n is None:
            raise ValueError("compare reads neurons: give each its layer and n.")
        _check_neuron(a, p.layer, p.n)
        neurons.append((int(p.layer), int(p.n)))
    texts = [(t.label or f"t{i + 1}", t.text) if isinstance(t, TextIn) else (f"t{i + 1}", str(t)) for i, t in enumerate(req.texts)]
    sc = a.scales()
    rows = {nk: {"layer": nk[0], "n": nk[1], "act_range": a.act_range(*nk, sc), "by_text": {}} for nk in neurons}
    for label, text in texts:
        ids = a.ids_of(text, req.chat)
        units = a.units(ids)
        unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
        got = ao.read_neurons(a.lm, ids, neurons)
        lo = 1 if len(ids) > 1 else 0  # the start mark left out
        for (l, m), vals in got.items():
            v = np.array(vals, np.float32)
            s = scale_scores(v[None, :, None], sc["pos"][l : l + 1, m : m + 1], sc["neg"][l : l + 1, m : m + 1])[0, :, 0]
            hi_i, lo_i, pk = (lo + int(x[lo:].argmax()) for x in (v, -v, np.abs(s)))
            rows[(l, m)]["by_text"][label] = {"max": [round(float(v[hi_i]), 3), units[unit_of[hi_i]][0]],
                                             "min": [round(float(v[lo_i]), 3), units[unit_of[lo_i]][0]],
                                             "peak_score": round(float(s[pk]), 2), "peak_at": units[unit_of[pk]][0]}
    return {"texts": [{"label": lb, "chars": len(t)} for lb, t in texts], "neurons": list(rows.values()), "scale_source": sc["source"]}


@router.post("/compare")
async def compare(req: CompareRequest) -> Dict[str, Any]:
    """Many neurons across several texts, one forward per text: per neuron and text its strongest positive and negative
    activation with its display unit, and its peak score (activation over its usual extreme on that side; the start
    mark left out)."""
    return await on_gpu(_compare, atlas(), req)


def _unit_at(units: List[Tuple[str, List[int]]], at: Optional[int], token: Optional[str]) -> Optional[int]:
    """The display unit a request points at: `at` (counting from the end when negative), else the first holding
    `token`; None when it names neither. A token no unit holds is refused in plain words."""
    if at is not None:
        if not -len(units) <= int(at) < len(units):
            raise ValueError(f"The text has {len(units)} units (0 to {len(units) - 1}); there is no unit {at}.")
        return int(at) % len(units)
    if token:
        k = _find_unit(units, token)
        if k is None:
            raise ValueError(f"No unit of the text holds {token!r}.")
        return k
    return None


def _fires(a: Atlas, req: FiresRequest) -> Dict[str, Any]:
    ids = a.ids_of(req.text, req.chat)
    sc = a.scales()
    units = a.units(ids)
    toks = [u for u, _ in units]
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    P = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))  # as atlas_index.scale_scores scales each side
    Q = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))
    k = _unit_at(units, req.at, req.token)
    if k is not None:
        # one unit: every neuron at its last word-piece
        pos = units[k][1][-1]
        A = ao.activations_at(a.lm, ids, pos).numpy() if pos > 0 else np.zeros((a.L, a.N), np.float32)
        sides = ((A / P, A, np.full(A.shape, pos), "+"), (A / Q, A, np.full(A.shape, pos), "-"))
    else:
        # the whole text: each neuron where it fires hardest on each side, past the start mark
        mx, amx, mn, amn = (t.numpy() for t in ao.activation_peaks(a.lm, ids))
        sides = ((mx / P, mx, amx, "+"), (mn / Q, mn, amn, "-"))
    lo, hi = req.layers if req.layers else (0, a.L - 1)
    cand = []
    for score, raw, at, sg in sides:
        s = np.where(score > 0, score, 0).astype(np.float32)
        s[: max(lo, 0)] = 0
        s[hi + 1 :] = 0
        flat = s.ravel()
        top = np.argpartition(-flat, req.limit)[: req.limit] if flat.size > req.limit else np.arange(flat.size)
        cand += [(float(flat[i]), int(i) // a.N, int(i) % a.N, sg, int(at.ravel()[i]), float(raw.ravel()[i])) for i in top if flat[i] > 0]
    cand.sort(key=lambda c: -c[0])
    edited = a.edited()
    res = [{"layer": l, "n": n, "face": sg, "score": round(s, 2), "act": round(r, 3), "pos": t, "token": toks[unit_of[t]],
            "words": a.face(l, n, sg, 6, edited)} for s, l, n, sg, t, r in cand[: req.limit]]
    return {"tokens": toks, "at": k, "scale_source": sc["source"],
            "note": "score = the activation over the neuron's usual extreme on that side; 1 = as strong as it usually gets",
            "results": res}


@router.post("/fires")
async def fires(req: FiresRequest) -> Dict[str, Any]:
    """The neurons that fire hardest on a text against their usual range (the reference's /api/fires): each neuron's
    strongest activation on each side as a multiple of its usual extreme there, with the word-piece where it is reached
    (the start mark left out); or, with `at` or `token`, at that one display unit. `layers` [lo, hi] keeps a range."""
    return await on_gpu(_fires, atlas(), req)


# -- the run view: a text run once, every activation kept


def _run(a: Atlas, req: RunRequest) -> Dict[str, Any]:
    ids = a.ids_of(req.text, req.chat, limit=MAX_RUN_POSITIONS)
    sc = a.scales()
    acts = ao.all_activations(a.lm, ids).numpy()
    units = a.units(ids)
    a.run = {"ids": ids, "acts": acts, "units": units, "scales": sc,
             "toks": [units[k][0] for i in range(len(ids)) for k, (_, idx) in enumerate(units) if i in idx]}
    return {"ready": True, "tokens": [u for u, _ in units], "pos": [idx[-1] for _, idx in units], "scale_source": sc["source"],
            "n_pieces": len(ids)}


@router.post("/run")
async def run(req: RunRequest) -> Dict[str, Any]:
    """Run a text once and keep every neuron's activation at every word-piece (at most MAX_RUN_POSITIONS) for
    /run/view, /run/heads and /run/paths; returns its display units and each one's last word-piece."""
    return await on_gpu(_run, atlas(), req)


def _run_scores(a: Atlas, pos: int, sign: str) -> Tuple[np.ndarray, np.ndarray]:
    """[L, N] scores at `pos` (or each neuron's strongest piece, pos -1, past the start mark) and the piece each comes
    from."""
    acts, sc = a.run["acts"], a.run["scales"]
    T = acts.shape[1]
    if pos >= 0:
        S = scale_scores(acts[:, pos : pos + 1], sc["pos"], sc["neg"])[:, 0] if pos > 0 else np.zeros(acts[:, 0].shape, np.float32)
        return S, np.full(S.shape, pos)
    best = np.zeros((a.L, a.N), np.float32)
    at = np.zeros((a.L, a.N), np.int64)
    for l in range(a.L):
        S = scale_scores(acts[l : l + 1], sc["pos"][l : l + 1], sc["neg"][l : l + 1])[0]  # [T, N]
        idx = S.argmax(0) if sign == "+" else S.argmin(0) if sign == "-" else np.abs(S).argmax(0)
        best[l] = np.take_along_axis(S, idx[None], 0)[0]
        at[l] = idx
    del T
    return best, at


@router.get("/run/view")
async def run_view(pos: int = -1, sign: str = "both", lo: int = 0, hi: int = 999, min_score: float = Query(1.0, alias="min"),
                   q: str = "", limit: int = 300) -> Dict[str, Any]:
    """The neurons that pass the filters for the text last run: at display position `pos` (a word-piece index), or each
    at its strongest piece (-1); `sign` "+" | "-" | "both"; layers lo..hi; |score| at least `min`; a face whose words
    hold `q`. Returns the list (strongest first) and every neuron's score there for the map (base64 float16, [L, N])."""
    a = atlas()
    if not a.run:
        return {"results": [], "total": 0}
    if sign not in ("+", "-", "both"):
        raise HTTPException(status_code=400, detail='sign is "+", "-" or "both".')
    if pos >= len(a.run["ids"]):
        return {"results": [], "total": 0}

    def work() -> Dict[str, Any]:
        best, at = _run_scores(a, pos, sign)
        view = best.copy()
        keep = np.abs(best) >= float(min_score)
        if sign == "+":
            keep &= best > 0
        if sign == "-":
            keep &= best < 0
        lay = np.arange(a.L)[:, None]
        keep &= (lay >= lo) & (lay <= hi)
        if q.strip():
            match = match_ids(a.vocab(), q)
            hit = np.zeros_like(keep)
            lens = a.index.get("lens")
            if lens is not None:
                lh = keep & np.where(best[..., None] >= 0, np.isin(lens["lens_pos"][:, :, :6], match) | np.isin(lens["eye_pos"][:, :, :6], match),
                                     np.isin(lens["lens_neg"][:, :, :6], match) | np.isin(lens["eye_neg"][:, :, :6], match)).any(-1)
                # a label the gate holds back is not the neuron's words in the main reading, so it does not match
                for l, n in zip(*np.nonzero(lh)):
                    hit[l, n] = a.index.label_passes(int(l), int(n), "+" if best[l, n] >= 0 else "-")
            st = a.index.get("steer_lens")
            if st is not None:
                hit |= np.where(best[..., None] >= 0, np.isin(st["up"][:, :, :6], match), np.isin(st["down"][:, :, :6], match)).any(-1)
            keep &= hit
        view[~keep] = 0
        ls, ns = np.nonzero(keep)
        order = np.argsort(-np.abs(best[ls, ns]))[: max(1, min(int(limit), 2000))]
        edited = a.edited()
        res = []
        for o in order:
            l, n = int(ls[o]), int(ns[o])
            sg = "+" if best[l, n] > 0 else "-"
            t = int(at[l, n])
            res.append({"layer": l, "n": n, "face": sg, "score": round(float(abs(best[l, n])), 2), "act": round(float(a.run["acts"][l, t, n]), 3),
                        "token": a.run["toks"][t], "pos": t, "words": a.face(l, n, sg, 6, edited)})
        blob = base64.b64encode(view.astype("<f2").tobytes()).decode()
        return {"results": res, "total": int(keep.sum()), "map": blob, "shape": list(view.shape), "scale_source": a.run["scales"]["source"]}

    return await asyncio.to_thread(work)


def _capture(a: Atlas, ids: List[int], streams: bool = False, act_pos: Optional[int] = None) -> ao.HeadCapture:
    key = (tuple(ids), streams, a.edit_key(), act_pos)
    hit = a.caps.get(key)
    if hit is None:
        if len(ids) > MAX_HEAD_POSITIONS:
            raise ValueError(f"The text is {len(ids)} word-pieces; heads are read on at most {MAX_HEAD_POSITIONS}.")
        hit = a.caps.put(key, ao.capture_heads(a.lm, ids, a.w, streams=streams, act_pos=act_pos))
    return hit


@router.get("/run/heads")
async def run_heads(pos: int) -> Dict[str, Any]:
    """Every head's write at a word-piece of the text last run, per layer, as a share of the largest write any head of
    that layer makes at any piece past the start mark."""
    a = atlas()
    if not a.run or not 0 <= pos < len(a.run["ids"]):
        return {"layers": None}

    def work() -> Dict[str, Any]:
        cap = _capture(a, a.run["ids"])
        out = []
        for l in range(a.L):
            W = cap.writes[l]  # [T, H]
            top = float(W[1:].max()) if W.shape[0] > 1 else float(W.max())
            out.append([round(float(x) / max(top, 1e-8), 3) for x in W[pos].tolist()])
        return {"layers": out}

    return await on_gpu(work)


@router.get("/run/paths")
async def run_paths(pos: int, top: int = 300, min_score: float = Query(1.0, alias="min"), model: str = "rest") -> Response:
    """The circuit active at one word-piece of the text last run (the reference's /api/infer/paths): among the `top`
    neurons with the largest |score| there, every wiring link whose two ends both fired, signed by the push the writer's
    firing gives the reader: from rest (a link kept only when its writer fired on its `gate` face), or with `model`
    "use" at the reader's operating point on the face it fired on (AtlasIndex.active_paths). Rows as /web's (hop 1);
    X-Active counts the neurons that fired, X-Model names the reading, and X-Use-Refused says why none could be read in
    use. Needs the wiring index."""
    a = atlas()
    _model(model)
    if not a.run or not 0 <= pos < len(a.run["ids"]):
        return Response(b"", media_type="application/octet-stream", headers={"X-Web-Total": "0", "X-Active": "0", "X-Model": model})

    def work() -> Tuple[np.ndarray, int]:
        S, _ = _run_scores(a, pos, "both")
        return a.index.active_paths(S, max(1, min(int(top), 2000)), float(min_score), model)

    rows, n_active = await asyncio.to_thread(work)
    why = await asyncio.to_thread(a.index.use_refused) if model == "use" else None
    return Response(rows.tobytes(), media_type="application/octet-stream",
                    headers={"X-Web-Total": str(len(rows)), "X-Active": str(n_active), "X-Model": model,
                             **({"X-Use-Refused": why} if why else {})})


@router.get("/head/row")
async def head_row(layer: int, head: int, pos: int) -> Dict[str, Any]:
    """One head's attention at a word-piece of the text last run, summed per display unit (units after it are null)."""
    a = atlas()
    _check_head(a, layer, head)
    if not a.run or not 0 <= pos < len(a.run["ids"]):
        return {"units": None}

    def work() -> Dict[str, Any]:
        cap = _capture(a, a.run["ids"])
        A = ao.head_pattern(cap, a.shapes[layer], head)[pos]
        return {"units": [round(float(A[idx].sum()), 4) if idx[0] <= pos else None for _, idx in a.run["units"]]}

    return await on_gpu(work)


@router.post("/head/pattern")
async def head_pattern(req: HeadTextRequest) -> Dict[str, Any]:
    """One head's attention on a text (the text given, else a sample sentence), by display unit: each row a unit
    reading (at its last piece), each column the weight it gives each unit up to itself."""
    a = atlas()
    _check_head(a, req.layer, req.head)

    def work() -> Dict[str, Any]:
        text = req.text or ao.HEAD_TEXTS[0]
        ids = a.ids_of(text, req.chat, limit=MAX_HEAD_POSITIONS)
        cap = _capture(a, ids)
        A = ao.head_pattern(cap, a.shapes[req.layer], req.head).numpy()
        units = a.units(ids)
        cols = np.stack([A[:, idx].sum(1) for _, idx in units], 1)
        rows = [[round(float(x), 4) for x in cols[idx[-1], : u + 1]] for u, (_, idx) in enumerate(units)]
        return {"text": text, "tokens": [t for t, _ in units], "rows": rows, "kind": a.shapes[req.layer].kind,
                "window": a.shapes[req.layer].window}

    return await on_gpu(work)


def _pool_head(a: Atlas, nodes: List[Dict[str, Any]], edited: set, top: int = 24) -> Dict[str, Any]:
    """A head's linked neurons' words pooled as the web summary pools a cone (weighted by z, ranked by lift over each
    layer's base rate): the steer lens's when built, else the lens's, and then only faces the gate lets through."""
    vocab = a.vocab()
    st = a.index.get("steer_lens")
    lens = a.index.get("lens")
    src = "steer" if st is not None else "lens" if lens is not None else None
    if src is None or not nodes:
        return {"src": src, "tokens": [], "gated": 0}
    pos = st["up"] if src == "steer" else lens["lens_pos"]
    base = a.index._base_rates(src, (pos, st["down"] if src == "steer" else lens["lens_neg"]), a.L, len(vocab))
    acc: Dict[str, Dict[str, Any]] = {}
    zmax = max([c["z"] for c in nodes] + [1e-8])
    gated = used = 0
    for c in nodes:
        if src == "lens" and not a.index.label_passes(c["layer"], c["n"], "+"):
            gated += 1
            continue
        used += 1
        for t in pos[c["layer"], c["n"], :8]:
            s = vocab[int(t)]
            key = s.strip().lower()
            if not key or "�" in key:
                continue
            e = acc.setdefault(key, {"n": 0, "w": 0.0, "base": 0.0, "forms": {}})
            e["n"] += 1
            e["w"] += c["z"] / zmax
            e["base"] += float(base[c["layer"], int(t)])
            e["forms"][s.strip()] = e["forms"].get(s.strip(), 0) + 1
    for e in acc.values():
        lift = (e["n"] / max(1, used)) / (e["base"] / e["n"])
        e["score"] = e["w"] * float(np.log(lift)) if lift > 1 else 0.0
    ranked = sorted((e for e in acc.values() if e["score"] > 0), key=lambda e: -e["score"])[:top]
    return {"src": src, "tokens": [{"tok": max(e["forms"], key=e["forms"].get), "n": e["n"], "w": round(e["score"], 2)} for e in ranked],
            "gated": gated}


def _head(a: Atlas, layer: int, head: int) -> Dict[str, Any]:
    key = (a.edit_key(), layer, head)
    links = a.cache["head"].get(key)
    if links is None:
        links = a.cache["head"].put(key, ao.head_wiring(a.spaces, layer, head, k=40))
    edited = a.edited()
    out: Dict[str, Any] = {"layer": layer, "head": head, "kind": a.shapes[layer].kind, "kv_head": a.shapes[layer].kv_of(head),
                           "values_from_keys": a.shapes[layer].values_from_keys}
    for role in ("q", "k", "v", "o"):
        rows = [{**c, "words": a.face(c["layer"], c["n"], "+", 5, edited)} for c in links[role]]
        out[role] = rows
        out[f"{role}_pooled"] = _pool_head(a, links[role], edited)
    return out


@router.get("/head")
async def head(layer: int, head: int) -> Dict[str, Any]:
    """What one attention head reads and writes, from the weights (atlas_ops.head_wiring): the earlier neurons feeding
    its query (q), key (k) and value (v), the later neurons reading its write (o), each with its words (positive face)
    and the role's words pooled."""
    a = atlas()
    _check_head(a, layer, head)
    return await on_gpu(_head, a, layer, head)


def _profiles(a: Atlas) -> Dict[str, Any]:
    if a._profiles is not None:
        return a._profiles
    path = a.index.root / "head_profiles.json"
    key = json.dumps({"texts": list(ao.HEAD_TEXTS), "model": getattr(a.rt, "model_id", None), "layers": a.L})
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("key") == key:
            a._profiles = d
            return d
    except (OSError, ValueError):
        pass
    ids = [a.rt.tokenize_prompt(t) for t in ao.HEAD_TEXTS]
    prof = ao.head_profiles(a.lm, a.w, a.rt.tokenizer, ids)
    d = {"key": key, "texts": list(ao.HEAD_TEXTS), **prof}
    if not getattr(a.rt, "atlas_stand_in", False):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    a._profiles = d
    return d


@router.get("/head/profile")
async def head_profile(layer: int, head: int) -> Dict[str, Any]:
    """Where one head looks over eight reference texts (atlas_ops.head_profiles), each pair with its reading and read
    units and the text it comes from. The first call reads every head (one forward per text)."""
    a = atlas()
    _check_head(a, layer, head)
    d = await on_gpu(_profiles, a)
    p = d["heads"].get(f"{layer}/{head}")
    if p is None:
        return {"error": "no such head"}
    pairs = [{"w": w_, "text": d["texts"][t], "q": qu, "k": ku, "qt": d["tokens"][t][qu], "kt": d["tokens"][t][ku]} for w_, t, qu, ku in p["pairs"]]
    return {"layer": layer, "head": head, "dist": p["dist"], "dist_names": ["self", "previous", "2-5 back", "further", "first"],
            "src": p["src"], "pairs": pairs}


def _attention(a: Atlas, req: HeadTextRequest) -> Dict[str, Any]:
    _check_head(a, req.layer, req.head)
    ids = a.ids_of(req.text or ao.HEAD_TEXTS[0], req.chat, limit=MAX_HEAD_POSITIONS)
    cap = _capture(a, ids)
    units = a.units(ids)
    toks = [u for u, _ in units]
    qu = _unit_at(units, req.at, req.token)
    qu = len(units) - 1 if qu is None else qu
    A = ao.head_pattern(cap, a.shapes[req.layer], req.head).numpy()
    Uu = np.stack([A[:, idx].sum(1) for _, idx in units], 1)[[idx[-1] for _, idx in units]]  # [U, U]
    row = [[toks[j], round(float(Uu[qu, j]), 3)] for j in range(qu + 1)]
    pairs = sorted(((float(Uu[i, j]), i, j) for i in range(len(units)) for j in range(1, i)), key=lambda x: -abs(x[0]))
    return {"layer": req.layer, "head": req.head, "kind": a.shapes[req.layer].kind, "window": a.shapes[req.layer].window,
            "tokens": toks, "query": {"index": qu, "token": toks[qu]}, "row": row,
            "top_sources": sorted(row, key=lambda r: -abs(r[1]))[:8],
            "first_unit_weight": round(float(np.abs(Uu[1:, 0]).mean()), 3) if len(units) > 1 else None,
            "strongest_reads": [{"query": toks[i], "source": toks[j], "weight": round(w_, 3), "back": i - j} for w_, i, j in pairs[:12]]}


@router.post("/attention")
async def attention(req: HeadTextRequest) -> Dict[str, Any]:
    """One head's attention on a text (the scientist's `attention`): at the reading unit (the first holding `token`, the
    unit `at`, else the last) the weight on every unit up to it, and the strongest single reads anywhere in the text."""
    return await on_gpu(_attention, atlas(), req)


DRIVE_NEURONS = 12  # the earlier neurons a drive names, the strongest shares across every block


def _drive(a: Atlas, req: DriveRequest, notes: Dict[str, Any]) -> Dict[str, Any]:
    _check_neuron(a, req.layer, req.n)
    if (req.text is None) == (req.ids is None):
        raise ValueError("Give a text to read (text) or its word-piece ids (ids), one of the two.")
    if req.ids is not None:
        ids = read_ids(a, req.ids)
        if len(ids) > MAX_HEAD_POSITIONS:
            raise ValueError(f"The ids are {len(ids)} word-pieces; heads are read on at most {MAX_HEAD_POSITIONS}. Give the last {MAX_HEAD_POSITIONS} or fewer.")
    else:
        ids = a.ids_of(req.text, req.chat, limit=MAX_HEAD_POSITIONS)
    units = a.units(ids)
    toks = [u for u, _ in units]
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    k = _unit_at(units, req.at, req.token)
    if k is None:
        acts = ao.read_neuron(a.lm, ids, req.layer, req.n)
        k = unit_of[int(np.abs(np.array(acts[1:])).argmax()) + 1] if len(acts) > 1 else 0
    pos = units[k][1][-1]
    # every neuron's activation at the word-piece read, kept in the same forward: each block's part split over them
    cap = _capture(a, ids, streams=True, act_pos=pos)
    d = ao.attention_drive(cap, a.w, a.shapes, req.layer, req.n, pos, top=req.top, neurons=DRIVE_NEURONS)
    attn = sum(c for _, _, c in d["heads"])
    heads = []
    for l, h, c in sorted(d["heads"], key=lambda r: -abs(r[2]))[: max(req.top, 12)]:
        row: Dict[str, Any] = {"layer": l, "head": h, "contribution": round(c, 5), "share": round(c / d["total"], 3) if d["total"] else None}
        if (l, h) in d["src"]:
            per: Dict[int, float] = {}
            for i, v in enumerate(d["src"][(l, h)]):
                per[unit_of[i]] = per.get(unit_of[i], 0.0) + v
            row["from"] = [[toks[u], round(v, 5)] for u, v in sorted(per.items(), key=lambda x: -abs(x[1]))[:5]]
        heads.append(row)
    mlps = sorted(d["mlps"], key=lambda r: -abs(r[1]))[:8]
    sums = dict(d.get("block_sums", []))
    edited = a.edited()
    neurons = []
    for l, m, act, c in d.get("neurons", []):
        face = "+" if act > 0 else "-"
        e = notes.get(fn.note_key(l, m)) or {}
        neurons.append({"layer": l, "n": m, "act": float(f"{act:.6g}"), "face": face, "contribution": float(f"{c:.6g}"),
                        "share": round(c / d["total"], 4) if d["total"] else None, "words": a.face(l, m, face, 5, edited),
                        "name": e.get("name", ""), "name_by": e.get("name_by") if e.get("name") else None})
    return {"layer": req.layer, "n": req.n, "read": "ids" if req.ids is not None else "text", "token": toks[k], "index": k,
            "read_pos": pos, "act": round(d["act"], 5),
            "act_range": a.act_range(req.layer, req.n), "linearised_total": round(d["total"], 5),
            "attention_sum": round(attn, 5), "mlp_sum": round(sum(c for _, c in d["mlps"]), 5), "embedding": round(d["embed"], 5),
            "rest": round(d["rest"], 6), "heads": heads,
            "mlps": [{"layer": l, "contribution": round(c, 5), "neurons_sum": float(f"{sums[l]:.6g}")} for l, c in mlps],
            "neurons": neurons,
            "note": ("the activation gelu(g) u linearised in the stream entering its feed-forward block, the norm's rms held "
                     "at its value there; the parts (heads, feed-forward blocks, the embedding) add up to the total"),
            "neurons_note": ("each earlier block's part split over its neurons: a neuron's direct write through its block's norm, "
                             "held at its scale there, read along the same line; its share of the total. What it does through "
                             "the layers between is in those layers' parts, not its own: leads to test, not causes found")}


@router.post("/attn_drive")
async def attn_drive(req: DriveRequest) -> Dict[str, Any]:
    """Which attention heads, feed-forward blocks and earlier neurons drive a neuron at one unit of a text
    (atlas_ops.attention_drive): the unit holding `token`, the unit `at`, else where it fires hardest; `text` read as
    POST /reads reads it, or `ids` exactly as given (a sighting's own word-pieces; at most 512). Which earlier units each
    top head read its part from; each block's part split over its neurons (`neurons_sum`), and the twelve strongest
    neurons across every block (`neurons`), each on the face it fired on, with that face's words, their check and its
    name."""
    notes = (await fn.all_notes())["neurons"]
    return await on_gpu(_drive, atlas(), req, notes)


# -- the scientist's whole-neuron views


@router.get("/neuron")
async def neuron(layer: int, n: int) -> Dict[str, Any]:
    """Everything kept for one neuron, for the scientist: both faces' words with their checks, the steer lens and KL,
    the activation range, its census rank, its 24 strongest links each way (each linked face's words and name)."""
    a = atlas()
    _check_neuron(a, layer, n)
    lens = await neuron_lens(layer, n)
    con = await neuron_wiring(layer, n, 24)
    u = await census_unit(layer, n)
    return {"layer": layer, "n": n, "name": u.get("name", ""), "faces": lens["faces"], "gate_trigger": lens["gate_trigger"],
            "steer": lens["steer"], "act_range": lens["act_range"], "edited": lens["edited"], "ceiling": lens.get("ceiling"),
            **({"steer_stale": lens["steer_stale"]} if lens.get("steer_stale") else {}),
            "census": {"rank": u.get("rank"), "n_ranked": u.get("n_ranked"), "stats": u.get("stats"),
                       "n_rollouts": len(u.get("rollouts", []))} if "run" in u else None,
            "writes_into": con["down"], "reads_from": con["up"], "wiring_source": con["source"],
            "label_gate": LABEL_GATE_CARD, "lens": a.rt.lens_card()}


@router.get("/cone")
async def cone(layer: int, n: int, k: int = 2, limit: int = 120, model: str = "rest") -> Dict[str, Any]:
    """The neuron's cone as a list of neurons (the scientist's `cone`), followed by the link model
    (atlas_ops.LINK_MODEL, AtlasIndex.walk): downstream from the neuron firing on its + face (`when: "+"`) and on its -
    face (`when: "-"`), upstream to the writers that push it toward either face (`drives`); each reached neuron with its
    hop, the strength of the link that reached it (w, over the strongest of its node's links), z, the face it is read on
    and that face's words. With `model` "use" each link is read at its reader's operating point (LINK_MODEL_USE; not
    ready without the passages index, saying why), upstream toward each face in turn. Multi-hop paths are weight-only
    guesses: leads to check, not circuits."""
    a = atlas()
    _check_neuron(a, layer, n)
    _model(model)
    notes = (await fn.all_notes())["neurons"]
    edited = a.edited()

    def work() -> Dict[str, Any]:
        if a.index.get("wiring") is None:
            return {"ready": False, "note": "the wiring index is not built: the cone needs every neuron's links (scripts/atlas_build.py wiring)"}
        if model == "use" and a.index.use_refused():
            return {"ready": False, "model": "use", "note": a.index.use_refused()}
        reached: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
        for f in ("+", "-"):
            reached += [(t, {**r, "when": f}) for t, r in a.index.walk(layer, n, "down", f, None, max(1, int(k)), model=model).items()]
        if model == "use":
            for f in ("+", "-"):
                reached += list(a.index.walk(layer, n, "up", f, None, max(1, int(k)), model="use").items())
        else:
            reached += list(a.index.walk(layer, n, "up", None, None, max(1, int(k))).items())
        per_hop: Dict[int, int] = {}
        for _, r in reached:
            per_hop[r["hop"]] = per_hop.get(r["hop"], 0) + 1
        nodes = sorted(reached, key=lambda kv: (abs(kv[1]["hop"]), -kv[1]["w"]))[: max(1, min(int(limit), 1000))]
        read = ("read from rest (" + ao.LINK_MODEL + ")") if model == "rest" else ("read " + ao.LINK_MODEL_USE)
        return {"ready": True, "neurons": len(reached), "per_hop": {str(h): c for h, c in sorted(per_hop.items())},
                **({"stale": STALE_WEB} if (layer, n) in edited else {}),
                **({"model": "use"} if model == "use" else {}),
                **({"use_stale": USE_STALE} if model == "use" and any(l_e < a.L - 1 for l_e, _ in edited) else {}),
                "note": ("weight-only direct-path links chained, " + read + " Paths past hop 1 are "
                         "leads, not circuits. hop > 0 reads the neuron (when: the face it fires on), hop < 0 feeds it "
                         "(drives: the face it pushes the neuron toward); face is the face each is read on"),
                "nodes": [{"layer": l, "n": m, "hop": r["hop"], "w": r["w"], "z": r["z"], "face": r["face"],
                           **({"when": r["when"]} if "when" in r else {}), **({"drives": r["drives"]} if "drives" in r else {}),
                           "words": a.face(l, m, r["face"], 5, edited), "census": census_words(l, a.rt.model_id).get(m, []),
                           "name": (notes.get(fn.note_key(l, m)) or {}).get("name") or None} for (l, m), r in nodes]}

    return await asyncio.to_thread(work)
