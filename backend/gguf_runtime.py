"""The instrument's runtime on a GGUF file (`backend/gguf_runtime.py`): `GGUFRuntime`, a `TheodoliteRuntime` whose
model is built from the file (`backend/gguf_model.py`), on the CPU or a GPU.

`backend/engine_runtime.py` is pinned by R4 (experiments/scripts/R4_common.py keeps its sha256, with backend/__init__.py,
theodolite_math.py and label_gate.py), so nothing here edits it. This subclass replaces only what touches the
checkpoint's packing:
- `load()` builds the model from the GGUF, then does everything the pinned `load()` does once its model is built, in
  the same order: `ie_missing_keys`, the EagerModel, the cached layers, final norm, lm_head and feed-forward norms, the
  J++ lens (and the Jacobian lens when asked) registered on interp_engine's lens stream, the allocator's settings and
  the log line. Two things differ: the lenses can be left out (`load_jpp=False`: a census and the atlas's stats,
  wiring, layout and steer builds do not read them, and they cost 3.4 GB; the sightings, the lens build and the vault's
  UPDATE and INSERT do, and fail without them), and the feed-forward norms are read for every layer the model has (the
  pinned loop counts 60, which only the 31B has).
- `slice_w_down`, `slice_w_gate`, `slice_w_up`, `get_full_w_down`, `get_full_w_gate`, `get_full_w_up` and
  `write_col_norms`, the seven accessors that read `weight_packed`, read through `gguf_weights.dense`, `dense_cols` and
  `dense_rows` instead, with the vault's overlays laid on as the pinned ones lay them (and, as there, none on
  `slice_w_up` or `get_full_w_up`). `write_col_norms` walks the layer's own width, which is the pinned D_FF (21,504) on
  the 31B.
Every other method is the pinned runtime's own, unchanged (the sightings, the writing, the contrasts, the lens card).

The model id it gives every reading (`model_id`, which every index, census run and lens card records, and which the
server compares before it reads an index) is `gguf:<file name>#<fingerprint>` (`gguf_model.gguf_stem_id`), so nothing
taken on one file is read as another file's or as the W4A16 checkpoint's. `weights_card` is the file's provenance with
how the model was built and where its tokenizer came from; GET /api/health sends it ("weights").

scripts/check_gguf.py drives it on the stand-in of scripts/atlas_stand_in.py written to tiny GGUF files.
"""

from __future__ import annotations

import logging
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from backend import engine_runtime as er
from backend import gguf_model as gm
from backend.gguf_weights import dense, dense_cols, dense_rows
from interp_engine import EagerModel

logger = logging.getLogger(__name__)

# Where the tokenizer is looked for first, when cached here: the instrument's checkpoint, then Google's own repo (the
# GGUF's own metadata otherwise).
TOKENIZER_REPOS = (er.DEFAULT_W4A16_MODEL_ID, er.CANONICAL_MODEL_ID)


class GGUFRuntime(er.TheodoliteRuntime):
    """`TheodoliteRuntime` on a GGUF of Gemma 4 (see the module's docstring)."""

    def __init__(
        self,
        gguf_path: Union[str, Path],
        lens_repo_id: str = er.LENS_REPO_ID,
        device: str = "cpu",
        load_jlens: bool = False,
        *,
        linears: str = "packed",
        dtype: torch.dtype = torch.bfloat16,
        hf_config: Union[None, str, Path, Dict[str, Any]] = None,
        tokenizer: Any = None,
        tokenizer_repos: Sequence[str] = TOKENIZER_REPOS,
        load_jpp: bool = True,
    ) -> None:
        super().__init__(model_id=str(gguf_path), lens_repo_id=lens_repo_id, device=device, load_jlens=load_jlens)
        self.gguf_path = Path(gguf_path)
        self.linears = linears
        self.dtype = dtype
        self.hf_config = hf_config
        self.tokenizer_repos = tuple(tokenizer_repos)
        self._given_tokenizer = tokenizer
        self.load_jpp = load_jpp
        # the file's provenance, how the model was built, where the tokenizer came from: {} until `load()`
        self.weights_card: Dict[str, Any] = {}

    def load(self) -> "GGUFRuntime":
        """Build the model from the GGUF, then everything the pinned `load()` does after its model is built."""
        t0 = time.perf_counter()
        logger.info("Loading Gemma 4 from the GGUF %s on %s (%s linear layers, %s)...", self.gguf_path, self.device,
                    self.linears, str(self.dtype).replace("torch.", ""))
        reader = gm.open_gguf(self.gguf_path)
        prov = gm.provenance(self.gguf_path, reader)
        hf_model, built = gm.build_gemma4(self.gguf_path, device=self.device, dtype=self.dtype, linears=self.linears,
                                          hf_config=self.hf_config, reader=reader)
        if self._given_tokenizer is not None:
            tokenizer, tok_rec = self._given_tokenizer, {"source": "given by the caller"}
        else:
            tokenizer, tok_rec = gm.load_tokenizer(self.gguf_path, self.tokenizer_repos, reader=reader)
        # the pinned load records transformers' missing keys here; the GGUF builder refuses a file that leaves any
        setattr(hf_model, "ie_missing_keys", frozenset())
        self.model_id = gm.gguf_stem_id(self.gguf_path, prov)
        self.weights_card = {**prov, "path": str(self.gguf_path), **built, "tokenizer": tok_rec,
                             "runtime": type(self).__name__}

        self.eager_model = EagerModel(
            self.model_id,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            hf_model=hf_model,
            tokenizer=tokenizer,
        )
        self.hf_model = hf_model
        self.tokenizer = tokenizer

        # Cache layer references and norm weights (every layer the model has)
        self._decoder_layers = self.eager_model.arch.decoder_layers
        self._final_norm_weight = self.eager_model.arch.final_norm.weight.detach()
        self._lm_head_weight = self.eager_model.arch.lm_head.weight.detach()
        for l, block in enumerate(self._decoder_layers):
            if hasattr(block, "pre_feedforward_layernorm"):
                self._pre_ffn_norm_weights[l] = block.pre_feedforward_layernorm.weight.detach()
            if hasattr(block, "post_feedforward_layernorm"):
                self._post_ffn_norm_weights[l] = block.post_feedforward_layernorm.weight.detach()

        if self.load_jpp:
            # Load & validate Pre-Fitted Coordinate Lenses (J++ and optionally JLens), as the pinned load does
            self._load_coordinate_lenses()
            self.eager_model._lens_jacobians["jpp"] = self.jpp_lenses
            if self.jlens_lenses:
                self.eager_model._lens_jacobians["jlens"] = self.jlens_lenses

        on_gpu = self.device.type == "cuda"
        peak_gb = (torch.cuda.max_memory_allocated(self.device) if on_gpu
                   else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) / (1024**3)
        self.allocator_settings = er.read_allocator_settings() if on_gpu else ""
        logger.info(
            "GGUFRuntime ready in %.2fs | %s %.2f GB | model %s | %s | allocator: %s | sliding window: %s",
            time.perf_counter() - t0,
            "Peak VRAM:" if on_gpu else "Peak RAM (process):",
            peak_gb,
            self.model_id,
            built["layers_as"],
            self.allocator_settings or "(default)",
            self.sliding_window,
        )
        if on_gpu and not er.expandable_segments_on(self.allocator_settings):
            logger.warning(
                "The CUDA allocator runs without expandable segments (%s): sweeps near MAX_SWEEP_POSITIONS=%d may "
                "run out of memory, and are then refused in plain words.",
                self.allocator_settings or "the default settings",
                er.MAX_SWEEP_POSITIONS,
            )
        return self

    # ------------------------------------------------- the accessors that read the packing, through gguf_weights

    @torch.no_grad()
    def slice_w_down(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """`W_down[l, idx]` of shape `[K, d_model]` (columns `idx` of `down_proj`), the vault's column edits laid on."""
        down_proj = self.get_mlp_module(layer_idx).down_proj
        cols = dense_cols(down_proj, neuron_indices)
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(down_proj, "_memit_col_overlays", None)
        if overlays:
            cols = cols.clone()
            for k_pos, n_id in enumerate(neuron_indices.cpu().tolist()):
                if int(n_id) in overlays:
                    cols[k_pos] = overlays[int(n_id)].to(device=cols.device, dtype=cols.dtype)
        return cols

    @torch.no_grad()
    def slice_w_gate(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """`W_gate[l, idx]` of shape `[K, d_model]` (rows `idx` of `gate_proj`), its row overlays laid on."""
        gate_proj = self.get_mlp_module(layer_idx).gate_proj
        rows = dense_rows(gate_proj, neuron_indices)
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(gate_proj, "_memit_row_overlays", None)
        if overlays:
            rows = rows.clone()
            for k_pos, n_id in enumerate(neuron_indices.cpu().tolist()):
                if int(n_id) in overlays:
                    rows[k_pos] = overlays[int(n_id)].to(device=rows.device, dtype=rows.dtype)
        return rows

    @torch.no_grad()
    def slice_w_up(self, layer_idx: int, neuron_indices: torch.Tensor) -> torch.Tensor:
        """`W_up[l, idx]` of shape `[K, d_model]` (rows `idx` of `up_proj`; no overlays, as the pinned one)."""
        return dense_rows(self.get_mlp_module(layer_idx).up_proj, neuron_indices)

    @torch.no_grad()
    def get_full_w_down(self, layer_idx: int) -> torch.Tensor:
        """The whole `down_proj` [d_model, d_ff] of layer `layer_idx`, its column overlays laid on."""
        down_proj = self.get_mlp_module(layer_idx).down_proj
        w = dense(down_proj)
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(down_proj, "_memit_col_overlays", None)
        if overlays:
            w = w.clone()
            for col_idx, new_col in overlays.items():
                w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
        return w

    @torch.no_grad()
    def get_full_w_gate(self, layer_idx: int) -> torch.Tensor:
        """The whole `gate_proj` [d_ff, d_model] of layer `layer_idx`, its row overlays laid on."""
        gate_proj = self.get_mlp_module(layer_idx).gate_proj
        w = dense(gate_proj)
        overlays: Optional[Dict[int, torch.Tensor]] = getattr(gate_proj, "_memit_row_overlays", None)
        if overlays:
            w = w.clone()
            for row_idx, new_row in overlays.items():
                w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)
        return w

    @torch.no_grad()
    def get_full_w_up(self, layer_idx: int) -> torch.Tensor:
        """The whole `up_proj` [d_ff, d_model] of layer `layer_idx` (no overlays, as the pinned one)."""
        return dense(self.get_mlp_module(layer_idx).up_proj)

    @torch.no_grad()
    def write_col_norms(self, layer_idx: int, chunk: int = 2048) -> torch.Tensor:
        """`[d_ff]` float32: ||W_down[l][:, i] (.) w_post|| for every neuron i of layer l (write-size ranking), as the
        pinned one: computed once per layer in column chunks and cached; a column under a vault overlay is recomputed
        from the overlay on every call."""
        down_proj = self.get_mlp_module(layer_idx).down_proj
        width = int(getattr(down_proj, "in_features", 0) or er.D_FF)
        w_post = self._post_ffn_norm_weights.get(layer_idx)
        w_post_f = w_post.to(self.device).float() if w_post is not None else None
        base = self._write_norm_cache.get(layer_idx)
        if base is None:
            parts: List[torch.Tensor] = []
            for c0 in range(0, width, chunk):
                idx = torch.arange(c0, min(c0 + chunk, width), device=self.device)
                cols = dense_cols(down_proj, idx).float()
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

