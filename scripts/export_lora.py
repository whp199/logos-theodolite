#!/usr/bin/env python3
"""The pushes a control vector cannot carry, for llama.cpp: a neuron's "mul" (0 silences it), a vault column edit and
one attention head's scale, written as a gemma4 LoRA adapter GGUF that `llama-server --lora FILE` loads and scales
per request; a whole feed-forward block or every head of a layer silenced, written as a copy of the model's GGUF.

    .venv/bin/python -I scripts/export_lora.py --gguf MODEL.gguf --layer 45 --neuron 12345 --mul 0     # silenced
    .venv/bin/python -I scripts/export_lora.py --gguf MODEL.gguf --page 46 --head 7 --scale 0.5        # one head halved
    .venv/bin/python -I scripts/export_lora.py --gguf MODEL.gguf -o mix.gguf \\
        --push '[{"layer": 45, "n": 12345, "mode": "mul", "dose": 2}, {"layer": 44, "head": 3, "dose": 0}]'
    .venv/bin/python -I scripts/export_lora.py --gguf MODEL.gguf --vault "UPDATE LAYER 45 SLOT 12345 SET object = 'Paris'"
    .venv/bin/python -I scripts/export_lora.py --gguf MODEL.gguf --layer 45 --block --model-out MODEL-L45-ffn0.gguf
    .venv/bin/python -I scripts/export_lora.py --show theodolite-L45-n12345-mul0.gguf

    llama-server -m MODEL.gguf --lora theodolite-L45-n12345-mul0.gguf --lora-init-without-apply
    curl localhost:8080/completion -d '{"prompt": "...", "lora": [{"id": 0, "scale": 1}]}'

The pushes, as the instrument makes them (`atlas_ops.resolve` and `installed`; the HTTP API's PushIn, `atlas.py`
`Atlas.parse_pushes`, whose defaults `--push` keeps: mul's dose 2, a head's or a whole block's 0):
- "mul" on neuron n of layer l: its activation a_n times the dose f at the down projection's input. As a LoRA on
  `blk.<l>.ffn_down`: A = e_n^T (one row, a one at n), B = (f - 1) W_down[:, n]; llama.cpp adds SCALE x B (A x) to the
  projection's output, so at SCALE s the factor is 1 + s (f - 1). Rank 1 per neuron.
- a vault edit (DELETE, UPDATE, INSERT: `larql_vault.LarqlVaultEngine.execute_lql`, run here on the instrument's own
  runtime): column n of W_down becomes v. As a LoRA: A = e_n^T, B = v - W_down[:, n]; at SCALE s the column is
  W + s (v - W). Rank 1. A neuron both edited and multiplied writes f v, as the instrument's forward does.
- a head's scale (PushIn `head` = h, dose f): head h's part of the attention output, o_proj's inputs h hd to
  (h + 1) hd (hd the layer's head width), times f. As a LoRA on `blk.<l>.attn_output`: A = the identity
  on those hd inputs, B = (f - 1) W_O[:, h hd:(h + 1) hd]. Rank hd per head (256 on the 31B's sliding layers, 512 on
  its full ones).
- a whole block silenced (PushIn `all: true`, mul 0) or every head of a layer (`head: "all"`, 0): not a LoRA (it would
  be of full rank). `--model-out` writes a copy of `--gguf` that does it (`--silence-by`):
  "scale" (the default): every tensor and key copied, with a one-number F32 tensor `blk.<l>.ffn_down.scale` or
  `blk.<l>.attn_output.scale` added, set to 0. llama.cpp's loader looks for these at every layer
  (src/llama-model.cpp, the "generic pass" after `load_arch_tensors`, lines 1407-1441 at cd26896) and Gemma 4's graph
  multiplies the projection's output by them (src/models/gemma4.cpp lines 240 and 245 for attention, 327 for the
  dense feed-forward block; 275 is the shared block of the mixture-of-experts model). An older llama.cpp, without
  that pass, refuses the file ("wrong number of tensors"). The record goes in the file under `theodolite.*`.
  "norm": the file copied (a reflink where the file system allows: no space taken) and the F32 post-feed-forward or
  post-attention norm (`blk.<l>.post_ffw_norm`, `blk.<l>.post_attention_norm`) zeroed in place, which every
  llama.cpp reads; the norm after the block writes 0 whatever comes in. The record goes beside it, FILE.theodolite.json.
  Either way the block writes nothing into the stream, as the instrument's x 0 does (the norm of 0 is 0).
- "add" is a control vector: scripts/export_control_vector.py writes it.
- "act" (a number added to one neuron's activation) is refused: llama.cpp has no hook for it. A LoRA adds B (A x),
  which grows with the input, never a fixed amount; a control vector adds after the layer's norm, which the
  instrument's act goes through.

What still differs from the instrument, by construction:
1. the start mark: the instrument's pushes and head scales leave `<bos>` (the attention sink) alone; a LoRA or a
   silenced copy acts at every position, the start mark included. A vault edit is a weight, everywhere, in both.
2. the arithmetic: llama.cpp adds B (A x) in float32 to its own product (on the CPU, its quantised weights times
   activations it quantises to 8 bits); the instrument multiplies a bfloat16 activation.
3. the weights: B is made from the columns read (`--gguf FILE`, or by default the W4A16 checkpoint in the local Hugging
   Face cache, read as scripts/export_control_vector.py reads it, `engine_runtime.dequantize_w4a16_cols`). Read from
   the GGUF you serve, a silenced neuron's column is that file's own times 0. Read from the checkpoint, the column left
   on a GGUF of other quantisation is (1 - f) (W_gguf - W_checkpoint)[:, n] away from f W_gguf[:, n]: name the GGUF.

What llama.cpp does with the adapter (read in its source at cd26896, 20 August 2026):
- `llama_adapter_lora_init_impl` (src/llama-adapter.cpp) asks `general.type` = adapter, `general.architecture` = the
  model's (gemma4), `adapter.type` = lora; pairs `<tensor>.lora_a` and `.lora_b` for each model tensor named, refusing a
  name the model lacks or shapes that do not fit (lora_a [n_in, r], lora_b [r, n_out] in ggml's order).
- `llm_graph_context::build_lora_mm` (src/llama-graph.cpp) adds `scale x B (A x)` to every matrix product of that
  tensor, scale = SCALE x alpha / r, or SCALE alone when `adapter.lora.alpha` is 0 or absent. None is written here, so
  SCALE alone multiplies the push.
- llama-server takes SCALE per request (`"lora": [{"id": ID, "scale": S}]`, an adapter left out at 0), without a
  restart; or for every request with `--lora-scaled FILE:S` (`--lora FILE`: 1). `GET /lora-adapters` lists what it
  loaded, with each id.

Watch the log, and the requests (seen through llama-server cd26896 on the stand-in, scripts/check_lora_export.py): a
control vector it cannot use is logged and the server serves the model unpushed (`common_init_from_params`,
common/common.cpp); an adapter it cannot read stops it at start ("failed to apply lora adapter: <why>"), and a good one
logs nothing at the default verbosity. A request naming an adapter id it does not hold, or a scale of 0, is answered
by the unpushed model without a word, and a request naming no adapter gets each at the scale `GET /lora-adapters` shows
(1, even with --lora-init-without-apply): name every adapter in every request. After writing, this prints what to look
for.

Every adapter records what it was made from under `theodolite.*` (the pushes as given, each entry, the weights read,
the push, the scale, the start mark, what of llama.cpp was read, the code's commit), beside llama.cpp's keys. `--show`
prints them. Nothing here uses the GPU or downloads anything; `--vault` loads the instrument's runtime on the CPU
(`runtime_factory.make_runtime`: the whole model, minutes and some 20 GB on the 31B, its J++ lens for UPDATE and
INSERT), the rest reads a few tensors.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("THEODOLITE_DEVICE", "cpu")  # the runtime --vault loads

import numpy as np  # noqa: E402
import torch  # noqa: E402

import export_control_vector as ecv  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402

MODEL_ID = ecv.MODEL_ID
ARCH = ecv.MODEL_HINT  # gemma4: an adapter carries its model's arch
KEY = ecv.KEY
FFN_DOWN = "blk.{}.ffn_down.weight"
ATTN_OUT = "blk.{}.attn_output.weight"
SILENCE_TENSORS = {"ffn": ("blk.{}.ffn_down.scale", "blk.{}.post_ffw_norm.weight"),
                   "attn": ("blk.{}.attn_output.scale", "blk.{}.post_attention_norm.weight")}
FICLONE = 0x40049409  # linux/fs.h: share the source's extents (btrfs, xfs), no copy

LLAMA_CPP_READ = ("llama.cpp cd26896 (20 August 2026): src/llama-adapter.cpp llama_adapter_lora_init_impl, src/llama-graph.cpp "
                  "build_lora_mm (scale x B (A x), alpha absent), src/models/gemma4.cpp build_ffn (ffn_down) and build_attn "
                  "(attn_output), src/llama-model.cpp's generic .scale pass, tools/server per-request lora")
MUL = ("mul: neuron n's activation times f at the down projection's input (atlas_ops.installed); A = e_n^T, B = (f - 1) "
       "W_down[:, n] on blk.<l>.ffn_down; at SCALE s the factor is 1 + s (f - 1)")
VAULT = ("vault edit: column n of W_down becomes v (larql_vault apply_patch_surgery, v rounded to the model's dtype as "
         "its forward lays it); A = e_n^T, B = f v - W_down[:, n] (f the neuron's mul, else 1); at SCALE s the column is "
         "W + s (f v - W)")
HEAD = ("head scale: head h's slice of o_proj's input (columns h hd to (h + 1) hd) times f (atlas_ops.installed); A = "
        "the identity on those inputs, B = (f - 1) W_O[:, h hd:(h + 1) hd] on blk.<l>.attn_output; at SCALE s the factor "
        "is 1 + s (f - 1)")
SCALE = ("llama-server multiplies B (A x) by the adapter's SCALE: per request (\"lora\": [{\"id\": ID, \"scale\": S}], an "
         "adapter left out at 0) or for every request (--lora FILE: 1, --lora-scaled FILE:S); no adapter.lora.alpha is "
         "written, so SCALE alone. SCALE 1 is the push as written, 0 the model as shipped; a neuron's file written at "
         "mul 0 gives the instrument's mul d at SCALE 1 - d")
START_MARK = ("the instrument's pushes and head scales leave the start mark (<bos>, the attention sink) alone; a LoRA "
              "or a silenced copy acts at every position, the start mark included (a vault edit is a weight: everywhere, "
              "in both)")
SILENCE = {"ffn": "every neuron of the feed-forward block silenced (PushIn all: true, mul 0): the block writes nothing "
                  "into the stream (atlas_ops.installed's x 0 at the down projection's input, then the norm of 0)",
           "attn": "every attention head silenced (PushIn head: \"all\", 0): attention writes nothing into the stream "
                   "(atlas_ops.installed's x 0 at o_proj's input, then the norm of 0)"}
SILENCE_BY = {"scale": ("a one-number F32 tensor blk.<l>.ffn_down.scale (or attn_output.scale) added, set to 0: llama.cpp "
                        "multiplies the projection's output by it (src/llama-model.cpp's generic pass, gemma4.cpp's "
                        "build_ffn and build_attn); a llama.cpp without that pass refuses the file"),
              "norm": ("the F32 blk.<l>.post_ffw_norm (or post_attention_norm) weight zeroed in place: the norm after "
                       "the block writes 0 whatever comes in")}
CAVEAT = ("the census read its labels from add pushes, not from these: a neuron's mul, a head's scale and a vault edit "
          "carry the instrument's arithmetic, not a measured effect. Pushes and head scales act on the start mark too")


# --------------------------------------------------------------------------------------------------- what is pushed


@dataclass
class Plan:
    """What one export carries, checked: neurons' factors and vault columns by layer, heads' factors by layer, and
    blocks silenced ((layer, "ffn" | "attn"))."""

    muls: Dict[int, Dict[int, float]] = field(default_factory=dict)
    columns: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    heads: Dict[int, Dict[int, float]] = field(default_factory=dict)
    silenced: List[Tuple[int, str]] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)  # each, in words, in the order given
    dropped: List[str] = field(default_factory=list)  # what a silenced block makes moot

    def lora_empty(self) -> bool:
        return not (self.muls or self.columns or self.heads)


def push_bodies(spec: str) -> List[Dict[str, Any]]:
    """`--push`: one PushIn object or a list of them, as JSON (the HTTP API's `neurons`)."""
    try:
        got = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--push takes JSON, a PushIn object or a list of them ({exc.msg} at {exc.pos}).") from None
    got = got if isinstance(got, list) else [got]
    if not got or not all(isinstance(b, dict) for b in got):
        raise ValueError('--push takes PushIn objects, such as {"layer": 45, "n": 12345, "mode": "mul", "dose": 0}.')
    return got


def pushes_of(bodies: Sequence[Dict[str, Any]]) -> Tuple[List[ao.Push], List[ao.HeadScale]]:
    """PushIn bodies as the HTTP API reads them (`atlas.Atlas.parse_pushes`, its defaults kept), before any check."""
    from pydantic import ValidationError

    from backend.atlas import PushIn, _head_index

    ps: List[ao.Push] = []
    hs: List[ao.HeadScale] = []
    for body in bodies:
        try:
            p = PushIn(**body)
        except ValidationError as exc:
            raise ValueError(f"Not a PushIn: {json.dumps(body)} ({exc.errors()[0]['msg']}).") from None
        if p.head is not None:
            hs.append(ao.HeadScale(int(p.layer), None if p.head == "all" else _head_index(p.head),
                                   float(p.dose if p.dose is not None else 0.0)))
            continue
        if p.all:
            ps.append(ao.Push(int(p.layer), None, "mul", float(p.dose if p.dose is not None else 0.0)))
        else:
            if p.n is None:
                raise ValueError("A push names its neuron (n), the whole layer's block (all: true), or a head (head).")
            mode = p.mode or "add"
            dose = p.dose if p.dose is not None else (2.0 if mode == "mul" else 0.8)
            ps.append(ao.Push(int(p.layer), int(p.n), mode, float(dose)))
    return ps, hs


def plan_of(ps: Sequence[ao.Push], hs: Sequence[ao.HeadScale], vault: Dict[Tuple[int, int], np.ndarray],
            source: Any, vault_said: Sequence[str] = ()) -> Plan:
    """Check every push, head scale and vault column against the weights' shape and refuse, in plain words, what
    llama.cpp cannot be given (`atlas_ops.check_pushes`'s words where they are the same). Several muls of one neuron
    multiply, several scales of one head multiply, as the instrument's hooks apply them in turn."""
    L = int(source.n_layers)
    out = Plan()

    def layer_ok(l: int) -> None:
        if not 0 <= l < L:
            raise ValueError(f"Layer {l} does not exist: layers count from 0 to {L - 1} (the atlas's page is layer + 1).")

    for p in ps:
        l = int(p.layer)
        layer_ok(l)
        if not math.isfinite(float(p.dose)):
            raise ValueError("A push's dose must be a finite number.")
        if p.mode == "add":
            raise ValueError("An add push is a control vector: scripts/export_control_vector.py writes it "
                             f"(--layer {l} --neuron {p.neuron} --dose {p.dose:g}).")
        if p.mode == "act":
            raise ValueError("An act push (a number added to one neuron's activation) cannot be exported: llama.cpp has no "
                             "hook for it. A LoRA adds B (A x), which grows with the input, and a control vector is added "
                             "after the layer's norm, which the activation's push goes through. Use mul (a factor on the "
                             "activation) instead.")
        if p.mode != "mul":
            raise ValueError(f'A push\'s mode is "add", "mul" or "act", not "{p.mode}".')
        if p.neuron is None:
            if float(p.dose) != 0.0:
                raise ValueError("A whole feed-forward block can only be silenced (dose 0): the norm after the block undoes any "
                                 f"other uniform scale, so x{p.dose:g} would change nothing but, below 0, its sign.")
            if (l, "ffn") not in out.silenced:
                out.silenced.append((l, "ffn"))
            out.entries.append(f"layer {l} (page {l + 1}) feed-forward block silenced")
            continue
        width = int(source.width(l))
        if not 0 <= int(p.neuron) < width:
            raise ValueError(f"Neuron {p.neuron} does not exist: layer {l} has {width} (0 to {width - 1}).")
        n = int(p.neuron)
        m = out.muls.setdefault(l, {})
        m[n] = m.get(n, 1.0) * float(p.dose)
        out.entries.append(f"layer {l} (page {l + 1}) neuron {n} mul x{p.dose:g}")
    for h in hs:
        l = int(h.layer)
        layer_ok(l)
        if not math.isfinite(float(h.scale)):
            raise ValueError("A head's scale must be a finite number.")
        if h.head is None:
            if float(h.scale) != 0.0:
                raise ValueError("Every head of a layer can only be silenced together (dose 0): the norm after attention undoes "
                                 f"any other uniform scale, so x{h.scale:g} would change nothing but, below 0, its sign. "
                                 "Scale heads one by one instead.")
            if (l, "attn") not in out.silenced:
                out.silenced.append((l, "attn"))
            out.entries.append(f"layer {l} (page {l + 1}) every attention head silenced")
            continue
        nh = int(source.n_heads(l))
        if not 0 <= int(h.head) < nh:
            raise ValueError(f"Layer {l} has {nh} attention heads (0 to {nh - 1}), not head {h.head}.")
        m = out.heads.setdefault(l, {})
        m[int(h.head)] = m.get(int(h.head), 1.0) * float(h.scale)
        out.entries.append(f"layer {l} (page {l + 1}) head {h.head} x{h.scale:g}")
    for (l, n), v in vault.items():
        layer_ok(l)
        out.columns.setdefault(int(l), {})[int(n)] = np.asarray(v, dtype=np.float32)
    out.entries += [f"vault: {s}" for s in vault_said]
    for l, which in out.silenced:  # 0 times anything: a silenced block's own pushes change nothing
        book = {"ffn": (out.muls, out.columns), "attn": (out.heads,)}[which]
        for d in book:
            if l in d:
                out.dropped.append(f"layer {l}'s {'neurons' if which == 'ffn' else 'heads'} {sorted(d.pop(l))}: its "
                                   f"{'feed-forward block' if which == 'ffn' else 'attention'} is silenced")
    return out


# --------------------------------------------------------------------------------------------------- the weights read


class Columns:
    """The columns a LoRA's B is made from, from the W4A16 checkpoint (`ecv.CheckpointColumns`) or a GGUF
    (`ecv.GGUFColumns`): the down projection's and o_proj's, and the layer's head count and width."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.n_layers = inner.n_layers
        self.is_gguf = isinstance(inner, ecv.GGUFColumns)
        self._heads: Optional[List[int]] = None

    def width(self, layer: int) -> int:
        return int(self.inner.width(layer))

    def record(self) -> str:
        return self.inner.record()

    def down_cols(self, layer: int, neurons: Sequence[int]) -> torch.Tensor:
        """[K, D] float32: W_down[:, n] for each n."""
        return self.inner.columns(layer, neurons)[0]

    def n_heads(self, layer: int) -> int:
        if self._heads is None:
            self._heads = self._read_heads()
        return int(self._heads[int(layer)])

    def _read_heads(self) -> List[int]:
        if self.is_gguf:
            from backend import gguf_model as gm

            v = gm.read_metadata(self.inner.reader, long_arrays=True).get(f"{ARCH}.attention.head_count")
            v = list(v) if isinstance(v, (list, tuple)) else [v] * self.n_layers
        else:
            cfg_path = self.inner.files[0].parent / "config.json"
            if not cfg_path.exists():
                raise ValueError(f"No config.json beside {self.inner.files[0].name}: the head count is read from it (or read a "
                                 "GGUF with --gguf).")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            v = [(cfg.get("text_config") or cfg).get("num_attention_heads")] * self.n_layers
        if not v or any(not isinstance(x, int) or x <= 0 for x in v):
            raise ValueError("The weights do not say how many attention heads each layer has.")
        return [int(x) for x in v]

    def o_width(self, layer: int) -> int:
        """o_proj's inputs: heads x head width."""
        if self.is_gguf:
            from backend import gguf_model as gm

            return int(gm.logical_shape(self.inner._tensor(layer, "self_attn.o_proj.weight"))[1])
        from safetensors import safe_open

        P = self.inner.PREFIX.format(int(layer))
        for name, mult in (("self_attn.o_proj.weight_packed", 8), ("self_attn.o_proj.weight", 1)):
            if P + name in self.inner.where:
                with safe_open(str(self.inner.where[P + name]), "pt") as f:
                    return int(f.get_slice(P + name).get_shape()[1]) * mult
        raise ValueError(f"The checkpoint holds no o_proj for layer {layer}.")

    def o_cols(self, layer: int, idx: Sequence[int]) -> torch.Tensor:
        """[K, D] float32: W_O[:, i] for each i."""
        ix = torch.tensor([int(i) for i in idx], dtype=torch.long)
        if self.is_gguf:
            from backend import gguf_model as gm
            from backend.gguf_weights import GGUFLinear

            t = self.inner._tensor(layer, "self_attn.o_proj.weight")
            D, N = gm.logical_shape(t)
            lin = GGUFLinear(gm._blocks(t, torch.device("cpu")), t.tensor_type, N, D, dtype=torch.float32, gguf_name=t.name)
            return lin.unpacked_cols(ix).float()
        from backend.engine_runtime import dequantize_w4a16_cols

        P = self.inner.PREFIX.format(int(layer))
        packed = self.inner._get(P + "self_attn.o_proj.weight_packed")
        if packed is not None:
            return dequantize_w4a16_cols(packed, self.inner._get(P + "self_attn.o_proj.weight_scale"), ix).float()
        dense = self.inner._get(P + "self_attn.o_proj.weight")
        if dense is None:
            raise ValueError(f"The checkpoint holds no o_proj for layer {layer}.")
        return dense[:, ix].T.float()


@contextmanager
def vault_kept(vault: Any) -> Iterator[None]:
    """The vault as it was before the block, after it: its records, its count and every layer's column overlays (not
    `rollback_patches`, which clears every edit, and walks the 31B's 60 layers)."""
    from backend.causal_check import language_model

    rt = vault.runtime
    n = len(language_model(rt.hf_model).layers)
    snap = [dict(d) if (d := getattr(rt.get_mlp_module(l).down_proj, "_memit_col_overlays", None)) is not None else None
            for l in range(n)]
    patches, counter = list(vault.active_patches), vault._patch_counter
    try:
        yield
    finally:
        for l, d in enumerate(snap):
            down = rt.get_mlp_module(l).down_proj
            if d is None:
                if hasattr(down, "_memit_col_overlays"):
                    down._memit_col_overlays = {}
            else:
                down._memit_col_overlays.clear()
                down._memit_col_overlays.update(d)
        vault.active_patches[:] = patches
        vault._patch_counter = counter


def vault_columns(vault: Any, statements: Sequence[str]) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[str]]:
    """({(layer, neuron): v float32 [D]}, each statement's edit in words): the vault's own edits
    (`LarqlVaultEngine.execute_lql`, in order, after any already in force; a later edit of a neuron replaces the
    earlier, as `edits_in_force` has it), each column rounded to the model's dtype as the forward lays it on. The vault
    is left as it was (`vault_kept`)."""
    said: List[str] = []
    with vault_kept(vault):
        for s in statements:
            head = s.strip().split(None, 1)[0].upper() if s.strip() else ""
            if head not in ("DELETE", "UPDATE", "INSERT"):
                raise ValueError(f"--vault takes an edit (DELETE, UPDATE or INSERT), not {s!r}.")
            rep = vault.execute_lql(s)
            pr = rep["patch"]
            said.append(f"{pr['statement']} -> layer {pr['layer_idx']} neuron {pr['neuron_idx']} ({pr['op_type']}, column "
                        f"length {pr['orig_norm']:g} -> {pr['new_norm']:g})")
        rt = vault.runtime
        dt = getattr(rt.hf_model, "dtype", None)
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for pr in vault.edits_in_force():
            v = rt.get_mlp_module(pr.layer_idx).down_proj._memit_col_overlays[pr.neuron_idx]
            v = v.to(dt) if isinstance(dt, torch.dtype) else v
            out[(int(pr.layer_idx), int(pr.neuron_idx))] = v.detach().float().cpu().numpy()
        return out, said


def lora_tensors(plan: Plan, source: Columns) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{model tensor: (A [r, n_in], B [n_out, r]) float32}, numpy's order (ggml's reversed: lora_a [n_in, r], lora_b
    [r, n_out]). One pair per tensor: llama.cpp keys its pairs by the model tensor's name."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for l in sorted(set(plan.muls) | set(plan.columns)):
        muls, cols = plan.muls.get(l, {}), plan.columns.get(l, {})
        ns = sorted(set(muls) | set(cols))
        W = source.down_cols(l, ns)  # [K, D]
        bs, picked = [], []
        for k, n in enumerate(ns):
            f = float(muls.get(n, 1.0))
            if n in cols:
                b = torch.from_numpy(cols[n]).float() * f - W[k]
            else:
                b = W[k] * (f - 1.0)
            if bool(b.abs().max() > 0):
                bs.append(b)
                picked.append(n)
        if not picked:
            continue
        A = np.zeros((len(picked), source.width(l)), np.float32)
        A[np.arange(len(picked)), picked] = 1.0
        out[FFN_DOWN.format(l)] = (A, np.ascontiguousarray(torch.stack(bs, dim=1).numpy(), dtype=np.float32))
    for l in sorted(plan.heads):
        nh, width = source.n_heads(l), source.o_width(l)
        if width % nh:
            raise ValueError(f"Layer {l}'s o_proj takes {width} inputs, not a whole number of its {nh} heads.")
        hd = width // nh
        hs = [h for h in sorted(plan.heads[l]) if float(plan.heads[l][h]) != 1.0]
        if not hs:
            continue
        A = np.zeros((hd * len(hs), width), np.float32)
        bs = []
        for k, h in enumerate(hs):
            A[np.arange(k * hd, (k + 1) * hd), np.arange(h * hd, (h + 1) * hd)] = 1.0
            bs.append(source.o_cols(l, range(h * hd, (h + 1) * hd)) * (float(plan.heads[l][h]) - 1.0))  # [hd, D]
        out[ATTN_OUT.format(l)] = (A, np.ascontiguousarray(torch.cat(bs, dim=0).T.numpy(), dtype=np.float32))
    return out


# --------------------------------------------------------------------------------------------------------- the files


def base_record(plan: Plan, source: Columns, given: Dict[str, Any], kinds: Sequence[str]) -> Dict[str, Any]:
    return {
        "model": MODEL_ID,
        "weights": source.record(),
        "entries": list(plan.entries),
        "given": json.dumps(given, sort_keys=True),
        "dropped": list(plan.dropped) or "nothing",
        "push": " | ".join(kinds),
        "start_mark": START_MARK,
        "llama_cpp": LLAMA_CPP_READ,
        "caveat": CAVEAT,
        "made_by": f"scripts/export_lora.py at {ecv.code_commit()}",
        "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_lora(path: Path, tensors: Dict[str, Tuple[np.ndarray, np.ndarray]], record: Dict[str, Any]) -> Path:
    """The adapter: llama.cpp's keys, `theodolite.*` from `record`, a lora_a and lora_b per model tensor, F32.
    Atomically: a temporary file beside it, then a rename."""
    import gguf

    if not tensors:
        raise ValueError("Nothing to write: every push asked for is the model as it is (a factor of 1).")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        w = gguf.GGUFWriter(str(tmp), ARCH)
        w.add_type(gguf.GGUFType.ADAPTER)
        w.add_string(gguf.Keys.Adapter.TYPE, "lora")
        w.add_name(str(record.get("name", "theodolite LoRA")))
        if record.get("description"):
            w.add_description(str(record["description"]))
        ecv.add_record(w, record)
        for name in sorted(tensors):
            A, B = tensors[name]
            w.add_tensor(f"{name}.lora_a", np.ascontiguousarray(A, dtype=np.float32))
            w.add_tensor(f"{name}.lora_b", np.ascontiguousarray(B, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_lora(path: Path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """({model tensor: (A, B)}, every key's value) from an adapter GGUF, paired as llama.cpp pairs them."""
    from backend import gguf_model as gm

    reader = gm.open_gguf(path)
    meta = gm.read_metadata(reader, long_arrays=True)
    half: Dict[str, Dict[str, np.ndarray]] = {}
    for t in reader.tensors:
        for suffix in (".lora_a", ".lora_b"):
            if t.name.endswith(suffix):
                shape = [int(x) for x in reversed(list(t.shape))]
                half.setdefault(t.name[: -len(suffix)], {})[suffix] = np.array(t.data, dtype=np.float32).reshape(shape)
                break
        else:
            raise ValueError(f"{Path(path).name}: llama.cpp reads only <tensor>.lora_a and .lora_b, not {t.name!r}.")
    out = {}
    for name, ab in half.items():
        if set(ab) != {".lora_a", ".lora_b"}:
            raise ValueError(f"{Path(path).name}: {name} lacks its {'lora_b' if '.lora_a' in ab else 'lora_a'}.")
        out[name] = (ab[".lora_a"], ab[".lora_b"])
    return out, meta


def _clone_or_copy(src: Path, dst: Path) -> str:
    """Copy `src` to `dst`: a reflink (FICLONE) where the file system allows, else bytes. Says which."""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        try:
            fcntl.ioctl(fo.fileno(), FICLONE, fi.fileno())
            return "reflink"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "copy"


def write_silenced(src: Path, out: Path, silenced: Sequence[Tuple[int, str]], by: str,
                   record: Dict[str, Any]) -> Tuple[Path, str]:
    """A copy of the GGUF `src` with each (layer, "ffn" | "attn") silenced, `by` "scale" or "norm" (module docstring);
    returns (the file, how). Atomically, as the adapter is written."""
    import gguf

    from backend import gguf_model as gm

    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("--model-out must be a new file, not the model itself.")
    reader = gm.open_gguf(src)
    meta = gm.read_metadata(reader)
    if meta.get("general.architecture") != ARCH:
        raise ValueError(f"{src.name} is a {meta.get('general.architecture')!r} model, not Gemma 4 ({ARCH}).")
    if meta.get("split.count") not in (None, 0, 1):
        raise ValueError(f"{src.name} is one part of a split GGUF: join it first (llama-gguf-split --merge).")
    names = {t.name: t for t in reader.tensors}
    for l, which in silenced:
        if f"blk.{l}.ffn_gate_inp.weight" in names:
            raise ValueError(f"Layer {l} of {src.name} has experts (Gemma 4 26B-A4B): this silences only the dense 31B's blocks.")
        scale, norm = (s.format(l) for s in SILENCE_TENSORS[which])
        if by == "scale" and scale in names:
            raise ValueError(f"{src.name} already holds {scale}: silence it with --silence-by norm instead.")
        t = names.get(norm)
        if t is None:
            raise ValueError(f"{src.name} holds no {norm}.")
        if by == "norm" and t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                                                  gguf.GGMLQuantizationType.BF16):
            raise ValueError(f"{norm} is {t.tensor_type.name}, not a float type whose zero bytes are 0.")
    tmp = out.with_name(out.name + ".tmp")
    try:
        if by == "norm":
            how = _clone_or_copy(src, tmp)
            with open(tmp, "r+b") as f:
                for l, which in silenced:
                    t = names[SILENCE_TENSORS[which][1].format(l)]
                    f.seek(int(t.data_offset))
                    f.write(b"\0" * int(t.n_bytes))
            side = out.with_name(out.name + ".theodolite.json")
            side.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        else:
            how = "rewritten"
            w = gguf.GGUFWriter(str(tmp), ARCH, endianess=reader.endianess)
            w.data_alignment = reader.alignment
            for fld in reader.fields.values():
                if fld.name == gguf.Keys.General.ARCHITECTURE or fld.name.startswith("GGUF.") or fld.name.startswith(KEY):
                    continue
                vt = fld.types[0]
                sub = fld.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
                w.add_key_value(fld.name, fld.contents(), vt, sub_type=sub)
            ecv.add_record(w, record)
            zero = np.zeros(1, np.float32)
            added = {SILENCE_TENSORS[which][0].format(l) for l, which in silenced}
            for t in reader.tensors:
                w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            for name in sorted(added):
                w.add_tensor_info(name, zero.shape, zero.dtype, zero.nbytes, gguf.GGMLQuantizationType.F32)
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_ti_data_to_file()
            for t in reader.tensors:
                w.write_tensor_data(t.data, tensor_endianess=reader.endianess)
            for _name in sorted(added):
                w.write_tensor_data(zero)
            w.close()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out, how


def show(path: Path) -> None:
    from backend import gguf_model as gm

    reader = gm.open_gguf(path)
    meta = gm.read_metadata(reader, long_arrays=True)
    side = Path(str(path) + ".theodolite.json")
    is_adapter = meta.get("general.type") == "adapter"
    for k, v in meta.items():
        if k.startswith(KEY) or (is_adapter and k.startswith(("general.", "adapter."))):
            print(f"{k}: {v}")
    if is_adapter:
        tensors, _ = read_lora(path)
        for name in sorted(tensors):
            A, B = tensors[name]
            print(f"{name}: rank {A.shape[0]}, lora_a {list(A.shape[::-1])}, lora_b {list(B.shape[::-1])} (ggml's order), "
                  f"|B| {float(np.linalg.norm(B.astype(np.float64))):.4f}")
        return
    if side.exists():
        print(side.read_text(encoding="utf-8").strip())
    for t in reader.tensors:
        if t.name.endswith((".ffn_down.scale", ".attn_output.scale", "post_ffw_norm.weight", "post_attention_norm.weight")):
            if not np.asarray(t.data).any():
                print(f"{t.name}: zero")


def default_name(plan: Plan) -> str:
    if len(plan.entries) == 1 and plan.muls:
        (l, m), = plan.muls.items()
        (n, f), = m.items()
        return f"theodolite-L{l}-n{n}-mul{f:g}.gguf"
    if len(plan.entries) == 1 and plan.heads:
        (l, m), = plan.heads.items()
        (h, f), = m.items()
        return f"theodolite-L{l}-h{h}-x{f:g}.gguf"
    if len(plan.columns) == 1 and len(next(iter(plan.columns.values()))) == 1 and not plan.muls and not plan.heads:
        (l, m), = plan.columns.items()
        return f"theodolite-L{l}-n{next(iter(m))}-vault.gguf"
    raise ValueError("Several pushes' adapter needs a name: -o FILE.")


def log_hints(lora: Optional[Path], n_pairs: int, model_out: Optional[Path], by: str) -> List[str]:
    """What to run and what to look for in llama-server's log: it serves whatever it could load."""
    out = []
    if lora is not None:
        out += [f"load it with: llama-server -m <the GGUF the columns came from> --lora {lora} --lora-init-without-apply",
                "  (several adapters: --lora A.gguf,B.gguf, ids 0, 1, ... in that order)",
                '  and push per request: "lora": [{"id": 0, "scale": 1}] (SCALE s: the factor 1 + s (f - 1)); name every '
                "adapter in every request, 0 for those left off: a request that names none gets each at the scale GET "
                "/lora-adapters shows, 1 even with --lora-init-without-apply. --lora-scaled FILE:S pushes every request",
                "  llama-server answers without a word, unpushed, a request naming an id it does not hold or a scale of 0: "
                "GET /lora-adapters lists the adapters it holds, with their ids and paths",
                "  in its log: a good adapter says nothing at the default verbosity (-lv 4 shows \"llama_adapter_lora_init_impl: "
                f"loaded {2 * n_pairs} tensors from lora file\"); one it cannot read says \"failed to apply lora adapter: "
                "<why>\" and the server stops (\"exiting due to model loading error\")"]
    if model_out is not None:
        out += [f"serve the silenced copy with: llama-server -m {model_out}"]
        out += (["  in its log, look for \"wrong number of tensors\": a llama.cpp without the generic .scale pass refuses "
                 "the file (cd26896 reads it)"] if by == "scale" else
                ["  the copy loads as the model itself; --show reads its zeroed norms back"])
    out += ["  a control vector beside it that the server cannot use is logged (\"control vector n_embd does not match "
            "model\", \"failed to load control vector file\", \"no valid control vector files passed\") and the server "
            "serves the model without it: read the log"]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--push", action="append", default=[], metavar="JSON",
                    help="a PushIn object or a list of them, as the HTTP API takes them (repeatable)")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--layer", type=int, help="the layer of --neuron, --head or --block, counted from 0")
    which.add_argument("--page", type=int, help="the same as the atlas's page (its address #/atlas/PAGE/...): layer + 1")
    ap.add_argument("--neuron", type=int, help="a neuron, with --mul F")
    ap.add_argument("--mul", type=float, help="the neuron's factor (0 silences it; the HTTP API's default is 2)")
    ap.add_argument("--head", help='a head (a number from 0, or "all"), with --scale F')
    ap.add_argument("--scale", type=float, help="the head's factor (0 silences it)")
    ap.add_argument("--block", action="store_true", help="the layer's whole feed-forward block, silenced (--model-out)")
    ap.add_argument("--vault", action="append", default=[], metavar="STATEMENT",
                    help="a vault edit (DELETE / UPDATE / INSERT), run on the instrument's runtime on the CPU (repeatable)")
    ap.add_argument("-o", "--out", help="the adapter to write (default theodolite-L<l>-n<n>-mul<f>.gguf and the like, here)")
    ap.add_argument("--model-out", help="the silenced copy of --gguf to write, for a whole block or every head")
    ap.add_argument("--silence-by", choices=("scale", "norm"), default="scale",
                    help="how a silenced copy is made: an added .scale tensor set to 0 (default), or the norm zeroed")
    ap.add_argument("--force", action="store_true", help="write over a file already there")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", help=f"the checkpoint's safetensors file or folder (default {MODEL_ID} in the local cache)")
    src.add_argument("--gguf", help="read the columns from the GGUF you will serve (exact for it; needed for --model-out)")
    ap.add_argument("--show", metavar="FILE", help="print what an adapter or a silenced copy was made from, and nothing else")
    a = ap.parse_args(argv)

    if a.show:
        show(Path(a.show))
        return 0
    try:
        bodies: List[Dict[str, Any]] = []
        for spec in a.push:
            bodies += push_bodies(spec)
        if a.block or any(x is not None for x in (a.neuron, a.mul, a.head, a.scale)):
            if a.layer is None and a.page is None:
                raise ValueError("--neuron, --head and --block need --layer L (or --page P).")
            layer = int(a.layer) if a.layer is not None else int(a.page) - 1
            if sum(x is not None for x in (a.neuron, a.head)) + bool(a.block) != 1:
                raise ValueError("Name one thing to push: --neuron N --mul F, --head H --scale F, or --block.")
            if a.neuron is not None:
                if a.mul is None or a.scale is not None:
                    raise ValueError("A neuron takes --mul F (its factor; 0 silences it).")
                bodies.append({"layer": layer, "n": a.neuron, "mode": "mul", "dose": a.mul})
            elif a.head is not None:
                if a.scale is None or a.mul is not None:
                    raise ValueError("A head takes --scale F (its factor; 0 silences it).")
                bodies.append({"layer": layer, "head": a.head if a.head == "all" else _int(a.head), "dose": a.scale})
            else:
                if a.mul is not None or a.scale is not None:
                    raise ValueError("--block silences the block: it takes no factor.")
                bodies.append({"layer": layer, "all": True, "mode": "mul", "dose": 0})
        elif a.layer is not None or a.page is not None:
            raise ValueError("--layer and --page go with --neuron, --head or --block.")
        if not bodies and not a.vault:
            raise ValueError("Name something to push: --neuron/--head/--block with --layer, --push JSON, or --vault STATEMENT.")
        if a.vault and a.checkpoint:
            raise ValueError("--vault runs on the instrument's runtime (the cached checkpoint, or --gguf): it does not read "
                             "--checkpoint's files.")
        source = Columns(ecv.GGUFColumns(Path(a.gguf)) if a.gguf else ecv.CheckpointColumns(ecv.checkpoint_files(a.checkpoint)))
        ps, hs = pushes_of(bodies)
        vault: Dict[Tuple[int, int], np.ndarray] = {}
        said: List[str] = []
        if a.vault:
            plan_of(ps, hs, {}, source)  # refuse what cannot be given before the model is loaded
            from backend.larql_vault import LarqlVaultEngine
            from backend.runtime_factory import GGUF_ENV, make_runtime

            if a.gguf:
                os.environ[GGUF_ENV] = str(Path(a.gguf).resolve())
            print("loading the instrument's runtime on the CPU for the vault's edits ...", flush=True)
            vault, said = vault_columns(LarqlVaultEngine(make_runtime(load_jlens=False).load()), a.vault)
        plan = plan_of(ps, hs, vault, source, said)
        if plan.silenced and not a.model_out:
            raise ValueError("A whole block or every head silenced is written as a copy of the model: name the GGUF you serve "
                             "with --gguf and the copy with --model-out FILE.")
        if plan.silenced and not a.gguf:
            raise ValueError("A silenced copy is made from the GGUF you serve: name it with --gguf.")
        if a.model_out and not plan.silenced:
            raise ValueError("--model-out is for a whole block or every head silenced (--block, all: true, head: \"all\").")
        kinds = [k for k, used in ((MUL, bool(plan.muls)), (VAULT, bool(plan.columns)), (HEAD, bool(plan.heads))) if used]
        out = None
        if not plan.lora_empty():
            out = Path(a.out) if a.out else Path(default_name(plan))
        elif a.out:
            raise ValueError("Nothing here is a LoRA (-o): a silenced block is the copy --model-out writes.")
        for p in (out, Path(a.model_out) if a.model_out else None):
            if p is not None and p.exists() and not a.force:
                raise ValueError(f"{p} is already there (--force writes over it).")
        given = {"pushes": bodies, "vault": list(a.vault)}
        tensors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if out is not None:
            tensors = lora_tensors(plan, source)
            rec = {"name": f"theodolite LoRA: {'; '.join(plan.entries)}"[:200],
                   "description": "Pushes from Theodolite that a control vector cannot carry: llama-server --lora FILE, its "
                                  "scale per request (1: the push as written).",
                   **base_record(plan, source, given, kinds),
                   "tensors": sorted(tensors), "ranks": [int(tensors[k][0].shape[0]) for k in sorted(tensors)],
                   "scale": SCALE}
            write_lora(out, tensors, rec)
        how = ""
        if plan.silenced:
            rec = {**base_record(plan, source, given, [SILENCE[w] for w in sorted({w for _l, w in plan.silenced})]),
                   "silenced": [f"layer {l} {w}" for l, w in plan.silenced], "silence_by": SILENCE_BY[a.silence_by]}
            _p, how = write_silenced(Path(a.gguf), Path(a.model_out), plan.silenced, a.silence_by, rec)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for e in plan.entries:
        print(f"  {e}")
    for d in plan.dropped:
        print(f"  left out, moot: {d}")
    if out is not None:
        print(f"wrote {out}: " + ", ".join(f"{k} rank {tensors[k][0].shape[0]}" for k in sorted(tensors)))
    if a.model_out:
        print(f"wrote {a.model_out}: {', '.join(f'layer {l} {w}' for l, w in plan.silenced)} silenced ({a.silence_by}, {how})")
    print(f"  weights: {source.record()}")
    if not a.gguf:
        print("  the columns are the checkpoint's: on a GGUF of other quantisation the push is near, not exact (name the "
              "GGUF you serve with --gguf)")
    for line in log_hints(out, len(tensors), Path(a.model_out) if a.model_out else None, a.silence_by):
        print(line)
    return 0


def _int(s: str) -> int:
    if not str(s).isdigit():
        raise ValueError(f'A head is a number counting from 0, or "all", not {s!r}.')
    return int(s)


if __name__ == "__main__":
    sys.exit(main())
