#!/usr/bin/env python3
"""A neuron's push as a llama.cpp control vector: the census's and the atlas's "add" push of one Gemma-4-31B-it MLP
neuron, or of a set of them, written as a GGUF that `llama-server --control-vector-scaled FILE:SCALE` loads.

    .venv/bin/python -I scripts/export_control_vector.py --layer 45 --neuron 12345            # dose +1
    .venv/bin/python -I scripts/export_control_vector.py --page 46 --neuron 12345 --dose -2   # the atlas's #/atlas/46/12345
    .venv/bin/python -I scripts/export_control_vector.py --set 45:12345,44:801:-0.5 -o mix.gguf
    .venv/bin/python -I scripts/export_control_vector.py --layer 45 --neuron 12345 --gguf gemma-4-31B-it-QAT-Q4_0.gguf
    .venv/bin/python -I scripts/export_control_vector.py --show theodolite-L45-n12345.gguf

    llama-server -m gemma-4-31B-it-Q4_K_M.gguf --control-vector-scaled theodolite-L45-n12345.gguf:1

The push. The instrument adds `dose x p50_l x u` to the stream leaving layer l, u = unit(w_post (.) W_down[:, n]) the
neuron's write direction as the stream receives it (`atlas_ops.Weights.write_dirs`) and p50_l the median length of
that stream (the stats index's `resid_q[:, 1]`), after the layer's post-feed-forward norm and its layer scalar, at
every word-piece but the start mark (`census.take_census`, `atlas_ops.resolve`). Layers count from 0, as every script
counts them; the atlas's page, and the first number in its address, is layer + 1 (`--page`).

What llama.cpp does with the file (read in its source at cd26896, 20 August 2026; the build beside the owner's models):
- `common_control_vector_load_one` (common/common.cpp) reads every tensor named `direction.<k>`, F32 and one-dimensional
  of the model's width, k >= 1 (direction.0 is refused: no vector is ever added after layer 0), multiplies it by the
  file's SCALE (1 for `--control-vector FILE`) and sums the files given; it reads no metadata.
- `llama_adapter_cvec::apply_to` (src/llama-adapter.cpp) adds the sum for k to the output of the graph's layer il = k,
  for start <= k <= end of `--control-vector-layer-range` (default 1 to the model's layer count).
- In Gemma 4's graph (src/models/gemma4.cpp) that is `build_cvec(cur, il)` right after the layer's
  `layer_output_scale` (the layer scalar), which comes after `ffn_post_norm` and the residual sum: the stream leaving
  layer il. So direction.<l> is the instrument's layer l, with no shift, and its vector is added in the same place.
- It is added at every position of every batch, the prompt's and each word-piece written, the start mark included.

So the file holds `dose x p50_l x u` in float32 for each entry, summed over a layer's entries as the atlas sums
several pushes at one layer, under direction.<l>; SCALE multiplies every dose. Written at dose 1 (the default), SCALE
is the census's dose: the break_rel design's labels were read at +-1 and +-2.

What still differs from the instrument's push, by construction:
1. the start mark: the instrument leaves `<bos>` (the attention sink) unpushed; llama.cpp pushes it too;
2. the sums: the instrument adds the push, cast to bfloat16, to a bfloat16 stream; llama.cpp adds float32 to float32;
3. the weights: a GGUF's quantisation is not the W4A16 checkpoint's, so the stream the push lands in, and its typical
   length, are not quite the instrument's. The direction and p50 are the checkpoint's unless `--gguf` says otherwise.

Where the direction comes from:
- by default the W4A16 checkpoint in the local Hugging Face cache (`census.cached_checkpoint`; `--checkpoint` names
  other files or a folder), read on the CPU, a few tensors of each layer named: the down projection's codes and scales
  and the post-feed-forward norm. The column is unpacked by the instrument's own unpacker
  (`engine_runtime.dequantize_w4a16_cols`, as `census.checkpoint_ceilings` reads it), so the vector is the census's push
  element for element (scripts/check_control_vector.py). The vault's edits are not in it: it is the model as shipped.
- `--gguf FILE`: the same two tensors from a GGUF of Gemma 4 (`blk.<l>.ffn_down`, `blk.<l>.post_ffw_norm`, by
  gguf-py's name map), unpacked to float32 by the instrument's GGUF unpackers (gguf-py's numbers, bit for bit). That
  direction is the file's own; p50 is still the stats index's.

p50 comes from the stats index, `<atlas folder>/index/act_scale.npz` (or `--stats`, such as the census kit's release
copy), refused when it was built for another model than `--model`. Every file records what it was made from under
`theodolite.*` (the model, the weights, each entry's layer, neuron, dose and p50, the stats index's sha256, the push,
the code's commit, and what a census label can bear), beside the keys llama.cpp's own cvector-generator writes
(`general.architecture` = controlvector, `controlvector.model_hint`, `controlvector.layer_count`). `--show` prints
them. Nothing here uses the GPU or downloads anything.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"  # engine_runtime.DEFAULT_W4A16_MODEL_ID
ARCH = "controlvector"  # general.architecture, as llama.cpp's cvector-generator writes it
MODEL_HINT = "gemma4"  # llama.cpp's arch name for Gemma 4
KEY = "theodolite."
LLAMA_CPP_READ = ("llama.cpp cd26896 (20 August 2026): common/common.cpp common_control_vector_load_one, "
                  "src/llama-adapter.cpp llama_adapter_cvec, src/models/gemma4.cpp build_cvec after layer_output_scale")
PUSH = ("add: dose x p50_l x unit(w_post (.) W_down[:, n]) for each entry, summed over a layer's entries, as float32 "
        "under direction.<l>, l the layer counted from 0 (the atlas's page l + 1); llama.cpp adds direction.<l> x SCALE "
        "to the stream leaving layer l (after ffn_post_norm and layer_output_scale), where the instrument adds its push")
START_MARK = ("the instrument leaves the start mark (<bos>, the attention sink) unpushed; llama.cpp adds a control "
              "vector at every position, the start mark included")
SCALE = ("llama-server --control-vector-scaled FILE:SCALE multiplies every dose by SCALE (--control-vector FILE: SCALE "
         "1); a file written at dose 1 takes the census's dose as its SCALE")
CAVEAT = ("a census label was read from six questions (break_rel) at +-1 and +-2 of the stream, one neuron pushed at a "
          "time, from greedy answers of 48 word-pieces with thinking off, on the W4A16 checkpoint, the start mark "
          "unpushed, and set against random directions pushed alike; it is a lead, not a finding (its p and q say how "
          "strong a lead). A dose of 1 is about 1.8 to 2.5 times the most one neuron at layers 40-50 can write by itself "
          "(its write ceiling). Other doses, sums of neurons, other prompts, sampling, other weights and a pushed start "
          "mark were not measured")


@dataclass(frozen=True)
class Entry:
    """One neuron's push: its layer (from 0), its index in that layer, and its dose (a share of the stream's p50)."""

    layer: int
    neuron: int
    dose: float


def parse_set(spec: str) -> List[Entry]:
    """'L:N[:DOSE],L:N[:DOSE],...' (layers from 0; a dose left out is 1) -> entries, in order."""
    out: List[Entry] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        bits = part.split(":")
        if len(bits) not in (2, 3):
            raise ValueError(f"A set's entry is LAYER:NEURON or LAYER:NEURON:DOSE, not {part!r}.")
        try:
            layer, neuron = int(bits[0]), int(bits[1])
            dose = float(bits[2]) if len(bits) == 3 else 1.0
        except ValueError:
            raise ValueError(f"A set's entry is LAYER:NEURON or LAYER:NEURON:DOSE in numbers, not {part!r}.") from None
        out.append(Entry(layer, neuron, dose))
    if not out:
        raise ValueError("The set names no neuron.")
    return out


def check_entries(entries: Sequence[Entry], n_layers: int, width_of: Any) -> None:
    """Refuse, in plain words, what cannot be written or would mean nothing (`width_of(layer)`: the layer's neurons)."""
    for e in entries:
        if e.layer == 0:
            raise ValueError("llama.cpp adds no control vector after layer 0 (its loader refuses direction.0), so a push "
                             "at layer 0 cannot be exported.")
        if not 0 < e.layer < n_layers:
            raise ValueError(f"Layer {e.layer} does not exist: layers count from 0 to {n_layers - 1} (the atlas's page is "
                             "layer + 1).")
        if not math.isfinite(e.dose):
            raise ValueError("A dose must be a finite number.")
    for layer in sorted({e.layer for e in entries}):
        width = int(width_of(layer))
        bad = [e.neuron for e in entries if e.layer == layer and not 0 <= e.neuron < width]
        if bad:
            raise ValueError(f"Neurons count from 0 to {width - 1}; {bad[0]} at layer {layer} does not exist.")


# ------------------------------------------------------------------------------------------------- the weights read


def write_dirs(cols: torch.Tensor, w_post: torch.Tensor) -> torch.Tensor:
    """[K, D] float32 unit write directions from down columns [K, D] and the post-feed-forward norm's weight [D]:
    `atlas_ops.Weights.write_dirs`'s own arithmetic, in its order, so the same numbers in give the same bits out."""
    v = cols.float() * w_post.float()
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)


class CheckpointColumns:
    """Down columns and post-feed-forward norms read on the CPU from a checkpoint's safetensors files (W4A16 packed, or
    a plain weight), one layer's tensors at a time, as `census.checkpoint_ceilings` reads them."""

    PREFIX = "model.language_model.layers.{}."

    def __init__(self, files: Sequence[Path]) -> None:
        from safetensors import safe_open

        if not files:
            raise ValueError(f"No checkpoint files: {MODEL_ID} is not all in the local Hugging Face cache (nothing is "
                             "downloaded here; name the files or their folder with --checkpoint, or read a GGUF with --gguf).")
        self.files = [Path(f) for f in files]
        self.where: Dict[str, Path] = {}
        for path in self.files:
            with safe_open(str(path), "pt") as f:
                for k in f.keys():
                    self.where.setdefault(k, path)
        found = {int(m.group(1)) for k in self.where if (m := re.match(r"model\.language_model\.layers\.(\d+)\.", k))}
        if not found:
            raise ValueError(f"{self.files[0].name} holds no layers of a Gemma 4 language model.")
        self.n_layers = max(found) + 1

    def _get(self, name: str) -> Optional[torch.Tensor]:
        from safetensors import safe_open

        path = self.where.get(name)
        if path is None:
            return None
        with safe_open(str(path), "pt") as f:
            return f.get_tensor(name)

    def width(self, layer: int) -> int:
        P = self.PREFIX.format(int(layer))
        if P + "mlp.down_proj.weight_packed" in self.where:
            from safetensors import safe_open

            with safe_open(str(self.where[P + "mlp.down_proj.weight_packed"]), "pt") as f:
                return int(f.get_slice(P + "mlp.down_proj.weight_packed").get_shape()[1]) * 8
        if P + "mlp.down_proj.weight" in self.where:
            from safetensors import safe_open

            with safe_open(str(self.where[P + "mlp.down_proj.weight"]), "pt") as f:
                return int(f.get_slice(P + "mlp.down_proj.weight").get_shape()[1])
        raise ValueError(f"The checkpoint holds no down projection for layer {layer}.")

    def columns(self, layer: int, neurons: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(W_down[:, neurons] as [K, D] float32, the post-feed-forward norm's weight [D] float32)."""
        from backend.engine_runtime import dequantize_w4a16_cols

        P = self.PREFIX.format(int(layer))
        w_post = self._get(P + "post_feedforward_layernorm.weight")
        if w_post is None:
            raise ValueError(f"The checkpoint holds no post-feed-forward norm for layer {layer}.")
        idx = torch.tensor([int(n) for n in neurons], dtype=torch.long)
        packed = self._get(P + "mlp.down_proj.weight_packed")
        if packed is not None:
            cols = dequantize_w4a16_cols(packed, self._get(P + "mlp.down_proj.weight_scale"), idx).float()
        else:
            dense = self._get(P + "mlp.down_proj.weight")
            if dense is None:
                raise ValueError(f"The checkpoint holds no down projection for layer {layer}.")
            cols = dense[:, idx].T.float()
        return cols, w_post.float()

    def record(self) -> str:
        """The files read: in the Hugging Face cache, the repo's snapshot and each file's blob name (its sha256, as the
        Hub stores large files); elsewhere each file's name and size."""
        parts = []
        for p in self.files:
            blob = Path(os.readlink(p)).name if p.is_symlink() else p.name  # the Hub's blob (a further link may follow)
            snap = p.parent.name if p.parent.parent.name == "snapshots" else None
            if snap and re.fullmatch(r"[0-9a-f]{64}", blob):
                parts.append(f"{p.name} sha256 {blob}")
            else:
                parts.append(f"{p.name} ({p.stat().st_size} bytes)")
        repo = self.files[0].parent.parent.parent.name if self.files[0].parent.parent.name == "snapshots" else None
        where = (f"{repo.replace('models--', '').replace('--', '/')} at revision {self.files[0].parent.name}: "
                 if repo and repo.startswith("models--") else "")
        return f"checkpoint {where}" + "; ".join(parts)


class GGUFColumns:
    """The same two tensors of each layer from a GGUF of Gemma 4, through its memory map: a down column costs one
    block per row (`gguf_weights.GGUFLinear.unpacked_cols`), unpacked to float32 (gguf-py's numbers)."""

    def __init__(self, path: Path) -> None:
        from backend import gguf_model as gm

        self.path = Path(path)
        self.reader = gm.open_gguf(self.path)
        self.tensors = gm.tensor_table(self.reader)
        meta = gm.read_metadata(self.reader)
        if meta.get("general.architecture") != MODEL_HINT:
            raise ValueError(f"{self.path.name} is a {meta.get('general.architecture')!r} model, not Gemma 4 ({MODEL_HINT}).")
        self.n_layers = int(meta.get(f"{MODEL_HINT}.block_count"))
        self.prov = gm.provenance(self.path, self.reader)
        self.name_of = gm.hf_to_gguf_names(self.n_layers)  # gguf-py's own map: down_proj -> blk.N.ffn_down, ...

    def _tensor(self, layer: int, hf_name: str) -> Any:
        name = self.name_of(f"model.layers.{int(layer)}.{hf_name}")
        t = self.tensors.get(name or "")
        if t is None:
            raise ValueError(f"{self.path.name} holds no {name or hf_name} for layer {layer}.")
        return t

    def width(self, layer: int) -> int:
        from backend import gguf_model as gm

        return int(gm.logical_shape(self._tensor(layer, "mlp.down_proj.weight"))[1])

    def columns(self, layer: int, neurons: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        from backend import gguf_model as gm
        from backend.gguf_weights import GGUFLinear

        down = self._tensor(layer, "mlp.down_proj.weight")
        post = self._tensor(layer, "post_feedforward_layernorm.weight")
        D, N = gm.logical_shape(down)
        lin = GGUFLinear(gm._blocks(down, torch.device("cpu")), down.tensor_type, N, D, dtype=torch.float32,
                         gguf_name=down.name)
        cols = lin.unpacked_cols(torch.tensor([int(n) for n in neurons], dtype=torch.long)).float()
        w_post = gm._dense(post, torch.device("cpu"), torch.float32).reshape(-1)
        return cols, w_post

    def record(self) -> str:
        p = self.prov
        return (f"gguf {p['file']} ({p['size']} bytes, fingerprint {p['id']}, header sha256 {p['sha256_header']}; "
                f"tensor types {json.dumps(p['tensor_types'])})")


def checkpoint_files(spec: Optional[str]) -> List[Path]:
    """The checkpoint's safetensors files: those named (a file, or a folder holding them, by its index when it has
    one), else the cached W4A16 checkpoint's."""
    if not spec:
        from backend.census import cached_checkpoint

        return cached_checkpoint(MODEL_ID)
    p = Path(spec).expanduser()
    if p.is_file():
        return [p]
    if (p / "model.safetensors.index.json").exists():
        names = sorted(set(json.loads((p / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"].values()))
        return [p / n for n in names]
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{p} holds no .safetensors files.")
    return files


# ------------------------------------------------------------------------------------------------------- the push


def read_stats(path: Path, model: str) -> Tuple[List[float], Dict[str, Any]]:
    """(each layer's p50, the stats index's record) from an act_scale index: `resid_q[:, 1]`, the median length of the
    stream leaving each layer, as census_run.py reads it. Refused when built for another model."""
    from backend.atlas_index import meta_of
    from backend.census import file_sha256

    path = Path(path).expanduser()
    if not path.exists():
        raise ValueError(f"No stats index at {path}: build it (scripts/atlas_build.py stats) or name the census kit's "
                         "copy with --stats.")
    with np.load(path, allow_pickle=False) as z:
        if "resid_q" not in z.files:
            raise ValueError(f"{path.name} holds no resid_q: it is not a stats index (act_scale).")
        resid_q = np.asarray(z["resid_q"])
        tokens = int(z["tokens"]) if "tokens" in z.files else None
        meta = meta_of(z)
    if meta.get("model") != model:
        raise ValueError(f"{path.name} was built for {meta.get('model')!r}, not {model!r}: its p50 is another model's "
                         "stream (name the right one with --stats, or the model with --model).")
    return [float(x) for x in resid_q[:, 1]], {
        "file": path.name, "sha256": file_sha256(path), "model": meta.get("model"), "built_at": meta.get("built_at"),
        "corpus": meta.get("corpus"), "tokens": tokens}


def pushes(entries: Sequence[Entry], source: Any, p50: Sequence[float]) -> Dict[int, np.ndarray]:
    """{layer: [D] float32}: each entry's push `u * (dose * p50_l)`, as take_census builds a row's (u from
    `write_dirs`, the scalar a Python float), summed in the entries' order over a layer's entries."""
    out: Dict[int, np.ndarray] = {}
    for layer in sorted({e.layer for e in entries}):
        es = [e for e in entries if e.layer == layer]
        cols, w_post = source.columns(layer, [e.neuron for e in es])
        u = write_dirs(cols, w_post)
        acc: Optional[torch.Tensor] = None
        for k, e in enumerate(es):
            d = u[k] * (float(e.dose) * float(p50[layer]))
            acc = d if acc is None else acc + d
        assert acc is not None
        out[layer] = np.ascontiguousarray(acc.numpy(), dtype=np.float32)
    return out


def code_commit() -> str:
    try:
        got = subprocess.run(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if got.returncode != 0 or not got.stdout.strip():
            return "unknown"
        st = subprocess.run(["git", "-C", str(ROOT_DIR), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, timeout=20)
        return got.stdout.strip() + ("" if st.returncode == 0 and not st.stdout.strip() else " (with changes)")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def describe(entries: Sequence[Entry]) -> str:
    return ", ".join(f"layer {e.layer} (page {e.layer + 1}) neuron {e.neuron} dose {e.dose:+g}" for e in entries)


def write_control_vector(path: Path, directions: Dict[int, np.ndarray], record: Dict[str, Any]) -> Path:
    """Write the file: llama.cpp's keys, `theodolite.*` from `record` (str, int, float, or lists of one of them), and
    a tensor direction.<l> per layer. Atomically: a temporary file beside it, then a rename."""
    import gguf

    if 0 in directions:
        raise ValueError("llama.cpp refuses direction.0: no vector is added after layer 0.")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        _write(gguf.GGUFWriter(str(tmp), ARCH), directions, record)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _write(w: Any, directions: Dict[int, np.ndarray], record: Dict[str, Any]) -> None:
    """The keys, then the tensors, through a gguf-py writer."""
    w.add_name(str(record.get("name", "theodolite control vector")))
    if record.get("description"):
        w.add_description(str(record["description"]))
    w.add_string(f"{ARCH}.model_hint", MODEL_HINT)
    w.add_int32(f"{ARCH}.layer_count", len(directions))
    for key, value in record.items():
        if key in ("name", "description"):
            continue
        k = KEY + key
        if isinstance(value, bool):
            w.add_bool(k, value)
        elif isinstance(value, int):
            w.add_int32(k, value)
        elif isinstance(value, float):
            w.add_float32(k, value)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(v, int) for v in value):
                w.add_array(k, [int(v) for v in value])
            else:
                w.add_array(k, list(value))
        else:
            w.add_string(k, str(value))
    for layer in sorted(directions):
        w.add_tensor(f"direction.{int(layer)}", np.ascontiguousarray(directions[layer], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_control_vector(path: Path) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """({layer: [D] float32}, every key's value) from a control-vector GGUF, as llama.cpp's loader names its tensors."""
    from backend import gguf_model as gm

    reader = gm.open_gguf(path)
    meta = gm.read_metadata(reader, long_arrays=True)
    out: Dict[int, np.ndarray] = {}
    for t in reader.tensors:
        m = re.fullmatch(r"direction\.(\d+)", t.name)
        if m is None:
            raise ValueError(f"{Path(path).name}: llama.cpp reads only direction.<layer> tensors, not {t.name!r}.")
        out[int(m.group(1))] = np.array(t.data, dtype=np.float32).reshape(-1)
    return out, meta


def export(entries: Sequence[Entry], out: Path, source: Any, stats: Path, model: str) -> Tuple[Path, Dict[int, np.ndarray],
                                                                                              Dict[str, Any]]:
    """Read, push and write; returns (the file, its directions, its record)."""
    check_entries(entries, source.n_layers, source.width)
    p50, stats_rec = read_stats(stats, model)
    if len(p50) != source.n_layers:
        raise ValueError(f"{stats_rec['file']} holds {len(p50)} layers' p50, the weights {source.n_layers} layers.")
    directions = pushes(entries, source, p50)
    one = len(entries) == 1
    name = (f"theodolite L{entries[0].layer} n{entries[0].neuron} dose {entries[0].dose:+g}" if one
            else f"theodolite set of {len(entries)} neurons at layers {', '.join(str(l) for l in sorted(directions))}")
    record: Dict[str, Any] = {
        "name": name,
        "description": f"The census's add push of {describe(entries)}, from Logos Theodolite: llama-server "
                       "--control-vector-scaled FILE:1 pushes it at the doses written here.",
        "model": model,
        "weights": source.record(),
        "layers": [int(e.layer) for e in entries],
        "pages": [int(e.layer) + 1 for e in entries],
        "neurons": [int(e.neuron) for e in entries],
        "doses": [float(e.dose) for e in entries],
        "p50": [float(p50[e.layer]) for e in entries],
        "stats_index": f"{stats_rec['file']}: built {stats_rec['built_at']} for {stats_rec['model']} over "
                       f"{stats_rec['tokens']} word-pieces of {stats_rec['corpus']}; p50 = resid_q[:, 1], the median length "
                       "of the stream leaving each layer",
        "stats_index_sha256": stats_rec["sha256"],
        "push": PUSH,
        "start_mark": START_MARK,
        "scale": SCALE,
        "llama_cpp": LLAMA_CPP_READ,
        "caveat": CAVEAT,
        "made_by": f"scripts/export_control_vector.py at {code_commit()}",
        "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return write_control_vector(out, directions, record), directions, record


def default_name(entries: Sequence[Entry]) -> str:
    e = entries[0]
    return f"theodolite-L{e.layer}-n{e.neuron}" + ("" if e.dose == 1.0 else f"-dose{e.dose:+g}") + ".gguf"


def show(path: Path) -> None:
    directions, meta = read_control_vector(path)
    for k, v in meta.items():
        if k.startswith(("general.", f"{ARCH}.", KEY)):
            print(f"{k}: {v}")
    p50 = dict(zip(meta.get(KEY + "layers") or [], meta.get(KEY + "p50") or []))
    for layer in sorted(directions):
        v = directions[layer]
        n = float(np.linalg.norm(v.astype(np.float64)))
        rel = f", {n / p50[layer]:.4f} x p50" if layer in p50 and p50[layer] else ""
        print(f"direction.{layer}: {v.size} floats, length {n:.4f}{rel}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--layer", type=int, help="the neuron's layer, counted from 0 as every script counts them")
    which.add_argument("--page", type=int, help="the neuron's page in the atlas (its address #/atlas/PAGE/NEURON): layer + 1")
    ap.add_argument("--neuron", type=int, help="the neuron's index in its layer")
    ap.add_argument("--dose", type=float,
                    help="the push as a share of the stream's p50 (default 1; llama.cpp's SCALE multiplies it)")
    ap.add_argument("--set", metavar="L:N[:DOSE],...",
                    help="several neurons (layers from 0, a dose left out is 1), summed at each layer as the atlas sums pushes")
    ap.add_argument("-o", "--out", help="the file to write (default theodolite-L<layer>-n<neuron>[-dose<d>].gguf here)")
    ap.add_argument("--force", action="store_true", help="write over a file already there")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", help=f"the checkpoint's safetensors file or folder (default {MODEL_ID} in the local cache)")
    src.add_argument("--gguf", help="read the direction from this GGUF of Gemma 4 instead (its own quantisation)")
    ap.add_argument("--stats", help="the stats index p50 is read from (default <atlas folder>/index/act_scale.npz)")
    ap.add_argument("--model", default=MODEL_ID, help=f"the model the stats index must be built for (default {MODEL_ID})")
    ap.add_argument("--show", metavar="FILE", help="print what a control-vector file was made from, and nothing else")
    a = ap.parse_args(argv)

    if a.show:
        show(Path(a.show))
        return 0
    try:
        if a.set:
            if a.layer is not None or a.page is not None or a.neuron is not None or a.dose is not None:
                raise ValueError("Give either --set (each entry's dose in it) or --layer/--page with --neuron and --dose, "
                                 "not both.")
            entries = parse_set(a.set)
        else:
            if a.neuron is None or (a.layer is None and a.page is None):
                raise ValueError("Name a neuron: --layer L (or --page P) and --neuron N, or --set L:N[:DOSE],...")
            dose = 1.0 if a.dose is None else float(a.dose)
            entries = [Entry(int(a.layer) if a.layer is not None else int(a.page) - 1, int(a.neuron), dose)]
        if a.set and not a.out:
            raise ValueError("A set's file needs a name: -o FILE.")
        out = Path(a.out) if a.out else Path(default_name(entries))
        if out.exists() and not a.force:
            raise ValueError(f"{out} is already there (--force writes over it).")
        if a.stats:
            stats = Path(a.stats)
        else:
            from backend.atlas_index import index_dir

            stats = index_dir() / "act_scale.npz"
        source = GGUFColumns(Path(a.gguf)) if a.gguf else CheckpointColumns(checkpoint_files(a.checkpoint))
        path, directions, record = export(entries, out, source, stats, a.model)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"wrote {path}: {describe(entries)}")
    for layer in sorted(directions):
        n = float(np.linalg.norm(directions[layer].astype(np.float64)))
        print(f"  direction.{layer}: length {n:.4f} (p50 at layer {layer}: {record['p50'][record['layers'].index(layer)]:.4f})")
    print(f"  weights: {record['weights']}")
    print(f"  stats index: {record['stats_index_sha256']}")
    print(f"load it with: llama-server -m <Gemma-4-31B-it GGUF> --control-vector-scaled {path}:1   (SCALE multiplies "
          "every dose; stay near 1: llama.cpp also pushes the start mark, which the census did not, and at 2 that "
          "breaks the text: ARCHITECTURE.md §6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
