"""A Gemma 4 text model from a GGUF file (`backend/gguf_model.py`), on the CPU or a GPU, and its provenance.

`build_gemma4(path)` reads the file with gguf-py (`gguf.GGUFReader`, a memory map), works out the model's config from
its metadata (or takes a matching Hugging Face config.json when given: it stands in for any key the file leaves out,
and is refused in plain words where it disagrees with one the file states), builds transformers' `Gemma4ForCausalLM`
without allocating its random weights, and fills it from the file by gguf-py's own name map (`gguf.get_tensor_name_map`
for the gemma4 arch: model.layers.N.mlp.down_proj -> blk.N.ffn_down, layer_scalar -> blk.N.layer_output_scale, and so
on). The seven linear layers of every block are `linears`:
- "packed": a block-quantised tensor becomes a `gguf_weights.GGUFLinear` that keeps its blocks and unpacks them at
  every forward (memory at rest about the file's size; on the CPU the blocks are read through the file's memory map);
- "resident": the same layer, unpacked once at load (memory the dense weight's);
- "dense": a plain `nn.Linear` holding the unpacked weight, as transformers' own GGUF loader would build it;
- "int4mm" (the CPU only): a Q4_0 tensor becomes an `int4_cpu.Int4GGUFLinear`, multiplied by torch's int4 CPU kernel,
  its blocks kept for the accessors; any other block type is packed. Not bit-identical to the other three: its model
  id says so (backend/int4_cpu.py).
A tensor stored as F32, F16 or BF16 is a plain `nn.Linear` in every mode. Every linear layer hears the vault's
overlays as the W4A16 forward does (`gguf_weights.with_overlays`). Embeddings, norms and layer scalars are unpacked to
`dtype` (bfloat16 by default, as the instrument runs; float32 for exact CPU checks).

Gemma 4's norms: `Gemma4RMSNorm` multiplies by its weight w (the unit offset Gemma 2 and 3 add at run time is already
in w), and a GGUF of Gemma 4 stores w as it is. Read from both converters: llama.cpp's `Gemma4Model.norm_shift` is 0.0
(conversion/gemma.py, master read 28 October 2026), where its Gemma 3 adds 1; transformers' GGUF loader subtracts 1
for gemma2 and gemma3 (`Gemma2TensorProcessor`) and has no processor for gemma4. So the weights are copied unchanged;
scripts/check_gguf.py shows the round trip, and scripts/compare_gguf_ct.py names a file whose norms sit exactly 1
above the checkpoint's.

Not read yet, and refused in plain words: Gemma 4's mixture-of-experts blocks (26B-A4B), per-layer embeddings
(hidden_size_per_layer_input > 0: E2B, E4B), shared key-value layers, and the tensor types gguf_weights.py lists as not
yet (the IQ types among them). The vision and audio towers are not in a text GGUF (llama.cpp writes them to a
separate mmproj file), and the instrument drops them anyway.

`load_tokenizer` takes the tokenizer from a Hugging Face repo when it is cached here and its vocabulary is the file's,
else builds it from the file's own metadata through transformers. `provenance` is the record every reading taken on
the model carries (`GGUFRuntime.weights_card`): the file's size, sha256 of its header, of the first MiB of its tensor
data and of its last MiB, the tensor types it holds, its arch and the metadata that matter.

`write_gemma4_gguf` writes a Gemma 4 text model in llama.cpp's layout (the keys and tensor names its converter writes,
norms unshifted, `rope_freqs` for the proportional rope of the full layers): for the CPU checks, which write the
stand-in of scripts/atlas_stand_in.py and read it back, and for scripts/compare_gguf_ct.py's self-test. It is not
llama.cpp's converter, and its Q4_K is a plain quantiser (each sub-block's range), not llama.cpp's search for scales.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from backend.gguf_weights import GGML_TYPES, GGUFLinear, install_overlay_forward, type_code, type_name, unpack

ARCH = "gemma4"
LINEAR_MODES = ("packed", "resident", "dense")  # the three that give the same forward bit for bit
KERNEL_MODES = ("int4mm",)  # torch's int4 CPU kernel: not the reference's bits (backend/int4_cpu.py)
MIB = 1 << 20
# The seven linear layers of a Gemma 4 block, by their transformers names under model.layers.N.
LINEAR_NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# The pieces that end a Gemma 4 turn beside <eos>, by name (1 <eos>, 106 <turn|>, 50 <|tool_response> on the 31B, as
# its generation_config.json and transformers' GGUF loader have them).
END_PIECES = ("<turn|>", "<|tool_response>")


# ------------------------------------------------------------------------------------------------- the file


def open_gguf(path: Union[str, Path]) -> Any:
    """A `gguf.GGUFReader` over the file (copy-on-write memory map: torch can read its blocks in place, and nothing is
    ever written back)."""
    from gguf import GGUFReader

    return GGUFReader(str(path), "c")


def read_metadata(reader: Any, *, long_arrays: bool = False) -> Dict[str, Any]:
    """The file's key-value metadata as Python values; arrays longer than 64 (the vocabulary, its merges and scores)
    are given as their length unless `long_arrays`."""
    out: Dict[str, Any] = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        n = len(field.data)
        is_array = bool(field.types) and int(field.types[0]) == 9  # GGUFValueType.ARRAY
        if is_array and n > 64 and not long_arrays:
            out[key] = {"array_of": n}
            continue
        out[key] = field.contents()
    return out


def tensor_table(reader: Any) -> Dict[str, Any]:
    """name -> the reader's tensor (`.tensor_type`, `.shape` in ggml's order, `.data` the bytes or numbers)."""
    return {t.name: t for t in reader.tensors}


def logical_shape(t: Any) -> Tuple[int, ...]:
    """A tensor's shape as torch has it (ggml lists the fastest dimension first)."""
    return tuple(int(x) for x in reversed(list(t.shape)))


def _key(meta: Dict[str, Any], name: str, default: Any = None) -> Any:
    return meta.get(f"{ARCH}.{name}", default)


# ------------------------------------------------------------------------------------------------ the config


# The gemma4 keys a Gemma 4 GGUF must state (or a config.json given beside it must stand in for).
NEEDED_KEYS = ("block_count", "embedding_length", "feed_forward_length", "attention.head_count",
               "attention.head_count_kv", "attention.key_length", "attention.key_length_swa",
               "attention.sliding_window", "attention.sliding_window_pattern", "attention.layer_norm_rms_epsilon")


def lacking(meta: Dict[str, Any], tensors: Dict[str, Any]) -> List[str]:
    """What a Gemma 4 needs that the file does not state: its missing NEEDED_KEYS (full names), and the rope_freqs
    tensor when it has none ([] for a file llama.cpp's converter wrote)."""
    out = [f"{ARCH}.{k}" for k in NEEDED_KEYS if _key(meta, k) is None]
    return out + (["rope_freqs.weight"] if "rope_freqs.weight" not in tensors else [])


def text_config_from_gguf(meta: Dict[str, Any], tensors: Dict[str, Any],
                          given: Optional[Tuple[Dict[str, Any], np.ndarray]] = None) -> Dict[str, Any]:
    """The `Gemma4TextConfig` keyword arguments a Gemma 4 GGUF's metadata and tensors give, under the keys llama.cpp's
    converter writes (conversion/gemma.py `Gemma4Model.set_gguf_parameters`). `given` (`gguf_metadata_of` of a
    config.json's config) stands in for every key and for the rope_freqs tensor the file leaves out; what the file
    states is read from the file. A ValueError in plain words for a file that is not a Gemma 4 text model this module
    reads, or that lacks what a Gemma 4 needs and no config was given for."""
    if given is not None:
        meta = {**given[0], **meta}
    arch = meta.get("general.architecture")
    if arch != ARCH:
        raise ValueError(f"This GGUF holds a {arch!r} model; only Gemma 4 ({ARCH!r}) is read here.")
    if _key(meta, "expert_count"):
        raise ValueError("This GGUF holds a Gemma 4 mixture-of-experts model (26B-A4B); its experts are not read yet.")
    if int(_key(meta, "embedding_length_per_layer_input", 0) or 0):
        raise ValueError("This GGUF holds a Gemma 4 with per-layer embeddings (E2B, E4B); they are not read yet.")
    if int(_key(meta, "attention.shared_kv_layers", 0) or 0):
        raise ValueError("This GGUF holds a Gemma 4 whose last layers share their keys and values; not read yet.")
    missing = [k for k in NEEDED_KEYS if _key(meta, k) is None]
    if missing:
        raise ValueError(f"This GGUF's metadata lacks {', '.join(f'{ARCH}.{k}' for k in missing)}, which a Gemma 4 needs; "
                         "give the model's config.json (hf_config, THEODOLITE_GGUF_CONFIG) to build it from.")
    L = int(_key(meta, "block_count"))
    swa = [bool(x) for x in _key(meta, "attention.sliding_window_pattern")]
    if len(swa) != L:
        raise ValueError(f"The sliding-window pattern names {len(swa)} layers, the file {L}.")
    layer_types = ["sliding_attention" if s else "full_attention" for s in swa]

    def per_layer(name: str) -> List[int]:
        v = _key(meta, name)
        return [int(x) for x in v] if isinstance(v, list) else [int(v)] * L

    def one_value(name: str, values: Sequence[int], what: str) -> int:
        if len(set(values)) != 1:
            raise ValueError(f"{ARCH}.{name} differs between {what} layers ({sorted(set(values))}); not read yet.")
        return int(values[0])

    kv = per_layer("attention.head_count_kv")
    n_kv = one_value("attention.head_count_kv", [k for k, s in zip(kv, swa) if s], "sliding")
    n_kv_full = one_value("attention.head_count_kv", [k for k, s in zip(kv, swa) if not s], "full")
    heads = one_value("attention.head_count", per_layer("attention.head_count"), "the")
    ff = one_value("feed_forward_length", per_layer("feed_forward_length"), "the")
    hd_full, hd = int(_key(meta, "attention.key_length")), int(_key(meta, "attention.key_length_swa"))
    for vname, kval in (("attention.value_length", hd_full), ("attention.value_length_swa", hd)):
        v = _key(meta, vname)
        if v is not None and int(v) != kval:
            raise ValueError(f"{ARCH}.{vname} is {v}, its key length {kval}: Gemma 4's values are as long as its keys.")
    rot_swa = _key(meta, "rope.dimension_count_swa")
    if rot_swa is not None and int(rot_swa) != hd:
        raise ValueError(f"The sliding layers rotate {rot_swa} of {hd} dimensions; Gemma 4's rotate all of them.")
    # The full layers' rope is "proportional" with a partial rotary factor: llama.cpp writes it as rope_freqs, one
    # factor per rotated pair, 1 for the first n_rot pairs and 1e30 (no rotation) for the rest.
    freqs = tensors.get("rope_freqs.weight")
    if freqs is not None:
        raw = torch.from_numpy(np.array(np.asarray(freqs.data).view(np.uint8).reshape(-1)))
        f = np.asarray(unpack(raw, freqs.tensor_type))
    elif given is not None:
        f = np.asarray(given[1], dtype=np.float32)
    else:
        raise ValueError("This GGUF has no rope_freqs tensor, so the full layers' partial rotary factor is not known; "
                         "give the model's config.json (hf_config, THEODOLITE_GGUF_CONFIG).")
    n_rot = int((f == 1.0).sum())
    if f.shape[0] != hd_full // 2 or not np.all(f[:n_rot] == 1.0) or not np.all(f[n_rot:] >= 1e29):
        raise ValueError(f"rope_freqs is not the proportional rope's pattern for a head of {hd_full} "
                         f"({f.shape[0]} factors, {n_rot} of them 1).")
    v_full = [t for t in tensors if t.startswith("blk.") and t.endswith(".attn_v.weight")]
    full_layers = [l for l, s in enumerate(swa) if not s]
    k_eq_v = not any(f"blk.{l}.attn_v.weight" in tensors for l in full_layers)
    if k_eq_v and len(v_full) != L - len(full_layers):
        raise ValueError("Some sliding layers have no value projection (attn_v); not a Gemma 4 this reads.")
    vocab = logical_shape(tensors["token_embd.weight"])[0]
    cfg: Dict[str, Any] = {
        "vocab_size": vocab,
        "vocab_size_per_layer_input": vocab,
        "hidden_size_per_layer_input": 0,
        "hidden_size": int(_key(meta, "embedding_length")),
        "intermediate_size": ff,
        "num_hidden_layers": L,
        "num_attention_heads": heads,
        "num_key_value_heads": n_kv,
        "num_global_key_value_heads": n_kv_full,
        "head_dim": hd,
        "global_head_dim": hd_full,
        "attention_k_eq_v": k_eq_v,
        "sliding_window": int(_key(meta, "attention.sliding_window")),
        "layer_types": layer_types,
        "rms_norm_eps": float(_key(meta, "attention.layer_norm_rms_epsilon")),
        "final_logit_softcapping": (float(_key(meta, "final_logit_softcapping"))
                                    if _key(meta, "final_logit_softcapping") is not None else None),
        "max_position_embeddings": int(_key(meta, "context_length", 131072)),
        "tie_word_embeddings": "output.weight" not in tensors,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 2 * n_rot / hd_full,
                               "rope_theta": float(_key(meta, "rope.freq_base", 1_000_000.0)),
                               "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": float(_key(meta, "rope.freq_base_swa", 10_000.0)),
                                  "rope_type": "default"},
        },
        "hidden_activation": "gelu_pytorch_tanh",  # not in the file: Gemma 4's GeGLU
    }
    for tk, ck in (("bos_token_id", "bos_token_id"), ("eos_token_id", "eos_token_id"), ("padding_token_id", "pad_token_id")):
        v = meta.get(f"tokenizer.ggml.{tk}")
        if v is not None:
            cfg[ck] = int(v)
    return cfg


def text_config_from_hf(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """A config.json (a path or its dict; a `Gemma4Config`'s `text_config` is taken from inside it) as text config
    keyword arguments."""
    d = json.loads(Path(config).read_text(encoding="utf-8")) if not isinstance(config, dict) else dict(config)
    d = dict(d.get("text_config") or d)
    for k in ("model_type", "architectures", "transformers_version", "dtype", "torch_dtype"):
        d.pop(k, None)
    return d


def gguf_metadata_of(cfg: Any) -> Tuple[Dict[str, Any], np.ndarray]:
    """What llama.cpp's converter would write for a `Gemma4TextConfig`, as `write_gemma4_gguf` writes it: the gemma4
    keys `text_config_from_gguf` reads, and the rope_freqs factors (1 for each rotated pair of the full layers' heads,
    1e30 for the rest). It stands in for what a file leaves out when a config.json is given (`build_gemma4`)."""
    L = int(cfg.num_hidden_layers)
    swa = [t == "sliding_attention" for t in cfg.layer_types]
    plc = cfg.per_layer_config
    rp = cfg.rope_parameters
    hd_full = int(plc[swa.index(False)].head_dim)
    hd_swa = int(plc[swa.index(True)].head_dim) if True in swa else hd_full
    keys: Dict[str, Any] = {
        "block_count": L, "context_length": int(cfg.max_position_embeddings), "embedding_length": int(cfg.hidden_size),
        "feed_forward_length": int(cfg.intermediate_size), "attention.head_count": int(cfg.num_attention_heads),
        "attention.head_count_kv": [int(plc[l].num_key_value_heads) for l in range(L)],
        "rope.freq_base": float(rp["full_attention"]["rope_theta"]),
        "rope.freq_base_swa": float(rp["sliding_attention"]["rope_theta"]),
        "attention.layer_norm_rms_epsilon": float(cfg.rms_norm_eps),
        "attention.key_length": hd_full, "attention.value_length": hd_full,
        "attention.key_length_swa": hd_swa, "attention.value_length_swa": hd_swa,
        "attention.sliding_window": int(cfg.sliding_window), "attention.sliding_window_pattern": swa,
        "rope.dimension_count": hd_full, "rope.dimension_count_swa": hd_swa,
    }
    if cfg.final_logit_softcapping:
        keys["final_logit_softcapping"] = float(cfg.final_logit_softcapping)
    n_rot = int(hd_full * float(rp["full_attention"].get("partial_rotary_factor", 1.0)) / 2)
    freqs = np.array([1.0] * n_rot + [1e30] * (hd_full // 2 - n_rot), dtype=np.float32)
    return {f"{ARCH}.{k}": v for k, v in keys.items()}, freqs


def config_shape(cfg: Any) -> Dict[str, Any]:
    """What a GGUF records of a model, read from a `Gemma4TextConfig` (the full layers' head size and key-value heads
    from its per-layer config; floats as the float32 a GGUF keeps)."""
    f32 = lambda v: None if v is None else float(np.float32(v))  # noqa: E731
    types_ = list(cfg.layer_types)
    sl = [i for i, t in enumerate(types_) if t != "full_attention"]
    fl = [i for i, t in enumerate(types_) if t == "full_attention"]
    plc = cfg.per_layer_config
    rp = cfg.rope_parameters or {}

    def rope(kind: str) -> Tuple[Any, ...]:
        r = rp.get(kind) or {}
        return (r.get("rope_type"), f32(r.get("rope_theta")), f32(r.get("partial_rotary_factor", 1.0)))

    return {
        "vocab_size": int(cfg.vocab_size), "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size), "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "head_dim": sorted({int(plc[i].head_dim) for i in sl}),
        "num_key_value_heads": sorted({int(plc[i].num_key_value_heads) for i in sl}),
        "global_head_dim": sorted({int(plc[i].head_dim) for i in fl}),
        "num_global_key_value_heads": sorted({int(plc[i].num_key_value_heads) for i in fl}),
        "attention_k_eq_v": bool(cfg.attention_k_eq_v), "sliding_window": int(cfg.sliding_window),
        "layer_types": types_, "rms_norm_eps": f32(cfg.rms_norm_eps),
        "final_logit_softcapping": f32(cfg.final_logit_softcapping),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "hidden_size_per_layer_input": int(cfg.hidden_size_per_layer_input or 0),
        "num_kv_shared_layers": int(cfg.num_kv_shared_layers or 0), "enable_moe_block": bool(cfg.enable_moe_block),
        "rope_full_attention": rope("full_attention"), "rope_sliding_attention": rope("sliding_attention"),
    }


def config_disagreements(from_gguf: Dict[str, Any], from_hf: Dict[str, Any]) -> List[str]:
    """Each thing the GGUF records that a given config states otherwise, in plain words ([] when they agree). Both
    are read through `Gemma4TextConfig`, so a default the config.json leaves out counts as stated."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    g, h = config_shape(Gemma4TextConfig(**from_gguf)), config_shape(Gemma4TextConfig(**from_hf))
    return [f"{k}: the GGUF says {g[k]!r}, the config.json {h[k]!r}" for k in g if g[k] != h[k]]


# ------------------------------------------------------------------------------------------------- the model


@contextmanager
def _parameters_on_meta() -> Iterator[None]:
    """Build modules with their parameters on the meta device (no memory, no random init) and their buffers real (the
    rotary frequencies, the embedding scale: computed at construction, as transformers computes them)."""
    register = nn.Module.register_parameter

    def on_meta(self: nn.Module, name: str, param: Optional[nn.Parameter]) -> None:
        register(self, name, param)
        if param is not None and param.device.type != "meta":
            self._parameters[name] = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    nn.Module.register_parameter = on_meta  # type: ignore[assignment]
    try:
        yield
    finally:
        nn.Module.register_parameter = register  # type: ignore[assignment]


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _blocks(t: Any, device: torch.device) -> torch.Tensor:
    """A tensor's bytes as uint8 [rows, row bytes] on `device` (on the CPU, the file's memory map itself)."""
    raw = np.asarray(t.data)
    rows = logical_shape(t)[0] if len(t.shape) > 1 else 1
    raw = raw.view(np.uint8).reshape(rows, -1)
    out = torch.from_numpy(raw)
    return out if device.type == "cpu" else out.to(device)


def _dense(t: Any, device: torch.device, dtype: torch.dtype, chunk_rows: int = 4096) -> torch.Tensor:
    """A tensor unpacked (float32, gguf's numbers) and cast to `dtype`, in its torch shape; a few thousand rows at a
    time, so the float32 temporaries of a 262,144-row embedding stay small."""
    shape = logical_shape(t)
    raw = _blocks(t, device)
    out = torch.empty(raw.shape[0], int(np.prod(shape)) // raw.shape[0], dtype=dtype, device=device)
    for r0 in range(0, raw.shape[0], chunk_rows):
        out[r0 : r0 + chunk_rows] = unpack(raw[r0 : r0 + chunk_rows], t.tensor_type).to(dtype)
    return out.reshape(shape)


def hf_to_gguf_names(n_layers: int) -> Callable[[str], Optional[str]]:
    """transformers' name of a Gemma4ForCausalLM tensor -> its GGUF name, by gguf-py's own map for the gemma4 arch."""
    import gguf

    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, n_layers)

    def to_gguf(hf_name: str) -> Optional[str]:
        base, suffix = hf_name, ""
        if hf_name.endswith((".weight", ".bias")):
            base, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix
        g = name_map.get_name(base)
        if g is None:
            return None
        # llama.cpp names the layer scalar layer_output_scale.weight (its converter adds ".weight" to layer_scalar)
        return g + (suffix or ".weight")

    return to_gguf


@torch.no_grad()
def build_gemma4(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    linears: str = "packed",
    hf_config: Union[None, str, Path, Dict[str, Any]] = None,
    attn_implementation: str = "sdpa",
    reader: Any = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """(Gemma4ForCausalLM in eval mode on `device`, a record of how it was built). See the module's docstring."""
    from transformers import GenerationConfig
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    if linears not in LINEAR_MODES + KERNEL_MODES:
        raise ValueError(f"linears is one of {', '.join(LINEAR_MODES + KERNEL_MODES)}, not {linears!r}.")
    device = torch.device(device)
    if linears == "int4mm":
        from backend import int4_cpu

        why = int4_cpu.kernel_missing() or (None if device.type == "cpu" else
                                            f"int4mm is torch's CPU kernel, and this model is built on {device}")
        if why:
            raise ValueError(f"The int4mm linears cannot be used: {why}. Use packed, resident or dense.")
    reader = reader if reader is not None else open_gguf(path)
    meta = read_metadata(reader)
    tensors = tensor_table(reader)
    for t in tensors.values():
        type_code(t.tensor_type)  # refuses a type not read yet, before anything is built
    # with a config.json, it stands in for whatever the file leaves out, and is checked against all the file states
    hf_kwargs = text_config_from_hf(hf_config) if hf_config is not None else None
    given = gguf_metadata_of(Gemma4TextConfig(**hf_kwargs)) if hf_kwargs is not None else None
    cfg_kwargs = text_config_from_gguf(meta, tensors, given)
    source = "gguf"
    if hf_kwargs is not None:
        differ = config_disagreements(cfg_kwargs, hf_kwargs)
        if differ:
            raise ValueError("The GGUF and the config.json given describe different models, so neither is used: "
                             + "; ".join(differ) + ".")
        cfg_kwargs = hf_kwargs
        lack = lacking(meta, tensors)
        source = (f"config.json ({hf_config if not isinstance(hf_config, dict) else 'given'}), checked against the GGUF"
                  + (f", which lacks {', '.join(lack)}" if lack else ""))
    cfg = Gemma4TextConfig(**cfg_kwargs)
    cfg._attn_implementation = attn_implementation
    cfg.architectures = ["Gemma4ForCausalLM"]

    with _default_dtype(dtype), _parameters_on_meta():
        hf = Gemma4ForCausalLM(cfg)
    hf.eval()
    to_gguf = hf_to_gguf_names(cfg.num_hidden_layers)
    used: set = set()
    kinds: Counter = Counter()

    def take(hf_name: str) -> Any:
        g = to_gguf(hf_name)
        t = tensors.get(g) if g else None
        if t is None:
            raise ValueError(f"This GGUF has no {g or hf_name} ({hf_name}), which the model needs.")
        used.add(g)
        return t

    lm = hf.model
    for l, layer in enumerate(lm.layers):
        for lname in LINEAR_NAMES:
            parent_name, attr = lname.split(".")
            parent = getattr(layer, parent_name)
            old = getattr(parent, attr, None)
            if old is None:
                continue  # a full layer whose values are its keys has no v_proj
            hf_name = f"model.layers.{l}.{lname}.weight"
            t = take(hf_name)
            shape = logical_shape(t)
            if shape != (old.out_features, old.in_features):
                raise ValueError(f"{t.name} is {list(shape)}, the model's {lname} [{old.out_features}, {old.in_features}].")
            qname = type_name(t.tensor_type)
            if linears == "int4mm" and qname == "Q4_0" and int4_cpu.fits(old.out_features, old.in_features):
                new = int4_cpu.Int4GGUFLinear(_blocks(t, device), old.in_features, old.out_features, dtype=dtype,
                                              gguf_name=t.name)
                kinds["Q4_0 int4mm"] += 1
            elif linears != "dense" and GGML_TYPES[type_code(t.tensor_type)][1] > 1:
                new: nn.Module = GGUFLinear(_blocks(t, device), t.tensor_type, old.in_features, old.out_features,
                                            dtype=dtype, resident=(linears == "resident"), gguf_name=t.name)
                kinds[f"{qname} {'resident' if linears == 'resident' else 'packed'}"] += 1
            else:
                new = nn.Linear(old.in_features, old.out_features, bias=False, device="meta")
                new.weight = nn.Parameter(_dense(t, device, dtype), requires_grad=False)
                install_overlay_forward(new)
                kinds[f"{qname} dense"] += 1
            setattr(parent, attr, new)
    # everything else: embeddings, norms, layer scalars (buffers), by name
    for name, p in list(hf.named_parameters()):
        if p.device.type != "meta":
            continue
        if name == "lm_head.weight" and cfg.tie_word_embeddings:
            continue
        t = take(name)
        mod = hf.get_submodule(name.rsplit(".", 1)[0])
        val = _dense(t, device, dtype)
        if tuple(val.shape) != tuple(p.shape):
            raise ValueError(f"{t.name} is {list(val.shape)}, the model's {name} {list(p.shape)}.")
        mod._parameters[name.rsplit(".", 1)[1]] = nn.Parameter(val, requires_grad=False)
    for name, b in list(hf.named_buffers()):
        if not name.endswith("layer_scalar"):
            continue
        t = take(name)
        mod = hf.get_submodule(name.rsplit(".", 1)[0])
        mod._buffers["layer_scalar"] = _dense(t, device, dtype).reshape(b.shape)
    if cfg.tie_word_embeddings:
        hf.lm_head.weight = lm.embed_tokens.weight
    # every other buffer (the rotary frequencies, the embedding scale) onto the device, keeping its own dtype
    for mod in hf.modules():
        for bname, b in list(mod._buffers.items()):
            if b is not None and b.device != device:
                mod._buffers[bname] = b.to(device)
    left = [n for n, p in hf.named_parameters() if p.device.type == "meta"]
    if left:
        raise ValueError(f"The GGUF did not fill {', '.join(left[:4])}{' and more' if len(left) > 4 else ''}.")
    unused = sorted(set(tensors) - used - {"rope_freqs.weight"})
    if unused:
        raise ValueError(f"The GGUF holds tensors this model has no place for ({', '.join(unused[:4])}"
                         f"{' and more' if len(unused) > 4 else ''}): not a Gemma 4 text model this reads.")
    eos = [int(cfg.eos_token_id)] if isinstance(cfg.eos_token_id, int) else list(cfg.eos_token_id or [])
    names = reader.fields["tokenizer.ggml.tokens"].contents() if "tokenizer.ggml.tokens" in reader.fields else []
    eos += [names.index(p) for p in END_PIECES if p in names]
    hf.generation_config = GenerationConfig(bos_token_id=cfg.bos_token_id, eos_token_id=list(dict.fromkeys(eos)),
                                            pad_token_id=cfg.pad_token_id)
    hf.requires_grad_(False)
    record = {"config_source": source, "linears": linears, "layers_as": dict(sorted(kinds.items())),
              "dtype": str(dtype).replace("torch.", ""), "device": str(device),
              "norms": "w as stored (Gemma 4: no unit offset; llama.cpp's norm_shift 0)"}
    return hf, record


# --------------------------------------------------------------------------------------------- the tokenizer


def _pieces(names: Sequence[str]) -> List[str]:
    return [str(n).replace("▁", " ") for n in names]


def load_tokenizer(path: Union[str, Path], repos: Sequence[str] = (), reader: Any = None) -> Tuple[Any, Dict[str, Any]]:
    """(tokenizer, a record of where it came from). From the first of `repos` that is in the local Hugging Face cache
    and holds the file's vocabulary piece for piece (the "▁" of SentencePiece read as a space on both sides);
    else from the file's own metadata through transformers' GGUF tokenizer steps
    (`integrations.gguf.get_gguf_tokenizer`, then `convert_gguf_tokenizer`, into a `PreTrainedTokenizerFast`; not
    `AutoTokenizer.from_pretrained(folder, gguf_file=name)`, which fails on a llama.cpp Gemma 4: see below), which for
    Gemma 4 build it from the file's pieces and merges (whether that one splits text exactly as the repo's
    tokenizer.json does is not checked here: no Gemma 4 tokenizer is cached on the machine this was written on)."""
    from transformers import AutoTokenizer

    reader = reader if reader is not None else open_gguf(path)
    names = reader.fields["tokenizer.ggml.tokens"].contents() if "tokenizer.ggml.tokens" in reader.fields else None
    problems: Dict[str, str] = {}
    for repo in repos:
        try:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        except Exception as exc:  # not cached, or not a tokenizer
            problems[repo] = f"not cached here ({type(exc).__name__})"
            continue
        theirs = _pieces(tok.convert_ids_to_tokens(list(range(len(tok)))))
        if names is not None and len(theirs) == len(names) and theirs == _pieces(names):
            return tok, {"source": f"hf:{repo}", "matches_gguf_vocab": True, "vocab": len(names)}
        problems[repo] = (f"its vocabulary ({len(theirs)} pieces) is not the file's "
                          f"({len(names) if names is not None else 'none'})")
    rec: Dict[str, Any] = {"repos_tried": problems} if problems else {}
    # transformers' own route, `AutoTokenizer.from_pretrained(folder, gguf_file=name)`, first builds the model's config
    # from the file, and for a llama.cpp Gemma 4 that fails (its head_count_kv is a per-layer list, which
    # Gemma4TextConfig refuses as num_key_value_heads); so its two tokenizer steps are called here without it
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.gguf import convert_gguf_tokenizer, get_gguf_tokenizer

    arch, tok_dict, tok_cfg = get_gguf_tokenizer(str(path))
    backend, extra = convert_gguf_tokenizer(arch, tok_dict)
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, **{k: v for k, v in {**tok_cfg, **extra}.items() if v is not None})
    tmpl = reader.fields.get("tokenizer.chat_template")
    if tmpl is not None and not getattr(tok, "chat_template", None):
        tok.chat_template = tmpl.contents()
    rec.update(source="gguf metadata (transformers' GGUF tokenizer converter)", vocab=len(tok),
               tokenizer_model=reader.fields["tokenizer.ggml.model"].contents() if "tokenizer.ggml.model" in reader.fields else None)
    return tok, rec


# --------------------------------------------------------------------------------------------- the provenance


def _sha256_range(f: Any, start: int, n: int) -> str:
    f.seek(start)
    h = hashlib.sha256()
    left = n
    while left > 0:
        chunk = f.read(min(left, 8 * MIB))
        if not chunk:
            break
        h.update(chunk)
        left -= len(chunk)
    return h.hexdigest()


# The metadata a reading's record keeps (beyond the arch's own keys, which are all kept).
_GENERAL_KEYS = ("general.name", "general.basename", "general.finetune", "general.size_label", "general.file_type",
                 "general.quantization_version", "general.quantized_by", "general.organization", "general.license",
                 "general.base_model.0.name", "general.base_model.0.repo_url", "general.alignment",
                 "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id",
                 "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id")


def provenance(path: Union[str, Path], reader: Any = None) -> Dict[str, Any]:
    """The file a model was read from, as every reading on it records it: its name and size; sha256 of its header
    (the bytes before the tensor data: every key and every tensor's name, type, shape and offset), of the first MiB of
    its tensor data (the header itself runs to some MiB on the 31B: its vocabulary is among the keys) and of its last
    MiB; `id`, a short fingerprint of those four; the tensor types and how many of each; its arch and the metadata that
    matter. The whole file is not hashed (an 18 GB file would take minutes); the header and the two ends of the data
    change with any change of key, tensor table or the data in its first or last MiB, not with one in between."""
    p = Path(path)
    reader = reader if reader is not None else open_gguf(p)
    size = p.stat().st_size
    start = int(reader.data_offset)
    with open(p, "rb") as f:
        head = _sha256_range(f, 0, start)
        first = _sha256_range(f, start, min(MIB, size - start))
        last = _sha256_range(f, max(0, size - MIB), min(MIB, size))
    meta = read_metadata(reader)
    types = Counter(type_name(t.tensor_type) for t in reader.tensors)
    kept = {k: v for k, v in meta.items() if k.startswith(f"{meta.get('general.architecture')}.") or k in _GENERAL_KEYS}
    fid = hashlib.sha256(f"{size}:{head}:{first}:{last}".encode()).hexdigest()[:16]
    n_params = int(sum(int(np.prod(list(t.shape))) for t in reader.tensors))
    return {
        "format": "gguf",
        "file": p.name,
        "size": size,
        "id": fid,
        "sha256_header": head,
        "header_bytes": int(reader.data_offset),
        "sha256_data_first_mib": first,
        "sha256_last_mib": last,
        "architecture": meta.get("general.architecture"),
        "name": meta.get("general.name"),
        "tensors": len(reader.tensors),
        "parameters": n_params,
        "tensor_types": dict(sorted(types.items())),
        "metadata": kept,
    }


# ------------------------------------------------------------------------------------------------- the writer


def quantize_q4_k(x: np.ndarray) -> np.ndarray:
    """A plain Q4_K quantiser for rows [..., n] (n a multiple of 256) -> uint8 [..., n / 256 x 144]: each sub-block of
    32 takes its own range (min clipped to at most 0), the super-block's d and dmin are the largest sub-scale and
    sub-minimum over 63 in fp16, and the codes are rounded to nearest. Valid Q4_K blocks, not llama.cpp's choice."""
    shape = x.shape
    b = np.asarray(x, dtype=np.float32).reshape(-1, 256)
    n = b.shape[0]
    sub = b.reshape(n, 8, 32)
    lo = np.minimum(sub.min(-1), 0.0)
    hi = sub.max(-1)
    scale = (hi - lo) / 15.0
    mneg = -lo
    d = (scale.max(-1, keepdims=True) / 63.0).astype(np.float16)
    dmin = (mneg.max(-1, keepdims=True) / 63.0).astype(np.float16)
    d32, dmin32 = d.astype(np.float32), dmin.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(d32 > 0, np.rint(scale / d32), 0).clip(0, 63).astype(np.uint8)
        mn = np.where(dmin32 > 0, np.rint(mneg / dmin32), 0).clip(0, 63).astype(np.uint8)
        eff_s = d32 * sc.astype(np.float32)
        eff_m = dmin32 * mn.astype(np.float32)
        q = np.where(eff_s[..., None] > 0, np.rint((sub + eff_m[..., None]) / eff_s[..., None]), 0).clip(0, 15)
    q = q.astype(np.uint8)
    # the 12 scale bytes, the inverse of gguf's Q4_K.get_scale_min
    d_b = (sc[:, :4] & 0x3F) | ((sc[:, 4:] >> 4) << 6)
    m_b = (mn[:, :4] & 0x3F) | ((mn[:, 4:] >> 4) << 6)
    md_b = (sc[:, 4:] & 0x0F) | ((mn[:, 4:] & 0x0F) << 4)
    scales = np.concatenate([d_b, m_b, md_b], axis=-1).astype(np.uint8)
    # qs: byte g*32 + j holds sub-block 2g's code j (low nibble) and sub-block 2g+1's (high)
    qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(n, 128).astype(np.uint8)
    out = np.concatenate([d.view(np.uint8), dmin.view(np.uint8), scales, qs], axis=-1)
    return out.reshape(*shape[:-1], shape[-1] // 256 * 144)


# llama.cpp's type for a row a 256-block type does not divide (src/llama-quant.cpp `tensor_type_fallback`, master read
# 28 October 2026); it turns to F16 when 32 does not divide the row either, and so does a 32-block type's row
FALLBACK_TYPES = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}


def quantize_rows(x: np.ndarray, qtype: str) -> Tuple[np.ndarray, str]:
    """Rows [..., n] in `qtype` -> (the array GGUFWriter takes, the type written). A row a block does not divide falls
    back as llama.cpp's quantiser does (`FALLBACK_TYPES`: Q4_K to Q5_0, Q6_K to Q8_0, ...; F16 when 32 does not divide
    it either)."""
    import gguf

    qt = qtype.upper()
    n = x.shape[-1]
    bs = GGML_TYPES[type_code(qt)][1]
    if n % bs:
        qt = FALLBACK_TYPES.get(qt, "F16") if bs > 32 else "F16"
        if n % GGML_TYPES[type_code(qt)][1]:
            qt = "F16"
    x = np.ascontiguousarray(x, dtype=np.float32)
    if qt == "F32":
        return x, qt
    if qt == "F16":
        return x.astype(np.float16), qt
    if qt == "Q4_K":
        return quantize_q4_k(x), qt
    return gguf.quants.quantize(x, gguf.GGMLQuantizationType[qt]), qt


_FILE_TYPES = {"F32": "ALL_F32", "F16": "MOSTLY_F16", "BF16": "MOSTLY_BF16", "Q8_0": "MOSTLY_Q8_0",
               "Q4_0": "MOSTLY_Q4_0", "Q4_1": "MOSTLY_Q4_1", "Q5_0": "MOSTLY_Q5_0", "Q5_1": "MOSTLY_Q5_1",
               "Q4_K": "MOSTLY_Q4_K_M"}


@torch.no_grad()
def write_gemma4_gguf(
    hf: nn.Module,
    path: Union[str, Path],
    *,
    tokenizer: Any = None,
    qtype: str = "Q8_0",
    embed_qtype: Optional[str] = None,
    blocks: Optional[Dict[str, Tuple[np.ndarray, str]]] = None,
    name: str = "gemma4 (written by backend/gguf_model.py)",
    norm_shift: float = 0.0,
) -> Path:
    """Write a Gemma 4 text model (a `Gemma4ForCausalLM`, or a `Gemma4ForConditionalGeneration`'s text part) as llama.cpp's
    converter lays it out: the gemma4 keys, `rope_freqs`, norms and layer scalars in F32 and unshifted (`norm_shift`
    other than 0 is for the checks' counter-example only), the seven linear layers of every block in `qtype` (with
    llama.cpp's fallback where a block does not divide a row), the embedding in `embed_qtype` (default: `qtype`).
    `blocks` gives some tensors' bytes outright ({gguf name: (uint8 array, type)}), for the comparison's self-test.
    With `tokenizer` (a byte-level BPE fast tokenizer, as the stand-in's), its vocabulary, merges, special ids and chat
    template are written as llama.cpp writes a "gpt2" tokenizer."""
    import gguf
    from backend.causal_check import language_model

    lm = language_model(hf)
    cfg = getattr(lm, "config", None) or hf.config
    cfg = getattr(cfg, "text_config", None) or cfg
    L = int(cfg.num_hidden_layers)
    swa = [t == "sliding_attention" for t in cfg.layer_types]
    plc = cfg.per_layer_config
    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_name(name)
    w.add_block_count(L)
    w.add_context_length(int(cfg.max_position_embeddings))
    w.add_embedding_length(int(cfg.hidden_size))
    w.add_feed_forward_length(int(cfg.intermediate_size))
    w.add_head_count(int(cfg.num_attention_heads))
    w.add_head_count_kv([int(plc[l].num_key_value_heads) for l in range(L)])
    rp = cfg.rope_parameters
    w.add_rope_freq_base(float(rp["full_attention"]["rope_theta"]))
    w.add_rope_freq_base_swa(float(rp["sliding_attention"]["rope_theta"]))
    w.add_layer_norm_rms_eps(float(cfg.rms_norm_eps))
    hd_full = int(plc[swa.index(False)].head_dim)
    hd_swa = int(plc[swa.index(True)].head_dim) if True in swa else hd_full
    w.add_key_length(hd_full)
    w.add_value_length(hd_full)
    w.add_key_length_swa(hd_swa)
    w.add_value_length_swa(hd_swa)
    if cfg.final_logit_softcapping:
        w.add_final_logit_softcapping(float(cfg.final_logit_softcapping))
    w.add_sliding_window(int(cfg.sliding_window))
    w.add_shared_kv_layers(int(getattr(cfg, "num_kv_shared_layers", 0) or 0))
    w.add_embedding_length_per_layer_input(int(getattr(cfg, "hidden_size_per_layer_input", 0) or 0))
    w.add_sliding_window_pattern(swa)
    w.add_rope_dimension_count(hd_full)
    w.add_rope_dimension_count_swa(hd_swa)
    w.add_file_type(int(getattr(gguf.LlamaFileType, _FILE_TYPES.get(qtype.upper(), "MOSTLY_Q8_0"))))
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if tokenizer is not None:
        V = len(tokenizer)
        toks = tokenizer.convert_ids_to_tokens(list(range(V)))
        special = set(tokenizer.all_special_ids)
        model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
        merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", [])]
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("default")
        w.add_token_list(toks)
        w.add_token_types([int(gguf.TokenType.CONTROL) if i in special else int(gguf.TokenType.NORMAL) for i in range(V)])
        w.add_token_merges(merges)
        for tk, fn in (("bos_token_id", w.add_bos_token_id), ("eos_token_id", w.add_eos_token_id),
                       ("pad_token_id", w.add_pad_token_id), ("unk_token_id", w.add_unk_token_id)):
            v = getattr(tokenizer, tk, None)
            if v is not None:
                fn(int(v))
        w.add_add_bos_token(True)
        if getattr(tokenizer, "chat_template", None):
            w.add_chat_template(tokenizer.chat_template)
    given = dict(blocks or {})
    written: Counter = Counter()

    def put(gname: str, t: torch.Tensor, qt: Optional[str]) -> None:
        if gname in given:
            arr, gt = given.pop(gname)
            w.add_tensor(gname, np.ascontiguousarray(arr), raw_dtype=gguf.GGMLQuantizationType[gt])
            written[gt] += 1
            return
        x = t.detach().float().cpu().numpy()
        if qt is None or x.ndim == 1:
            w.add_tensor(gname, np.ascontiguousarray(x.reshape(-1) if x.ndim <= 1 else x, dtype=np.float32))
            written["F32"] += 1
            return
        arr, got = quantize_rows(x, qt)
        if arr.dtype == np.uint8:
            w.add_tensor(gname, arr, raw_dtype=gguf.GGMLQuantizationType[got])
        else:
            w.add_tensor(gname, arr)
        written[got] += 1

    # the full layers' proportional rope, as llama.cpp's converter writes it
    prf = float(rp["full_attention"].get("partial_rotary_factor", 1.0))
    n_rot = int(hd_full * prf / 2)
    put("rope_freqs.weight", torch.tensor([1.0] * n_rot + [1e30] * (hd_full // 2 - n_rot)), None)
    put("token_embd.weight", lm.embed_tokens.weight, embed_qtype or qtype)
    put("output_norm.weight", lm.norm.weight.float() + norm_shift, None)
    if not getattr(cfg, "tie_word_embeddings", True):
        put("output.weight", hf.lm_head.weight, embed_qtype or qtype)
    for l, layer in enumerate(lm.layers):
        b = f"blk.{l}."
        a = layer.self_attn
        put(b + "attn_norm.weight", layer.input_layernorm.weight.float() + norm_shift, None)
        put(b + "attn_q.weight", a.q_proj.weight, qtype)
        put(b + "attn_q_norm.weight", a.q_norm.weight.float() + norm_shift, None)
        put(b + "attn_k.weight", a.k_proj.weight, qtype)
        put(b + "attn_k_norm.weight", a.k_norm.weight.float() + norm_shift, None)
        if getattr(a, "v_proj", None) is not None:
            put(b + "attn_v.weight", a.v_proj.weight, qtype)
        put(b + "attn_output.weight", a.o_proj.weight, qtype)
        put(b + "post_attention_norm.weight", layer.post_attention_layernorm.weight.float() + norm_shift, None)
        put(b + "ffn_norm.weight", layer.pre_feedforward_layernorm.weight.float() + norm_shift, None)
        put(b + "ffn_gate.weight", layer.mlp.gate_proj.weight, qtype)
        put(b + "ffn_up.weight", layer.mlp.up_proj.weight, qtype)
        put(b + "ffn_down.weight", layer.mlp.down_proj.weight, qtype)
        put(b + "post_ffw_norm.weight", layer.post_feedforward_layernorm.weight.float() + norm_shift, None)
        put(b + "layer_output_scale.weight", layer.layer_scalar.reshape(-1).float(), None)
    if given:
        raise ValueError(f"No tensor of the model is named {', '.join(sorted(given))}.")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return Path(path)


def gguf_stem_id(path: Union[str, Path], prov: Dict[str, Any]) -> str:
    """The model id a GGUF runtime gives its readings, before `runtime_factory.reading_class` says where and in what
    they were taken: the file's name and its fingerprint (`provenance()["id"]`), so an index or census taken on one file
    is not read as another's (as far as the fingerprint tells them apart), nor as the W4A16 checkpoint's."""
    return f"gguf:{Path(path).name}#{prov['id']}"
