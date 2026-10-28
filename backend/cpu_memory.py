"""Memory on the CPU, refused in plain words (`backend/cpu_memory.py`): before a reading that holds large buffers, an
estimate of what it needs against what the kernel says is available, and a CPU allocator's failure turned into the
same plain refusal.

On a GPU a reading that does not fit raises `torch.cuda.OutOfMemoryError`, which the server, the atlas, the vault,
the causal check and the scientist's tools catch and answer in plain words. The CPU raises nothing of the kind: torch's
allocator raises a `RuntimeError` ("DefaultCPUAllocator: can't allocate memory") only for a request the kernel refuses
outright, and under Linux's default overcommit a request it grants may later be met by the kernel's out-of-memory
killer, which ends the process without a word (the server, a census, an atlas build). So on the CPU the readings that
hold large buffers first ask `ensure`: a sweep (`sweep_need`), a census's batch and the vault's voice
(`writing_need`), a causal check's forward and a steer lens batch (`forward_need`). It raises `CPUMemoryRefused` (a
ValueError, so every caller that answers a ValueError in plain words answers this one) when MemAvailable
(/proc/meminfo, `w4a16_fast.free_bytes`) cannot hold the estimate; `is_oom` and `plain` let the callers answer an
allocator failure in the same words. Nothing here touches a reading on a GPU.

The estimates count the largest buffers a reading holds at once, from the model's own config and layers
(`Dims`): the keys and values a row keeps for each of its word-pieces across every layer, the widest layer's
activations, the attention's scores of a whole-text forward, the logits (the model's dtype at every position of a
first forward that keeps them, float32 at the last), and the largest transient of a packed layer's forward (the
unpacked weight and its temporaries: `unpack_bytes`). Worked out for the 31B in bf16 (the CPU's unpacking
temporaries of the widest W4A16 layer, 1.4 GB, among them): 1.07 MB a row-piece cached, and 4.7 GB for a census batch
of 32 at --max-new 64, where the card measured about 0.8 MB and 3.1 GiB (scripts/census_run.py's help); 6.3 GB for a
2048-piece sweep's writing and 6.9 GB for its reading, where the card measured 3.8 and 3.9 GiB (the comment above
engine_runtime.MAX_SWEEP_POSITIONS; there the captured states went to the host, uncounted). So the estimates lean
high, by about half: arithmetic set against the card's measurements, not measured on a CPU. A reading near the edge
may still be killed; THEODOLITE_CPU_MEMORY=off leaves the estimates out.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

CPU_MEMORY_ENV = "THEODOLITE_CPU_MEMORY"
GIB = 1 << 30
# what the thinking comparison keeps of the first pair's sweep while it writes (0.49 GiB of pole logits on the 31B:
# about 32 bytes a layer and vocabulary entry; the comment above engine_runtime.MAX_SWEEP_POSITIONS)
BUNDLE_BYTES_PER_LAYER_PIECE = 32


class CPUMemoryRefused(ValueError):
    """A reading on the CPU that MemAvailable cannot hold, refused in plain words before anything is read."""


def _model(x: Any) -> nn.Module:
    model = getattr(x, "hf_model", None)
    return x if model is None else model


def device_of(x: Any) -> torch.device:
    """Where a runtime's (or a model's) readings are taken."""
    dev = getattr(x, "device", None)
    if isinstance(dev, torch.device):
        return dev
    for p in _model(x).parameters():
        return p.device
    return torch.device("cpu")


def on_cpu(x: Any) -> bool:
    return device_of(x).type == "cpu" and os.environ.get(CPU_MEMORY_ENV, "").strip().lower() != "off"


def available() -> Optional[int]:
    """Bytes the kernel says new allocations can take on this machine (MemAvailable); None where it cannot be read."""
    from backend.w4a16_fast import free_bytes

    return free_bytes("cpu")


@dataclass(frozen=True)
class Dims:
    """What sizes a reading's buffers: layers, width, the widest feed-forward, vocabulary, attention heads, the bytes
    of keys and values one word-piece of one row keeps across every layer, and the model's bytes per number."""

    layers: int
    d_model: int
    d_ff: int
    vocab: int
    heads: int
    kv_piece: int
    item: int


def dims(x: Any) -> Optional[Dims]:
    """The model's sizes, from its config; None for one that has none to read (a check's stand-in for a runtime)."""
    model = _model(x)
    cfg = getattr(model, "config", None)
    cfg = getattr(cfg, "text_config", None) or cfg
    if not isinstance(model, nn.Module) or getattr(cfg, "num_hidden_layers", None) is None:
        return None
    L = int(cfg.num_hidden_layers)
    D = int(cfg.hidden_size)
    F = int(getattr(cfg, "intermediate_size", 4 * D))
    if getattr(cfg, "use_double_wide_mlp", False):
        F *= 2
    item = next((p.element_size() for p in model.parameters() if p.dtype.is_floating_point), 2)
    plc = getattr(cfg, "per_layer_config", None)
    if plc:
        kv = sum(2 * int(plc[l].num_key_value_heads) * int(plc[l].head_dim) for l in range(L))
    else:
        hd = int(getattr(cfg, "head_dim", None) or D // int(cfg.num_attention_heads))
        kv = 2 * L * int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads) * hd
    return Dims(L, D, F, int(cfg.vocab_size), int(cfg.num_attention_heads), kv * item, item)


def piece_bytes(d: Dims) -> int:
    """One word-piece of one row: its keys and values in every layer, and the widest layer's activations (the stream,
    its norms and the attention's output; the gate, up and their product)."""
    return d.kv_piece + (4 * d.d_model + 3 * d.d_ff) * d.item


def forward_piece_bytes(d: Dims) -> int:
    """One word-piece of one row in a forward that keeps no cache: the widest layer's activations, and one layer's
    keys and values (each layer's freed before the next runs)."""
    return d.kv_piece // max(1, d.layers) + (4 * d.d_model + 3 * d.d_ff) * d.item


def attention_bytes(d: Dims, rows: int, positions: int) -> int:
    """The attention's scores of a whole-text forward, in float32 (one layer at a time)."""
    return int(rows) * d.heads * int(positions) ** 2 * 4


def logits_bytes(d: Dims, rows_full: int, rows_last: int) -> int:
    """Logits kept at every position of `rows_full` row-pieces (the model's dtype, twice over in the softcap), and at
    the last of `rows_last` rows in float32 (the logits and their log-softmax)."""
    return int(rows_full) * d.vocab * 2 * d.item + int(rows_last) * d.vocab * 8


def unpack_bytes(x: Any) -> int:
    """The largest transient of one packed layer's forward, the layers being unpacked one at a time: a W4A16 module
    through the pinned unpacker (two int32 temporaries, the pieces and the weight: 12 bytes a weight, as
    w4a16_fast.resident_need counts them), a packed GGUF layer (its weight in the model's dtype, and about 12 bytes a
    weight of the rows unpacked at once), an int4mm layer whose kernel is turned off as the packed layer it then is;
    nothing for a plain or resident layer, or an int4mm one through the kernel (an edit in force unpacks only its
    columns or rows)."""
    from backend.gguf_weights import GGUFLinear

    worst = 0
    for mod in _model(x).modules():
        if (getattr(mod, "int4mm", False) and getattr(mod, "int4_enabled", True)) or "_resident_w" in mod.__dict__:
            continue
        if isinstance(mod, GGUFLinear):
            if mod.resident:
                continue
            out, inn = mod.out_features, mod.in_features
            item = torch.empty((), dtype=mod.compute_dtype).element_size()
            worst = max(worst, out * inn * item + min(out, mod.chunk_rows) * inn * 12)
        elif hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
            worst = max(worst, int(mod.weight_packed.shape[0]) * int(mod.weight_packed.shape[1]) * 8 * 12)
    return worst


def forward_need(x: Any, rows: int, positions: int, *, logit_rows: Optional[int] = None, cached: int = 0) -> int:
    """One forward of `rows` rows of `positions` word-pieces keeping no cache, read at one position each (`logit_rows`
    rows of logits, default `rows`), beside a cache of `cached` word-pieces of one row: a causal check's (its prefix
    cache in given mode), a steer lens batch's."""
    d = dims(x)
    if d is None:
        return 0
    lr = rows if logit_rows is None else logit_rows
    return (int(rows) * int(positions) * forward_piece_bytes(d) + attention_bytes(d, rows, positions)
            + logits_bytes(d, 0, lr) + int(cached) * d.kv_piece + unpack_bytes(x))


def writing_need(x: Any, rows: int, prompt: int, new: int, *, full_logits: bool = False) -> int:
    """`rows` answers of up to `new` word-pieces each written after a `prompt`-piece question, keys and values cached:
    a census batch (`atlas_ops.write_rows`, logits at the last position) or the vault's voice and a sweep's writing
    (`full_logits`: the first forward keeps logits at every position of the question)."""
    d = dims(x)
    if d is None:
        return 0
    T = int(prompt) + int(new)
    return (int(rows) * T * piece_bytes(d) + attention_bytes(d, rows, prompt)
            + logits_bytes(d, int(rows) * int(prompt) if full_logits else 0, rows) + unpack_bytes(x))


def sweep_need(x: Any, positions: int, *, thinking: bool = False) -> int:
    """A sighting of `positions` word-pieces, question and answer together: the larger of its writing (logits kept at
    every position of its first forward) and its reading (the states of every layer kept in float32 for the kinship
    of layers and in the model's dtype as captured, with its [layers, n, n] Gram matrices), and with the thinking
    comparison the first pair's bundle besides."""
    d = dims(x)
    if d is None:
        return 0
    n = int(positions)
    writing = writing_need(x, 1, n, 0, full_logits=True)
    reading = (n * d.layers * d.d_model * (4 + d.item) + 4 * d.layers * n * n + attention_bytes(d, 1, n)
               + unpack_bytes(x))
    return max(writing, reading) + (BUNDLE_BYTES_PER_LAYER_PIECE * d.layers * d.vocab if thinking else 0)


def size(n: Optional[int]) -> str:
    """Bytes in words: GiB from one GiB up, MiB below, KiB below that."""
    if n is None:
        return "an unknown amount"
    if n >= GIB:
        return f"{n / GIB:.1f} GiB"
    return f"{n / (1 << 20):.1f} MiB" if n >= 1 << 20 else f"{max(1, round(n / 1024))} KiB"


Need = Union[int, Callable[[], int]]


def fits(x: Any, need: Need) -> bool:
    """Whether `need` bytes (or what `need()` estimates, asked only on the CPU) fit in what is available, on the CPU;
    always on a GPU, with the estimates off, or where MemAvailable cannot be read."""
    if not on_cpu(x):
        return True
    free = available()
    return free is None or int(need() if callable(need) else need) <= free


def ensure(x: Any, need: Need, what: str, hint: str) -> None:
    """Refuse (`CPUMemoryRefused`) a reading on the CPU whose estimated `need` (bytes, or a function giving them, asked
    only on the CPU) MemAvailable cannot hold, in plain words: `what` names the reading ("This sweep"), `hint` what to
    do ("shorten the question"). Nothing on a GPU."""
    if fits(x, need):
        return
    need = need() if callable(need) else need
    raise CPUMemoryRefused(
        f"{what} needs about {size(need)} of memory on the CPU (estimated: backend/cpu_memory.py), and "
        f"{size(available())} is available (MemAvailable): {hint}. Nothing was read. ({CPU_MEMORY_ENV}=off runs it "
        "anyway, at the risk of the kernel stopping the process.)"
    )


_ALLOC_WORDS = ("DefaultCPUAllocator: can't allocate memory", "DefaultCPUAllocator: not enough memory")


def is_cpu_oom(exc: BaseException) -> bool:
    """Whether `exc` is torch's CPU allocator refusing an allocation: a plain RuntimeError in its words. torch's own
    out-of-memory error (`torch.cuda.OutOfMemoryError` is `torch.OutOfMemoryError` in torch 2.14) is the GPU's here,
    the only accelerator the instrument runs on, and is not the CPU's."""
    if isinstance(exc, torch.OutOfMemoryError):
        return False
    return isinstance(exc, RuntimeError) and any(w in str(exc) for w in _ALLOC_WORDS)


def is_oom(exc: BaseException) -> bool:
    """Running out of memory on a GPU or on the CPU."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or is_cpu_oom(exc)


def plain(exc: BaseException, what: str) -> str:
    """Running out of memory, as words for `what` ("this sweep"): on the GPU as before, on the CPU with the bytes the
    allocator was asked for when it says."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return f"The GPU ran out of memory during {what}."
    asked = re.search(r"allocate (\d+) bytes", str(exc))
    more = f" (one allocation of {size(int(asked.group(1)))} was refused)" if asked else ""
    return f"The CPU ran out of memory during {what}{more}."
