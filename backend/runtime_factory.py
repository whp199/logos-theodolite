"""Which runtime a process runs (`backend/runtime_factory.py`): one choice, read from the environment, for the server,
scripts/census_run.py and scripts/atlas_build.py.

    THEODOLITE_GGUF=<file.gguf>        a GGUF of Gemma 4 (backend/gguf_runtime.py), else the W4A16 checkpoint as today
    THEODOLITE_DEVICE=cpu | cuda:N     where the model runs (default cuda:0, as today)
    THEODOLITE_GGUF_LINEARS=packed | resident | dense   how a GGUF's quantised layers are held (default packed)
    THEODOLITE_GGUF_CONFIG=<config.json>                build a GGUF's model from this config, checked against the file
    THEODOLITE_DTYPE=bfloat16 | float32                 a GGUF model's dtype (default bfloat16, as the instrument runs)
    THEODOLITE_GGUF_LENS=jpp | none                     load the J++ lens beside a GGUF model (default), or leave it
                                                        out (3.4 GB): a census and the atlas's stats, wiring, layout
                                                        and steer builds do not read it; the sightings, the lens
                                                        build and the vault's UPDATE and INSERT do

With none of them set, `make_runtime()` is `TheodoliteRuntime(load_jlens=...)`, exactly as these entry points built it
before. On the CPU (THEODOLITE_DEVICE=cpu) the GPUs are hidden from this process before anything touches CUDA
(CUDA_VISIBLE_DEVICES=""), as the stand-in's scripts hide them. That is a precaution: the pinned runtime asks
`torch.cuda.max_memory_allocated(self.device)` whenever CUDA is available, which torch 2.14.1 answers with 0 while CUDA
is not initialised in the process and refuses for a CPU device once it is (`memory_stats_as_nested_dict`; read from
torch's code, no GPU here to show it); hidden GPUs keep a library from initialising CUDA in a CPU process.

A reading taken off the GPU, or in float32, is its own numeric class, and says so in its model id (`reading_class`):
the checkpoint on the CPU is `google/gemma-4-31B-it-qat-w4a16-ct@cpu/bfloat16` (`CheckpointRuntime`), a GGUF on the CPU
`gguf:<file>#<fingerprint>@cpu/bfloat16`. Every census run, index and lens card records that id, a run resumes only on
the same one and the server reads an index only for it, so readings of two classes never mix in one file. bfloat16
on a GPU, as the instrument has always run, adds nothing.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from backend.engine_runtime import TheodoliteRuntime

logger = logging.getLogger(__name__)

GGUF_ENV = "THEODOLITE_GGUF"
DEVICE_ENV = "THEODOLITE_DEVICE"
LINEARS_ENV = "THEODOLITE_GGUF_LINEARS"
CONFIG_ENV = "THEODOLITE_GGUF_CONFIG"
DTYPE_ENV = "THEODOLITE_DTYPE"
LENS_ENV = "THEODOLITE_GGUF_LENS"
DEFAULT_DEVICE = "cuda:0"
_DTYPES = ("bfloat16", "float32")


def runtime_device() -> str:
    """THEODOLITE_DEVICE as torch names it ("cpu", "cuda:0", ...), DEFAULT_DEVICE when unset; a ValueError in plain words
    for anything else."""
    d = os.environ.get(DEVICE_ENV, "").strip().lower() or DEFAULT_DEVICE
    if d == "cuda":
        d = "cuda:0"
    if d != "cpu" and not re.fullmatch(r"cuda:\d+", d):
        raise ValueError(f"{DEVICE_ENV}={d!r}: the runtime runs on cpu or cuda:N (cuda:0 by default).")
    return d


def hide_gpus() -> None:
    """Hide every GPU from this process (CUDA_VISIBLE_DEVICES=""), for a runtime on the CPU. It holds only if CUDA has
    not been initialised yet; when it has, a warning says so."""
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if torch.cuda.is_initialized():
        logger.warning("CUDA was initialised before the runtime was chosen, so the GPUs could not be hidden: set "
                       "CUDA_VISIBLE_DEVICES= before starting this process for a CPU runtime.")


def reading_class(device: Any, dtype: Any = None) -> str:
    """What a model id carries past the weights' own name when its readings are not taken as the instrument takes them
    (bfloat16 on a GPU): "@cpu/bfloat16", "@cpu/float32", "@cuda/float32"; "" for bfloat16 on any GPU. `dtype` defaults
    to bfloat16 (the checkpoint's)."""
    import torch

    dev = torch.device(device)
    dt = str(dtype if dtype is not None else torch.bfloat16).replace("torch.", "")
    return "" if dev.type == "cuda" and dt == "bfloat16" else f"@{dev.type}/{dt}"


class CheckpointRuntime(TheodoliteRuntime):
    """The pinned runtime on the W4A16 checkpoint, off the GPU: its own `load()`, unchanged, then its model id tagged
    with where the readings are taken (`reading_class`), so nothing taken on the CPU is read as a GPU reading."""

    def load(self) -> "CheckpointRuntime":
        super().load()  # loads the weights by self.model_id, the repo id, so the tag goes on after
        self.model_id = f"{self.model_id}{reading_class(self.device)}"
        return self


def make_runtime(*, load_jlens: bool = False, **kwargs: Any) -> Any:
    """The runtime this process runs, not yet loaded (call `.load()`): a `GGUFRuntime` on THEODOLITE_GGUF when it is
    set, else the pinned `TheodoliteRuntime` on the W4A16 checkpoint (a `CheckpointRuntime` on the CPU); on
    THEODOLITE_DEVICE (default cuda:0). `kwargs` go to `GGUFRuntime` (`load_jpp=False` leaves the lenses out)."""
    device = runtime_device()
    if device == "cpu":
        hide_gpus()
    path = os.environ.get(GGUF_ENV, "").strip()
    if path:
        import torch

        from backend.gguf_runtime import GGUFRuntime

        dtype_name = os.environ.get(DTYPE_ENV, "").strip().lower() or "bfloat16"
        if dtype_name not in _DTYPES:
            raise ValueError(f"{DTYPE_ENV}={dtype_name!r}: a GGUF model runs in {' or '.join(_DTYPES)}.")
        if not os.path.isfile(os.path.expanduser(path)):
            raise ValueError(f"{GGUF_ENV}={path!r}: there is no such file.")
        lens = os.environ.get(LENS_ENV, "").strip().lower() or "jpp"
        if lens not in ("jpp", "none"):
            raise ValueError(f"{LENS_ENV}={lens!r}: jpp (load the J++ lens) or none.")
        kwargs.setdefault("load_jpp", lens == "jpp")
        return GGUFRuntime(
            os.path.expanduser(path),
            device=device,
            load_jlens=load_jlens,
            linears=os.environ.get(LINEARS_ENV, "").strip().lower() or "packed",
            dtype=getattr(torch, dtype_name),
            hf_config=os.environ.get(CONFIG_ENV, "").strip() or None,
            **kwargs,
        )
    if device == DEFAULT_DEVICE:
        return TheodoliteRuntime(load_jlens=load_jlens)
    if device == "cpu":
        return CheckpointRuntime(device=device, load_jlens=load_jlens)
    return TheodoliteRuntime(device=device, load_jlens=load_jlens)
