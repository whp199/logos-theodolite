"""Faster ways to unpack the checkpoint's 4-bit weights, each giving the very bits the runtime's own unpacker gives
(`engine_runtime.dequantize_w4a16_full`), installed from outside the runtime (`install`): R4 pins engine_runtime.py by
sha256 (experiments/scripts/R4_common.py), so the file is left byte for byte as it is.

Every forward of the runtime unpacks all 410 quantised linears from their packed int32 words (eight 4-bit pieces a
word, a bf16 scale per 32 weights) with five eager operations, three of them over int32 tensors four bytes a weight
wide: some 32.6 bytes of memory traffic a weight, where the matrix product itself reads 2. By that arithmetic most of
the 0.71 s decode step measured on the RTX 5090 is unpacking; the split has not been timed. The modes:

- "reference": the runtime's own unpacker, as loaded (the default);
- "lut": each packed byte (two pieces, the low nibble first on a little-endian machine, as every CUDA card is) looked
  up in a 256-entry table of (piece - 8) in the scale's dtype, then the reference's own multiply by the scale: about
  12.6 bytes a weight;
- "compiled": the reference's own arithmetic through `torch.compile` (Inductor: a C++ loop on the CPU, a Triton kernel
  on CUDA), one graph per weight shape, the unpacker alone: the model stays eager, so every hook of the instrument
  still fires; about 4.6 bytes a weight on a GPU (one read of the words and scales, one write);
- "resident": each linear unpacked once, at install, by the reference unpacker, its bf16 weight kept beside the packed
  one (every reader of `weight_packed` still finds it), and a forward that reads the kept weight under the pinned
  forward's overlay rules: a vault column or row overlay is applied to a clone, and nothing writes the kept weight.

Why the bits are the same. The pieces minus 8 are integers in [-8, 7], exact in bf16, so the only rounding on any route
is the one multiply by the scale; the product of two bf16 values is exact in float32 (8 + 8 significant bits), and each
route rounds it once, to nearest even, back to bf16. That is an argument, not a measurement, so it is tested twice:
scripts/check_dequant.py compares "lut" with the reference on every one of the 16 pieces x 65,536 bf16 scale bit
patterns (NaNs by their bits), and every mode on a 4-bit stand-in (forward, overlays, a pushed answer, a census) byte
for byte on the CPU; and `install` tests "compiled" and "lut" (this on CUDA, whose multiply is another kernel, or when
asked) against the reference on every linear of the model, by bits and strides, and on the whole piece-by-scale table
too, there by bits but for the NaNs' own (Inductor writes a NaN 0x7fc0 where eager torch on the CPU writes 0xffff;
only a NaN or an infinite scale makes one, and no finite checkpoint holds either); and "resident" by each kept weight
against a fresh unpacking and each linear's new forward against the pinned one on a fixed two-row probe, by bits and
strides (one probe's product, not every batch's); and refuses in plain words on any other difference, before anything
is read.

How it is installed. The pinned forward (`_w4a16_forward`), `dequantize_w4a16_rows` (so `slice_w_gate` and
`slice_w_up`) and `get_full_w_down`, `_gate` and `_up` look `dequantize_w4a16_full` up as a module global at call
time, and `atlas_ops.dense_weight` and labkit's gradient linears import it at call time, so "lut" and "compiled"
rebind `engine_runtime.dequantize_w4a16_full` (scripts/check_dequant.py §2 counts each call site). The column slicer
(`dequantize_w4a16_cols`: `slice_w_down`, `write_col_norms`, the vault's edits) has arithmetic of its own, O(K) and
off the forward's path, and is left as it is. A module that imported the name before the install
(experiments/scripts/V6_quiet_dims_01_stats.py) keeps the reference, which gives the same bits. "resident" binds a new
forward on each packed module, and `install(..., "reference")` puts every one back.

What a reading records: `describe()`, under "unpacker" in a census run's meta and an index's meta, and in GET
/api/health.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import types
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import backend.engine_runtime as _er

logger = logging.getLogger("theodolite.w4a16_fast")

MODES: Tuple[str, ...] = ("reference", "lut", "compiled", "resident")
# the server reads its unpacker from here (scripts take --unpacker)
UNPACKER_ENV = "THEODOLITE_UNPACKER"

Unpacker = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# the runtime's own unpacker, as the pinned file defines it (an installed one carries it as `.reference`, so a second
# import of this module still finds it)
REFERENCE: Unpacker = getattr(_er.dequantize_w4a16_full, "reference", _er.dequantize_w4a16_full)
# how many (piece, scale) products the table holds: the 16 pieces twice over (one group of 32) x 65,536 scales
TABLE_PRODUCTS = 32 * 65536


class Refused(RuntimeError):
    """An unpacker that could not be installed, in plain words. The runtime is left on the reference unpacker."""


# ------------------------------------------------------------------------------------------------------- bits, tables


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors hold the same bits: the same shape and dtype and every element's bit pattern, so a NaN
    equals the same NaN and -0 differs from +0 (torch.equal says neither for floats)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(as_int), b.contiguous().view(as_int))
    return torch.equal(a, b)


def same_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
    """`same_bits` and the same strides too: a weight laid out otherwise (the same numbers, transposed in memory)
    sends F.linear to another kernel, whose sums may round otherwise."""
    return same_bits(a, b) and a.stride() == b.stride()


def table_agrees(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, int, int]:
    """Two unpackings of `piece_scale_table` (bf16) compared product by product: (whether they agree: the same bits
    wherever `b`, the reference's, is a number, and a NaN wherever it is NaN; how many numbers differ or NaNs moved;
    how many NaNs carry other bits). The NaNs come from NaN scales and from 0 x an infinite one, which no finite
    checkpoint holds; eager torch on the CPU writes them 0xffff, Inductor 0x7fc0."""
    na, nb = torch.isnan(a), torch.isnan(b)
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    wrong = int((na != nb).sum()) + int(((ai != bi) & ~nb & ~na).sum())
    payload = int(((ai != bi) & na & nb).sum())
    return wrong == 0, wrong, payload


def piece_scale_table(device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Every 4-bit piece beside every bf16 scale bit pattern, as one packed matrix: packed `[65536, 4]` int32 (each row
    the pieces 0..15 twice over, stored as compressed-tensors stores them, nibble j of word k column 8k + j) and scale
    `[65536, 1]` bf16 (row i the bit pattern i - 32768: zeros, subnormals, infinities and NaNs among them), so one
    unpacking writes all 16 x 65,536 products, twice."""
    pieces = torch.arange(16, dtype=torch.int64).repeat(2).view(4, 8)
    words = (pieces << (torch.arange(8, dtype=torch.int64) * 4)).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    packed = words.expand(65536, 4).contiguous()
    scale = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(65536, 1)
    return packed.to(device), scale.contiguous().to(device)


def packed_linears(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """The model's 4-bit linears, as `patch_compressed_tensors_for_low_vram` finds them (weight_packed and
    weight_scale), with their names."""
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "weight_packed") and hasattr(m, "weight_scale")]


# --------------------------------------------------------------------------------------------------------- "lut"

_LUT: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """[256, 2] in `dtype`: byte b -> ((b & 15) - 8, (b >> 4) - 8), a packed byte's two pieces minus 8, the low nibble
    first (exact: integers in [-8, 7])."""
    key = (str(device), dtype)
    t = _LUT.get(key)
    if t is None:
        b = torch.arange(256, dtype=torch.int32)
        t = torch.stack([(b & 15) - 8, (b >> 4) - 8], dim=-1).to(dtype=dtype).to(device)
        _LUT[key] = t
    return t


@torch.no_grad()
def dequantize_w4a16_full_lut(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The reference's weight through a lookup table: the packed words read as bytes (`[out, in // 2]`, byte b of a row
    holding columns 2b and 2b + 1 on a little-endian machine), each byte's two pieces minus 8 looked up (`_lut`), then
    the same multiply by the scale as `engine_runtime.dequantize_w4a16_full`. bfloat16 `[out, in]`."""
    rows = packed.shape[0]
    cols = packed.shape[1] * 8
    w_q = F.embedding(packed.contiguous().view(torch.uint8).to(torch.int32), _lut(packed.device, scale.dtype))
    return (w_q.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, cols)


# ---------------------------------------------------------------------------------------------------- "compiled"

_Key = Tuple[Tuple[int, ...], Tuple[int, ...], str, torch.dtype]
_COMPILED: Dict[_Key, Callable[..., torch.Tensor]] = {}


def _unpack(packed: torch.Tensor, shifts: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # engine_runtime.dequantize_w4a16_full's arithmetic, line for line, with the shifts passed in
    rows = packed.shape[0]
    cols = packed.shape[1] * 8
    w_q = (((packed.unsqueeze(-1) >> shifts) & 0xF) - 8).to(dtype=scale.dtype)
    return (w_q.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, cols)


def _key(packed: torch.Tensor, scale: torch.Tensor) -> _Key:
    return (tuple(packed.shape), tuple(scale.shape), str(packed.device), scale.dtype)


def _compiled_for(key: _Key) -> Callable[..., torch.Tensor]:
    """`_unpack` compiled for one shape. Each shape gets a function of its own (its own code object, named after the
    shape), so each its own compile cache: torch's dynamo keeps one per code object and stops compiling a ninth shape
    of one (`recompile_limit`, 8), and the 31B's linears come in some eight shapes before the table's."""
    fn = _COMPILED.get(key)
    if fn is None:
        name = f"_unpack_{key[0][0]}x{key[0][1] * 8}_{key[3]}".replace("torch.", "")
        code = _unpack.__code__.replace(co_name=name)
        if hasattr(code, "co_qualname"):
            code = code.replace(co_qualname=name)
        fn = torch.compile(types.FunctionType(code, _unpack.__globals__, name), dynamic=False, fullgraph=True)
        _COMPILED[key] = fn
    return fn


@torch.no_grad()
def dequantize_w4a16_full_compiled(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The reference's own arithmetic, compiled for this shape (on first use: seconds on a GPU, longer on the CPU),
    bfloat16 `[out, in]`. Once installed, only the shapes it was tested on at install go through it (`install`)."""
    packed, scale = packed.contiguous(), scale.contiguous()
    return _compiled_for(_key(packed, scale))(packed, _er._get_nibble_shifts(packed.device), scale)


def _installed_compiled(tested: Set[_Key]) -> Unpacker:
    @torch.no_grad()
    def dequantize_w4a16_full_compiled_tested(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """The compiled unpacker for the shapes `install` tested by bits; any other shape (a slice of rows, a weight
        of another model) through the reference. Only the graphs compiled at install run: a call whose guards those
        graphs fail (under torch.inference_mode or autocast, say), which dynamo would compile afresh, untested, goes
        through the reference instead (`fail_on_recompile`), and is counted (`.fell_back`)."""
        key = _key(packed, scale)
        if key in tested and packed.is_contiguous() and scale.is_contiguous():
            try:
                with torch.compiler.set_stance("fail_on_recompile"):
                    return _COMPILED[key](packed, _er._get_nibble_shifts(packed.device), scale)
            except RuntimeError as exc:
                if "recompile" not in str(exc):
                    raise
                dequantize_w4a16_full_compiled_tested.fell_back += 1  # type: ignore[attr-defined]
                if dequantize_w4a16_full_compiled_tested.fell_back == 1:  # type: ignore[attr-defined]
                    lines = [ln.strip() for ln in str(exc).splitlines()]
                    why = next((ln[2:] for ln in lines if ln.startswith("- ")), lines[0] if lines else "")
                    logger.warning("unpacker: a call the compiled graphs were not tested for (%s) goes through the "
                                   "reference unpacker, as will any other", why[:200])
        return REFERENCE(packed, scale)

    dequantize_w4a16_full_compiled_tested.tested = tested  # type: ignore[attr-defined]
    dequantize_w4a16_full_compiled_tested.fell_back = 0  # type: ignore[attr-defined]
    return dequantize_w4a16_full_compiled_tested


# ---------------------------------------------------------------------------------------------------- "resident"


def _make_resident_forward(module: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """A forward over the kept weight with `_w4a16_forward`'s overlay rules, line for line: the vault's column
    overlays (down_proj) and row overlays (gate_proj, up_proj; none is written today) applied to a clone, never to
    the kept weight, which nothing writes in place. Its version counter is checked on every call, which sees any
    in-place op on it or on a view of it (under torch.no_grad too), though not a write through `.data` or straight
    into its storage."""

    def _w4a16_resident_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        w = self._resident_w
        if w._version != self._resident_version:
            raise RuntimeError(
                f"The resident weight of {self._resident_name} was written in place since it was unpacked, so it may no "
                "longer be the checkpoint's: reinstall the unpacker (w4a16_fast.install) before reading anything more."
            )
        col_overlays: Optional[Dict[int, torch.Tensor]] = getattr(self, "_memit_col_overlays", None)
        if col_overlays:
            w = w.clone()
            for col_idx, new_col in col_overlays.items():
                w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
        row_overlays: Optional[Dict[int, torch.Tensor]] = getattr(self, "_memit_row_overlays", None)
        if row_overlays:
            if not col_overlays:
                w = w.clone()
            for row_idx, new_row in row_overlays.items():
                w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)

        bias = getattr(self, "bias", None)
        return F.linear(x.to(dtype=w.dtype), w, bias)

    return types.MethodType(_w4a16_resident_forward, module)


_PROBES: Dict[Tuple[int, str], torch.Tensor] = {}


def _probe(module: nn.Module) -> torch.Tensor:
    """The fixed input `install` passes through a linear's resident and pinned forwards: two rows of N(0, 1) draws
    (seed 0) as wide as the linear reads, in bf16 on its device."""
    key = (int(module.weight_packed.shape[1]) * 8, str(module.weight_packed.device))
    x = _PROBES.get(key)
    if x is None:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, key[0], generator=g).to(torch.bfloat16).to(module.weight_packed.device)
        _PROBES[key] = x
    return x


def free_bytes(device: Any) -> Optional[int]:
    """Bytes free for new tensors on `device`: on CUDA the driver's free memory (torch.cuda.mem_get_info) plus what
    torch's allocator holds unused, on the CPU the kernel's MemAvailable (/proc/meminfo); None where neither can be
    read."""
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free) + int(torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev))
    if dev.type == "cpu":
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def resident_need(linears: Sequence[Tuple[str, nn.Module]]) -> Dict[str, Tuple[int, int]]:
    """Per device, (bytes the kept weights take, bytes the unpacking of the largest needs while it runs: two int32
    temporaries, the pieces and the weight, 12 bytes a weight)."""
    need: Dict[str, Tuple[int, int]] = {}
    for _n, m in linears:
        dev = str(m.weight_packed.device)
        numel = int(m.weight_packed.shape[0]) * int(m.weight_packed.shape[1]) * 8
        kept, peak = need.get(dev, (0, 0))
        need[dev] = (kept + numel * m.weight_scale.element_size(), max(peak, 12 * numel))
    return need


# ----------------------------------------------------------------------------------------------- install, describe

_STATE: Dict[str, Any] = {"mode": "reference"}
_RESIDENT_MODEL: Optional["weakref.ReferenceType[nn.Module]"] = None


def _model_of(rt_or_model: Any) -> nn.Module:
    model = getattr(rt_or_model, "hf_model", None)
    model = rt_or_model if model is None else model
    if not isinstance(model, nn.Module):
        raise Refused("An unpacker is installed on a loaded runtime (its hf_model) or on a model; this is neither.")
    return model


def _uninstall(model: Optional[nn.Module]) -> None:
    """Back to the reference: the runtime's own unpacker, and each resident module's forward as it was, its kept
    weight let go."""
    global _RESIDENT_MODEL
    if _er.dequantize_w4a16_full is not REFERENCE:
        _er.dequantize_w4a16_full = REFERENCE
    models = [model] if model is not None else []
    held = _RESIDENT_MODEL() if _RESIDENT_MODEL is not None else None
    if held is not None and all(held is not m for m in models):
        models.append(held)
    freed = False
    for m_ in models:
        for _n, mod in packed_linears(m_):
            if "_resident_w" in mod.__dict__:
                saved = mod.__dict__.pop("_w4a16_fast_saved_forward", None)
                if saved is not None:
                    mod.forward = saved
                else:
                    mod.__dict__.pop("forward", None)
                for attr in ("_resident_w", "_resident_version", "_resident_name"):
                    mod.__dict__.pop(attr, None)
                freed = True
    _RESIDENT_MODEL = None
    _STATE.clear()
    _STATE["mode"] = "reference"
    if freed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _self_test(linears: Sequence[Tuple[str, nn.Module]], fast: Unpacker, *, table: bool,
               tested: Optional[Set[_Key]] = None) -> Tuple[Dict[str, Any], List[Tuple[str, str]], Optional[str]]:
    """Every linear's whole weight through `fast` against the reference, by bits and strides (`same_layout`; each
    shape that matched added to `tested`), and with `table` the piece-by-scale table on each device the linears are
    on (`table_agrees`: by bits but for the NaNs' own). Returns (the record, the linears that differed with how, how
    the table differed or None)."""
    t0 = time.perf_counter()
    bad: List[Tuple[str, str]] = []
    for name, mod in linears:
        p, s = mod.weight_packed, mod.weight_scale
        got, want = fast(p, s), REFERENCE(p, s)
        if same_layout(got, want):
            if tested is not None:
                tested.add(_key(p.contiguous(), s.contiguous()))
        elif same_bits(got, want):
            bad.append((name, f"the same bits laid out otherwise, strides {tuple(got.stride())} where the reference's "
                              f"are {tuple(want.stride())}"))
        else:
            bad.append((name, "other bits"))
        del got, want
    table_bad: Optional[str] = None
    payloads = 0
    devices = sorted({str(m.weight_packed.device) for _n, m in linears}) or ["cpu"]
    if table:
        for dev in devices:
            p, s = piece_scale_table(dev)
            a, b = fast(p, s), REFERENCE(p, s)
            ok, wrong, payload = table_agrees(a, b)
            payloads += payload
            if not ok:
                na, nb = torch.isnan(a), torch.isnan(b)
                rows = ((na != nb) | ((a.view(torch.int16) != b.view(torch.int16)) & ~nb)).any(-1).nonzero().flatten()
                pats = [f"0x{int(r) ^ 0x8000:04x}" for r in rows[:4].tolist()]
                table_bad = (f"on {dev}, {wrong:,} of its {TABLE_PRODUCTS:,} products differ, at {rows.numel()} of the "
                             f"65,536 bf16 scale patterns ({', '.join(pats)}{', ...' if rows.numel() > 4 else ''})")
                break
            del a, b
    rec = {"linears": len(linears), "same_bits": not bad and table_bad is None,
           "table": f"16 pieces x 65,536 bf16 scales on {', '.join(devices)}" if table else None,
           "table_nan_payloads": payloads if table else None,
           "seconds": round(time.perf_counter() - t0, 2)}
    return rec, bad, table_bad


def install(rt_or_model: Any, mode: str = "reference", *, self_test: Optional[bool] = None,
            log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Install the unpacker `mode` (MODES) on a loaded runtime (or its model), after putting back whatever was
    installed before, and return the record a reading keeps (`describe()`).

    "compiled" is tested first on every 4-bit linear of the model, by bits and strides, against the reference, and
    "lut" is when the model is on CUDA, or when `self_test` is true ("lut" on the CPU is checked exhaustively by
    scripts/check_dequant.py); each, when tested, on the piece-by-scale table too. "resident" checks first that its
    kept weights fit where the model is, then each kept weight against a fresh unpacking and each linear's forward
    against the pinned forward on a fixed probe (`_probe`). Any of these failing raises `Refused`, in plain words,
    with the runtime left on the reference unpacker and nothing read."""
    say = log or logger.info
    model = _model_of(rt_or_model)
    _uninstall(model)
    if mode not in MODES:
        raise Refused(f"The unpackers are {', '.join(MODES)}, not {mode!r}; the runtime keeps the reference unpacker.")
    linears = packed_linears(model)
    t0 = time.perf_counter()
    devices = sorted({str(m.weight_packed.device) for _n, m in linears})
    rec: Dict[str, Any] = {"mode": mode, "linears": len(linears),
                           "shapes": len({_key(m.weight_packed, m.weight_scale) for _n, m in linears}),
                           "devices": devices}
    if mode == "reference":
        _STATE.update(rec)
        return dict(rec)
    if mode == "lut" and sys.byteorder != "little":
        raise Refused("The lut unpacker reads the packed words as bytes, low nibble first, which holds only on a "
                      "little-endian machine; this one is big-endian. Use the reference unpacker.")
    on_cuda = any(d.startswith("cuda") for d in devices)

    if mode in ("lut", "compiled"):
        fast: Unpacker = dequantize_w4a16_full_lut if mode == "lut" else dequantize_w4a16_full_compiled
        test = True if mode == "compiled" else (on_cuda if self_test is None else bool(self_test))
        tested: Set[_Key] = set()
        if test:
            table = all(m.weight_scale.dtype == torch.bfloat16 for _n, m in linears)
            try:
                st, bad, table_bad = _self_test(linears, fast, table=table, tested=tested)
            except Exception as exc:  # a compiler missing, a kernel that would not build: said, not raised as is
                raise Refused(f"The {mode} unpacker could not be run here ({type(exc).__name__}: {exc}); nothing was "
                              "installed, and the runtime keeps the reference unpacker.") from exc
            if bad or table_bad:
                what = []
                if bad:
                    what.append(f"on {len(bad)} of {len(linears)} linears ({bad[0][0]} first: {bad[0][1]})")
                if table_bad:
                    what.append(f"on the piece-by-scale table ({table_bad})")
                raise Refused(f"The {mode} unpacker's weights differ from the reference's {' and '.join(what)}; "
                              "nothing was installed, and the runtime keeps the reference unpacker.")
            rec["self_test"] = st
        else:
            rec["self_test"] = None
        installed: Unpacker = fast if mode == "lut" else _installed_compiled(tested)
        installed.mode = mode  # type: ignore[attr-defined]
        installed.reference = REFERENCE  # type: ignore[attr-defined]
        _er.dequantize_w4a16_full = installed
    else:
        need = resident_need(linears)
        mem: Dict[str, Any] = {}
        for dev, (kept, peak) in need.items():
            free = free_bytes(dev)
            mem[dev] = {"kept_bytes": kept, "unpack_bytes": peak, "free_bytes": free}
            if free is not None and kept + peak > free:
                raise Refused(
                    f"Resident weights need {kept:,} bytes ({_gib(kept)}) on {dev} for {len(linears)} linears, beside "
                    f"the 4-bit ones, and about {_gib(peak)} more while each is unpacked; {free:,} bytes "
                    f"({_gib(free)}) are free there. Use --unpacker lut or compiled, which keep only the 4-bit weights."
                )
        rec["memory"] = mem
        global _RESIDENT_MODEL
        _RESIDENT_MODEL = weakref.ref(model)
        try:
            for name, mod in linears:
                w = REFERENCE(mod.weight_packed, mod.weight_scale)
                mod._w4a16_fast_saved_forward = mod.__dict__.get("forward")
                mod._resident_w = w
                mod._resident_version = w._version
                mod._resident_name = name
                mod.forward = _make_resident_forward(mod)
            # each kept weight against a fresh unpacking (which says only that the unpacker gives what it gave), and
            # what differs from the pinned path, the new forward's product over the kept weight, against the pinned
            # forward's on a fixed probe, by bits and strides, overlays as they stand (none at load)
            t1 = time.perf_counter()
            bad = [name for name, mod in linears
                   if not same_layout(mod._resident_w, REFERENCE(mod.weight_packed, mod.weight_scale))]
            fwd_bad = [name for name, mod in linears
                       if not same_layout(mod.forward(_probe(mod)), _er._make_w4a16_linear_forward(mod)(_probe(mod)))]
        except Exception as exc:
            _uninstall(model)
            raise Refused(f"The resident weights could not be kept ({type(exc).__name__}: {exc}); nothing was "
                          "installed, and the runtime keeps the reference unpacker.") from exc
        if bad or fwd_bad:
            _uninstall(model)
            what = []
            if bad:
                what.append(f"a kept weight differs from a fresh unpacking on {len(bad)} of {len(linears)} linears "
                            f"({bad[0]} first)")
            if fwd_bad:
                what.append(f"the resident forward's output on a probe differs from the pinned forward's on "
                            f"{len(fwd_bad)} of {len(linears)} linears ({fwd_bad[0]} first)")
            raise Refused(f"The resident unpacker failed its test: {' and '.join(what)}; nothing was installed, and the "
                          "runtime keeps the reference unpacker.")
        rec["self_test"] = {"linears": len(linears), "same_bits": True, "table": None,
                            "probe": "2 rows of N(0, 1), seed 0, through each linear's resident and pinned forwards",
                            "seconds": round(time.perf_counter() - t1, 2)}
        rec["resident_bytes"] = sum(kept for kept, _p in need.values())
    rec["install_s"] = round(time.perf_counter() - t0, 2)
    _STATE.clear()
    _STATE.update(rec)
    st = rec.get("self_test")
    if not st:
        tested_how = "not tested here (scripts/check_dequant.py checks it exhaustively on the CPU)"
    elif mode == "resident":
        tested_how = "every kept weight the reference's bits, and every forward the pinned one's on a probe"
    else:
        tested_how = ("the same bits as the reference on every one"
                      + (" and on the piece-by-scale table" if st.get("table") else ""))
    say(f"unpacker: {mode} on {len(linears)} linears ({rec['shapes']} shapes) in {rec['install_s']} s; {tested_how}")
    return dict(rec)


def describe(rt_or_model: Any = None) -> Dict[str, Any]:
    """Which unpacker this process runs, as a reading records it: the record `install` returned, or {"mode":
    "reference"}. It is read from what is bound, so a rebinding put back by hand, or a model whose linears are not
    resident, reads as the reference."""
    fn = _er.dequantize_w4a16_full
    bound = "reference" if fn is REFERENCE else getattr(fn, "mode", "unknown")
    mode = _STATE.get("mode", "reference")
    if bound != "reference":
        actual = bound
    elif mode == "resident":
        model = None if rt_or_model is None else _model_of(rt_or_model)
        actual = "resident" if model is None or any("_resident_w" in m.__dict__ for _n, m in packed_linears(model)) else "reference"
    else:
        actual = "reference"
    return dict(_STATE) if actual == mode else {"mode": actual}
