"""A faster way to unpack a GGUF's packed layers (`backend/gguf_fast.py`), giving the very bits the reference gives
(`gguf_weights.unpack` to float32, then the cast to the layer's dtype: `GGUFLinear.unpacked_reference`), installed
from outside the layer (`install`, which `w4a16_fast.install` hands a model with packed GGUF layers and no W4A16
linears to, so `--unpacker compiled` and $THEODOLITE_UNPACKER=compiled serve a GGUF as they serve the W4A16
checkpoint). The model id and the weights card do not change: they name the weights, and these are the same.

The mode:
- "compiled": the reference's own arithmetic, `unpack(blocks, qtype).to(dtype)`, through `torch.compile` (Inductor:
  a C++ loop on the CPU, Triton kernels on CUDA), one graph per (type, blocks shape, device, dtype), the whole weight
  in one call (no chunks of rows: the fused kernels keep no float32 temporaries, and compiled in chunks it was about
  twice as slow), the unpacker alone: the model stays eager, so every hook of the instrument still fires. On the RTX
  5090 a 21,504 x 5,376 layer of the 31B's types (Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS; random blocks) unpacked in
  0.24-0.46 ms, against the reference's 1.3-3.6 ms (a probe on random blocks, 2 November 2026); on the 31B
  itself a written word-piece took 0.12-0.13 s against the reference's 0.54-0.82 s, the same bits
  (experiments/results/gguf_31b_gpu).
"lut" and "resident" are the checkpoint's, and refused on a GGUF's packed layers in plain words (a GGUF held resident
or dense has none, and keeps the reference, the record saying what was asked: backend/w4a16_fast.py): a lookup table,
compiled, was no faster than "compiled" there (and slower eager), and a GGUF's layers are held resident at load
(THEODOLITE_GGUF_LINEARS=resident, or THEODOLITE_GGUF_RESIDENT_GB for a budget: backend/gguf_model.py).

Why the bits are the same. In every unpacker every product (the fp16 scale, widened exactly to float32, times integer
codes and a sub-block scale) needs at most 23 significant bits, so each is exact in float32; the only roundings are one
add or subtract of the minimum, in float32 (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K), and the cast to the layer's dtype, each
done once on either route, so ordering the products otherwise or fusing a multiply with an add (an FMA) changes no
bit. That is an argument, not a measurement, so it is tested: on CUDA the graphs are compiled with contraction off all
the same (`emulate_precision_casts`, which runs ptxas with --fmad=false; on the CPU Inductor's C++ is compiled with
-ffp-contract=off already), and `install` compares every packed layer's whole weight with the reference's, by bits
and strides, and then, through each graph that serves a layer (Inductor builds each shape's code apart, so one graph's
test says nothing of another's: `table_through`), a table of every one of the 65,536 patterns of each of the type's
fp16 fields (`gguf_weights.F16_FIELDS`) beside random bytes, laid out as that layer's blocks, by bits but for the NaNs'
own (a NaN wherever the reference has one; on CUDA both write 0x7fff, so none differ there, as measured; on the CPU
some graphs write other NaN bits, counted in the record and said in the log line); and refuses in plain words on any
difference, before anything is read. scripts/check_gguf_fast.py checks every type on random blocks and on every fp16
pattern through each graph, where the hook reaches, and readings and a census byte for byte, on the CPU and on cuda:0.

How it is installed. `GGUFLinear.unpacked()` looks the module global `gguf_weights._FAST_UNPACK` up at each call, so
the forward, `dense()` (so `get_full_w_*` and `atlas_ops.dense_weight`), labkit's gradient twin and an int4mm layer
with its kernel turned off see it. The row and column slicers (`unpacked_rows`, `unpacked_cols`: `slice_w_*`, the
vault's edits, `write_col_norms`), `make_resident` and the load (`gguf_model._dense`) stay on `unpack`, as do resident
layers and the layers dense at load (counted in the record). Only the shapes tested at install go through the graphs,
under `fail_on_recompile`: a call they were not tested for (under torch.inference_mode, say) goes through the
reference instead, and is counted (`.fell_back`).

Memory: the graphs, and the unpacked weight alone (2 bytes a weight in bf16) while a layer is unpacked. The self-test
briefly holds one layer's compiled weight beside its reference one and the reference's chunk temporaries (some 12
bytes a weight of 2,048 rows), never two layers' at once: about 1 GB for the 31B's down_proj (estimated); a table
through a layer's graph as much, its blocks laid out as that layer's and unpacked both ways.

What a reading records: `w4a16_fast.describe()` (this module's `describe()` when installed), under "unpacker" in a
census run's meta and an index's meta, and in GET /api/health.
"""

from __future__ import annotations

import logging
import math
import time
import types
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from backend import gguf_weights as gw
from backend.w4a16_fast import Refused, same_bits, same_layout

logger = logging.getLogger("theodolite.gguf_fast")

# the modes of a GGUF's layers (w4a16_fast.MODES names the checkpoint's; lut and resident are refused here)
MODES: Tuple[str, ...] = ("reference", "compiled")
# the patterns one fp16 field holds
F16_PATTERNS = 65536

FastUnpacker = Callable[[gw.GGUFLinear], Optional[torch.Tensor]]


# ------------------------------------------------------------------------------------------------- the layers


def packed_layers(model: nn.Module) -> List[Tuple[str, gw.GGUFLinear]]:
    """The model's packed GGUF layers (a `GGUFLinear` holding its blocks: an `int4_cpu.Int4GGUFLinear` among them),
    with their names: what "compiled" unpacks."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, gw.GGUFLinear) and m.weight_blocks is not None]


def skipped(model: nn.Module) -> Dict[str, int]:
    """The layers "compiled" has nothing to do for, counted: GGUF layers held resident (THEODOLITE_GGUF_LINEARS=resident,
    or within THEODOLITE_GGUF_RESIDENT_GB), and the blocks' plain linears (a GGUF's layers dense at load: the types
    gguf-py reads, F32, F16 and BF16 tensors, or THEODOLITE_GGUF_LINEARS=dense)."""
    resident = sum(1 for m in model.modules() if isinstance(m, gw.GGUFLinear) and m.weight_dense is not None)
    dense = sum(1 for n, m in model.named_modules() if type(m) is nn.Linear and ".layers." in f".{n}.")
    return {"resident": resident, "dense": dense}


# ------------------------------------------------------------------------------------------------- the tables


def f16_table(qtype: str, field: int, seed: int = 0) -> torch.Tensor:
    """65,536 blocks of `qtype` (uint8 [65536, bytes], on the CPU) whose fp16 field `field` (the field-th of
    `gguf_weights.F16_FIELDS[qtype]`) runs through every pattern, block i holding pattern i (zeros, subnormals,
    infinities and NaNs among them), every other byte random and the type's other fp16 fields finite (seeded)."""
    _bs, ts = gw.block_shape(qtype)
    g = torch.Generator().manual_seed(seed * 1009 + field)
    b = torch.randint(0, 256, (F16_PATTERNS, ts), generator=g, dtype=torch.int64)
    for off in gw.F16_FIELDS[qtype]:
        u = torch.randint(0, 65536, (F16_PATTERNS,), generator=g, dtype=torch.int64)
        u = torch.where(((u >> 10) & 0x1F) == 0x1F, u ^ 0x4000, u)  # an infinity's or a NaN's exponent made finite
        b[:, off], b[:, off + 1] = u & 0xFF, u >> 8
    off = gw.F16_FIELDS[qtype][field]
    i = torch.arange(F16_PATTERNS, dtype=torch.int64)
    b[:, off], b[:, off + 1] = i & 0xFF, i >> 8  # little-endian, as the file holds it
    return b.to(torch.uint8)


def table_agrees(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, int, int]:
    """Two unpackings of a table compared number by number, in any float dtype (`w4a16_fast.table_agrees` is bf16's):
    (whether they agree: the same bits wherever `b`, the reference's, is a number, and a NaN wherever it is NaN; how
    many numbers differ or NaNs moved; how many NaNs carry other bits). The NaNs come from NaN or infinite fp16
    fields, which no finite file holds; eager torch on the CPU writes them 0xffff in bf16."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, int(b.numel()), 0
    as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    na, nb = torch.isnan(a), torch.isnan(b)
    ai, bi = a.contiguous().view(as_int), b.contiguous().view(as_int)
    wrong = int((na != nb).sum()) + int(((ai != bi) & ~nb & ~na).sum())
    payload = int(((ai != bi) & na & nb).sum())
    return wrong == 0, wrong, payload


# ---------------------------------------------------------------------------------------------------- "compiled"

_Key = Tuple[str, Tuple[int, ...], str, torch.dtype]
_COMPILED: Dict[_Key, Callable[..., torch.Tensor]] = {}


def _unpack(blocks: torch.Tensor, qtype: str, dtype: torch.dtype) -> torch.Tensor:
    # GGUFLinear.unpacked_reference()'s numbers in one call: gguf_weights.unpack's float32, then the one cast
    return gw.unpack(blocks, qtype).to(dtype)


def _key(blocks: torch.Tensor, qtype: gw.QType, dtype: torch.dtype) -> _Key:
    return (gw.type_name(gw.type_code(qtype)), tuple(blocks.shape), str(blocks.device), dtype)


def options(device: str) -> Optional[Dict[str, Any]]:
    """Inductor's options for a graph on `device`, passed to that compile alone (no global config is set): on CUDA
    contraction off (`emulate_precision_casts`: Triton then runs ptxas with --fmad=false), so the bits rest on no
    argument; on the CPU none (Inductor's C++ is compiled with -ffp-contract=off)."""
    return {"emulate_precision_casts": True} if device.startswith("cuda") else None


def _compiled_for(key: _Key) -> Callable[..., torch.Tensor]:
    """`_unpack` compiled for one (type, blocks shape, device, dtype). Each gets a function of its own (its own code
    object, named after it), so each its own compile cache: dynamo keeps one per code object and stops compiling a
    ninth shape of one (`recompile_limit`, 8, past which it runs eager, silently), and the 31B has eight whole-layer
    shapes of a type. The unpacker's small tables are made on the device first (one eager unpacking of a block), so
    the graph reads them as they are."""
    fn = _COMPILED.get(key)
    if fn is None:
        qt, shape, dev, dt = key
        bs, ts = gw.block_shape(qt)
        gw.unpack(torch.zeros(1, ts, dtype=torch.uint8, device=dev), qt)  # _shifts and the IQ4 levels, on the device
        name = f"_unpack_{qt}_{shape[0]}x{shape[-1] // ts * bs}_{dev}_{dt}".replace("torch.", "").replace(":", "")
        code = _unpack.__code__.replace(co_name=name)
        if hasattr(code, "co_qualname"):
            code = code.replace(co_qualname=name)
        fn = torch.compile(types.FunctionType(code, _unpack.__globals__, name), dynamic=False, fullgraph=True,
                           options=options(dev))
        _COMPILED[key] = fn
    return fn


@torch.no_grad()
def unpack_compiled(blocks: torch.Tensor, qtype: gw.QType, dtype: torch.dtype) -> torch.Tensor:
    """`gguf_weights.unpack(blocks, qtype).to(dtype)` compiled for this type, shape, device and dtype (on first use:
    about a second on a GPU, a few on the CPU), [..., row elements] contiguous. Once installed, only the layers' shapes
    tested at install go through it (`install`)."""
    blocks = blocks.contiguous()
    key = _key(blocks, qtype, dtype)
    return _compiled_for(key)(blocks, key[0], dtype)


def _differing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The rows of two unpackings [n, block] where `table_agrees` finds them differing (a number's bits, or a NaN
    moved), as indices."""
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    na, nb = torch.isnan(a), torch.isnan(b)
    return ((na != nb) | ((a.view(bits) != b.view(bits)) & ~nb)).any(-1).nonzero().flatten()


def table_through(key: _Key, field: int, seed: int = 0) -> Tuple[bool, int, int, List[int]]:
    """The table of every pattern of the fp16 field `field` of the key's type (`f16_table`) through the very graph that
    serves the key's layers: its blocks laid out as theirs ([rows, row bytes], as many calls as it takes, the last
    filled from the table's start), under `fail_on_recompile`, so no other graph is built for it; each call against the
    reference as a layer of those blocks unpacks itself (`GGUFLinear.unpacked_reference`: eager torch's NaN bits too
    depend on the shape), by `table_agrees`. Returns (whether they agree, how many numbers differ or NaNs moved, how
    many NaNs carry other bits, the patterns that differ)."""
    qt, shape, dev, dt = key
    bs, ts = gw.block_shape(qt)
    rows, in_f = shape[0], shape[-1] // ts * bs
    table = f16_table(qt, field, seed).to(dev)
    nb = math.prod(shape) // ts
    ok, wrong, payload = True, 0, 0
    bad = torch.zeros(F16_PATTERNS, dtype=torch.bool, device=dev)
    with torch.compiler.set_stance("fail_on_recompile"):
        for s in range(0, F16_PATTERNS, nb):
            idx = torch.arange(s, s + nb, device=dev) % F16_PATTERNS  # block i of the table holds pattern i
            tile = table[idx].reshape(shape)
            got = unpack_compiled(tile, qt, dt)
            want = gw.GGUFLinear(tile, qt, in_f, rows, dtype=dt).unpacked_reference()
            o, w, pl = table_agrees(got, want)
            ok, wrong, payload = ok and o, wrong + w, payload + pl
            if not o:
                bad[idx[_differing(got.reshape(nb, bs), want.reshape(nb, bs))]] = True
            del tile, got, want
    return ok, wrong, payload, bad.nonzero().flatten().tolist()


def _installed_compiled(tested: Set[_Key]) -> FastUnpacker:
    @torch.no_grad()
    def unpack_compiled_tested(mod: gw.GGUFLinear) -> Optional[torch.Tensor]:
        """A packed layer's weight through the graph compiled and tested by bits for its shape at install; None (so
        `GGUFLinear.unpacked()` takes the reference) for any other: another shape or dtype, blocks not contiguous, or a
        call whose guards that graph fails (under torch.inference_mode, say), which dynamo would compile afresh,
        untested (`fail_on_recompile`), counted (`.fell_back`)."""
        blocks = mod.weight_blocks
        if blocks is None or not blocks.is_contiguous():
            return None
        key = (mod.qtype, tuple(blocks.shape), str(blocks.device), mod.compute_dtype)
        if key not in tested:
            return None
        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                return _COMPILED[key](blocks, key[0], key[3])
        except RuntimeError as exc:
            if "recompile" not in str(exc):
                raise
            unpack_compiled_tested.fell_back += 1  # type: ignore[attr-defined]
            if unpack_compiled_tested.fell_back == 1:  # type: ignore[attr-defined]
                lines = [ln.strip() for ln in str(exc).splitlines()]
                why = next((ln[2:] for ln in lines if ln.startswith("- ")), lines[0] if lines else "")
                logger.warning("unpacker: a GGUF layer's call the compiled graphs were not tested for (%s) goes through "
                               "the reference unpacker, as will any other", why[:200])
        return None

    unpack_compiled_tested.tested = tested  # type: ignore[attr-defined]
    unpack_compiled_tested.fell_back = 0  # type: ignore[attr-defined]
    return unpack_compiled_tested


# ----------------------------------------------------------------------------------------------- install, describe

_STATE: Dict[str, Any] = {"mode": "reference"}


def uninstall() -> None:
    """Back to the reference: no faster unpacker bound in `gguf_weights` (the graphs kept, for a reinstall)."""
    gw._FAST_UNPACK = None
    _STATE.clear()
    _STATE["mode"] = "reference"


def installed() -> bool:
    """Whether a faster unpacker is bound for GGUF layers."""
    return gw._FAST_UNPACK is not None


def _self_test(layers: List[Tuple[str, gw.GGUFLinear]], tested: Set[_Key]
               ) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]], Dict[str, str]]:
    """Every packed layer's whole weight through `unpack_compiled` against the reference (`unpacked_reference`), by
    bits and strides (`same_layout`; each key that matched added to `tested`), one layer at a time; then through each
    graph tested so, the table of every pattern of each of its type's fp16 fields laid out as its layers' blocks
    (`table_through`), against the reference such a layer unpacks, by `table_agrees`. Returns (the record, the layers
    that differed with their type and how, how each type's table differed)."""
    t0 = time.perf_counter()
    bad: List[Tuple[str, str, str]] = []
    for name, mod in layers:
        got, want = unpack_compiled(mod.weight_blocks, mod.qtype, mod.compute_dtype), mod.unpacked_reference()
        if same_layout(got, want):
            tested.add(_key(mod.weight_blocks, mod.qtype, mod.compute_dtype))
        elif same_bits(got, want):
            bad.append((name, mod.qtype, f"the same bits laid out otherwise, strides {tuple(got.stride())} where the "
                                         f"reference's are {tuple(want.stride())}"))
        else:
            ok, _wrong, payload = table_agrees(got, want)
            bad.append((name, mod.qtype, "other bits" if not ok else
                        f"other bits, though only in {payload:,} NaNs' own (its fp16 scales hold infinities or NaNs, "
                        "which no finite file holds, and the compiled graph writes their NaNs otherwise than the "
                        "reference here)"))
        del got, want
    table: Dict[str, Dict[str, Any]] = {}
    table_bad: Dict[str, str] = {}
    for key in sorted(tested, key=str):  # each graph that will serve a layer, the table through it
        qt, shape, dev, dt = key
        bs, ts = gw.block_shape(qt)
        on = f"{dev}/{str(dt).replace('torch.', '')}"
        fields = gw.F16_FIELDS[qt]
        rec = table.setdefault(qt, {"fields": list(fields), "patterns": F16_PATTERNS * len(fields), "graphs": 0,
                                    "nan_payloads": 0, "on": []})
        rec["graphs"] += 1
        if on not in rec["on"]:
            rec["on"].append(on)
        for k, off in enumerate(fields):
            ok, wrong, payload, pats = table_through(key, k)
            rec["nan_payloads"] += payload
            if not ok and qt not in table_bad:
                shown = ", ".join(f"0x{p:04x}" for p in pats[:4]) + (", ..." if len(pats) > 4 else "")
                table_bad[qt] = (f"on {on}, through the graph of its [{shape[0]}, {shape[-1] // ts * bs}] layers, the fp16 "
                                 f"field at byte {off}: {wrong:,} numbers differ, at {len(pats)} of its {F16_PATTERNS:,} "
                                 f"patterns ({shown})")
    rec_ = {"linears": len(layers), "same_bits": not bad and not table_bad, "table": table,
            "seconds": round(time.perf_counter() - t0, 2)}
    return rec_, bad, table_bad


def install(model: nn.Module, mode: str, *, self_test: Optional[bool] = None,
            log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Install the unpacker `mode` on a model's packed GGUF layers (`w4a16_fast.install` hands a model with some and
    no W4A16 linears to this), after putting back whatever was installed before, and return the record a reading
    keeps (`describe()`).

    "compiled" is tested first, whatever `self_test` says, on every packed layer by bits and strides against the
    reference, and on the table of every fp16 pattern of each type's fields through each graph that serves a layer
    (`_self_test`); only then is it bound. A difference, an exception or a compile that fails raises `Refused`, in plain words (how many layers and
    which types differ), with the reference left bound and nothing read. "lut" and "resident" are refused, naming what
    to use instead; "reference" unbinds."""
    say = log or logger.info
    uninstall()
    if mode not in MODES + ("lut", "resident"):
        raise Refused(f"The unpackers of a GGUF's layers are {', '.join(MODES)}, not {mode!r}; the runtime keeps the "
                      "reference unpacker.")
    layers = packed_layers(model)
    t0 = time.perf_counter()
    kinds = Counter(m.qtype for _n, m in layers)
    rec: Dict[str, Any] = {"mode": mode, "weights": "gguf", "linears": len(layers), "types": dict(sorted(kinds.items())),
                           "shapes": len({_key(m.weight_blocks, m.qtype, m.compute_dtype) for _n, m in layers}),
                           "devices": sorted({str(m.weight_blocks.device) for _n, m in layers}),
                           "int4mm": sum(1 for _n, m in layers if getattr(m, "int4mm", False)),
                           "skipped": skipped(model)}
    if mode == "reference":
        _STATE.update(rec)
        return dict(rec)
    if mode == "lut":
        raise Refused("A GGUF's layers have no lut unpacker: on a GPU it is no faster than compiled (measured on the RTX "
                      "5090); use compiled. Nothing was installed, and the runtime keeps the reference unpacker.")
    if mode == "resident":
        raise Refused("A GGUF's layers are held resident at load: THEODOLITE_GGUF_LINEARS=resident, or "
                      "THEODOLITE_GGUF_RESIDENT_GB for a budget (the resident unpacker is the W4A16 checkpoint's). "
                      "Nothing was installed, and the runtime keeps the reference unpacker.")
    tested: Set[_Key] = set()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="bf16 and fp16 are mixed")  # Inductor's C++, on the fp16 fields
            st, bad, table_bad = _self_test(layers, tested)
    except Exception as exc:  # a compiler missing, a kernel that would not build: said, not raised as is
        raise Refused(f"The compiled unpacker could not be run on this GGUF's layers here ({type(exc).__name__}: {exc}); "
                      "nothing was installed, and the runtime keeps the reference unpacker.") from exc
    if bad or table_bad:
        what = []
        if bad:
            types_bad = ", ".join(sorted({qt for _n, qt, _w in bad}))
            what.append(f"on {len(bad)} of {len(layers)} GGUF layers, of type {types_bad} ({bad[0][0]} first: {bad[0][2]})")
        if table_bad:
            first = sorted(table_bad)[0]
            what.append(f"on the table of every fp16 pattern of {', '.join(sorted(table_bad))} "
                        f"({first} {table_bad[first]})")
        raise Refused(f"The compiled unpacker's weights differ from the reference's {' and '.join(what)}; nothing was "
                      "installed, and the runtime keeps the reference unpacker.")
    rec["self_test"] = st
    fast = _installed_compiled(tested)
    fast.mode = mode  # type: ignore[attr-defined]
    gw._FAST_UNPACK = fast
    rec["install_s"] = round(time.perf_counter() - t0, 2)
    _STATE.clear()
    _STATE.update(rec)
    left = rec["skipped"]
    also = f"; left as they are: {left['resident']} resident, {left['dense']} dense at load" if any(left.values()) else ""
    moved = sum(t["nan_payloads"] for t in st["table"].values())
    nans = (f", but for {moved:,} NaNs' own bits there (from infinite or NaN fp16 fields, which no finite file holds)"
            if moved else "")
    say(f"unpacker: compiled on {len(layers)} GGUF layers ({', '.join(f'{n} {qt}' for qt, n in rec['types'].items())}; "
        f"{rec['shapes']} shapes) in {rec['install_s']} s; the same bits as the reference on every one, and on every fp16 "
        f"pattern of each type's fields through each of the {rec['shapes']} graphs{nans}{also}")
    return dict(rec)


def describe() -> Dict[str, Any]:
    """Which unpacker GGUF layers run, as a reading records it: the record `install` returned, or {"mode":
    "reference"}; read from what is bound, so a hook put back by hand reads as the reference."""
    fn = gw._FAST_UNPACK
    bound = "reference" if fn is None else getattr(fn, "mode", "unknown")
    return dict(_STATE) if bound == _STATE.get("mode", "reference") else {"mode": bound}
