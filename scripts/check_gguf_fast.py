#!/usr/bin/env python3
"""A check of the faster unpacker of a GGUF's packed layers (backend/gguf_fast.py: --unpacker compiled, through
backend/w4a16_fast.py's install): that it gives the very bits the reference gives (`gguf_weights.unpack` to float32,
then the cast to the layer's dtype), and that every reading through it is the same, byte for byte. It runs on the CPU,
the GPU hidden, unless given --cuda, when the device sections run on cuda:0 too; it loads no real model unless given
--gguf. It checks:
1. every type compiled against the reference, by bits and strides: random blocks (their fp16 fields any finite
   pattern, subnormals, negatives and +-0 among them, or of a real file's sizes) as whole weights of three shapes (one
   of more rows than a layer unpacks at once, its last chunk one row), in bf16 and float32, and through each shape's
   own graph the table of every pattern of each fp16 field beside random bytes (NaNs by where they are, their own bits
   counted apart: none on CUDA), each (type, shape, device, dtype) a graph of its own, the tables building none, each
   compile given contraction off on CUDA (emulate_precision_casts, as torch.compile receives it) and no options on the
   CPU; and the table of fp16 fields itself (gguf_weights.F16_FIELDS: scripts/check_gguf.py's and int4_cpu's the same,
   each field one whose NaN makes every weight NaN, and no other two bytes so), and f16_table sweeping the field asked,
   the type's others finite;
2. where the hook reaches, compiled installed: GGUF layers of several types, their forward bare and under a column
   overlay, a row overlay and both; on a stand-in GGUFRuntime, dense, get_full_w_*, atlas_ops.dense_weight and
   labkit's gradient twin through the compiled graphs (counted), dense_cols and dense_rows through the reference, all
   byte for byte the reference's; a call under torch.inference_mode through the reference, counted, no graph compiled
   after install; a layer of a shape not tested at install (a graph built for it beside the install, then a wrong one)
   and one whose blocks are a strided view, both through the reference, served by no graph; layers held resident (THEODOLITE_GGUF_LINEARS=resident, a THEODOLITE_GGUF_RESIDENT_GB budget) and
   layers dense at load (a type gguf-py reads, or linears dense) left as they are and counted; an int4mm model on the
   CPU: the kernel's product unchanged, and with the kernel off the reference's bits;
3. readings byte for byte, compiled against the reference, on the stand-in four times as wide written as Q4_K (its
   Q5_0 fallbacks among its layers), in bf16: logits, three answers written at once under their own pushes, a vault
   UPDATE heard and seen and a ROLLBACK, and a census run's rows, its meta recording {"mode": "compiled", "weights":
   "gguf", ...} under the same model id, the weights card unchanged;
4. refusals in plain words, the reference left bound: a compiled unpacker one bit off on one layer, one off on a type's
   table alone (through a layer's graph), lut and resident on a GGUF, an unknown unpacker; installed twice, and the reference after compiled
   unbinding it; a dense model's record as before ("No W4A16 linears", what was asked); the W4A16 checkpoint's
   compiled record with the keys it had;
5. the command lines and the server: census_run.py --unpacker compiled on a GGUF records the GGUF's record in its meta,
   its rows the reference's, and --unpacker lut is refused there; the server's lifespan with THEODOLITE_GGUF and
   $THEODOLITE_UNPACKER=compiled shows it in /api/health, the health keys' order kept; labd's /health and the
   standalone scientist install it too, the scientist answering a tool call with a refusal's words as the call's error
   (not ending its server); pod_tool.py steps records what ran beside what was asked and chooses the fastest by it, on
   a GGUF, and on the dense stand-in, where lut and compiled run as the reference, choosing none; pod_tool.py
   selftest names the GGUF's own table test; verify_control_vector.py records install's record (its code read: it
   runs only beside a census of the 31B), and its instrument step on a GGUF refuses lut in plain words, a name not an
   unpacker before anything loads;
6. with --gguf FILE, a file loaded packed, once (on cuda:0 with --cuda, else on the CPU; with --cuda and no GPU, or
   with nothing compiling here, not loaded at all, and the run fails saying so): compiled installed (its record), one
   forward's logits on a fixed prompt the reference's bits, that forward and the timed ones through the compiled graphs
   (counted, none sent to the reference), one row's forward timed each way after a warm-up, the memory;
7. the files R4 pins unchanged (R4_common.pinned_differ), and w4a16_fast's checkpoint-facing code as at the volunteer
   kit's tag (census-kit-1: every function and constant but install, _uninstall and describe, which hand a GGUF on,
   its REFERENCE the pinned unpacker); importing it loads no GGUF module and sets no compiler config, and a GGUF's
   install sets none either.

"compiled" on the CPU goes through Inductor's C++ compiler: a child process tries one tiny compile first (240 s at
most), and when that fails the sections that compile on the CPU are left out, with the reason said.

Run: .venv/bin/python -I scripts/check_gguf_fast.py [--cuda] [--gguf FILE]   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    if "--cuda" not in sys.argv:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")

import ast
import asyncio
import contextlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn as nn

import atlas_stand_in as asi
import check_gguf as cg  # its GGUF stand-ins and the fp16 fields' table
from backend import atlas_ops as ao
from backend import census as cz
from backend import engine_runtime as er
from backend import gguf_fast as gf
from backend import gguf_model as gm
from backend import gguf_weights as gw
from backend import int4_cpu
from backend import larql_vault
from backend import runtime_factory as rf
from backend import w4a16_fast as wf
from backend.causal_check import language_model
from backend.gguf_runtime import GGUFRuntime
from backend.larql_vault import LarqlVaultEngine

warnings.filterwarnings("ignore", message="bf16 and fp16 are mixed")  # Inductor's C++ on the fp16 fields: a note

failures: List[str] = []
passed = 0
BASE = "http://127.0.0.1:8765"  # the server's guards refuse TestClient's own "testserver" Host
TYPES = list(gw.F16_FIELDS)
ENV_KEYS = (rf.GGUF_ENV, rf.DEVICE_ENV, rf.LINEARS_ENV, rf.CONFIG_ENV, rf.DTYPE_ENV, rf.LENS_ENV, rf.RESIDENT_ENV,
            rf.CHECKPOINT_LINEARS_ENV, wf.UNPACKER_ENV)


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def quiet(*_a: Any, **_k: Any) -> None:
    pass


@contextlib.contextmanager
def environment(**values: Any) -> Iterator[None]:
    """The runtime's variables (and $THEODOLITE_UNPACKER) set as given inside the block, none else; as they were after."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in values.items():
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def gguf_env(path: Path, dtype: str = "float32", **more: Any) -> Dict[str, Any]:
    """A GGUF on the CPU without the lens."""
    return {rf.GGUF_ENV: path, rf.DEVICE_ENV: "cpu", rf.DTYPE_ENV: dtype, rf.LENS_ENV: "none", **more}


def compile_works() -> Tuple[bool, str]:
    """Whether torch.compile (Inductor, a C++ compiler on the CPU) builds a tiny GGUF unpacker here: tried in a child
    process given 240 s (as scripts/check_dequant.py tries the W4A16 one)."""
    code = ("import torch, sys; sys.path.insert(0, sys.argv[1]); from backend import gguf_fast as gf, gguf_weights as gw; "
            "b = torch.zeros(2, 18, dtype=torch.uint8); b[:, 1] = 0x3c; "
            "print(torch.equal(gf.unpack_compiled(b, 'Q4_0', torch.bfloat16), gw.unpack(b, 'Q4_0').to(torch.bfloat16)))")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT_DIR)], capture_output=True, text=True, timeout=240,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    except subprocess.TimeoutExpired:
        return False, "a tiny compile took more than 240 s here"
    if r.returncode != 0 or r.stdout.strip() != "True":
        tail = (r.stderr.strip().splitlines() or [r.stdout.strip()])[-1][:300]
        return False, f"a tiny compile failed here ({tail})"
    return True, f"a tiny compile took {time.time() - t0:.0f} s"


# ------------------------------------------------------------------------------------------------ random blocks


def f16_bits(n: int, g: torch.Generator, dist: str) -> torch.Tensor:
    """n fp16 bit patterns (int64 in 0..65535): "wide", any finite pattern (an infinity's or a NaN's exponent made
    finite), +-0, the smallest and largest subnormals and the largest number first; "real", |d| log-uniform in 1e-3 to
    1e-1, either sign, as a quantised file's scales are."""
    if dist == "wide":
        u = torch.randint(0, 65536, (n,), generator=g, dtype=torch.int64)
        u = torch.where(((u >> 10) & 0x1F) == 0x1F, u ^ 0x4000, u)
        first = torch.tensor([0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x7BFF, 0xFBFF], dtype=torch.int64)
        u[: min(n, 8)] = first[: min(n, 8)]
        return u
    mag = 10 ** (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 3)
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    return (mag * sign).to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF


def random_blocks(qtype: str, rows: int, in_f: int, seed: int, dist: str = "mixed") -> torch.Tensor:
    """A [rows, in_f] weight of `qtype` as blocks (uint8 [rows, row bytes]): random bytes, each fp16 field "wide" or
    "real" (`f16_bits`), or block by block one then the other ("mixed")."""
    bs, ts = gw.block_shape(qtype)
    n = rows * in_f // bs
    g = torch.Generator().manual_seed(seed)
    b = torch.randint(0, 256, (n, ts), generator=g, dtype=torch.int64)
    for off in gw.F16_FIELDS[qtype]:
        if dist == "mixed":
            u = torch.where(torch.arange(n) % 2 == 0, f16_bits(n, g, "wide"), f16_bits(n, g, "real"))
        else:
            u = f16_bits(n, g, dist)
        b[:, off], b[:, off + 1] = u & 0xFF, u >> 8
    return b.to(torch.uint8).reshape(rows, -1).contiguous()


class Counting:
    """The installed GGUF unpacker wrapped to count its calls (`calls`) and those it served (`served`, not None)."""

    def __enter__(self) -> "Counting":
        self.inner = gw._FAST_UNPACK
        self.calls = self.served = 0

        def counting(mod: gw.GGUFLinear) -> Optional[torch.Tensor]:
            self.calls += 1
            w = self.inner(mod) if self.inner is not None else None
            self.served += w is not None
            return w

        counting.mode = getattr(self.inner, "mode", None)  # type: ignore[attr-defined]
        gw._FAST_UNPACK = counting
        return self

    def __exit__(self, *_exc: Any) -> None:
        gw._FAST_UNPACK = self.inner


def graphs() -> int:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def give_lens(rt: Any) -> None:
    """A stand-in J++ lens on the runtime, as scripts/atlas_stand_in.py gives its own (the vault's UPDATE pulls its
    direction back through it)."""
    lm = language_model(rt.hf_model)
    d = lm.norm.weight.shape[0]
    g = torch.Generator().manual_seed(5)
    for l in range(len(lm.layers)):
        rt.jpp_lenses[l] = (torch.eye(d) + 0.3 * torch.randn(d, d, generator=g)).to(torch.bfloat16).to(rt.device)


def load_labkit() -> Any:
    spec = importlib.util.spec_from_file_location("labkit_fast_check", ROOT_DIR / "experiments" / "lab" / "labkit.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- 1. every type against the reference


SHAPES = [(3, 256), (16, 1024), (2049, 256)]  # the last: more rows than chunk_rows (2048), the reference's last chunk one row
DTYPES = (torch.bfloat16, torch.float32)


def check_fields() -> None:
    before = len(failures)
    check(gw.F16_FIELDS == {k: tuple(v) for k, v in cg.F16_FIELDS.items()} and list(gw.F16_FIELDS) == list(cg.F16_FIELDS)
          and gw.F16_FIELDS == {k: tuple(v) for k, v in int4_cpu._F16_AT.items()} and set(TYPES) == {
              n for c, (n, bs, _t) in gw.GGML_TYPES.items() if bs > 1},
          "gguf_weights.F16_FIELDS: every block type, the same as scripts/check_gguf.py's and int4_cpu._F16_AT")
    for qt in TYPES:
        _bs, ts = gw.block_shape(qt)
        base = random_blocks(qt, 4, gw.block_shape(qt)[0] * 2, 11, "real")  # [4, 2 blocks]
        base = base.reshape(-1, ts)
        fields = gw.F16_FIELDS[qt]
        nan_all = []
        for off in range(ts - 1):
            b = base.clone()
            b[:, off], b[:, off + 1] = 0x00, 0x7E  # fp16 NaN 0x7e00
            nan_all.append(bool(torch.isnan(gw.unpack(b, qt)).all()))
        claimed = [nan_all[off] for off in fields]
        others = [off for off in range(ts - 1) if nan_all[off] and off not in fields]
        check(all(claimed) and not others and not torch.isnan(gw.unpack(base, qt)).any(),
              f"{qt}: a NaN in each of its fp16 fields {list(fields)} makes every weight NaN, in no other two bytes ({others})")
        for k, off in enumerate(fields):
            t = gf.f16_table(qt, k).long()
            u = {o: t[:, o] | (t[:, o + 1] << 8) for o in fields}
            finite = all(bool((((u[o] >> 10) & 0x1F) != 0x1F).all()) for o in fields if o != off)
            check(torch.equal(u[off], torch.arange(gf.F16_PATTERNS)) and finite
                  and bool(torch.isnan(gw.unpack(t.to(torch.uint8), qt)).any()),
                  f"{qt}: f16_table({k}) runs its field at byte {off} through every pattern, block i holding pattern i, "
                  "its other fp16 fields finite")
    section("the table of fp16 fields: one, the unpackers' own, every field one whose NaN reaches every weight; "
            "f16_table sweeping the field asked", before)


def check_types(devices: List[str]) -> None:
    before = len(failures)
    n0, g0 = len(gf._COMPILED), graphs()
    keys = set()
    payloads: Dict[str, int] = {}
    compiles: List[Tuple[str, Any, Any, Any]] = []
    true_compile = torch.compile

    def spy(fn: Any, *a: Any, **k: Any) -> Any:  # what each graph's compile is given, as torch.compile receives it
        compiles.append((getattr(fn, "__name__", ""), k.get("options"), k.get("dynamic"), k.get("fullgraph")))
        return true_compile(fn, *a, **k)

    torch.compile = spy  # type: ignore[assignment]
    t0 = time.time()
    try:
        for dev in devices:
            for dt in DTYPES:
                on = f"{dev}/{str(dt).replace('torch.', '')}"
                for i, qt in enumerate(TYPES):
                    for j, (rows, in_f) in enumerate(SHAPES):
                        b = random_blocks(qt, rows, in_f, 100 * i + j).to(dev)
                        got = gf.unpack_compiled(b, qt, dt)
                        want = gw.unpack(b, qt).to(dt)
                        lay = gw.GGUFLinear(b, qt, in_f, rows, dtype=dt).unpacked_reference()
                        key = gf._key(b, qt, dt)
                        keys.add(key)
                        check(wf.same_layout(got, want) and wf.same_layout(got, lay) and bool(torch.isfinite(want).all()),
                              f"{qt} [{rows}, {in_f}] on {dev} in {dt}: the reference's bits and strides, whole and as the "
                              f"layer unpacks it ({int((got.float() != want.float()).sum())} differ)")
                        del b, got, want, lay
                        for k, off in enumerate(gw.F16_FIELDS[qt]):  # through this shape's own graph, as install tests it
                            ok, wrong, payload, pats = gf.table_through(key, k, seed=7)
                            payloads[on] = payloads.get(on, 0) + payload
                            check(ok and (payload == 0 or not dev.startswith("cuda")),
                                  f"{qt} [{rows}, {in_f}] on {dev} in {dt}: every fp16 pattern of its field at byte {off} "
                                  f"through this shape's graph, the reference's numbers and NaNs ({wrong} differ, at "
                                  f"{len(pats)} patterns; {payload} NaNs with other bits)")
                print(f"        {on}: {len(TYPES)} types done ({time.time() - t0:.0f} s)")
    finally:
        torch.compile = true_compile  # type: ignore[assignment]
    new_keys = len(gf._COMPILED) - n0
    check(new_keys == len(keys) and graphs() - g0 == new_keys,
          f"each (type, shape, device, dtype) its own graph, the tables through them building none: {len(keys)} keys, "
          f"{new_keys} functions, {graphs() - g0} graphs")
    off = [c for c in compiles if c[1] != ({"emulate_precision_casts": True} if "_cuda" in c[0] else None)
           or c[2] is not False or c[3] is not True]
    check(len(compiles) == new_keys and not off,
          f"each graph compiled with contraction off on CUDA (emulate_precision_casts) and no options on the CPU, "
          f"dynamic=False, fullgraph=True, as torch.compile was given them ({len(compiles)} compiles; {off[:2]})")
    print("        NaNs whose own bits differ, the rest the same: "
          + ", ".join(f"{k} {v:,}" for k, v in payloads.items())
          + " (on the CPU eager torch and Inductor's C++ write some NaNs otherwise, by shape and dtype)")
    section(f"every type compiled against the reference on {', '.join(devices)}, in bf16 and float32: random blocks of "
            f"{len(SHAPES)} shapes by bits and strides, every pattern of every fp16 field through each shape's graph, "
            "the compile options as said", before)


# ------------------------------------------------------------------------------------------- 2. the hook's reach


def layer_holder(dev: str, dtype: torch.dtype) -> nn.Module:
    """GGUF layers of several types and shapes (real-sized scales), the first with a bias."""
    holder = nn.Module()
    holder.lins = nn.ModuleList()
    for i, (qt, rows, in_f) in enumerate((("Q4_0", 40, 64), ("Q4_K", 24, 512), ("Q6_K", 2050, 256), ("IQ4_XS", 8, 256),
                                          ("Q5_K", 33, 256), ("Q8_0", 5, 96))):
        bias = torch.randn(rows, generator=torch.Generator().manual_seed(i)) if i == 0 else None
        holder.lins.append(gw.GGUFLinear(random_blocks(qt, rows, in_f, 300 + i, "real").to(dev), qt, in_f, rows, dtype=dtype,
                                         bias=None if bias is None else bias.to(dev, dtype)))
    return holder


def forwards(holder: nn.Module, xs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Each layer's output, bare, under a column overlay, under a row overlay, and under both."""
    out: List[torch.Tensor] = []
    for lin, x in zip(holder.lins, xs):
        r, c = lin.out_features, lin.in_features
        col = torch.linspace(-1, 1, r).to(x.device)
        row = torch.linspace(1, -1, c).to(x.device)
        out.append(lin(x))
        lin._memit_col_overlays = {c - 1: col}
        out.append(lin(x))
        lin._memit_col_overlays, lin._memit_row_overlays = {}, {0: row}
        out.append(lin(x))
        lin._memit_col_overlays = {1: col}
        out.append(lin(x))
        lin._memit_col_overlays, lin._memit_row_overlays = {}, {}
    return out


def accessors(rt: Any, labkit: Any) -> Dict[str, Any]:
    lm = language_model(rt.hf_model)
    r: Dict[str, Any] = {}
    for l in (0, 5):
        mlp = lm.layers[l].mlp
        r[f"dense{l}"] = gw.dense(mlp.down_proj)
        r[f"full{l}"] = (rt.get_full_w_down(l), rt.get_full_w_gate(l), rt.get_full_w_up(l))
        r[f"dense_weight{l}"] = ao.dense_weight(lm.layers[l].self_attn.o_proj)
    return r


def slicers(rt: Any) -> Dict[str, Any]:
    lm = language_model(rt.hf_model)
    idx = torch.tensor([5, 0, 127, 64, 5])
    return {f"{k}{l}": v for l in (0, 5) for k, v in (
        ("cols", gw.dense_cols(lm.layers[l].mlp.down_proj, idx)), ("rows", gw.dense_rows(lm.layers[l].mlp.gate_proj, idx)),
        ("slice_down", rt.slice_w_down(l, idx)), ("slice_gate", rt.slice_w_gate(l, idx)), ("slice_up", rt.slice_w_up(l, idx)))}


def gradients(rt: Any, labkit: Any) -> List[torch.Tensor]:
    """labkit's GGUF twin on two layers: output and input gradient of sum(y^2), a vault column in force on the second."""
    lm = language_model(rt.hf_model)
    out = []
    for mod, col in ((lm.layers[1].mlp.up_proj, None), (lm.layers[2].mlp.down_proj, 7)):
        if col is not None:
            mod._memit_col_overlays[col] = torch.linspace(-1, 1, mod.out_features).to(rt.device)
        x = torch.randn(3, mod.in_features, generator=torch.Generator().manual_seed(8)).to(rt.device, mod.compute_dtype)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            y = labkit._GGUFLinearFn.apply(x, mod)
            y.float().pow(2).sum().backward()
        out += [y.detach(), x.grad]
        mod._memit_col_overlays.clear()
    return out


def check_reach(dev: str, files: Dict[str, Path], tok: Any, labkit: Any) -> None:
    before = len(failures)
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        holder = layer_holder(dev, dtype)
        xs = [torch.randn(2, 3, l.in_features, generator=torch.Generator().manual_seed(9)).to(dev, dtype) for l in holder.lins]
        ref = forwards(holder, xs)
        rec = wf.install(holder, "compiled", log=quiet)
        with Counting() as c:
            got = forwards(holder, xs)
        check(rec["mode"] == "compiled" and rec["linears"] == 6 and wf.describe(holder) == rec and c.served == c.calls == 24
              and all(wf.same_bits(a, b) for a, b in zip(got, ref)),
              f"{dev} {dtype}: six layers' forwards through the compiled graphs ({c.served} of {c.calls}), bare and under "
              "column and row overlays, the reference's bits")
        # a layer of a shape not tested at install, a graph built for it beside the install (as a probe's would be) and
        # then put wrong; and a layer of a tested shape whose blocks are a strided view: both through the reference
        fast = gw._FAST_UNPACK
        untested = gw.GGUFLinear(random_blocks("Q4_0", 12, 96, 390, "real").to(dev), "Q4_0", 96, 12, dtype=dtype)
        key = gf._key(untested.weight_blocks, "Q4_0", dtype)
        gf.unpack_compiled(untested.weight_blocks, "Q4_0", dtype)
        true_graph = gf._COMPILED[key]
        gf._COMPILED[key] = lambda b, q, d: true_graph(b, q, d) + 1  # noqa: E731  (wrong: any call through it shows)
        strided = gw.GGUFLinear(random_blocks("Q4_0", 40, 128, 391, "real").to(dev)[:, :36], "Q4_0", 64, 40, dtype=dtype)
        x_u = torch.randn(2, 96, generator=torch.Generator().manual_seed(10)).to(dev, dtype)
        x_s = torch.randn(2, 64, generator=torch.Generator().manual_seed(11)).to(dev, dtype)
        gw._FAST_UNPACK = None
        want_u, want_s = untested(x_u), strided(x_s)
        gw._FAST_UNPACK = fast
        fb, g1 = fast.fell_back, graphs()  # type: ignore[union-attr]
        try:
            with Counting() as c:
                got_u, got_s = untested(x_u), strided(x_s)
        finally:
            gf._COMPILED[key] = true_graph
        check(key not in fast.tested and not strided.weight_blocks.is_contiguous() and c.calls == 2 and c.served == 0  # type: ignore[union-attr]
              and fast.fell_back == fb and graphs() == g1 and wf.same_bits(got_u, want_u) and wf.same_bits(got_s, want_s),  # type: ignore[union-attr]
              f"{dev} {dtype}: a shape not tested at install, its graph built beside it, and blocks laid out strided: both "
              f"through the reference ({c.served} of {c.calls} served, {fast.fell_back - fb} sent back by a guard, "  # type: ignore[union-attr]
              f"{graphs() - g1} graphs built)")
        wf.install(holder, "reference")
    rt = GGUFRuntime(files["Q4_0"], device=dev, dtype=torch.bfloat16, tokenizer=tok, load_jpp=False).load()
    acc_ref, sl_ref, gr_ref = accessors(rt, labkit), slicers(rt), gradients(rt, labkit)
    rec = wf.install(rt, "compiled", log=quiet)
    check(rec["types"] == {"Q4_0": 41} and rec["skipped"] == {"resident": 0, "dense": 0} and rec["self_test"]["same_bits"],
          f"{dev}: compiled on the Q4_0 stand-in's 41 layers ({rec['shapes']} shapes)")
    with Counting() as c:
        acc = accessors(rt, labkit)
    n_acc = c.served
    with Counting() as c:
        sl = slicers(rt)
    n_sl = c.calls
    with Counting() as c:
        gr = gradients(rt, labkit)
    n_gr = c.served
    same = lambda a, b: all(wf.same_layout(x, y) for x, y in zip(a, b)) if isinstance(a, tuple) else wf.same_layout(a, b)  # noqa: E731
    check(all(same(acc[k], acc_ref[k]) for k in acc_ref) and n_acc == 10,
          f"{dev}: dense, get_full_w_down, _gate, _up and atlas_ops.dense_weight through the compiled graphs ({n_acc} of 10), "
          "the reference's bits and strides")
    check(all(wf.same_layout(sl[k], sl_ref[k]) for k in sl_ref) and n_sl == 0,
          f"{dev}: dense_cols, dense_rows and slice_w_* through the reference's own slicers ({n_sl} calls of the hook)")
    check(all(wf.same_bits(a, b) for a, b in zip(gr, gr_ref)) and n_gr == 4,
          f"{dev}: labkit's gradient twin forward and backward through the compiled graphs ({n_gr} of 4), a column overlay "
          "in force on one: the reference's outputs and gradients")
    down = language_model(rt.hf_model).layers[3].mlp.down_proj
    g0 = graphs()
    with torch.inference_mode():
        w_im = gw.dense(down)
        w_ref = down.unpacked_reference()
    check(wf.same_layout(w_im, w_ref) and graphs() == g0 and gw._FAST_UNPACK.fell_back == 1,  # type: ignore[union-attr]
          f"{dev}: a call under torch.inference_mode through the reference, counted, nothing compiled after install "
          f"({graphs() - g0} graphs)")
    wf.install(rt, "reference")
    check(gw._FAST_UNPACK is None and wf.describe(rt) == {"mode": "reference", "linears": 0, "shapes": 0, "devices": []},
          f"{dev}: reference unbinds it ({wf.describe(rt)})")
    del rt
    section(f"the hook's reach on {dev}: forwards with overlays, dense, get_full_w_*, dense_weight and labkit's gradients "
            "through the graphs, the slicers through the reference; an untested shape, strided blocks and a call under "
            "inference_mode sent to the reference", before)


def check_held_otherwise(files: Dict[str, Path], tok: Any, tmp: Path) -> None:
    before = len(failures)
    ids = torch.tensor([tok.encode(asi.CORPUS[0])[:16]])
    rt_b = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.bfloat16, tokenizer=tok, load_jpp=False, resident_gb=1e-4).load()
    n_res = sum(1 for m in rt_b.hf_model.modules() if isinstance(m, gw.GGUFLinear) and m.resident)
    want = rt_b.hf_model(ids).logits
    rec = wf.install(rt_b, "compiled", log=quiet)
    check(0 < n_res < 41 and rec["linears"] == 41 - n_res and rec["skipped"] == {"resident": n_res, "dense": 0}
          and wf.same_bits(rt_b.hf_model(ids).logits, want),
          f"a resident budget: {n_res} layers resident left as they are and counted, {rec['linears']} packed compiled, the "
          f"same logits ({rec['skipped']})")
    for linears, held in (("resident", "resident"), ("dense", "dense")):
        rt_h = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.bfloat16, tokenizer=tok, load_jpp=False, linears=linears).load()
        said: List[str] = []
        rec = wf.install(rt_h, "compiled", log=said.append)
        check(rec == {"mode": "reference", "linears": 0, "shapes": 0, "devices": [], "asked": "compiled"}
              and said and said[0].startswith("No W4A16 linears in this model") and gw._FAST_UNPACK is None,
              f"THEODOLITE_GGUF_LINEARS={linears}: nothing packed, the reference recorded with what was asked ({rec})")
        del rt_h
    # a type gguf-py unpacks at load, dense, beside Q8_0 layers packed
    wide = cg.wide_stand_in(tok, 256, 512)
    path = cg.typed_stand_in(tok, wide, "IQ2_XXS", tmp / "wide-IQ2_XXS.gguf", np.random.default_rng(3))
    rt_t = GGUFRuntime(path, device="cpu", dtype=torch.bfloat16, tokenizer=tok, load_jpp=False).load()
    want = rt_t.hf_model(ids).logits
    rec = wf.install(rt_t, "compiled", log=quiet)
    check(rec["skipped"] == {"resident": 0, "dense": 30} and rec["linears"] == 11 and rec["types"] == {"Q8_0": 11}
          and wf.same_bits(rt_t.hf_model(ids).logits, want),
          f"IQ2_XXS layers dense at load (gguf-py) left as they are and counted, the Q8_0 ones compiled ({rec['skipped']})")
    del rt_t
    wf.install(rt_b, "reference")
    del rt_b
    # int4mm on the CPU: the kernel's product unchanged, the kernel off the reference's bits
    why = int4_cpu.kernel_missing()
    if why:
        print(f"  --    int4mm left out: {why}")
    else:
        rt_k = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.bfloat16, tokenizer=tok, linears="int4mm", load_jpp=False).load()
        n_k = len(int4_cpu.int4_layers(rt_k.hf_model))
        on_ref = rt_k.hf_model(ids).logits
        int4_cpu.enable(rt_k, False)
        off_ref = rt_k.hf_model(ids).logits
        int4_cpu.enable(rt_k, True)
        mid = rt_k.model_id
        rec = wf.install(rt_k, "compiled", log=quiet)
        with Counting() as c:
            on = rt_k.hf_model(ids).logits
        n_on = c.calls
        int4_cpu.enable(rt_k, False)
        with Counting() as c:
            off = rt_k.hf_model(ids).logits
        int4_cpu.enable(rt_k, True)
        check(rec["int4mm"] == n_k and rec["linears"] == 41 and n_k > 0 and wf.same_bits(on, on_ref)
              and n_on == 41 - n_k and rt_k.model_id == mid,
              f"int4mm: its {n_k} layers tested and counted; the kernel on, its product unchanged, the hook called only by "
              f"the {41 - n_k} layers it cannot take ({n_on}), the model id kept")
        check(wf.same_bits(off, off_ref) and c.served == 41 and not wf.same_bits(off_ref, on_ref),
              f"int4mm turned off: the reference forward's bits through the compiled graphs ({c.served} of 41 layers)")
        wf.install(rt_k, "reference")
        del rt_k
    section("layers held otherwise: a resident budget, resident and dense layers, gguf-py's at load, counted and left as "
            "they are; int4mm unchanged with the kernel on, the reference's bits with it off", before)


# ----------------------------------------------------------------------------------------- 3. readings, byte for byte


def readings(rt: Any, vault: LarqlVaultEngine, out: Path) -> Dict[str, Any]:
    """Every reading the check compares, in the runtime's current mode."""
    hf, lm = rt.hf_model, language_model(rt.hf_model)
    tok = rt.tokenizer
    ids = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("Who are you?", chat=True))]).to(rt.device)
    idx = torch.tensor([5, 7, 0, 127, 64])
    r: Dict[str, Any] = {"logits": hf(input_ids=ids).logits}
    w = ao.Weights(lm, rt)
    u = w.write_dirs(2, [9])[0]
    delta = torch.stack([torch.zeros_like(u), u * 3.0, u * -2.0])
    steps = ao.write_rows(lm, rt.lm_head_weight, [ids[0].tolist()] * 3, ao.Interventions(rows=3, adds={2: delta}),
                          max_new=8, stop_ids=rt.end_of_turn_ids(), bos_id=tok.bos_token_id, spare_bos=True,
                          softcap=ao.softcap_of(hf), top_k=5)
    r["write_rows"] = [(s.row, s.token_id, s.logprob, s.top_ids, s.top_lps, s.end) for s in steps]
    rep = vault.apply_patch_surgery("UPDATE", 3, 5, target_concept=100, strength=1.5, statement="check_gguf_fast")
    r["update"] = {k: v for k, v in rep["patch"].items() if k not in ("timestamp", "patch_id")}
    r["logits_update"] = hf(input_ids=ids).logits
    r["slice_w_down"], r["full_w_down"] = rt.slice_w_down(3, idx), rt.get_full_w_down(3)
    n_layers = larql_vault.N_LAYERS
    larql_vault.N_LAYERS = len(lm.layers)  # ROLLBACK sweeps the 31B's 60 layers; the stand-in has six
    try:
        r["rollback"] = vault.execute_lql("ROLLBACK")
    finally:
        larql_vault.N_LAYERS = n_layers
    r["logits_after"] = hf(input_ids=ids).logits
    res = cz.take_census(rt, 3, [0, 5], out, p50=1.5, p50_source="check_gguf_fast", design_name="main_rel", controls=1,
                         seed=3, max_new=6, top_k=5, batch=5, log=quiet)
    r["census_counts"] = res
    r["census_rows"] = out.read_bytes()
    r["census_meta"] = json.loads(cz.meta_path_of(out).read_text(encoding="utf-8"))
    return r


def check_readings(dev: str, files: Dict[str, Path], tok: Any, tmp: Path) -> None:
    before = len(failures)
    rt = GGUFRuntime(files["Q4_K"], device=dev, dtype=torch.bfloat16, tokenizer=tok, load_jpp=False).load()
    give_lens(rt)
    vault = LarqlVaultEngine(rt)
    card, mid = json.dumps(rt.weights_card, sort_keys=True), rt.model_id
    torch.manual_seed(0)
    t0 = time.time()
    ref = readings(rt, vault, tmp / f"ref-{dev.replace(':', '')}.jsonl")
    t_ref = time.time() - t0
    check(ref["census_counts"] == {"written": 34, "skipped": 0, "stopped": False} and ref["census_meta"]["unpacker"]["mode"] == "reference",
          f"{dev}: the reference's census, 34 answers ({ref['census_counts']}, {ref['census_meta']['unpacker']})")
    check(not wf.same_bits(ref["logits"], ref["logits_update"]) and wf.same_bits(ref["logits"], ref["logits_after"])
          and len({s[1] for s in ref["write_rows"] if s[0] == 1} ^ {s[1] for s in ref["write_rows"] if s[0] == 0}) > 0,
          f"{dev}: the edit moves the logits and ROLLBACK puts them back; the push changes what is written")
    t0 = time.time()
    rec = wf.install(rt, "compiled", log=quiet)
    t_install = time.time() - t0
    check(set(rec["types"]) == {"Q4_K", "Q5_0"} and rec["types"]["Q5_0"] == 6 and rec["linears"] == 41,
          f"{dev}: compiled on the wide Q4_K stand-in, its Q5_0 fallbacks among them ({rec['types']})")
    torch.manual_seed(0)
    t0 = time.time()
    with Counting() as c:
        got = readings(rt, vault, tmp / f"compiled-{dev.replace(':', '')}.jsonl")
    t_got = time.time() - t0
    check(c.served > 0 and c.served == c.calls and gw._FAST_UNPACK.fell_back == 0,  # type: ignore[union-attr]
          f"{dev}: every whole weight of every reading through the graphs tested at install ({c.served} of {c.calls}, none "
          "sent to the reference)")
    for k, v in ref.items():
        if k == "census_meta":
            strip = lambda m: {a: b for a, b in m.items() if a not in ("unpacker", "started_at")}  # noqa: E731
            un = got[k]["unpacker"]
            check(strip(got[k]) == strip(v) and un["mode"] == "compiled" and un["weights"] == "gguf"
                  and un["self_test"]["same_bits"] is True and got[k]["model"] == mid,
                  f"{dev}: the census meta as the reference's but for its unpacker, recorded {{'mode': 'compiled', 'weights': "
                  f"'gguf', ...}} under the same model id ({un.get('mode')}, {got[k]['model']})")
        elif isinstance(v, torch.Tensor):
            check(wf.same_layout(got[k], v), f"{dev}: {k} the reference's bits")
        else:
            check(got[k] == v, f"{dev}: {k} as the reference's")
    check(rt.model_id == mid and json.dumps(rt.weights_card, sort_keys=True) == card, f"{dev}: the model id and weights card unchanged")
    print(f"        {dev}: the reference's readings in {t_ref:.1f} s; compiled installed in {t_install:.1f} s, read in {t_got:.1f} s")
    wf.install(rt, "reference")
    del rt, vault
    section(f"readings on {dev}, byte for byte: logits, three pushed answers, a vault UPDATE and ROLLBACK, a census run, "
            "recorded as the GGUF's compiled unpacker", before)


# ----------------------------------------------------------------------------------------------- 4. refusals, records


def refused(run: Callable[[], Any], *words: str) -> Optional[str]:
    try:
        run()
    except wf.Refused as exc:
        msg = str(exc)
        check(all(w in msg for w in words), f"the refusal says {words}: {msg}")
        return msg
    check(False, f"refused ({words})")
    return None


def check_refusals(files: Dict[str, Path], tok: Any) -> None:
    before = len(failures)
    rt = GGUFRuntime(files["Q4_K"], device="cpu", dtype=torch.bfloat16, tokenizer=tok, load_jpp=False).load()
    lm = language_model(rt.hf_model)
    ids = torch.tensor([tok.encode(asi.CORPUS[1])[:16]])
    want = rt.hf_model(ids).logits
    odd = lm.layers[4].mlp.up_proj
    true_fast = gf.unpack_compiled
    own = {m.weight_blocks.data_ptr() for _n, m in gf.packed_layers(rt.hf_model)}

    def one_bit_off(blocks: torch.Tensor, qtype: Any, dtype: torch.dtype) -> torch.Tensor:
        w = true_fast(blocks, qtype, dtype)
        if blocks.data_ptr() == odd.weight_blocks.data_ptr():
            w = w.clone()
            w.view(torch.int16).view(-1)[5] ^= 1
        return w

    def off_on_the_table(blocks: torch.Tensor, qtype: Any, dtype: torch.dtype) -> torch.Tensor:
        w = true_fast(blocks, qtype, dtype)
        if blocks.data_ptr() not in own and gw.type_name(gw.type_code(qtype)) == "Q5_0":  # a table, laid out as a layer
            b = blocks.reshape(-1, 22)
            at = ((b[:, 0] == 0x00) & (b[:, 1] == 0x3C)).nonzero().flatten()
            if at.numel():
                w = w.clone()
                w.view(torch.int16).view(-1, 32)[at[0], 3] ^= 1  # under d = 1.0, one weight one bit off, nowhere a layer reads
        return w

    def laid_out_otherwise(blocks: torch.Tensor, qtype: Any, dtype: torch.dtype) -> torch.Tensor:
        w = true_fast(blocks, qtype, dtype)
        return w.t().contiguous().t() if blocks.data_ptr() == odd.weight_blocks.data_ptr() else w

    for fake, words, what in (
            (one_bit_off, ("weights differ from the reference's on 1 of 41 GGUF layers", f"of type {odd.qtype}",
                           "layers.4.mlp.up_proj first: other bits"), "one bit off on one layer"),
            (off_on_the_table, ("on the table of every fp16 pattern of Q5_0", "through the graph of its [",
                                "the fp16 field at byte 0", "(0x3c00)"), "one weight off on a table alone"),
            (laid_out_otherwise, ("on 1 of 41 GGUF layers", "the same bits laid out otherwise, strides"), "laid out otherwise")):
        gf.unpack_compiled = fake
        try:
            msg = refused(lambda: wf.install(rt, "compiled"), *words, "nothing was installed")
        finally:
            gf.unpack_compiled = true_fast
        print(f"        e.g. {msg}")
        check(gw._FAST_UNPACK is None and wf.describe(rt)["mode"] == "reference"
              and wf.same_bits(rt.hf_model(ids).logits, want),
              f"compiled with {what}: refused, the reference left bound, the forward as before")
    refused(lambda: wf.install(rt, "lut"), "A GGUF's layers have no lut unpacker", "no faster than compiled", "use compiled")
    refused(lambda: wf.install(rt, "resident"), "held resident at load", "THEODOLITE_GGUF_LINEARS=resident",
            "THEODOLITE_GGUF_RESIDENT_GB")
    refused(lambda: wf.install(rt, "fast"), "reference, lut, compiled, resident", "'fast'")
    check(gw._FAST_UNPACK is None and wf.describe(rt) == {"mode": "reference"},
          "lut, resident and an unknown unpacker refused: the reference left bound")
    true_for = gf._compiled_for

    def no_compiler(key: Any) -> Any:
        def fails(*_a: Any) -> torch.Tensor:
            raise RuntimeError("InductorError: CppCompileError: C++ compile error (stood in for)")

        return fails

    gf._compiled_for = no_compiler  # a graph that would not build
    try:
        refused(lambda: wf.install(rt, "compiled"), "could not be run on this GGUF's layers here", "CppCompileError",
                "nothing was installed")
    finally:
        gf._compiled_for = true_for
    check(gw._FAST_UNPACK is None and wf.describe(rt) == {"mode": "reference"}, "a compile that fails: the reference left bound")
    a = wf.install(rt, "compiled", log=quiet)
    b = wf.install(rt, "compiled", log=quiet)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("install_s", "self_test")}  # noqa: E731
    check(strip(a) == strip(b) and wf.describe(rt) == b and gw._FAST_UNPACK.mode == "compiled"  # type: ignore[union-attr]
          and wf.same_bits(rt.hf_model(ids).logits, want),
          "installed twice: the same record, bound once, the same forward")
    check(list(a) == ["mode", "weights", "linears", "types", "shapes", "devices", "int4mm", "skipped", "self_test", "install_s"]
          and list(a["self_test"]) == ["linears", "same_bits", "table", "seconds"] and a["self_test"]["same_bits"] is True
          and sorted(a["self_test"]["table"]) == sorted(a["types"])
          and sum(t["graphs"] for t in a["self_test"]["table"].values()) == a["shapes"]
          and all(list(t) == ["fields", "patterns", "graphs", "nan_payloads", "on"] and t["patterns"] == 65536 * len(gw.F16_FIELDS[q])
                  and t["on"] == ["cpu/bfloat16"] for q, t in a["self_test"]["table"].items()),
          f"the record: mode, weights, linears, types, shapes, devices, int4mm, skipped, the self-test (a table per type, "
          f"through each of its graphs) and install_s ({list(a)}, {a['self_test']['table']})")
    wf.install(rt, "reference")
    check(gw._FAST_UNPACK is None and wf.describe(rt)["mode"] == "reference" and gf.describe() == {"mode": "reference"}
          and wf.same_bits(rt.hf_model(ids).logits, want),
          "reference after compiled: the hook unbound, both describe() the reference")
    wf.install(rt, "compiled", log=quiet)
    gw._FAST_UNPACK = None  # put back by hand
    check(wf.describe(rt) == {"mode": "reference"}, f"a hook put back by hand reads as the reference ({wf.describe(rt)})")
    wf.install(rt, "reference")
    del rt
    # a dense model: as before
    rt_d, _v = asi.stand_in()
    said: List[str] = []
    rec = wf.install(rt_d, "compiled", log=said.append)
    check(rec == {"mode": "reference", "linears": 0, "shapes": 0, "devices": [], "asked": "compiled"}
          and wf.describe(rt_d) == rec and said and said[0].startswith("No W4A16 linears in this model, so the compiled unpacker "
                                                                       "has nothing to unpack: the reference is recorded"),
          f"the dense stand-in: the reference recorded, compiled asked, the words as before ({rec})")
    wf.install(rt_d, "reference")
    # the W4A16 checkpoint's compiled record, its keys as before (a holder of 4-bit linears, patched as the checkpoint's)
    import check_dequant as cd

    holder = nn.Module()
    holder.lins = nn.ModuleList()
    for i, (r_, c_) in enumerate(((3, 64), (40, 256))):
        lin = nn.Linear(c_, r_, bias=False).to(torch.bfloat16)
        del lin.weight
        packed, scale = cd.random_packed(r_, c_, 50 + i)
        lin.register_buffer("weight_packed", packed)
        lin.register_buffer("weight_scale", scale)
        lin.weight = None
        holder.lins.append(lin)
    er.patch_compressed_tensors_for_low_vram(holder)
    rec = wf.install(holder, "compiled", log=quiet)
    check(list(rec) == ["mode", "linears", "shapes", "devices", "self_test", "install_s"]
          and list(rec["self_test"]) == ["linears", "same_bits", "table", "table_nan_payloads", "seconds"]
          and rec["mode"] == "compiled" and rec["self_test"]["same_bits"] is True and gw._FAST_UNPACK is None
          and er.dequantize_w4a16_full is not wf.REFERENCE and "weights" not in rec,
          f"the W4A16 compiled record: its keys as before, nothing GGUF in it ({list(rec)})")
    wf.install(holder, "reference")
    check(er.dequantize_w4a16_full is wf.REFERENCE, "and the reference put back")
    section("refusals in plain words, the reference left bound: one bit off on a layer, on a table alone, laid out "
            "otherwise, a compile that fails, lut, resident, an unknown name; the records as said", before)


# ------------------------------------------------------------------------------------------------------ 5. wiring


def check_wiring(files: Dict[str, Path], tok: Any, tmp: Path) -> None:
    before = len(failures)
    py = [sys.executable, "-I"]
    base_env = {**{k: v for k, v in os.environ.items() if k not in ENV_KEYS and k != "THEODOLITE_CENSUS_DIR"},
                "THEODOLITE_ATLAS_DIR": str(tmp / "atlas"), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    env = {**base_env, **{k: str(v) for k, v in gguf_env(files["Q4_0"]).items()}}
    rt_q = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.float32, tokenizer=tok, load_jpp=False).load()
    out_dir = tmp / "cli-census"
    args = ["--design", "main_rel", "--layer", "2", "--neurons", "0-1", "--random-controls", "1", "--max-new", "6",
            "--batch", "5", "--top-k", "3", "--no-rank"]
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), *args, "--out-dir", str(out_dir), "--unpacker", "compiled"],
                       env=env, capture_output=True, text=True, timeout=900)
    run = out_dir / "L2-main_rel.jsonl"
    meta = json.loads(cz.meta_path_of(run).read_text()) if run.exists() else {}
    un = meta.get("unpacker") or {}
    check(r.returncode == 0 and un.get("mode") == "compiled" and un.get("weights") == "gguf" and un.get("linears") == 41
          and (un.get("self_test") or {}).get("same_bits") is True and meta.get("model") == rt_q.model_id
          and "unpacker: compiled on 41 GGUF layers" in r.stdout,
          f"census_run.py --unpacker compiled on a GGUF: the GGUF's record in its meta, the model id unchanged ({un}; {r.stderr[-300:]})")
    if run.exists():
        ref_out = tmp / "cli-ref.jsonl"
        cz.take_census(rt_q, 2, [0, 1], ref_out, p50=meta["p50"], p50_source=meta["p50_source"], design_name="main_rel", controls=1,
                       seed=0, max_new=6, top_k=3, batch=5, log=quiet)
        check(list(cz.read_rows(run)) == list(cz.read_rows(ref_out)), "its rows are the reference's, read in this process at the same p50")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), *args, "--out-dir", str(tmp / "cli-lut"), "--unpacker", "lut"],
                       env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode != 0 and "A GGUF's layers have no lut unpacker" in r.stderr and not (tmp / "cli-lut" / "L2-main_rel.jsonl").exists(),
          f"census_run.py --unpacker lut on a GGUF: refused in plain words, nothing written ({r.stderr[-200:]})")

    # the server's own lifespan, its runtime built from THEODOLITE_GGUF by runtime_factory
    from fastapi.testclient import TestClient

    import backend.server as server

    with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "compiled"})):
        try:
            with TestClient(server.app, base_url=BASE) as client:
                health = client.get("/api/health").json()
        finally:
            server._RUNTIME = server._VAULT = None
    un = health.get("unpacker") or {}
    check(un.get("mode") == "compiled" and un.get("weights") == "gguf" and un.get("linears") == 41
          and (un.get("self_test") or {}).get("same_bits") is True and str(health.get("model_id", "")).startswith("gguf:stand-in-Q4_0.gguf#")
          and (health.get("weights") or {}).get("format") == "gguf",
          f"the server on THEODOLITE_GGUF with $THEODOLITE_UNPACKER=compiled: /api/health says so ({un.get('mode')}, "
          f"{health.get('model_id')})")
    check(list(health)[-3:] == ["atlas", "label_gate", "lens"], f"/api/health: atlas, label_gate and lens still last ({list(health)[-4:]})")
    async def lifespan_alone() -> None:  # the lifespan itself: TestClient hands a SystemExit on as a bare cancel
        async with server.lifespan(server.app):
            pass

    with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "lut"})):
        try:
            asyncio.run(lifespan_alone())
            check(False, "the server refuses lut on a GGUF")
        except SystemExit as exc:
            check("A GGUF's layers have no lut unpacker" in str(exc) and gw._FAST_UNPACK is None,
                  f"the server stops in plain words on lut over a GGUF: {exc}")
        finally:
            server._RUNTIME = server._VAULT = None
    wf.install(rt_q, "reference")

    # the lab daemon and the standalone scientist
    sys.path.insert(0, str(ROOT_DIR / "experiments" / "lab"))
    cwd = os.getcwd()
    import labd  # changes to the repository's root, as it does when run
    import labkit

    os.chdir(cwd)
    labd.LOG = tmp / "labd_exec_log.jsonl"
    gender = labkit.Lab._build_gender_ids
    labkit.Lab._build_gender_ids = lambda self: {}  # the stand-in's vocabulary has not the study's paired words
    try:
        with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "compiled"})):
            asyncio.run(labd._startup())
        h = TestClient(labd.app, base_url=BASE).get("/health").json()
        check(h["ready"] and (h.get("unpacker") or {}).get("mode") == "compiled" and h["unpacker"].get("weights") == "gguf"
              and h["model"] == labd.STATE["rt"].model_id,
              f"labd with $THEODOLITE_UNPACKER=compiled on a GGUF: /health names the unpacker ({(h.get('unpacker') or {}).get('mode')})")
        labd.STATE.clear()
        with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "quick"})):
            try:
                asyncio.run(labd._startup())
                check(False, "labd refuses an unpacker that is not one")
            except SystemExit as exc:
                check("not 'quick'" in str(exc) and "nothing was loaded" in str(exc) and "rt" not in labd.STATE,
                      f"labd stops in plain words, before loading: {exc}")
    finally:
        labkit.Lab._build_gender_ids = gender
        labd.STATE.clear()
        labd.TOKEN["value"] = None
    import backend.mcp_scientist as ms

    bound = (ms._SHARED_RUNTIME, ms._SHARED_VAULT)
    try:
        ms._SHARED_RUNTIME, ms._SHARED_VAULT = None, None
        with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "compiled"})):
            rt_ms, _vault_ms = ms.get_runtime_and_vault()
        check(isinstance(rt_ms, GGUFRuntime) and wf.describe(rt_ms)["mode"] == "compiled",
              f"the standalone scientist with $THEODOLITE_UNPACKER=compiled on a GGUF: installed ({wf.describe(rt_ms)['mode']})")
        ms._SHARED_RUNTIME, ms._SHARED_VAULT = None, None
        with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: "quick"})):
            try:
                ms.get_runtime_and_vault()
                check(False, "the standalone scientist refuses an unpacker that is not one")
            except wf.Refused as exc:
                check("not 'quick'" in str(exc) and "nothing was loaded" in str(exc) and ms._SHARED_RUNTIME is None,
                      f"the standalone scientist refuses in plain words, an Exception, before loading: {exc}")
        # over MCP, the refusal is the call's error, its words, and the server answers the next call (a SystemExit inside
        # a tool would end the stdio server's task group, the call never answered)
        from fastmcp import Client

        async def two_calls() -> Tuple[Any, Any]:
            async with Client(ms.mcp) as client:
                first = await asyncio.wait_for(client.call_tool("query_larql_vault", {"lql_query": "ROLLBACK"},
                                                                raise_on_error=False), 300)
                return first, await asyncio.wait_for(client.list_tools(), 60)

        for name, words in (("lut", "A GGUF's layers have no lut unpacker"), ("quick", "not 'quick'")):
            ms._SHARED_RUNTIME, ms._SHARED_VAULT = None, None
            with environment(**gguf_env(files["Q4_0"], **{wf.UNPACKER_ENV: name})):
                try:
                    res, tools = asyncio.run(two_calls())
                    text = " ".join(getattr(x, "text", "") for x in res.content)
                    check(res.is_error and words in text and len(tools) >= 5 and ms._SHARED_RUNTIME is None,
                          f"the scientist over MCP with $THEODOLITE_UNPACKER={name} on a GGUF: the call answered with the "
                          f"refusal's words, the server still answering ({text[:160]})")
                except (Exception, SystemExit, BaseExceptionGroup) as exc:  # noqa: B036
                    check(False, f"the scientist over MCP with $THEODOLITE_UNPACKER={name}: the call not answered "
                                 f"({type(exc).__name__}: {str(exc)[:200]})")
    finally:
        ms._SHARED_RUNTIME, ms._SHARED_VAULT = bound
        wf.install(rt_q, "reference")

    # pod_tool.py steps: what ran beside what was asked, the fastest chosen by what ran
    # (on the dense stand-in lut and compiled alone: both run as the reference, so none ran as asked and none is chosen,
    # whichever is fastest by chance)
    for what, extra, e, asked in (("a GGUF", [], env, "reference,lut,compiled"),
                                  ("the dense stand-in", ["--stand-in"], base_env, "lut,compiled")):
        steps = tmp / f"steps-{len(extra)}.json"
        r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "pod" / "pod_tool.py"), "steps", *extra, "--layer", "2", "--batches", "2",
                                 "--unpackers", asked, "--max-new", "4", "--out", str(steps)],
                           env=e, capture_output=True, text=True, timeout=900)
        js = json.loads(steps.read_text()) if steps.exists() else {}
        rows = {x["unpacker"]: x for x in js.get("rows", [])}
        best = js.get("best")
        if not extra:
            ok = (r.returncode == 0 and rows.get("compiled", {}).get("installed") == "compiled"
                  and rows.get("reference", {}).get("installed") == "reference"
                  and "no lut unpacker" in rows.get("lut", {}).get("refused", "") and rows.get("lut", {}).get("installed") == "reference"
                  and (best or {}).get("unpacker") in ("reference", "compiled") and (best or {}).get("installed") == (best or {}).get("unpacker"))
        else:
            ok = (r.returncode == 1 and steps.exists() and set(rows) == {"lut", "compiled"}
                  and all(x.get("installed") == "reference" and x.get("fits") for x in rows.values()) and best is None)
        check(ok, f"pod_tool.py steps on {what} ({asked}): each row what ran beside what was asked, the fastest that fit "
              f"chosen among those that ran as asked ({ {k: (x.get('installed'), x.get('step_s')) for k, x in rows.items()} }, "
              f"best {(best or {}).get('unpacker')}, exit {r.returncode}; {r.stderr[-300:]})")
    sel = tmp / "selftest.json"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "pod" / "pod_tool.py"), "selftest", "--unpacker", "compiled", "--out", str(sel)],
                       env=env, capture_output=True, text=True, timeout=900)
    test = json.loads(sel.read_text()).get("unpacker_test", "") if sel.exists() else ""
    check(r.returncode == 0 and "on every fp16 pattern of each type's fields" in test and "piece-by-scale" not in test,
          f"pod_tool.py selftest on a GGUF names the GGUF's own table test, not the checkpoint's ({test}; {r.stderr[-200:]})")
    src = (ROOT_DIR / "scripts" / "verify_control_vector.py").read_text(encoding="utf-8")
    check("unp = w4a16_fast.install(rt, unpacker" in src and '"unpacker": unp})' in src and '"unpacker": None if stand_in else unpacker' not in src,
          "verify_control_vector.py records install's record, not the name asked (its code read: it runs beside a census of the 31B)")
    vcv = [str(ROOT_DIR / "scripts" / "verify_control_vector.py"), "instrument", "--out", str(tmp / "vcv")]
    r = subprocess.run(py + vcv + ["--unpacker", "lut"], env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode != 0 and "A GGUF's layers have no lut unpacker" in r.stderr and "Traceback" not in r.stderr
          and not (tmp / "vcv" / "instrument.json").exists(),
          f"verify_control_vector.py instrument --unpacker lut on a GGUF: refused in plain words, no traceback, nothing "
          f"written ({r.stderr.strip()[-200:]})")
    t0 = time.time()
    r = subprocess.run(py + vcv + ["--unpacker", "quick"], env=env, capture_output=True, text=True, timeout=600)
    check(r.returncode == 2 and "invalid choice: 'quick'" in r.stderr and "Loading Gemma 4" not in r.stderr + r.stdout,
          f"verify_control_vector.py --unpacker quick: refused by its flags before anything loads ({time.time() - t0:.1f} s; "
          f"{r.stderr.strip()[-160:]})")
    del rt_q
    section("the flags, the server, labd, the scientist, the pod's steps and selftest, verify_control_vector: compiled on "
            "a GGUF installed, recorded as it ran; lut refused there in plain words", before)


# ------------------------------------------------------------------------------------------------ 6. a real file


def check_file(path: Path, dev: str) -> None:
    before = len(failures)
    t0 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    rt = GGUFRuntime(path, device=dev, dtype=torch.bfloat16, load_jpp=False).load()
    load_s = time.time() - t0
    print(f"        {path.name} on {dev}: loaded in {load_s:.1f} s, {rt.model_id}, {rt.weights_card.get('layers_as')}")
    ids = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("What is the capital of France?", chat=True))]).to(rt.device)
    lm = language_model(rt.hf_model)

    def timed(n: int = 3) -> float:
        one = ids[:, -1:]
        ts = []
        for _ in range(n):
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            lm(input_ids=one, use_cache=False)
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t1)
        return statistics.median(ts)

    ref = rt.hf_model(input_ids=ids).logits
    timed(1)  # a warm-up, so neither way is timed cold
    t_ref = timed()
    try:
        rec = wf.install(rt, "compiled", log=lambda s: print(f"        {s}"))
    except wf.Refused as exc:
        check(False, f"{path.name}: compiled installed ({exc})")
        del rt
        return
    fast = gw._FAST_UNPACK
    with Counting() as c:
        got = rt.hf_model(input_ids=ids).logits
        timed(1)
        t_fast = timed()
    fell = getattr(fast, "fell_back", None)
    st = rec.get("self_test") or {}
    check(rec["mode"] == "compiled" and st.get("same_bits") is True and rec["linears"] > 0,
          f"{path.name}: compiled installed on {rec['linears']} layers ({rec['types']}, {rec['shapes']} shapes, skipped "
          f"{rec['skipped']}) in {rec['install_s']} s, every layer the reference's bits")
    check(c.calls > 0 and c.served == c.calls and fell == 0,
          f"{path.name}: the forward and the timed rows through the compiled graphs ({c.served} of {c.calls} unpackings, "
          f"{fell} sent to the reference)")
    check(wf.same_layout(got, ref), f"{path.name}: one forward's logits on a fixed prompt ({ids.shape[1]} pieces), the reference's bits")
    print(f"        one row's forward: {t_ref * 1e3:.1f} ms through the reference, {t_fast * 1e3:.1f} ms compiled "
          f"(median of 3 after a warm-up, timed, not checked{'; compiled not faster here' if t_fast >= t_ref else ''})")
    print(f"        the record: {json.dumps(rec)[:1500]}")
    if dev.startswith("cuda"):
        print(f"        memory on {dev}: {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB held, "
              f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB at the peak")
    wf.install(rt, "reference")
    del rt
    section(f"{path.name} on {dev}: compiled tested on every layer and serving each, the logits the reference's bits", before)


# ------------------------------------------------------------------------------------------------------- 7. pins


def check_pins() -> None:
    before = len(failures)
    sys.path.insert(0, str(ROOT_DIR / "experiments" / "scripts"))
    import R4_common

    check(R4_common.pinned_differ() == [], f"the files R4 pins are unchanged: {R4_common.pinned_differ()}")
    check(wf.REFERENCE.__module__ == "backend.engine_runtime" and wf.REFERENCE.__name__ == "dequantize_w4a16_full"
          and not hasattr(wf.REFERENCE, "mode"), "w4a16_fast.REFERENCE is the pinned unpacker itself")
    r = subprocess.run(["git", "-C", str(ROOT_DIR), "show", "census-kit-1:backend/w4a16_fast.py"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  --    the census-kit-1 tag is not here ({r.stderr.strip()[:100]}): w4a16_fast's code not compared with it")
    else:
        def parts(src: str) -> Dict[str, str]:
            tree = ast.parse(src)
            out: Dict[str, str] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    out[node.name] = ast.get_source_segment(src, node) or ""
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    t = node.targets[0] if isinstance(node, ast.Assign) else node.target
                    if isinstance(t, ast.Name):
                        out[t.id] = ast.get_source_segment(src, node) or ""
            return out

        kit, now = parts(r.stdout), parts((ROOT_DIR / "backend" / "w4a16_fast.py").read_text(encoding="utf-8"))
        differ = sorted(k for k in set(kit) | set(now) if kit.get(k) != now.get(k))
        check(differ == ["_uninstall", "describe", "install"] and all(k in kit for k in ("_compiled_for", "_unpack", "_key",
                                                                                       "_installed_compiled", "REFERENCE")),
              f"w4a16_fast: every function and constant as at census-kit-1 but install, _uninstall and describe ({differ})")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import torch, torch._dynamo.config as dc, torch._inductor.config as ic; "
            "b = (dc.get_config_copy(), ic.get_config_copy()); import backend.w4a16_fast; "
            "print('backend.gguf_fast' in sys.modules, b == (dc.get_config_copy(), ic.get_config_copy()))")
    r = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT_DIR)], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    check(r.stdout.split() == ["False", "True"], f"importing w4a16_fast loads no GGUF module and changes no compiler config ({r.stdout.strip()} {r.stderr[-200:]})")
    import torch._inductor.config as ic

    check(ic.emulate_precision_casts is False, "a GGUF's install set no global Inductor config (emulate_precision_casts is per compile)")
    section("the R4 pins, and w4a16_fast's checkpoint-facing code, as they were", before)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cuda", action="store_true", help="run the device sections on cuda:0 too")
    ap.add_argument("--gguf", default="", help="a GGUF file to load packed and check (section 6)")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    devices = ["cpu"]
    if a.cuda:
        if torch.cuda.is_available():
            devices.append("cuda:0")
        else:
            print("  --    --cuda, but no GPU here: the CPU alone")
    compiled, why = compile_works()
    print(f"  {'..' if compiled else '--'}    \"compiled\" on the CPU {'checked' if compiled else 'left out'}: {why}")
    on = devices if compiled else [d for d in devices if d != "cpu"]
    tmp = Path(tempfile.mkdtemp(prefix="gguf-fast-check-"))
    os.environ["THEODOLITE_ATLAS_DIR"] = str(tmp / "atlas")
    asi.use_stand_in_atlas()
    try:
        tok = asi.stand_in_tokenizer()
        hf = asi.stand_in_hf(tok)
        files = {"Q4_0": gm.write_gemma4_gguf(hf, tmp / "stand-in-Q4_0.gguf", tokenizer=tok, qtype="Q4_0"),
                 "Q4_K": gm.write_gemma4_gguf(cg.wide_stand_in(tok, 256, 512), tmp / "stand-in-wide-Q4_K.gguf", tokenizer=tok,
                                              qtype="Q4_K")}
        check_fields()
        if on:
            check_types(on)
            labkit = load_labkit()
            for dev in on:
                check_reach(dev, files, tok, labkit)
            if compiled:
                check_held_otherwise(files, tok, tmp)
            for dev in on:
                check_readings(dev, files, tok, tmp)
            if compiled:
                check_refusals(files, tok)
                check_wiring(files, tok, tmp)
        if a.gguf:  # one device: cuda:0 with --cuda, else the CPU (a 31B is loaded once, and never where it was not asked)
            dev = "cuda:0" if a.cuda else "cpu"
            if dev in on:
                check_file(Path(a.gguf), dev)
            else:
                check(False, f"--gguf {Path(a.gguf).name}: not loaded, nothing of it checked: "
                             + ("--cuda, but no GPU here (it is not loaded on the CPU instead: run without --cuda for that)"
                                if a.cuda else "\"compiled\" does not compile on the CPU here"))
        check_pins()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
