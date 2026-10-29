#!/usr/bin/env python3
"""A CPU-only check of the faster 4-bit unpackers (backend/w4a16_fast.py): that each gives the very bits the runtime's
own unpacker gives (`engine_runtime.dequantize_w4a16_full`, R4-pinned and left as it is), and every reading through
them is the same, byte for byte. It loads neither the real model nor the lens, and hides the GPU. It checks:
1. the piece-by-scale table, exhaustively: each of the 16 pieces beside each of the 65,536 bf16 scale bit patterns,
   through "lut" and the reference, the same bits (NaNs by their bits, -0 apart from +0); the reference's own product
   is (piece - 8) x scale rounded once to the nearest bf16; through "compiled", the same bits but for the NaNs' own;
   and table_agrees, which install's table test uses, tells a number one bit off and a NaN moved from a NaN's bits;
2. where the runtime looks the unpacker up: the pinned forward, dequantize_w4a16_rows (so slice_w_gate and
   slice_w_up), get_full_w_down, _gate and _up, atlas_ops.dense_weight and labkit's gradient linears call whatever
   engine_runtime.dequantize_w4a16_full is bound to at the time; slice_w_down goes through dequantize_w4a16_cols,
   which is never rebound;
3. random packed matrices of several shapes (bf16 scales, and float32 ones), whole and through the row and column
   slicers, in each mode: the same bits as the reference; and each mode's forward over them, bare and under a column
   and a row overlay, gives the pinned forward's output; a "compiled" call its graphs were not tested for (under
   torch.inference_mode) goes through the reference, nothing compiled after install;
4. refusals, in plain words, the runtime left on the reference: an unpacker that is not one; a "lut" whose bits
   differ on a linear, one whose bits differ only on the piece-by-scale table, and one whose bits are right but laid
   out otherwise (its self-test); resident weights that do not fit (the bytes needed said); a resident forward one
   bit off the pinned one on the probe; and a resident weight written in place is caught at the next forward;
5. a 4-bit stand-in: scripts/atlas_stand_in.py's tiny Gemma 4 in bf16, every block linear quantised to group-32
   symmetric 4-bit with bf16 scales in compressed-tensors' pack-quantized layout (as the dequantizer reads it) and
   patched by patch_compressed_tensors_for_low_vram, as the real checkpoint is; in each mode, byte for byte against
   the reference: a forward's logits, bare, under a vault edit (a down_proj column overlay by apply_patch_surgery) and
   under a gate_proj row overlay; the runtime's slices and whole weights with each in force; the stream's p50 over the
   study's texts; the wiring check's reads under +-d and each reader's g0 and u0, read by hooks on gate_proj and up_proj
   (whose forwards the unpackers replace; on the reference, what plain hooks give, to bf16's rounding, and the unpushed
   read gelu(g0) u0); three answers written at once under their own pushes (write_rows); a census run's rows, with the
   mode in its meta, and a run continued under another unpacker, or merged with one, keeping each session's; the
   resident weights untouched after all of it, and "reference" putting every forward back;
6. the command lines and the server: census_run.py --stand-in --unpacker lut records it in the run's meta and an
   unknown unpacker is refused; atlas_build.py --unpacker lut records it in an index's meta; the server's own lifespan
   installs $THEODOLITE_UNPACKER and GET /api/health says which, and a name that is not one stops it in plain words;
7. the four runtime files R4 pins (experiments/scripts/R4_common.py's PINNED_UNCHANGED: backend/__init__.py,
   theodolite_math.py, engine_runtime.py, label_gate.py) still have their pinned sha256.

"compiled" goes through Inductor, which on the CPU needs a C++ compiler: a child process tries one tiny compile first
(240 s at most), and when that fails "compiled" is left out, with the reason said (as with --no-compiled).

Run: .venv/bin/python -I scripts/check_dequant.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn as nn

import backend.atlas_ops as ao
import backend.engine_runtime as ER
from atlas_stand_in import stand_in_hf, stand_in_runtime, stand_in_tokenizer, use_stand_in_atlas
from backend import census as cz
from backend.atlas_index import FILES
from backend import w4a16_fast as wf
from backend.causal_check import language_model
from backend.larql_vault import LarqlVaultEngine

failures: List[str] = []
passed = 0
# the files R4 pins that the instrument's runtime is (R4_common.PINNED_UNCHANGED holds them beside R4's own)
R4_RUNTIME_FILES = ("backend/__init__.py", "backend/theodolite_math.py", "backend/engine_runtime.py",
                    "backend/label_gate.py")


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


def pack_w4a16(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(packed int32 [out, in / 8], scale bf16 [out, in / 32]): symmetric group-32 4-bit quantisation in
    compressed-tensors' pack-quantized layout, nibble j of word k column 8k + j, stored as q + 8 (as
    experiments/scripts/R3_gpu.py's pack_w4a16, with bf16 scales as the checkpoint has them)."""
    out_f, in_f = w.shape
    g = w.float().reshape(out_f, in_f // 32, 32)
    scale = (g.abs().amax(-1) / 7.0).clamp_min(1e-8).to(torch.bfloat16)
    q = torch.round(g / scale.float()[..., None]).clamp(-8, 7).to(torch.int64) + 8
    word = (q.reshape(out_f, in_f // 8, 8) << (torch.arange(8, dtype=torch.int64) * 4)).sum(-1)
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32), scale


def to_packed(mod: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a dense linear into the checkpoint's form: its weight gone, weight_packed and weight_scale buffers."""
    packed, scale = pack_w4a16(mod.weight.detach())
    del mod.weight
    mod.register_buffer("weight_packed", packed)
    mod.register_buffer("weight_scale", scale)
    mod.weight = None
    return packed, scale


def random_packed(rows: int, cols: int, seed: int, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(-(2**31), 2**31 - 1, (rows, cols // 8), generator=g, dtype=torch.int64).to(torch.int32)
    scale = torch.randn(rows, cols // 32, generator=g) * 0.02
    scale.view(-1)[:: 7] = 0.0  # a zero scale now and then: pieces x 0, signed zeros among them
    return packed, scale.to(dtype)


def compile_works() -> Tuple[bool, str]:
    """Whether torch.compile (Inductor, a C++ compiler on the CPU) builds a tiny unpacker here: tried in a child
    process given 240 s, so a compiler that hangs or is missing costs this check nothing but the time."""
    code = ("import torch, sys; sys.path.insert(0, sys.argv[1]); from backend import w4a16_fast as wf; "
            "p = torch.zeros(2, 4, dtype=torch.int32); s = torch.ones(2, 1, dtype=torch.bfloat16); "
            "print(wf.same_bits(wf.dequantize_w4a16_full_compiled(p, s), wf.REFERENCE(p, s)))")
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


def packed_stand_in() -> Tuple[Any, LarqlVaultEngine, int, int, float]:
    """The atlas's stand-in in bf16 with every block linear 4-bit, patched as `load()` patches the checkpoint:
    (runtime, vault, linears packed, linears patched, the largest |w - unpacked| in units of its group's scale)."""
    tok = stand_in_tokenizer()
    hf = stand_in_hf(tok, torch.bfloat16)
    n, worst = 0, 0.0
    for name, mod in list(hf.named_modules()):
        if isinstance(mod, nn.Linear) and ".layers." in f".{name}.":
            dense = mod.weight.detach().float().clone()
            packed, scale = to_packed(mod)
            err = (dense - wf.REFERENCE(packed, scale).float()).abs().view(dense.shape[0], -1, 32)
            worst = max(worst, float((err / scale.float()[..., None]).max()))
            n += 1
    patched = ER.patch_compressed_tensors_for_low_vram(hf)
    rt = stand_in_runtime(hf, tok)
    return rt, LarqlVaultEngine(rt), n, patched, worst


def pinned_forward(mod: nn.Module) -> bool:
    f = getattr(mod.forward, "__func__", None)
    return f is not None and f.__name__ == "_w4a16_forward" and f.__module__ == "backend.engine_runtime"


# ------------------------------------------------------------------------------------------------- 1. the table


def check_table(compiled: bool) -> None:
    before = len(failures)
    p, s = wf.piece_scale_table()
    ref = wf.REFERENCE(p, s)
    lut = wf.dequantize_w4a16_full_lut(p, s)
    nan = torch.isnan(ref)
    bits = ref.view(torch.int16)
    check(tuple(ref.shape) == (65536, 32) and ref.dtype == torch.bfloat16 and wf.TABLE_PRODUCTS == ref.numel(),
          f"the table: {tuple(ref.shape)} {ref.dtype}")
    check(len(set(s.view(torch.int16).flatten().tolist())) == 65536 and int(nan.sum()) > 0 and bool((bits == -32768).any()),
          "the table holds every bf16 scale bit pattern, and NaN and -0 products among them")
    check(wf.same_bits(lut, ref), "lut: the same bits as the reference on all 16 x 65,536 (piece, scale) pairs, NaNs by "
          f"their bits ({int((lut.view(torch.int16) != bits).sum())} differ)")
    check(not torch.equal(lut, ref) and wf.same_bits(ref, ref), "same_bits holds where torch.equal cannot (NaN != NaN)")
    pieces = (torch.arange(32) % 16 - 8).float()
    want = (pieces[None, :] * s.float()).to(torch.bfloat16)  # exact in float32, then one rounding to nearest even
    check(torch.equal(want.view(torch.int16)[~nan], bits[~nan]) and bool(torch.isnan(want)[nan].all()),
          "the reference: (piece - 8) x scale, exact in float32, rounded once to the nearest bf16")
    one = (s.view(torch.int16).flatten() == 0x3F80).nonzero().item()
    check(ref[one].tolist() == pieces.tolist(), "at scale 1 the pieces minus 8 themselves, in column order 8k + j")
    # table_agrees, as install's table test reads it: a number one bit off and a NaN moved disagree, a NaN's own bits not
    nan_at, num_at = int(nan.view(-1).nonzero()[0]), one * 32 + 3

    def edited(at: int, to: Optional[int] = None) -> torch.Tensor:
        t = ref.clone()
        flat = t.view(torch.int16).view(-1)
        flat[at] = (flat[at] ^ 1) if to is None else to
        return t

    check(wf.table_agrees(ref, ref) == (True, 0, 0), "table_agrees: the reference agrees with itself")
    check(wf.table_agrees(edited(num_at), ref) == (False, 1, 0), "table_agrees: one product one bit off disagrees")
    check(wf.table_agrees(edited(nan_at, 0), ref) == (False, 1, 0)
          and wf.table_agrees(edited(num_at, 0x7FC0), ref) == (False, 1, 0),
          "table_agrees: a NaN where the reference has a number, or a number where it has a NaN, disagrees")
    check(int(bits.view(-1)[nan_at]) == -1 and wf.table_agrees(edited(nan_at, 0x7FC0), ref) == (True, 0, 1),
          "table_agrees: a NaN with other bits (0x7fc0 for 0xffff) agrees, counted apart")
    t0 = time.time()
    for _ in range(3):
        wf.dequantize_w4a16_full_lut(p, s)
    lut_s = time.time() - t0
    if compiled:
        comp = wf.dequantize_w4a16_full_compiled(p, s)
        ok, wrong, payload = wf.table_agrees(comp, ref)
        check(ok, f"compiled: the table's products agree with the reference ({wrong} differ)")
        print(f"        compiled on the CPU: every product's bits as the reference's but for {payload:,} NaNs' own "
              f"(Inductor 0x7fc0, eager 0xffff), all from NaN or infinite scales")
        check(payload == 0 or (bool((comp.view(torch.int16)[nan] == 0x7FC0).all()) and bool((bits[nan] == -1).all())),
              "compiled: its NaNs 0x7fc0 where eager torch writes 0xffff")
    section(f"the piece-by-scale table: lut the reference's bits on every pair (3 passes of 2,097,152 in {lut_s:.2f} s)"
            + (", compiled too but for NaNs' own bits" if compiled else ""), before)


# -------------------------------------------------------------------------------------- 2. where it is looked up


def check_call_sites(rt: Any) -> None:
    before = len(failures)
    lm = language_model(rt.hf_model)
    mlp = lm.layers[2].mlp
    idx = torch.tensor([3, 0, 9])
    x = torch.randn(1, 4, mlp.down_proj.weight_packed.shape[1] * 8).to(torch.bfloat16)
    calls: List[Tuple[int, ...]] = []

    def counting(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(packed.shape))
        return wf.REFERENCE(packed, scale)

    spec = importlib.util.spec_from_file_location("labkit", ROOT_DIR / "experiments" / "lab" / "labkit.py")
    assert spec is not None and spec.loader is not None
    labkit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labkit)
    looked_up = [
        ("the pinned forward (_w4a16_forward)", lambda: mlp.down_proj(x)),
        ("dequantize_w4a16_rows", lambda: ER.dequantize_w4a16_rows(mlp.gate_proj.weight_packed, mlp.gate_proj.weight_scale, idx)),
        ("slice_w_gate", lambda: rt.slice_w_gate(2, idx)),
        ("slice_w_up", lambda: rt.slice_w_up(2, idx)),
        ("get_full_w_down", lambda: rt.get_full_w_down(2)),
        ("get_full_w_gate", lambda: rt.get_full_w_gate(2)),
        ("get_full_w_up", lambda: rt.get_full_w_up(2)),
        ("atlas_ops.dense_weight", lambda: ao.dense_weight(mlp.up_proj)),
        ("labkit's _W4A16LinearFn", lambda: labkit._W4A16LinearFn.apply(x, mlp.down_proj.weight_packed, mlp.down_proj.weight_scale)),
    ]
    ER.dequantize_w4a16_full = counting
    try:
        for what, run in looked_up:
            n0 = len(calls)
            run()
            check(len(calls) > n0, f"{what} looks engine_runtime.dequantize_w4a16_full up at call time")
        n0 = len(calls)
        rt.slice_w_down(2, idx)
        check(len(calls) == n0, "slice_w_down goes through dequantize_w4a16_cols, never rebound")
    finally:
        ER.dequantize_w4a16_full = wf.REFERENCE
    section(f"{len(looked_up)} call sites look the unpacker up at call time; slice_w_down's column slicer does not", before)


# ----------------------------------------------------------------------------------- 3. random packed matrices


SHAPES = [(1, 32, torch.bfloat16), (3, 64, torch.bfloat16), (17, 96, torch.bfloat16), (40, 256, torch.bfloat16),
          (5, 1024, torch.bfloat16), (6, 64, torch.float32)]


def holder_of(shapes: List[Tuple[int, int, torch.dtype]]) -> nn.Module:
    """Packed linears of the given shapes (the first with a bias), patched as the checkpoint's are."""
    holder = nn.Module()
    holder.lins = nn.ModuleList()
    for i, (r, c, dt) in enumerate(shapes):
        lin = nn.Linear(c, r, bias=(i == 0)).to(torch.bfloat16)
        del lin.weight
        packed, scale = random_packed(r, c, 100 + i, dt)
        lin.register_buffer("weight_packed", packed)
        lin.register_buffer("weight_scale", scale)
        lin.weight = None
        holder.lins.append(lin)
    check(ER.patch_compressed_tensors_for_low_vram(holder) == len(shapes), "the holder's linears are patched")
    return holder


def forwards(holder: nn.Module, xs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Each linear's output, bare, under a column overlay, under a row overlay, and under both."""
    out: List[torch.Tensor] = []
    for lin, x in zip(holder.lins, xs):
        r, c = lin.weight_packed.shape[0], lin.weight_packed.shape[1] * 8
        col = torch.linspace(-1, 1, r).to(torch.bfloat16)
        row = torch.linspace(1, -1, c).to(torch.bfloat16)
        out.append(lin(x))
        lin._memit_col_overlays = {c - 1: col}
        out.append(lin(x))
        lin._memit_col_overlays, lin._memit_row_overlays = {}, {0: row}
        out.append(lin(x))
        lin._memit_col_overlays = {1: col}
        out.append(lin(x))
        lin._memit_col_overlays, lin._memit_row_overlays = {}, {}
    return out


def check_random(modes: List[str]) -> None:
    before = len(failures)
    for i, (r, c, dt) in enumerate(SHAPES):
        p, s = random_packed(r, c, i, dt)
        ref = wf.REFERENCE(p, s)
        check(wf.same_bits(wf.dequantize_w4a16_full_lut(p, s), ref), f"lut [{r}, {c}] {dt}: the reference's bits")
        if "compiled" in modes:
            check(wf.same_bits(wf.dequantize_w4a16_full_compiled(p, s), ref), f"compiled [{r}, {c}] {dt}: the reference's bits")
        g = torch.Generator().manual_seed(i)
        ri, ci = torch.randint(0, r, (5,), generator=g), torch.randint(0, c, (7,), generator=g)
        check(wf.same_bits(ER.dequantize_w4a16_cols(p, s, ci), ref[:, ci].T.contiguous()),
              f"[{r}, {c}]: the column slicer gives the whole weight's columns")
    holder = holder_of(SHAPES)
    xs = [torch.randn(2, 3, c).to(torch.bfloat16) for _r, c, _d in SHAPES]
    ref_out = forwards(holder, xs)
    for mode in modes:
        rec = wf.install(holder, mode, self_test=(mode == "lut") or None, log=lambda s: None)
        check(rec["mode"] == mode and rec["linears"] == len(SHAPES) and wf.describe(holder)["mode"] == mode, f"{mode}: installed ({rec})")
        for i, (r, c, dt) in enumerate(SHAPES):
            p, s = random_packed(r, c, i, dt)
            ref = wf.REFERENCE(p, s)
            g = torch.Generator().manual_seed(i)
            ri, ci = torch.randint(0, r, (5,), generator=g), torch.randint(0, c, (7,), generator=g)
            check(wf.same_bits(ER.dequantize_w4a16_full(p, s), ref), f"{mode} [{r}, {c}] {dt}: dequantize_w4a16_full")
            check(wf.same_bits(ER.dequantize_w4a16_rows(p, s, ri), ref[ri]), f"{mode} [{r}, {c}] {dt}: dequantize_w4a16_rows")
            check(wf.same_bits(ER.dequantize_w4a16_cols(p, s, ci), ref[:, ci].T.contiguous()), f"{mode} [{r}, {c}]: cols")
        got = forwards(holder, xs)
        check(all(wf.same_bits(a, b) for a, b in zip(got, ref_out)),
              f"{mode}: every forward the pinned forward's output, bare and under column and row overlays")
        if mode == "compiled":
            from torch._dynamo.utils import counters

            graphs = counters["stats"]["unique_graphs"]
            p, s = random_packed(40, 256, 3)
            with torch.inference_mode():  # dynamo's guards fail, so it would compile a graph never tested
                got_im = ER.dequantize_w4a16_full(p, s)
            new = counters["stats"]["unique_graphs"] - graphs
            check(wf.same_layout(got_im, wf.REFERENCE(p, s)) and new == 0 and ER.dequantize_w4a16_full.fell_back == 1,
                  f"compiled: a call its graphs were not tested for, through the reference ({new} compiled after install)")
        if mode == "resident":
            for lin in holder.lins:
                w = lin._resident_w
                ri = torch.tensor([0, w.shape[0] - 1])
                ci = torch.tensor([w.shape[1] - 1, 2])
                check(wf.same_bits(w[ri], ER.dequantize_w4a16_rows(lin.weight_packed, lin.weight_scale, ri))
                      and wf.same_bits(w[:, ci].T.contiguous(), ER.dequantize_w4a16_cols(lin.weight_packed, lin.weight_scale, ci)),
                      "resident: the kept weight's rows and columns are the slicers'")
                check(w._version == lin._resident_version, "resident: the kept weight never written by an overlay")
    wf.install(holder, "reference")
    check(ER.dequantize_w4a16_full is wf.REFERENCE and all(pinned_forward(l) and "_resident_w" not in l.__dict__ for l in holder.lins),
          "reference: the runtime's unpacker and the pinned forwards back, the kept weights let go")
    section(f"random packed matrices ({len(SHAPES)} shapes): whole, rows, columns and forwards with overlays in "
            f"{', '.join(modes)}", before)


# ------------------------------------------------------------------------------------------------- 4. refusals


def check_refusals(rt: Any) -> None:
    before = len(failures)
    lm = language_model(rt.hf_model)

    def refused(run: Any, *words: str) -> Optional[str]:
        try:
            run()
        except wf.Refused as exc:
            msg = str(exc)
            check(all(w in msg for w in words), f"the refusal says {words}: {msg}")
            return msg
        check(False, f"refused ({words})")
        return None

    refused(lambda: wf.install(rt, "fast"), "reference, lut, compiled, resident", "'fast'")
    true_lut = wf.dequantize_w4a16_full_lut

    def off_by_one_bit(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        w = true_lut(packed, scale)
        if packed.shape == lm.layers[4].mlp.up_proj.weight_packed.shape:
            w = w.clone()
            w.view(torch.int16).view(-1)[5] ^= 1
        return w

    wf.dequantize_w4a16_full_lut = off_by_one_bit
    try:
        msg = refused(lambda: wf.install(rt, "lut", self_test=True), "weights differ from the reference's on", "linears",
                      "first: other bits", "nothing was installed")
    finally:
        wf.dequantize_w4a16_full_lut = true_lut
    print(f"        e.g. {msg}")
    check(ER.dequantize_w4a16_full is wf.REFERENCE and wf.describe(rt)["mode"] == "reference",
          "a self-test that fails leaves the reference bound")

    def off_on_the_table(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        w = true_lut(packed, scale)
        if tuple(packed.shape) == (65536, 4):  # one product at scale 1.0 one bit off, nowhere a linear reads
            w = w.clone()
            w.view(torch.int16)[0x3F80 ^ 0x8000, 3] ^= 1
        return w

    def laid_out_otherwise(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        w = true_lut(packed, scale)
        return w.t().contiguous().t() if packed.shape == lm.layers[4].mlp.up_proj.weight_packed.shape else w

    for fake, words, what in ((off_on_the_table, ("reference's on the piece-by-scale table",
                                                   "1 of its 2,097,152 products differ", "(0x3f80)"),
                               "one product off on the table alone"),
                              (laid_out_otherwise, ("reference's on", "linears", "the same bits laid out otherwise, strides"),
                               "the same bits laid out otherwise")):
        wf.dequantize_w4a16_full_lut = fake
        try:
            msg = refused(lambda: wf.install(rt, "lut", self_test=True), *words, "nothing was installed")
        finally:
            wf.dequantize_w4a16_full_lut = true_lut
        print(f"        e.g. {msg}")
        check(ER.dequantize_w4a16_full is wf.REFERENCE and wf.describe(rt)["mode"] == "reference",
              f"a lut with {what} refused, the reference left bound")
    true_make = wf._make_resident_forward
    odd = lm.layers[3].mlp.down_proj

    def one_bit_off(module: nn.Module) -> Any:
        f = true_make(module)
        if module is not odd:
            return f

        def off(x: torch.Tensor) -> torch.Tensor:
            y = f(x).clone()
            y.view(torch.int16).view(-1)[0] ^= 1
            return y

        return off

    wf._make_resident_forward = one_bit_off
    try:
        msg = refused(lambda: wf.install(rt, "resident"), "output on a probe differs from the pinned forward's",
                      "on 1 of 41 linears", "layers.3.mlp.down_proj first", "nothing was installed")
    finally:
        wf._make_resident_forward = true_make
    print(f"        e.g. {msg}")
    check(all(pinned_forward(m) and "_resident_w" not in m.__dict__ for _n, m in wf.packed_linears(rt.hf_model)),
          "a resident forward off the pinned one refused: every forward the pinned one, nothing kept")
    true_free = wf.free_bytes
    wf.free_bytes = lambda device: 4096
    try:
        msg = refused(lambda: wf.install(rt, "resident"), "Resident weights need", "bytes", "are free there")
    finally:
        wf.free_bytes = true_free
    print(f"        e.g. {msg}")
    need = sum(int(m.weight_packed.numel()) * 8 * 2 for _n, m in wf.packed_linears(rt.hf_model))
    check(msg is not None and f"{need:,} bytes" in msg, f"the bytes needed said: {need:,}")
    check(all(pinned_forward(m) and "_resident_w" not in m.__dict__ for _n, m in wf.packed_linears(rt.hf_model)),
          "resident refused: every forward the pinned one, nothing kept")
    check(isinstance(wf.free_bytes("cpu"), int) and wf.free_bytes("cpu") > 0, f"MemAvailable read: {wf.free_bytes('cpu')}")
    wf.install(rt, "resident", log=lambda s: None)
    lin = lm.layers[1].self_attn.q_proj
    lin._resident_w.add_(0)  # an edit in place, however small
    ids = torch.tensor([[2, 30, 31]])
    try:
        rt.hf_model(input_ids=ids)
        check(False, "a resident weight written in place is caught")
    except RuntimeError as exc:
        check("written in place" in str(exc) and "layers.1.self_attn.q_proj" in str(exc), f"caught: {exc}")
    wf.install(rt, "reference")
    section("refusals in plain words, the reference left bound: an unknown unpacker, a lut off by one bit on a linear or "
            "on the table alone or laid out otherwise, resident weights that do not fit, a resident forward one bit off; "
            "a kept weight written in place caught", before)


# ----------------------------------------------------------------------------------------- 5. the 4-bit stand-in


def readings(rt: Any, vault: LarqlVaultEngine, out: Path) -> Dict[str, Any]:
    """Every reading the check compares across unpackers, in the runtime's current mode."""
    from backend.mcp_scientist import _study_ids

    hf, lm = rt.hf_model, language_model(rt.hf_model)
    tok = rt.tokenizer
    ids = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("Who are you?", chat=True))])
    idx = torch.tensor([5, 7, 0, 127, 64])
    r: Dict[str, Any] = {"logits": hf(input_ids=ids).logits}
    # a vault edit: one down_proj column overlaid (UPDATE, the real apply_patch_surgery)
    vault.apply_patch_surgery("UPDATE", 3, 5, target_concept=100, strength=1.5, statement="check_dequant")
    r["logits_col"] = hf(input_ids=ids).logits
    r["slice_w_down"], r["full_w_down"] = rt.slice_w_down(3, idx), rt.get_full_w_down(3)
    lm.layers[3].mlp.down_proj._memit_col_overlays.clear()
    vault.active_patches.clear()
    # a row overlay on gate_proj (the pinned forward applies one; no vault statement writes one today)
    gate = lm.layers[2].mlp.gate_proj
    gate._memit_row_overlays = {7: torch.linspace(-1, 1, gate.weight_packed.shape[1] * 8).to(torch.bfloat16)}
    r["logits_row"] = hf(input_ids=ids).logits
    r["slice_w_gate"], r["full_w_gate"] = rt.slice_w_gate(2, idx), rt.get_full_w_gate(2)
    r["slice_w_up"], r["full_w_up"] = rt.slice_w_up(2, idx), rt.get_full_w_up(2)
    r["dense_weight"] = ao.dense_weight(lm.layers[1].self_attn.o_proj)
    gate._memit_row_overlays = {}
    r["logits_after"] = hf(input_ids=ids).logits
    p50 = ao.stream_p50(ao.activation_extremes(lm, _study_ids(rt)))
    r["p50"] = [float(x) for x in p50]
    w = ao.Weights(lm, rt)
    # the wiring check's reads under +-d, and where each reader sits unpushed: its gate_proj and up_proj outputs, read by
    # hooks on those modules, whose forwards each unpacker replaces (with, beside them, plain hooks of the check's own
    # on an unpushed forward, which the reference's are set against)
    readers = [[(2, 5), (4, 17)], [(3, 1), (5, 2), (4, 0)]]
    reads, g0, u0 = ao.pushed_reads(lm, w, [ids[0].tolist()], [(1, 3), (2, 9)], readers, [0.5, 1.0], bos_id=tok.bos_token_id, batch=5)
    r["pushed_reads"], r["pushed_g0"], r["pushed_u0"] = (torch.from_numpy(x) for x in (reads, g0, u0))
    sits: Dict[str, Any] = {}
    hooks = [h for j in (2, 3, 4, 5) for h in (
        lm.layers[j].mlp.gate_proj.register_forward_hook(lambda _m, _a, o, j=j: sits.__setitem__(f"g{j}", o[0, 1:].float().clone())),
        lm.layers[j].mlp.up_proj.register_forward_hook(lambda _m, _a, o, j=j: sits.__setitem__(f"u{j}", o[0, 1:].float().clone())))]
    try:
        lm(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    r["sits_by_hook"] = torch.stack([torch.stack([sits[f"{k}{j}"][:, b] for k in ("g", "u")]) for rs in readers for j, b in rs])
    # three answers at once, each under its own push (none, +3, -2 x p50 along neuron 9's write direction)
    u = w.write_dirs(2, [9])[0]
    delta = torch.stack([torch.zeros_like(u), u * (3.0 * p50[2]), u * (-2.0 * p50[2])])
    steps = ao.write_rows(lm, rt.lm_head_weight, [ids[0].tolist()] * 3, ao.Interventions(rows=3, adds={2: delta}),
                          max_new=8, stop_ids=rt.end_of_turn_ids(), bos_id=tok.bos_token_id, spare_bos=True,
                          softcap=ao.softcap_of(hf), top_k=5)
    r["write_rows"] = [(s.row, s.token_id, s.logprob, s.top_ids, s.top_lps, s.end) for s in steps]
    res = cz.take_census(rt, 3, [0, 5], out, p50=p50[3], p50_source="check_dequant", design_name="main_rel", controls=1,
                         seed=3, max_new=6, top_k=5, batch=5, log=lambda s: None)
    r["census_counts"] = res
    r["census_rows"] = out.read_bytes()
    r["census_meta"] = json.loads(cz.meta_path_of(out).read_text(encoding="utf-8"))
    return r


def check_stand_in(rt: Any, vault: LarqlVaultEngine, modes: List[str], tmp: Path) -> None:
    before = len(failures)
    torch.manual_seed(0)
    t0 = time.time()
    ref = readings(rt, vault, tmp / "reference.jsonl")
    print(f"        the reference's readings in {time.time() - t0:.1f} s: {ref['census_counts']['written']} census answers")
    check(ref["census_counts"] == {"written": 34, "skipped": 0, "stopped": False}, f"the census: {ref['census_counts']}")
    check(ref["census_meta"]["unpacker"]["mode"] == "reference" and ref["census_meta"]["unpacker"]["linears"] == 41,
          f"the census's meta names its unpacker: {ref['census_meta']['unpacker']}")
    check(not wf.same_bits(ref["logits"], ref["logits_col"]) and not wf.same_bits(ref["logits"], ref["logits_row"])
          and wf.same_bits(ref["logits"], ref["logits_after"]), "the overlays move the logits, and lifting them puts them back")
    check(len({s[1] for s in ref["write_rows"] if s[0] == 1} ^ {s[1] for s in ref["write_rows"] if s[0] == 0}) > 0,
          "the push changes what is written")
    # where each reader sits, as the wiring check reads it on the packed weights: what plain hooks give (another batch
    # shape, so to bf16's rounding), every value read, and the unpushed read gelu_tanh(g0) u0 to bf16's rounding
    sat = torch.stack([torch.stack([ref[f"pushed_{k}"][a, r_, 0] for k in ("g0", "u0")]) for a, n_r in enumerate((2, 3)) for r_ in range(n_r)])
    act = torch.nn.functional.gelu(ref["pushed_g0"][..., -1], approximate="tanh") * ref["pushed_u0"][..., -1]
    have = torch.isfinite(ref["pushed_reads"][:, :, 0, 0, 0])
    check(bool(torch.isfinite(sat).all()) and float((sat - ref["sits_by_hook"]).abs().max()) <= 2e-2 * max(1.0, float(ref["sits_by_hook"].abs().max()))
          and float((act[:, :, 0][have] - ref["pushed_reads"][:, :, 0, 0, 0][have]).abs().max()) <= 2e-2 * float(act[:, :, 0][have].abs().max()),
          "the wiring check on the 4-bit weights: each reader's g0 and u0 read through the hooks on gate_proj and up_proj, as plain hooks give "
          f"them, and the unpushed read gelu(g0) u0 ({float((sat - ref['sits_by_hook']).abs().max()):.1e} off)")
    for mode in modes:
        t0 = time.time()
        rec = wf.install(rt, mode, log=lambda s: None)
        check(rec["linears"] == 41 and rec["shapes"] == 4 and (rec["self_test"] or {}).get("same_bits", mode == "lut"),
              f"{mode}: installed on the stand-in's 41 linears ({rec.get('self_test')})")
        if mode == "compiled":
            check(len(getattr(ER.dequantize_w4a16_full, "tested", ())) == 4,
                  "compiled: the stand-in's 4 shapes go through the compiled unpacker, each tested at install")
        got = readings(rt, vault, tmp / f"{mode}.jsonl")
        dt = time.time() - t0
        if mode == "compiled":
            check(ER.dequantize_w4a16_full.fell_back == 0, "compiled: every reading through the graphs tested at install, "
                  f"none sent to the reference ({ER.dequantize_w4a16_full.fell_back})")
        for k, v in ref.items():
            if k == "census_meta":
                strip = lambda m: {a: b for a, b in m.items() if a not in ("unpacker", "started_at")}  # noqa: E731
                check(strip(got[k]) == strip(v) and got[k]["unpacker"]["mode"] == mode,
                      f"{mode}: the census meta as the reference's, but for its unpacker ({got[k]['unpacker']['mode']})")
            elif isinstance(v, torch.Tensor):
                check(wf.same_bits(got[k], v), f"{mode}: {k} the reference's bits")
            else:
                check(got[k] == v, f"{mode}: {k} as the reference's")
        if mode == "resident":
            lins = wf.packed_linears(rt.hf_model)
            check(all(wf.same_bits(m._resident_w, wf.REFERENCE(m.weight_packed, m.weight_scale))
                      and m._resident_w._version == m._resident_version for _n, m in lins),
                  "resident: every kept weight still the reference's bits, never written in place, after all of it")
        print(f"        {mode}: installed and read in {dt:.1f} s")
    # the reference's run continued under lut (nothing is left to write, so only its meta moves), then merged with lut's
    wf.install(rt, "lut", log=lambda s: None)
    res = cz.take_census(rt, 3, [0, 5], tmp / "reference.jsonl", p50=ref["p50"][3], p50_source="check_dequant",
                         design_name="main_rel", controls=1, seed=3, max_new=6, top_k=5, batch=5, log=lambda s: None)
    meta = json.loads(cz.meta_path_of(tmp / "reference.jsonl").read_text(encoding="utf-8"))
    check(res["written"] == 0 and [u["mode"] for u in meta.get("unpacker_sessions", [])] == ["reference", "lut"]
          and meta["unpacker"]["mode"] == "lut",
          f"a run continued under another unpacker keeps each session's: {meta.get('unpacker_sessions')}")
    cz.merge_runs([tmp / "reference.jsonl", tmp / "lut.jsonl"], tmp / "merged.jsonl")
    merged = json.loads(cz.meta_path_of(tmp / "merged.jsonl").read_text(encoding="utf-8"))
    check([u["mode"] for u in merged.get("unpacker_sessions", [])] == ["reference", "lut", "lut"],
          f"a merged run keeps every shard's sessions' unpackers: {[u['mode'] for u in merged.get('unpacker_sessions', [])]}")
    wf.install(rt, "reference")
    lins = wf.packed_linears(rt.hf_model)
    check(ER.dequantize_w4a16_full is wf.REFERENCE and all(pinned_forward(m) and "_resident_w" not in m.__dict__ for _n, m in lins),
          "reference: the runtime's own unpacker and the pinned forward on every linear, as loaded")
    section(f"the 4-bit stand-in, byte for byte in {', '.join(modes)}: logits bare and under a column and a row overlay, "
            "slices and whole weights, p50, three pushed answers at once, a census run's rows; each session's unpacker kept "
            "on resume and merge", before)


# ---------------------------------------------------------------------------------- 6. command lines and server


def check_wiring(rt: Any, vault: LarqlVaultEngine, tmp: Path) -> None:
    before = len(failures)
    env = {**os.environ, "THEODOLITE_ATLAS_DIR": str(tmp / "atlas"), "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    env.pop("THEODOLITE_CENSUS_DIR", None)
    py = [sys.executable, "-I"]
    runs = tmp / "runs"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--neurons", "0",
                             "--random-controls", "0", "--max-new", "3", "--batch", "7", "--unpacker", "lut",
                             "--out-dir", str(runs), "--no-rank"], env=env, capture_output=True, text=True, timeout=600)
    meta_p = cz.meta_path_of(runs / "L2-main_rel.jsonl")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    # the stand-in's linears are dense, not W4A16: the reference is recorded, with what was asked
    un = meta.get("unpacker") or {}
    check(r.returncode == 0 and un.get("mode") == "reference" and un.get("asked") == "lut" and un.get("linears") == 0,
          f"census_run.py --unpacker lut on a model with no W4A16 linears: the reference recorded, lut asked ({un}; {r.stderr[-300:]})")
    check("No W4A16 linears in this model" in r.stdout, f"census_run.py says why: {r.stdout[:300]}")
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "census_run.py"), "--stand-in", "--layer", "2", "--unpacker", "fast"],
                       env=env, capture_output=True, text=True, timeout=300)
    check(r.returncode != 0 and "invalid choice: 'fast'" in r.stderr, f"census_run.py: an unknown unpacker refused ({r.stderr[-200:]})")
    idx = tmp / "index"
    r = subprocess.run(py + [str(ROOT_DIR / "scripts" / "atlas_build.py"), "layout", "--stand-in", "--unpacker", "lut",
                             "--layout-method", "pca", "--out", str(idx)], env=env, capture_output=True, text=True, timeout=600)
    lay = idx / FILES["layout"]
    m = json.loads(str(np.load(lay)["meta"])) if lay.exists() else {}
    check(r.returncode == 0 and (m.get("unpacker") or {}).get("mode") == "reference" and (m.get("unpacker") or {}).get("asked") == "lut",
          f"atlas_build.py --unpacker lut: recorded in the index's meta, the reference on a dense model ({m.get('unpacker')}; {r.stderr[-300:]})")

    # the server's own lifespan, over the 4-bit stand-in in place of the model it loads
    from fastapi.testclient import TestClient

    import backend.server as server

    class Loads:
        def __init__(self, **_kw: Any) -> None:
            pass

        def load(self) -> Any:
            return rt

    # the server builds its runtime through runtime_factory.make_runtime (THEODOLITE_GGUF / THEODOLITE_DEVICE), so both
    # names it could reach are stood in for
    true_rt, true_make, true_env = server.TheodoliteRuntime, server.make_runtime, os.environ.get(wf.UNPACKER_ENV)
    server.TheodoliteRuntime = Loads  # type: ignore[assignment,misc]
    server.make_runtime = Loads  # type: ignore[assignment]
    try:
        os.environ[wf.UNPACKER_ENV] = "resident"
        with TestClient(server.app, base_url="http://127.0.0.1:8765") as client:
            health = client.get("/api/health").json()
        un = health.get("unpacker") or {}
        check(un.get("mode") == "resident" and un.get("linears") == 41 and (un.get("self_test") or {}).get("same_bits") is True,
              f"the server installs $THEODOLITE_UNPACKER and /api/health says so: {un}")
        check(list(health)[-2:] == ["label_gate", "lens"], f"/api/health: label_gate and lens still last ({list(health)[-3:]})")
        os.environ[wf.UNPACKER_ENV] = "quick"
        try:
            with TestClient(server.app, base_url="http://127.0.0.1:8765"):
                pass
            check(False, "the server refuses an unpacker that is not one")
        except wf.Refused as exc:
            check("not 'quick'" in str(exc), f"the server stops in plain words: {exc}")
        os.environ.pop(wf.UNPACKER_ENV)
        with TestClient(server.app, base_url="http://127.0.0.1:8765") as client:
            un = client.get("/api/health").json().get("unpacker") or {}
        check(un.get("mode") == "reference", f"without $THEODOLITE_UNPACKER, the reference: {un}")
    finally:
        server.TheodoliteRuntime = true_rt  # type: ignore[misc]
        server.make_runtime = true_make  # type: ignore[assignment]
        if true_env is None:
            os.environ.pop(wf.UNPACKER_ENV, None)
        else:
            os.environ[wf.UNPACKER_ENV] = true_env
        wf.install(rt, "reference")
    section("the flags and the server: --unpacker in census_run.py and atlas_build.py, recorded in their metas; "
            "$THEODOLITE_UNPACKER in the server's lifespan and /api/health", before)


# ------------------------------------------------------------------------------------------------------ 7. pins


def check_pins() -> None:
    before = len(failures)
    tree = ast.parse((ROOT_DIR / "experiments" / "scripts" / "R4_common.py").read_text(encoding="utf-8"))
    table: Dict[str, str] = {}
    for node in ast.walk(tree):
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "PINNED_UNCHANGED" and node.value is not None:
            table = ast.literal_eval(node.value)
    for rel in R4_RUNTIME_FILES:
        got = hashlib.sha256((ROOT_DIR / rel).read_bytes()).hexdigest()
        check(table.get(rel) == got, f"{rel}: sha256 {got[:12]}, R4 pins {str(table.get(rel))[:12]}")
    section("the four runtime files R4 pins keep their sha256 (R4_common.PINNED_UNCHANGED)", before)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-compiled", action="store_true", help='leave "compiled" out (it needs a C++ compiler here)')
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    tmp = Path(tempfile.mkdtemp(prefix="dequant-check-"))
    os.environ["THEODOLITE_ATLAS_DIR"] = str(tmp / "atlas")
    use_stand_in_atlas()
    try:
        if a.no_compiled:
            compiled, why = False, "--no-compiled"
        else:
            compiled, why = compile_works()
        print(f"  {'..' if compiled else '--'}    \"compiled\" {'checked' if compiled else 'left out'}: {why}")
        modes = ["lut"] + (["compiled"] if compiled else []) + ["resident"]
        rt, vault, n, patched, worst = packed_stand_in()
        print(f"the 4-bit stand-in: {n} linears packed, {patched} patched, |w - unpacked| at most {worst:.3f} of a scale")
        check(n == patched == 41 and worst <= 0.5 + 1 / 16,
              "the stand-in's linears quantised as the checkpoint's are, within half a step and a bf16 rounding")
        check_table(compiled)
        check_call_sites(rt)
        check_random(modes)
        check_refusals(rt)
        check_stand_in(rt, vault, modes, tmp)
        check_wiring(rt, vault, tmp)
        check_pins()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
