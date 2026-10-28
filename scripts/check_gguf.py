#!/usr/bin/env python3
"""A CPU check of the GGUF path (backend/gguf_weights.py, backend/gguf_model.py, backend/gguf_runtime.py,
backend/runtime_factory.py, scripts/compare_gguf_ct.py) on the tiny random Gemma 4 of scripts/atlas_stand_in.py:
1. the unpackers against gguf-py's own dequantiser (gguf 0.19), bit for bit (NaNs where its NaNs are): every F16 and
   BF16 pattern; every int8 code of Q8_0 and every nibble of Q4_0 under every fp16 scale; every fp16 scale of Q4_1,
   Q5_0, Q5_1 and the K types (and every minimum of Q4_1) beside random bytes; random blocks of every type; the block
   sizes against gguf's table; the IQ types refused in plain words; on the GPU too when there is one;
2. the GGUF layer and the accessors: rows and columns unpacked alone equal the whole weight's; packed, resident and
   chunked unpacking give the same weight and forward; the vault's column and row overlays laid on as the pinned W4A16
   forward lays them, columns and rows together and a row alone (a Q4_0 layer holding a W4A16 module's codes and scales
   computes its forward bit for bit, overlays and all, its stored weight untouched); dense, dense_cols and dense_rows
   alike on a plain Linear, a W4A16 module and a GGUF layer;
3. the stand-in written to tiny GGUF files (F32, F16, BF16, Q8_0, Q4_0; Q4_K, with llama.cpp's Q8_0 fallback where a
   row is not 256 long, on a stand-in four times as wide) and read back: the config from the metadata is the
   stand-in's; every tensor is gguf's dequantised; the forward equals the stand-in carrying those numbers, bit for bit
   in float32 on the CPU, with the layers packed, resident and dense; the norm convention by a round trip (the file
   holds w, transformers' own GGUF reader reads w for gemma4 and w - 1 under Gemma 3's rule, and a file holding w + 1
   does not give the stand-in's forward); a matching config.json taken, one that differs refused, and one standing in
   for keys or the rope_freqs tensor a file leaves out (which alone is refused, asking for it); an IQ tensor, a
   mixture of experts, per-layer embeddings and another arch refused; the provenance (size, the header's and both
   ends' sha256, the tensor types, a fingerprint that moves when a byte of the data does); the tokenizer from the
   file's own metadata encodes as the stand-in's does; the 31B's metadata, as llama.cpp writes it, built on the meta
   device into its 410 linear layers and 29.287 billion quantised weights;
4. GGUFRuntime on the Q4_0 file, packed, against the stand-in's runtime carrying the same numbers: every accessor that
   reads the packing, write_col_norms by hand, a push written in three voices and a neuron read, the lens-free parts
   of a sighting (the rows it captures, the naked eye's readout), a vault DELETE on a packed layer, a column written
   in its place and a gate row (heard by the forward, seen by every accessor and write_col_norms as the pinned ones see
   them, the blocks untouched), and a census taken row for row alike;
5. the choice of runtime from the environment (runtime_factory.make_runtime), and census_run.py on a GGUF on the CPU;
6. scripts/compare_gguf_ct.py's self-test: a checkpoint in the W4A16 layout and a Q4_0 GGUF written from the same
   codes and scales are found identical, again with every other block written the other way round (d = -scale, codes
   16 - q, as llama.cpp's quantiser writes a negative d), and a changed code, a changed scale and a +1 norm convention
   are each found;
7. the files R4 pins (backend/engine_runtime.py among them) are unchanged (experiments/scripts/R4_common.pinned_differ).

Run: .venv/bin/python -I scripts/check_gguf.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")  # transformers' GGUF reader shows a progress bar

import json
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gguf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import atlas_stand_in as asi
import compare_gguf_ct as cmp
from backend import atlas_ops as ao
from backend import census as cz
from backend import engine_runtime as er
from backend import gguf_model as gm
from backend import gguf_weights as gw
from backend import runtime_factory as rf
from backend.causal_check import language_model
from backend.gguf_runtime import GGUFRuntime
from backend.larql_vault import LarqlVaultEngine
from backend.theodolite_math import unembed_gemma4
from interp_engine import Address

failures: List[str] = []
passed = 0
QT = gguf.GGMLQuantizationType


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


def same_bits(a: torch.Tensor, b: Any) -> bool:
    """float32 `a` against gguf's float32 `b`: NaN where it has NaN, every other element the same bits."""
    b = b.detach().cpu().float().clone() if isinstance(b, torch.Tensor) else torch.from_numpy(np.array(b, dtype=np.float32))
    b = b.reshape(a.shape)
    a = a.detach().cpu().contiguous()
    na, nb = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(na, nb) and torch.equal(a[~na].view(torch.int32), b[~nb].view(torch.int32)))


def gg(raw: np.ndarray, qtype: str) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gguf.quants.dequantize(raw, QT[qtype])


def ours(raw: np.ndarray, qtype: str, device: str = "cpu") -> torch.Tensor:
    return gw.unpack(torch.from_numpy(np.ascontiguousarray(raw)).to(device), qtype).cpu()


# The byte ranges of each type's fp16 fields (scale d, and minimum or dmin where it has one).
F16_FIELDS = {"Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q2_K": [80, 82],
              "Q3_K": [108], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}
BLOCK_TYPES = list(F16_FIELDS)


def random_blocks(qtype: str, n: int, rng: np.random.Generator, finite: bool = True) -> np.ndarray:
    """n random blocks of `qtype` (uint8 [n, bytes]), their fp16 fields finite (uniform in +-2) unless not asked."""
    bs, ts = gw.block_shape(qtype)
    b = rng.integers(0, 256, (n, ts), dtype=np.uint8)
    if finite:
        for off in F16_FIELDS[qtype]:
            b[:, off : off + 2] = rng.uniform(-2, 2, (n, 1)).astype(np.float16).view(np.uint8)
    return b


def every_f16(qtype: str, rng: np.random.Generator, field: int = 0) -> np.ndarray:
    """65,536 random blocks whose `field`-th fp16 field runs through every fp16 pattern (NaNs and infinities too)."""
    b = random_blocks(qtype, 65536, rng)
    off = F16_FIELDS[qtype][field]
    b[:, off : off + 2] = np.arange(65536, dtype=np.uint16).view(np.uint8).reshape(-1, 2)
    return b


def stand_in_with(tok: Any, path: Path, hidden: int = asi.HIDDEN, ffn: int = asi.FFN, seed: int = 0) -> nn.Module:
    """The stand-in (same seed, same shape as the file), carrying the file's dequantised numbers (float32)."""
    ref = wide_stand_in(tok, hidden, ffn, seed)
    r = gguf.GGUFReader(str(path))
    tt = {t.name: t for t in r.tensors}
    to_g = gm.hf_to_gguf_names(len(ref.model.layers))
    with torch.no_grad():
        for name, prm in ref.named_parameters():
            if name == "lm_head.weight":
                continue
            t = tt[to_g(name)]
            prm.copy_(torch.from_numpy(np.array(gg(t.data, t.tensor_type.name))).reshape(prm.shape))
        for name, b in ref.named_buffers():
            if name.endswith("layer_scalar"):
                t = tt[to_g(name)]
                b.copy_(torch.from_numpy(np.array(gg(t.data, t.tensor_type.name))).reshape(b.shape))
    return ref


def wide_stand_in(tok: Any, hidden: int, ffn: int, seed: int = 0) -> nn.Module:
    old = (asi.HIDDEN, asi.FFN)
    asi.HIDDEN, asi.FFN = hidden, ffn
    try:
        return asi.stand_in_hf(tok, seed=seed)
    finally:
        asi.HIDDEN, asi.FFN = old


def model_tensors_equal(m: nn.Module, path: Path) -> bool:
    """Every weight the built model holds (unpacked when packed) is gguf's dequantised tensor, bit for bit."""
    r = gguf.GGUFReader(str(path))
    tt = {t.name: t for t in r.tensors}
    to_g = gm.hf_to_gguf_names(len(m.model.layers))
    ok = True
    for name, mod in m.named_modules():
        if isinstance(mod, (gw.GGUFLinear, nn.Linear)) and name != "lm_head":
            t = tt[to_g(name + ".weight")]
            ok &= same_bits(gw.dense(mod).float(), gg(t.data, t.tensor_type.name).reshape(mod.out_features, mod.in_features))
    for name, p in m.named_parameters():
        if p.dim() == 1 or name.endswith("embed_tokens.weight"):
            t = tt[to_g(name)]
            ok &= same_bits(p.float(), gg(t.data, t.tensor_type.name).reshape(p.shape))
    for name, b in m.named_buffers():
        if name.endswith("layer_scalar"):
            t = tt[to_g(name)]
            ok &= same_bits(b.float(), gg(t.data, t.tensor_type.name).reshape(b.shape))
    return bool(ok)


class W4A16Module(nn.Module):
    """A compressed-tensors W4A16 linear as the pinned runtime patches one (`_make_w4a16_linear_forward`)."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.weight_packed = packed
        self.weight_scale = scale
        self.bias = None
        self._memit_col_overlays: Dict[int, torch.Tensor] = {}
        self._memit_row_overlays: Dict[int, torch.Tensor] = {}
        self.forward = er._make_w4a16_linear_forward(self)


def quiet(*_a: Any, **_k: Any) -> None:
    pass


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gguf-check-"))
    try:
        return run_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_checks(tmp: Path) -> int:
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

    # ---------------------------------------------------------------------- 1. the unpackers against gguf-py's own
    before = len(failures)
    for code, (name, bs, ts) in gw.GGML_TYPES.items():
        check(gguf.GGML_QUANT_SIZES[QT(code)] == (bs, ts) and QT(code).name == name, f"{name}: block size {bs}, {ts} bytes, as gguf's table")
    for bad in ("IQ4_NL", "IQ2_XXS", "IQ4_XS", "TQ1_0", 16, 23):
        try:
            gw.type_code(bad)
            check(False, f"{bad} refused")
        except ValueError as exc:
            check("not read yet" in str(exc), f"{bad} refused in plain words: {exc}")
    for dev in devices:
        pat = np.arange(65536, dtype=np.uint16).view(np.uint8).reshape(1, -1)
        for qt in ("F16", "BF16"):
            check(same_bits(ours(pat, qt, dev), gg(pat, qt)), f"{qt} on {dev}: all 65,536 patterns")
        f32 = rng.integers(0, 256, (64, 4096), dtype=np.uint8)
        f32[0, :8] = np.array([0x7F800001, 0xFFC00000], dtype=np.uint32).view(np.uint8)  # two NaN patterns
        check(same_bits(ours(f32, "F32", dev), gg(f32, "F32")), f"F32 on {dev}: random patterns")
        # Q8_0: every fp16 scale x every int8 code (8 blocks of 32 codes per scale)
        q8 = np.zeros((65536 * 8, 34), dtype=np.uint8)
        q8[:, :2] = np.repeat(np.arange(65536, dtype=np.uint16), 8).view(np.uint8).reshape(-1, 2)
        q8[:, 2:] = np.tile(np.arange(256, dtype=np.uint8).reshape(8, 32), (65536, 1))
        check(same_bits(ours(q8, "Q8_0", dev), gg(q8, "Q8_0")), f"Q8_0 on {dev}: every int8 code under every fp16 scale (16.8M)")
        # Q4_0: every fp16 scale x every nibble, low and high
        q4 = np.zeros((65536, 18), dtype=np.uint8)
        q4[:, :2] = np.arange(65536, dtype=np.uint16).view(np.uint8).reshape(-1, 2)
        q4[:, 2:] = (np.arange(16) | ((15 - np.arange(16)) << 4)).astype(np.uint8)
        check(same_bits(ours(q4, "Q4_0", dev), gg(q4, "Q4_0")), f"Q4_0 on {dev}: every nibble under every fp16 scale")
        for qt in BLOCK_TYPES:
            for field in range(len(F16_FIELDS[qt])):
                b = every_f16(qt, rng, field)
                check(same_bits(ours(b, qt, dev), gg(b, qt)), f"{qt} on {dev}: every fp16 pattern of its field {field}, random bytes beside")
            b = random_blocks(qt, 4096, rng)
            check(same_bits(ours(b, qt, dev), gg(b, qt)), f"{qt} on {dev}: 4,096 random blocks")
            rows = random_blocks(qt, 6 * 4, rng).reshape(6, -1)  # [rows, row bytes], as a file holds a weight
            check(same_bits(ours(rows, qt, dev), gg(rows, qt)), f"{qt} on {dev}: rows of several blocks")
    if len(devices) == 1:
        print("  --    no GPU here: the unpackers ran on the CPU only")
    section("the unpackers equal gguf-py's dequantiser bit for bit: F16/BF16/Q8_0/Q4_0 exhaustively, every fp16 field "
            "of every block type, random blocks of all " + str(len(gw.GGML_TYPES)) + " types", before)

    # --------------------------------------------------------------------- 2. the GGUF layer and the accessors
    before = len(failures)
    out_f, in_f = 24, 512
    x = torch.randn(3, in_f, generator=torch.Generator().manual_seed(1))
    idx = torch.tensor([0, 31, 32, 255, 256, 300, 511, 31])
    ridx = torch.tensor([5, 0, 23, 5])
    for qt in ["F32", "F16", "BF16"] + BLOCK_TYPES:
        bs, ts = gw.block_shape(qt)
        if qt in F16_FIELDS:
            raw = random_blocks(qt, out_f * in_f // bs, rng).reshape(out_f, -1)
        else:
            raw = gguf.quants.quantize(rng.standard_normal((out_f, in_f)).astype(np.float32), QT[qt]).view(np.uint8).reshape(out_f, -1)
        full = torch.from_numpy(np.asarray(gg(raw, qt), dtype=np.float32)).reshape(out_f, in_f)
        lin = gw.GGUFLinear(torch.from_numpy(raw.copy()), qt, in_f, out_f, dtype=torch.float32)
        chunked = gw.GGUFLinear(torch.from_numpy(raw.copy()), qt, in_f, out_f, dtype=torch.float32, chunk_rows=7)
        res = gw.GGUFLinear(torch.from_numpy(raw.copy()), qt, in_f, out_f, dtype=torch.float32, resident=True)
        check(same_bits(lin.unpacked(), full) and same_bits(chunked.unpacked(), full) and same_bits(res.unpacked(), full)
              and res.weight_blocks is None, f"{qt}: packed, chunked and resident unpack to gguf's weight (resident drops the blocks)")
        for m_ in (lin, res):
            check(same_bits(m_.unpacked_cols(idx), full[:, idx].T) and same_bits(m_.unpacked_rows(ridx), full[ridx]),
                  f"{qt} {'resident' if m_.resident else 'packed'}: columns (one block per row each) and rows alone equal the whole weight's")
        check(torch.equal(lin(x), F.linear(x, full)) and torch.equal(res(x), F.linear(x, full)), f"{qt}: the forward is F.linear on the weight")
        check(torch.equal(gw.dense(lin), full) and torch.equal(gw.dense_cols(lin, idx), full[:, idx].T)
              and torch.equal(gw.dense_rows(lin, ridx), full[ridx]), f"{qt}: dense, dense_cols, dense_rows")
    # a W4A16 module and a Q4_0 layer holding the same codes and scales (bf16 scales an fp16 also holds)
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 16, (out_f, in_f), generator=g, dtype=torch.uint8)
    scale = (2.0 ** -torch.randint(4, 9, (out_f, in_f // 32), generator=g).float()
             * (1 + torch.randint(0, 128, (out_f, in_f // 32), generator=g).float() / 128)).to(torch.bfloat16)
    ct = W4A16Module(cmp.ct_pack(codes), scale)
    q40 = gw.GGUFLinear(torch.from_numpy(cmp.q4_0_blocks(codes, scale.float())), "Q4_0", in_f, out_f, dtype=torch.bfloat16)
    plain = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16)
    plain.weight.data.copy_(er.dequantize_w4a16_full(ct.weight_packed, ct.weight_scale))
    gw.install_overlay_forward(plain)
    w_ct = er.dequantize_w4a16_full(ct.weight_packed, ct.weight_scale)
    check(torch.equal(gw.dense(ct), w_ct) and torch.equal(gw.dense_cols(ct, idx), er.dequantize_w4a16_cols(ct.weight_packed, ct.weight_scale, idx))
          and torch.equal(gw.dense_rows(ct, ridx), er.dequantize_w4a16_rows(ct.weight_packed, ct.weight_scale, ridx)),
          "W4A16: dense, dense_cols, dense_rows are the pinned unpackers' numbers")
    for m_, what in ((q40, "a Q4_0 layer"), (plain, "a plain Linear")):
        check(torch.equal(gw.dense(m_), w_ct) and torch.equal(gw.dense_cols(m_, idx), gw.dense_cols(ct, idx))
              and torch.equal(gw.dense_rows(m_, ridx), gw.dense_rows(ct, ridx)), f"{what} holding the same numbers: the three accessors alike")
    xb = x.to(torch.bfloat16)
    check(torch.equal(q40(xb), ct(xb)) and torch.equal(plain(xb), ct(xb)), "the same forward as the pinned W4A16 one, bit for bit (bf16)")
    col, row = torch.randn(out_f, generator=g), torch.randn(in_f, generator=g)
    saved = q40.weight_blocks.clone()
    for m_ in (ct, q40, plain):
        m_._memit_col_overlays[37] = col.clone()
        m_._memit_row_overlays[4] = row.clone()
    want = w_ct.clone()
    want[:, 37] = col.to(torch.bfloat16)
    want[4, :] = row.to(torch.bfloat16)
    check(torch.equal(ct(xb), F.linear(xb, want)), "the pinned W4A16 forward: column overlays, then row overlays, on a clone")
    check(torch.equal(q40(xb), ct(xb)) and torch.equal(plain(xb), ct(xb)), "a Q4_0 layer and a plain Linear lay the overlays on as the pinned forward does")
    check(torch.equal(q40.weight_blocks, saved) and torch.equal(gw.dense(q40), w_ct) and torch.equal(plain.weight, w_ct),
          "the overlays touch neither the blocks nor the stored weight, and dense() reads the weight itself")
    q40r = gw.GGUFLinear(torch.from_numpy(cmp.q4_0_blocks(codes, scale.float())), "Q4_0", in_f, out_f, dtype=torch.bfloat16, resident=True)
    q40r._memit_col_overlays[37], q40r._memit_row_overlays[4] = col, row
    check(torch.equal(q40r(xb), ct(xb)) and torch.equal(q40r.weight_dense, w_ct), "resident: the same, its resident weight untouched")
    # a row overlay alone, no column: the pinned forward then clones for the rows (nothing else would)
    ct_r = W4A16Module(cmp.ct_pack(codes), scale)
    plain_r = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16)
    plain_r.weight.data.copy_(w_ct)
    gw.install_overlay_forward(plain_r)
    rows_only = [(gw.GGUFLinear(torch.from_numpy(cmp.q4_0_blocks(codes, scale.float())), "Q4_0", in_f, out_f, dtype=torch.bfloat16, resident=res_), what)
                 for res_, what in ((False, "packed"), (True, "resident"))] + [(plain_r, "a plain Linear")]
    for m_ in [ct_r] + [m_ for m_, _ in rows_only]:
        m_._memit_row_overlays[4] = row.clone()
    want_r = w_ct.clone()
    want_r[4, :] = row.to(torch.bfloat16)
    check(torch.equal(ct_r(xb), F.linear(xb, want_r)), "the pinned W4A16 forward with a row overlay alone")
    for m_, what in rows_only:
        stored = m_.weight if isinstance(m_, nn.Linear) else (m_.weight_dense if m_.resident else gw.dense(m_))
        check(torch.equal(m_(xb), ct_r(xb)) and torch.equal(stored, w_ct) and torch.equal(gw.dense(m_), w_ct),
              f"{what} with a row overlay alone: the pinned forward's numbers, laid on a clone (its weight untouched)")
    section("GGUFLinear: slices, packed/resident/chunked, the vault's overlays as _w4a16_forward lays them; the accessors alike "
            "on plain, W4A16 and GGUF layers", before)

    # ------------------------------------------------------------- 3. the stand-in written to GGUF files and read back
    before = len(failures)
    tok = asi.stand_in_tokenizer()
    hf = asi.stand_in_hf(tok)
    ids = torch.tensor([tok.encode(asi.CORPUS[0] + " " + asi.CORPUS[4])])
    ids2 = torch.tensor([tok.encode(asi.CORPUS[3])[:12]])
    files: Dict[str, Path] = {}
    for qt in ("F32", "F16", "BF16", "Q8_0", "Q4_0"):
        files[qt] = gm.write_gemma4_gguf(hf, tmp / f"stand-in-{qt}.gguf", tokenizer=tok, qtype=qt)
    wide = wide_stand_in(tok, 256, 512)
    files["Q4_K"] = gm.write_gemma4_gguf(wide, tmp / "stand-in-wide-Q4_K.gguf", tokenizer=tok, qtype="Q4_K")
    for qt, path in files.items():
        hidden, ffn = (256, 512) if qt == "Q4_K" else (asi.HIDDEN, asi.FFN)
        r = gguf.GGUFReader(str(path))
        types = Counter(t.tensor_type.name for t in r.tensors)
        if qt == "Q4_K":
            check(types["Q4_K"] > 0 and types["Q8_0"] == 6, f"Q4_K: rows of 64 and 128 (o_proj) fall back to Q8_0 as llama.cpp's do: {dict(types)}")
        ref = stand_in_with(tok, path, hidden, ffn)
        want = ref(ids).logits
        want2 = ref(ids2).logits
        src = wide if qt == "Q4_K" else hf
        for mode in gm.LINEAR_MODES:
            m, rec = gm.build_gemma4(path, dtype=torch.float32, linears=mode)
            n_lin = sum(rec["layers_as"].values())
            check(n_lin == 41 and rec["config_source"] == "gguf", f"{qt} {mode}: 41 linear layers built ({rec['layers_as']})")
            check(model_tensors_equal(m, path), f"{qt} {mode}: every tensor is gguf's dequantised one")
            check(torch.equal(m(ids).logits, want) and torch.equal(m(ids2).logits, want2),
                  f"{qt} {mode}: the forward equals the stand-in's carrying the same numbers, bit for bit (fp32, past the window)")
            kinds = {type(mod).__name__ for n_, mod in m.named_modules() if n_.endswith(("_proj",))}
            if mode == "dense" or qt in ("F32", "F16", "BF16"):
                check(kinds == {"Linear"}, f"{qt} {mode}: plain Linear layers ({kinds})")
            else:
                check("GGUFLinear" in kinds, f"{qt} {mode}: the quantised layers are GGUFLinear ({kinds})")
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

        got = gm.config_shape(Gemma4TextConfig(**gm.text_config_from_gguf(gm.read_metadata(r), gm.tensor_table(r))))
        check(got == gm.config_shape(src.config), f"{qt}: the config from the metadata is the stand-in's")
    # bf16, as the instrument runs: the GGUF model against the stand-in cast to bf16 with the same numbers
    ref16 = stand_in_with(tok, files["Q4_0"])
    for p_ in ref16.parameters():  # the weights in bf16, the rotary frequencies kept in float32 (as from_pretrained keeps them)
        p_.data = p_.data.to(torch.bfloat16)
    for layer in ref16.model.layers:
        layer.layer_scalar = layer.layer_scalar.to(torch.bfloat16)
    want16 = ref16(ids).logits
    for mode in gm.LINEAR_MODES:
        m16, _ = gm.build_gemma4(files["Q4_0"], dtype=torch.bfloat16, linears=mode)
        check(torch.equal(m16(ids).logits, want16), f"Q4_0 {mode} in bf16: the forward equals the bf16 stand-in's, bit for bit")
    section("the stand-in as GGUF (F32, F16, BF16, Q8_0, Q4_0, Q4_K): config, tensors and forward as the stand-in carrying "
            "the file's numbers, packed, resident and dense", before)

    before = len(failures)
    lm0 = hf.model
    r = gguf.GGUFReader(str(files["F32"]))
    tt = {t.name: t for t in r.tensors}
    check(torch.equal(torch.from_numpy(np.array(tt["blk.2.ffn_norm.weight"].data)), lm0.layers[2].pre_feedforward_layernorm.weight)
          and torch.equal(torch.from_numpy(np.array(tt["output_norm.weight"].data)), lm0.norm.weight),
          "the file holds w as Gemma4RMSNorm uses it (no unit offset added, llama.cpp's norm_shift 0)")
    from transformers.modeling_gguf_pytorch_utils import Gemma2TensorProcessor, TENSOR_PROCESSORS, load_gguf_checkpoint

    parsed = load_gguf_checkpoint(str(files["F32"]), return_tensors=True, model_to_load=asi.stand_in_hf(tok))["tensors"]
    check("gemma4" not in TENSOR_PROCESSORS and torch.equal(parsed["model.layers.2.pre_feedforward_layernorm.weight"],
                                                           lm0.layers[2].pre_feedforward_layernorm.weight),
          "transformers' own GGUF reader has no processor for gemma4 and reads w unchanged")
    g3 = Gemma2TensorProcessor().process(np.asarray(tt["blk.2.ffn_norm.weight"].data), "blk.2.ffn_norm.weight").weights
    check(np.array_equal(g3, np.asarray(tt["blk.2.ffn_norm.weight"].data) - 1), "under Gemma 2 and 3's rule the same bytes would read w - 1")
    if "model.layers.0.layer_scalar" not in parsed:
        print("  --    note: transformers' own GGUF reader leaves Gemma 4's layer scalars out "
              "(llama.cpp names them layer_output_scale.weight); backend/gguf_model.py maps them")
    shifted = gm.write_gemma4_gguf(hf, tmp / "stand-in-plus-one.gguf", tokenizer=tok, qtype="F32", norm_shift=1.0)
    m1, _ = gm.build_gemma4(shifted, dtype=torch.float32)
    base_logits = hf(ids).logits
    check(not torch.allclose(m1(ids).logits, base_logits, atol=1e-3) and torch.equal(gm.build_gemma4(files["F32"], dtype=torch.float32)[0](ids).logits, base_logits),
          "round trip: the file as written (w) gives the stand-in's own forward; one holding w + 1 does not")
    section("the norm convention: Gemma 4's GGUF holds w, read unchanged; w + 1 would show", before)

    before = len(failures)
    cfg_json = tmp / "config.json"
    cfg_json.write_text(json.dumps(hf.config.to_dict()), encoding="utf-8")
    m_hf, rec = gm.build_gemma4(files["Q8_0"], dtype=torch.float32, hf_config=cfg_json)
    check(rec["config_source"].startswith("config.json") and torch.equal(m_hf(ids).logits, gm.build_gemma4(files["Q8_0"], dtype=torch.float32)[0](ids).logits),
          "a matching config.json is taken (checked against the file), the same model")
    for k, v in (("sliding_window", 16), ("rms_norm_eps", 1e-5), ("num_hidden_layers", 5)):
        d = dict(hf.config.to_dict(), **{k: v})
        if k == "num_hidden_layers":
            d["layer_types"] = d["layer_types"][:4] + ["full_attention"]
            d.pop("per_layer_config", None)
        try:
            gm.build_gemma4(files["Q8_0"], dtype=torch.float32, hf_config=d)
            check(False, f"a config.json with another {k} refused")
        except ValueError as exc:
            check(k in str(exc) and "different models" in str(exc), f"a config.json with another {k} refused in plain words: {str(exc)[:120]}")
    # a file that leaves out a key a Gemma 4 needs, or the rope_freqs tensor: refused alone, built with a config.json
    r = gguf.GGUFReader(str(files["Q8_0"]))
    gmeta, gfreqs = gm.gguf_metadata_of(hf.config)
    fmeta = gm.read_metadata(r)
    check(all((np.float32(fmeta[k]) == np.float32(v)) if isinstance(v, float) else fmeta[k] == v for k, v in gmeta.items())
          and np.array_equal(np.asarray(gm.tensor_table(r)["rope_freqs.weight"].data), gfreqs),
          f"gguf_metadata_of the stand-in's config: the {len(gmeta)} keys and the rope_freqs the writer put in the file")
    logits_q8 = gm.build_gemma4(files["Q8_0"], dtype=torch.float32)[0](ids).logits

    def short_reader(what: str) -> Any:
        """The Q8_0 file's reader without `what` (a key, the rope_freqs tensor, or "all": every gemma4 key and it)."""
        r_ = gm.open_gguf(files["Q8_0"])
        if what in ("rope_freqs.weight", "all"):
            r_.tensors = [t for t in r_.tensors if t.name != "rope_freqs.weight"]
        for k in [k for k in r_.fields if k == what or (what == "all" and k.startswith("gemma4."))]:
            del r_.fields[k]
        return r_

    for what in ("gemma4.attention.sliding_window_pattern", "gemma4.attention.layer_norm_rms_epsilon", "rope_freqs.weight", "all"):
        try:
            gm.build_gemma4(files["Q8_0"], dtype=torch.float32, reader=short_reader(what))
            check(False, f"a file without {what} refused when no config.json is given")
        except ValueError as exc:
            check("config.json" in str(exc) and (what == "all" or what in str(exc) or "rope_freqs" in str(exc)),
                  f"a file without {what}, no config.json: refused, asking for one ({str(exc)[:90]})")
        m_l, rec_l = gm.build_gemma4(files["Q8_0"], dtype=torch.float32, hf_config=cfg_json, reader=short_reader(what))
        check(torch.equal(m_l(ids).logits, logits_q8) and "which lacks" in rec_l["config_source"]
              and (what == "all" or what in rec_l["config_source"]),
              f"a file without {what}, with the config.json: built from it, the same model ({rec_l['config_source'][-70:]})")
    d = dict(hf.config.to_dict(), sliding_window=16)
    try:
        gm.build_gemma4(files["Q8_0"], dtype=torch.float32, hf_config=d, reader=short_reader("gemma4.attention.layer_norm_rms_epsilon"))
        check(False, "a file lacking a key, with a config.json that differs on a key it states: refused")
    except ValueError as exc:
        check("sliding_window" in str(exc) and "rms_norm_eps" not in str(exc),
              "a file lacking a key, with a config.json that differs on one it states: refused for that one alone")
    iq = gm.write_gemma4_gguf(hf, tmp / "stand-in-iq.gguf", qtype="Q8_0",
                              blocks={"blk.0.ffn_down.weight": (rng.integers(0, 256, (asi.HIDDEN, asi.FFN // 32 * 18), dtype=np.uint8), "IQ4_NL")})
    try:
        gm.build_gemma4(iq, dtype=torch.float32)
        check(False, "a file with an IQ4_NL tensor refused")
    except ValueError as exc:
        check("IQ4_NL" in str(exc) and "not read yet" in str(exc), f"a file with an IQ4_NL tensor refused: {exc}")
    r = gguf.GGUFReader(str(files["Q8_0"]))
    meta, tabl = gm.read_metadata(r), gm.tensor_table(r)
    for change, word in (({"gemma4.expert_count": 128}, "mixture-of-experts"),
                         ({"gemma4.embedding_length_per_layer_input": 256}, "per-layer embeddings"),
                         ({"general.architecture": "llama"}, "'llama'"),
                         ({"gemma4.attention.shared_kv_layers": 4}, "share")):
        try:
            gm.text_config_from_gguf({**meta, **change}, tabl)
            check(False, f"{change} refused")
        except ValueError as exc:
            check(word in str(exc), f"{list(change)[0]}: refused in plain words ({exc})")
    section("a matching config.json taken, a differing one refused, one standing in for what a file leaves out; IQ "
            "tensors, experts, per-layer embeddings, shared key-values and other archs refused", before)

    before = len(failures)
    # the 31B's shape, as llama.cpp's converter would write its metadata (ARCHITECTURE §2; the head counts as the GPU
    # runtime study of 28 October 2026 worked them out), built on the meta device: the shapes only, no weights
    swa31 = [(l + 1) % 6 != 0 for l in range(60)]
    meta31 = {"general.architecture": "gemma4", "gemma4.block_count": 60, "gemma4.embedding_length": 5376,
              "gemma4.feed_forward_length": 21504, "gemma4.attention.head_count": 32,
              "gemma4.attention.head_count_kv": [16 if s_ else 4 for s_ in swa31], "gemma4.attention.key_length": 512,
              "gemma4.attention.value_length": 512, "gemma4.attention.key_length_swa": 256,
              "gemma4.attention.value_length_swa": 256, "gemma4.attention.sliding_window": 1024,
              "gemma4.attention.sliding_window_pattern": swa31, "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
              "gemma4.final_logit_softcapping": 30.0, "gemma4.context_length": 262144, "gemma4.rope.freq_base": 1e6,
              "gemma4.rope.freq_base_swa": 1e4, "gemma4.rope.dimension_count": 512, "gemma4.rope.dimension_count_swa": 256,
              "gemma4.embedding_length_per_layer_input": 0, "gemma4.attention.shared_kv_layers": 0}

    class Fake:
        def __init__(self, name: str, shape: tuple, data: Any = None) -> None:
            self.name, self.shape, self.data, self.tensor_type = name, list(reversed(shape)), data, QT.F32

    freqs = np.array([1.0] * 64 + [1e30] * 192, dtype=np.float32)
    tabl31 = {"rope_freqs.weight": Fake("rope_freqs.weight", (256,), freqs), "token_embd.weight": Fake("token_embd.weight", (262144, 5376))}
    tabl31.update({f"blk.{l}.attn_v.weight": Fake(f"blk.{l}.attn_v.weight", (4096, 5376)) for l in range(60) if swa31[l]})
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg31 = Gemma4TextConfig(**gm.text_config_from_gguf(meta31, tabl31))
    sh = gm.config_shape(cfg31)
    check(sh["head_dim"] == [256] and sh["global_head_dim"] == [512] and sh["num_key_value_heads"] == [16]
          and sh["num_global_key_value_heads"] == [4] and sh["attention_k_eq_v"] and sh["layer_types"].count("full_attention") == 10
          and sh["rope_full_attention"] == ("proportional", 1e6, 0.25) and sh["vocab_size"] == 262144,
          f"the 31B's metadata: 10 full layers, heads 256/512, key-values 16/4, rope {sh['rope_full_attention']}")
    with gm._parameters_on_meta():
        m31 = Gemma4ForCausalLM(cfg31)
    lins = [mod for n_, mod in m31.named_modules() if isinstance(mod, nn.Linear) and n_ != "lm_head"]
    n31 = sum(mod.in_features * mod.out_features for mod in lins)
    check(len(lins) == 410 and n31 == 29_286_727_680 and all(p_.device.type == "meta" for p_ in m31.parameters()),
          f"built on the meta device: {len(lins)} linear layers, {n31:,} quantised weights, nothing allocated")
    del m31
    section("the 31B's shape from metadata as llama.cpp writes it: 410 linear layers, 29.287 billion weights", before)

    before = len(failures)
    path = files["Q4_0"]
    data = path.read_bytes()
    pv = gm.provenance(path)
    import hashlib

    r = gguf.GGUFReader(str(path))
    check(pv["size"] == len(data) and pv["header_bytes"] == r.data_offset
          and pv["sha256_header"] == hashlib.sha256(data[: r.data_offset]).hexdigest()
          and pv["sha256_first_mib"] == hashlib.sha256(data[: 1 << 20]).hexdigest()
          and pv["sha256_last_mib"] == hashlib.sha256(data[-(1 << 20):]).hexdigest(), "provenance: the size and the three sha256")
    check(pv["tensor_types"] == dict(Counter(t.tensor_type.name for t in r.tensors)) and pv["tensors"] == len(r.tensors)
          and pv["architecture"] == "gemma4" and pv["metadata"]["gemma4.block_count"] == 6 and "general.file_type" in pv["metadata"],
          f"provenance: tensor types {pv['tensor_types']}, arch and metadata")
    check(gm.provenance(path)["id"] == pv["id"], "provenance: the same file, the same fingerprint")
    flipped = tmp / "flipped.gguf"
    b = bytearray(data)
    b[-1] ^= 1
    flipped.write_bytes(bytes(b))
    pf = gm.provenance(flipped)
    check(pf["id"] != pv["id"] and pf["sha256_header"] == pv["sha256_header"], "provenance: one bit of the last tensor flipped moves the fingerprint, not the header's sha256")
    section("the provenance record", before)

    before = len(failures)
    t2, trec = gm.load_tokenizer(path, ("google/not-cached-here",))
    check(trec["source"].startswith("gguf metadata") and "google/not-cached-here" in trec.get("repos_tried", {}), f"the tokenizer from the file's metadata: {trec}")
    texts = list(asi.CORPUS) + [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
                                for t in asi.CORPUS[:3]]
    check(all(t2.encode(t, add_special_tokens=False) == tok.encode(t, add_special_tokens=False) for t in texts),
          "it encodes the corpus and chat turns as the stand-in's tokenizer does")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Who are you?"}]
    check(all(t2.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=th)
              == tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=th) for th in (False, True)),
          "its chat template is the file's: the same turns, thinking on and off")
    v_s = tok.get_vocab()
    v_g = t2.get_vocab()
    extra = sorted(set(v_g) - set(v_s))
    check(all(v_g.get(k) == i for k, i in v_s.items()) and all(v_g[k] >= len(v_s) for k in extra),
          f"every stand-in piece keeps its id; the converter's additions lie past the model's rows ({extra})")
    section("the tokenizer from the GGUF's own metadata", before)

    # --------------------------------------------------------------- 4. GGUFRuntime against the stand-in's runtime
    before = len(failures)
    rt_g = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.float32, tokenizer=tok, load_jpp=False).load()
    ref = stand_in_with(tok, files["Q4_0"])
    rt_s = asi.stand_in_runtime(ref, tok)
    check(rt_g.model_id == f"gguf:stand-in-Q4_0.gguf#{rt_g.weights_card['id']}@cpu/float32"
          and rt_g.weights_card["linears"] == "packed" and rt_g.weights_card["layers_as"] == {"Q4_0 packed": 41}
          and rt_g.lens_card() is None and rt_g.end_of_turn_ids() == rt_s.end_of_turn_ids() and rt_g.sliding_window == asi.WINDOW,
          f"loaded: {rt_g.model_id}, {rt_g.weights_card['layers_as']}, no lens, the stand-in's turn ends and window")
    lm_g, lm_s = language_model(rt_g.hf_model), language_model(rt_s.hf_model)
    check(isinstance(lm_g.layers[2].mlp.down_proj, gw.GGUFLinear), "its down_proj layers are packed GGUF layers")
    nidx = torch.tensor([5, 0, 127, 64, 5])
    for l in (0, 5):
        check(torch.equal(rt_g.slice_w_down(l, nidx), rt_s.slice_w_down(l, nidx)) and torch.equal(rt_g.slice_w_gate(l, nidx), rt_s.slice_w_gate(l, nidx))
              and torch.equal(rt_g.slice_w_up(l, nidx), rt_s.slice_w_up(l, nidx)), f"layer {l}: slice_w_down, slice_w_gate, slice_w_up as the stand-in's")
        check(torch.equal(rt_g.get_full_w_down(l), rt_s.get_full_w_down(l)) and torch.equal(rt_g.get_full_w_gate(l), rt_s.get_full_w_gate(l))
              and torch.equal(rt_g.get_full_w_up(l), rt_s.get_full_w_up(l)), f"layer {l}: get_full_w_down, _gate, _up as the stand-in's")
    w_post = lm_s.layers[2].post_feedforward_layernorm.weight.float()
    by_hand = torch.linalg.vector_norm(lm_s.layers[2].mlp.down_proj.weight[:, torch.arange(asi.FFN)].T.contiguous().float() * w_post, dim=-1)
    wn = rt_g.write_col_norms(2)
    check(wn.shape == (asi.FFN,) and torch.equal(wn, by_hand), "write_col_norms: ||w_post (.) W_down[:, i]|| for every neuron, the layer's own width")
    W = ao.Weights(lm_g, rt_g)
    Ws = ao.Weights(lm_s, rt_s)
    check(W.d_ff == asi.FFN and torch.equal(W.write_ceiling(3, [1, 9]), Ws.write_ceiling(3, [1, 9]))
          and torch.equal(W.attn_weight(5, "v"), Ws.attn_weight(5, "v")), "atlas_ops.Weights: width, write ceilings and attention weights alike")
    prompt = [rt_s.tokenize_prompt(rt_s.compose_prompt("Who are you?", chat=True))]
    stop = rt_s.end_of_turn_ids()

    def voices(rt: Any, lm: Any, w: Any) -> List[Any]:
        iv = ao.resolve(w, [ao.Push(3, 7, "add", 0.8), ao.Push(2, 5, "act", 1.0)], [], ao.THREE_VOICES, lambda l: 3.0, lambda l, i: 2.0)
        return [(s.row, s.token_id, s.logprob, s.top_ids, s.top_lps) for s in ao.write_rows(
            lm, rt.lm_head_weight, prompt * 3, iv, max_new=12, stop_ids=stop, bos_id=tok.bos_token_id, top_k=3)]

    check(voices(rt_g, lm_g, W) == voices(rt_s, lm_s, Ws), "a push written in three voices: the same pieces and log-probabilities")
    check(ao.read_neuron(lm_g, prompt[0], 4, 11, bos_id=tok.bos_token_id) == ao.read_neuron(lm_s, prompt[0], 4, 11, bos_id=tok.bos_token_id),
          "read_neuron: the same activations")
    every = [Address("resid_post", l) for l in range(6)]
    read = [Address(n_, l) for l in range(6) for n_ in ("resid_pre", "resid_mid", "mlp_act", "mlp_out")]
    cg = rt_g.capture_rows(prompt[0], every=every, read=read, n_read=2)
    cs = rt_s.capture_rows(prompt[0], every=every, read=read, n_read=2)
    check(set(cg) == set(cs) and all(torch.equal(cg[k], cs[k]) for k in cs), "a sighting's row capture (resid_post everywhere, four points at the read rows): the same states")
    h = cg[Address("resid_post", 5)][-1].float()
    eye_g = unembed_gemma4(h, rt_g.final_norm_weight, rt_g.lm_head_weight)
    eye_s = unembed_gemma4(cs[Address("resid_post", 5)][-1].float(), rt_s.final_norm_weight, rt_s.lm_head_weight)
    dg = rt_g.decode_top_tokens(eye_g, top_k=5, tracked_token_ids={"bob": 5})
    ds = rt_s.decode_top_tokens(eye_s, top_k=5, tracked_token_ids={"bob": 5})
    check(torch.equal(eye_g, eye_s) and dg == ds, "the naked eye's readout of the last state: the same logits and top pieces")
    print("  --    a whole sighting is not run here: the pinned sweep counts 60 layers of 5,376 dims, and reads the J++ lens")
    section("GGUFRuntime: its accessors, a push, a neuron read and a sighting's lens-free parts as the stand-in's", before)

    before = len(failures)
    vault = LarqlVaultEngine(rt_g)
    down = lm_g.layers[2].mlp.down_proj
    saved, w_before = down.weight_blocks.clone(), gw.dense(down).clone()
    clean = rt_g.hf_model(torch.tensor(prompt)).logits
    rep = vault.apply_patch_surgery("DELETE", layer_idx=2, neuron_idx=5)
    zeroed = stand_in_with(tok, files["Q4_0"])
    zeroed.model.layers[2].mlp.down_proj.weight.data[:, 5] = 0.0
    check(rep["status"] == "applied" and 5 in down._memit_col_overlays, "a vault DELETE on a packed GGUF down_proj: applied as a column overlay")
    check(torch.equal(rt_g.hf_model(torch.tensor(prompt)).logits, zeroed(torch.tensor(prompt)).logits),
          "heard: the forward is the stand-in's with that column zeroed, bit for bit")
    check(torch.equal(rt_g.slice_w_down(2, torch.tensor([5, 6]))[0], torch.zeros(asi.HIDDEN)) and float(rt_g.write_col_norms(2)[5]) == 0.0
          and torch.equal(rt_g.write_col_norms(2)[6:], wn[6:]) and torch.equal(rt_g.get_full_w_down(2)[:, 5], torch.zeros(asi.HIDDEN)),
          "seen: slice_w_down, get_full_w_down and write_col_norms read the edit; the other neurons as before")
    check(torch.equal(down.weight_blocks, saved) and torch.equal(gw.dense(down), w_before), "the blocks, and the weight they unpack to, are untouched")
    down._memit_col_overlays.clear()
    vault.active_patches.clear()
    check(torch.equal(rt_g.hf_model(torch.tensor(prompt)).logits, clean), "with the edit cleared, the forward is as before")
    # a column that is not zero (as an UPDATE or INSERT writes one; they read the J++ lens, not loaded here, so it is laid
    # on as the vault lays it) and a row on gate_proj, on the GGUF runtime and on the stand-in's alike
    g9 = torch.Generator().manual_seed(9)
    v7, r9 = torch.randn(asi.HIDDEN, generator=g9), torch.randn(asi.HIDDEN, generator=g9)
    gate_g = lm_g.layers[2].mlp.gate_proj
    down_s, gate_s = lm_s.layers[2].mlp.down_proj, lm_s.layers[2].mlp.gate_proj
    down._memit_col_overlays[7] = v7.clone()
    gate_g._memit_row_overlays[9] = r9.clone()
    down_s._memit_col_overlays, gate_s._memit_row_overlays = {7: v7.clone()}, {9: r9.clone()}
    nidx2 = torch.tensor([7, 9, 0, 7])
    check(torch.equal(rt_g.slice_w_down(2, nidx2), rt_s.slice_w_down(2, nidx2)) and torch.equal(rt_g.slice_w_down(2, nidx2)[0], v7)
          and torch.equal(rt_g.get_full_w_down(2), rt_s.get_full_w_down(2)) and torch.equal(rt_g.get_full_w_down(2)[:, 7], v7),
          "a non-zero column overlay: slice_w_down and get_full_w_down as the pinned ones read it")
    check(torch.equal(rt_g.slice_w_gate(2, nidx2), rt_s.slice_w_gate(2, nidx2)) and torch.equal(rt_g.slice_w_gate(2, nidx2)[1], r9)
          and torch.equal(rt_g.get_full_w_gate(2), rt_s.get_full_w_gate(2)) and torch.equal(rt_g.get_full_w_gate(2)[9], r9)
          and torch.equal(rt_g.slice_w_up(2, nidx2), rt_s.slice_w_up(2, nidx2)),
          "a gate row overlay: slice_w_gate and get_full_w_gate as the pinned ones read it (up_proj, as there, without)")
    wn7 = rt_g.write_col_norms(2)
    want7 = torch.linalg.vector_norm(v7.float() * w_post)
    check(torch.equal(wn7[7], want7) and not torch.equal(want7, torch.linalg.vector_norm(v7)) and torch.equal(wn7[:7], wn[:7])
          and torch.equal(wn7[8:], wn[8:]), "write_col_norms: the overlay column's ||w_post (.) v||, the others as before")
    edited = stand_in_with(tok, files["Q4_0"])
    edited.model.layers[2].mlp.down_proj.weight.data[:, 7] = v7
    edited.model.layers[2].mlp.gate_proj.weight.data[9] = r9
    check(torch.equal(rt_g.hf_model(torch.tensor(prompt)).logits, edited(torch.tensor(prompt)).logits)
          and torch.equal(down.weight_blocks, saved) and torch.equal(gw.dense(gate_g), gw.dense(gate_s)),
          "heard: the forward is the stand-in's with that column and that row written in, the blocks untouched")
    down._memit_col_overlays.clear()
    gate_g._memit_row_overlays.clear()
    del down_s._memit_col_overlays, gate_s._memit_row_overlays
    check(torch.equal(rt_g.hf_model(torch.tensor(prompt)).logits, clean), "with both cleared, the forward is as before")
    section("vault edits on packed GGUF layers (a zeroed and a written column, a gate row): heard, seen, the blocks untouched", before)

    before = len(failures)
    runs = {}
    for name, rt in (("gguf", rt_g), ("stand-in", rt_s)):
        out = tmp / f"census-{name}.jsonl"
        res = cz.take_census(rt, 2, [0, 1], out, p50=1.5, p50_source="fixed", controls=1, seed=0, max_new=6, top_k=3, batch=5, log=quiet)
        runs[name] = (list(cz.read_rows(out)), json.loads(cz.meta_path_of(out).read_text()), res)
    (rg, mg, _), (rs, ms, _) = runs["gguf"], runs["stand-in"]
    check(len(rg) == 34 and rg == rs, f"a tiny census (2 neurons, 1 control, main_rel): {len(rg)} answers, row for row the stand-in's")
    check(mg["model"] == rt_g.model_id and mg["read_gain"] == ms["read_gain"] and mg["prompt_ids"] == ms["prompt_ids"],
          "its meta: the GGUF's model id, the same read gain and prompts")
    rt_16 = GGUFRuntime(files["Q4_0"], device="cpu", dtype=torch.bfloat16, tokenizer=tok, load_jpp=False).load()
    try:
        cz.take_census(rt_16, 2, [0, 1, 2], tmp / "census-gguf.jsonl", p50=1.5, p50_source="fixed", controls=1, seed=0,
                       max_new=6, top_k=3, batch=5, log=quiet)
        check(False, "the same census continued in bf16 on a float32 run refused")
    except ValueError as exc:
        check("model" in str(exc) and rt_16.model_id.endswith("@cpu/bfloat16") and len(list(cz.read_rows(tmp / "census-gguf.jsonl"))) == 34,
              f"the same census continued in bf16 onto the float32 run: refused, nothing written ({exc})")
    del rt_16
    section("a census on the GGUF runtime, row for row the stand-in's", before)

    # ------------------------------------------------------------ 5. the runtime from the environment, and the CLI
    before = len(failures)
    keys = (rf.GGUF_ENV, rf.DEVICE_ENV, rf.LINEARS_ENV, rf.CONFIG_ENV, rf.DTYPE_ENV, rf.LENS_ENV, "CUDA_VISIBLE_DEVICES")
    env_before = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        rt0 = rf.make_runtime(load_jlens=False)
        check(type(rt0) is er.TheodoliteRuntime and str(rt0.device) == "cuda:0" and rt0.load_jlens is False,
              "nothing set: the pinned TheodoliteRuntime on cuda:0, as the entry points built it")
        check([rf.reading_class(d, t) for d, t in (("cuda:0", None), ("cuda:1", torch.bfloat16), ("cpu", None),
                                                   ("cpu", torch.float32), ("cuda:0", torch.float32))]
              == ["", "", "@cpu/bfloat16", "@cpu/float32", "@cuda/float32"],
              "reading_class: nothing for bf16 on a GPU; the device and dtype otherwise")
        os.environ[rf.DEVICE_ENV] = "cpu"
        rt_c = rf.make_runtime(load_jlens=False)
        pinned_load = er.TheodoliteRuntime.load
        er.TheodoliteRuntime.load = lambda self: self  # the checkpoint is not here: the pinned load stood in for
        try:
            id_before = rt_c.model_id
            rt_c.load()
        finally:
            er.TheodoliteRuntime.load = pinned_load
        check(type(rt_c) is rf.CheckpointRuntime and str(rt_c.device) == "cpu" and id_before == er.DEFAULT_W4A16_MODEL_ID
              and rt_c.model_id == er.DEFAULT_W4A16_MODEL_ID + "@cpu/bfloat16" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
              f"THEODOLITE_DEVICE=cpu: the checkpoint loads by its repo id, its readings then recorded as {rt_c.model_id}")
        os.environ.pop(rf.DEVICE_ENV)
        os.environ.update({rf.GGUF_ENV: str(files["Q4_0"]), rf.DEVICE_ENV: "cpu", rf.DTYPE_ENV: "float32", rf.LENS_ENV: "none",
                           rf.LINEARS_ENV: "resident"})
        rt1 = rf.make_runtime(load_jlens=False)
        check(isinstance(rt1, GGUFRuntime) and str(rt1.device) == "cpu" and rt1.dtype == torch.float32 and rt1.linears == "resident"
              and rt1.load_jpp is False and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "THEODOLITE_GGUF on the CPU: a GGUFRuntime, the GPUs hidden")
        for k, v, word in ((rf.DEVICE_ENV, "tpu", "cpu or cuda"), (rf.DTYPE_ENV, "float16", "bfloat16 or float32"),
                           (rf.LENS_ENV, "jlens", "jpp"), (rf.GGUF_ENV, str(tmp / "nowhere.gguf"), "no such file")):
            old = os.environ[k] if k in os.environ else None
            os.environ[k] = v
            try:
                rf.make_runtime()
                check(False, f"{k}={v} refused")
            except ValueError as exc:
                check(word in str(exc), f"{k}={v} refused in plain words")
            finally:
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    finally:
        for k, v in env_before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env = {**os.environ, rf.GGUF_ENV: str(files["Q4_0"]), rf.DEVICE_ENV: "cpu", rf.DTYPE_ENV: "float32", rf.LENS_ENV: "none",
           "THEODOLITE_ATLAS_DIR": str(tmp / "atlas"), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    env.pop("THEODOLITE_CENSUS_DIR", None)
    out_dir = tmp / "cli-census"
    r = subprocess.run([sys.executable, "-I", str(ROOT_DIR / "scripts" / "census_run.py"), "--layer", "2", "--neurons", "0-1",
                        "--random-controls", "1", "--max-new", "6", "--batch", "5", "--top-k", "3", "--out-dir", str(out_dir), "--no-rank"],
                       env=env, capture_output=True, text=True, timeout=600)
    run = out_dir / "L2-main_rel.jsonl"
    check(r.returncode == 0 and run.exists(), f"census_run.py with THEODOLITE_GGUF on the CPU: {r.returncode} {r.stderr[-400:]}")
    if run.exists():
        meta = json.loads(cz.meta_path_of(run).read_text())
        ref_out = tmp / "cli-ref.jsonl"
        cz.take_census(rt_s, 2, [0, 1], ref_out, p50=meta["p50"], p50_source=meta["p50_source"], controls=1, seed=0, max_new=6, top_k=3,
                       batch=5, log=quiet)
        check(meta["model"] == rt_g.model_id and list(cz.read_rows(run)) == list(cz.read_rows(ref_out)),
              "its rows are the stand-in runtime's at the same p50 (the GGUF's own tokenizer reads the questions alike)")
    section("make_runtime from the environment; census_run.py on a GGUF on the CPU", before)

    # ------------------------------------------------------------------ 6. compare_gguf_ct.py's synthetic self-test
    before = len(failures)
    for ok, what in cmp.self_test(tmp / "compare"):
        check(ok, f"compare_gguf_ct: {what}")
    section("compare_gguf_ct.py: identical pairs found identical (blocks of either sign), a changed code, scale and norm "
            "convention each found", before)

    before = len(failures)
    sys.path.insert(0, str(ROOT_DIR / "experiments" / "scripts"))
    import R4_common

    check(R4_common.pinned_differ() == [], f"the files R4 pins (engine_runtime.py among them) are unchanged: {R4_common.pinned_differ()}")
    section("the R4-pinned runtime untouched: everything here is in new modules", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
