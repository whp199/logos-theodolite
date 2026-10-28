"""GGUF weights in torch (`backend/gguf_weights.py`): llama.cpp's block formats unpacked on the CPU or a GPU, a linear
layer that keeps a GGUF tensor's blocks, and three accessors that read any of the instrument's linear layers alike.

A GGUF file (llama.cpp's format, written by its `convert_hf_to_gguf.py` and `llama-quantize`) stores each weight
[out, in] as rows of blocks along `in`: a block of 32 elements (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) or 256 (the K types
Q2_K to Q6_K), each with its own scale (and for some types a minimum) in fp16, or plain F32, F16 or BF16 numbers. The
unpackers here (`unpack`, one per type) are pure torch, so they run wherever the blocks are (CPU or CUDA), and each one
mirrors gguf-py's own numpy dequantiser (`gguf.quants.dequantize`, gguf 0.19) operation for operation in float32: the
same products in the same order, so the same numbers bit for bit (scripts/check_gguf.py compares them on random
blocks, and exhaustively over every fp16 scale where that is cheap). The IQ types (IQ1_S to IQ4_XS), the ternary
TQ1_0 and TQ2_0, MXFP4, NVFP4 and Q1_0 are refused in plain words: not yet.

`GGUFLinear` is a linear layer that keeps a tensor's raw blocks and unpacks them on the fly at every forward (as
`engine_runtime._w4a16_forward` unpacks the compressed-tensors W4A16 checkpoint), or once at load ("resident", the
blocks then dropped). Its forward applies the vault's column and row overlays (`_memit_col_overlays`,
`_memit_row_overlays`: the one-neuron edits, named after MEMIT in the code, not MEMIT) exactly as `_w4a16_forward`
does: on a clone of the unpacked weight, column by column, then row by row. `install_overlay_forward` gives a plain
`nn.Linear` the same forward, so a model dequantised to dense layers hears the vault too.

`dense(mod)`, `dense_cols(mod, idx)` and `dense_rows(mod, idx)` read a linear layer's own weight (no overlays: the
runtime's `slice_w_*` and `get_full_w_*` lay those on) whatever holds it: a plain `nn.Linear`, a compressed-tensors
W4A16 module (`weight_packed`, `weight_scale`: through the pinned `engine_runtime` unpackers, looked up at each call,
so their numbers are unchanged) or a `GGUFLinear`. Their shapes are the pinned unpackers': `dense` [out, in],
`dense_cols` [K, out] (row k is column idx[k], a neuron's write vector on a down_proj), `dense_rows` [K, in].

Nothing here imports gguf-py; `backend/gguf_model.py` reads the files with it.
"""

from __future__ import annotations

import types
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

QK_K = 256

# ggml's type codes (gguf.GGMLQuantizationType) -> (name, elements per block, bytes per block), for the types unpacked
# here; scripts/check_gguf.py checks the sizes against gguf.GGML_QUANT_SIZES.
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 2 + 16),
    3: ("Q4_1", 32, 2 + 2 + 16),
    6: ("Q5_0", 32, 2 + 4 + 16),
    7: ("Q5_1", 32, 2 + 2 + 4 + 16),
    8: ("Q8_0", 32, 2 + 32),
    10: ("Q2_K", QK_K, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: ("Q3_K", QK_K, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", QK_K, 2 + 2 + QK_K // 2 + 12),
    13: ("Q5_K", QK_K, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", QK_K, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    30: ("BF16", 1, 2),
}
# Types a GGUF file may hold that are not unpacked here yet (named so a refusal can say which).
NOT_YET: Dict[int, str] = {
    9: "Q8_1", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}
_CODE_OF = {name: code for code, (name, _b, _t) in GGML_TYPES.items()}
# A type given by ggml's code (or gguf's GGMLQuantizationType) or by name
QType = Union[int, str]


def type_code(qtype: QType) -> int:
    """ggml's code for a type given by code or name ("Q4_0", "q4_k"); a ValueError in plain words for a type that is
    not unpacked here (an IQ type, say)."""
    if isinstance(qtype, str):
        code = _CODE_OF.get(qtype.upper())
        if code is None:
            known = {v: k for k, v in NOT_YET.items()}.get(qtype.upper())
            if known is None:
                raise ValueError(f"{qtype!r} is not a GGUF tensor type.")
            code = known
    else:
        code = int(getattr(qtype, "value", qtype))
    if code in GGML_TYPES:
        return code
    name = NOT_YET.get(code)
    if name is not None:
        raise ValueError(
            f"GGUF tensors of type {name} are not read yet (backend/gguf_weights.py unpacks "
            f"{', '.join(n for n, _b, _t in GGML_TYPES.values())}): use a file quantised to one of those."
        )
    raise ValueError(f"GGUF tensor type {code} is not a type ggml defines.")


def type_name(qtype: QType) -> str:
    try:
        return GGML_TYPES[type_code(qtype)][0]
    except ValueError:
        code = int(getattr(qtype, "value", qtype)) if not isinstance(qtype, str) else -1
        return NOT_YET.get(code, str(qtype))


def block_shape(qtype: QType) -> Tuple[int, int]:
    """(elements per block, bytes per block)."""
    _n, bs, ts = GGML_TYPES[type_code(qtype)]
    return bs, ts


def row_bytes(qtype: QType, n: int) -> int:
    bs, ts = block_shape(qtype)
    if n % bs:
        raise ValueError(f"A row of {n} cannot be stored as {type_name(qtype)}, whose blocks hold {bs}.")
    return n // bs * ts


# ------------------------------------------------------------------------------------------------- the unpackers
# Each takes blocks [n, bytes per block] (uint8) and returns [n, elements per block] float32, as gguf-py's
# `dequantize_blocks` of the same name: every reshape, shift, mask and product in its order.

_U8 = torch.uint8


def _shifts(values: Tuple[int, ...], shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=_U8, device=device).reshape(shape)


def _f16(b: torch.Tensor) -> torch.Tensor:
    """Two bytes a block, read as fp16 and widened (exactly) to float32: [n, 1]."""
    return b.contiguous().view(torch.float16).to(torch.float32)


def _u32(b: torch.Tensor) -> torch.Tensor:
    """Four bytes a block, read as a little-endian uint32 (held in int64): [n, 1]."""
    w = b.to(torch.int64)
    return (w[:, 0:1] | (w[:, 1:2] << 8) | (w[:, 2:3] << 16) | (w[:, 3:4] << 24))


def _q4_0(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    d = _f16(b[:, :2])
    qs = b[:, 2:].reshape(n, -1, 1, 16) >> _shifts((0, 4), (1, 1, 2, 1), b.device)
    qs = (qs & 0x0F).reshape(n, -1).to(torch.int8) - 8
    return d * qs.to(torch.float32)


def _q4_1(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    qs = b[:, 4:].reshape(n, -1, 1, 16) >> _shifts((0, 4), (1, 1, 2, 1), b.device)
    qs = (qs & 0x0F).reshape(n, -1).to(torch.float32)
    return (d * qs) + m


def _q5_bits(qh: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """The 32 five-bit codes of a Q5_0 / Q5_1 block, as uint8: low nibbles as Q4_0's, the fifth bits from qh."""
    n = qs.shape[0]
    qh = (_u32(qh) >> torch.arange(32, device=qs.device, dtype=torch.int64).reshape(1, 32)) & 1
    ql = qs.reshape(n, -1, 1, 16) >> _shifts((0, 4), (1, 1, 2, 1), qs.device)
    ql = (ql & 0x0F).reshape(n, -1)
    return ql | (qh.to(_U8) << 4)


def _q5_0(b: torch.Tensor) -> torch.Tensor:
    d = _f16(b[:, :2])
    q = _q5_bits(b[:, 2:6], b[:, 6:]).to(torch.int8) - 16
    return d * q.to(torch.float32)


def _q5_1(b: torch.Tensor) -> torch.Tensor:
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    q = _q5_bits(b[:, 4:8], b[:, 8:]).to(torch.float32)
    return (d * q) + m


def _q8_0(b: torch.Tensor) -> torch.Tensor:
    d = _f16(b[:, :2])
    x = b[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    return x * d


def _q2_k(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    scales, qs = b[:, : QK_K // 16], b[:, QK_K // 16 : QK_K // 16 + QK_K // 4]
    d, dmin = _f16(b[:, -4:-2]), _f16(b[:, -2:])
    dl = (d * (scales & 0xF).to(torch.float32)).reshape(n, QK_K // 16, 1)
    ml = (dmin * (scales >> 4).to(torch.float32)).reshape(n, QK_K // 16, 1)
    q = (qs.reshape(n, -1, 1, 32) >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), b.device)) & 3
    q = q.reshape(n, QK_K // 16, 16).to(torch.float32)
    return (dl * q - ml).reshape(n, -1)


def _q3_k(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    hmask, qs = b[:, : QK_K // 8], b[:, QK_K // 8 : QK_K // 8 + QK_K // 4]
    scales, d = b[:, QK_K // 8 + QK_K // 4 : -2], _f16(b[:, -2:])
    lscales = (scales[:, :8].reshape(n, 1, 8) >> _shifts((0, 4), (1, 2, 1), b.device)).reshape(n, 16)
    hscales = (scales[:, 8:].reshape(n, 1, 4) >> _shifts((0, 2, 4, 6), (1, 4, 1), b.device)).reshape(n, 16)
    sc = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    sc = (sc.to(torch.int8) - 32).to(torch.float32)
    dl = (d * sc).reshape(n, 16, 1)
    ql = qs.reshape(n, -1, 1, 32) >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), b.device)
    qh = hmask.reshape(n, -1, 1, 32) >> _shifts(tuple(range(8)), (1, 1, 8, 1), b.device)
    ql = ql.reshape(n, 16, QK_K // 16) & 3
    qh = (qh.reshape(n, 16, QK_K // 16) & 1) ^ 1  # the offset is zero where the mask bit is 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8)).to(torch.float32)
    return (dl * q).reshape(n, QK_K)


def _k_scale_min(scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The eight 6-bit scales and minimums of a Q4_K / Q5_K block from its 12 bytes (gguf's Q4_K.get_scale_min)."""
    n = scales.shape[0]
    s = scales.reshape(n, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc, mn


def _q4_k(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, m = _k_scale_min(b[:, 4:16])
    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * m.to(torch.float32)).reshape(n, -1, 1)
    qs = b[:, 16:].reshape(n, -1, 1, 32) >> _shifts((0, 4), (1, 1, 2, 1), b.device)
    qs = (qs & 0x0F).reshape(n, -1, 32).to(torch.float32)
    return (d * qs - dm).reshape(n, QK_K)


def _q5_k(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, m = _k_scale_min(b[:, 4:16])
    qh, qs = b[:, 16 : 16 + QK_K // 8], b[:, 16 + QK_K // 8 :]
    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * m.to(torch.float32)).reshape(n, -1, 1)
    ql = qs.reshape(n, -1, 1, 32) >> _shifts((0, 4), (1, 1, 2, 1), b.device)
    qh = qh.reshape(n, -1, 1, 32) >> _shifts(tuple(range(8)), (1, 1, 8, 1), b.device)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = (qh & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)
    return (d * q - dm).reshape(n, QK_K)


def _q6_k(b: torch.Tensor) -> torch.Tensor:
    n = b.shape[0]
    ql, qh = b[:, : QK_K // 2], b[:, QK_K // 2 : QK_K // 2 + QK_K // 4]
    scales, d = b[:, QK_K // 2 + QK_K // 4 : -2], _f16(b[:, -2:])
    scales = scales.contiguous().view(torch.int8).to(torch.float32)
    d = (d * scales).reshape(n, QK_K // 16, 1)
    ql = ql.reshape(n, -1, 1, 64) >> _shifts((0, 4), (1, 1, 2, 1), b.device)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = qh.reshape(n, -1, 1, 32) >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), b.device)
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape(n, QK_K // 16, -1).to(torch.float32)
    return (d * q).reshape(n, QK_K)


_UNPACK_BLOCKS = {2: _q4_0, 3: _q4_1, 6: _q5_0, 7: _q5_1, 8: _q8_0, 10: _q2_k, 11: _q3_k, 12: _q4_k, 13: _q5_k, 14: _q6_k}


@torch.no_grad()
def unpack(raw: torch.Tensor, qtype: QType) -> torch.Tensor:
    """A GGUF tensor's bytes [..., row bytes] (uint8, on any device) as float32 [..., row elements], the numbers
    `gguf.quants.dequantize` gives for the same bytes. F32, F16 and BF16 are read as they are stored (F16 and BF16
    widened exactly), every other type block by block."""
    code = type_code(qtype)
    if raw.dtype != _U8:
        raise ValueError(f"GGUF blocks are bytes (uint8), not {raw.dtype}.")
    raw = raw.contiguous()
    lead, nbytes = tuple(raw.shape[:-1]), int(raw.shape[-1])
    if code == 0:
        return raw.view(torch.float32)
    if code == 1:
        return raw.view(torch.float16).to(torch.float32)
    if code == 30:
        return (raw.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
    name, bs, ts = GGML_TYPES[code]
    if nbytes % ts:
        raise ValueError(f"{nbytes} bytes a row is not a whole number of {name} blocks ({ts} bytes each).")
    out = _UNPACK_BLOCKS[code](raw.reshape(-1, ts))
    return out.reshape(*lead, nbytes // ts * bs)


# ------------------------------------------------------------------------------------------ the vault's overlays


def with_overlays(mod: nn.Module, w: torch.Tensor) -> torch.Tensor:
    """`w` with the module's live overlays laid on, as `engine_runtime._w4a16_forward` lays them: on a clone, every
    column overlay (`_memit_col_overlays`: w[:, i] = v), then every row overlay (`_memit_row_overlays`: w[i, :] = v);
    `w` itself when there are none."""
    col_overlays: Optional[Dict[int, torch.Tensor]] = getattr(mod, "_memit_col_overlays", None)
    if col_overlays:
        w = w.clone()
        for col_idx, new_col in col_overlays.items():
            w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
    row_overlays: Optional[Dict[int, torch.Tensor]] = getattr(mod, "_memit_row_overlays", None)
    if row_overlays:
        if not col_overlays:
            w = w.clone()
        for row_idx, new_row in row_overlays.items():
            w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)
    return w


def install_overlay_forward(linear: nn.Linear) -> nn.Linear:
    """Give a plain `nn.Linear` the W4A16 forward's handling of the vault's overlays (an empty dict of each, and a
    forward that lays them on a clone of its weight); with none in force it computes `F.linear` on its own weight, as
    `nn.Linear` does. Returns the layer."""
    linear._memit_col_overlays = {}
    linear._memit_row_overlays = {}

    def _overlay_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        w = with_overlays(self, self.weight)
        return F.linear(x.to(dtype=w.dtype), w, self.bias)

    linear.forward = types.MethodType(_overlay_forward, linear)
    return linear


# ------------------------------------------------------------------------------------------------- the layer


class GGUFLinear(nn.Module):
    """A linear layer y = x W^T (+ b) whose weight W [out, in] is a GGUF tensor's blocks.

    Packed (the default), it keeps the blocks (`weight_blocks`, uint8 [out, row bytes]) and unpacks the whole weight at
    every forward into `dtype`, as the W4A16 forward does: memory at rest is the file's, and every forward pays the
    unpacking. Resident (`resident=True`, or `make_resident()`), it unpacks once into `weight_dense` and drops the
    blocks: memory is the dense weight's, and a forward is a plain product. Either way the weight it computes with is
    the unpacked numbers (float32, those of `gguf.quants.dequantize`) cast to `dtype`, and the forward lays the vault's
    overlays on it as `_w4a16_forward` does (`with_overlays`). `in_features`, `out_features`, `qtype` (the type's name)
    and `gguf_name` (the tensor's name in its file) say what it holds."""

    def __init__(
        self,
        blocks: torch.Tensor,
        qtype: QType,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        resident: bool = False,
        bias: Optional[torch.Tensor] = None,
        gguf_name: str = "",
        chunk_rows: int = 2048,
    ) -> None:
        super().__init__()
        code = type_code(qtype)
        want = (int(out_features), row_bytes(code, int(in_features)))
        if blocks.dtype != _U8 or tuple(blocks.shape) != want:
            raise ValueError(
                f"{gguf_name or 'a GGUF tensor'}: blocks of shape {tuple(blocks.shape)} ({blocks.dtype}) do not hold a "
                f"[{out_features}, {in_features}] {GGML_TYPES[code][0]} weight (uint8 {list(want)})."
            )
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.qtype = GGML_TYPES[code][0]
        self.compute_dtype = dtype
        self.gguf_name = gguf_name
        self.chunk_rows = int(chunk_rows)
        self.register_buffer("weight_blocks", blocks, persistent=False)
        self.register_buffer("weight_dense", None, persistent=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self._memit_col_overlays: Dict[int, torch.Tensor] = {}
        self._memit_row_overlays: Dict[int, torch.Tensor] = {}
        if resident:
            self.make_resident()

    @property
    def resident(self) -> bool:
        return self.weight_dense is not None

    def extra_repr(self) -> str:
        how = "resident" if self.resident else "packed"
        return f"in_features={self.in_features}, out_features={self.out_features}, {self.qtype} {how}, {self.compute_dtype}"

    @torch.no_grad()
    def make_resident(self) -> None:
        """Unpack once and keep the weight in `dtype`; the blocks are dropped (the numbers do not change)."""
        if self.weight_dense is None:
            self.weight_dense = self._unpack_rows_all()
            self.weight_blocks = None

    @torch.no_grad()
    def _unpack_rows_all(self) -> torch.Tensor:
        blocks = self.weight_blocks
        out = torch.empty(self.out_features, self.in_features, dtype=self.compute_dtype, device=blocks.device)
        for r0 in range(0, self.out_features, self.chunk_rows):
            r1 = min(r0 + self.chunk_rows, self.out_features)
            out[r0:r1] = unpack(blocks[r0:r1], self.qtype).to(self.compute_dtype)
        return out

    @torch.no_grad()
    def unpacked(self) -> torch.Tensor:
        """W [out, in] in `dtype`, no overlays (the resident weight itself when resident: do not write into it)."""
        if self.weight_dense is not None:
            return self.weight_dense
        if self.out_features <= self.chunk_rows:
            return unpack(self.weight_blocks, self.qtype).to(self.compute_dtype)
        return self._unpack_rows_all()

    @torch.no_grad()
    def unpacked_rows(self, idx: torch.Tensor) -> torch.Tensor:
        """W[idx, :] as [K, in] in `dtype`: only those rows' blocks are unpacked."""
        if self.weight_dense is not None:
            return self.weight_dense.index_select(0, idx.to(device=self.weight_dense.device, dtype=torch.long)).contiguous()
        rows = self.weight_blocks.index_select(0, idx.to(device=self.weight_blocks.device, dtype=torch.long))
        return unpack(rows, self.qtype).to(self.compute_dtype)

    @torch.no_grad()
    def unpacked_cols(self, idx: torch.Tensor) -> torch.Tensor:
        """W[:, idx] as [K, out] in `dtype` (row k is column idx[k]): only the blocks holding those columns are
        unpacked, one per row for each distinct block, so a neuron's down column costs out x one block."""
        if self.weight_dense is not None:
            return self.weight_dense[:, idx.to(device=self.weight_dense.device, dtype=torch.long)].T.contiguous()
        blocks = self.weight_blocks
        idx = idx.to(device=blocks.device, dtype=torch.long)
        bs, ts = block_shape(self.qtype)
        which, inverse = torch.unique(idx // bs, return_inverse=True)
        per_row = blocks.reshape(self.out_features, self.in_features // bs, ts)
        picked = per_row.index_select(1, which)  # [out, U, ts]
        vals = unpack(picked.reshape(self.out_features, -1), self.qtype).reshape(self.out_features, -1, bs)
        cols = vals[:, inverse, idx % bs]  # [out, K]
        return cols.T.to(self.compute_dtype).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = with_overlays(self, self.unpacked())
        return F.linear(x.to(dtype=w.dtype), w, self.bias)


# ------------------------------------------------------------------------------------------------ the accessors


def _w4a16(mod: nn.Module) -> bool:
    return hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale")


@torch.no_grad()
def dense(mod: nn.Module) -> torch.Tensor:
    """The layer's own weight [out, in] as its forward uses it (no overlays): unpacked from GGUF blocks or W4A16, else
    the plain weight (detached, not copied)."""
    if isinstance(mod, GGUFLinear):
        return mod.unpacked()
    if _w4a16(mod):
        from backend import engine_runtime as er  # looked up at each call: the pinned unpacker, as atlas_ops reads it

        return er.dequantize_w4a16_full(mod.weight_packed, mod.weight_scale)
    return mod.weight.detach()


@torch.no_grad()
def dense_cols(mod: nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Columns `idx` of the layer's own weight as [K, out] (no overlays), as `engine_runtime.dequantize_w4a16_cols`."""
    if isinstance(mod, GGUFLinear):
        return mod.unpacked_cols(idx)
    if _w4a16(mod):
        from backend import engine_runtime as er

        return er.dequantize_w4a16_cols(mod.weight_packed, mod.weight_scale, idx)
    return mod.weight[:, idx.to(mod.weight.device)].T.contiguous()


@torch.no_grad()
def dense_rows(mod: nn.Module, idx: torch.Tensor) -> torch.Tensor:
    """Rows `idx` of the layer's own weight as [K, in] (no overlays), as `engine_runtime.dequantize_w4a16_rows`."""
    if isinstance(mod, GGUFLinear):
        return mod.unpacked_rows(idx)
    if _w4a16(mod):
        from backend import engine_runtime as er

        return er.dequantize_w4a16_rows(mod.weight_packed, mod.weight_scale, idx)
    return mod.weight.index_select(0, idx.to(mod.weight.device)).contiguous()


def packed_kind(mod: nn.Module) -> str:
    """What holds the layer's weight: "gguf:<type>" (packed or resident), "w4a16" or "dense"."""
    if isinstance(mod, GGUFLinear):
        return f"gguf:{mod.qtype}{'' if not mod.resident else ' resident'}"
    return "w4a16" if _w4a16(mod) else "dense"
