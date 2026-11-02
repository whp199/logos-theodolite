"""torch's own int4 kernel on the CPU (`backend/int4_cpu.py`): the 4-bit linears of a GGUF (Q4_0, Q4_1 and Q4_K
layers), or of the W4A16 checkpoint, held as torch's packed int4 weights and multiplied by
`torch.ops.aten._weight_int4pack_mm_for_cpu`, never unpacked. Opt in with THEODOLITE_GGUF_LINEARS=int4mm (a GGUF) or
THEODOLITE_CHECKPOINT_LINEARS=int4mm (the checkpoint), on the CPU only (backend/runtime_factory.py).

Why. Every packed forward of the reference path unpacks every weight (the GGUF unpackers, or the pinned W4A16 one),
some 3 to 16 ns a weight on the 4 vCPUs this was written on, so minutes a forward for the 31B's 29.3 billion. torch
2.14's CPU kernel reads the 4-bit codes and a scale and zero per 32 in place. Measured there on one 5,376 x 21,504
layer (ARCHITECTURE §5's table; three runs beside other work, so a spread): at one row 1.7 to 8 ms, against 0.5 to 1.5 s
for the reference and 7 to 17 ms for the resident bf16 product; at 8 rows about as fast as resident bf16; at 64 rows
3 to 5 times slower than it, still 4 to 14 times faster than the reference. It needs 0.625 bytes a weight (18 GB for
the 31B's linears), where resident bf16 needs 2 (59 GB).

What it computes. The kernel's weight is (q - 8) x scale + zero, a scale and zero per group of 32 in the activations'
dtype, its sums its own (`kernel_codes` maps each type onto it, the scale and zero worked out in float32 from the
block's fp16 fields, then rounded to the model's dtype once):
- Q4_0, d x (q - 8): scale = d, zero 0. bf16 holds 8 of fp16's 11 significant bits, so the scale is rounded.
- Q4_1, d x q + m: scale = d, zero = 8d + m.
- Q4_K, a super-block of 256 whose eight sub-blocks of 32 are (d x sc) x q - dmin x m, sc and m 6-bit: per sub-block
  scale = d x sc, zero = 8 x d x sc - dmin x m (the sub-block is the kernel's group).
- A W4A16 group, (q - 8) x scale with scale already bf16: exact.
So the weight is the file's to within the rounding of a scale and a zero (Q4_0, Q4_1, Q4_K) or exactly (W4A16), and
the products are summed in the kernel's order, not F.linear's: the readings are NOT the reference's bits. The 5-, 6-
and 8-bit types (Q5_0, Q5_1, Q5_K, Q6_K, Q8_0) and Q2_K, Q3_K (sub-blocks of 16) have no place in a 4-bit code and a
group of 32; they stay on the reference path, packed, or resident within THEODOLITE_GGUF_RESIDENT_GB
(backend/gguf_model.py). Every reading taken through the kernel says so: the model id
ends `+int4mm` (after `runtime_factory.reading_class`'s `@cpu/bfloat16`), so a census run, an index or a lens card
taken on it is never mixed with the reference's, and the weights card names the mode (`linears`, and `layers_as`:
"Q4_0 int4mm", "Q4_K int4mm", ...). How far it departs from the reference is measured by `compare_rows` (the shared
prefix of greedy answers, the largest |delta log-prob| over it, the overlap of the five likeliest): on the stand-in by
scripts/check_gguf_cpu.py §4, on the real model by scripts/compare_int4mm.py; the weight itself against the reference's
unpacking by `kernel_weight`.

What it leaves alone. The accessors (`slice_w_*`, `get_full_w_*`, `write_col_norms`, the atlas's weights, the vault's
reads) read the layer's own blocks or `weight_packed` as before, so they give the file's numbers, not the kernel's. A
layer with a vault edit in force keeps the kernel's product and adds the edit to it as its difference from those
numbers (`with_overlay_terms`), so an edit that changes no weight adds exactly nought, and what a reading credits to
an edit (the voice before and after it, a sweep or causal check with it in force) is the edit's alone, not a change of
numeric path as well. The lab's gradient passes (labkit.Lab.grad_linears) run the reference forward and backward (the
kernel has no backward). A layer the kernel cannot take (out features not a multiple of 16, in features
not of 32, a GGUF type other than Q4_0, Q4_1 and Q4_K) stays on the reference path, as `layers_as` counts; a model of
which it can take none (a Q6_K file) is refused, not run on the reference under a `+int4mm` id. On a GPU it is
refused: the kernel is the CPU's.
"""

from __future__ import annotations

import sys
import time
import types
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.gguf_weights import GGUFLinear, _k_scale_min, block_shape, dense_cols, dense_rows

KERNEL = "int4mm"
SUFFIX = "+int4mm"  # after the model id's reading class: gguf:<file>#<fp>@cpu/bfloat16+int4mm
GROUP = 32
KERNEL_TYPES = ("Q4_0", "Q4_1", "Q4_K")  # the GGUF types whose weight is (q - 8) x scale + zero per group of 32


def kernel_missing() -> Optional[str]:
    """Why torch's int4 CPU kernel cannot be used here, in plain words; None when it can."""
    ops = torch.ops.aten
    if not (hasattr(ops, "_weight_int4pack_mm_for_cpu") and hasattr(ops, "_convert_weight_to_int4pack_for_cpu")):
        return (f"this torch ({torch.__version__}) has no _weight_int4pack_mm_for_cpu: int4mm needs torch's int4 CPU "
                "kernel (torch 2.5 or later)")
    if sys.byteorder != "little":
        return "int4mm reads 4-bit codes packed little-endian; this machine is big-endian"
    return None


def fits(out_features: int, in_features: int) -> bool:
    """Whether the kernel takes a layer of this shape (its packing wants out features in sixteens and groups of 32)."""
    return int(out_features) % 16 == 0 and int(in_features) % GROUP == 0


def pack(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype,
         zero: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(torch's packed int4 weight uint8 [out, in / 2], scales and zeros [in / 32, out, 2] in `dtype`) for the weight
    (codes - 8) x scale + zero: codes [out, in] in 0..15, scale and zero [out, in / 32] (zero None: zeros 0)."""
    packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes.to(torch.int32).contiguous(), 1)
    s = scale.to(torch.float32).to(dtype).T
    z = torch.zeros_like(s) if zero is None else zero.to(torch.float32).to(dtype).T
    return packed, torch.stack([s, z], -1).contiguous()


def mm(x: torch.Tensor, packed: torch.Tensor, scales_zeros: torch.Tensor, out_features: int,
       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x @ W.T (+ b) through the kernel, any leading shape (the kernel takes a contiguous 2-D input in the dtype of its
    scales)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).to(scales_zeros.dtype).contiguous()
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(x2, packed, GROUP, scales_zeros).reshape(*lead, int(out_features))
    return y if bias is None else y + bias.to(y.dtype)


@torch.no_grad()
def kernel_weight(packed: torch.Tensor, scales_zeros: torch.Tensor, in_features: int, out_features: int,
                  chunk: int = 2048) -> torch.Tensor:
    """The weight [out, in] as the kernel forms it, in its scales' dtype: the identity's rows multiplied through it,
    `chunk` at a time, so each entry is one weight, (q - 8) x scale + zero, alone in its sum."""
    n = int(in_features)
    out = torch.empty(int(out_features), n, dtype=scales_zeros.dtype)
    for r0 in range(0, n, int(chunk)):
        r1 = min(r0 + int(chunk), n)
        eye = torch.zeros(r1 - r0, n, dtype=scales_zeros.dtype)
        eye[:, r0:r1] = torch.eye(r1 - r0, dtype=scales_zeros.dtype)
        out[:, r0:r1] = mm(eye, packed, scales_zeros, out_features).T
    return out


@torch.no_grad()
def kernel_departure(mod: "Int4GGUFLinear") -> Dict[str, Any]:
    """How far an int4mm layer's weight, as the kernel forms it (`kernel_weight`), departs from the reference's (the
    file's numbers unpacked in float32, as gguf-py's, then rounded to the model's dtype: `mod.unpacked()`). `max_share`
    is the largest |difference| as a share of its group's bound 8|scale| + |zero| (scale and zero in float32, before
    the kernel's rounding); `tolerance` what rounding allows, 2 x the dtype's eps: a scale and a zero each rounded to
    the dtype (8 x |scale| and |zero| times half an eps), the kernel's weight rounded and the reference's rounded
    (half an eps of |weight| each, which is at most the bound) make 3 x half an eps of the bound, under 4. `max_rel` is
    the largest |difference| over the largest |weight|."""
    w_k = kernel_weight(mod.int4_packed, mod.int4_scales_zeros, mod.in_features, mod.out_features).to(torch.float64)
    w_r = mod.unpacked().to(torch.float64)
    _codes, scale, zero = kernel_codes(mod.weight_blocks, mod.qtype, mod.in_features)
    bound = 8.0 * scale.abs().to(torch.float64) + (0.0 if zero is None else zero.abs().to(torch.float64))
    diff = (w_k - w_r).abs()
    share = diff / bound.repeat_interleave(GROUP, dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    return {"qtype": mod.qtype, "dtype": str(mod.compute_dtype).replace("torch.", ""), "max_share": float(share.max()),
            "tolerance": 2.0 * torch.finfo(mod.compute_dtype).eps, "max_rel": float(diff.max() / w_r.abs().max().clamp_min(1e-300)),
            "identical_share": float((diff == 0).double().mean())}


def with_overlay_terms(mod: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The kernel's product `y` (x @ W.T + b) with the module's live vault overlays laid on as their differences from
    its own weight W (`gguf_weights.dense_cols` / `dense_rows`: the file's numbers, which the accessors and the vault
    read), summed in float32 and rounded once to y's dtype; `y` itself when none is in force. A column overlay
    (`_memit_col_overlays`: w[:, i] = v) adds x[..., i] (v - W[:, i]); a row overlay (`_memit_row_overlays`: w[i, :] = v,
    laid on after the columns, as `_w4a16_forward` lays them) adds x (v - W'[i, :]) at output i, W' the weight with the
    column overlays on. Each v is first rounded to the model's dtype, as the reference rounds it. So an overlay that
    changes no weight adds exactly nought, and the edited layer is the kernel's weight moved by the edit."""
    cols: Optional[Dict[int, torch.Tensor]] = getattr(mod, "_memit_col_overlays", None)
    rows: Optional[Dict[int, torch.Tensor]] = getattr(mod, "_memit_row_overlays", None)
    if not cols and not rows:
        return y
    dt = y.dtype  # the model's, as the kernel reads the activations
    acc = y.to(torch.float32)
    if cols:
        cidx = torch.tensor(list(cols), dtype=torch.long, device=x.device)
        new_c = torch.stack([v.to(device=x.device, dtype=dt) for v in cols.values()]).to(torch.float32)  # [K, out]
        delta_c = new_c - dense_cols(mod, cidx).to(device=x.device, dtype=torch.float32)
        acc = acc + x.index_select(-1, cidx).to(dt).to(torch.float32) @ delta_c
    if rows:
        ridx = torch.tensor(list(rows), dtype=torch.long, device=x.device)
        new_r = torch.stack([v.to(device=x.device, dtype=dt) for v in rows.values()]).to(torch.float32)  # [R, in]
        base = dense_rows(mod, ridx).to(device=x.device, dtype=torch.float32).clone()
        if cols:
            base[:, cidx] = new_c[:, ridx].T  # the rows as the column overlays left them
        acc = acc.index_add(-1, ridx, x.to(dt).to(torch.float32) @ (new_r - base).T)
    return acc.to(dt)


def q4_0_codes(blocks: torch.Tensor, in_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """A Q4_0 tensor's bytes uint8 [out, in / 32 x 18] -> (codes uint8 [out, in] in 0..15, d float32 [out, in / 32]):
    each block d (fp16) then 16 bytes, byte j holding element j low and j + 16 high (gguf_weights._q4_0)."""
    out = blocks.shape[0]
    b = blocks.reshape(out, int(in_features) // GROUP, 18)
    d = b[..., :2].contiguous().view(torch.float16).to(torch.float32).reshape(out, -1)
    qs = b[..., 2:]
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1).reshape(out, int(in_features))
    return codes, d


def q4_1_codes(blocks: torch.Tensor, in_features: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A Q4_1 tensor's bytes uint8 [out, in / 32 x 20] -> (codes uint8 [out, in], scale, zero float32 [out, in / 32]):
    each block d and m (fp16) then 16 bytes laid out as Q4_0's, the weight d x q + m (gguf_weights._q4_1), so
    scale = d and zero = 8d + m."""
    out = blocks.shape[0]
    b = blocks.reshape(out, int(in_features) // GROUP, 20)
    d = b[..., :2].contiguous().view(torch.float16).to(torch.float32).reshape(out, -1)
    m = b[..., 2:4].contiguous().view(torch.float16).to(torch.float32).reshape(out, -1)
    qs = b[..., 4:]
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1).reshape(out, int(in_features))
    return codes, d, 8.0 * d + m


def q4_k_codes(blocks: torch.Tensor, in_features: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A Q4_K tensor's bytes uint8 [out, in / 256 x 144] -> (codes uint8 [out, in], scale, zero float32 [out, in / 32]):
    each super-block d and dmin (fp16), twelve bytes of eight 6-bit scales sc and minimums m, then 128 bytes whose 32
    bytes g hold sub-block 2g's codes low and 2g + 1's high, a sub-block's weight (d x sc) x q - dmin x m
    (gguf_weights._q4_k), so per sub-block scale = d x sc and zero = 8 x d x sc - dmin x m."""
    out = blocks.shape[0]
    b = blocks.reshape(-1, 144)
    d = b[:, :2].contiguous().view(torch.float16).to(torch.float32)
    dmin = b[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
    sc, m = _k_scale_min(b[:, 4:16])
    scale = d * sc.to(torch.float32)
    zero = 8.0 * scale - dmin * m.to(torch.float32)
    qs = b[:, 16:].reshape(-1, 4, 1, 32)
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-2).reshape(out, int(in_features))
    return codes, scale.reshape(out, -1), zero.reshape(out, -1)


def kernel_codes(blocks: torch.Tensor, qtype: str, in_features: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """A GGUF tensor of one of KERNEL_TYPES as the kernel's (codes [out, in], scale [out, in / 32], zero or None), its
    weight (codes - 8) x scale + zero; a ValueError for another type."""
    if qtype == "Q4_0":
        codes, d = q4_0_codes(blocks, in_features)
        return codes, d, None
    if qtype == "Q4_1":
        return q4_1_codes(blocks, in_features)
    if qtype == "Q4_K":
        return q4_k_codes(blocks, in_features)
    raise ValueError(f"torch's int4 kernel takes {', '.join(KERNEL_TYPES)} layers, not {qtype}.")


def w4a16_codes(weight_packed: torch.Tensor) -> torch.Tensor:
    """compressed-tensors' pack-quantized words int32 [out, in / 8] -> codes uint8 [out, in] in 0..15 (nibble j of word
    k is column 8k + j, the weight (code - 8) x scale, as engine_runtime.dequantize_w4a16_full reads it)."""
    w = weight_packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=w.device)
    return ((w.unsqueeze(-1) >> shifts) & 0x0F).reshape(w.shape[0], -1).to(torch.uint8)


class Int4GGUFLinear(GGUFLinear):
    """A Q4_0, Q4_1 or Q4_K `GGUFLinear` whose forward is torch's int4 CPU kernel. It keeps its blocks (on the CPU the
    file's memory map) for every accessor, which read the file's numbers as a packed layer's do, and torch's packing of
    the same codes with each group's scale and zero in the model's dtype (`kernel_codes`) for the forward; a vault
    overlay in force is added to the kernel's product (`with_overlay_terms`); with `int4_enabled` false its forward is
    the packed layer's own (the reference)."""

    int4mm = True

    def __init__(self, blocks: torch.Tensor, in_features: int, out_features: int, *, qtype: str = "Q4_0",
                 dtype: torch.dtype = torch.bfloat16, gguf_name: str = "") -> None:
        super().__init__(blocks, qtype, in_features, out_features, dtype=dtype, gguf_name=gguf_name)
        codes, scale, zero = kernel_codes(blocks, self.qtype, in_features)
        packed, sz = pack(codes, scale, dtype, zero)
        self.register_buffer("int4_packed", packed, persistent=False)
        self.register_buffer("int4_scales_zeros", sz, persistent=False)
        self.int4_enabled = True

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, {self.qtype} int4mm, {self.compute_dtype}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.int4_enabled:
            return super().forward(x)
        return with_overlay_terms(self, x, mm(x, self.int4_packed, self.int4_scales_zeros, self.out_features, self.bias))


def int4_layers(model: nn.Module) -> List[nn.Module]:
    """The layers whose forward is the kernel's: Int4GGUFLinear, and W4A16 modules `install_w4a16` bound."""
    return [m for m in model.modules() if getattr(m, "int4mm", False)]


def _w4a16_forward(module: nn.Module) -> Any:
    from backend import engine_runtime as er

    pinned = er._make_w4a16_linear_forward(module)

    def _int4mm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.int4_enabled:
            return pinned(x)  # the reference: the pinned forward, its overlays laid on
        y = mm(x, self.int4_packed, self.int4_scales_zeros, int(self.weight_packed.shape[0]), getattr(self, "bias", None))
        return with_overlay_terms(self, x, y)

    return types.MethodType(_int4mm_forward, module)


def install_w4a16(rt_or_model: Any, dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    """Bind the kernel's forward on every W4A16 module of a loaded checkpoint (weight_packed and weight_scale kept, for
    the accessors, the vault's edits laid on the kernel's product and the reference forward the layer runs when the
    kernel is turned off), its scales exact (they are bf16). Returns how the linears are held ({"w4a16 int4mm": n,
    "w4a16 packed": m} for those the kernel cannot take); refused when it can take none."""
    why = kernel_missing()
    if why:
        raise ValueError(f"The int4mm linears cannot be used: {why}.")
    model = getattr(rt_or_model, "hf_model", None) or rt_or_model
    kinds: Counter = Counter()
    for _n, mod in model.named_modules():
        if not (hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale")):
            continue
        out_f, in_f = int(mod.weight_packed.shape[0]), int(mod.weight_packed.shape[1]) * 8
        if mod.weight_packed.device.type != "cpu":
            raise ValueError("The int4mm linears are the CPU's (torch.ops.aten._weight_int4pack_mm_for_cpu): this model "
                             f"is on {mod.weight_packed.device}.")
        if not fits(out_f, in_f):
            kinds["w4a16 packed"] += 1
            continue
        packed, sz = pack(w4a16_codes(mod.weight_packed), mod.weight_scale, dtype)
        mod.int4_packed, mod.int4_scales_zeros, mod.int4_enabled, mod.int4mm = packed, sz, True, True
        mod.forward = _w4a16_forward(mod)
        kinds["w4a16 int4mm"] += 1
    if not kinds["w4a16 int4mm"]:
        raise ValueError(f"The int4mm linears cannot be used: the kernel takes none of this model's W4A16 layers "
                         f"({dict(kinds) or 'it has none'}), so its readings would be the reference's under another model id.")
    return dict(sorted(kinds.items()))


def enable(rt_or_model: Any, on: bool) -> int:
    """Turn the kernel on or off in every int4mm layer (off: each runs the reference forward), and the model id's
    `+int4mm` with it when given a runtime, so a reading taken either way is named for the way it was taken. Returns how
    many layers were turned."""
    model = getattr(rt_or_model, "hf_model", None) or rt_or_model
    layers = int4_layers(model)
    for m in layers:
        m.int4_enabled = bool(on)
    mid = getattr(rt_or_model, "model_id", None)
    if layers and isinstance(mid, str):
        base = mid[: -len(SUFFIX)] if mid.endswith(SUFFIX) else mid
        rt_or_model.model_id = base + (SUFFIX if on else "")
    return len(layers)


# ------------------------------------------------------------------------------------- how far it departs, measured


def compare_rows(ref: Sequence[Dict[str, Any]], new: Sequence[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    """Two sets of greedy answers to the same jobs (census rows: `token_ids`, `logprob`, `top_ids`), matched by job,
    compared as neuron-explorer's sweep/check_equivalence.py compares an engine with the reference, on what a census
    reads (scripts/compare_int4mm.py, scripts/check_gguf_cpu.py §4): per
    answer the shared prefix of word-pieces, the step where they split (None when they do not), the largest
    |delta log-prob| of the chosen piece over the shared prefix, and the overlap of the `k` likeliest over it and at
    the split (whose context is still shared); and over
    all, the answers identical, the mean shared prefix as a share of the reference's length, the splits' steps, the
    largest |delta log-prob| and the mean top-k overlap."""
    def key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (r.get("kind"), r.get("neuron"), r.get("prompt"), r.get("dose"))

    by = {key(r): r for r in new}
    per: List[Dict[str, Any]] = []
    for a in ref:
        b = by.get(key(a))
        if b is None:
            continue
        ta, tb = list(a["token_ids"]), list(b["token_ids"])
        n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
            n += 1
        split = None if (n == len(ta) == len(tb)) else n
        dlp = max((abs(float(a["logprob"][i]) - float(b["logprob"][i])) for i in range(n)), default=0.0)
        # the five likeliest compare over the shared prefix and at the split itself (its context still shared)
        steps = min(len(a.get("top_ids") or []), len(b.get("top_ids") or []), n + (1 if n < min(len(ta), len(tb)) else 0))
        ov = [len(set(a["top_ids"][i][:k]) & set(b["top_ids"][i][:k])) / float(k) for i in range(steps)]
        per.append({"job": list(key(a)), "len": len(ta), "shared": n, "split_at": split, "max_abs_dlogprob": dlp,
                    "topk_overlap": (sum(ov) / len(ov)) if ov else None})
    splits = [p["split_at"] for p in per if p["split_at"] is not None]
    ovs = [p["topk_overlap"] for p in per if p["topk_overlap"] is not None]
    return {
        "answers": len(per),
        "identical": sum(p["split_at"] is None for p in per),
        "shared_share": round(sum(p["shared"] / max(1, p["len"]) for p in per) / max(1, len(per)), 4),
        "split_steps": sorted(splits),
        "max_abs_dlogprob": round(max((p["max_abs_dlogprob"] for p in per), default=0.0), 6),
        "topk_overlap": round(sum(ovs) / len(ovs), 4) if ovs else None,
        "k": int(k),
        "per_answer": per,
    }


# the byte offsets of each type's fp16 fields (d, and m or dmin where it has one), for random blocks to time
_F16_AT = {"Q4_0": (0,), "Q4_1": (0, 2), "Q4_K": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,),
           "Q8_0": (0,)}
TIMED_TYPES = ("Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K", "Q8_0")


def random_blocks(qtype: str, out_features: int, in_features: int, g: torch.Generator) -> torch.Tensor:
    """Random `qtype` blocks uint8 [out, row bytes]: random bytes, each fp16 field a small positive number (so every
    weight is finite and of a layer's size)."""
    bs, ts = block_shape(qtype)
    n = out_features * (in_features // bs)
    b = torch.randint(0, 256, (n, ts), generator=g, dtype=torch.int64).to(torch.uint8)
    for off in _F16_AT[qtype]:
        f = (torch.rand(n, 1, generator=g) * 0.018 + 0.002).to(torch.float16)
        b[:, off : off + 2] = f.view(torch.uint8)
    return b.reshape(out_features, -1).contiguous()


def timed_layer(out_features: int, in_features: int, rows: Sequence[int] = (1, 8, 64), reps: int = 3,
                seed: int = 0, qtypes: Sequence[str] = TIMED_TYPES) -> List[Dict[str, Any]]:
    """The kernel against the reference on one random layer of this shape, at each batch of `rows`: the best of `reps`
    wall times (seconds), all in bf16 on this machine's threads (scripts/compare_int4mm.py --time). Per row count:
    `types`, for each of `qtypes` held as random blocks of that type, the packed layer's forward (the reference: unpack,
    then F.linear) and, for KERNEL_TYPES, the kernel's (`int4mm_s`, else None: those stay on the reference path, or
    resident); the resident bf16 product (`resident_bf16_s`, what THEODOLITE_GGUF_RESIDENT_GB buys a layer the kernel
    cannot take); and the W4A16 layout of the Q4_0 layer's codes, its scales as bf16, through the pinned forward
    (`w4a16_reference_s`) and the kernel (`w4a16_int4mm_s`)."""
    from backend import engine_runtime as er

    g = torch.Generator().manual_seed(seed)
    layers: Dict[str, Tuple[GGUFLinear, Optional[Int4GGUFLinear]]] = {}
    for qt in qtypes:
        blocks = random_blocks(qt, out_features, in_features, g)
        layers[qt] = (GGUFLinear(blocks, qt, in_features, out_features, dtype=torch.bfloat16),
                      Int4GGUFLinear(blocks, in_features, out_features, qtype=qt, dtype=torch.bfloat16)
                      if qt in KERNEL_TYPES else None)
    w = next(iter(layers.values()))[0].unpacked()
    # the Q4_0 layer's codes in compressed-tensors' layout: nibble j of word k is column 8k + j
    q40 = random_blocks("Q4_0", out_features, in_features, g)
    codes, d = q4_0_codes(q40, in_features)
    c = codes.to(torch.int64).reshape(out_features, -1, 8) << torch.arange(0, 32, 4, dtype=torch.int64)
    words = c.sum(-1)
    ct = nn.Module()
    ct.weight_packed = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    ct.weight_scale = d.to(torch.bfloat16)
    ct.bias, ct._memit_col_overlays, ct._memit_row_overlays = None, {}, {}
    ct_forward = er._make_w4a16_linear_forward(ct)
    ct_packed, ct_sz = pack(w4a16_codes(ct.weight_packed), ct.weight_scale, torch.bfloat16)

    def best(fn: Any, x: torch.Tensor) -> float:
        fn(x)
        ts = []
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            fn(x)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    out = []
    with torch.no_grad():
        for b in rows:
            x = torch.randn(int(b), in_features, generator=g).to(torch.bfloat16)
            types = {qt: {"reference_s": best(ref, x), "int4mm_s": best(k, x) if k is not None else None}
                     for qt, (ref, k) in layers.items()}
            out.append({"rows": int(b), "types": types, "resident_bf16_s": best(lambda v: F.linear(v, w), x),
                        "w4a16_reference_s": best(ct_forward, x),
                        "w4a16_int4mm_s": best(lambda v: mm(v, ct_packed, ct_sz, out_features), x)})
    return out
