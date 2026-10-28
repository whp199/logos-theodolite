"""torch's own int4 kernel on the CPU (`backend/int4_cpu.py`): the 4-bit linears of a Q4_0 GGUF, or of the W4A16
checkpoint, held as torch's packed int4 weights and multiplied by `torch.ops.aten._weight_int4pack_mm_for_cpu`, never
unpacked. Opt in with THEODOLITE_GGUF_LINEARS=int4mm (a GGUF) or THEODOLITE_CHECKPOINT_LINEARS=int4mm (the checkpoint),
on the CPU only (backend/runtime_factory.py).

Why. Every packed forward of the reference path unpacks every weight (the GGUF unpackers, or the pinned W4A16 one),
some 7 to 16 ns a weight on the 4 vCPUs this was written on, so minutes a forward for the 31B's 29.3 billion. torch
2.14's CPU kernel reads the 4-bit codes and a scale and zero per 32 in place. Measured there on one 5,376 x 21,504
layer (ARCHITECTURE §5's table): at one row 2 to 8 ms, against 0.5 to 1.5 s for the reference and 7 to 17 ms for the
resident bf16 product; at 8 rows about as fast as resident bf16; at 64 rows 3 to 5 times slower than it, still 7 to 14
times faster than the reference. It needs 0.625 bytes a weight (18 GB for the 31B's linears), where resident bf16
needs 2 (59 GB).

What it computes. The kernel's weight is (q - 8) x scale + zero, a scale and zero per group of 32 in the activations'
dtype, its sums its own. A Q4_0 block is d x (q - 8) with d in fp16: here scale = d rounded to the model's dtype (bf16
holds 8 of fp16's 11 significant bits), zero 0. A W4A16 group is (q - 8) x scale with scale already bf16: exact. So
the weight is the file's to within the scale's rounding (Q4_0) or exactly (W4A16), but the products are summed in the
kernel's order, not F.linear's: the readings are NOT the reference's bits. Every reading taken so says so: the model id
ends `+int4mm` (`runtime_factory.reading_class`), so a census run, an index or a lens card taken on it is never mixed
with the reference's, and the weights card names the mode (`linears`, and `layers_as`: "Q4_0 int4mm"). How far it
departs from the reference is measured by `compare_rows` after scripts/check_gguf_cpu.py §4 (the shared prefix of
greedy answers, the largest |delta log-prob| over it, the overlap of the five likeliest), on the stand-in there and by
scripts/compare_int4mm.py on the real model.

What it leaves alone. The accessors (`slice_w_*`, `get_full_w_*`, `write_col_norms`, the atlas's weights, the vault's
reads) read the layer's own blocks or `weight_packed` as before, so they give the file's numbers, not the kernel's. A
layer with a vault edit in force runs the reference forward, overlays laid on as `_w4a16_forward` lays them (`int4mm`
has no overlays of its own). The lab's gradient passes (labkit.Lab.grad_linears) run the reference forward and
backward (the kernel has no backward). A layer the kernel cannot take (out features not a multiple of 16, in features
not of 32, a GGUF type other than Q4_0) stays packed on the reference path, as `layers_as` counts. On a GPU it is
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

from backend.gguf_weights import GGUFLinear, with_overlays

KERNEL = "int4mm"
SUFFIX = "+int4mm"  # after the model id's reading class: gguf:<file>#<fp>@cpu/bfloat16+int4mm
GROUP = 32


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


def pack(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """(torch's packed int4 weight uint8 [out, in / 2], scales and zeros [in / 32, out, 2] in `dtype`) for the weight
    (codes - 8) x scale: codes [out, in] in 0..15, scale [out, in / 32] (zeros 0)."""
    packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes.to(torch.int32).contiguous(), 1)
    s = scale.to(torch.float32).to(dtype).T
    return packed, torch.stack([s, torch.zeros_like(s)], -1).contiguous()


def mm(x: torch.Tensor, packed: torch.Tensor, scales_zeros: torch.Tensor, out_features: int,
       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x @ W.T (+ b) through the kernel, any leading shape (the kernel takes a contiguous 2-D input in the dtype of its
    scales)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).to(scales_zeros.dtype).contiguous()
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(x2, packed, GROUP, scales_zeros).reshape(*lead, int(out_features))
    return y if bias is None else y + bias.to(y.dtype)


def q4_0_codes(blocks: torch.Tensor, in_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """A Q4_0 tensor's bytes uint8 [out, in / 32 x 18] -> (codes uint8 [out, in] in 0..15, d float32 [out, in / 32]):
    each block d (fp16) then 16 bytes, byte j holding element j low and j + 16 high (gguf_weights._q4_0)."""
    out = blocks.shape[0]
    b = blocks.reshape(out, int(in_features) // GROUP, 18)
    d = b[..., :2].contiguous().view(torch.float16).to(torch.float32).reshape(out, -1)
    qs = b[..., 2:]
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1).reshape(out, int(in_features))
    return codes, d


def w4a16_codes(weight_packed: torch.Tensor) -> torch.Tensor:
    """compressed-tensors' pack-quantized words int32 [out, in / 8] -> codes uint8 [out, in] in 0..15 (nibble j of word
    k is column 8k + j, the weight (code - 8) x scale, as engine_runtime.dequantize_w4a16_full reads it)."""
    w = weight_packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=w.device)
    return ((w.unsqueeze(-1) >> shifts) & 0x0F).reshape(w.shape[0], -1).to(torch.uint8)


class Int4GGUFLinear(GGUFLinear):
    """A Q4_0 `GGUFLinear` whose forward is torch's int4 CPU kernel. It keeps its blocks (on the CPU the file's memory
    map) for every accessor, which read the file's numbers as a packed layer's do, and torch's packing of the same codes
    with d in the model's dtype for the forward; with a vault overlay in force, or `int4_enabled` false, its forward is
    the packed layer's own (the reference)."""

    int4mm = True

    def __init__(self, blocks: torch.Tensor, in_features: int, out_features: int, *, dtype: torch.dtype = torch.bfloat16,
                 gguf_name: str = "") -> None:
        super().__init__(blocks, "Q4_0", in_features, out_features, dtype=dtype, gguf_name=gguf_name)
        codes, d = q4_0_codes(blocks, in_features)
        packed, sz = pack(codes, d, dtype)
        self.register_buffer("int4_packed", packed, persistent=False)
        self.register_buffer("int4_scales_zeros", sz, persistent=False)
        self.int4_enabled = True

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, Q4_0 int4mm, {self.compute_dtype}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.int4_enabled or self._memit_col_overlays or self._memit_row_overlays:
            return super().forward(x)
        return mm(x, self.int4_packed, self.int4_scales_zeros, self.out_features, self.bias)


def int4_layers(model: nn.Module) -> List[nn.Module]:
    """The layers whose forward is the kernel's: Int4GGUFLinear, and W4A16 modules `install_w4a16` bound."""
    return [m for m in model.modules() if getattr(m, "int4mm", False)]


def _w4a16_forward(module: nn.Module) -> Any:
    from backend import engine_runtime as er

    pinned = er._make_w4a16_linear_forward(module)

    def _int4mm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.int4_enabled or getattr(self, "_memit_col_overlays", None) or getattr(self, "_memit_row_overlays", None):
            return pinned(x)  # the reference: the pinned forward, its overlays laid on
        return mm(x, self.int4_packed, self.int4_scales_zeros, int(self.weight_packed.shape[0]), getattr(self, "bias", None))

    return types.MethodType(_int4mm_forward, module)


def install_w4a16(rt_or_model: Any, dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    """Bind the kernel's forward on every W4A16 module of a loaded checkpoint (weight_packed and weight_scale kept, for
    the accessors and for the reference forward an edited layer runs), its scales exact (they are bf16). Returns how the
    linears are held ({"w4a16 int4mm": n, "w4a16 packed": m} for those the kernel cannot take)."""
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
    compared as scripts/check_gguf_cpu.py §4 and gpu_runtime.md §3.5 compare an engine against the reference: per
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


def timed_layer(out_features: int, in_features: int, rows: Sequence[int] = (1, 8, 64), reps: int = 3,
                seed: int = 0) -> List[Dict[str, Any]]:
    """The kernel against the reference on one random layer of this shape, held as Q4_0 blocks and in the W4A16
    layout (the same codes, the scales as bf16), at each batch of `rows`: the best of `reps` wall times (seconds) of
    the packed GGUF layer's forward (unpack, then F.linear), the pinned W4A16 forward (its unpacker, then F.linear),
    the resident bf16 product, and the kernel on each, all in bf16 on this machine's threads
    (scripts/compare_int4mm.py --time)."""
    from backend import engine_runtime as er

    g = torch.Generator().manual_seed(seed)
    n_blocks = in_features // GROUP
    d = (torch.rand(out_features, n_blocks, 1, generator=g) * 0.018 + 0.002).to(torch.float16)
    qs = torch.randint(0, 256, (out_features, n_blocks, 16), generator=g, dtype=torch.int64).to(torch.uint8)
    blocks = torch.cat([d.view(torch.uint8).reshape(out_features, n_blocks, 2), qs], -1).reshape(out_features, -1)
    packed_layer = GGUFLinear(blocks.contiguous(), "Q4_0", in_features, out_features, dtype=torch.bfloat16)
    int4_layer = Int4GGUFLinear(blocks.contiguous(), in_features, out_features, dtype=torch.bfloat16)
    w = packed_layer.unpacked()
    # the same codes in compressed-tensors' layout: nibble j of word k is column 8k + j
    codes, _d = q4_0_codes(blocks, in_features)
    c = codes.to(torch.int64).reshape(out_features, -1, 8) << torch.arange(0, 32, 4, dtype=torch.int64)
    words = c.sum(-1)
    ct = nn.Module()
    ct.weight_packed = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    ct.weight_scale = d.reshape(out_features, n_blocks).to(torch.bfloat16)
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
            out.append({"rows": int(b), "reference_s": best(packed_layer, x), "w4a16_reference_s": best(ct_forward, x),
                        "resident_bf16_s": best(lambda v: F.linear(v, w), x), "int4mm_s": best(int4_layer, x),
                        "w4a16_int4mm_s": best(lambda v: mm(v, ct_packed, ct_sz, out_features), x)})
    return out
