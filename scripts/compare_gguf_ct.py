#!/usr/bin/env python3
"""Does a GGUF hold the same weights as the compressed-tensors W4A16 checkpoint the instrument reads?

    .venv/bin/python -I scripts/compare_gguf_ct.py model.gguf                      # against the cached checkpoint
    .venv/bin/python -I scripts/compare_gguf_ct.py model.gguf --ct /path/to/ckpt --json report.json --layers 40-59
    .venv/bin/python -I scripts/compare_gguf_ct.py --self-test                    # CPU: synthetic files, both ways

Both are streamed, tensor by tensor (the GGUF through gguf-py's memory map, the checkpoint's safetensors shards through
safe_open, so only one tensor pair is in memory at a time), their names mapped by gguf-py's own map for Gemma 4
(model.language_model.layers.N.mlp.down_proj -> blk.N.ffn_down), and both dequantised to float32: the checkpoint's
W4A16 as (nibble - 8) x scale (exact in float32), the GGUF's as gguf-py dequantises it (backend/gguf_weights.py, bit
for bit). Each pair is compared bit for bit, and where they differ the report gives the largest |difference| and the
share of elements equal. Equal in float32 means equal in the bfloat16 the instrument computes in too.

For a Q4_0 tensor against a W4A16 one it also compares the raw codes and the scales, since both are group-32 symmetric
4-bit, w = (q - 8) x scale:
- W4A16 (compressed-tensors pack-quantized): weight_packed int32 [out, in / 8], eight consecutive elements per int32,
  low nibble first; weight_scale bfloat16 [out, in / 32];
- Q4_0: a block of 32 is an fp16 scale and 16 bytes, byte j holding element j (low nibble) and element j + 16 (high).
Q4_0's scale is fp16 and the checkpoint's bf16, so the report says how many of the GGUF's scales a bf16 holds exactly
(the GGUF can be the checkpoint only if every one is) and how many of the checkpoint's an fp16 holds (a Q4_0 copy of
the checkpoint can be exact only if every one is: fp16 has three more mantissa bits than bf16 but a far narrower
range, and loses bits below 2^-14).

Norms are compared as stored; a GGUF whose norm weights sit exactly 1 above the checkpoint's is named as holding Gemma 2
and 3's convention (Gemma 4's GGUFs store w itself: backend/gguf_model.py). The vision and audio towers are not in a
text GGUF and are counted apart. The exit code is 0 when every tensor of the GGUF is the checkpoint's, bit for bit.

`--self-test` writes, from the same quantised weights of a tiny random Gemma 4 (scripts/atlas_stand_in.py), a
checkpoint in the W4A16 layout and a Q4_0 GGUF, and checks that the comparison finds them identical; then that it
finds one perturbed code, one perturbed scale and a +1 norm convention, each where it was put.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

CT_PREFIX = "model.language_model."
OTHER_TOWERS = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")


# ------------------------------------------------------------------------------- the two packings, codes and scales


def ct_codes(packed: torch.Tensor) -> torch.Tensor:
    """W4A16 weight_packed int32 [out, in / 8] -> the 4-bit codes uint8 [out, in] (0..15; w = (q - 8) x scale)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(packed.shape[0], -1).to(torch.uint8)


def ct_pack(codes: torch.Tensor) -> torch.Tensor:
    """The inverse of `ct_codes`: codes uint8 [out, in] -> weight_packed int32 [out, in / 8]."""
    c = codes.to(torch.int64).reshape(codes.shape[0], -1, 8)
    word = (c << torch.arange(0, 32, 4, dtype=torch.int64)).sum(-1)
    return torch.from_numpy((word.numpy() & 0xFFFFFFFF).astype(np.uint32).view(np.int32))


def q4_0_codes_scales(blocks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q4_0 bytes uint8 [out, in / 32 x 18] -> (codes uint8 [out, in], scales float32 [out, in / 32])."""
    out = blocks.shape[0]
    b = blocks.reshape(out, -1, 18)
    d = b[..., :2].contiguous().view(torch.float16).to(torch.float32).reshape(out, -1)
    qs = b[..., 2:]
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1).reshape(out, -1)
    return codes, d


def q4_0_blocks(codes: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """codes uint8 [out, in] and fp16-held scales [out, in / 32] -> Q4_0 bytes uint8 [out, in / 32 x 18]."""
    out = codes.shape[0]
    c = codes.reshape(out, -1, 32).to(torch.uint8)
    qs = c[..., :16] | (c[..., 16:] << 4)
    d = scales.to(torch.float16).reshape(out, -1, 1).contiguous().view(torch.uint8).reshape(out, -1, 2)
    return torch.cat([d, qs], dim=-1).reshape(out, -1).numpy()


def ct_dense(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W4A16 as float32 [out, in]: (q - 8) x scale, exact (a 4-bit integer times a bf16 fits a float32)."""
    q = ct_codes(packed).to(torch.float32) - 8.0
    return (q.reshape(q.shape[0], -1, 32) * scale.to(torch.float32).unsqueeze(-1)).reshape(q.shape[0], -1)


def representable(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Where a float32 holds a value `dtype` holds exactly (bool, same shape)."""
    y = x.to(dtype).to(torch.float32)
    return (y == x) | (torch.isnan(x) & torch.isnan(y))


# ------------------------------------------------------------------------------------------- reading the two sides


class Checkpoint:
    """The W4A16 checkpoint's tensors by name, read one at a time from its safetensors shards."""

    def __init__(self, folder: Path) -> None:
        from safetensors import safe_open

        self.folder = folder
        self.where: Dict[str, Path] = {}
        shards = sorted(folder.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"No .safetensors in {folder}.")
        for s in shards:
            with safe_open(str(s), "pt") as f:
                for k in f.keys():
                    self.where[k] = s
        self._open: Dict[Path, Any] = {}

    def get(self, name: str) -> Optional[torch.Tensor]:
        from safetensors import safe_open

        s = self.where.get(name)
        if s is None:
            return None
        if s not in self._open:
            self._open = {s: safe_open(str(s), "pt")}  # one shard open at a time
        return self._open[s].get_tensor(name)

    def names(self) -> List[str]:
        return sorted(self.where)


def find_checkpoint(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(spec, local_files_only=True, allow_patterns=["*.safetensors", "*.json"]))
    except Exception as exc:
        raise SystemExit(f"The checkpoint {spec!r} is neither a folder nor in the local Hugging Face cache ({type(exc).__name__}).")


def hf_name_of(ct_name: str) -> Optional[str]:
    """A checkpoint tensor's name as Gemma4ForCausalLM has it (None for the vision and audio towers)."""
    if ct_name.startswith(OTHER_TOWERS):
        return None
    for suffix in (".weight_packed", ".weight_scale", ".weight_shape", ".weight_zero_point", ".weight_g_idx"):
        if ct_name.endswith(suffix):
            ct_name = ct_name[: -len(suffix)] + ".weight"
    return "model." + ct_name[len(CT_PREFIX):] if ct_name.startswith(CT_PREFIX) else ct_name


# ------------------------------------------------------------------------------------------------- the comparison


CHUNK_ROWS = 2048  # rows compared at a time, so a 262,144-row embedding never sits whole in float32 twice


def compare_tensor(name: str, t: Any, shape: Tuple[int, ...], packed: Optional[torch.Tensor],
                   scale: Optional[torch.Tensor], plain: Optional[torch.Tensor]) -> Dict[str, Any]:
    """One GGUF tensor against its checkpoint tensor, a chunk of rows at a time: bit for bit, else the largest
    |difference| and the share equal; for a norm, whether the GGUF's is the checkpoint's + 1 exactly; for Q4_0 against
    W4A16, the codes and the scales."""
    from backend.gguf_weights import type_name, unpack

    rows = shape[0] if len(shape) > 1 else 1
    raw = torch.from_numpy(np.asarray(t.data).view(np.uint8).reshape(rows, -1))
    rec: Dict[str, Any] = {"shape": list(shape)}
    if packed is None:
        if plain is None or plain.numel() != int(np.prod(shape)):
            rec.update(equal=False, problem=f"shapes differ: GGUF {list(shape)}, checkpoint {list(plain.shape) if plain is not None else None}")
            return rec
        plain = plain.reshape(rows, -1)
    q4 = packed is not None and type_name(t.tensor_type) == "Q4_0"
    n = n_eq = codes_eq = n_codes = scales_eq = n_scales = bf16_ok = fp16_ok = 0
    max_diff, plus_one = 0.0, True
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, rows)
        g = unpack(raw[r0:r1], t.tensor_type)
        c = ct_dense(packed[r0:r1], scale[r0:r1]) if packed is not None else plain[r0:r1].to(torch.float32)
        if tuple(g.shape) != tuple(c.shape):
            rec.update(equal=False, problem=f"shapes differ: GGUF {list(shape)}, checkpoint rows of {c.shape[-1]}")
            return rec
        same = g.contiguous().view(torch.int32) == c.contiguous().view(torch.int32)
        n += same.numel()
        n_eq += int(same.sum())
        if not bool(same.all()):
            max_diff = max(max_diff, float((g - c).abs().max()))
        plus_one = plus_one and bool(torch.equal(g, c + 1.0))
        if q4:
            gc, gs = q4_0_codes_scales(raw[r0:r1])
            cc, cs = ct_codes(packed[r0:r1]), scale[r0:r1].to(torch.float32)
            codes_eq += int((gc == cc).sum())
            n_codes += gc.numel()
            scales_eq += int((gs == cs).sum())
            n_scales += gs.numel()
            bf16_ok += int(representable(gs, torch.bfloat16).sum())
            fp16_ok += int(representable(cs, torch.float16).sum())
    rec["equal"] = n_eq == n
    if not rec["equal"]:
        rec["share_equal"] = n_eq / n
        rec["max_abs_diff"] = max_diff
        if name.endswith("norm.weight") and plus_one:
            rec["norm_plus_one"] = True
    if q4:
        rec.update(codes_equal=codes_eq == n_codes, codes_share_equal=codes_eq / n_codes, scales_equal=scales_eq == n_scales,
                   gguf_scales_bf16_exact=bf16_ok / n_scales, ct_scales_fp16_exact=fp16_ok / n_scales)
    return rec


def compare(gguf_path: Path, ckpt: Checkpoint, *, layers: Optional[List[int]] = None, limit: int = 0,
            log: Any = print) -> Dict[str, Any]:
    """The report: one entry per GGUF tensor, the counts, and the tensors either side lacks."""
    from backend import gguf_model as gm
    from backend.gguf_weights import type_name

    reader = gm.open_gguf(gguf_path)
    tensors = gm.tensor_table(reader)
    n_layers = int(gm.read_metadata(reader).get(f"{gm.ARCH}.block_count", 0) or 0)
    to_gguf = gm.hf_to_gguf_names(max(n_layers, 1))
    ct_by_gguf: Dict[str, str] = {}
    towers = 0
    for k in ckpt.names():
        hf = hf_name_of(k)
        if hf is None:
            towers += 1
            continue
        g = to_gguf(hf)
        if g is not None:
            base = k.rsplit(".", 1)[0] if k.endswith((".weight_packed", ".weight_scale", ".weight_shape")) else k
            ct_by_gguf.setdefault(g, base)
    report: Dict[str, Any] = {"gguf": str(gguf_path), "checkpoint": str(ckpt.folder), "tensors": {},
                              "checkpoint_tensors_in_other_towers": towers}
    names = [n for n in tensors if n != "rope_freqs.weight"]
    if layers is not None:
        keep = {f"blk.{l}." for l in layers}
        names = [n for n in names if not n.startswith("blk.") or any(n.startswith(k) for k in keep)]
    if limit:
        names = names[:limit]
    for i, name in enumerate(names):
        t = tensors[name]
        base = ct_by_gguf.get(name)
        rec: Dict[str, Any] = {"gguf_type": type_name(t.tensor_type)}
        if base is None:
            rec.update(equal=False, problem="not in the checkpoint")
            report["tensors"][name] = rec
            continue
        packed = ckpt.get(base + ".weight_packed")
        scale = ckpt.get(base + ".weight_scale") if packed is not None else None
        plain = ckpt.get(base) if packed is None else None
        rec["ct_type"] = "W4A16" if packed is not None else (str(plain.dtype).replace("torch.", "") if plain is not None else None)
        if packed is None and plain is None:
            rec.update(equal=False, problem="not in the checkpoint")
        else:
            rec.update(compare_tensor(name, t, gm.logical_shape(t), packed, scale, plain))
        del packed, scale, plain
        report["tensors"][name] = rec
        if log and (not rec.get("equal") or i % 50 == 0):
            log(f"  {name:<36} {rec['gguf_type']:>5} vs {rec.get('ct_type')}: "
                + ("equal" if rec.get("equal") else
                   rec.get("problem") or f"{rec['share_equal']:.6f} equal, max |diff| {rec['max_abs_diff']:.3g}"
                   + (" (norm = checkpoint + 1: Gemma 2/3's convention)" if rec.get("norm_plus_one") else "")))
    done = report["tensors"]
    seen = set(done)
    report["summary"] = {
        "compared": len(done),
        "equal": sum(1 for r in done.values() if r.get("equal")),
        "differ": sorted(n for n, r in done.items() if not r.get("equal")),
        "norms_plus_one": sorted(n for n, r in done.items() if r.get("norm_plus_one")),
        "q4_0_codes_equal": sum(1 for r in done.values() if r.get("codes_equal")),
        "q4_0_scales_equal": sum(1 for r in done.values() if r.get("scales_equal")),
        "q4_0_gguf_scales_bf16_exact": min((r["gguf_scales_bf16_exact"] for r in done.values() if "gguf_scales_bf16_exact" in r),
                                           default=None),
        "checkpoint_only": sorted(g for g in ct_by_gguf if g not in seen and (layers is None and not limit)),
    }
    return report


# -------------------------------------------------------------------------------------------------- the self-test


def write_pair(tmp: Path, *, bad_code: Optional[str] = None, bad_scale: Optional[str] = None,
               norm_shift: float = 0.0, seed: int = 0) -> Tuple[Path, Path, Dict[str, Any]]:
    """A checkpoint folder in the W4A16 layout and a Q4_0 GGUF, written from the same codes, scales and bf16 tensors
    of the stand-in's shape; `bad_code` / `bad_scale` name a GGUF tensor to perturb (one code, one scale)."""
    from safetensors.torch import save_file

    from atlas_stand_in import stand_in_hf, stand_in_tokenizer
    from backend import gguf_model as gm

    tok = stand_in_tokenizer()
    hf = stand_in_hf(tok, seed=seed)
    to_gguf = gm.hf_to_gguf_names(len(hf.model.layers))
    g = torch.Generator().manual_seed(seed + 11)
    ct: Dict[str, torch.Tensor] = {}
    blocks: Dict[str, Tuple[np.ndarray, str]] = {}
    where: Dict[str, Any] = {}
    with torch.no_grad():
        for name, p in hf.named_parameters():
            ct_name = CT_PREFIX + name[len("model."):] if name.startswith("model.") else name
            if name == "lm_head.weight":
                continue
            gname = to_gguf(name)
            if p.dim() == 2 and ".layers." in name:
                out, inn = p.shape
                codes = torch.randint(0, 16, (out, inn), generator=g, dtype=torch.uint8)
                # bf16 scales that an fp16 also holds: 2^-e x (1 + m / 128), e in 6..9
                e = torch.randint(6, 10, (out, inn // 32), generator=g).float()
                m = torch.randint(0, 128, (out, inn // 32), generator=g).float()
                scale = (2.0 ** -e * (1 + m / 128)).to(torch.bfloat16)
                ct[ct_name.replace(".weight", ".weight_packed")] = ct_pack(codes)
                ct[ct_name.replace(".weight", ".weight_scale")] = scale
                ct[ct_name.replace(".weight", ".weight_shape")] = torch.tensor([out, inn], dtype=torch.int64)
                gcodes, gscale = codes.clone(), scale.to(torch.float32).clone()
                if gname == bad_code:
                    gcodes[1, 5] = (gcodes[1, 5] + 1) % 16
                    where["code"] = (gname, 1, 5)
                if gname == bad_scale:
                    gscale[2, 0] = gscale[2, 0] * 2
                    where["scale"] = (gname, 2, 0)
                blocks[gname] = (q4_0_blocks(gcodes, gscale), "Q4_0")
                p.copy_(ct_dense(ct[ct_name.replace(".weight", ".weight_packed")], scale))
            else:
                v = p.to(torch.bfloat16)
                ct[ct_name] = v.clone()
                p.copy_(v.float())
        for name, b in hf.named_buffers():
            if name.endswith("layer_scalar"):
                v = b.to(torch.bfloat16)
                ct[CT_PREFIX + name[len("model."):]] = v.clone()
                b.copy_(v.float())
    folder = tmp / f"ct-{seed}-{bad_code}-{bad_scale}-{norm_shift}"
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ct.items()}, str(folder / "model.safetensors"))
    path = gm.write_gemma4_gguf(hf, tmp / f"pair-{seed}-{bad_code}-{bad_scale}-{norm_shift}.gguf", tokenizer=tok,
                                qtype="F32", blocks=blocks, norm_shift=norm_shift)
    return path, folder, where


def self_test(tmp: Path, log: Any = None) -> List[Tuple[bool, str]]:
    """[(held, what)] for the four synthetic cases (see the module's docstring)."""
    out: List[Tuple[bool, str]] = []
    path, folder, _ = write_pair(tmp)
    rep = compare(path, Checkpoint(folder), log=log)
    s = rep["summary"]
    n_lin = sum(1 for r in rep["tensors"].values() if r.get("gguf_type") == "Q4_0")
    out.append((s["compared"] > 0 and not s["differ"] and s["equal"] == s["compared"],
                f"written both ways from the same weights: every tensor equal ({s['equal']} of {s['compared']})"))
    out.append((n_lin > 0 and s["q4_0_codes_equal"] == n_lin and s["q4_0_scales_equal"] == n_lin
                and s["q4_0_gguf_scales_bf16_exact"] == 1.0 and not s["checkpoint_only"],
                f"the {n_lin} Q4_0 tensors: codes and scales equal, every scale bf16-exact, nothing left over"))
    bad = "blk.3.ffn_down.weight"
    path, folder, where = write_pair(tmp, bad_code=bad)
    rep = compare(path, Checkpoint(folder), log=log)
    s, r = rep["summary"], rep["tensors"][bad]
    out.append((s["differ"] == [bad] and not r["codes_equal"] and r["scales_equal"]
                and abs(r["codes_share_equal"] - (1 - 1 / (r["shape"][0] * r["shape"][1]))) < 1e-9,
                f"one code changed: found in {s['differ']} alone, one element of its codes"))
    bad = "blk.0.attn_q.weight"
    path, folder, where = write_pair(tmp, bad_scale=bad)
    rep = compare(path, Checkpoint(folder), log=log)
    r = rep["tensors"][bad]
    n = r["shape"][0] * r["shape"][1]
    out.append((rep["summary"]["differ"] == [bad] and r["codes_equal"] and not r["scales_equal"]
                and 1 - 32 / n - 1e-9 <= r["share_equal"] < 1,
                f"one scale doubled: found in {bad} alone, within one block of 32 (its zero codes still equal)"))
    path, folder, _ = write_pair(tmp, norm_shift=1.0)
    s = compare(path, Checkpoint(folder), log=log)["summary"]
    norms = [n for n in s["differ"]]
    out.append((bool(norms) and set(norms) == set(s["norms_plus_one"]) and all(n.endswith("norm.weight") for n in norms),
                f"a GGUF in Gemma 2/3's norm convention: all {len(norms)} norms named as the checkpoint's + 1, nothing else"))
    return out


def parse_layers(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        if a:
            out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("gguf", nargs="?", help="the GGUF file")
    ap.add_argument("--ct", default="", help="the W4A16 checkpoint: a folder, or a repo id in the local Hugging Face "
                    "cache (default: the instrument's, google/gemma-4-31B-it-qat-w4a16-ct)")
    ap.add_argument("--layers", default="", help='compare only these blocks (and the rest of the model): "40-59"')
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tensors")
    ap.add_argument("--json", default="", help="write the whole report here")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic self-test on the CPU")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    if a.self_test:
        with tempfile.TemporaryDirectory(prefix="compare-gguf-ct-") as d:
            results = self_test(Path(d))
        for ok, what in results:
            print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        bad = sum(1 for ok, _ in results if not ok)
        print(f"\nself-test: {len(results) - bad} of {len(results)} hold")
        return 1 if bad else 0
    if not a.gguf:
        ap.error("name a GGUF file (or --self-test)")
    from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID

    ckpt = Checkpoint(find_checkpoint(a.ct or DEFAULT_W4A16_MODEL_ID))
    rep = compare(Path(a.gguf), ckpt, layers=parse_layers(a.layers) if a.layers else None, limit=a.limit)
    s = rep["summary"]
    print(f"\n{s['equal']} of {s['compared']} GGUF tensors are the checkpoint's bit for bit; {len(s['differ'])} differ"
          + (f" ({len(s['norms_plus_one'])} of them norms exactly 1 above: Gemma 2/3's convention)" if s["norms_plus_one"] else ""))
    if s["q4_0_gguf_scales_bf16_exact"] is not None:
        print(f"Q4_0 tensors: codes equal in {s['q4_0_codes_equal']}, scales equal in {s['q4_0_scales_equal']}; the share "
              f"of the GGUF's scales a bf16 holds exactly is at least {s['q4_0_gguf_scales_bf16_exact']:.6f}")
    if s["checkpoint_only"]:
        print(f"In the checkpoint, not the GGUF: {len(s['checkpoint_only'])} tensors ({', '.join(s['checkpoint_only'][:4])} ...)")
    print(f"({rep['checkpoint_tensors_in_other_towers']} checkpoint tensors belong to the vision and audio towers, which a text GGUF leaves out)")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"-> {a.json}")
    return 0 if not s["differ"] and not s["checkpoint_only"] else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
