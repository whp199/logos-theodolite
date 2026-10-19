#!/usr/bin/env python3
"""A CPU-only check that `TheodoliteRuntime.capture_rows` returns what interp_engine's `EagerModel.capture` returns, bit
for bit (torch.equal, same dtype and shape), at every row it keeps. Every sweep's recapture, every matched control and
the readers trace read the model through capture_rows, so this is what stands behind "the same values as capture".
It loads neither the real model nor the lens, and hides the GPU.

The stand-in is a tiny random Gemma 4 from transformers (`Gemma4ForCausalLM`): 6 layers, 5 with sliding-window
attention and the sixth with full attention, the real model's pattern (its layer_types put a full layer at every sixth,
50 sliding and 10 full); a window of 8 word-pieces, so every length past 8 crosses it as a sweep past 1024 does on the
real model; attention_k_eq_v, partial rotary on the full layers and the final softcap of 30, as on Gemma-4-31B. It is
wrapped in interp_engine's `EagerModel` with the instrument's own tokenizer (from the local cache), and capture_rows
runs on a stand-in runtime holding only that model. For eager and sdpa attention, in float32 and bfloat16:
1. lengths 1, 3, 8, 9, 17 and 40 (shorter than the window, at it, and past it), read counts 1, 5 and the whole length;
2. the three ways the instrument calls it: a sighting (resid_post at every row, resid_pre / resid_mid / mlp_act /
   mlp_out at the read rows, `TheodoliteRuntime._sweep_points`), a matched control (resid_post at the read rows
   only) and the readers trace (mlp_act over a run of layers, mlp_out at the first, resid_mid after it, last row);
3. unsteered, and inside the steering block a sweep opens (`TheodoliteRuntime._steering_context`, the block the
   writing and its recapture share) with the server's two kinds of nudge (a norm-scaled push and a fixed one, at two
   layers), with and without the spared start mark (position mask [0]); the nudge must change the states, and with
   the mark spared, row 0 must be the unsteered one;
4. capture_rows refuses what it cannot keep exactly: an address asked for twice, a point outside
   `_ROW_CAPTURE_POINTS`, a stream.
What it cannot show is the real weights' numbers at real lengths on the GPU; it checks the code path they go through.

Run: .venv/bin/python -I scripts/check_capture_rows.py   (exits 0 when every check holds; verify_e2e runs it with the
other CPU checks)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # nothing here needs the GPU, and nothing may be fetched: the tokenizer comes from the local cache
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID, TheodoliteRuntime, _ROW_CAPTURE_POINTS
from interp_engine import Address, EagerModel
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, NormScaledAddSpec, SteeringSpec

N_LAYERS = 6
WINDOW = 8
VOCAB = 512
HIDDEN = 64
LENGTHS = (1, 3, 8, 9, 17, 40)
READ_COUNTS = (1, 5)  # and the whole length
BOS = 2

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def stand_in_model(attn: str, dtype: torch.dtype, tokenizer: Any) -> EagerModel:
    """A tiny random Gemma 4 with both kinds of layer and a window shorter than most lengths checked."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size_per_layer_input=0,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        sliding_window=WINDOW,
        layer_types=["sliding_attention"] * (N_LAYERS - 1) + ["full_attention"],
        final_logit_softcapping=30.0,
        max_position_embeddings=4096,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    )
    cfg._attn_implementation = attn
    hf = Gemma4ForCausalLM(cfg).to(dtype).eval()
    # the layer norms and scalars start at plain values: give them some spread, so no layer passes its input through
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if p.dim() == 1 or name.endswith("layer_scalar"):
                p.copy_((torch.rand_like(p, dtype=torch.float32) * 0.8 + 0.6).to(p.dtype))
    return EagerModel(f"stand-in ({attn}, {dtype})", device="cpu", dtype=dtype, hf_model=hf, tokenizer=tokenizer)


def sweep_points() -> tuple:
    every = [Address("resid_post", l) for l in range(N_LAYERS)]
    read = [Address(n, l) for l in range(N_LAYERS) for n in ("resid_pre", "resid_mid", "mlp_act", "mlp_out")]
    return every, read


def styles() -> Dict[str, tuple]:
    every, read = sweep_points()
    readers = [Address("mlp_act", l) for l in range(1, 5)] + [Address("mlp_out", 1)]
    readers += [Address("resid_mid", l) for l in range(2, 5)]
    return {
        "sighting": (every, read),
        "control": ([], [Address("resid_post", l) for l in range(N_LAYERS)]),
        "readers": ([], readers),
    }


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)


def compare(
    em: EagerModel, label: str, ids: List[int], every: Sequence[Address], read: Sequence[Address], n_read: int
) -> Dict[Address, torch.Tensor]:
    """capture_rows against capture on the same ids, inside whatever block is open; returns capture_rows' rows."""
    runtime = SimpleNamespace(eager_model=em)
    full = asyncio.run(em.capture(ids, list(every) + list(read)))
    rows = TheodoliteRuntime.capture_rows(runtime, ids, every=every, read=read, n_read=n_read)  # type: ignore[arg-type]
    seq = len(ids)
    check(set(rows) == set(every) | set(read), f"{label}: came back with {len(rows)} addresses")
    bad = [str(a) for a in every if not same(full[a], rows[a])]
    bad += [str(a) for a in read if not same(full[a][seq - n_read :], rows[a])]
    check(not bad, f"{label}: differs from capture at {bad[:6]}{' ...' if len(bad) > 6 else ''}")
    return rows


def steering_spec(seed: int) -> SteeringSpec:
    """The server's two kinds of nudge (`server._steering_op`), at two layers, on resid_post."""
    g = torch.Generator().manual_seed(seed)
    v1, v2 = torch.randn(HIDDEN, generator=g), torch.randn(HIDDEN, generator=g)
    return SteeringSpec(
        layers={
            1: LayerSteeringSpec(
                operations=[NormScaledAddSpec(vector=v1, strength=0.5, max_fraction=1.0, normalize=True)]
            ),
            3: LayerSteeringSpec(operations=[AddSpec(vector=v2, scale=4.0, normalize=True)]),
        },
        point="resid_post",
    )


def check_model(em: EagerModel, tag: str) -> None:
    before = len(failures)
    g = torch.Generator().manual_seed(1)
    n_compared = 0
    for seq in LENGTHS:
        ids = [BOS] + [int(t) for t in torch.randint(3, VOCAB, (seq - 1,), generator=g)]
        for n_read in sorted({min(c, seq) for c in READ_COUNTS} | {seq}):
            for style, (every, read) in styles().items():
                compare(em, f"{tag}, {style}, {seq} rows, last {n_read} read", ids, every, read, n_read)
                n_compared += 1
    # steered, as a sweep is: the block opened over the prompt, the recapture over prompt + answer inside it
    every, read = sweep_points()
    seq, n_read = 20, 6
    ids = [BOS] + [int(t) for t in torch.randint(3, VOCAB, (seq - 1,), generator=g)]
    prompt = ids[: seq - n_read + 1]
    plain = TheodoliteRuntime.capture_rows(  # type: ignore[arg-type]
        SimpleNamespace(eager_model=em), ids, every=every, read=read, n_read=n_read
    )
    for mask in (None, [0]):
        stand_in = SimpleNamespace(eager_model=em)
        ctx = TheodoliteRuntime._steering_context(stand_in, prompt, steering_spec(7), mask)  # type: ignore[arg-type]
        check(not isinstance(ctx, nullcontext), f"{tag}: no steering block opened (mask {mask})")
        with ctx:
            rows = compare(em, f"{tag}, sighting, steered, mask {mask}", ids, every, read, n_read)
            n_compared += 1
            for style, (s_every, s_read) in styles().items():
                if style != "sighting":
                    compare(em, f"{tag}, {style}, steered, mask {mask}", ids, s_every, s_read, 1)
                    n_compared += 1
        last = Address("resid_post", N_LAYERS - 1)
        check(not torch.equal(rows[last][-1], plain[last][-1]), f"{tag}: the nudge (mask {mask}) changed nothing")
        if mask == [0]:
            check(all(torch.equal(rows[a][0], plain[a][0]) for a in every),
                  f"{tag}: with the start mark spared, row 0 is not the unsteered one")
    if len(failures) == before:
        print(f"  ok    {tag}: {n_compared} comparisons with capture, every one equal")


def check_refusals(em: EagerModel) -> None:
    before = len(failures)
    runtime = SimpleNamespace(eager_model=em)
    ids = [BOS, 5, 6, 7]

    def refused(**kw: Any) -> bool:
        try:
            TheodoliteRuntime.capture_rows(runtime, ids, **kw)  # type: ignore[arg-type]
        except ValueError:
            return True
        return False

    a = Address("resid_post", 1)
    check(refused(every=[a], read=[a]), "an address asked for twice was not refused")
    outside = next(n for n in ("attn_out", "attn_out_post", "mlp_out_post", "mlp_in") if n not in _ROW_CAPTURE_POINTS)
    check(refused(read=[Address(outside, 1)]), f"a point outside _ROW_CAPTURE_POINTS ({outside}) was not refused")
    check(refused(read=[Address("resid_post", 1, stream=0)]), "an address with a stream was not refused")
    if len(failures) == before:
        print("  ok    an address twice, a point it cannot keep exactly and a stream are refused")


def load_tokenizer() -> Optional[Any]:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)
    except Exception as exc:
        failures.append(f"the instrument's tokenizer is not in the local Hugging Face cache ({exc})")
        print(f"  FAIL  {failures[-1]}")
        return None


def main() -> int:
    tokenizer = load_tokenizer()
    if tokenizer is not None:
        first: Optional[EagerModel] = None
        for attn in ("eager", "sdpa"):
            for dtype in (torch.float32, torch.bfloat16):
                em = stand_in_model(attn, dtype, tokenizer)
                first = first or em
                check_model(em, f"{attn}, {str(dtype).replace('torch.', '')}")
        assert first is not None
        check_refusals(first)
    if failures:
        print(f"\ncapture_rows: {len(failures)} of {passed + len(failures)} checks FAILED")
        return 1
    print(f"\ncapture_rows: all {passed} checks hold, every row kept equal to capture's, bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
