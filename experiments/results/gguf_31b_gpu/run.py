#!/usr/bin/env python3
"""Gemma-4-31B from a GGUF on the GPU: the reference unpacker against `compiled` (backend/gguf_fast.py), one model.

    run.py gpu GGUF OUT.json [--lens]   # cuda:0, packed, bf16: measured on the reference, then compiled installed in
                                        # place and measured again; every reading compared bit for bit
    run.py cpu GGUF OUT.json            # on the CPU: one forward of the first prompt each way (no generation)

Measured each way: a forward of "The capital of France is" (three times, the first warming), a chat prefill, up to 32
written word-pieces (greedy; it stops at an end mark), and 8 written at a batch of 32 copies of the chat prompt (the
end mark suppressed, so all 8 are written); with --lens also the optics check's first sighting (scripts/verify_optics.py:
60 layers through the J++ lens, 6 written), taken twice on the reference. Compared: the logits by their bits, the
pieces written, and every number of the sighting but its clock and memory gauge (elapsed_ms, analyse_ms,
peak_vram_gb), the reference's two sightings with each other too. `compiled_fell_back` counts the calls the compiled
graphs were not tested for, which went through the reference instead.
"""

import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["HF_HUB_OFFLINE"] = "1"
mode, gguf, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if mode == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
lens = "--lens" in sys.argv

import torch  # noqa: E402

from backend import gguf_weights as gw  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.gguf_runtime import GGUFRuntime  # noqa: E402

dev = "cuda:0" if mode == "gpu" else "cpu"
gpu = dev != "cpu"
torch.set_grad_enabled(False)


def sync():
    if gpu:
        torch.cuda.synchronize()


def gib(x):
    return round(x / 2**30, 2)


def bits(t):
    if not t.is_floating_point():
        return t.detach().cpu()
    return t.detach().contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]).cpu()


# a sighting's own clock and memory gauge, which differ run to run whatever the numbers
NOT_READINGS = ("elapsed_ms", "analyse_ms", "peak_vram_gb")


def differ(a, b, at="sighting"):
    """Where two sightings' records hold other values (tensors by their bits; NaN equal to NaN), their clock and memory
    gauge left out: a list of paths, empty where they agree."""
    if dataclasses.is_dataclass(a):
        if type(a) is not type(b):
            return [at]
        return [p for f in dataclasses.fields(a) if f.name not in NOT_READINGS
                for p in differ(getattr(a, f.name), getattr(b, f.name), f"{at}.{f.name}")]
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [at]
        return [p for k in a for p in differ(a[k], b[k], f"{at}[{k!r}]")]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [at]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differ(x, y, f"{at}[{i}]")]
    if isinstance(a, torch.Tensor):
        ok = isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))
        return [] if ok else [at]
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return []
    return [] if a == b else [at]


rec = {"mode": mode, "gguf": gguf, "device": dev, "lens": lens, "threads": torch.get_num_threads()}
t0 = time.perf_counter()
rt = GGUFRuntime(gguf, device=dev, dtype=torch.bfloat16, linears="packed", load_jpp=lens).load()
sync()
rec["load_s"] = round(time.perf_counter() - t0, 1)
rec.update(model_id=rt.model_id, layers_as=rt.weights_card.get("layers_as"))
if gpu:
    rec["vram_after_load_gib"] = gib(torch.cuda.memory_allocated())
print(json.dumps(rec, indent=1), flush=True)

raw = torch.tensor([rt.tokenizer.encode("The capital of France is")])
if rt.tokenizer.bos_token_id is not None and int(raw[0, 0]) != rt.tokenizer.bos_token_id:
    raw = torch.cat([torch.tensor([[rt.tokenizer.bos_token_id]]), raw], 1)
raw = raw.to(dev)
chat = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("Name three rivers in Europe, one line each.",
                                                          chat=True))]).to(dev)
batch = chat.repeat(32, 1)


def sighting():
    from verify_optics import BASE_PROMPT

    sw = asyncio.run(rt.run_theodolite_sweep(
        BASE_PROMPT, monument_concepts=("spider", "eight", "insect"), survey_mode="monument", max_new_tokens=6,
        top_k_poles=15, tracked_concepts=("spider", "arachnid", "eight", "8", "insect", "six", "6")))
    sync()
    return sw


def measure(tag):
    m, keep = {}, {}
    times = []
    for _ in range(3 if gpu else 1):
        t0 = time.perf_counter()
        lg = rt.hf_model(raw).logits[0, -1]
        sync()
        times.append(round(time.perf_counter() - t0, 3))
    m["paris_forward_s"] = times
    lp = torch.log_softmax(lg.float(), -1)
    v, i = lp.topk(5)
    m["paris_top5"] = [(rt.tokenizer.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]
    keep["paris"] = bits(lg)
    if gpu:
        t0 = time.perf_counter()
        lc = rt.hf_model(chat).logits[0, -1]
        sync()
        m["chat_prefill_pieces"], m["chat_prefill_s"] = int(chat.shape[1]), round(time.perf_counter() - t0, 3)
        keep["chat"] = bits(lc)
        t0 = time.perf_counter()
        g = rt.hf_model.generate(chat, attention_mask=torch.ones_like(chat), max_new_tokens=32, do_sample=False)
        sync()
        dt = time.perf_counter() - t0
        n = int(g.shape[1] - chat.shape[1])
        m["written"], m["pieces_written"], m["generate_s"] = rt.tokenizer.decode(g[0, chat.shape[1]:]), n, round(dt, 3)
        m["s_per_step_after_prefill"] = round((dt - m["chat_prefill_s"]) / (n - 1), 4) if n > 1 else None
        keep["written"] = g[0].cpu()
        t0 = time.perf_counter()
        rt.hf_model(batch)
        sync()
        pre32 = time.perf_counter() - t0
        t0 = time.perf_counter()
        g32 = rt.hf_model.generate(batch, attention_mask=torch.ones_like(batch), max_new_tokens=8, min_new_tokens=8,
                                   do_sample=False)
        sync()
        dt = time.perf_counter() - t0
        m["batch32_prefill_s"] = round(pre32, 3)
        m["batch32_generate_s"] = round(dt, 3)
        m["batch32_s_per_step_after_prefill"] = round((dt - pre32) / 7, 4)
        keep["batch32"] = g32.cpu()
    if lens:
        t0 = time.perf_counter()
        sw = sighting()
        m["sighting_s"] = round(time.perf_counter() - t0, 2)
        m["sighting_written"] = sw.generated_text
        keep["sighting"] = sw
    if gpu:
        m["vram_peak_gib"] = gib(torch.cuda.max_memory_allocated())
    print(tag, json.dumps(m), flush=True)
    return m, keep


rec["reference"], ref = measure("reference")
if lens:  # the sighting twice on the reference: what differs between two runs of the same reading, if anything
    rec["sighting_reference_twice_differs"] = differ(ref["sighting"], sighting())[:40]
t0 = time.perf_counter()
rec["install"] = w4a16_fast.install(rt, "compiled", log=lambda s: print(s, flush=True))
rec["install_wall_s"] = round(time.perf_counter() - t0, 1)
if gpu:
    torch.cuda.reset_peak_memory_stats()
rec["compiled"], cmp = measure("compiled")
fast = gw._FAST_UNPACK
rec["compiled_fell_back"] = getattr(fast, "fell_back", None)
rec["describe"] = w4a16_fast.describe(rt)
rec["model_id_after_install"] = rt.model_id
rec["same_bits"] = {k: (bool(torch.equal(ref[k], cmp[k])) if isinstance(ref[k], torch.Tensor) else not differ(ref[k], cmp[k]))
                    for k in ref}
if "sighting" in ref:
    rec["sighting_differs"] = differ(ref["sighting"], cmp["sighting"])[:40]
out.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("install_wall_s", "compiled_fell_back", "same_bits")}, indent=1))
