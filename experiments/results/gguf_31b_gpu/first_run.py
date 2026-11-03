#!/usr/bin/env python3
"""The GGUF path on the GPU before `compiled` existed (2 November 2026, main at 1c676c7): the whole model on cuda:0.

    first_run.py GGUF OUT.json [--linears packed] [--resident-gb N]
    first_run.py --cpu-ref GGUF OUT.json          # the same Paris forward on the CPU, packed, for the comparison
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["HF_HUB_OFFLINE"] = "1"
cpu_ref = "--cpu-ref" in sys.argv
if cpu_ref:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
args = [a for a in sys.argv[1:] if a != "--cpu-ref"]

import torch  # noqa: E402

from backend.gguf_runtime import GGUFRuntime  # noqa: E402

gguf, out = args[0], Path(args[1])
linears = args[args.index("--linears") + 1] if "--linears" in args else "packed"
budget = float(args[args.index("--resident-gb") + 1]) if "--resident-gb" in args else None
dev = "cpu" if cpu_ref else "cuda:0"
torch.set_grad_enabled(False)


def sync():
    if dev != "cpu":
        torch.cuda.synchronize()


def gib(x):
    return round(x / 2**30, 2)


rec = {"gguf": gguf, "device": dev, "linears": linears, "resident_gb": budget}
t0 = time.perf_counter()
rt = GGUFRuntime(gguf, device=dev, dtype=torch.bfloat16, linears=linears, load_jpp=False, resident_gb=budget).load()
sync()
rec["load_s"] = round(time.perf_counter() - t0, 1)
card = rt.weights_card
rec.update(model_id=rt.model_id, layers_as=card.get("layers_as"), resident_bytes=card.get("resident_bytes"))
if dev != "cpu":
    rec["vram_after_load_gib"] = gib(torch.cuda.memory_allocated())
print(json.dumps(rec, indent=1), flush=True)

raw = torch.tensor([rt.tokenizer.encode("The capital of France is")])
if rt.tokenizer.bos_token_id is not None and int(raw[0, 0]) != rt.tokenizer.bos_token_id:
    raw = torch.cat([torch.tensor([[rt.tokenizer.bos_token_id]]), raw], 1)
raw = raw.to(dev)
times = []
for _ in range(1 if cpu_ref else 3):
    t0 = time.perf_counter()
    lg = rt.hf_model(raw).logits[0, -1].float().cpu()
    sync()
    times.append(round(time.perf_counter() - t0, 3))
rec["paris_forward_s"] = times
lp = torch.log_softmax(lg, -1)
v, i = lp.topk(5)
rec["paris_top5"] = [(rt.tokenizer.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]
torch.save(lp, out.with_suffix(".paris_lp.pt"))
print("paris:", times, rec["paris_top5"], flush=True)

if not cpu_ref:
    chat = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("Name three rivers in Europe, one line each.",
                                                              chat=True))]).to(dev)
    t0 = time.perf_counter()
    rt.hf_model(chat)
    sync()
    rec["chat_prefill_pieces"], rec["chat_prefill_s"] = int(chat.shape[1]), round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    g = rt.hf_model.generate(chat, attention_mask=torch.ones_like(chat), max_new_tokens=32, do_sample=False)
    sync()
    dt = time.perf_counter() - t0
    n = int(g.shape[1] - chat.shape[1])
    rec["written"], rec["pieces_written"], rec["generate_s"] = rt.tokenizer.decode(g[0, chat.shape[1]:]), n, round(dt, 2)
    rec["s_per_step_after_prefill"] = round((dt - rec["chat_prefill_s"]) / (n - 1), 3) if n > 1 else None
    print("written:", n, rec["s_per_step_after_prefill"], repr(rec["written"]), flush=True)
    rec["vram_peak_gib"] = gib(torch.cuda.max_memory_allocated())
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
