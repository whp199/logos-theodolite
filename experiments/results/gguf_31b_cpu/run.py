#!/usr/bin/env python3
"""The merged GGUF code on a real 31B file, on the CPU.

    run.py int4mm GGUF OUT.json [--resident-gb 12]   # int4mm + budget: load, layers_as, prefill, 16 written,
                                                                # then the kernel off in place, the same prefill
    run.py packed GGUF OUT.json                        # every quantised layer packed: one forward, timed
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402

from backend.gguf_runtime import GGUFRuntime  # noqa: E402


def rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 2**20, 1)


def top(rt, logits, k=5):
    lp = torch.log_softmax(logits.float(), -1)
    v, i = lp.topk(k)
    return [(rt.tokenizer.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)], lp


mode, gguf, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
budget = float(sys.argv[sys.argv.index("--resident-gb") + 1]) if "--resident-gb" in sys.argv else None
torch.set_grad_enabled(False)
rec = {"mode": mode, "gguf": gguf, "threads": torch.get_num_threads(), "loadavg_at_start": os.getloadavg()}
t0 = time.perf_counter()
rt = GGUFRuntime(gguf, device="cpu", dtype=torch.bfloat16, linears="int4mm" if mode == "int4mm" else "packed",
                 load_jpp=False, resident_gb=budget).load()
rec["load_s"] = round(time.perf_counter() - t0, 1)
card = rt.weights_card
rec.update(model_id=rt.model_id, layers_as=card.get("layers_as"), resident_gb=card.get("resident_gb"),
           resident_bytes=card.get("resident_bytes"), at_load_bytes=card.get("at_load_bytes"),
           tensor_types=card.get("tensor_types"), rss_gib_after_load=rss_gib())
print(json.dumps(rec, indent=1), flush=True)
raw = torch.tensor([rt.tokenizer.encode("The capital of France is")])
if rt.tokenizer.bos_token_id is not None and int(raw[0, 0]) != rt.tokenizer.bos_token_id:
    raw = torch.cat([torch.tensor([[rt.tokenizer.bos_token_id]]), raw], 1)
chat = torch.tensor([rt.tokenize_prompt(rt.compose_prompt("Name three rivers in Europe, one line each.", chat=True))])
t0 = time.perf_counter()
lg = rt.hf_model(raw).logits[0, -1]
rec["paris_forward_s"] = round(time.perf_counter() - t0, 1)
rec["paris_top5"], lp_k = top(rt, lg)
print("paris:", rec["paris_forward_s"], rec["paris_top5"], flush=True)
if mode == "int4mm":
    t0 = time.perf_counter()
    lg_c = rt.hf_model(chat).logits[0, -1]
    rec["chat_prefill_pieces"], rec["chat_prefill_s"] = int(chat.shape[1]), round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    g = rt.hf_model.generate(chat, attention_mask=torch.ones_like(chat), max_new_tokens=16, do_sample=False)
    dt = time.perf_counter() - t0
    rec["written_16"] = rt.tokenizer.decode(g[0, chat.shape[1]:])
    rec["s_per_piece_incl_prefill"] = round(dt / 16, 2)
    print("written:", rec["s_per_piece_incl_prefill"], repr(rec["written_16"]), flush=True)
    from backend import int4_cpu as ic

    ic.enable(rt, False)
    t0 = time.perf_counter()
    lg_r = rt.hf_model(raw).logits[0, -1]
    rec["kernel_off_paris_forward_s"] = round(time.perf_counter() - t0, 1)
    rec["kernel_off_paris_top5"], lp_r = top(rt, lg_r)
    rec["kernel_off_model_id"] = rt.model_id
    p_r = lp_r.exp()
    rec["KL(reference || int4mm) at the last position"] = round(float((p_r * (lp_r - lp_k)).sum()), 5)
    ids5 = lp_r.topk(5).indices
    rec["largest |dlogprob| among the reference's five"] = round(float((lp_r[ids5] - lp_k[ids5]).abs().max()), 4)
    print("kernel off:", rec["kernel_off_paris_forward_s"], rec["kernel_off_paris_top5"], flush=True)
rec["rss_gib_end"] = rss_gib()
rec["max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
rec["loadavg_at_end"] = os.getloadavg()
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
