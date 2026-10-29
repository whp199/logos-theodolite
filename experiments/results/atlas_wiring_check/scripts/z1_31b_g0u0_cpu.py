# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: the 31B's own gate and up values (g0, u0) at the check's read sites, unpushed, on the CPU.
One prefill per census question on the W4A16 checkpoint (packed, the reference path), hooks on every layer's
gate_proj and up_proj outputs and down_proj's input. Saves last-word-piece g0, u0, act [Q, L, N] float32 and the
per-position g, u (float16) for the mean site. Writes only to the scratch folder."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["THEODOLITE_DEVICE"] = "cpu"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(22)
OUT = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/z1_out"
os.makedirs(OUT, exist_ok=True)

from backend.runtime_factory import make_runtime
from backend.causal_check import language_model
from backend import census as cz

t0 = time.time()
rt = make_runtime(load_jlens=False).load()
print("loaded", rt.model_id, f"{time.time()-t0:.0f}s", flush=True)
lm = language_model(rt.hf_model)
L = len(lm.layers)
prompts = {k: rt.tokenize_prompt(rt.compose_prompt(v, chat=True)) for k, v in cz.PROMPTS.items()}
print({k: len(v) for k, v in prompts.items()}, flush=True)
for qi, (name, ids) in enumerate(prompts.items()):
    if os.path.exists(f"{OUT}/q{qi}.npz"):
        continue
    G, U, A = {}, {}, {}
    hooks = []
    for j in range(L):
        mlp = lm.layers[j].mlp
        hooks.append(mlp.gate_proj.register_forward_hook(lambda m, a, o, j=j: G.__setitem__(j, o[0].float().cpu())))
        hooks.append(mlp.up_proj.register_forward_hook(lambda m, a, o, j=j: U.__setitem__(j, o[0].float().cpu())))
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, a, j=j: A.__setitem__(j, a[0][0].float().cpu())))
    t1 = time.time()
    try:
        x = torch.tensor([ids], dtype=torch.long)
        lm(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    print(name, "forward", f"{time.time()-t1:.0f}s", len(G), len(U), len(A), flush=True)
    g = torch.stack([G[j] for j in range(L)])  # [L, T, N]
    u = torch.stack([U[j] for j in range(L)])
    a = torch.stack([A[j] for j in range(L)])
    np.savez(f"{OUT}/q{qi}.npz", name=name, ids=np.array(ids), g_last=g[:, -1].numpy(), u_last=u[:, -1].numpy(),
             a_last=a[:, -1].numpy(), g_all=g.numpy().astype(np.float16), u_all=u.numpy().astype(np.float16),
             a_mean=a[:, 1:].mean(1).numpy())
print("done", f"{time.time()-t0:.0f}s", flush=True)
