# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: the push itself on the 31B (CPU, packed W4A16): for a few layers, 6 of the check's writers each, pushed
along their write direction by +-0.25 and +-1.0 of their ceiling (the check's own lengths at 0.25), on the first
census question; every layer's gate and up values and activation read at the last word-piece. Answers: does a +d push
move each reader's gate and up values with the sign of the index's g and u; how large is the change against |g0|; do
the reads reproduce the stored check's."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["THEODOLITE_DEVICE"] = "cpu"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(22)
OUT = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/z4_out"
os.makedirs(OUT, exist_ok=True)
from backend.runtime_factory import make_runtime
from backend.causal_check import language_model
from backend import census as cz
from backend import atlas_ops as ao

z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
rt = make_runtime(load_jlens=False).load()
lm = language_model(rt.hf_model)
w = ao.Weights(lm, rt)
L = len(lm.layers)
ids = rt.tokenize_prompt(rt.compose_prompt(cz.PROMPTS["who"], chat=True))
T = len(ids)
DOSES = (0.25, -0.25, 1.0, -1.0)
NW = 6
for l in (int(a) for a in sys.argv[1:]):
    if os.path.exists(f"{OUT}/L{l}.npz"):
        continue
    t0 = time.time()
    writers = [int(a) for a in z[f"L{l}_writers"][:NW]]
    ceil_npz = z[f"L{l}_ceilings"][:NW]
    ceil_now = w.write_ceiling(l, writers).cpu().numpy()
    dirs = w.write_dirs(l, writers).float()
    B = 1 + len(DOSES) * NW
    iv = ao.Interventions(rows=B)
    iv.adds[l] = torch.zeros((B, w.d_model), dtype=torch.float32)
    for k in range(NW):
        for di, d in enumerate(DOSES):
            iv.adds[l][1 + k * len(DOSES) + di] = float(d) * float(ceil_npz[k]) * dirs[k]
    x = torch.tensor([ids] * B, dtype=torch.long)
    keep = ao._Keep()
    keep.mask = ao.keep_mask(x, None, rt.tokenizer.bos_token_id, True)
    G, U, A = {}, {}, {}
    hooks = []
    for j in range(l + 1, L):
        mlp = lm.layers[j].mlp
        hooks.append(mlp.gate_proj.register_forward_hook(lambda m, a, o, j=j: G.__setitem__(j, o[:, -1].float().cpu())))
        hooks.append(mlp.up_proj.register_forward_hook(lambda m, a, o, j=j: U.__setitem__(j, o[:, -1].float().cpu())))
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, a, j=j: A.__setitem__(j, a[0][:, -1].float().cpu()), prepend=True))
    try:
        with ao.installed(lm, iv, keep):
            lm(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    js = list(range(l + 1, L))
    np.savez(f"{OUT}/L{l}.npz", writers=np.array(writers), doses=np.array(DOSES), ceil_npz=ceil_npz, ceil_now=ceil_now,
             js=np.array(js), g=np.stack([G[j].numpy() for j in js]), u=np.stack([U[j].numpy() for j in js]),
             a=np.stack([A[j].numpy() for j in js]))
    print(f"L{l} forward B={B} {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
