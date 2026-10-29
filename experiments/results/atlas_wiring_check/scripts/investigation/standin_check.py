# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""On the CPU stand-in: are the wiring check's conventions (push direction, pair_products' g and u, reads = activations)
consistent, and how does the sign from rest fare by where the reader's gate sits (g0)?"""
import os, sys, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/home/eric/bin/ai/theodolite2'); sys.path.insert(0, '/home/eric/bin/ai/theodolite2/scripts')
import numpy as np, torch
import torch.nn.functional as F
from atlas_stand_in import stand_in_hf, stand_in_tokenizer
from backend import atlas_ops as ao
from backend.causal_check import language_model
tok = stand_in_tokenizer()
hf = stand_in_hf(tok, torch.float32)
lm = language_model(hf)
w = ao.Weights(lm, None)
N, L = w.d_ff, w.n_layers
texts = ["The committee rejected the proposal because it was rushed.", "def merge(a, b):\n    out = []", "Ingredients: 2 cups flour, 1 tsp salt, 3 eggs", "Alice called Bob because she wanted his car."]
prompts = [[tok.bos_token_id] + tok(t, add_special_tokens=False)['input_ids'] for t in texts]
rows = []
for l in (0, 1, 2, 3):
    writers = list(range(0, N, 8))
    readers = [[(j, b) for j in range(l + 1, L) for b in range(N)] for _ in writers]
    ceil = w.write_ceiling(l, writers).numpy()
    for dose in (0.02, 0.25):
        lengths = dose * ceil
        reads = ao.pushed_reads(lm, w, prompts, [(l, a) for a in writers], readers, lengths, bos_id=tok.bos_token_id, batch=33)
        g, u = ao.pair_products(w, l, writers, readers)
        # the readers' own gate and up values, unpushed and pushed, at the last word-piece (separate forwards)
        for q, ids in enumerate(prompts):
            x = torch.tensor([ids])
            cap = {}
            hooks = []
            for j in range(l + 1, L):
                hooks.append(lm.layers[j].mlp.gate_proj.register_forward_hook(lambda m, a, o, j=j: cap.__setitem__(('g', j), o[0, -1].detach().clone())))
                hooks.append(lm.layers[j].mlp.up_proj.register_forward_hook(lambda m, a, o, j=j: cap.__setitem__(('u', j), o[0, -1].detach().clone())))
            def run(delta):
                cap.clear()
                iv = ao.Interventions(rows=1, adds={l: delta[None]}) if delta is not None else None
                keep = ao._Keep(); keep.mask = ao.keep_mask(x, None, tok.bos_token_id, True)
                with ao.installed(lm, iv, keep):
                    lm(input_ids=x, use_cache=False)
                return {k: v.clone() for k, v in cap.items()}
            base = run(None)
            dirs = w.write_dirs(l, writers)
            for a, A in enumerate(writers):
                pushed = run(float(lengths[a]) * dirs[a])
                for r, (j, b) in enumerate(readers[a]):
                    g0, u0 = float(base[('g', j)][b]), float(base[('u', j)][b])
                    dgt, dut = float(pushed[('g', j)][b]) - g0, float(pushed[('u', j)][b]) - u0
                    a0, ap, am = reads[a, r, q, :, 0]
                    rows.append((l, j, dose, g0, u0, dgt, dut, float(g[a, r]), float(u[a, r]), a0, ap, am))
            for h in hooks: h.remove()
R = np.array(rows)
np.save('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/standin_rows.npy', R)
l, j, dose, g0, u0, dgt, dut, g, u, a0, ap, am = R.T
sg = np.sign
def gelu(x): return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
dgelu = lambda x: ao.gelu_tanh_grads(torch.tensor(x)).numpy()
print('check: unpushed read == gelu(g0) u0 (max abs diff)', np.abs(a0 - gelu(g0) * u0).max())
for ds in (0.02, 0.25):
    for nb, m in (('j = l+1', (j == l + 1) & (dose == ds)), ('j > l+1', (j > l + 1) & (dose == ds))):
        print(f'dose {ds} {nb}: sign(true dg) == sign(g) {np.mean(sg(dgt[m]) == sg(g[m])):.3f}; sign(true du) == sign(u) {np.mean(sg(dut[m]) == sg(u[m])):.3f}; '
              f'corr(dg, g) {np.corrcoef(dgt[m], g[m])[0,1]:.3f}')
    m = dose == ds
    dp = ap - a0
    fo = dgelu(g0) * u0 * dgt + gelu(g0) * dut   # first order with the true operating point and the true input changes
    nz = m & (dp != 0)
    print(f'dose {ds}: sign(measured +d change) == sign(first order at the true g0,u0) {np.mean(sg(dp[nz]) == sg(fo[nz])):.3f}')
    dgf = np.where(g > 0, ap - a0, am - a0)
    for nm, mm in (('g0 > 0 (gate open)', g0 > 0), ('-0.75 < g0 < 0', (g0 < 0) & (g0 > -0.75)), ('g0 < -0.75 (below gelu min)', g0 < -0.75)):
        k = nz & mm & (dgf != 0)
        print(f'   dose {ds} {nm:30s} n={k.sum():6d}: gate-face push vs sign from rest {np.mean(sg(dgf[k]) == sg(g[k] * u[k])):.3f}; '
              f'+d push vs sign from rest {np.mean(sg(dp[k]) == sg(g[k]*u[k])):.3f}; odd part vs -sign(u) {np.mean(sg((ap-am)[k]) == -sg(u[k])):.3f}')
