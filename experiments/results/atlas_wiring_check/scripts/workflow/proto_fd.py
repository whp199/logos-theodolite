# Kept from the session of the night of 28-29 October 2026 (its scratchpad): the wiring workflow's own checks of the
# new summary; only the paths changed (see ../../summary.json).
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path[:0] = ["/home/eric/bin/ai/theodolite2", "/home/eric/bin/ai/theodolite2/scripts"]
import numpy as np, torch, torch.nn.functional as F
torch.set_grad_enabled(False)
import backend.atlas_ops as ao
from atlas_stand_in import stand_in_hf, stand_in_runtime, stand_in_tokenizer
from backend.causal_check import language_model
import check_atlas_ops as C

tok = stand_in_tokenizer(); hf = stand_in_hf(tok, torch.float32); lm = language_model(hf); w = ao.Weights(lm)
prompts = [C.ids_of(tok, "Who are you?", chat=True), C.ids_of(tok, "Describe the color orange.", chat=True)]
print([len(p) for p in prompts])
writers = [(1, 3), (2, 40), (1, 7), (3, 90), (2, 11)]
readers = [[(2, 5), (4, 17), (5, 100)], [(3, 1), (5, 2)], [(2, 5), (3, 64), (4, 0), (5, 127)], [(4, 9)], [(5, 77), (3, 3)]]
lengths = [0.5, 1.2, 0.3, 2.0, 0.8]
def alone(ids, l, a, d, rs):
    iv = ao.Interventions(rows=1, adds={l: d * w.write_dirs(l, [a])}) if d else None
    at = sorted({j for j, _ in rs})
    watch = [(lm.layers[j].mlp.down_proj, "pre") for j in at] + [(lm.layers[j].mlp.gate_proj, "post") for j in at] + [(lm.layers[j].mlp.up_proj, "post") for j in at]
    got = C.observe(lm, ids, iv, watch); n = len(at)
    A = {j: c[0][0] for j, c in zip(at, got[:n])}; G = {j: c[0][0] for j, c in zip(at, got[n:2*n])}; U = {j: c[0][0] for j, c in zip(at, got[2*n:])}
    return [(A[j][:, b], G[j][:, b], U[j][:, b]) for j, b in rs]
for frac in (0.1, 0.02, 0.005):
    small = [frac * x for x in lengths]
    reads, g0, u0 = ao.pushed_reads(lm, w, prompts, writers, readers, small, bos_id=tok.bos_token_id, batch=64)
    worst, agree, n_, ksite = 0.0, 0, 0, []
    for a, ((l, nn_), rs) in enumerate(zip(writers, readers)):
        g, u = ao.pair_products(w, l, [nn_], [rs])
        for q, ids in enumerate(prompts):
            p, m = alone(ids, l, nn_, small[a], rs), alone(ids, l, nn_, -small[a], rs)
            for r, (j, b) in enumerate(rs):
                gv, uv = float(g0[a, r, q, -1]), float(u0[a, r, q, -1])
                al0 = ao.gelu_tanh_grad(gv) * uv; be0 = float(F.gelu(torch.tensor(gv, dtype=torch.float64), approximate="tanh"))
                dg = (float(p[r][1][-1]) - float(m[r][1][-1])) / 2; du = (float(p[r][2][-1]) - float(m[r][2][-1])) / 2
                da = (float(reads[a, r, q, 1, 0]) - float(reads[a, r, q, 2, 0])) / 2
                pred = al0 * dg + be0 * du
                rel = abs(da - pred) / max(abs(da), 1e-12)
                worst = max(worst, rel)
                agree += (da > 0) == (pred > 0); n_ += 1
                ksite.append((da > 0) == (al0 * g[0, r] + be0 * u[0, r] > 0))
                if frac == 0.02 and q == 0: print(f"  {a} {r} da {da:.3e} pred {pred:.3e} rel {rel:.2e} g0 {gv:.3f}")
    print(frac, "worst rel", worst, "sign agree", agree, n_, "kappa from weights", sum(ksite), len(ksite))
