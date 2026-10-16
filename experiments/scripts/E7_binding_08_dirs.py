#!/usr/bin/env python3
"""E7_binding_08: cross-fitted per-layer gender directions (raw coordinates) for directional interchange patching.
For half h in {0: passages 0-11, 1: passages 12-23} compute, from the OTHER half's variant-averaged residual means:
  A_self: ref = ds (G_self effect), non = du (G_user effect) orthogonalised against ref
  A_user: ref = du, non = ds orthogonalised against ref
Saved as CACHE/dirs_xfit{h}.npz with arrays [60, D] (unit vectors) + random control directions."""
import numpy as np
CACHE = "./experiments/results/E7_binding_cache"
RES = "./experiments/results"
z = np.load(f"{RES}/E7_binding_factorial_chunk0.npz"); TYPES = list(z["types"]); CELLN = list(z["cell_names"])
R = np.concatenate([np.load(f"{CACHE}/resid_means_chunk{c}.npy", mmap_mode="r") for c in range(3)], 0)  # [24,10,7,60,D]
GSW = [CELLN.index(x) for x in ("WW", "WM", "WU")]; GSM = [CELLN.index(x) for x in ("MW", "MM", "MU")]
GUW = [CELLN.index(x) for x in ("WW", "MW", "UW")]; GUM = [CELLN.index(x) for x in ("WM", "MM", "UM")]
rng = np.random.default_rng(0)
def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
for h in (0, 1):
    other = list(range(12, 24)) if h == 0 else list(range(0, 12))
    out = {}
    for t in ("A_self", "A_user"):
        ti = TYPES.index(t)
        X = np.asarray(R[other][:, :, ti], dtype=np.float32)  # [12, 10, 60, D]
        ds = ((X[:, GSW].mean(1) - X[:, GSM].mean(1)) / 2).mean(0)  # [60, D]
        du = ((X[:, GUW].mean(1) - X[:, GUM].mean(1)) / 2).mean(0)
        ref, non = (ds, du) if t == "A_self" else (du, ds)
        ur = unit(ref)
        un = unit(non - (non * ur).sum(-1, keepdims=True) * ur)
        out[f"{t}_ref"] = ur.astype(np.float32); out[f"{t}_non"] = un.astype(np.float32)
        out[f"{t}_ref_norm"] = np.linalg.norm(ref, axis=-1); out[f"{t}_non_norm"] = np.linalg.norm(non, axis=-1)
        out[f"{t}_cos_ref_non"] = (unit(ref) * unit(non)).sum(-1)
        r = rng.standard_normal(ref.shape).astype(np.float32)
        out[f"{t}_rand"] = unit(r)
    np.savez(f"{CACHE}/dirs_xfit{h}.npz", **out)
    print(h, "cos(ref,non) A_self by band", [round(float(out["A_self_cos_ref_non"][a:b].mean()), 2) for a, b in ((10,20),(20,30),(30,40),(40,50),(50,59))],
          "norm ratio non/ref", [round(float((out["A_self_non_norm"][a:b] / out["A_self_ref_norm"][a:b]).mean()), 2) for a, b in ((10,20),(20,30),(30,40),(40,50),(50,59))])
