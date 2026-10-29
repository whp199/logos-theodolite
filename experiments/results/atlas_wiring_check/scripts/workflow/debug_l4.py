# Kept from the session of the night of 28-29 October 2026 (its scratchpad): the wiring workflow's own checks of the
# new summary; only the paths changed (see ../../summary.json).
import os, sys, math
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
import backend.atlas_ops as ao
W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/z1_out"
qs = [np.load(f"{W}/q{q}.npz") for q in range(4)]
print([len(q["ids"]) for q in qs], [str(q["name"]) for q in qs])
G = np.stack([q["g_last"] for q in qs]); U = np.stack([q["u_last"] for q in qs])
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz")
l = 4
reads = z[f"L{l}_reads"][..., 0]; rd = z[f"L{l}_readers"]; lk = z[f"L{l}_linked"]; ga = z[f"L{l}_gate"].astype(int)
g = z[f"L{l}_g"]; u = z[f"L{l}_u"]
valid = rd[..., 0] >= 0
j = np.where(valid, rd[..., 0], 0); b = np.where(valid, rd[..., 1], 0)
g0 = np.moveaxis(G[:, j, b], 0, -1); u0 = np.moveaxis(U[:, j, b], 0, -1)
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
c = math.sqrt(2 / math.pi)
gelu = lambda x: 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
def dgelu(x):
    t = np.tanh(c * (x + 0.044715 * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
dp = reads[..., 1] - reads[..., 0]; dm = reads[..., 2] - reads[..., 0]
dgf = np.where(ga[..., None] > 0, dp, dm)
M = np.broadcast_to((lk & valid)[..., None], dp.shape); nz = M & (dgf != 0)
k32 = sg(dgelu(g0) * u0 * g[..., None] + gelu(g0) * u[..., None])
a64, b64 = ao.site_coefficients(g0, u0)
k64 = sg(a64 * g[..., None] + b64 * u[..., None])
print("float32 numpy:", (sg(dgf) == ga[..., None] * k32)[nz].mean(), " float64:", (sg(dgf) == ga[..., None] * k64)[nz].mean())
print("kappa differ share:", (k32 != k64)[nz].mean())
print("g0 dtype", g0.dtype, "g", g.dtype)
x = (dgelu(g0) * u0 * g[..., None])[nz]; y = (gelu(g0) * u[..., None])[nz]
print("sample", x[:5], y[:5], (dgelu(g0)*u0*g[...,None] + gelu(g0)*u[...,None])[nz][:5], (a64 * g[..., None] + b64 * u[..., None])[nz][:5])
