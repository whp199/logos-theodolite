# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: does the first-order size at the measured op rank the change better than |P| f^2? (last site)"""
import sys, math, json
sys.path.insert(0, "/home/eric/bin/ai/theodolite2")
import numpy as np
from backend.atlas_build import spearman
W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b"
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
qs = [np.load(f"{W}/z1_out/q{q}.npz") for q in range(4)]
G = np.stack([q["g_last"] for q in qs]); U = np.stack([q["u_last"] for q in qs])
c = math.sqrt(2 / math.pi)
gelu = lambda x: 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
def dgelu(x):
    t = np.tanh(c * (x + 0.044715 * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
per = {}
for l in [int(x) for x in z["layers"]]:
    reads = z[f"L{l}_reads"][..., 0]; rd = z[f"L{l}_readers"]; lk = z[f"L{l}_linked"]; ga = z[f"L{l}_gate"].astype(int)
    g = z[f"L{l}_g"]; u = z[f"L{l}_u"]; st = z[f"L{l}_strength"]; ln = z[f"L{l}_lengths"]
    valid = rd[..., 0] >= 0
    j = np.where(valid, rd[..., 0], 0); b = np.where(valid, rd[..., 1], 0)
    g0 = np.moveaxis(G[:, j, b], 0, -1); u0 = np.moveaxis(U[:, j, b], 0, -1)
    f = np.sqrt(st / np.maximum(np.abs(g * u), 1e-30))
    first = (f[..., None] * np.abs(dgelu(g0) * u0 * g[..., None] + gelu(g0) * u[..., None])).mean(-1) * ln[:, None]
    dp = reads[..., 1] - reads[..., 0]; dm = reads[..., 2] - reads[..., 0]
    size = np.abs(np.where(ga[..., None] > 0, dp, dm)).mean(-1)
    for grp, m in (("linked", lk & valid), ("null", ~lk & valid)):
        per.setdefault(f"rho_strength:{grp}", []).append(spearman(size[m], st[m]))
        per.setdefault(f"rho_first_order_at_op:{grp}", []).append(spearman(size[m], first[m]))
print(json.dumps({k: round(float(np.median([x for x in v if x is not None])), 3) for k, v in per.items()}, indent=1))
