# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: the mean site (mean over word-pieces past the start mark) at the 31B's measured per-position op."""
import json, math
import numpy as np

W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b"
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
qs = [np.load(f"{W}/z1_out/q{q}.npz") for q in range(4)]
c = math.sqrt(2 / math.pi)
gelu = lambda x: 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
def dgelu(x):
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
# per question, per layer: mean over positions of alpha_t and beta_t (the first-order response to a push held equal at
# every word-piece, the direct path's own scale per position left out)
AL = [dgelu(q["g_all"][:, 1:].astype(np.float32)) * q["u_all"][:, 1:].astype(np.float32) for q in qs]
BE = [gelu(q["g_all"][:, 1:].astype(np.float32)) for q in qs]
alm = np.stack([a.mean(1) for a in AL]); bem = np.stack([b.mean(1) for b in BE])  # [Q, L, N]
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
per = {}
for l in [int(x) for x in z["layers"]]:
    reads = z[f"L{l}_reads"][..., 1]  # mean site [W,R,Q,3]
    rd = z[f"L{l}_readers"]; lk = z[f"L{l}_linked"]; s = z[f"L{l}_sign"].astype(int); ga = z[f"L{l}_gate"].astype(int)
    g = z[f"L{l}_g"]; u = z[f"L{l}_u"]
    valid = rd[..., 0] >= 0
    j = np.where(valid, rd[..., 0], 0); b = np.where(valid, rd[..., 1], 0)
    al = np.moveaxis(alm[:, j, b], 0, -1); be = np.moveaxis(bem[:, j, b], 0, -1)
    k = sg(al * g[..., None] + be * u[..., None])
    dp = reads[..., 1] - reads[..., 0]; dm = reads[..., 2] - reads[..., 0]
    dgf = np.where(ga[..., None] > 0, dp, dm)
    for grp, m in (("linked", lk & valid), ("null", ~lk & valid)):
        M = np.broadcast_to(m[..., None], dp.shape) & (dgf != 0)
        per.setdefault(f"sign_gate_mean:{grp}", []).append(float((sg(dgf) == s[..., None])[M].mean()))
        per.setdefault(f"use_gate_true_mean:{grp}", []).append(float((sg(dgf) == ga[..., None] * k)[M].mean()))
print(json.dumps({k: {"layer_median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 3)} for k, v in per.items()}, indent=1))
