# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
sg = np.sign
res = {k: {"linked": [], "null": []} for k in ("toward0_gate", "toward0_plus", "toward0_both", "abs_shrinks_gate", "even_vs_sign", "even_vs_minus_sign_of_act0")}
for l in z['layers'].tolist():
    R = z[f'L{l}_reads'][..., 0]
    lk = z[f'L{l}_linked']; valid = z[f'L{l}_readers'][..., 0] >= 0; gt = z[f'L{l}_gate'].astype(int)[..., None]
    s = z[f'L{l}_sign'].astype(int)[..., None]
    a0, dp, dm = R[..., 0], R[..., 1] - R[..., 0], R[..., 2] - R[..., 0]
    dg = np.where(gt > 0, dp, dm); even = (dp + dm) / 2
    for nm, m in (("linked", lk & valid), ("null", ~lk & valid)):
        M = np.broadcast_to(m[..., None], a0.shape) & (dp != 0) & (dm != 0)
        res["toward0_gate"][nm].append((sg(dg) == -sg(a0))[M].mean())
        res["toward0_plus"][nm].append((sg(dp) == -sg(a0))[M].mean())
        res["toward0_both"][nm].append(((sg(dp) == -sg(a0)) & (sg(dm) == -sg(a0)))[M].mean())
        res["abs_shrinks_gate"][nm].append((np.abs(a0 + dg) < np.abs(a0))[M].mean())
        res["even_vs_sign"][nm].append((sg(even) == s)[M].mean())
        res["even_vs_minus_sign_of_act0"][nm].append((sg(even) == -sg(a0))[M].mean())
for k, v in res.items():
    print(f"{k:28s} linked median {np.median(v['linked']):.3f}  null {np.median(v['null']):.3f}")
