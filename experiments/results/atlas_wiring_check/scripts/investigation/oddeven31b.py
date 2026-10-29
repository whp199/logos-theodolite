# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
layers = z['layers'].tolist()
sg = np.sign
acc = {k: {"linked": [], "null": []} for k in ["gate_face", "plus", "even_vs_sign", "gateodd_vs_sign", "odd_vs_kappa", "gateodd_vs_minus_sign", "toward0_gate", "frac_odd_dom", "face_pos", "kappa_vs_sign_gate", "kappa_on_face_vs_rest", "even_vs_sign_when_odd_small"]}
per_layer = []
for l in layers:
    R = z[f'L{l}_reads'][..., 0]  # last site [W,R,Q,3]
    lk = z[f'L{l}_linked']; valid = z[f'L{l}_readers'][..., 0] >= 0
    g, u = z[f'L{l}_g'], z[f'L{l}_u']; s = z[f'L{l}_sign'].astype(int); gt = z[f'L{l}_gate'].astype(int)
    op = z[f'L{l}_op']
    a0, dp, dm = R[..., 0], R[..., 1] - R[..., 0], R[..., 2] - R[..., 0]
    odd, even = (dp - dm) / 2, (dp + dm) / 2
    face = np.where(a0 > 0, 0, 1)
    al = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 0], face.shape + (2,)), face[..., None], -1)[..., 0]
    be = np.take_along_axis(np.broadcast_to(op[:, :, None, :, 1], face.shape + (2,)), face[..., None], -1)[..., 0]
    kap = sg(al * g[..., None] + be * u[..., None])
    G = gt[..., None]; S = s[..., None]
    dg = np.where(G > 0, dp, dm)
    row = {"l": l}
    for name, m in (("linked", lk & valid), ("null", ~lk & valid)):
        mm = np.broadcast_to(m[..., None], dp.shape)
        nz = mm & (dp != 0) & (dm != 0)
        def sh(x): return float(x[nz].mean())
        acc["gate_face"][name].append(sh(sg(dg) == S))
        acc["plus"][name].append(sh(sg(dp) == S))
        acc["even_vs_sign"][name].append(sh(sg(even) == S))
        acc["gateodd_vs_sign"][name].append(sh(sg(G * odd) == S))
        hk = nz & np.isfinite(kap)
        acc["odd_vs_kappa"][name].append(float((sg(odd) == kap)[hk].mean()))
        acc["frac_odd_dom"][name].append(sh(np.abs(odd) > np.abs(even)))
        acc["face_pos"][name].append(sh(a0 > 0))
        acc["kappa_vs_sign_gate"][name].append(float((G * kap == S)[hk].mean()))
        row[name] = (sh(sg(dg) == S), sh(sg(even) == S), sh(sg(G * odd) == S), float((sg(odd) == kap)[hk].mean()), sh(np.abs(odd) > np.abs(even)))
    per_layer.append(row)
for k, v in acc.items():
    print(f"{k:28s} linked median {np.median(v['linked']):.3f}  null median {np.median(v['null']):.3f}")
print("per layer (gate_face, even_vs_sign, gateodd_vs_sign, odd_vs_kappa, odd_dominates):")
for r in per_layer[::6]:
    print(r["l"], "linked", tuple(round(x, 2) for x in r["linked"]), "null", tuple(round(x, 2) for x in r["null"]))
