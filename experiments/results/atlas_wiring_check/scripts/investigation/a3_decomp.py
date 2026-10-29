# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np, json
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz')
A = np.load('/home/eric/bin/ai/theodolite2/atlas_data/index/act_scale.npz')
pos, neg = A['pos'], A['neg']
layers = z['layers'].tolist()
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
acc = {}
def add(key, m):
    a = acc.setdefault(key, [0, 0]); a[0] += int(m.sum()); a[1] += int(m.size)
bins = [0, 0.01, 0.03, 0.1, 0.3, 10]
for l in layers:
    R = z[f'L{l}_reads'][..., 0]  # last word-piece [W,R,Q,3]
    rd = z[f'L{l}_readers']; valid = rd[..., 0] >= 0
    lk = z[f'L{l}_linked']
    s = z[f'L{l}_sign'].astype(int)[:, :, None]; gt = z[f'L{l}_gate'].astype(int)[:, :, None]
    a0, ap, am = R[..., 0], R[..., 1], R[..., 2]
    dp, dm = ap - a0, am - a0
    dg = np.where(gt > 0, dp, dm)
    odd_g = gt * (dp - dm) / 2      # odd part, oriented to the gate face
    even = (dp + dm) / 2
    j, b = np.clip(rd[..., 0], 0, None), np.clip(rd[..., 1], 0, None)
    sc = np.maximum(pos[j, b], -neg[j, b])[:, :, None]
    rel = np.abs(a0) / np.maximum(sc, 1e-30)
    for name, grp in (('linked', lk & valid), ('null', ~lk & valid)):
        m = np.broadcast_to(grp[:, :, None], a0.shape)
        S = np.broadcast_to(s, a0.shape)
        add((name, 'sign_gate'), (sg(dg) == S)[m])
        add((name, 'sign_plus'), (sg(dp) == S)[m])
        add((name, 'odd_gate==sign'), (sg(odd_g) == S)[m])
        add((name, 'even==sign'), (sg(even) == S)[m])
        add((name, '|odd|>|even|'), (np.abs(odd_g) > np.abs(even))[m])
        add((name, 'act0>0'), (a0 > 0)[m])
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (rel >= lo) & (rel < hi)
            add((name, f'sign_gate | rel in [{lo},{hi})'), (sg(dg) == S)[mm])
            add((name, f'share rel in [{lo},{hi})'), np.array([mm.sum()]) )  # placeholder
for k, (a, n) in acc.items():
    if 'share rel' in k[1]:
        continue
    print(k, f'{a/n:.3f}' if n else None, n)
