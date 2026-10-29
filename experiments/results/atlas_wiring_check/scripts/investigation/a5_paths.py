# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""31B: which of the reader's two paths carries the sign. Pairs where the writer reaches the reader mostly through its
up row (|u| > 4|g|) show sign(beta) = sign(gelu(g0)) at the reader's real operating point; pairs mostly through its
gate row (|g| > 4|u|) show sign(alpha) = sign(gelu'(g0) u0). Stratified by how active the reader sits unpushed."""
import numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz')
A = np.load('/home/eric/bin/ai/theodolite2/atlas_data/index/act_scale.npz')
pos, neg = A['pos'], A['neg']
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
acc = {}
def add(key, m):
    a = acc.setdefault(key, [0, 0]); a[0] += int(m.sum()); a[1] += int(m.size)
bins = [0, 0.01, 0.1, 10]
for l in z['layers'].tolist():
    R = z[f'L{l}_reads'][..., 0]
    rd = z[f'L{l}_readers']; valid = rd[..., 0] >= 0; lk = z[f'L{l}_linked']
    g = z[f'L{l}_g'][:, :, None]; u = z[f'L{l}_u'][:, :, None]
    a0, dp, dm = R[..., 0], R[..., 1] - R[..., 0], R[..., 2] - R[..., 0]
    odd = (dp - dm) / 2  # the response to +d, odd part
    j, b = np.clip(rd[..., 0], 0, None), np.clip(rd[..., 1], 0, None)
    rel = (np.abs(a0) / np.maximum(np.maximum(pos[j, b], -neg[j, b])[:, :, None], 1e-30))
    upd = np.broadcast_to(np.abs(u) > 4 * np.abs(g), a0.shape)
    gtd = np.broadcast_to(np.abs(g) > 4 * np.abs(u), a0.shape)
    for name, grp in (('linked', lk & valid), ('null', ~lk & valid)):
        m = np.broadcast_to(grp[:, :, None], a0.shape)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (rel >= lo) & (rel < hi)
            add((name, f'up-path pairs, rel [{lo},{hi}): sign(odd)=sign(u)  [=P(gelu(g0)>0)]'), (sg(odd) == np.broadcast_to(sg(u), a0.shape))[mm & upd])
            add((name, f'gate-path pairs, rel [{lo},{hi}): sign(odd)=sign(g)  [=P(alpha>0)]'), (sg(odd) == np.broadcast_to(sg(g), a0.shape))[mm & gtd])
            add((name, f'gate-path pairs, rel [{lo},{hi}): sign(odd)=sign(g)*sign(act0)  [=P(sign alpha=sign act0)]'),
                (sg(odd) == np.broadcast_to(sg(g), a0.shape) * sg(a0))[mm & gtd])
for k, (a, n) in sorted(acc.items()):
    print(f'{k[0]:6s} {k[1]:95s} {a/n if n else float("nan"):.3f}  n={n}')
