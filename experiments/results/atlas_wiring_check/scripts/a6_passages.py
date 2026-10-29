# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
P = '/home/eric/bin/ai/theodolite2/atlas_data/index/passages.d/'
for f in ('pos', 'neg'):
    g = np.load(P + f'{f}_g.npy', mmap_mode='r'); u = np.load(P + f'{f}_u.npy', mmap_mode='r'); win = np.load(P + f'{f}_win.npy', mmap_mode='r')
    tot = [0, 0, 0]
    for l in range(0, g.shape[0], 1):
        ok = win[l] >= 0
        gl, ul = g[l][ok].astype(np.float32), u[l][ok].astype(np.float32)
        tot[0] += ok.sum(); tot[1] += (gl > 0).sum(); tot[2] += ((ul > 0) == (f == 'pos')).sum()
    print(f'31B stored {f} events: g > 0 at {tot[1]/tot[0]:.3f}; u with the face sign at {tot[2]/tot[0]:.3f}  (n={tot[0]})')
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz')
be = np.concatenate([z[f'L{l}_op'][..., 1].ravel() for l in z['layers']]); al = np.concatenate([z[f'L{l}_op'][..., :, 0].reshape(-1, 2) for l in z['layers']])
print('31B check op: beta > 0 at', np.nanmean(be > 0) if np.isfinite(be).any() else None, '; alpha>0 on + face', np.nanmean(al[:, 0] > 0), ', alpha<0 on - face', np.nanmean(al[:, 1] < 0))
