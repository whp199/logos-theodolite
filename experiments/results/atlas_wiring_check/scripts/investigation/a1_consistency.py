# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import sys, json, numpy as np
sys.path.insert(0, '/home/eric/bin/ai/theodolite2')
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
layers = z['layers'].tolist()
print('layers', len(layers), layers[:3], layers[-3:])
meta = json.loads(str(z['meta']))
print('meta keys', list(meta.keys()))
print({k: meta[k] for k in ('dose','writers','seed','batch','dose_unit','reads') if k in meta})
l0 = layers[0]
for k in z.files:
    if k.startswith(f'L{l0}_'):
        print(k, z[k].shape, z[k].dtype)
tot = dict(n=0, sign_mis=0, gate_mis=0, n_null=0)
for l in layers:
    g, u = z[f'L{l}_g'], z[f'L{l}_u']
    lk = z[f'L{l}_linked']; valid = z[f'L{l}_readers'][..., 0] >= 0
    s, gt = z[f'L{l}_sign'], z[f'L{l}_gate']
    m = lk & valid
    sp = np.where(g*u > 0, 1, -1); gp = np.where(g > 0, 1, -1)
    tot['n'] += m.sum(); tot['sign_mis'] += (s[m] != sp[m]).sum(); tot['gate_mis'] += (gt[m] != gp[m]).sum()
print('linked: index sign/gate vs pair_products', tot)
