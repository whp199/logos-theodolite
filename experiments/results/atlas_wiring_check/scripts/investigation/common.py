# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
d = dict(np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/flat.npz'))
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)
lk = d['linked']
dp, dm = d['ap'] - d['a0'], d['am'] - d['a0']
gate = d['gate'].astype(np.int64); sign = d['sign'].astype(np.int64)
dg = np.where(gate > 0, dp, dm)     # push on the gate face
do = np.where(gate > 0, dm, dp)     # push on the other face
odd = (dp - dm) / 2; even = (dp + dm) / 2
face = sg(d['a0'])
nz = dg != 0
band = np.digitize(d['L'], [15, 30, 45])
gap = d['j'] - d['L']
def tab(mask_fn, rows, label, pred=None, resp=None):
    pred = sign if pred is None else pred
    resp = dg if resp is None else resp
    ok = resp != 0
    A = sg(resp) == pred
    print(f'--- {label}')
    print(f'{"cell":38s} {"linked":>8s} {"n":>9s} {"null":>8s} {"n":>9s}')
    for name, m in rows:
        a = m & ok & lk; b = m & ok & ~lk
        print(f'{name:38s} {A[a].mean() if a.any() else np.nan:8.3f} {a.sum():9d} {A[b].mean() if b.any() else np.nan:8.3f} {b.sum():9d}')
