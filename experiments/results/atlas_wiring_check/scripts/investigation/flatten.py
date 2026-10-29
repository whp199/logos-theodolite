# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Flatten the 31B wiring check into one row per (writer, reader, question) at both read sites; save as npz."""
import sys, json, numpy as np
z = np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz', allow_pickle=False)
S = np.load('/home/eric/bin/ai/theodolite2/atlas_data/index/act_scale.npz', allow_pickle=False)
pos, neg = S['pos'], S['neg']
cols = {k: [] for k in ('L', 'j', 'b', 'w', 'linked', 'sign', 'gate', 'g', 'u', 'strength', 'q', 'a0', 'ap', 'am', 'm0', 'mp', 'mm',
                        'alp', 'bep', 'alm', 'bem', 'spos', 'sneg', 'length')}
for l in [int(x) for x in z['layers']]:
    p = f'L{l}_'
    rd = z[p + 'reads']  # W R Q 3 2
    W, R, Q = rd.shape[:3]
    readers = z[p + 'readers']
    valid = readers[..., 0] >= 0
    wi, ri = np.nonzero(valid)
    for q in range(Q):
        n = len(wi)
        cols['L'].append(np.full(n, l, np.int16)); cols['j'].append(readers[wi, ri, 0].astype(np.int16)); cols['b'].append(readers[wi, ri, 1])
        cols['w'].append(z[p + 'writers'][wi]); cols['linked'].append(z[p + 'linked'][wi, ri])
        for k in ('sign', 'gate', 'g', 'u', 'strength'):
            cols[k].append(z[p + k][wi, ri])
        cols['q'].append(np.full(n, q, np.int8))
        for k, (s, site) in {'a0': (0, 0), 'ap': (1, 0), 'am': (2, 0), 'm0': (0, 1), 'mp': (1, 1), 'mm': (2, 1)}.items():
            cols[k].append(rd[wi, ri, q, s, site])
        op = z[p + 'op'][wi, ri]  # n 2 2
        cols['alp'].append(op[:, 0, 0]); cols['bep'].append(op[:, 0, 1]); cols['alm'].append(op[:, 1, 0]); cols['bem'].append(op[:, 1, 1])
        jj, bb = readers[wi, ri, 0], readers[wi, ri, 1]
        cols['spos'].append(pos[jj, bb]); cols['sneg'].append(neg[jj, bb])
        cols['length'].append(z[p + 'lengths'][wi])
out = {k: np.concatenate(v) for k, v in cols.items()}
np.savez('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/flat.npz', **out)
print({k: (v.shape, v.dtype) for k, v in out.items()})
