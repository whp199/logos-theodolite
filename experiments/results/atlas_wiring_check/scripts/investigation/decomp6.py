# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
g, u = d['g'].astype(np.float64), d['u'].astype(np.float64)
su, sgv = sg(u).astype(np.int64), sg(g).astype(np.int64)
lr = np.log10(np.abs(g) / np.maximum(np.abs(u), 1e-30))
edges = [-9, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 9]
rows = [(f'log10|g/u| in [{edges[i]}, {edges[i+1]})', (lr >= edges[i]) & (lr < edges[i + 1])) for i in range(len(edges) - 1)]
tab(None, rows, 'odd part vs -sign(u) (the up path through a shut gate, gelu(g0) < 0)', pred=-su, resp=odd)
tab(None, rows, 'odd part vs +face*sign(g) (the gate path below gelu\'s minimum, gelu\'(g0) < 0)', pred=face * sgv, resp=odd)
tab(None, rows, 'gate-face push vs the sign from rest', pred=sign, resp=dg)
# both paths agree when face != sign
agree_cells = face * sign < 0
tab(None, [('face != sign (both paths: -sign(u))', agree_cells), ('face == sign (the paths oppose)', face * sign > 0)],
    'odd part vs -sign(u), split where the two shut-gate paths agree or oppose', pred=-su, resp=odd)
tab(None, [(f'band {b}', band == b) for b in range(4)], 'odd part vs -sign(u) by writer band', pred=-su, resp=odd)
tab(None, [(f'gap {a}-{b}', (gap >= a) & (gap <= b)) for a, b in ((1, 1), (2, 3), (4, 8), (9, 20), (21, 60))], 'odd part vs -sign(u) by layer gap j - l', pred=-su, resp=odd)
