# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
a0 = d['a0']
ext = np.where(a0 >= 0, d['spos'], np.abs(d['sneg']))   # the reader's extreme on the face it sits on (stats index)
rel = np.abs(a0) / np.maximum(ext, 1e-12)
qs = np.quantile(rel, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
print('|a0| / face extreme quantiles (10,25,50,75,90,99%):', np.round(qs, 4))
print('share |a0| > 0.1 x extreme: %.4f; > 0.25: %.4f; > 0.5: %.4f' % ((rel > 0.1).mean(), (rel > 0.25).mean(), (rel > 0.5).mean()))
edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 10]
rows = [(f'rel in [{edges[i]}, {edges[i+1]})', (rel >= edges[i]) & (rel < edges[i + 1])) for i in range(len(edges) - 1)]
tab(None, rows, 'sign from rest, gate-face push, by |a0|/extreme')
su = sg(d['u']).astype(np.int64)
tab(None, rows, 'odd part vs +sign(u) (open gate predicts > 0.5), by |a0|/extreme', pred=su, resp=odd)
