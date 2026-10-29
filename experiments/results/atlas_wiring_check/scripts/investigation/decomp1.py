# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
import numpy as np
d = dict(np.load('/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/flat.npz'))
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)
lk = d['linked']
sgu, sgg = sg(d['g'] * d['u']), sg(d['g'])
print('stored sign == sign(g u) from pair_products: linked %.4f null %.4f' % ((d['sign'] == sgu)[lk].mean(), (d['sign'] == sgu)[~lk].mean()))
print('stored gate == sign(g): linked %.4f null %.4f' % ((d['gate'] == sgg)[lk].mean(), (d['gate'] == sgg)[~lk].mean()))
dp, dm = d['ap'] - d['a0'], d['am'] - d['a0']
dg = np.where(d['gate'] > 0, dp, dm)
nz = dg != 0
print('zero share', 1 - nz.mean())
A = sg(dg) == d['sign']
def share(m):
    m = m & nz
    return A[m].mean(), int(m.sum())
print('overall rest_gate (nonzero): linked %.3f null %.3f' % (share(lk)[0], share(~lk)[0]))
# medians over layers as the task reports
for nm, m in (('linked', lk), ('null', ~lk)):
    per = [A[m & nz & (d['L'] == l)].mean() for l in range(59)]
    print(nm, 'median over layers %.3f' % np.median(per))
