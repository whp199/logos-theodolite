# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
import math
g, u, a0 = d['g'].astype(np.float64), d['u'].astype(np.float64), d['a0'].astype(np.float64)
fp = face > 0
al = np.where(d['a0'] > 0, d['alp'], d['alm']).astype(np.float64)
be = np.where(d['a0'] > 0, d['bep'], d['bem']).astype(np.float64)
has = np.isfinite(al) & np.isfinite(be)
kop = sg(al * g + be * u).astype(np.int64)
T = np.ones_like(lk)
def both(pred_odd, label, mask=None):
    m = T if mask is None else mask
    tab(None, [('odd part (+d minus -d)', m)], label + ' : odd part', pred=pred_odd, resp=odd)
    tab(None, [('gate-face push', m)], label + ' : gate-face push', pred=gate * pred_odd, resp=dg)
both(kop, 'in use, op as stored (strongest events)', has)
print('op as stored: share beta>0 %.3f; share sign(alpha)==face %.3f' % ((be[has] > 0).mean(), (sg(al[has]) == face[has]).mean()))
both(sg(al * g - np.abs(be) * u).astype(np.int64), 'in use, beta forced < 0 (shut gate), alpha as stored', has)
def gelu(x): return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def dgelu(x):
    c = math.sqrt(2 / math.pi); t = math.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
for g0 in (-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 2):
    r, b0 = dgelu(g0) / gelu(g0), gelu(g0)
    k = sg(a0 * r * g + b0 * u).astype(np.int64)   # alpha = gelu'(g0) u0 with u0 = a0 / gelu(g0)
    ok = odd != 0
    A = sg(odd) == k
    Ag = sg(dg) == gate * k
    nzg = dg != 0
    print(f'one g0 = {g0:+.1f} for every reader (u0 = a0/gelu(g0)): odd part linked {A[ok & lk].mean():.3f} null {A[ok & ~lk].mean():.3f}; '
          f'gate-face push linked {Ag[nzg & lk].mean():.3f} null {Ag[nzg & ~lk].mean():.3f}')
