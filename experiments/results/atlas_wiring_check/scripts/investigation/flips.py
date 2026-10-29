# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
g, u = d['g'].astype(np.float64), d['u'].astype(np.float64)
su = sg(u).astype(np.int64)
al = np.where(d['a0'] > 0, d['alp'], d['alm']).astype(np.float64)
be = np.where(d['a0'] > 0, d['bep'], d['bem']).astype(np.float64)
has = np.isfinite(al) & np.isfinite(be)
kop = sg(al * g + be * u).astype(np.int64)
kshut = sg(al * g - np.abs(be) * u).astype(np.int64)
fz = np.where(face == 0, -1, face)
cands = {
    'F0 sign from rest, as built': sign,
    'F1 -sign (every link)': -sign,
    'F2 -sign where gate is -, sign where +': np.where(gate < 0, -sign, sign),
    'F3 -sign where gate is +, sign where -': np.where(gate > 0, -sign, sign),
    'F4 sign x reader face': sign * fz,
    'F5 -sign x reader face': -sign * fz,
    'F6 -sign where sign != face, sign where ==': np.where(sign != fz, -sign, sign),
    'F7 in use, op as built (gate*kappa)': gate * kop,
    'F8 -(in use) ': -gate * kop,
    'F9 in use, beta < 0 (shut gate)': gate * kshut,
}
nzg = dg != 0
print(f'{"gate-face push, nonzero changes":48s} {"linked":>7s} {"null":>7s} {"gap":>6s}')
for nm, p in cands.items():
    A = sg(dg) == p
    m = nzg & (has if 'in use' in nm or 'beta' in nm else True)
    print(f'{nm:48s} {A[m & lk].mean():7.3f} {A[m & ~lk].mean():7.3f} {A[m & lk].mean() - A[m & ~lk].mean():+6.3f}')
# per layer medians of the main ones
print()
for nm in ('F0 sign from rest, as built', 'F1 -sign (every link)', 'F9 in use, beta < 0 (shut gate)', 'F7 in use, op as built (gate*kappa)'):
    A = sg(dg) == cands[nm]
    pl = np.array([[A[nzg & has & (lk == side) & (d['L'] == l)].mean() for side in (True, False)] for l in range(59)])
    gapl = pl[:, 0] - pl[:, 1]
    print(f'{nm:40s} per-layer median linked {np.median(pl[:,0]):.3f} null {np.median(pl[:,1]):.3f}; layers linked > null: {(gapl > 0).sum()}/59; '
          f'min/max linked {pl[:,0].min():.3f}/{pl[:,0].max():.3f} (layers {pl[:,0].argmin()}/{pl[:,0].argmax()})')
A = sg(dg) == sign
pl = np.array([[A[nzg & (lk == side) & (d['L'] == l)].mean() for side in (True, False)] for l in range(59)])
print('per-layer sign_gate (nonzero), linked/null:')
print(' '.join(f'{l}:{pl[l,0]:.2f}/{pl[l,1]:.2f}' for l in range(59)))
