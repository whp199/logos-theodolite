# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
z = dg == 0
print('zero share of the gate-face change at the last word-piece: linked %.4f null %.4f' % (z[lk].mean(), z[~lk].mean()))
zm = np.where(gate > 0, d['mp'] - d['m0'], d['mm'] - d['m0']) == 0
print('zero share at the mean site: linked %.4f null %.4f' % (zm[lk].mean(), zm[~lk].mean()))
pl = [z[d['L'] == l].mean() for l in range(59)]
print('per layer (last):', ' '.join(f'{l}:{p:.3f}' for l, p in enumerate(pl)))
print('share with a0 == 0 exactly: %.4f' % (d['a0'] == 0).mean())
