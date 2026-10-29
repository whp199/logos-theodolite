# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
su, sgv = sg(d['u']).astype(np.int64), sg(d['g']).astype(np.int64)
T = np.ones_like(lk)
print('share of odd part zero', (odd == 0).mean(), 'even zero', (even == 0).mean())
print('|even| > |odd| share: linked %.3f null %.3f' % ((np.abs(even) > np.abs(odd))[lk].mean(), (np.abs(even) > np.abs(odd))[~lk].mean()))
for nm, pred in (('+sign(u)', su), ('+sign(g)*face', sgv * face), ('+sign(g u)', sign), ('+sign(g)', sgv), ('+face', face)):
    tab(None, [('all', T), ('face+', face > 0), ('face-', face < 0)], f'odd part (dp-dm)/2 vs {nm}', pred=pred, resp=odd)
rows = [(f'u {a:+d} g {b:+d} face {c:+d}', (su == a) & (sgv == b) & (face == c)) for a in (1, -1) for b in (1, -1) for c in (1, -1)]
tab(None, rows, 'odd part: share with sign = +sign(u), per cell', pred=su, resp=odd)
tab(None, [('all', T), ('face+', face > 0), ('face-', face < 0)], 'even part vs -face (toward zero)', pred=-face, resp=even)
tab(None, [('all', T), ('face+', face > 0), ('face-', face < 0)], 'even part vs sign (from rest)', pred=sign, resp=even)
