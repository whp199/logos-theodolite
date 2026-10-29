# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
exec(open('/home/eric/bin/ai/theodolite2/experiments/results/atlas_wiring_check/scripts/investigation/common.py').read())
T = np.ones_like(lk)
tab(None, [('all', T)], 'sign from rest vs gate-face push (nonzero)')
tab(None, [('gate +', gate > 0), ('gate -', gate < 0)], 'by gate')
tab(None, [('sign +', sign > 0), ('sign -', sign < 0)], 'by index sign')
tab(None, [('face +', face > 0), ('face -', face < 0), ('face 0', face == 0)], 'by reader unpushed face (last)')
tab(None, [(f'band {b}', band == b) for b in range(4)], 'by writer layer band (0-14,15-29,30-44,45-58)')
rows = []
for g_ in (1, -1):
    for s_ in (1, -1):
        for f_ in (1, -1):
            rows.append((f'gate {g_:+d} sign {s_:+d} face {f_:+d}', (gate == g_) & (sign == s_) & (face == f_)))
tab(None, rows, 'gate x sign x face')
rows = [(f'sign {s_:+d} face {f_:+d}', (sign == s_) & (face == f_)) for s_ in (1, -1) for f_ in (1, -1)]
tab(None, rows, 'sign x face')
rows = [(f'gate {g_:+d} face {f_:+d}', (gate == g_) & (face == f_)) for g_ in (1, -1) for f_ in (1, -1)]
tab(None, rows, 'gate x face')
tab(None, [('all', T)], 'sign from rest vs +d push', resp=dp)
tab(None, [('gate +', gate > 0), ('gate -', gate < 0)], '+d push by gate', resp=dp)
tab(None, [('gate +', gate > 0), ('gate -', gate < 0)], 'other-face push by gate', resp=do)
