#!/usr/bin/env python3
"""morning.sh against what the hashed tonight.sh writes (R = the worktree), and the calibration figures the
pass criterion quotes (calib_expect.json, layers 20, 30, 40, 45, 50, 57 in order). Reads only."""
import difflib, json
from pathlib import Path
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
src = (T / 'prereg/tonight.sh').read_text()
body = src.split('cat > "$D/morning.sh" <<EOF\n', 1)[1].split('\nEOF\n', 1)[0] + '\n'
vals = {'$THEODOLITE_ATLAS_DIR': '/home/eric/bin/ai/theodolite2/atlas_data', '$PY': '/home/eric/bin/ai/theodolite2/.venv/bin/python -I',
        '$P': str(T) + '/prereg', '$D': str(T), '$R': '/home/eric/bin/ai/theodolite2/.claude/worktrees/census-break-rel'}
for k in sorted(vals, key=len, reverse=True):
    body = body.replace(k, vals[k])
body = body.replace('\\\\\n', '\\\n')
got = (T / 'morning.sh').read_text()
print('morning.sh == tonight.sh heredoc expanded:', body == got)
if body != got:
    print(''.join(difflib.unified_diff(body.splitlines(1), got.splitlines(1))))
c = json.loads((T / 'prereg/calib_expect.json').read_text())['calib_rel cut 48']['per']
for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'rep_res', 'cov_lp'):
    print(k, 'L45', round(c[k][3][0], 4), 'L57', round(c[k][5][0], 4))
