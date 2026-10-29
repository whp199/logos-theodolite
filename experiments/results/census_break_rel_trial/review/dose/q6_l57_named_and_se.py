"""L57 named neurons: where their top-5 words occur (question, dose) and one +-2 and one +-1 answer each; and
Hanley-McNeil standard errors for the AUCs reported. Writes q6_l57_named.txt."""
import json
import math

import common as C
from common import R, cz


def se(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


for a, n1, n2, what in ((0.7064, 1022, 1024, 'L45 pm1 mass_c'), (0.5708, 1022, 1024, 'L45 pm2 mass_c'),
                        (0.641, 705, 942, 'L45 pm1 intact units mass_c'), (0.5225, 507, 512, 'L57 pm1 mass_c'),
                        (0.5737, 507, 512, 'L57 pm2 mass_c'), (0.5995, 507, 512, 'L57 all mass_c')):
    print(f'SE {what}: AUC {a} +- {se(a, n1, n2):.4f}')
rows = C.rows('L57')
V = json.loads((C.TRIALS / 'L57_verdict.json').read_text())['readings']['48']
cnt = R.counts(rows, C.keep_all, 'own')
m = R.mass_from_counts(cnt, 'controls')
base = R.baselines(rows)
by = {(r['kind'], r['neuron'], r['prompt'], r['dose']): r for r in rows if r['kind'] != 'baseline' and C.keep_all(r)}
lines = []
for n in sorted(V['named'], key=lambda n: -m[('neuron', n)][0]):
    u = ('neuron', n)
    lines.append(f'neuron {n} mass {m[u][0]:.2f}')
    for z, w in m[u][1]:
        where = sorted(f'{k[2]}{k[3]:+g}' for k, r in by.items() if k[0] == 'neuron' and k[1] == n and w in R._words(r) - R._words(base[r['prompt']]))
        lines.append(f'   {w[:30]!r} z {z:.2f}: {" ".join(where)}')
    for d in (2.0, -2.0, 1.0, -1.0):
        r = by[('neuron', n, 'walk', d)]
        lines.append(f'   [walk {d:+g}] d4 {cz.distinct4(r["token_ids"]):.2f}: ' + r['text'].replace('\n', '\\n')[:160])
(C.HERE / 'q6_l57_named.txt').write_text('\n'.join(lines))
