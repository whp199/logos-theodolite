"""Q2: decoded answers of 10 named L45 neurons (spread over the named list by mass rank: ranks 0, 26, 52, ... of the
262 sorted by mass_c) and 10 controls (rng(0) draw), at -2, -1, +1, +2 for 'who' (half A) and 'bicycle' (half B),
with each answer's distinct-4, the counts of its top-5 words, and whether those words came at +-1 or +-2.
Writes q2_samples.txt."""
import json

import numpy as np

import common as C
from common import R, cz

layer = 'L45'
V = json.loads((C.TRIALS / 'L45_verdict.json').read_text())['readings']['48']
rows = C.rows(layer)
cnt = R.counts(rows, C.keep_all, 'own')
m = R.mass_from_counts(cnt, 'controls')
named = sorted(V['named'], key=lambda n: -m[('neuron', n)][0])
pick_n = [named[i] for i in range(0, len(named), 26)][:10]
rng = np.random.default_rng(0)
pick_c = sorted(int(x) for x in rng.choice(1024, 10, replace=False))
by = {}
for r in rows:
    if r['kind'] != 'baseline' and C.keep_all(r):
        by[(r['kind'], r['neuron'], r['prompt'], r['dose'])] = r
base = R.baselines(rows)
lines = []
for kind, ids in (('neuron', pick_n), ('random', pick_c)):
    for n in ids:
        u = (kind, n)
        top = [(w, round(z, 2)) for z, w in m[u][1]]
        lines.append('=' * 110)
        rank = named.index(n) if kind == 'neuron' else None
        lines.append(f'{kind} {n}  mass_c {m[u][0]:.2f}' + (f'  (named rank {rank} of {len(named)})' if rank is not None else '')
                     + f'  top-5 {top}')
        if kind == 'neuron':
            lines.append(f'  verdict named_words: {V["named_words"][str(n)]}')
        # where the top words occur
        for z, w in m[u][1]:
            where = [f'{r["prompt"]}{r["dose"]:+g}' for k, r in by.items() if k[0] == kind and k[1] == n and w in R._words(r) - R._words(base[r['prompt']])]
            lines.append(f'    {w!r} z {z:.2f} in: {" ".join(sorted(where))}')
        for q in ('who', 'bicycle'):
            for d in (-2.0, 2.0, -1.0, 1.0):
                r = by[(kind, n, q, d)]
                t = r['text'].replace('\n', '\\n')
                lines.append(f'  [{q} {d:+g}] d4 {cz.distinct4(r["token_ids"]):.2f}: {t[:260]}')
lines.append('=' * 110)
for q in ('who', 'bicycle'):
    lines.append(f'unpushed [{q}]: ' + base[q]['text'].replace('\n', '\\n')[:260])
(C.HERE / 'q2_samples.txt').write_text('\n'.join(lines))
print('neurons', pick_n, 'controls', pick_c)
