"""V1: full reading (check against the verdict), +-1 only, +-2 only, single doses; overlap of named sets.
Writes v1_split.json and v1_perunit_<layer>.pkl."""
import json
import pickle
import time

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcore as V

out = {}
for layer in ('L45', 'L57'):
    t = time.time()
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    verdict = json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']
    L = {}
    per = {}
    for name, keep in (('all', V.K()), ('pm1', V.K(absd=1.0)), ('pm2', V.K(absd=2.0))):
        d = V.reading(rows, keep, fresh)
        L[name] = V.public(d)
        per[name] = {'named': d['_named'], 'p': d['_p'], 'q': d['_q'], 'top': d['_top'],
                     'mass_c': d['_stat']['mass_c']}
        print(layer, name, L[name], f'{time.time() - t:.0f}s', flush=True)
    L['verdict_figures'] = {k: verdict['figures'][k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'mass_A')}
    L['verdict_named'] = len(verdict['named'])
    L['my_all_named_equals_verdict'] = sorted(per['all']['named']) == sorted(verdict['named'])
    single = {}
    for dose in (-2.0, -1.0, 1.0, 2.0):
        d = V.reading(rows, V.K(dose=dose), fresh, want_named=False)
        single[f'{dose:+g}'] = {'mass_c': d['mass_c'], 'rep_AB': d['rep_AB'], 'rep_BA': d['rep_BA']}
    L['single'] = single
    vn = set(verdict['named'])
    L['overlap'] = {
        'verdict named also named on pm1': len(vn & set(per['pm1']['named'])),
        'verdict named with q<=0.05 on pm1': sum(1 for n in vn if per['pm1']['q'].get(n, 1) <= 0.05),
        'verdict named with q<=0.05 on pm2': sum(1 for n in vn if per['pm2']['q'].get(n, 1) <= 0.05),
        'verdict named with p<=0.05 on pm2': sum(1 for n in vn if per['pm2']['p'].get(n, 1) <= 0.05),
        'verdict named with p<=0.05 on pm1': sum(1 for n in vn if per['pm1']['p'].get(n, 1) <= 0.05),
        'expected by chance p<=0.05 of named': round(0.05 * len(vn), 1),
        'all neurons with p<=0.05 on pm2': sum(1 for v in per['pm2']['p'].values() if v <= 0.05),
        'n neurons read': len(per['pm2']['p']),
    }
    print(layer, 'single', single, L['overlap'], flush=True)
    out[layer] = L
    (V.HERE / f'v1_perunit_{layer}.pkl').write_bytes(pickle.dumps(per))
(V.HERE / 'v1_split.json').write_text(json.dumps(out, indent=1, default=str))
