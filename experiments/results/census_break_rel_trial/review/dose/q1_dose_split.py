"""Q1/Q4: the pre-registered reading (readlib.read, unchanged) on the +-1 answers alone and on the +-2 answers alone,
counts, controls' mean, covariates and residual all re-computed on each subset. Also mass_c at each single dose.
Writes q1_dose_split.json and q1_perunit_<layer>.pkl here."""
import json
import pickle
import time

import common as C
from common import R

GATES45 = {'mass_c': 0.60, 'mass_c_res': 0.58, 'rep_AB': 0.57, 'rep_BA': 0.57, 'mass_B': 0.56}
GATES57 = {'mass_c': 0.58, 'mass_c_res': 0.56, 'rep_AB': 0.55, 'rep_BA': 0.55}
KEYS = ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_A', 'mass_B', 'mass_u', 'shift', 'xq',
        'cov_lnw', 'cov_d4', 'cov_cut', 'cov_lp', 'cov_dep', 'cov_lost']

out = {}
for layer in ('L45', 'L57'):
    t = time.time()
    rows = C.rows(layer)
    fresh = C.FRESH[layer]
    gates = GATES45 if layer == 'L45' else GATES57
    out[layer] = {}
    per = {}
    for name, keep in (('all', C.keep_all), ('pm1', C.keep_abs(1.0)), ('pm2', C.keep_abs(2.0))):
        fig, stat, named, pp = R.read(rows, keep, C.A, C.B, units_only=fresh, null_splits=20)
        d = {k: fig[k]['auc'] for k in KEYS}
        d['counts'] = fig['counts']
        d['null_split_auc'] = fig['null_split_auc']
        d['rho_AB_neurons'] = round(fig['rho_AB_neurons'], 3)
        d['rho_AB_controls'] = round(fig['rho_AB_controls'], 3)
        d['bars'] = {k: (fig[k]['auc'], bar, fig[k]['auc'] >= bar) for k, bar in gates.items()}
        d['passes_all_bars'] = all(v[2] for v in d['bars'].values())
        out[layer][name] = d
        per[name] = {'stat': {k: stat[k] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_A', 'mass_B')},
                     'top': {u: v[1] for u, v in pp['top'].items()}, 'named': [u[1] for u in named],
                     'p': pp['p'], 'q': pp['q']}
        print(layer, name, {k: d[k] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B')},
              'named', d['counts']['named'], 'q05', d['counts']['q05'], 'null', d['null_split_auc']['mean'],
              f'{time.time() - t:.0f}s', flush=True)
    # each single dose (6 answers a unit): mass_c only
    single = {}
    for dose in C.DOSES:
        k = (lambda r, dd=dose: C.keep_all(r) and abs(r['dose'] - dd) < 1e-9)
        m = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k, 'own'), 'controls').items()}
        if fresh is not None:
            m = {u: v for u, v in m.items() if u[0] == 'random' or u[1] in fresh}
        single[f'{dose:+g}'] = round(R.auc(*R.split(m))[0], 4)
    out[layer]['single_dose_mass_c'] = single
    print(layer, 'single doses', single, flush=True)
    with open(C.HERE / f'q1_perunit_{layer}.pkl', 'wb') as f:
        pickle.dump(per, f)

(C.HERE / 'q1_dose_split.json').write_text(json.dumps(out, indent=1, default=str))
