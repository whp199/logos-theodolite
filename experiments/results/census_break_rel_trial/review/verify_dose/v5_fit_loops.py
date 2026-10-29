"""V5: the residual's fit (R2 on controls, predicted means, sd, range, betas) on all / +-1 / +-2; drop-one covariate
residual AUCs; per-unit +-2 loop composition (no word / collapse / unit-specific / other) and the residual with the
two loop shares added; the controls' range of those shares; the collapse vocabulary (words with 'thought' + words that
>= 10 controls hold in >= 2 of their 12 +-2 answers). Writes v5_fit_loops.json and v5_drop_<layer>.json."""
import collections
import json
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import vcore as V  # noqa: E402
from vcore import R, cz  # noqa: E402

out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    named = set(json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']['named'])
    L = {}
    fit = {}
    for name, keep in (('all', V.K()), ('pm1', V.K(absd=1.0)), ('pm2', V.K(absd=2.0))):
        cov = R.covariates(rows, keep)
        mass = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, keep, 'own'), 'controls').items()}
        cu = sorted(u for u in cov if u[0] == 'random')
        nu = sorted(u for u in cov if u[0] == 'neuron' and (fresh is None or u[1] in fresh))
        X = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in cu])
        Xn = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in nu])
        y = np.array([mass[u] for u in cu])
        yn = np.array([mass[u] for u in nu])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fit[name] = {'R2': round(float(r2), 3), 'pred_mean_neurons': round(float((Xn @ b).mean()), 2),
                     'pred_mean_controls': round(float((X @ b).mean()), 2), 'mass_mean_neurons': round(float(yn.mean()), 2),
                     'mass_mean_controls': round(float(y.mean()), 2),
                     'beta': {c: round(float(v), 2) for c, v in zip(('const',) + R.COVS, b)},
                     'sd_controls': {c: round(float(X[:, i + 1].std()), 4) for i, c in enumerate(R.COVS)},
                     'sd_neurons': {c: round(float(Xn[:, i + 1].std()), 4) for i, c in enumerate(R.COVS)},
                     'neurons_below_ctl_min': {c: int((Xn[:, i + 1] < X[:, i + 1].min()).sum()) for i, c in enumerate(R.COVS)},
                     'neurons_above_ctl_max': {c: int((Xn[:, i + 1] > X[:, i + 1].max()).sum()) for i, c in enumerate(R.COVS)}}
        # drop-one covariate residual AUCs, and each covariate alone
        dd = {}
        for c in R.COVS:
            rest = tuple(x for x in R.COVS if x != c)
            dd['without_' + c] = round(V.auc_n(R.residualise({u: mass[u] for u in cov}, cov, rest), fresh)[0], 4)
            dd['only_' + c] = round(V.auc_n(R.residualise({u: mass[u] for u in cov}, cov, (c,)), fresh)[0], 4)
        dd['all_six'] = round(V.auc_n(R.residualise({u: mass[u] for u in cov}, cov), fresh)[0], 4)
        dd['raw'] = round(V.auc_n(mass, fresh)[0], 4)
        fit[name]['drop_one'] = dd
    L['fit'] = fit
    # collapse vocabulary and the +-2 loop composition
    c2 = R.counts(rows, V.K(absd=2.0), 'own')
    held = collections.Counter()
    allw = set()
    for u, c in c2.items():
        allw.update(c)
        if u[0] == 'random':
            held.update(w for w, n in c.items() if n >= 2)
    drop = frozenset({w for w in allw if 'thought' in w} | {w for w, n in held.items() if n >= 10})
    L['collapse_vocab_non_thought'] = sorted(w for w in drop if 'thought' not in w)
    L['collapse_vocab_size'] = len(drop)
    base = {r['prompt']: set(cz.words(r['text'])) for r in rows if r['kind'] == 'baseline'}
    per = collections.defaultdict(collections.Counter)
    per1 = collections.defaultdict(collections.Counter)
    for r in rows:
        if not V.in_design(r):
            continue
        u = (r['kind'], r['neuron'])
        toks = cz.TOKEN.findall(r['text'].lower())
        if abs(r['dose']) == 2.0:
            P = per[u]
            P['n'] += 1
            if not toks:
                P['no_word'] += 1
                continue
            w = collections.Counter(toks).most_common(1)[0][0]
            if w in drop:
                P['collapse'] += 1
            elif w not in base[r['prompt']] and c2[u][w] >= 2:
                P['specific'] += 1
            else:
                P['other'] += 1
        elif r['d4'] < 0.5:  # looping +-1 answers: what they loop on
            if not toks:
                per1[u]['no_word'] += 1
            else:
                w = collections.Counter(toks).most_common(1)[0][0]
                per1[u]['collapse' if w in drop else 'noncollapse'] += 1
    comp = {}
    for f in ('no_word', 'collapse', 'specific', 'other'):
        d = {u: per[u][f] / per[u]['n'] for u in per}
        n_, c_ = R.split({u: v for u, v in d.items() if u[0] == 'random' or fresh is None or u[1] in fresh})
        comp[f] = {'neurons': round(float(np.mean(n_)), 3), 'controls': round(float(np.mean(c_)), 3),
                   'named': round(float(np.mean([d[('neuron', n)] for n in named])), 3),
                   'auc': round(V.auc_n(d, fresh)[0], 3),
                   'controls_max': round(float(np.max(c_)), 3), 'controls_p99': round(float(np.percentile(c_, 99)), 3),
                   'neurons_above_ctl_max': int((np.array(n_) > np.max(c_)).sum())}
    L['pm2_loop_composition'] = comp
    tot1 = {g: collections.Counter() for g in ('named', 'neurons', 'controls')}
    for u, c in per1.items():
        if u[0] == 'random':
            tot1['controls'].update(c)
        elif fresh is None or u[1] in fresh:
            tot1['neurons'].update(c)
            if u[1] in named:
                tot1['named'].update(c)
    L['pm1_looping_answers_loop_on'] = {g: dict(c) for g, c in tot1.items()}
    ext = {u: {'spec2': per[u]['specific'] / per[u]['n'], 'noword2': per[u]['no_word'] / per[u]['n']} for u in per}
    for name, keep in (('all', V.K()), ('pm2', V.K(absd=2.0))):
        d = V.reading(rows, keep, fresh, extra_cov=ext, want_named=False)
        L[f'{name}_with_loop_covariates'] = {'mass_c': d['mass_c'], 'mass_c_res': d['mass_c_res'], 'mass_c_res_plus': d['mass_c_res_plus']}
    print(layer, json.dumps(L), flush=True)
    out[layer] = L
    (V.HERE / f'v5_drop_{layer}.json').write_text(json.dumps(sorted(drop)))
(V.HERE / 'v5_fit_loops.json').write_text(json.dumps(out, indent=1))
