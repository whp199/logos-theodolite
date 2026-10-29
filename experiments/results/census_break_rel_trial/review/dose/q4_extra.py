"""Follow-ups on what the lead is made of (CPU only, read-only on the runs; writes q4_extra.json here).

1. Intact text at +-1: mass_c / rep / residual on the +-1 answers whose distinct-4 >= 0.9 (answer filter), and on the
   units whose 12 +-1 answers all have distinct-4 >= 0.9 (unit filter).
2. Collapse vocabulary removed: mass_c with the words the controls themselves loop on taken out of every count
   (any word containing 'thought', plus every word that >= 10 controls hold in >= 2 of their 12 +-2 answers).
3. The named neurons' top-5 z split exactly by |dose| x (intact d4 >= 0.9 / broken d4 < 0.9) answers.
4. The residual's fit: OLS betas on the controls, and how many neurons lie outside the controls' covariate range.
5. +-2 loops per unit: share of a unit's +-2 answers that hold no word, whose most repeated word is collapse
   vocabulary, or whose most repeated word is one the unit repeats in >= 2 of its +-2 answers and is not collapse
   vocabulary (a unit-specific loop word). AUC neurons over controls of each.
6. Share of +-1 answers that loop (d4 < 0.5), by question, named / fresh / controls.
"""
import collections
import json
import math

import numpy as np

import common as C
from common import R, cz


def auc_of(d, fresh):
    if fresh is not None:
        d = {u: v for u, v in d.items() if u[0] == 'random' or u[1] in fresh}
    return round(R.auc(*R.split(d))[0], 4)


def reading(rows, keep, fresh, drop=frozenset(), extra_cov=None, units_sel=None):
    """mass_c, residual, rep_AB, rep_BA, mass_B on `keep`, the words in `drop` taken out of every count.
    `units_sel`: read only these units (R.counts lists every unit of the run, kept answers or not, so a unit filter
    must drop the others from the counts, or they enter the controls' mean and the AUC with mass 0)."""
    def cnt_of(k):
        c = R.counts(rows, k, 'own')
        if units_sel is not None:
            c = collections.defaultdict(collections.Counter, {u: v for u, v in c.items() if u in units_sel})
        if drop:
            for u in c:
                for w in list(c[u]):
                    if w in drop:
                        del c[u][w]
        return c
    kA = lambda r: keep(r) and r['prompt'] in C.A  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in C.B  # noqa: E731
    cnt, cA, cB = cnt_of(keep), cnt_of(kA), cnt_of(kB)
    m = R.mass_from_counts(cnt, 'controls')
    mA, mB = R.mass_from_counts(cA, 'controls'), R.mass_from_counts(cB, 'controls')
    repAB = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    repBA = R.z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    mass = {u: v[0] for u, v in m.items()}
    cov = R.covariates(rows, keep)
    out = {'mass_c': auc_of(mass, fresh), 'rep_AB': auc_of(repAB, fresh), 'rep_BA': auc_of(repBA, fresh),
           'mass_B': auc_of({u: v[0] for u, v in mB.items()}, fresh)}
    units = [u for u in mass if u in cov]
    mass = {u: mass[u] for u in units}
    out['mass_c_res'] = auc_of(R.residualise(mass, cov), fresh)
    if extra_cov is not None:
        cov2 = {u: {**cov[u], **extra_cov.get(u, {})} for u in units}
        names = R.COVS + tuple(sorted(next(iter(extra_cov.values())).keys()))
        out['mass_c_res_plus'] = auc_of(R.residualise(mass, cov2, names), fresh)
    out['units'] = {'neurons': sum(1 for u in units if u[0] == 'neuron' and (fresh is None or u[1] in fresh)),
                    'controls': sum(1 for u in units if u[0] == 'random')}
    return out


res = {}
for layer in ('L45', 'L57'):
    rows = C.rows(layer)
    fresh = C.FRESH[layer]
    V = json.loads((C.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']
    named = set(V['named'])
    for r in rows:
        r['_d4'] = cz.distinct4(r['token_ids'])
    pm1, pm2 = C.keep_abs(1.0), C.keep_abs(2.0)
    L = {}

    # 1. intact text at +-1
    intact = lambda r: pm1(r) and r['_d4'] >= 0.9  # noqa: E731
    n_int = collections.Counter((r['kind'], r['neuron']) for r in rows if intact(r))
    n_pm1 = collections.Counter((r['kind'], r['neuron']) for r in rows if pm1(r))
    ext = {u: {'n_ans': math.log(n_int.get(u, 0) + 1)} for u in n_pm1}
    L['pm1_answers_d4>=0.9'] = reading(rows, intact, fresh, extra_cov=ext)
    sel = {u for u in n_pm1 if n_int.get(u, 0) == n_pm1[u]}
    unitkeep = lambda r: pm1(r) and (r['kind'], r['neuron']) in sel  # noqa: E731
    L['pm1_units_all12_d4>=0.9'] = reading(rows, unitkeep, fresh, units_sel=sel)
    fr = [u for u in n_pm1 if u[0] == 'neuron' and (fresh is None or u[1] in fresh)]
    L['pm1 intact answers per unit (mean of 12)'] = {
        'fresh neurons': round(float(np.mean([n_int.get(u, 0) for u in fr])), 2),
        'named': round(float(np.mean([n_int.get(('neuron', n), 0) for n in named])), 2),
        'controls': round(float(np.mean([n_int.get(u, 0) for u in n_pm1 if u[0] == 'random'])), 2),
        'units kept by the unit filter (neurons, controls)': (sum(1 for u in sel if u in fr), sum(1 for u in sel if u[0] == 'random')),
        'named kept by the unit filter': sum(1 for n in named if ('neuron', n) in sel)}
    cov_sel = R.covariates(rows, unitkeep)
    L['pm1_units_all12_d4>=0.9']['covariate AUCs'] = {c: auc_of({u: v[c] for u, v in cov_sel.items()}, fresh) for c in R.COVS}
    print(layer, '1', json.dumps({k: L[k] for k in list(L)[:3]}), flush=True)

    # 2. collapse vocabulary removed
    c2 = R.counts(rows, pm2, 'own')
    held = collections.Counter()
    for u, c in c2.items():
        if u[0] == 'random':
            held.update(w for w, n in c.items() if n >= 2)
    allw = set()
    for c in c2.values():
        allw.update(c)
    drop = frozenset({w for w in allw if 'thought' in w} | {w for w, n in held.items() if n >= 10})
    L['collapse_vocabulary'] = sorted(w for w in drop if 'thought' not in w) + [f'+ {sum(1 for w in drop if "thought" in w)} words containing "thought"']
    L['collapse_removed'] = {name: reading(rows, k, fresh, drop=drop) for name, k in (('all', C.keep_all), ('pm1', pm1), ('pm2', pm2))}
    print(layer, '2', L['collapse_vocabulary'][:40], json.dumps(L['collapse_removed']), flush=True)

    # 3. named neurons' z by |dose| x intact/broken
    cls = lambda r: (int(abs(r['dose'])), 'intact' if r['_d4'] >= 0.9 else 'broken')  # noqa: E731
    classes = [(1, 'intact'), (1, 'broken'), (2, 'intact'), (2, 'broken')]
    cc = {k: R.counts(rows, (lambda r, k=k: C.keep_all(r) and cls(r) == k), 'own') for k in classes}
    cnt = R.counts(rows, C.keep_all, 'own')
    m = R.mass_from_counts(cnt, 'controls')
    ctl = [u for u in cnt if u[0] == 'random']
    tots = {k: collections.Counter() for k in classes}
    for k in classes:
        for u in ctl:
            tots[k].update(cc[k][u])
    share_ans = {}
    for g, us in (('named', [('neuron', n) for n in named]), ('fresh', fr), ('controls', ctl)):
        parts = collections.Counter()
        Z = 0.0
        for u in us:
            loo = u[0] == 'random'
            den = len(ctl) - 1 if loo else len(ctl)
            for z, w in m[u][1]:
                if z <= 0:
                    continue
                mus = {k: (tots[k][w] - (cc[k][u][w] if loo else 0)) / den for k in classes}
                s = math.sqrt(sum(mus.values()) + 0.5)
                for k in classes:
                    parts[k] += (cc[k][u][w] - mus[k]) / s
                Z += z
        share_ans[g] = {f'|d|={k[0]} {k[1]}': round(parts[k] / Z, 3) for k in classes}
        # share of the group's answers in each class
        nn = collections.Counter(cls(r) for r in rows if C.keep_all(r) and (r['kind'], r['neuron']) in set(us))
        share_ans[g]['answers in each class'] = {f'|d|={k[0]} {k[1]}': round(nn[k] / sum(nn.values()), 3) for k in classes}
    L['top5_z_by_dose_and_text'] = share_ans
    print(layer, '3', json.dumps(share_ans), flush=True)

    # 4. the residual's fit on each subset
    diag = {}
    for name, k in (('all', C.keep_all), ('pm1', pm1), ('pm2', pm2)):
        cov = R.covariates(rows, k)
        mm = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k, 'own'), 'controls').items()}
        cu = [u for u in cov if u[0] == 'random']
        nu = [u for u in cov if u[0] == 'neuron' and (fresh is None or u[1] in fresh)]
        X = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in cu])
        y = np.array([mm[u] for u in cu])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        Xn = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in nu])
        outside = {c: int(np.sum((Xn[:, i + 1] < X[:, i + 1].min()) | (Xn[:, i + 1] > X[:, i + 1].max()))) for i, c in enumerate(R.COVS)}
        pred_n, pred_c = Xn @ beta, X @ beta
        diag[name] = {'beta': dict(zip(('const',) + R.COVS, [round(float(b), 2) for b in beta])),
                      'R2_on_controls': round(float(r2), 3),
                      'neurons outside the controls range, per covariate': outside,
                      'mean predicted mass: neurons, controls': (round(float(pred_n.mean()), 2), round(float(pred_c.mean()), 2)),
                      'mean mass: neurons, controls': (round(float(np.mean([mm[u] for u in nu])), 2), round(float(y.mean()), 2)),
                      'controls covariate sd': {c: round(float(X[:, i + 1].std()), 4) for i, c in enumerate(R.COVS)},
                      'neurons covariate sd': {c: round(float(Xn[:, i + 1].std()), 4) for i, c in enumerate(R.COVS)}}
    L['residual_fit'] = diag
    print(layer, '4', json.dumps(diag), flush=True)

    # 5. +-2 loops per unit
    base = R.baselines(rows)
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if not pm2(r):
            continue
        u = (r['kind'], r['neuron'])
        wc = collections.Counter(cz.TOKEN.findall(r['text'].lower()))
        per[u]['n'] += 1
        if not wc:
            per[u]['no_word'] += 1
            continue
        w = wc.most_common(1)[0][0]
        if w in drop:
            per[u]['collapse'] += 1
        elif w not in R._words(base[r['prompt']]) and c2[u][w] >= 2:
            per[u]['specific'] += 1
        else:
            per[u]['other'] += 1
    loops = {}
    for f in ('no_word', 'collapse', 'specific', 'other'):
        d = {u: per[u][f] / per[u]['n'] for u in per}
        n_, c_ = R.split({u: v for u, v in d.items() if u[0] == 'random' or fresh is None or u[1] in fresh})
        loops[f] = {'mean share, neurons': round(float(np.mean(n_)), 3), 'mean share, controls': round(float(np.mean(c_)), 3),
                    'mean share, named': round(float(np.mean([d[('neuron', n)] for n in named])), 3),
                    'AUC neurons over controls': auc_of(d, fresh)}
    # a covariate on the unit-specific loop share: does it take the +-2 lead and the whole lead away
    ext = {u: {'spec2': per[u]['specific'] / per[u]['n'], 'noword2': per[u]['no_word'] / per[u]['n']} for u in per}
    loops['pm2 reading with the two loop shares added as covariates'] = reading(rows, pm2, fresh, extra_cov=ext)
    loops['all reading with the two +-2 loop shares added as covariates'] = reading(rows, C.keep_all, fresh, extra_cov=ext)
    L['pm2_loops'] = loops
    print(layer, '5', json.dumps(loops), flush=True)

    # 6. +-1 loops by question
    byq = {}
    for g, test in (('named', lambda u: u[0] == 'neuron' and u[1] in named),
                    ('fresh', lambda u: u[0] == 'neuron' and (fresh is None or u[1] in fresh)),
                    ('controls', lambda u: u[0] == 'random')):
        byq[g] = {}
        for q in C.A + C.B:
            v = [r['_d4'] < 0.5 for r in rows if pm1(r) and r['prompt'] == q and test((r['kind'], r['neuron']))]
            byq[g][q] = round(float(np.mean(v)), 3)
        v = [r['_d4'] < 0.5 for r in rows if pm1(r) and test((r['kind'], r['neuron']))]
        byq[g]['all'] = round(float(np.mean(v)), 3)
        us = {(r['kind'], r['neuron']) for r in rows if pm1(r) and test((r['kind'], r['neuron'])) and r['_d4'] < 0.5}
        byq[g]['units with >=1 looping +-1 answer'] = len(us)
    L['pm1_loop_share_by_question'] = byq
    print(layer, '6', json.dumps(byq), flush=True)
    res[layer] = L
    for r in rows:
        r.pop('_d4', None)

(C.HERE / 'q4_extra.json').write_text(json.dumps(res, indent=1, default=str))
