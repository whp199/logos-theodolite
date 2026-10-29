#!/usr/bin/env python3
"""Re-derive the reviewer's headline figures with myread (independent of readlib): the gates at 48 and 32 pieces, SEs,
the max_new rule, the null split and its variants, G5b against units.jsonl, every neuron, covariates left out one at a
time, covariate overlap, the lead by dose, the text per dose, a tie-break check and a bootstrap of the residual gate.
Then readlib's per-unit values for a unit-by-unit cross-check. CPU only; reads only; writes gates_<run>.json here.

    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<worktree> .venv/bin/python -I -B verify_gates.py L45|L57
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import myread as M  # noqa: E402

T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
which = sys.argv[1]
run = T / ('L45-break_rel-trial.jsonl' if which == 'L45' else 'L57-break_rel-probe.jsonl')
fresh = set(int(x) for x in (T / 'prereg/trial_L45_fresh.txt').read_text().split(',') if x.strip()) if which == 'L45' else None
BARS = ({'mass_c': 0.60, 'mass_c_res': 0.58, 'rep_AB': 0.57, 'rep_BA': 0.57, 'mass_B': 0.56} if which == 'L45' else
        {'mass_c': 0.58, 'mass_c_res': 0.56, 'rep_AB': 0.55, 'rep_BA': 0.55})
KEYS = ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'mass_A', 'rep', 'rep_res'] + ['cov_' + c for c in M.COVS]
t0 = time.time()
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(M.TOK_PATH)
out = {'run': run.name}


def log(*a):
    print(f'[{time.time() - t0:6.1f}s]', *a, flush=True)


def null_split(cnt, order, variant, n=50, seed=0):
    ctl = [u for u in order if u[0] == 'random']
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        perm = rng.permutation(len(ctl))
        fake = {ctl[i] for i in perm[: len(perm) // 2]}
        relab = {(('neuron', u[1] + 10_000_000) if u in fake else u): cnt[u] for u in ctl}
        kw = {'reader': {}, 'no_loo': {'loo': False}, 'units_ref': {'ref': 'units'}}[variant]
        m = M.mass(M.tops(relab, **kw))
        res.append(M.auc([v for u, v in m.items() if u[0] == 'neuron'], [v for u, v in m.items() if u[0] == 'random']))
    return {'mean': round(float(np.mean(res)), 4), 'sd': round(float(np.std(res)), 4), 'min': round(float(min(res)), 4),
            'max': round(float(max(res)), 4), 'split': [len(ctl) // 2, len(ctl) - len(ctl) // 2]}


readings = {}
for cut in (48, 32):
    rows = M.load(run, cut, tok)
    rd = M.Reading(rows)
    st, cov, tp, c_all = M.statistics(rd)
    a = M.aucs(st, fresh, KEYS)
    r = {'auc': {k: v['auc'] for k, v in a.items()}, 'n': a['mass_c']['n'], 'c': a['mass_c']['c']}
    r['se'] = {k: round(M.hanley_mcneil(r['auc'][k], r['n'], r['c']), 4) for k in BARS}
    r['gates'] = {k: {'auc': round(r['auc'][k], 5), 'bar': b, 'pass': r['auc'][k] >= b,
                      'margin': round(r['auc'][k] - b, 5), 'margin_in_se': round((r['auc'][k] - b) / r['se'][k], 3)}
                  for k, b in BARS.items()}
    r['null_reader'] = null_split(c_all, rd.order, 'reader')
    r['pass_gates_1_4'] = all(g['pass'] for g in r['gates'].values())
    readings[cut] = r
    log(cut, {k: round(v, 4) for k, v in r['auc'].items()}, r['null_reader'])
    if cut == 48:
        rows48, rd48, st48, cov48, tp48, c48 = rows, rd, st, cov, tp, c_all
out['readings'] = {str(k): v for k, v in readings.items()}
f1, f2 = readings[48]['auc'], readings[32]['auc']
mr = {'mass_c_drop': round(f1['mass_c'] - f2['mass_c'], 4),
      'rep_min_drop': round(min(f1['rep_AB'], f1['rep_BA']) - min(f2['rep_AB'], f2['rep_BA']), 4),
      'shorter_passes_gates': readings[32]['pass_gates_1_4']}
mr['take_shorter_if_G5_holds'] = bool(mr['shorter_passes_gates'] and mr['mass_c_drop'] <= 0.02 and mr['rep_min_drop'] <= 0.02)
out['max_new_rule'] = mr
log('max_new', mr)

# the null split's variants, and the real comparison under each
out['null_variants'] = {v: null_split(c48, rd48.order, v) for v in ('reader', 'no_loo', 'units_ref')}
real = {}
for v, kw in (('reader', {}), ('no_loo', {'loo': False}), ('units_ref', {'ref': 'units'})):
    real[v] = round(M.aucs({'m': M.mass(M.tops(c48, **kw))}, fresh)['m']['auc'], 4)
out['null_variants']['real_mass_auc_under_each'] = real
log('null variants', out['null_variants'])

# G5b: units.jsonl against this mass_c, and against other readings
uf = {}
with open(str(run)[:-6] + '.units.jsonl', encoding='utf-8') as f:
    for line in f:
        u = json.loads(line)
        uf[(u['kind'], int(u['neuron']))] = u
mc = st48['mass_c']
c_allq = rd48.counts(exclude='all')
old_capped = M.Reading(rows48, doses=(-1.0, 1.0)).counts(exclude='all')  # READING_DEFAULT: max_rel 1.6 drops +-2
alt = {'units_ref_every_question_excluded_every_answer': M.mass(M.tops(c_allq, ref='units')),
       'old_reading_with_cap_1.6 (units ref, every question excluded, |dose| 1 only)': M.mass(M.tops(old_capped, ref='units')),
       'units_ref_own_excluded': M.mass(M.tops(c48, ref='units')),
       'controls_ref_every_question_excluded': M.mass(M.tops(c_allq))}
out['G5b'] = {'units_in_file': len(uf), 'units_read': len(mc), 'reading_in_file': sorted({json.dumps(u.get('reading'), sort_keys=True) for u in uf.values()}),
              'max_abs_diff': round(max(abs(uf[u]['mass'] - mc[u]) for u in mc), 6),
              'max_abs_diff_vs_round3': round(max(abs(uf[u]['mass'] - round(mc[u], 3)) for u in mc), 6),
              'top5_same_order': sum(1 for u in mc if [w[0] for w in uf[u]['words'][:5]] == [w for _, w in tp48['all'][u]]),
              'other_readings_max_abs_diff': {k: round(max(abs(uf[u]['mass'] - d[u]) for u in mc), 3) for k, d in alt.items()}}
log('G5b', out['G5b'])

# every neuron (re-asked included)
if fresh is not None:
    out['every_neuron'] = {k: round(v['auc'], 4) for k, v in M.aucs(st48, None, ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B']).items()}
    out['every_neuron']['n'] = M.aucs(st48, None, ['mass_c'])['mass_c']['n']
    log('every neuron', out['every_neuron'])

# covariates: one left out at a time, the fit's R2, overlap with the controls
sel = lambda u: u[0] == 'random' or fresh is None or u[1] in fresh  # noqa: E731
loo_cov = {}
for c in M.COVS:
    rest = tuple(x for x in M.COVS if x != c)
    loo_cov['without_' + c] = round(M.aucs({'r': M.resid(st48['mass_c'], cov48, rest)}, fresh)['r']['auc'], 4)
ctl = [u for u in st48['mass_c'] if u[0] == 'random']
neu = [u for u in st48['mass_c'] if u[0] == 'neuron' and sel(u)]
X = np.array([[1.0] + [cov48[u][c] for c in M.COVS] for u in ctl])
y = np.array([st48['mass_c'][u] for u in ctl])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
overlap = {}
for c in M.COVS:
    cv, nv = np.array([cov48[u][c] for u in ctl]), np.array([cov48[u][c] for u in neu])
    overlap[c] = {'above_ctl_p99': round(float((nv > np.percentile(cv, 99)).mean()), 3),
                  'below_ctl_p1': round(float((nv < np.percentile(cv, 1)).mean()), 3)}
out['residual'] = {'six': round(readings[48]['auc']['mass_c_res'], 4), 'leave_one_out': loo_cov,
                   'range': [min(loo_cov.values()), max(loo_cov.values())],
                   'R2_controls': round(float(1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
                   'beta': dict(zip(['const'] + list(M.COVS), [round(float(b), 3) for b in beta])), 'overlap': overlap}
log('residual', out['residual'])

# bootstrap of the residual gate (units resampled, the fit redone on the resampled controls; mass held fixed)
rng = np.random.default_rng(7)
ctl_a, neu_a = np.array(ctl, dtype=object), np.array(neu, dtype=object)
Xc = np.array([[1.0] + [cov48[u][c] for c in M.COVS] for u in ctl])
Xn = np.array([[1.0] + [cov48[u][c] for c in M.COVS] for u in neu])
yc, yn = np.array([st48['mass_c'][u] for u in ctl]), np.array([st48['mass_c'][u] for u in neu])
bs = []
for _ in range(1000):
    ic, ineu = rng.integers(0, len(ctl), len(ctl)), rng.integers(0, len(neu), len(neu))
    b = np.linalg.lstsq(Xc[ic], yc[ic], rcond=None)[0]
    bs.append(M.auc(yn[ineu] - Xn[ineu] @ b, yc[ic] - Xc[ic] @ b))
bar = BARS['mass_c_res']
out['residual_bootstrap'] = {'n': 1000, 'mean': round(float(np.mean(bs)), 4), 'sd': round(float(np.std(bs)), 4),
                             'share_below_bar': round(float(np.mean(np.array(bs) < bar)), 3),
                             'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
log('bootstrap', out['residual_bootstrap'])

# the lead by dose
by = {}
for name, ds in (('|dose| 1', (-1.0, 1.0)), ('|dose| 2', (-2.0, 2.0))):
    s2, _, _, _ = M.statistics(M.Reading(rows48, doses=ds))
    by[name] = {k: round(v['auc'], 4) for k, v in M.aucs(s2, fresh, ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res', 'cov_lp', 'cov_d4', 'cov_lost']).items()}
out['by_dose'] = by
log('by dose', by)

# text per dose: distinct-4 minus the unpushed, and share cut off (fresh neurons, every neuron, controls)
text = {}
for d in M.DOSES:
    for label, f in (('fresh_neurons', lambda r: r['kind'] == 'neuron' and (fresh is None or r['neuron'] in fresh)),
                     ('every_neuron', lambda r: r['kind'] == 'neuron'), ('controls', lambda r: r['kind'] == 'random')):
        rs = [r for r in rows48 if f(r) and abs(r['dose'] - d) < 1e-9 and r['prompt'] in M.A_HALF + M.B_HALF]
        text[f'{d:+g} {label}'] = {'d4_minus_unpushed': round(float(np.mean([M.distinct4(r['ids']) - M.distinct4(rd48.base[r['prompt']]['ids']) for r in rs])), 3),
                                   'cut_off': round(float(np.mean([r['finish'] == 'length' for r in rs])), 3), 'n': len(rs)}
out['text_per_dose'] = text
log('text', text)

# tie-break: the same reading with ties among words broken the other way
st_alt, _, _, _ = M.statistics(rd48, tiebreak='alt')
out['tiebreak_alt'] = {k: round(v['auc'], 4) for k, v in M.aucs(st_alt, fresh, ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B']).items()}
log('tiebreak', out['tiebreak_alt'])

# unit-by-unit against readlib (the pre-registered library), on the 48-piece reading
sys.path.insert(0, str(T / 'prereg'))
import readlib as R  # noqa: E402
lrows = [{'layer': 0, 'kind': r['kind'], 'neuron': r['neuron'], 'prompt': r['prompt'], 'dose': r['dose'], 'rel_push': r['rel_push'],
          'finish': r['finish'], 'text': r['text'], 'token_ids': r['ids'], 'lp': r['lp']} for r in rows48]
keep = lambda r: r['prompt'] in set(M.A_HALF + M.B_HALF) and any(abs(r['dose'] - d) < 1e-9 for d in M.DOSES)  # noqa: E731
fig, stat, _, _ = R.read(lrows, keep, list(M.A_HALF), list(M.B_HALF), units_only=fresh, null_splits=50)
cmp = {}
for k in ('mass_c', 'mass_B', 'rep_AB', 'rep_BA', 'mass_c_res', 'rep_res', 'cov_lp'):
    cmp[k] = {'max_abs_unit_diff': float(max(abs(stat[k][u] - st48[k][u]) for u in stat[k])),
              'readlib_auc': fig[k]['auc'], 'mine_auc': round(readings[48]['auc'][k], 4)}
cmp['null_readlib'] = fig['null_split_auc']
out['vs_readlib'] = cmp
log('vs readlib', cmp)
(HERE / f'gates_{which}.json').write_text(json.dumps(out, indent=1, default=str))
log('done')
