"""The synthesised design's reading (break_rel), on any census run's rows: CPU only, reads, never writes, the repo.

Rows are dicts with kind, neuron, prompt, dose, rel_push, finish, text, token_ids, lp (per-piece log-probs). Every
figure is taken over the answers `keep` passes, every answer counted (no max_rel cap).

Per unit (neuron or control):
- mass_c: the census's mass, against the controls' mean count alone (a control left out of its own mean), the
  answer's own question's unpushed words left out, words met in at least two answers, summed positive z of the top 5;
- mass_u: the same with compute_units' reference (every unit's mean) and exclusion (every question's unpushed words),
  as the code reads today (a cross-check: it reproduces compute_units);
- shift, shift_up, shift_down: census.shift_scores (every word, controls' mean, leave-one-out);
- rep_AB / rep_BA: word-level replication, the summed half-B z (control-referenced, half B's answers alone) of the
  unit's top-5 half-A words (and the reverse): words picked on one half, scored on the other;
- named: q <= 0.05 on mass_c over the layer's neurons (Benjamini-Hochberg), its half-A words recurring on half B and
  its half-B words on half A (rep_AB and rep_BA p <= 0.05 each), and p <= 0.05 on mass_c's covariate residual; the
  design's first wording (p <= 0.05 on mass_A and on mass_B, which a neuron strong on each half with other words
  passes) is reported beside it as named_by_halves;
- xq: question recurrence, the number of questions whose answers hold a word (own unpushed words left out), z against
  the controls' mean, summed positive z of the top 5 among words met on 2 or more questions;
- covariates (how the text survives, not what it says): log1p new words an answer, mean distinct-4, share cut off,
  mean log-prob, mean first departure from the own question's unpushed answer (a share of the answer's length), mean
  share of the own question's unpushed words lost.
"""
import collections
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

import os
import sys
# the checkout whose backend/census.py is read (THEODOLITE_TRIAL_REPO: the break_rel commit the trial ran from; else the
# main checkout); every census function used here is the same in both
sys.path.insert(0, os.environ.get('THEODOLITE_TRIAL_REPO') or '/home/eric/bin/ai/theodolite2')
from backend import census as cz  # noqa: E402

COVS = ('lnw', 'd4', 'cut', 'lp', 'dep', 'lost')


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan')
    u = stats.mannwhitneyu(a, b, alternative='greater')
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def stouffer(ps):
    ps = [min(max(p, 1e-12), 1 - 1e-12) for p in ps if p == p]
    return float(np.sum(stats.norm.isf(ps)) / math.sqrt(len(ps))) if ps else float('nan')


def _words(r):
    w = r.get('_w')
    if w is None:
        w = r['_w'] = frozenset(cz.words(r['text']))
    return w


def baselines(rows):
    b = {}
    for r in rows:
        if r['kind'] == 'baseline':
            b.setdefault(r['prompt'], r)
    return b


def _z_top(c: collections.Counter, tot: collections.Counter, den: int, loo: bool, k=5, min_count=2):
    zs = []
    for w, cnt in c.items():
        if cnt < min_count:
            continue
        mu = (tot[w] - (cnt if loo else 0)) / den
        zs.append(((cnt - mu) / math.sqrt(mu + 0.5), w))
    zs.sort(reverse=True)
    return zs[:k]


def counts(rows, keep, exclude='own'):
    """{unit: Counter of word -> answers holding it}, own (or every) question's unpushed words left out."""
    base = baselines(rows)
    allbase = frozenset().union(*(_words(r) for r in base.values())) if base else frozenset()
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        if r['kind'] == 'baseline':
            continue
        u = (r['kind'], int(r['neuron']))
        out[u]
        if not keep(r):
            continue
        ex = _words(base[r['prompt']]) if exclude == 'own' else allbase
        out[u].update(_words(r) - ex)
    return out


def mass_from_counts(cnt, reference='controls', k=5):
    """{unit: (mass, top words [(z, w)])}."""
    units = list(cnt)
    ctl = [u for u in units if u[0] == 'random']
    tot = collections.Counter()
    pool = ctl if reference == 'controls' else units
    for u in pool:
        tot.update(cnt[u])
    out = {}
    for u in units:
        if reference == 'controls':
            loo = u[0] == 'random'
            den = len(ctl) - 1 if loo else len(ctl)
        else:  # compute_units: every unit's mean, the unit itself included
            loo, den = False, len(units)
        top = _z_top(cnt[u], tot, den, loo, k=k)
        out[u] = (float(sum(max(z, 0.0) for z, _ in top)), top)
    return out


def z_of(cnt, words_of_unit, reference='controls'):
    """{unit: summed z of the given words} on these counts (no min-count), control-referenced, leave-one-out."""
    ctl = [u for u in cnt if u[0] == 'random']
    tot = collections.Counter()
    for u in ctl:
        tot.update(cnt[u])
    out = {}
    for u, ws in words_of_unit.items():
        loo = u[0] == 'random'
        den = len(ctl) - 1 if loo else len(ctl)
        s = 0.0
        for w in ws:
            c = cnt[u][w]
            mu = (tot[w] - (c if loo else 0)) / den
            s += (c - mu) / math.sqrt(mu + 0.5)
        out[u] = s
    return out


def word_z(cnt, u, words):
    """[(word, z)] of the given words for one unit on these counts, control-referenced (leave-one-out for a control)."""
    ctl = [v for v in cnt if v[0] == 'random']
    tot = collections.Counter()
    for v in ctl:
        tot.update(cnt[v])
    loo = u[0] == 'random'
    den = len(ctl) - 1 if loo else len(ctl)
    out = []
    for w in words:
        c = cnt[u][w]
        mu = (tot[w] - (c if loo else 0)) / den
        out.append((w, round((c - mu) / math.sqrt(mu + 0.5), 2)))
    return out


def xq_scores(rows, keep):
    base = baselines(rows)
    qs = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if r['kind'] == 'baseline':
            continue
        u = (r['kind'], int(r['neuron']))
        qs[u]
        if not keep(r):
            continue
        for w in _words(r) - _words(base[r['prompt']]):
            qs[u][w].add(r['prompt'])
    cnt = {u: collections.Counter({w: len(s) for w, s in d.items()}) for u, d in qs.items()}
    m = mass_from_counts(cnt, 'controls')
    return {u: v[0] for u, v in m.items()}


def covariates(rows, keep):
    base = baselines(rows)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r['kind'] == 'baseline' or not keep(r):
            continue
        u = (r['kind'], int(r['neuron']))
        b = base[r['prompt']]
        bw = _words(b)
        ids = r['token_ids']
        acc[u]['lnw'].append(len(_words(r) - bw))
        acc[u]['d4'].append(cz.distinct4(ids))
        acc[u]['cut'].append(1.0 if r['finish'] == 'length' else 0.0)
        acc[u]['lp'].append(float(np.mean(r['lp'])) if r['lp'] else 0.0)
        d = cz.first_departure(ids, b['token_ids'])
        acc[u]['dep'].append((len(ids) if d is None else d) / max(len(ids), 1))
        acc[u]['lost'].append(len(bw - _words(r)) / max(len(bw), 1))
    out = {}
    for u, a in acc.items():
        out[u] = {k: float(np.mean(v)) for k, v in a.items()}
        out[u]['lnw'] = math.log1p(out[u]['lnw'])
    return out


def residualise(stat: Dict, cov: Dict, covs=COVS):
    """stat minus its OLS fit on the covariates, the fit over the controls alone."""
    units = [u for u in stat if u in cov]
    ctl = [u for u in units if u[0] == 'random']
    X = np.array([[1.0] + [cov[u][c] for c in covs] for u in ctl])
    y = np.array([stat[u] for u in ctl])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {u: stat[u] - float(np.dot([1.0] + [cov[u][c] for c in covs], beta)) for u in units}


def split(d):
    return ([v for u, v in d.items() if u[0] == 'neuron'], [v for u, v in d.items() if u[0] == 'random'])


def emp_p(d):
    """{neuron: (1 + controls at or above) / (C + 1)}, and {control: leave-one-out p among controls}."""
    ctl = np.sort([v for u, v in d.items() if u[0] == 'random'])
    C = len(ctl)
    out = {}
    for u, v in d.items():
        ge = C - int(np.searchsorted(ctl, v, side='left'))
        out[u] = (1 + ge) / (C + 1) if u[0] == 'neuron' else (ge - 1 + 1) / C  # a control counts itself among the ge
    return out


def bh(ps: Dict) -> Dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    n = len(items)
    q, out = 1.0, {}
    for i in range(n - 1, -1, -1):
        q = min(q, items[i][1] * n / (i + 1))
        out[items[i][0]] = q
    return out


def read(rows, keep, half_a: Sequence[str], half_b: Sequence[str], units_only: Optional[set] = None, null_splits=20, seed=0):
    """Every figure of the reading. `units_only`: restrict the neurons read (e.g. the fresh ones) after the
    statistics are computed against every control."""
    rows = list(rows)
    A, B = set(half_a), set(half_b)
    kA = lambda r: keep(r) and r['prompt'] in A  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in B  # noqa: E731
    cnt = counts(rows, keep, 'own')
    m_c = mass_from_counts(cnt, 'controls')
    m_u = mass_from_counts(counts(rows, keep, 'all'), 'units')
    cA, cB = counts(rows, kA, 'own'), counts(rows, kB, 'own')
    mA, mB = mass_from_counts(cA, 'controls'), mass_from_counts(cB, 'controls')
    repAB = z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    repBA = z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    sl = cz.SlimRows()
    sl.extend(cz.SlimRows.slim(sl, {**r, 'logprob': r['lp'], 'rel_push': 0.0}) for r in rows if r['kind'] == 'baseline' or keep(r))
    sh = cz.shift_scores(sl, max_rel=float('inf'))
    xq = xq_scores(rows, keep)
    cov = covariates(rows, keep)
    stat = {
        'mass_c': {u: v[0] for u, v in m_c.items()},
        'mass_u': {u: v[0] for u, v in m_u.items()},
        'shift': {u: (v['shift'] or 0.0) for u, v in sh.items()},
        'shift_up': {u: sum(z for _, z in v['up']) for u, v in sh.items()},
        'shift_down': {u: -sum(z for _, z in v['down']) for u, v in sh.items()},
        'mass_A': {u: v[0] for u, v in mA.items()},
        'mass_B': {u: v[0] for u, v in mB.items()},
        'rep_AB': repAB, 'rep_BA': repBA,
        'rep': {u: repAB[u] + repBA[u] for u in repAB},
        'xq': xq,
    }
    for c in COVS:
        stat['cov_' + c] = {u: cov[u][c] for u in cov}
    for key in ('mass_c', 'shift', 'shift_up', 'rep'):
        stat[key + '_res'] = residualise(stat[key], cov)
    if units_only is not None:
        stat = {k: {u: v for u, v in d.items() if u[0] == 'random' or u[1] in units_only} for k, d in stat.items()}
    fig = {}
    for k, d in stat.items():
        n, c = split(d)
        a, p = auc(n, c)
        fig[k] = {'auc': round(a, 4), 'p': float(f'{p:.3g}')}
    # replication of the mass rank across halves (P2's P4, kept as a report)
    neuA, neuB = split(stat['mass_A']), split(stat['mass_B'])
    nkeys = [u for u in stat['mass_A'] if u[0] == 'neuron']
    ckeys = [u for u in stat['mass_A'] if u[0] == 'random']
    fig['rho_AB_neurons'] = float(stats.spearmanr([stat['mass_A'][u] for u in nkeys], [stat['mass_B'][u] for u in nkeys])[0])
    fig['rho_AB_controls'] = float(stats.spearmanr([stat['mass_A'][u] for u in ckeys], [stat['mass_B'][u] for u in ckeys])[0])
    # per-neuron p, q and the naming rule
    p_all, p_A, p_B, p_r = emp_p(stat['mass_c']), emp_p(stat['mass_A']), emp_p(stat['mass_B']), emp_p(stat['mass_c_res'])
    p_AB, p_BA = emp_p(stat['rep_AB']), emp_p(stat['rep_BA'])
    q_all = bh({u: v for u, v in p_all.items() if u[0] == 'neuron'})
    # named: its words replicate across the halves both ways (not only strong on each half, perhaps with other words)
    named = [u for u in q_all if q_all[u] <= 0.05 and p_AB[u] <= 0.05 and p_BA[u] <= 0.05 and p_r[u] <= 0.05]
    named_by_halves = [u for u in q_all if q_all[u] <= 0.05 and p_A[u] <= 0.05 and p_B[u] <= 0.05 and p_r[u] <= 0.05]
    # a named neuron's replicated words: its top half-A words with a positive half-B z, and the reverse
    named_words = {u[1]: {'A_on_B': [x for x in word_z(cB, u, [w for z, w in mA[u][1] if z > 0]) if x[1] > 0],
                          'B_on_A': [x for x in word_z(cA, u, [w for z, w in mB[u][1] if z > 0]) if x[1] > 0]} for u in named}
    N, C = len(nkeys), len(ckeys)
    ctl_mass = np.array(split(stat['mass_c'])[1])
    neu_mass = np.array(split(stat['mass_c'])[0])
    fig['counts'] = {'neurons': N, 'controls': C, 'above_every_control': int((neu_mass > ctl_mass.max()).sum()),
                     'by_chance': round(N / (C + 1), 1), 'q05': int(sum(1 for v in q_all.values() if v <= 0.05)),
                     'named': len(named), 'named_by_halves': len(named_by_halves),
                     'above_ctl_p95': round(float((neu_mass > np.percentile(ctl_mass, 95)).mean()), 4),
                     'pA05_and_pB05': int(sum(1 for u in nkeys if p_A[u] <= 0.05 and p_B[u] <= 0.05)),
                     'pA05_and_pB05_by_chance': round(N * 0.05 * 0.05, 1)}
    # null check: half the controls relabelled as neurons, the reference from the other half alone
    rng = np.random.default_rng(seed)
    ctl_units = [u for u in cnt if u[0] == 'random']
    nulls = []
    for _ in range(null_splits):
        perm = rng.permutation(len(ctl_units))
        fake = {ctl_units[i] for i in perm[: len(perm) // 2]}
        relab = {(('neuron', u[1] + 10_000_000) if u in fake else u): c for u, c in cnt.items() if u[0] == 'random'}
        mm = mass_from_counts(relab, 'controls')
        n, c = split({u: v[0] for u, v in mm.items()})
        nulls.append(auc(n, c)[0])
    fig['null_split_auc'] = {'mean': round(float(np.mean(nulls)), 4), 'min': round(float(np.min(nulls)), 4),
                             'max': round(float(np.max(nulls)), 4), 'n': null_splits}
    return fig, stat, named, {'p': p_all, 'q': q_all, 'pA': p_A, 'pB': p_B, 'pr': p_r, 'pAB': p_AB, 'pBA': p_BA, 'top': m_c,
                              'named_by_halves': named_by_halves, 'named_words': named_words}
