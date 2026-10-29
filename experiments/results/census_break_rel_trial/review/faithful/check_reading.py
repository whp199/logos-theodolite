#!/usr/bin/env python3
"""Faithfulness check, part 2 (CPU only, reads only). For one run:
1. re-run the pre-registered reading (readlib.read, as trial_read calls it) and compare every AUC with the verdict JSON;
2. the gated AUCs unrounded, with Hanley-McNeil SEs;
3. an independent implementation (written here from the pre-registration's words, not readlib) of mass_c, mass_B,
   rep_AB, rep_BA and mass_c's residual, and its AUCs;
4. G5b in detail: units.jsonl mass against the reader's mass_c (unrounded and rounded to 3 places), top-5 words, and
   against mass_u (the old reading), to show whether the check can tell readings apart;
5. the null split: the reader's own procedure, then with the control's leave-one-out removed and with every unit's
   mean, to see how far a plausible bug moves it; and a wider null (half the controls relabelled, every gated
   statistic read by readlib.read);
6. the gates on every neuron of the run (re-asked included), to see whether excluding the 64 matters;
7. decoder check for the 32-piece reading: the tokenizers decode of an uncut answer against the text the writer stored.
Writes reading_<layer>.json here."""
import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
P = T / 'prereg'
sys.path.insert(0, str(P))
import readlib as R  # noqa: E402
import trial_read as TR  # noqa: E402
from backend import census as cz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--verdict', required=True)
ap.add_argument('--fresh', default='')
ap.add_argument('--wide-null', type=int, default=10)
ap.add_argument('--out', required=True)
a = ap.parse_args()

t0 = time.time()
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(TR.TOK)
run = Path(a.run)
meta = json.loads(cz.meta_path_of(run).read_text())
V = json.loads(Path(a.verdict).read_text())
qs = list(meta['prompts'])
A, B = qs[:3], qs[3:6]
doses = [-2.0, -1.0, 1.0, 2.0]
keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
fresh = cz.parse_neurons(Path(a.fresh).read_text().strip(), int(meta['d_ff'])) if a.fresh else None
fresh = set(fresh) if fresh is not None else None
out = {'run': run.name, 'halves': [A, B], 'verdict_halves': [V['half_a'], V['half_b']]}

rows = TR.load(run, 48, tok)
out['rows'] = len(rows)
cut_by_load = sum(1 for r in rows if r['finish'] == 'length' and len(r['token_ids']) < 48)
out['rows_shorter_than_48_marked_length'] = cut_by_load

# 1. the pre-registered reading, again
fig, stat, named, per = R.read(rows, keep, A, B, units_only=fresh, null_splits=50)
vf = V['readings']['48']['figures']
diffs = {k: round(fig[k]['auc'] - vf[k]['auc'], 6) for k in fig if isinstance(fig[k], dict) and 'auc' in fig[k]}
out['rerun_vs_verdict'] = {'max_abs_auc_diff': max(abs(x) for x in diffs.values()), 'null': fig['null_split_auc'],
                           'verdict_null': vf['null_split_auc'], 'counts_same': fig['counts'] == vf['counts']}
print('rerun done', round(time.time() - t0, 1), out['rerun_vs_verdict'], flush=True)


# 2. unrounded gated AUCs and SEs
def hm_se(A_, n1, n2):
    q1, q2 = A_ / (2 - A_), 2 * A_ * A_ / (1 + A_)
    return math.sqrt((A_ * (1 - A_) + (n1 - 1) * (q1 - A_ * A_) + (n2 - 1) * (q2 - A_ * A_)) / (n1 * n2))


gated = ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res', 'shift_up_res', 'cov_lp', 'cov_lost', 'cov_d4']
out['gated_unrounded'] = {}
for k in gated:
    n, c = R.split(stat[k])
    au = R.auc(n, c)[0]
    out['gated_unrounded'][k] = {'auc': au, 'se_hanley_mcneil': round(hm_se(au, len(n), len(c)), 4), 'n_neurons': len(n),
                                 'n_controls': len(c)}

# 3. an independent implementation from the pre-registration's words
base = {}
for r in rows:
    if r['kind'] == 'baseline':
        base.setdefault(r['prompt'], r)
bw = {q: cz.words(base[q]['text']) for q in base}
ans = collections.defaultdict(list)  # unit -> [(prompt, words-own-baseline, row)]
for r in rows:
    if r['kind'] == 'baseline' or not keep(r):
        continue
    ans[(r['kind'], int(r['neuron']))].append((r['prompt'], frozenset(cz.words(r['text']) - bw[r['prompt']]), r))
units = sorted(ans)
ctl = [u for u in units if u[0] == 'random']
C = len(ctl)


def cnts(qset):
    return {u: collections.Counter(w for p, ws, _ in ans[u] if p in qset for w in ws) for u in units}


def zfun(cn, tot):
    def z(u, w):
        c = cn[u][w]
        mu = (tot[w] - c) / (C - 1) if u[0] == 'random' else tot[w] / C
        return (c - mu) / math.sqrt(mu + 0.5)
    return z


def tops(cn):
    tot = collections.Counter()
    for u in ctl:
        tot.update(cn[u])
    z = zfun(cn, tot)
    t = {}
    for u in units:
        zs = sorted(((z(u, w), w) for w, c in cn[u].items() if c >= 2), reverse=True)[:5]
        t[u] = zs
    return t, z


cAll, cA_, cB_ = cnts(set(A) | set(B)), cnts(set(A)), cnts(set(B))
tAll, _ = tops(cAll)
tA, zA = tops(cA_)
tB, zB = tops(cB_)
mine = {'mass_c': {u: sum(max(z, 0) for z, _ in tAll[u]) for u in units},
        'mass_B': {u: sum(max(z, 0) for z, _ in tB[u]) for u in units},
        'rep_AB': {u: sum(zB(u, w) for z, w in tA[u] if z > 0) for u in units},
        'rep_BA': {u: sum(zA(u, w) for z, w in tB[u] if z > 0) for u in units}}
cov = {}
for u in units:
    lnw, d4, cut, lp, dep, lost = [], [], [], [], [], []
    for p, ws, r in ans[u]:
        b = base[p]
        lnw.append(len(ws))
        d4.append(cz.distinct4(r['token_ids']))
        cut.append(1.0 if r['finish'] == 'length' else 0.0)
        lp.append(float(np.mean(r['lp'])) if r['lp'] else 0.0)
        d = cz.first_departure(r['token_ids'], b['token_ids'])
        dep.append((len(r['token_ids']) if d is None else d) / max(len(r['token_ids']), 1))
        lost.append(len(bw[p] - cz.words(r['text'])) / max(len(bw[p]), 1))
    cov[u] = [1.0, math.log1p(np.mean(lnw)), np.mean(d4), np.mean(cut), np.mean(lp), np.mean(dep), np.mean(lost)]
X = np.array([cov[u] for u in ctl])
beta = np.linalg.lstsq(X, np.array([mine['mass_c'][u] for u in ctl]), rcond=None)[0]
mine['mass_c_res'] = {u: mine['mass_c'][u] - float(np.dot(cov[u], beta)) for u in units}


def auc_rank(d):
    n = [v for u, v in d.items() if u[0] == 'neuron' and (fresh is None or u[1] in fresh)]
    c = [v for u, v in d.items() if u[0] == 'random']
    rk = stats.rankdata(n + c)
    return (rk[:len(n)].sum() - len(n) * (len(n) + 1) / 2) / (len(n) * len(c))


out['independent'] = {}
for k, d in mine.items():
    ref = {u: v for u, v in stat[k].items()}
    common = [u for u in d if u in ref]
    out['independent'][k] = {'auc': round(auc_rank(d), 4), 'reader_auc': fig[k]['auc'],
                             'max_abs_unit_diff_vs_readlib': float(max(abs(d[u] - ref[u]) for u in common))}
print('independent done', round(time.time() - t0, 1), json.dumps(out['independent']), flush=True)

# 4. G5b in detail
up = cz.units_path_of(run)
uf = {(u['kind'], int(u['neuron'])): u for u in cz.read_rows(up)}
top = per['top']
d_raw = [abs(uf[u]['mass'] - top[u][0]) for u in top]
d_rnd = [abs(uf[u]['mass'] - round(top[u][0], 3)) for u in top]
w_same = sum(1 for u in top if [w for w, _, _ in uf[u]['words'][:5]] == [w for _, w in top[u][1]])
m_u = R.mass_from_counts(R.counts(rows, keep, 'all'), 'units')
m_own_units = R.mass_from_counts(R.counts(rows, keep, 'own'), 'units')
m_all_ctl = R.mass_from_counts(R.counts(rows, keep, 'all'), 'controls')
ufm = {u: v['mass'] for u, v in uf.items()}
out['G5b'] = {'units_in_file': len(uf), 'units_compared': len(top), 'missing_from_file': len([u for u in top if u not in uf]),
              'max_abs_diff_raw': round(max(d_raw), 6), 'max_abs_diff_vs_rounded_3dp': round(max(d_rnd), 6),
              'units_diff_over_0.0005': sum(1 for x in d_raw if x > 0.0005 + 1e-9),
              'top5_words_same_order': w_same,
              'file_mass_auc_fresh_vs_controls': round(auc_rank(ufm), 4),
              'would_it_tell_readings_apart': {
                  'max_abs_diff_vs_old_reading(units ref, every question excluded)': round(max(abs(ufm[u] - m_u[u][0]) for u in top), 3),
                  'max_abs_diff_vs_units_ref_own_excl': round(max(abs(ufm[u] - m_own_units[u][0]) for u in top), 3),
                  'max_abs_diff_vs_controls_ref_every_question_excl': round(max(abs(ufm[u] - m_all_ctl[u][0]) for u in top), 3)}}
print('G5b', json.dumps(out['G5b']), flush=True)

# 5. the null split: the reader's procedure, without the control's leave-one-out, and with every unit's mean
cnt = R.counts(rows, keep, 'own')
ctl_units = [u for u in cnt if u[0] == 'random']


def mass_variant(cn, loo=True, ref='controls'):
    us = list(cn)
    cs = [u for u in us if u[0] == 'random']
    tot = collections.Counter()
    for u in (cs if ref == 'controls' else us):
        tot.update(cn[u])
    res = {}
    for u in us:
        if ref == 'controls':
            l = loo and u[0] == 'random'
            den = len(cs) - 1 if l else len(cs)
        else:
            l, den = False, len(us)
        res[u] = sum(max(z, 0) for z, _ in R._z_top(cn[u], tot, den, l))
    return res


rng = np.random.default_rng(0)
nul = {'reader': [], 'no_loo': [], 'units_ref': []}
for _ in range(50):
    perm = rng.permutation(len(ctl_units))
    fake = {ctl_units[i] for i in perm[: len(perm) // 2]}
    relab = {(('neuron', u[1] + 10_000_000) if u in fake else u): c for u, c in cnt.items() if u[0] == 'random'}
    for k, kw in (('reader', {}), ('no_loo', {'loo': False}), ('units_ref', {'ref': 'units'})):
        m = mass_variant(relab, **kw)
        nul[k].append(R.auc(*R.split(m))[0])
out['null_split'] = {k: {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v)), 4)} for k, v in nul.items()}
# the real comparison under the two faulty variants, for scale
real = {}
for k, kw in (('reader', {}), ('no_loo', {'loo': False}), ('units_ref', {'ref': 'units'})):
    m = mass_variant(cnt, **kw)
    m = {u: v for u, v in m.items() if u[0] == 'random' or fresh is None or u[1] in fresh}
    real[k] = round(R.auc(*R.split(m))[0], 4)
out['null_split']['real_mass_auc_under_each'] = real
print('null', json.dumps(out['null_split']), flush=True)

# a wider null: half the controls relabelled as neurons, the run's own neurons left out, every figure read by readlib
wide = collections.defaultdict(list)
rng2 = np.random.default_rng(12345)
ctl_ids = sorted({int(r['neuron']) for r in rows if r['kind'] == 'random'})
for s in range(a.wide_null):
    perm = rng2.permutation(len(ctl_ids))
    fake = {ctl_ids[i] for i in perm[: len(perm) // 2]}
    nr = []
    for r in rows:
        if r['kind'] == 'neuron':
            continue
        if r['kind'] == 'random' and int(r['neuron']) in fake:
            r2 = dict(r)
            r2['kind'], r2['neuron'] = 'neuron', int(r['neuron']) + 10_000_000
            r2.pop('_w', None)
            nr.append(r2)
        else:
            nr.append(r)
    f2, _, _, _ = R.read(nr, keep, A, B, null_splits=1)
    for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res', 'shift_up', 'xq'):
        wide[k].append(f2[k]['auc'])
    print('wide null split', s, {k: v[-1] for k, v in wide.items()}, round(time.time() - t0, 1), flush=True)
out['wide_null'] = {k: {'mean': round(float(np.mean(v)), 4), 'min': round(float(np.min(v)), 4), 'max': round(float(np.max(v)), 4),
                        'n': len(v)} for k, v in wide.items()}

# 6. the gates on every neuron of the run (re-asked included)
if fresh is not None:
    f_all, _, _, _ = R.read(rows, keep, A, B, units_only=None, null_splits=1)
    out['every_neuron'] = {k: f_all[k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B')}
    out['every_neuron']['n'] = f_all['counts']['neurons']
    print('every neuron', out['every_neuron'], flush=True)

# 7. decoder check, for the 32-piece reading
mis = tot_ = mis_short = short = 0
for r in cz.read_rows(run, tolerate_tail=True):
    ids, fin = r['token_ids'], r['finish']
    body = ids[:-1] if fin == 'stop' else ids
    t = tok.decode(body, skip_special_tokens=True)
    tot_ += 1
    if t != r['text']:
        mis += 1
    if len(ids) <= 32:
        short += 1
        mis_short += t != r['text']
out['decoder'] = {'rows': tot_, 'tokenizers_decode_ne_stored_text': mis, 'rows_32_or_fewer_pieces': short,
                  'of_which_mismatch': mis_short}
print('decoder', out['decoder'], flush=True)
out['seconds'] = round(time.time() - t0, 1)
Path(a.out).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
