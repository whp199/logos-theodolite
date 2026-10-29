"""An independent implementation of the break_rel reading, written from the pre-registration's words (not from
readlib): CPU only, reads only. Nothing from readlib or trial_read is imported here; census.py is not imported either
(words, distinct-4 and first departure are re-implemented from their one-line definitions).

Definitions (pre-registration, trial_read docstring, the task text):
- a word: a run of 3+ letters (any script), lower-cased, counted once per answer;
- an answer's words leave out its own question's unpushed answer's words;
- mass_c: per unit, count of answers holding each word over the kept answers; z = (c - mu) / sqrt(mu + 0.5), mu the
  controls' mean count (a control left out of its own mean); words met in >= 2 answers; summed positive z of the top 5;
- mass_A / mass_B: the same on one half's answers alone;
- rep_AB: the summed half-B z (no min count) of the unit's positive top-5 half-A words; rep_BA the reverse;
- covariates: log1p(mean new words), mean distinct-4, share cut off, mean log-prob, mean first departure (share of the
  answer's length), mean share of the own unpushed words lost; residual = stat minus its OLS fit on the controls;
- AUC: Mann-Whitney, neurons over controls, ties counted half.
"""
import collections
import json
import math
import re

import numpy as np

TOK_PATH = ('/home/eric/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/'
            '52f3f65bc7a02d555763bc923bd1d9094898219d/tokenizer.json')
WORD = re.compile(r"[^\W\d_]{3,}")
COVS = ('lnw', 'd4', 'cut', 'lp', 'dep', 'lost')
A_HALF = ('who', 'orange', 'free')
B_HALF = ('bicycle', 'walk', 'chair')
DOSES = (-2.0, -1.0, 1.0, 2.0)


def words(text):
    return frozenset(WORD.findall(text.lower()))


def distinct4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 1.0


def departure(ids, base):
    for i, (a, b) in enumerate(zip(ids, base)):
        if a != b:
            return i
    return None if len(ids) == len(base) else min(len(ids), len(base))


def load(path, cut=None, tok=None):
    """Rows with the fields the reading needs. With `cut`, an answer longer than `cut` pieces is cut to its first `cut`
    pieces, its text decoded from them, marked cut off (finish 'length'); the stop piece is not text."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids, lp, text, fin = r['token_ids'], r['logprob'], r['text'], r['finish']
            if cut is not None and len(ids) > cut:
                body = ids[:-1] if fin == 'stop' else ids
                text = tok.decode(body[:cut], skip_special_tokens=True)
                ids, lp, fin = ids[:cut], lp[:cut], 'length'
            out.append({'kind': r['kind'], 'neuron': int(r['neuron']), 'prompt': r['prompt'], 'dose': float(r['dose']),
                        'rel_push': float(r['rel_push']), 'push': r.get('push'), 'seed': r.get('seed'), 'finish': fin,
                        'text': text, 'ids': ids, 'lp': lp, 'w': words(text)})
    return out


def auc(n, c):
    n, c = np.asarray(n, float), np.asarray(c, float)
    allv = np.concatenate([n, c])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(n)].sum() - len(n) * (len(n) + 1) / 2) / (len(n) * len(c)))


def hanley_mcneil(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


class Reading:
    """Per-unit answers (kept rows only), with their new-word sets, ready for any statistic."""

    def __init__(self, rows, doses=DOSES, qs=A_HALF + B_HALF):
        self.base = {}
        for r in rows:
            if r['kind'] == 'baseline':
                self.base.setdefault(r['prompt'], r)
        self.ans = collections.defaultdict(list)
        self.order = []  # units in order of first appearance in the file
        seen = set()
        for r in rows:
            if r['kind'] == 'baseline':
                continue
            u = (r['kind'], r['neuron'])
            if u not in seen:
                seen.add(u)
                self.order.append(u)
            if r['prompt'] in qs and any(abs(r['dose'] - d) < 1e-9 for d in doses):
                self.ans[u].append(r)
        self.units = list(self.order)

    def counts(self, qset=None, exclude='own'):
        allb = frozenset().union(*(b['w'] for b in self.base.values()))
        out = {}
        for u in self.units:
            c = collections.Counter()
            for r in self.ans[u]:
                if qset is not None and r['prompt'] not in qset:
                    continue
                c.update(r['w'] - (self.base[r['prompt']]['w'] if exclude == 'own' else allb))
            out[u] = c
        return out


def ref_totals(cnt, pool):
    t = collections.Counter()
    for u in pool:
        t.update(cnt[u])
    return t


def tops(cnt, ref='controls', loo=True, k=5, tiebreak='readlib'):
    """{unit: [(z, word)] top k among words met >= 2 times}. ref 'controls' (a control left out of its own mean when
    loo) or 'units' (every unit's mean, itself included)."""
    units = list(cnt)
    ctl = [u for u in units if u[0] == 'random']
    pool = ctl if ref == 'controls' else units
    tot = ref_totals(cnt, pool)
    out = {}
    for u in units:
        if ref == 'controls':
            l = loo and u[0] == 'random'
            den = len(ctl) - 1 if l else len(ctl)
        else:
            l, den = False, len(units)
        zs = []
        for w, c in cnt[u].items():
            if c < 2:
                continue
            mu = (tot[w] - (c if l else 0)) / den
            zs.append(((c - mu) / math.sqrt(mu + 0.5), w))
        if tiebreak == 'readlib':
            zs.sort(reverse=True)  # z then word, both descending
        else:
            zs.sort(key=lambda t: (-t[0], t[1]))  # z descending, word ascending
        out[u] = zs[:k]
    return out


def mass(tp):
    return {u: sum(max(z, 0.0) for z, _ in t) for u, t in tp.items()}


def zsum(cnt, picked):
    """{unit: summed z (control-referenced, a control left out) of its picked words on these counts}."""
    ctl = [u for u in cnt if u[0] == 'random']
    tot = ref_totals(cnt, ctl)
    C = len(ctl)
    out = {}
    for u, ws in picked.items():
        l = u[0] == 'random'
        den = C - 1 if l else C
        s = 0.0
        for w in ws:
            c = cnt[u][w]
            mu = (tot[w] - (c if l else 0)) / den
            s += (c - mu) / math.sqrt(mu + 0.5)
        out[u] = s
    return out


def covariates(rd):
    cov = {}
    for u in rd.units:
        acc = collections.defaultdict(list)
        for r in rd.ans[u]:
            b = rd.base[r['prompt']]
            ids = r['ids']
            acc['lnw'].append(len(r['w'] - b['w']))
            acc['d4'].append(distinct4(ids))
            acc['cut'].append(1.0 if r['finish'] == 'length' else 0.0)
            acc['lp'].append(float(np.mean(r['lp'])) if r['lp'] else 0.0)
            d = departure(ids, b['ids'])
            acc['dep'].append((len(ids) if d is None else d) / max(len(ids), 1))
            acc['lost'].append(len(b['w'] - r['w']) / max(len(b['w']), 1))
        cov[u] = {k: float(np.mean(v)) for k, v in acc.items()}
        cov[u]['lnw'] = math.log1p(cov[u]['lnw'])
    return cov


def resid(stat, cov, covs=COVS):
    ctl = [u for u in stat if u[0] == 'random']
    X = np.array([[1.0] + [cov[u][c] for c in covs] for u in ctl])
    y = np.array([stat[u] for u in ctl])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return {u: stat[u] - float(np.dot([1.0] + [cov[u][c] for c in covs], beta)) for u in stat}


def statistics(rd, tiebreak='readlib'):
    A, B = set(A_HALF), set(B_HALF)
    c_all, c_A, c_B = rd.counts(), rd.counts(A), rd.counts(B)
    t_all, t_A, t_B = tops(c_all, tiebreak=tiebreak), tops(c_A, tiebreak=tiebreak), tops(c_B, tiebreak=tiebreak)
    st = {'mass_c': mass(t_all), 'mass_A': mass(t_A), 'mass_B': mass(t_B)}
    st['rep_AB'] = zsum(c_B, {u: [w for z, w in t_A[u] if z > 0] for u in rd.units})
    st['rep_BA'] = zsum(c_A, {u: [w for z, w in t_B[u] if z > 0] for u in rd.units})
    st['rep'] = {u: st['rep_AB'][u] + st['rep_BA'][u] for u in rd.units}
    cov = covariates(rd)
    for c in COVS:
        st['cov_' + c] = {u: cov[u][c] for u in rd.units}
    st['mass_c_res'] = resid(st['mass_c'], cov)
    st['rep_res'] = resid(st['rep'], cov)
    return st, cov, {'all': t_all, 'A': t_A, 'B': t_B}, c_all


def aucs(st, neurons=None, keys=None):
    out = {}
    for k in keys or st:
        d = st[k]
        n = [v for u, v in d.items() if u[0] == 'neuron' and (neurons is None or u[1] in neurons)]
        c = [v for u, v in d.items() if u[0] == 'random']
        out[k] = {'auc': auc(n, c), 'n': len(n), 'c': len(c)}
    return out
