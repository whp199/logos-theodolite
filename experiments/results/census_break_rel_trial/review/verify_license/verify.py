"""Independent re-derivation of the license reviewer's figures on the break_rel trial (CPU only; reads the runs through
the pre-registered loader and readlib's statistics; p-values, BH, AUC, bootstraps, nulls, splits are my own code here).
Writes only into this folder.  Usage: verify.py L45:48 L45:32 L57:48 L57:32"""
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

PRE = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
sys.path.insert(0, str(PRE))
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from backend import census as cz  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
OUT = Path(__file__).resolve().parent
NB = 2000


def ids(f):
    return {int(x) for x in Path(f).read_text().replace('\n', ',').split(',') if x.strip()}


# ---------------------------------------------------------------- my own primitives
def my_auc(n, c):
    n, c = np.asarray(n, float), np.asarray(c, float)
    r = stats.rankdata(np.concatenate([n, c]))
    return float((r[:len(n)].sum() - len(n) * (len(n) + 1) / 2) / (len(n) * len(c)))


def my_p(d):
    """neuron: (1 + #controls >= v) / (C + 1); control: leave-one-out, (#controls >= v incl. itself) / C."""
    ctl = np.sort(np.array([v for u, v in d.items() if u[0] == 'random'], float))
    C = len(ctl)
    out = {}
    for u, v in d.items():
        ge = C - int(np.searchsorted(ctl, v, side='left'))
        out[u] = (1 + ge) / (C + 1) if u[0] == 'neuron' else ge / C
    return out


def my_bh(p):
    keys = sorted(p, key=lambda k: p[k])
    n = len(keys)
    pv = np.array([p[k] for k in keys])
    q = np.minimum.accumulate((pv * n / np.arange(1, n + 1))[::-1])[::-1]
    return dict(zip(keys, np.minimum(q, 1.0)))


def spear(x, y):
    return float(stats.spearmanr(x, y)[0])


def boot_spear(x, y, rng, n=NB):
    x, y = np.asarray(x), np.asarray(y)
    k = len(x)
    return np.array([spear(x[j], y[j]) for j in (rng.integers(0, k, k) for _ in range(n))])


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def ols_resid(stat, cov, fit_units):
    """stat minus OLS fit on the six covariates, fitted on fit_units (a list, may repeat for bootstrap)."""
    X = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in fit_units])
    y = np.array([stat[u] for u in fit_units])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return {u: stat[u] - float(np.dot([1.0] + [cov[u][c] for c in R.COVS], beta)) for u in stat}


def need(N, C, q=0.05):
    return math.ceil(N / ((C + 1) * q))


# ---------------------------------------------------------------- the reading
def setup(tag, cut, tok):
    path = TR / ('L45-break_rel-trial.jsonl' if tag == 'L45' else 'L57-break_rel-probe.jsonl')
    meta = json.loads(Path(str(path)[:-6] + '.meta.json').read_text())
    asked = [q for q in meta['prompts'] if any(p == q for p, _ in meta['design'])]
    A, B = asked[:3], asked[3:6]
    doses = (-2.0, -1.0, 1.0, 2.0)
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
    kA = lambda r: keep(r) and r['prompt'] in set(A)  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in set(B)  # noqa: E731
    rows = T.load(path, cut, tok)
    return meta, A, B, keep, kA, kB, rows


def statistics(cnt, cA, cB, cov):
    m_c = R.mass_from_counts(cnt, 'controls')
    mA, mB = R.mass_from_counts(cA, 'controls'), R.mass_from_counts(cB, 'controls')
    repAB = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    repBA = R.z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    st = {'mass_c': {u: v[0] for u, v in m_c.items()}, 'mass_A': {u: v[0] for u, v in mA.items()},
          'mass_B': {u: v[0] for u, v in mB.items()}, 'rep_AB': repAB, 'rep_BA': repBA,
          'rep': {u: repAB[u] + repBA[u] for u in repAB}}
    ctl = [u for u in st['mass_c'] if u[0] == 'random']
    for k in ('mass_c', 'rep', 'rep_AB', 'rep_BA'):
        st[k + '_res'] = ols_resid(st[k], cov, ctl)
    return st, m_c, mA, mB


def restrict(st, only):
    if only is None:
        return st
    return {k: {u: v for u, v in d.items() if u[0] == 'random' or u[1] in only} for k, d in st.items()}


def naming(st):
    p = {k: my_p(st[k]) for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA')}
    q = my_bh({u: v for u, v in p['mass_c'].items() if u[0] == 'neuron'})
    q05 = [u for u in q if q[u] <= 0.05]
    named = [u for u in q05 if p['rep_AB'][u] <= 0.05 and p['rep_BA'][u] <= 0.05 and p['mass_c_res'][u] <= 0.05]
    neu = np.array([v for u, v in st['mass_c'].items() if u[0] == 'neuron'])
    ctl = np.array([v for u, v in st['mass_c'].items() if u[0] == 'random'])
    above = [u for u, v in st['mass_c'].items() if u[0] == 'neuron' and v > ctl.max()]
    thr = max((p['mass_c'][u] for u in q05), default=None)
    cu = [u for u in st['mass_c'] if u[0] == 'random']
    c_mass = [u for u in cu if thr is not None and p['mass_c'][u] <= thr]
    c_named = [u for u in c_mass if p['rep_AB'][u] <= 0.05 and p['rep_BA'][u] <= 0.05 and p['mass_c_res'][u] <= 0.05]
    return {'N': len(neu), 'C': len(ctl), 'q05': q05, 'named': named, 'above': above, 'thr': thr,
            'ctl_mass_at_thr': len(c_mass), 'ctl_named_at_thr': len(c_named), 'p': p, 'q': q,
            'mass_auc': my_auc(neu, ctl)}


def relabel(d, fake):
    return {(('neuron', u[1] + 10_000_000) if u in fake else u): v for u, v in d.items() if u[0] == 'random'}


def boot_auc(d, rng, n=NB):
    neu = np.array([v for u, v in d.items() if u[0] == 'neuron'])
    ctl = np.array([v for u, v in d.items() if u[0] == 'random'])
    bs = np.array([my_auc(neu[rng.integers(0, len(neu), len(neu))], ctl[rng.integers(0, len(ctl), len(ctl))]) for _ in range(n)])
    return {'auc': round(my_auc(neu, ctl), 4), 'se': round(float(bs.std(ddof=1)), 4), 'ci': ci(bs)}


def boot_auc_refit(stat, cov, rng, n=NB):
    """AUC of the residual, the OLS refit on each bootstrap sample of the controls (the fit's own noise included)."""
    neu = [u for u in stat if u[0] == 'neuron']
    ctl = [u for u in stat if u[0] == 'random']
    bs = []
    for _ in range(n):
        nb = [neu[i] for i in rng.integers(0, len(neu), len(neu))]
        cb = [ctl[i] for i in rng.integers(0, len(ctl), len(ctl))]
        X = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in cb])
        y = np.array([stat[u] for u in cb])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        f = lambda u: stat[u] - float(np.dot([1.0] + [cov[u][c] for c in R.COVS], beta))  # noqa: E731
        bs.append(my_auc([f(u) for u in nb], [f(u) for u in cb]))
    bs = np.array(bs)
    return {'se': round(float(bs.std(ddof=1)), 4), 'ci': ci(bs)}


def hanley_se(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def sel_rep(sa, sb, rule_b=0.05):
    pa, pb = my_p(sa), my_p(sb)
    qa = my_bh({u: v for u, v in pa.items() if u[0] == 'neuron'})
    disc = [u for u in qa if qa[u] <= 0.05]
    thr = max((pa[u] for u in disc), default=None)
    cdisc = [u for u, v in pa.items() if u[0] == 'random' and thr is not None and v <= thr]
    # a larger control baseline: the top 5% of controls on the selecting half (how often a strong control replicates)
    cs = sorted([u for u in pa if u[0] == 'random'], key=lambda u: -sa[u])[:max(1, len([u for u in pa if u[0] == 'random']) // 20)]
    return {'discovered': len(disc), 'thr': thr, 'replicate': int(sum(pb[u] <= rule_b for u in disc)),
            'share': round(float(np.mean([pb[u] <= rule_b for u in disc])), 3) if disc else None,
            'ctl_at_thr': len(cdisc), 'ctl_replicate': int(sum(pb[u] <= rule_b for u in cdisc)),
            'ctl_top5pct': len(cs), 'ctl_top5pct_replicate': int(sum(pb[u] <= rule_b for u in cs))}


def run(tag, cut, tok, verdict, extra=True):
    t0 = time.time()
    fresh = ids(PRE / 'trial_L45_fresh.txt') if tag == 'L45' else None
    reasked = ids(PRE / 'trial_L45_reasked.txt') if tag == 'L45' else None
    meta, A, B, keep, kA, kB, rows = setup(tag, cut, tok)
    cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    cov = R.covariates(rows, keep)
    st_all, m_c, mA, mB = statistics(cnt, cA, cB, cov)
    st = restrict(st_all, fresh)
    nm = naming(st)
    vr = verdict['readings'][str(cut)]
    out = {'tag': tag, 'cut': cut, 'A': A, 'B': B, 'N': nm['N'], 'C': nm['C']}
    out['gates_auc'] = {k: round(my_auc(*R.split(st[k])), 4) for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'mass_A', 'rep_res', 'rep_AB_res', 'rep_BA_res')}
    out['verdict_auc'] = {k: vr['figures'][k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'mass_A', 'rep_res')}
    out['counts'] = {'q05': len(nm['q05']), 'named': len(nm['named']), 'above_every_control': len(nm['above']),
                     'named_matches_verdict': sorted(u[1] for u in nm['named']) == sorted(vr['named']),
                     'verdict_counts': vr['figures']['counts'], 'bh_thr_p': nm['thr'],
                     'q05_equals_above_every_control': set(nm['q05']) == set(nm['above']),
                     'need_floor': need(nm['N'], nm['C']), 'chance_above_every_control': round(nm['N'] / (nm['C'] + 1), 2)}
    out['empirical_null'] = {'ctl_pass_mass_at_thr': nm['ctl_mass_at_thr'], 'ctl_named_at_thr': nm['ctl_named_at_thr'],
                             'false_q05_if_all_null': round(nm['N'] * nm['ctl_mass_at_thr'] / nm['C'], 2),
                             'false_named_if_all_null': round(nm['N'] * nm['ctl_named_at_thr'] / nm['C'], 2),
                             'share_of_named': round(nm['N'] * nm['ctl_named_at_thr'] / nm['C'] / max(len(nm['named']), 1), 4),
                             # Poisson 95% upper bound on the 7 (or k) controls, scaled
                             'false_named_poisson_upper95': round(nm['N'] / nm['C'] * stats.chi2.ppf(0.975, 2 * (nm['ctl_named_at_thr'] + 1)) / 2, 1)}
    rng = np.random.default_rng(2026)
    # AUC bootstraps
    out['boot'] = {k: boot_auc(st[k], rng) for k in ('mass_c', 'mass_c_res', 'rep_res', 'rep_AB_res', 'rep_BA_res')}
    out['boot']['mass_c_res_refit'] = boot_auc_refit(restrict({'x': st_all['mass_c']}, fresh)['x'], cov, rng)
    out['boot']['rep_res_refit'] = boot_auc_refit(restrict({'x': st_all['rep']}, fresh)['x'], cov, rng)
    a = out['gates_auc']['mass_c_res']
    out['boot']['mass_c_res_hanley_se'] = round(hanley_se(a, nm['N'], nm['C']), 4)
    # rho_AB
    nk = [u for u in st['mass_A'] if u[0] == 'neuron']
    ck = [u for u in st['mass_A'] if u[0] == 'random']
    xn, yn = [st['mass_A'][u] for u in nk], [st['mass_B'][u] for u in nk]
    xc, yc = [st['mass_A'][u] for u in ck], [st['mass_B'][u] for u in ck]
    bn, bc = boot_spear(xn, yn, rng), boot_spear(xc, yc, rng)
    out['rho_AB'] = {'neurons': round(spear(xn, yn), 3), 'neurons_ci': ci(bn), 'controls': round(spear(xc, yc), 3),
                     'controls_ci': ci(bc), 'diff': round(spear(xn, yn) - spear(xc, yc), 3), 'diff_ci': ci(bn - bc),
                     'verdict': [round(vr['figures']['rho_AB_neurons'], 3), round(vr['figures']['rho_AB_controls'], 3)]}
    covA, covB = R.covariates(rows, kA), R.covariates(rows, kB)
    ctl_all = [u for u in st_all['mass_A'] if u[0] == 'random']
    resA, resB = ols_resid(st_all['mass_A'], covA, ctl_all), ols_resid(st_all['mass_B'], covB, ctl_all)
    rn = spear([resA[u] for u in nk], [resB[u] for u in nk])
    rc = spear([resA[u] for u in ck], [resB[u] for u in ck])
    out['rho_AB_res_per_half'] = {'neurons': round(rn, 3), 'neurons_ci': ci(boot_spear([resA[u] for u in nk], [resB[u] for u in nk], rng)),
                                  'controls': round(rc, 3), 'controls_ci': ci(boot_spear([resA[u] for u in ck], [resB[u] for u in ck], rng))}
    # tail / bulk on half A only
    p95A = float(np.percentile(xc, 95))
    tail = [u for u in nk if st['mass_A'][u] >= p95A]
    bulk = [u for u in nk if st['mass_A'][u] < p95A]
    bulk_c = [u for u in ck if st['mass_A'][u] < p95A]
    tail_c = [u for u in ck if st['mass_A'][u] >= p95A]
    f = lambda us: ([st['mass_A'][u] for u in us], [st['mass_B'][u] for u in us])  # noqa: E731
    bt, bb, bbc = boot_spear(*f(tail), rng), boot_spear(*f(bulk), rng), boot_spear(*f(bulk_c), rng)
    out['tail_bulk'] = {'p95A': round(p95A, 3), 'tail_n': len(tail), 'tail_rho': round(spear(*f(tail)), 3), 'tail_ci': ci(bt),
                        'bulk_n': len(bulk), 'bulk_rho': round(spear(*f(bulk)), 3), 'bulk_ci': ci(bb),
                        'bulk_ctl_n': len(bulk_c), 'bulk_ctl_rho': round(spear(*f(bulk_c)), 3), 'bulk_ctl_ci': ci(bbc),
                        'bulk_minus_ctl_ci': ci(bb - bbc), 'tail_ctl_n': len(tail_c),
                        # do bulk neurons still lead the bulk controls on half B (a group lead without an order)?
                        'bulk_massB_auc_vs_bulk_ctl': round(my_auc([st['mass_B'][u] for u in bulk], [st['mass_B'][u] for u in bulk_c]), 4),
                        'bulk_massA_auc_vs_bulk_ctl': round(my_auc([st['mass_A'][u] for u in bulk], [st['mass_A'][u] for u in bulk_c]), 4)}
    # select on one half, test on the other
    rr = restrict({'A': resA, 'B': resB}, fresh)
    out['select'] = {'mass A->B': sel_rep(st['mass_A'], st['mass_B']), 'mass B->A': sel_rep(st['mass_B'], st['mass_A']),
                     'res A->B': sel_rep(rr['A'], rr['B']), 'res B->A': sel_rep(rr['B'], rr['A']),
                     'words A->B (rep_AB)': sel_rep(st['mass_A'], st['rep_AB']), 'words B->A (rep_BA)': sel_rep(st['mass_B'], st['rep_BA'])}
    if reasked:
        rk = [u for u in st_all['mass_A'] if u[0] == 'neuron' and u[1] in reasked]
        x, y = [st_all['mass_A'][u] for u in rk], [st_all['mass_B'][u] for u in rk]
        out['rho_reasked64'] = {'n': len(rk), 'rho': round(spear(x, y), 3), 'ci': ci(boot_spear(x, y, rng))}
    # calibration-sized halves inside half A
    cal = []
    for held in A:
        a2 = [q for q in A if q != held]
        k1 = lambda r, a2=a2: keep(r) and r['prompt'] in set(a2)  # noqa: E731
        k2 = lambda r, h=held: keep(r) and r['prompt'] == h  # noqa: E731
        m1 = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k1, 'own'), 'controls').items()}
        m2 = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k2, 'own'), 'controls').items()}
        row = [held, spear([m1[u] for u in nk], [m2[u] for u in nk]), spear([m1[u] for u in ck], [m2[u] for u in ck])]
        if reasked:
            rk = [u for u in m1 if u[0] == 'neuron' and u[1] in reasked]
            row.append(spear([m1[u] for u in rk], [m2[u] for u in rk]))
        cal.append(row)
    out['calib_sized_halves'] = {'splits': [[c[0]] + [round(v, 3) for v in c[1:]] for c in cal],
                                 'mean_neurons': round(float(np.mean([c[1] for c in cal])), 3),
                                 'mean_controls': round(float(np.mean([c[2] for c in cal])), 3)}
    if reasked:
        out['calib_sized_halves']['mean_reasked'] = round(float(np.mean([c[3] for c in cal])), 3)
    # covariate range
    lo = {k: min(cov[u][k] for u in ck) for k in R.COVS}
    hi = {k: max(cov[u][k] for u in ck) for k in R.COVS}
    oo = lambda u: any(cov[u][k] < lo[k] or cov[u][k] > hi[k] for k in R.COVS)  # noqa: E731
    out['cov_range'] = {'named_out': int(sum(oo(u) for u in nm['named'])), 'named': len(nm['named']),
                        'fresh_share_out': round(float(np.mean([oo(u) for u in nk])), 4),
                        'by_cov_named': {k: int(sum(cov[u][k] < lo[k] or cov[u][k] > hi[k] for u in nm['named'])) for k in R.COVS}}
    # shared vocabulary
    tn, tc = collections.Counter(), collections.Counter()
    for u in nk:
        tn.update({w for z, w in m_c[u][1] if z > 0})
    for u in ck:
        tc.update({w for z, w in m_c[u][1] if z > 0})
    nw = vr['named_words']
    inlist = collections.Counter()
    for k, v in nw.items():
        inlist.update({w for w, _ in v['A_on_B']} | {w for w, _ in v['B_on_A']})
    out['vocab'] = {w: {'named_lists': inlist[w], 'top5_neuron_share': round(tn[w] / len(nk), 3), 'top5_ctl_share': round(tc[w] / len(ck), 3)}
                    for w in ('same', 'own', 'thought', 'thoughtthought')}
    out['vocab_top10'] = [(w, n, round(n / len(nk), 3), round(tc[w] / len(ck), 3)) for w, n in tn.most_common(10)]
    drop = {w for w, n in tn.items() if n >= 0.02 * len(nk)}
    out['n_drop'] = len(drop)
    if extra and tag == 'L45' and cut == 48:
        # relabelled-control null (G5's splits: rng(0) permutations of the controls in counts order)
        r0 = np.random.default_rng(0)
        cu = [u for u in cnt if u[0] == 'random']
        nulls = []
        for _ in range(50):
            perm = r0.permutation(len(cu))
            fake = {cu[i] for i in perm[: len(perm) // 2]}
            s0, *_ = statistics(relabel(cnt, fake), relabel(cA, fake), relabel(cB, fake), relabel(cov, fake))
            n0 = naming(s0)
            nulls.append((len(n0['named']), len(n0['q05']), len(n0['above']), n0['mass_auc']))
        nulls = np.array(nulls)
        out['relabelled_null'] = {'named_nonzero': int((nulls[:, 0] > 0).sum()), 'q05_nonzero': int((nulls[:, 1] > 0).sum()),
                                  'above_every_mean': round(float(nulls[:, 2].mean()), 2), 'above_every_max': int(nulls[:, 2].max()),
                                  'mass_auc_mean': round(float(nulls[:, 3].mean()), 4), 'floor_512_512': need(512, 512)}
        # words dropped everywhere and re-read
        named0 = set(nm['named'])
        for r in rows:
            r['_w'] = frozenset(cz.words(r['text'])) - drop
        cnt2, cA2, cB2 = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
        cov2 = R.covariates(rows, keep)
        st2 = restrict(statistics(cnt2, cA2, cB2, cov2)[0], fresh)
        nm2 = naming(st2)
        out['dropped'] = {'words': sorted(drop), 'named': len(nm2['named']), 'overlap': len(named0 & set(nm2['named'])),
                          'q05': len(nm2['q05']),
                          'auc': {k: round(my_auc(*R.split(st2[k])), 4) for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res')},
                          'select_words_A->B': sel_rep(st2['mass_A'], st2['rep_AB']),
                          'select_mass_A->B': sel_rep(st2['mass_A'], st2['mass_B']),
                          'rho_AB': round(spear([st2['mass_A'][u] for u in nk], [st2['mass_B'][u] for u in nk]), 3),
                          'rho_AB_ctl': round(spear([st2['mass_A'][u] for u in ck], [st2['mass_B'][u] for u in ck]), 3)}
    if extra and tag == 'L57':
        # the above-every-control share against the number of controls (sub-sampled controls, mass fixed): how the
        # BH floor 20/(C+1) compares with the share of neurons above every one of C controls
        cm = np.array([st['mass_c'][u] for u in ck])
        nmv = np.array([st['mass_c'][u] for u in nk])
        r1 = np.random.default_rng(5)
        sub = {}
        for C in (32, 64, 128, 256, 512):
            fr = [float((nmv > cm[r1.choice(len(cm), C, replace=False)].max()).mean()) for _ in range(400 if C < 512 else 1)]
            sub[C] = {'share_above_all': round(float(np.mean(fr)), 4), 'floor_share': round(20 / (C + 1), 4)}
        out['floor_vs_controls'] = sub
    out['seconds'] = round(time.time() - t0, 1)
    return out


if __name__ == '__main__':
    tok = Tokenizer.from_file(T.TOK)
    res = {}
    for w in sys.argv[1:]:
        tag, cut = w.split(':')
        verdict = json.loads((TR / f'{tag}_verdict.json').read_text())
        res[w] = run(tag, int(cut), tok, verdict)
        print(json.dumps(res[w], default=str), flush=True)
        json.dump(res, open(OUT / f'verify_{"_".join(x.replace(":", "-") for x in sys.argv[1:])}.json', 'w'), indent=1, default=str)
