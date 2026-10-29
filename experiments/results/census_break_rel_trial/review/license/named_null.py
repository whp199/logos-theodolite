"""Does the trial's per-neuron figure answer "the gates test a population lead, not whether a neuron's rank reproduces"?
CPU only; reads the runs through the pre-registered reader (trial_read.load, readlib), writes only into this folder.

Per (layer, cut):
  1. the naming rule re-implemented from readlib.read's own lines (mass_c BH q <= 0.05 over the layer's neurons,
     rep_AB p <= 0.05, rep_BA p <= 0.05, mass_c residual p <= 0.05), checked against the verdict's named list;
  2. the null the way readlib's G5 null does it: half the controls relabelled as neurons (rng(0) permutations, the
     same splits as G5), the reference from the other half alone, every figure recomputed, the naming rule applied;
  3. an empirical-null count at full size: controls (leave-one-out p) passing the naming rule at the BH threshold
     the real neurons reached, scaled to the number of neurons;
  4. rho_AB (Spearman of half-A mass with half-B mass) among the fresh neurons and among the controls, a bootstrap CI
     for each and for the difference; the same on each half's mass residualised on that half's own six covariates;
     the 64 re-asked calibration neurons alone; calibration-sized halves (two of who/orange/free against the third);
  5. select-on-one-half, test-on-the-other: neurons discovered on half A (BH q <= 0.05 on mass_A) and the share whose
     half-B p <= 0.05, beside controls passing the same half-A threshold;
  6. AUC bootstrap CIs for the gate statistics and the residualised replication (rep_AB_res, rep_BA_res)."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

PRE = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
sys.path.insert(0, str(PRE))
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

OUT = Path(__file__).resolve().parent
TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
NBOOT = 2000


def ids(file):
    return {int(x) for x in Path(file).read_text().replace('\n', ',').split(',') if x.strip()}


def stats_of(cnt, cA, cB, cov):
    m_c = R.mass_from_counts(cnt, 'controls')
    mA, mB = R.mass_from_counts(cA, 'controls'), R.mass_from_counts(cB, 'controls')
    repAB = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    repBA = R.z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    st = {'mass_c': {u: v[0] for u, v in m_c.items()}, 'mass_A': {u: v[0] for u, v in mA.items()},
          'mass_B': {u: v[0] for u, v in mB.items()}, 'rep_AB': repAB, 'rep_BA': repBA}
    st['mass_c_res'] = R.residualise(st['mass_c'], cov)
    st['rep_AB_res'] = R.residualise(repAB, cov)
    st['rep_BA_res'] = R.residualise(repBA, cov)
    st['rep_res'] = R.residualise({u: repAB[u] + repBA[u] for u in repAB}, cov)
    return st


def restrict(st, only):
    if only is None:
        return st
    return {k: {u: v for u, v in d.items() if u[0] == 'random' or u[1] in only} for k, d in st.items()}


def naming(st):
    p_all, p_r = R.emp_p(st['mass_c']), R.emp_p(st['mass_c_res'])
    p_AB, p_BA = R.emp_p(st['rep_AB']), R.emp_p(st['rep_BA'])
    q = R.bh({u: v for u, v in p_all.items() if u[0] == 'neuron'})
    named = [u for u in q if q[u] <= 0.05 and p_AB[u] <= 0.05 and p_BA[u] <= 0.05 and p_r[u] <= 0.05]
    q05 = [u for u in q if q[u] <= 0.05]
    neu = np.array([v for u, v in st['mass_c'].items() if u[0] == 'neuron'])
    ctl = np.array([v for u, v in st['mass_c'].items() if u[0] == 'random'])
    thr = max((p_all[u] for u in q05), default=None)  # the BH threshold reached
    # empirical null at full size: controls (leave-one-out p) passing the same rule at that threshold
    ctl_units = [u for u in st['mass_c'] if u[0] == 'random']
    if thr is not None:
        c_q = [u for u in ctl_units if p_all[u] <= thr]
        c_named = [u for u in c_q if p_AB[u] <= 0.05 and p_BA[u] <= 0.05 and p_r[u] <= 0.05]
    else:
        c_q, c_named = [], []
    return {'named': named, 'q05': len(q05), 'above_every_control': int((neu > ctl.max()).sum()),
            'bh_threshold_p': thr, 'controls_at_threshold': len(c_q), 'controls_named_at_threshold': len(c_named),
            'N': len(neu), 'C': len(ctl),
            'mass_auc': R.auc(neu, ctl)[0], 'p': {'all': p_all, 'r': p_r, 'AB': p_AB, 'BA': p_BA}}


def relabel(d, fake):
    return {(('neuron', u[1] + 10_000_000) if u in fake else u): v for u, v in d.items() if u[0] == 'random'}


def spearman(x, y):
    return float(stats.spearmanr(x, y)[0])


def boot_rho(x, y, rng, n=NBOOT):
    x, y = np.asarray(x), np.asarray(y)
    k = len(x)
    out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, k, k)
        out[i] = spearman(x[j], y[j])
    return out


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def rho_block(sa, sb, keys_n, keys_c, rng):
    xn, yn = [sa[u] for u in keys_n], [sb[u] for u in keys_n]
    xc, yc = [sa[u] for u in keys_c], [sb[u] for u in keys_c]
    rn, rc = spearman(xn, yn), spearman(xc, yc)
    bn, bc = boot_rho(xn, yn, rng), boot_rho(xc, yc, rng)
    return {'neurons': round(rn, 3), 'neurons_ci': ci(bn), 'controls': round(rc, 3), 'controls_ci': ci(bc),
            'diff': round(rn - rc, 3), 'diff_ci': ci(bn - bc), 'n_neurons': len(xn), 'n_controls': len(xc)}


def boot_auc(st_key_d, rng, n=NBOOT):
    neu = np.array([v for u, v in st_key_d.items() if u[0] == 'neuron'])
    ctl = np.array([v for u, v in st_key_d.items() if u[0] == 'random'])
    a = R.auc(neu, ctl)[0]
    bs = np.empty(n)
    for i in range(n):
        x = neu[rng.integers(0, len(neu), len(neu))]
        y = ctl[rng.integers(0, len(ctl), len(ctl))]
        # AUC by ranks (ties half)
        r = stats.rankdata(np.concatenate([x, y]))
        bs[i] = (r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    return {'auc': round(a, 4), 'se': round(float(bs.std(ddof=1)), 4), 'ci': ci(bs)}


def run(layer_tag, path, cut, fresh, reasked, null_splits=50):
    t0 = time.time()
    tok = Tokenizer.from_file(T.TOK)
    meta = json.loads((Path(str(path)[:-len('.jsonl')] + '.meta.json')).read_text())
    qs = list(meta.get('prompts') or {})
    asked = [q for q in qs if any(p == q for p, _ in meta['design'])]
    A, B = asked[:3], asked[3:6]
    doses = [-2.0, -1.0, 1.0, 2.0]
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
    kA = lambda r: keep(r) and r['prompt'] in set(A)  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in set(B)  # noqa: E731
    rows = T.load(path, cut, tok)
    cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    cov = R.covariates(rows, keep)
    covA, covB = R.covariates(rows, kA), R.covariates(rows, kB)
    st_all = stats_of(cnt, cA, cB, cov)
    st = restrict(st_all, fresh)
    nm = naming(st)
    verdict = json.loads((TR / f'{layer_tag}_verdict.json').read_text())
    vnamed = sorted(verdict['readings'][str(cut)]['named'])
    out = {'layer': layer_tag, 'cut': cut, 'A': A, 'B': B, 'named': len(nm['named']), 'q05': nm['q05'],
           'above_every_control': nm['above_every_control'], 'N': nm['N'], 'C': nm['C'],
           'matches_verdict_named_list': sorted(u[1] for u in nm['named']) == vnamed,
           'verdict_named': len(vnamed), 'bh_threshold_p': nm['bh_threshold_p'],
           'empirical_null': {'controls_passing_mass_at_threshold': nm['controls_at_threshold'],
                              'controls_named_at_threshold': nm['controls_named_at_threshold'],
                              'expected_false_q05_if_all_null': round(nm['N'] * nm['controls_at_threshold'] / nm['C'], 2),
                              'expected_false_named_if_all_null': round(nm['N'] * nm['controls_named_at_threshold'] / nm['C'], 2)}}
    # 2. relabelled-control null (the G5 splits: rng(0), permutations over the controls in row order)
    rng = np.random.default_rng(0)
    ctl_units = [u for u in cnt if u[0] == 'random']
    nulls = []
    for _ in range(null_splits):
        perm = rng.permutation(len(ctl_units))
        fake = {ctl_units[i] for i in perm[: len(perm) // 2]}
        st0 = stats_of(relabel(cnt, fake), relabel(cA, fake), relabel(cB, fake), relabel(cov, fake))
        n0 = naming(st0)
        nulls.append({'named': len(n0['named']), 'q05': n0['q05'], 'above_every_control': n0['above_every_control'],
                      'mass_auc': round(n0['mass_auc'], 4), 'controls_named_at_threshold': n0['controls_named_at_threshold']})
    nn = np.array([x['named'] for x in nulls]); nq = np.array([x['q05'] for x in nulls])
    out['relabelled_null'] = {'splits': null_splits, 'fake_neurons': len(ctl_units) // 2,
                              'reference_controls': len(ctl_units) - len(ctl_units) // 2,
                              'named_mean': round(float(nn.mean()), 3), 'named_max': int(nn.max()),
                              'named_splits_nonzero': int((nn > 0).sum()),
                              'q05_mean': round(float(nq.mean()), 3), 'q05_max': int(nq.max()),
                              'above_every_control_mean': round(float(np.mean([x['above_every_control'] for x in nulls])), 2),
                              'mass_auc_mean': round(float(np.mean([x['mass_auc'] for x in nulls])), 4),
                              'g5_null_split_mean_in_verdict': verdict['readings'][str(cut)]['figures']['null_split_auc']['mean']}
    # 4. rho_AB with bootstrap
    rb = np.random.default_rng(12345)
    nkeys = [u for u in st['mass_A'] if u[0] == 'neuron']
    ckeys = [u for u in st['mass_A'] if u[0] == 'random']
    out['rho_AB_mass'] = rho_block(st['mass_A'], st['mass_B'], nkeys, ckeys, rb)
    out['rho_AB_mass']['verdict'] = [round(verdict['readings'][str(cut)]['figures']['rho_AB_neurons'], 3),
                                     round(verdict['readings'][str(cut)]['figures']['rho_AB_controls'], 3)]
    resA = R.residualise(st_all['mass_A'], covA)
    resB = R.residualise(st_all['mass_B'], covB)
    out['rho_AB_mass_residualised_per_half'] = rho_block(resA, resB, nkeys, ckeys, rb)
    # rho with the units well inside the controls' range only (mass_c below the controls' 95th percentile)
    ctl95 = np.percentile([st['mass_c'][u] for u in ckeys], 95)
    bulk = [u for u in nkeys if st['mass_c'][u] < ctl95]
    bulk_c = [u for u in ckeys if st['mass_c'][u] < ctl95]
    out['rho_AB_mass_below_ctl_p95'] = rho_block(st['mass_A'], st['mass_B'], bulk, bulk_c, rb)
    if reasked:
        rk = [u for u in st_all['mass_A'] if u[0] == 'neuron' and u[1] in reasked]
        out['rho_AB_mass_reasked64'] = rho_block(st_all['mass_A'], st_all['mass_B'], rk, ckeys, rb)
    # calibration-sized halves inside half A (two questions, 8 answers, against the third, 4 answers)
    cal = []
    for held in A:
        a2 = [q for q in A if q != held]
        k1 = lambda r, a2=a2: keep(r) and r['prompt'] in set(a2)  # noqa: E731
        k2 = lambda r, h=held: keep(r) and r['prompt'] == h  # noqa: E731
        m1 = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k1, 'own'), 'controls').items()}
        m2 = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k2, 'own'), 'controls').items()}
        cal.append((held, spearman([m1[u] for u in nkeys], [m2[u] for u in nkeys]),
                    spearman([m1[u] for u in ckeys], [m2[u] for u in ckeys])))
        if reasked:
            rk = [u for u in m1 if u[0] == 'neuron' and u[1] in reasked]
            cal[-1] = cal[-1] + (spearman([m1[u] for u in rk], [m2[u] for u in rk]),)
    out['rho_calibration_sized_halves'] = {
        'splits': [[c[0]] + [round(x, 3) for x in c[1:]] for c in cal],
        'mean_neurons': round(float(np.mean([c[1] for c in cal])), 3),
        'mean_controls': round(float(np.mean([c[2] for c in cal])), 3)}
    if reasked:
        out['rho_calibration_sized_halves']['mean_reasked64'] = round(float(np.mean([c[3] for c in cal])), 3)
    # 5. select on half A, test on half B (and the reverse), mass and residualised mass
    def sel_rep(sa, sb):
        pa, pb = R.emp_p(sa), R.emp_p(sb)
        res = {}
        for name, (pp, qq) in {'A->B': (pa, pb), 'B->A': (pb, pa)}.items():
            qn = R.bh({u: v for u, v in pp.items() if u[0] == 'neuron'})
            disc = [u for u in qn if qn[u] <= 0.05]
            thr = max((pp[u] for u in disc), default=None)
            cdisc = [u for u, v in pp.items() if u[0] == 'random' and thr is not None and v <= thr]
            res[name] = {'discovered': len(disc), 'threshold_p': thr,
                         'replicate_p05': int(sum(qq[u] <= 0.05 for u in disc)),
                         'share': round(float(np.mean([qq[u] <= 0.05 for u in disc])), 3) if disc else None,
                         'controls_at_threshold': len(cdisc),
                         'controls_replicate_p05': int(sum(qq[u] <= 0.05 for u in cdisc)),
                         'controls_share': round(float(np.mean([qq[u] <= 0.05 for u in cdisc])), 3) if cdisc else None}
        return res
    rr = restrict({'A': resA, 'B': resB}, fresh)
    out['select_on_one_half'] = {'mass': sel_rep(st['mass_A'], st['mass_B']),
                                 'mass_residualised_per_half': sel_rep(rr['A'], rr['B']),
                                 'words_rep': sel_rep(st['mass_A'], st['rep_AB'])}
    # 6. AUC CIs
    out['auc'] = {k: boot_auc(st[k], rb) for k in ('mass_c', 'mass_c_res', 'mass_B', 'rep_AB', 'rep_BA', 'rep_AB_res', 'rep_BA_res', 'rep_res')}
    # named neurons' words (the replicated ones), a sample
    nw = verdict['readings'][str(cut)].get('named_words') or {}
    out['named_words_sample'] = {k: nw[k] for k in list(nw)[:25]}
    out['seconds'] = round(time.time() - t0, 1)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['L45:48', 'L45:32', 'L57:48', 'L57:32']
    fresh45, re45 = ids(PRE / 'trial_L45_fresh.txt'), ids(PRE / 'trial_L45_reasked.txt')
    res = {}
    for w in which:
        tag, cut = w.split(':')
        cut = int(cut)
        if tag == 'L45':
            r = run('L45', TR / 'L45-break_rel-trial.jsonl', cut, fresh45, re45)
        else:
            r = run('L57', TR / 'L57-break_rel-probe.jsonl', cut, None, None, null_splits=50)
        res[w] = r
        print(json.dumps({k: v for k, v in r.items() if k != 'named_words_sample'}, indent=None, default=str)[:6000], flush=True)
        json.dump(res, open(OUT / f'named_null_{"_".join(x.replace(":", "-") for x in which)}.json', 'w'), indent=1, default=str)
