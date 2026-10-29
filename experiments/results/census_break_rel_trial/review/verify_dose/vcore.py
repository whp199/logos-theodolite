"""Independent re-derivation of the dose-split review (verifier's own code). CPU only; reads the runs read-only and
writes only into this folder. Rows are loaded with the pre-registered trial_read.load (cut at 48 pieces by the
checkpoint's tokenizer) and cached here. The reading is composed here from readlib's primitives (counts,
mass_from_counts, z_of, covariates, residualise, emp_p, bh, auc), not by calling readlib.read, and every count can be
restricted to a set of units (a unit filter drops the other units from the counts, the controls' mean and the AUC)."""
import collections
import math
import os
import pickle
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PREREG = '/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg'
sys.path.insert(0, PREREG)
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from backend import census as cz  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
TRIALS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
RUN = {'L45': TRIALS / 'L45-break_rel-trial.jsonl', 'L57': TRIALS / 'L57-break_rel-probe.jsonl'}
QA = ('who', 'orange', 'free')
QB = ('bicycle', 'walk', 'chair')


def fresh_of(layer):
    if layer != 'L45':
        return None
    return {int(x) for x in Path(PREREG + '/trial_L45_fresh.txt').read_text().replace('\n', ',').split(',') if x.strip()}


def load(layer):
    p = HERE / f'_cache_{layer}.pkl'
    if p.exists():
        return pickle.loads(p.read_bytes())
    from tokenizers import Tokenizer
    rows = T.load(RUN[layer], 48, Tokenizer.from_file(T.TOK))
    for r in rows:
        r['d4'] = cz.distinct4(r['token_ids'])
    p.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return rows


def in_design(r):
    return r['kind'] != 'baseline' and r['prompt'] in QA + QB and r['dose'] in (-2.0, -1.0, 1.0, 2.0)


def K(absd=None, dose=None, extra=None):
    """A keep filter: design answers, optionally only |dose| == absd, or dose == dose, and extra(r)."""
    def f(r):
        if not in_design(r):
            return False
        if absd is not None and abs(r['dose']) != absd:
            return False
        if dose is not None and r['dose'] != dose:
            return False
        return extra is None or extra(r)
    return f


def counts_sel(rows, keep, units=None, drop=frozenset()):
    c = R.counts(rows, keep, 'own')
    if units is not None:
        c = collections.defaultdict(collections.Counter, {u: v for u, v in c.items() if u in units})
    if drop:
        for u in c:
            for w in [w for w in c[u] if w in drop]:
                del c[u][w]
    return c


def auc_n(d, fresh):
    d = {u: v for u, v in d.items() if u[0] == 'random' or fresh is None or u[1] in fresh}
    n, c = R.split(d)
    a, p = R.auc(n, c)
    return a, len(n), len(c)


def hanley_se(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def reading(rows, keep, fresh, units=None, drop=frozenset(), extra_cov=None, want_named=True):
    """mass_c, mass_A, mass_B, rep_AB, rep_BA, mass_c_res (six covariates on the controls), xq, named, q05."""
    kA = lambda r: keep(r) and r['prompt'] in QA  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in QB  # noqa: E731
    cnt, cA, cB = counts_sel(rows, keep, units, drop), counts_sel(rows, kA, units, drop), counts_sel(rows, kB, units, drop)
    m = R.mass_from_counts(cnt, 'controls')
    mA, mB = R.mass_from_counts(cA, 'controls'), R.mass_from_counts(cB, 'controls')
    rAB = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    rBA = R.z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    mass = {u: v[0] for u, v in m.items()}
    cov = R.covariates(rows, lambda r: keep(r) and (units is None or (r['kind'], r['neuron']) in units))
    res = R.residualise({u: mass[u] for u in mass if u in cov}, cov)
    st = {'mass_c': mass, 'mass_A': {u: v[0] for u, v in mA.items()}, 'mass_B': {u: v[0] for u, v in mB.items()},
          'rep_AB': rAB, 'rep_BA': rBA, 'mass_c_res': res}
    if extra_cov is not None:
        names = R.COVS + tuple(sorted(next(iter(extra_cov.values()))))
        cov2 = {u: {**cov[u], **extra_cov[u]} for u in cov}
        st['mass_c_res_plus'] = R.residualise({u: mass[u] for u in mass if u in cov2}, cov2, names)
    out = {}
    for k, d in st.items():
        a, n1, n2 = auc_n(d, fresh)
        out[k] = round(a, 4)
    out['n'] = (n1, n2)
    out['se_mass_c'] = round(hanley_se(out['mass_c'], n1, n2), 4)
    out['cov_auc'] = {c: round(auc_n({u: cov[u][c] for u in cov}, fresh)[0], 3) for c in R.COVS}
    if want_named:
        sel = lambda d: {u: v for u, v in d.items() if u[0] == 'random' or fresh is None or u[1] in fresh}  # noqa: E731
        p, pAB, pBA, pr = (R.emp_p(sel(st[k])) for k in ('mass_c', 'rep_AB', 'rep_BA', 'mass_c_res'))
        q = R.bh({u: v for u, v in p.items() if u[0] == 'neuron'})
        named = sorted(u[1] for u in q if q[u] <= 0.05 and pAB[u] <= 0.05 and pBA[u] <= 0.05 and pr[u] <= 0.05)
        neu = np.array(R.split(sel(mass))[0])
        ctlm = np.array(R.split(sel(mass))[1])
        out['named'] = len(named)
        out['q05'] = int(sum(1 for v in q.values() if v <= 0.05))
        out['above_every_control'] = int((neu > ctlm.max()).sum())
        out['_named'] = named
        out['_p'] = {u[1]: v for u, v in p.items() if u[0] == 'neuron'}
        out['_q'] = {u[1]: v for u, v in q.items()}
    out['_top'] = {u: v[1] for u, v in m.items()}
    out['_stat'] = st
    out['_cov'] = cov
    return out


def public(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}
