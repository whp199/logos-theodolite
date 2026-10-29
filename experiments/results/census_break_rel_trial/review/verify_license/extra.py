"""Validity checks on the residual figures (CPU only; writes only extra_<tag>-<cut>.json here): do they hold for the
neurons whose six covariates lie inside the controls' range (where the OLS fit on the controls is not extrapolated)?
Usage: extra.py L57:48 (or L45:48 ...)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify as V  # noqa: E402
from verify import R, T, my_auc, my_p, my_bh, spear, ols_resid  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(T.TOK)
res = {}
for w in sys.argv[1:]:
    tag, cut = w.split(':')
    cut = int(cut)
    fresh = V.ids(V.PRE / 'trial_L45_fresh.txt') if tag == 'L45' else None
    meta, A, B, keep, kA, kB, rows = V.setup(tag, cut, tok)
    cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    cov, covA, covB = R.covariates(rows, keep), R.covariates(rows, kA), R.covariates(rows, kB)
    st_all, *_ = V.statistics(cnt, cA, cB, cov)
    st = V.restrict(st_all, fresh)
    nk = [u for u in st['mass_c'] if u[0] == 'neuron']
    ck = [u for u in st['mass_c'] if u[0] == 'random']

    def inrange(c):
        lo = {k: min(c[u][k] for u in ck) for k in R.COVS}
        hi = {k: max(c[u][k] for u in ck) for k in R.COVS}
        return {u for u in nk if all(lo[k] <= c[u][k] <= hi[k] for k in R.COVS)}

    ins = inrange(cov)
    ins_AB = inrange(covA) & inrange(covB)
    out = {'neurons': len(nk), 'in_range': len(ins), 'in_range_both_halves': len(ins_AB)}
    for name, group in (('in', ins), ('out', set(nk) - ins)):
        g = {}
        for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'rep', 'rep_res', 'rep_AB_res', 'rep_BA_res', 'mass_B'):
            g[k] = round(my_auc([st[k][u] for u in group], [st[k][u] for u in ck]), 4)
        out['auc_' + name] = g
    resA = ols_resid(st_all['mass_A'], covA, ck)
    resB = ols_resid(st_all['mass_B'], covB, ck)
    for name, group in (('all', set(nk)), ('in_both', ins_AB), ('out_any', set(nk) - ins_AB)):
        gl = sorted(group)
        out['rho_res_' + name] = {'n': len(gl), 'rho': round(spear([resA[u] for u in gl], [resB[u] for u in gl]), 3)}
    out['rho_res_controls'] = round(spear([resA[u] for u in ck], [resB[u] for u in ck]), 3)
    # select on residualised half A, test on residualised half B, in-range neurons only (BH over them)
    for name, group in (('in_both', ins_AB), ('all', set(nk))):
        sa = {u: v for u, v in resA.items() if u[0] == 'random' or u in group}
        sb = {u: v for u, v in resB.items() if u[0] == 'random' or u in group}
        out['select_res_A->B_' + name] = V.sel_rep(sa, sb)
    # the named neurons and the range (L57: are the 20 named mostly extrapolated?)
    nm = V.naming(st)
    out['named'] = len(nm['named'])
    out['named_in_range'] = int(sum(u in ins for u in nm['named']))
    # naming with the residual criterion dropped (how many named rest on the residual p)
    p = nm['p']
    out['q05_rep_both'] = int(sum(1 for u in nm['q05'] if p['rep_AB'][u] <= 0.05 and p['rep_BA'][u] <= 0.05))
    # mean residual of out-of-range neurons vs in-range neurons (sign of the extrapolation)
    for k in ('mass_c_res', 'rep_res'):
        out['mean_' + k] = {'in': round(float(np.mean([st[k][u] for u in ins])), 3),
                            'out': round(float(np.mean([st[k][u] for u in set(nk) - ins])), 3),
                            'controls': round(float(np.mean([st[k][u] for u in ck])), 3)}
    res[w] = out
    print(w, json.dumps(out), flush=True)
json.dump(res, open(HERE / f'extra_{"_".join(x.replace(":", "-") for x in sys.argv[1:])}.json', 'w'), indent=1)
