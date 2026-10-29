"""Robustness of the covariate-range split (CPU only; writes only extra2.json here).
1. L45 with the controls' range taken from 512 random controls (as many as L57 has): is L57's 30% out of range a
   layer difference or the smaller control set?
2. L57 (and L45) gate AUCs on the neurons inside the controls' range widened by 10% / 25% of that range each side."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify as V  # noqa: E402
from verify import R, T, my_auc  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(T.TOK)
res = {}
for tag in ('L45', 'L57'):
    fresh = V.ids(V.PRE / 'trial_L45_fresh.txt') if tag == 'L45' else None
    meta, A, B, keep, kA, kB, rows = V.setup(tag, 48, tok)
    cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    cov = R.covariates(rows, keep)
    st = V.restrict(V.statistics(cnt, cA, cB, cov)[0], fresh)
    nk = [u for u in st['mass_c'] if u[0] == 'neuron']
    ck = [u for u in st['mass_c'] if u[0] == 'random']
    out = {}
    if tag == 'L45':
        rng = np.random.default_rng(11)
        sh = []
        for _ in range(20):
            sub = [ck[i] for i in rng.choice(len(ck), 512, replace=False)]
            lo = {k: min(cov[u][k] for u in sub) for k in R.COVS}
            hi = {k: max(cov[u][k] for u in sub) for k in R.COVS}
            sh.append(np.mean([any(cov[u][k] < lo[k] or cov[u][k] > hi[k] for k in R.COVS) for u in nk]))
        out['out_of_range_share_512ctl_range'] = round(float(np.mean(sh)), 4)
    for widen in (0.0, 0.10, 0.25):
        lo = {k: min(cov[u][k] for u in ck) for k in R.COVS}
        hi = {k: max(cov[u][k] for u in ck) for k in R.COVS}
        lo, hi = ({k: lo[k] - widen * (hi[k] - lo[k]) for k in R.COVS}, {k: hi[k] + widen * (hi[k] - lo[k]) for k in R.COVS})
        ins = [u for u in nk if all(lo[k] <= cov[u][k] <= hi[k] for k in R.COVS)]
        out[f'widen_{widen}'] = {'in_range': len(ins), **{k: round(my_auc([st[k][u] for u in ins], [st[k][u] for u in ck]), 4)
                                                          for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res')}}
    res[tag] = out
    print(tag, json.dumps(out), flush=True)
json.dump(res, open(HERE / 'extra2.json', 'w'), indent=1)
