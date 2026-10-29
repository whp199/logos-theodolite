"""Two checks on the per-neuron reading at L45 (48 pieces). CPU only; writes only into this folder.
1. Where the rank agreement lives, without selecting on a sum of both halves (Berkson): Spearman of mass_A with
   mass_B among units whose mass_A is below the controls' 95th percentile of mass_A (selection on one half only),
   and among the rest.
2. Extrapolation of the residual: the share of fresh neurons, of named neurons and of controls with any of the six
   covariates outside the controls' [min, max] (the OLS fit is made on the controls and extrapolated there).
3. The calibration's resolution floor: the fewest neurons above every control BH needs for q <= 0.05 with N neurons
   and C controls (p = 1/(C+1) for each), against each run's above-every-control count."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PRE = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
sys.path.insert(0, str(PRE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from named_null import stats_of, restrict, naming, ids, boot_rho, ci  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

OUT = Path(__file__).resolve().parent
TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')


def need(N, C, q=0.05):
    return math.ceil((1 / (C + 1)) * N / q)


res = {'bh_floor': {'calibration 64/64': need(64, 64), 'L45 trial 1022/1024': need(1022, 1024), 'L57 probe 507/512': need(507, 512),
                    'whole layer 21504/1024': need(21504, 1024), 'whole layer 21504/2048': need(21504, 2048)},
       'calibration_above_every_control_L45': {'cut 32': 12, 'cut 48': 15, 'cut 64': 18}}
tok = Tokenizer.from_file(T.TOK)
fresh = ids(PRE / 'trial_L45_fresh.txt')
A, B = ['who', 'orange', 'free'], ['bicycle', 'walk', 'chair']
keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in (-2.0, -1.0, 1.0, 2.0))  # noqa: E731
kA = lambda r: keep(r) and r['prompt'] in set(A)  # noqa: E731
kB = lambda r: keep(r) and r['prompt'] in set(B)  # noqa: E731
rows = T.load(TR / 'L45-break_rel-trial.jsonl', 48, tok)
cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
cov = R.covariates(rows, keep)
st = restrict(stats_of(cnt, cA, cB, cov), fresh)
nm = naming(st)
nk = [u for u in st['mass_A'] if u[0] == 'neuron']
ck = [u for u in st['mass_A'] if u[0] == 'random']
p95A = np.percentile([st['mass_A'][u] for u in ck], 95)
rb = np.random.default_rng(7)
for name, sel in (('mass_A below controls p95', lambda u: st['mass_A'][u] < p95A), ('mass_A at or above controls p95', lambda u: st['mass_A'][u] >= p95A)):
    n = [u for u in nk if sel(u)]
    c = [u for u in ck if sel(u)]
    xn, yn = [st['mass_A'][u] for u in n], [st['mass_B'][u] for u in n]
    xc, yc = [st['mass_A'][u] for u in c], [st['mass_B'][u] for u in c]
    res[name] = {'neurons': len(n), 'rho_neurons': round(float(stats.spearmanr(xn, yn)[0]), 3), 'rho_neurons_ci': ci(boot_rho(xn, yn, rb)),
                 'controls': len(c), 'rho_controls': round(float(stats.spearmanr(xc, yc)[0]), 3) if len(c) > 5 else None,
                 'mass_B_auc_vs_all_controls': round(R.auc(yn, [st['mass_B'][u] for u in ck])[0], 4)}
lo = {k: min(cov[u][k] for u in ck) for k in R.COVS}
hi = {k: max(cov[u][k] for u in ck) for k in R.COVS}
out_of = lambda u: any(cov[u][k] < lo[k] or cov[u][k] > hi[k] for k in R.COVS)  # noqa: E731
named = set(nm['named'])
res['outside_control_covariate_range'] = {
    'fresh_neurons': round(float(np.mean([out_of(u) for u in nk])), 4), 'named': round(float(np.mean([out_of(u) for u in named])), 4),
    'named_count': int(sum(out_of(u) for u in named)),
    'by_covariate_fresh': {k: int(sum(cov[u][k] < lo[k] or cov[u][k] > hi[k] for u in nk)) for k in R.COVS}}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / 'bulk_extra.json', 'w'), indent=1)
