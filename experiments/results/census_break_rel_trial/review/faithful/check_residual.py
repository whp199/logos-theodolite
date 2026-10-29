#!/usr/bin/env python3
"""Reported only (no gate): how the G2/S2 residual depends on its linear fit. Covariate overlap of neurons and
controls, the OLS coefficients (fitted on the controls, as readlib.residualise), and the residual's AUC with each
covariate left out in turn. Reads only; writes residual.json here."""
import json, sys
from pathlib import Path
import numpy as np
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
sys.path.insert(0, str(T / 'prereg'))
import readlib as R  # noqa: E402
import trial_read as TR  # noqa: E402
from backend import census as cz  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(TR.TOK)
A, B = ['who', 'orange', 'free'], ['bicycle', 'walk', 'chair']
keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in (-2.0, -1.0, 1.0, 2.0))  # noqa: E731
out = {}
for run, fresh in (('L45-break_rel-trial', T / 'prereg/trial_L45_fresh.txt'), ('L57-break_rel-probe', None)):
    rows = TR.load(T / f'{run}.jsonl', 48, tok)
    fr = set(cz.parse_neurons(fresh.read_text().strip(), 21504)) if fresh else None
    m = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, keep, 'own'), 'controls').items()}
    cov = R.covariates(rows, keep)
    sel = lambda d: {u: v for u, v in d.items() if u[0] == 'random' or fr is None or u[1] in fr}  # noqa: E731
    o = out[run] = {}
    ctl = [u for u in m if u[0] == 'random']
    neu = [u for u in m if u[0] == 'neuron' and (fr is None or u[1] in fr)]
    for c in R.COVS:
        cv = np.array([cov[u][c] for u in ctl]); nv = np.array([cov[u][c] for u in neu])
        o[f'overlap {c}'] = {'neurons_above_ctl_p99': round(float((nv > np.percentile(cv, 99)).mean()), 3),
                             'neurons_below_ctl_p1': round(float((nv < np.percentile(cv, 1)).mean()), 3)}
    X = np.array([[1.0] + [cov[u][c] for c in R.COVS] for u in ctl]); y = np.array([m[u] for u in ctl])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    o['beta'] = dict(zip(['const'] + list(R.COVS), [round(float(b), 3) for b in beta]))
    o['R2_on_controls'] = round(float(1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)), 3)
    o['res_auc_six'] = round(R.auc(*R.split(sel(R.residualise(m, cov))))[0], 4)
    for c in R.COVS:
        rest = tuple(x for x in R.COVS if x != c)
        o[f'res_auc_without_{c}'] = round(R.auc(*R.split(sel(R.residualise(m, cov, rest))))[0], 4)
    print(run, json.dumps(o), flush=True)
Path(__file__).resolve().parent.joinpath('residual.json').write_text(json.dumps(out, indent=1))
