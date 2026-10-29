#!/usr/bin/env python3
"""The wider null with myread: the run's neuron rows dropped, half the controls relabelled as neurons, every gated
statistic read in full (the reference, the residual fit and the halves from the remaining controls). First the
reviewer's split procedure (rng 12345 over the sorted control ids, 10 splits), then 50 more splits from another seed to
see whether a mean below 0.5 is noise or a lean. Reads only; writes widenull_<run>.json here."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import myread as M  # noqa: E402

T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
which = sys.argv[1]
run = T / ('L45-break_rel-trial.jsonl' if which == 'L45' else 'L57-break_rel-probe.jsonl')
rows = [r for r in M.load(run) if r['kind'] != 'neuron']
ctl_ids = sorted({r['neuron'] for r in rows if r['kind'] == 'random'})
KEYS = ['mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res']


def splits(seed, n):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in KEYS}
    for _ in range(n):
        perm = rng.permutation(len(ctl_ids))
        fake = {ctl_ids[i] for i in perm[: len(perm) // 2]}
        nr = [dict(r, kind='neuron', neuron=r['neuron'] + 10_000_000) if r['kind'] == 'random' and r['neuron'] in fake else r for r in rows]
        st, _, _, _ = M.statistics(M.Reading(nr))
        a = M.aucs(st, None, KEYS)
        for k in KEYS:
            res[k].append(a[k]['auc'])
    return {k: {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v)), 4), 'n': len(v),
                'se_of_mean': round(float(np.std(v) / np.sqrt(len(v))), 4)} for k, v in res.items()}


out = {'reviewer_procedure_rng12345_10': splits(12345, 10)}
print(json.dumps(out), flush=True)
out['rng99_50'] = splits(99, 50)
print(json.dumps(out['rng99_50']), flush=True)
(HERE / f'widenull_{which}.json').write_text(json.dumps(out, indent=1))
