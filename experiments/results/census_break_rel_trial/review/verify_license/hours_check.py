"""Whole-layer hours at L45 from the trial's own log and plan (CPU only; reads, writes only hours_check.json here)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
import readlib  # noqa: E402,F401  (puts THEODOLITE_TRIAL_REPO on sys.path)
from backend import census as cz  # noqa: E402

TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
OUT = Path(__file__).resolve().parent
res = {}
for stem in ('L45-break_rel-trial', 'L57-break_rel-probe'):
    meta = json.loads((TR / f'{stem}.meta.json').read_text())
    jobs = cz.census_jobs(cz.parse_neurons(meta['neuron_ids'], int(meta['d_ff'])), int(meta['random_controls']),
                          [tuple(x) for x in meta['design']], cz.meta_questions(meta))
    assert cz.jobs_digest(jobs) == meta['jobs']
    plan = cz.plan_batches(jobs, int(meta['batch']), meta['plan'])
    ng = {}
    for r in cz.read_rows(TR / f'{stem}.jsonl', tolerate_tail=True):
        ng[cz.row_key(r)] = int(r['n_gen'])
    steps = np.array([max(ng[cz.row_key(jobs[i])] for i in idx) for _, _, idx in plan])
    last = [l for l in (TR / f'{stem}.log').read_text(errors='replace').splitlines() if re.match(r'^\d+/\d+ answers,', l)][-1]
    n, rate = int(last.split('/')[0]), float(re.search(r'([\d.]+) answers/s', last).group(1))
    wall = float((TR / f'{stem}.t1').read_text().split()[0]) - float((TR / f'{stem}.t0').read_text().split()[0])
    res[stem] = {'answers': len(jobs), 'batches': len(plan), 'batch': meta['batch'], 'max_new': meta['max_new'],
                 'batches_at_max': int((steps == meta['max_new']).sum()), 'min_steps': int(steps.min()),
                 'rate_write': rate, 'write_s': round(n / rate, 1), 'wall_s': wall, 'rate_wall': round(n / wall, 3),
                 's_per_batch': round(n / rate / len(plan), 4), 'mean_n_gen': round(float(np.mean(list(ng.values()))), 2)}
tr = res['L45-break_rel-trial']
meta = json.loads((TR / 'L45-break_rel-trial.meta.json').read_text())
design, qs = [tuple(x) for x in meta['design']], cz.meta_questions(meta)
layer = {}
for C in (1024, 2048):
    jobs = cz.census_jobs(range(int(meta['d_ff'])), C, design, qs)
    nb = len(cz.plan_batches(jobs, 64, meta['plan']))
    n = len(jobs)
    spb = tr['s_per_batch']
    step = (spb - 0.45) / 48
    layer[C] = {'answers': n, 'batches': nb,
                'h48_write_rate': round(n / tr['rate_write'] / 3600, 2), 'h48_wall_rate': round(n / tr['rate_wall'] / 3600, 2),
                'h48_per_batch': round(nb * spb / 3600, 2), 'h48_formula': round(nb * (0.45 + 48 * 0.109) / 3600, 2),
                'h32_proportional': round(nb * spb * 32 / 48 / 3600, 2), 'h32_fixed045_refit': round(nb * (0.45 + 32 * step) / 3600, 2),
                'h32_formula_ratio': round(nb * spb * (0.45 + 32 * 0.109) / (0.45 + 48 * 0.109) / 3600, 2),
                'step_refit': round(step, 4)}
res['layer_L45_b64'] = layer
res['extra_2048_h48'] = round(layer[2048]['h48_write_rate'] - layer[1024]['h48_write_rate'], 3)
res['extra_2048_h32'] = [round(layer[2048]['h32_proportional'] - layer[1024]['h32_proportional'], 3),
                         round(layer[2048]['h32_fixed045_refit'] - layer[1024]['h32_fixed045_refit'], 3)]
res['save_32_vs_48_h'] = [round(layer[1024]['h48_per_batch'] - layer[1024]['h32_fixed045_refit'], 2),
                          round(layer[1024]['h48_per_batch'] - layer[1024]['h32_proportional'], 2)]
res['formula_rate_b64_48'] = round(64 / (0.45 + 48 * 0.109), 3)
res['measured_over_formula'] = round(tr['rate_write'] / (64 / (0.45 + 48 * 0.109)), 4)
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / 'hours_check.json', 'w'), indent=1)
