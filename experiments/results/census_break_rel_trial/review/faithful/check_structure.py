#!/usr/bin/env python3
"""Faithfulness check, part 1 (CPU only, reads only): neuron lists, meta against the plan, the job list, every row of
both runs, and the units files. Writes only to this folder (structure.json)."""
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
P = T / 'prereg'
RUNS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/runs')
sys.path.insert(0, str(P))
sys.path.insert(0, os.environ['THEODOLITE_TRIAL_REPO'])
from backend import census as cz  # noqa: E402
import trial_read as TR  # noqa: E402  (for batch_sizes, the reader's own rebuild of the plan)

out = {}
D_FF = 21504


def ids(path):
    return cz.parse_neurons(Path(path).read_text().strip().replace('\n', ','), D_FF)



fresh, reasked, l45all, l57 = (ids(P / 'trial_L45_fresh.txt'), ids(P / 'trial_L45_reasked.txt'),
                               cz.parse_neurons(Path(P / 'trial_L45_neurons.txt').read_text().strip(), D_FF),
                               ids(P / 'probe_L57_neurons.txt'))
cal45 = json.loads((RUNS / 'L45-calib_rel.meta.json').read_text())
cal57 = json.loads((RUNS / 'L57-calib_rel.meta.json').read_text())
cal45_n = set(cz.parse_neurons(cal45['neuron_ids'], int(cal45['d_ff'])))
cal57_n = set(cz.parse_neurons(cal57['neuron_ids'], int(cal57['d_ff'])))
draw45 = set(cz.parse_neurons('random:1024', D_FF, seed=3))   # rng(4)
draw0 = set(cz.parse_neurons('random:64', D_FF, seed=0))      # rng(1)
draw57 = set(cz.parse_neurons('random:512', D_FF, seed=6))    # rng(7)
out['lists'] = {
    'fresh_n': len(fresh), 'fresh_unique': len(set(fresh)), 'reasked_n': len(reasked), 'l45_neurons_file_n': len(l45all),
    'fresh_and_reasked_overlap': sorted(set(fresh) & set(reasked)),
    'fresh_union_reasked_eq_l45_file': set(fresh) | set(reasked) == set(l45all),
    'fresh_eq_draw_seed3_less_calib': set(fresh) == draw45 - cal45_n,
    'draw_seed3_and_calib45_overlap': sorted(draw45 & cal45_n),
    'reasked_eq_calib45_neurons': set(reasked) == cal45_n, 'reasked_eq_draw_seed0': set(reasked) == draw0,
    'calib45_seed': cal45.get('seed'), 'calib45_controls': cal45.get('random_controls'),
    'l57_n': len(l57), 'l57_eq_draw_seed6_less_calib57': set(l57) == draw57 - cal57_n,
    'draw_seed6_and_calib57_overlap': sorted(draw57 & cal57_n), 'l57_and_calib57_overlap': sorted(set(l57) & cal57_n),
}

for run, nfile, ctl_n, layer in (('L45-break_rel-trial', set(l45all), 1024, 45), ('L57-break_rel-probe', set(l57), 512, 57)):
    rp = T / f'{run}.jsonl'
    meta = json.loads(cz.meta_path_of(rp).read_text())
    o = out[run] = {}
    meta_neu = set(cz.parse_neurons(meta['neuron_ids'], int(meta['d_ff'])))
    o['meta'] = {
        'design_name': meta['design_name'], 'layer': meta['layer'], 'max_new': meta['max_new'], 'batch': meta['batch'],
        'seed': meta['seed'], 'random_controls': meta['random_controls'], 'plan': meta.get('plan'),
        'plan_kept': meta.get('plan_kept'), 'dose_unit': meta.get('dose_unit'), 'p50': meta.get('p50'),
        'git': meta['runtime'].get('git'), 'git_clean': meta['runtime'].get('git_clean'),
        'mixed_runtime': meta.get('mixed_runtime', 'absent'), 'unpacker_sessions': meta.get('unpacker_sessions', 'absent'),
        'stage': meta.get('stage', 'absent'), 'prompts_order': list(meta['prompts']),
        'design_eq_break_rel_formula': [tuple(x) for x in meta['design']] == [(p, s * d) for p in ['who', 'orange', 'free', 'bicycle', 'walk', 'chair'] for d in (1.0, 2.0) for s in (-1.0, 1.0)],
        'neuron_ids_eq_prereg_list': meta_neu == nfile, 'neuron_ids_n': len(meta_neu),
        'neurons_spec_eq_ids': set(cz.parse_neurons(meta['neurons'], int(meta['d_ff']))) == meta_neu,
        'read_gain': meta.get('read_gain'), 'jobs': meta['jobs'], 'started_at': meta.get('started_at'),
    }
    jobs = cz.census_jobs(sorted(meta_neu), int(meta['random_controls']), [tuple(x) for x in meta['design']], cz.meta_questions(meta))
    o['jobs_rebuilt'] = {'n': len(jobs), 'digest_matches_meta': cz.jobs_digest(jobs) == meta['jobs'],
                         'expected_n': (len(meta_neu) + ctl_n) * 24 + 6}
    bs = TR.batch_sizes(meta)
    o['batch_sizes_rebuilt'] = None if bs is None else dict(collections.Counter(bs.values()))
    # every row
    keys = collections.Counter()
    bad = collections.Counter()
    finish = collections.Counter()
    ngen = collections.Counter()
    seeds = collections.Counter()
    for r in cz.read_rows(rp, tolerate_tail=False):
        k = cz.row_key(r)
        keys[k] += 1
        finish[r['finish']] += 1
        n = len(r['token_ids'])
        ngen[min(n, 49)] += 0  # keep keys bounded
        if n > int(meta['max_new']):
            bad['longer_than_max_new'] += 1
        if r.get('n_gen') is not None and r['n_gen'] != n:
            bad['n_gen_ne_len'] += 1
        if len(r['logprob']) != n:
            bad['logprob_len_ne_ids'] += 1
        if r['layer'] != layer:
            bad['layer'] += 1
        if r['kind'] == 'baseline':
            if r['dose'] != 0.0:
                bad['baseline_dose'] += 1
            continue
        if abs(float(r['rel_push']) - abs(float(r['dose']))) > 1e-9:
            bad['rel_push_ne_abs_dose'] += 1
        if float(r['rel_push']) > 2.5:
            bad['rel_push_over_2.5'] += 1
        if abs(float(r['push']) - round(float(r['dose']) * float(meta['p50']), 4)) > 1e-3:
            bad['push_ne_dose_p50'] += 1
        if r['kind'] == 'random':
            seeds[r.get('seed')] += 1
            if not 0 <= int(r['neuron']) < ctl_n:
                bad['control_id_range'] += 1
        elif int(r['neuron']) not in meta_neu:
            bad['neuron_not_in_list'] += 1
        if r['finish'] == 'stop' and int(r['token_ids'][-1]) not in meta['stop_ids']:
            bad['stop_without_stop_id'] += 1
    jk = collections.Counter(cz.row_key(j) for j in jobs)
    o['rows'] = {'n': sum(keys.values()), 'distinct_keys': len(keys), 'duplicates': sum(v - 1 for v in keys.values() if v > 1),
                 'keys_eq_job_keys': set(keys) == set(jk), 'missing_vs_jobs': len(set(jk) - set(keys)),
                 'extra_vs_jobs': len(set(keys) - set(jk)), 'finish': dict(finish), 'control_row_seeds': dict(seeds),
                 'faults': dict(bad)}
    per_unit = collections.Counter((k[0], k[1]) for k in keys if k[0] != 'baseline')
    o['rows']['units'] = len(per_unit)
    o['rows']['units_cells'] = dict(collections.Counter(per_unit.values()))
    o['rows']['baselines'] = sorted(k[2] for k in keys if k[0] == 'baseline')
    # the units file
    up = cz.units_path_of(rp)
    us = list(cz.read_rows(up))
    o['units_file'] = {'n': len(us), 'kinds': dict(collections.Counter(u['kind'] for u in us)),
                       'readings': [json.loads(x) for x in {json.dumps(u.get('reading'), sort_keys=True) for u in us}],
                       'covers_every_unit': {(u['kind'], int(u['neuron'])) for u in us} == set(per_unit),
                       'mtime_utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime(up.stat().st_mtime))}

(HERE / 'structure.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
