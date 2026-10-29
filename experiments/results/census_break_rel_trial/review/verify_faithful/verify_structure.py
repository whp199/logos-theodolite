#!/usr/bin/env python3
"""Neuron lists, completeness and the job digest, re-derived without census.py (its one-line definitions re-written
here: parse_neurons' random:K = sorted(rng(seed + 1).choice(n, K)), unit_order, census_jobs, jobs_digest). Also the
read-gain sample's overlap with the run's neurons. Reads only; writes structure.json here."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
RUNS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/runs')
N = 21504


def rand(k, seed):
    return sorted(np.random.default_rng(seed + 1).choice(N, k, replace=False).tolist())


def ranges(spec):
    out = []
    for part in spec.split(','):
        if part.strip():
            a, _, b = part.partition('-')
            out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return sorted(set(out))


def plain(path):
    return sorted(int(x) for x in Path(path).read_text().replace('\n', ',').split(',') if x.strip())


def unit_order(neurons, controls):
    units = [('neuron', int(n)) for n in neurons]
    for k in range(controls):
        units.insert(round((k + 1) * len(units) / (controls + 1)), ('random', k))
    return units


def jobs(neurons, controls, design, questions):
    js = [('baseline', -1, q, 0.0) for q in questions]
    js += [(kind, n, p, float(d)) for kind, n in unit_order(neurons, controls) for p, d in design]
    return js


def digest(js):
    return hashlib.sha256(json.dumps([list(j) for j in js]).encode('utf-8')).hexdigest()[:16]


out = {}
calib = {L: json.loads((RUNS / f'L{L}-calib_rel.meta.json').read_text()) for L in (45, 57)}
cal45, cal57 = ranges(calib[45]['neuron_ids']), ranges(calib[57]['neuron_ids'])
r64 = rand(64, 0)
fresh_drawn = rand(1024, 3)
fresh = plain(T / 'prereg/trial_L45_fresh.txt')
reasked = plain(T / 'prereg/trial_L45_reasked.txt')
m45 = json.loads((T / 'L45-break_rel-trial.meta.json').read_text())
m57 = json.loads((T / 'L57-break_rel-probe.meta.json').read_text())
l57_drawn = rand(512, 6)
l57 = ranges(Path(T / 'prereg/probe_L57_neurons.txt').read_text().strip())
out['lists'] = {
    'calib45_is_random64_seed0': cal45 == r64, 'calib57_is_random64_seed0': cal57 == r64,
    'fresh_overlap_with_calib64': sorted(set(fresh_drawn) & set(cal45)),
    'fresh_file_eq_draw_less_overlap': fresh == sorted(set(fresh_drawn) - set(cal45)), 'fresh_n': len(fresh),
    'reasked_file_eq_calib45': reasked == cal45, 'reasked_n': len(reasked),
    'fresh_and_reasked_disjoint': not (set(fresh) & set(reasked)),
    'meta45_neurons_eq_union': ranges(m45['neuron_ids']) == sorted(set(fresh) | set(reasked)),
    'meta45_n': len(ranges(m45['neuron_ids'])),
    'l57_overlap_with_calib64': sorted(set(l57_drawn) & set(cal57)),
    'l57_file_eq_draw_less_overlap': l57 == sorted(set(l57_drawn) - set(cal57)), 'l57_n': len(l57),
    'meta57_neurons_eq_file': ranges(m57['neuron_ids']) == l57,
    'l57_shared_with_calib57': len(set(l57) & set(cal57)),
}
# the read gain's neuron sample (census.gain_sample: rng(seed + 2).choice(n, 512)) against the run's neurons
gs = set(np.random.default_rng(3 + 2).choice(N, 512, replace=False).tolist())
out['gain_sample_overlap'] = {'L45_run_neurons_in_gain_sample': len(gs & set(ranges(m45['neuron_ids']))),
                              'L45_fresh_in_gain_sample': len(gs & set(fresh)),
                              'L57_run_neurons_in_gain_sample': len(gs & set(l57)),
                              'expected_L45': round(512 * 1086 / N, 1), 'expected_L57': round(512 * 507 / N, 1)}

for name, meta in (('L45-break_rel-trial', m45), ('L57-break_rel-probe', m57)):
    design = [tuple(x) for x in meta['design']]
    qs = list(dict.fromkeys(p for p, _ in design))
    js = jobs(ranges(meta['neuron_ids']), int(meta['random_controls']), design, qs)
    keys = []
    bad = collections.Counter()
    p50 = float(meta['p50'])
    with open(T / f'{name}.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            k = (r['kind'], int(r['neuron']), r['prompt'], float(r['dose']))
            keys.append(k)
            if r['kind'] == 'random' and r.get('seed') != 3:
                bad['control_seed_not_3'] += 1
            if r['kind'] != 'baseline':
                if abs(float(r['rel_push']) - abs(float(r['dose']))) > 1e-9:
                    bad['rel_push_ne_abs_dose'] += 1
                if float(r['rel_push']) > 2.5:
                    bad['rel_push_over_2.5'] += 1
                if abs(float(r['push']) - round(float(r['dose']) * p50, 4)) > 1e-6:
                    bad['push_ne_dose_p50'] += 1
            if r['finish'] not in ('stop', 'length'):
                bad['finish_odd'] += 1
    per_unit = collections.defaultdict(set)
    for k in keys:
        if k[0] != 'baseline':
            per_unit[(k[0], k[1])].add((k[2], k[3]))
    cells = {(p, float(d)) for p, d in design}
    out[name] = {
        'design': design, 'questions': qs, 'halves': [qs[:3], qs[3:6]],
        'rows': len(keys), 'jobs_n_meta': meta['jobs']['n'], 'duplicates': len(keys) - len(set(keys)),
        'missing': len(set(js) - set(keys)), 'extra': len(set(keys) - set(js)),
        'rows_in_job_order': keys == js,
        'units': len(per_unit), 'units_with_all_24': sum(1 for v in per_unit.values() if v == cells), 'cells': len(cells),
        'digest_rebuilt': digest(js), 'digest_meta': meta['jobs']['sha256'],
        'faults': dict(bad),
        'meta': {k: meta.get(k) for k in ('layer', 'design_name', 'batch', 'max_new', 'seed', 'random_controls', 'plan',
                                           'read_gain', 'p50')},
        'git': meta['runtime'].get('git'), 'git_clean': meta['runtime'].get('git_clean'),
        'mixed_runtime': meta.get('mixed_runtime'), 'unpacker_sessions': meta.get('unpacker_sessions'),
    }
print(json.dumps(out, indent=1))
(HERE / 'structure.json').write_text(json.dumps(out, indent=1))
