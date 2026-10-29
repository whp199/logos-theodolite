#!/usr/bin/env python3
"""Which re-asked answers were written in batches of one size in both runs (the trial and L45-calib_rel), rebuilt
without census.py: each run's job list (baselines, then units with the controls spread evenly, each design cell) and
its batch plan (plan x batch jobs a chunk, grouped by question in first-asked order, batch at a time). Then how many of
those expected-identical answers are bit-identical (both cut to 48 pieces). Reads only; prints."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import myread as M  # noqa: E402


def ranges(spec):
    out = []
    for part in spec.split(','):
        if part.strip():
            a, _, b = part.partition('-')
            out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return sorted(set(out))


def unit_order(neurons, controls):
    units = [('neuron', int(n)) for n in neurons]
    for k in range(controls):
        units.insert(round((k + 1) * len(units) / (controls + 1)), ('random', k))
    return units


T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
RUNS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/runs')
PLANS = {'10 x batch': 10, '16 x batch': 16, '12 x batch': 12}


def sizes(meta):
    design = [tuple(x) for x in meta['design']]
    qs = list(meta['prompts'])
    jobs = [('baseline', -1, q, 0.0) for q in qs]
    jobs += [(k, n, p, float(d)) for k, n in unit_order(ranges(meta['neuron_ids']), int(meta['random_controls'])) for p, d in design]
    B = int(meta['batch'])
    size = B * PLANS[meta.get('plan') or '16 x batch']
    order = list(dict.fromkeys(j[2] for j in jobs))
    out = {}
    for c0 in range(0, len(jobs), size):
        part = range(c0, min(c0 + size, len(jobs)))
        for p in order:
            js = [i for i in part if jobs[i][2] == p]
            for b0 in range(0, len(js), B):
                for i in js[b0:b0 + B]:
                    out[jobs[i]] = len(js[b0:b0 + B])
    return out


mt = json.loads((T / 'L45-break_rel-trial.meta.json').read_text())
mc = json.loads((RUNS / 'L45-calib_rel.meta.json').read_text())
st, sc = sizes(mt), sizes(mc)
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(M.TOK_PATH)
reasked = set(int(x) for x in (T / 'prereg/trial_L45_reasked.txt').read_text().split(',') if x.strip())
trial = {(r['kind'], r['neuron'], r['prompt'], r['dose']): r['ids'] for r in M.load(T / 'L45-break_rel-trial.jsonl')}
ref = {(r['kind'], r['neuron'], r['prompt'], r['dose']): r['ids'] for r in M.load(RUNS / 'L45-calib_rel.jsonl', 48, tok)}
keys = [k for k in trial if k[0] == 'neuron' and k[1] in reasked and k[2] in ('who', 'orange', 'free') and abs(k[3]) in (1.0, 2.0)]
exp = [k for k in keys if st.get(k) == sc.get(k)]
print(json.dumps({'answers': len(keys), 'expected_identical': len(exp), 'identical_where_expected': sum(trial[k] == ref[k] for k in exp),
                  'identical_elsewhere': sum(trial[k] == ref[k] for k in keys if k not in exp),
                  'plan_trial': [mt['batch'], mt.get('plan')], 'plan_calib': [mc['batch'], mc.get('plan')]}))
