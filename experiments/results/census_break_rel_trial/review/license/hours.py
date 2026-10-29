"""Whole-layer hours for break_rel at L45 from the measured rates (CPU only; reads runs, writes only hours.json here).

Per-batch time T(m) = a + b*m (m = decode steps a batch runs = its longest answer). Points:
  - the L45 trial and L57 probe: break_rel, batch 64, plan 12 x batch, max_new 48 (census_run's own writing rate);
  - the six calib_rel ladders of 28 October: batch 64, plan 10 x batch, max_new 64, the same card and compiled unpacker.
Per-batch seconds = seconds spent writing / batches planned (census.plan_batches on each run's own job list)."""
import json, sys, re, os
from pathlib import Path
import numpy as np
REPO = os.environ['THEODOLITE_TRIAL_REPO']
sys.path.insert(0, REPO)
sys.path.insert(0, '/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
from backend import census as cz
import trial_read as T
OUT = Path(__file__).resolve().parent
TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
CAL = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/runs')
CALLOG = Path('/tmp/claude-1000/-home-eric-bin-ai-theodolite2/55e2a1fc-688e-4ae1-82e6-1ef32bab9bd9/scratchpad/calib/calib_rel.log')

def plan_of(meta):
    jobs = cz.census_jobs(cz.parse_neurons(meta['neuron_ids'], int(meta['d_ff'])), int(meta['random_controls']),
                          [tuple(x) for x in meta['design']], cz.meta_questions(meta))
    assert cz.jobs_digest(jobs) == meta['jobs'], 'job list not rebuilt'
    return jobs, cz.plan_batches(jobs, int(meta['batch']), meta.get('plan') or cz.PLAN_BEFORE)

def steps_per_batch(run, jobs, plan):
    ng = {}
    for r in cz.read_rows(run, tolerate_tail=True):
        ng[(r['kind'], int(r['neuron']), r['prompt'], float(r['dose']))] = int(r['n_gen'])
    out = []
    for _, _, idx in plan:
        out.append(max(ng[(jobs[i]['kind'], int(jobs[i]['neuron']), jobs[i]['prompt'], float(jobs[i]['dose']))] for i in idx))
    return np.array(out)

def session_rates(log):
    """[(answers written, seconds)] per census_run session in a log (the last chunk line of each session)."""
    out, cur = [], None
    for line in Path(log).read_text(errors='replace').splitlines():
        m = re.match(r'^layer (\d+): .*?(\d+) answers to write \((\d+) already written\)', line)
        if m:
            cur = [int(m.group(1)), int(m.group(3)), None]; out.append(cur); continue
        m = re.match(r'^(\d+)/(\d+) answers, ([\d,]+) pieces/s, ([\d.]+) answers/s', line)
        if m and cur is not None:
            cur[2] = (int(m.group(1)), float(m.group(4)), float(m.group(3).replace(',', '')))
    return [(L, n - s, (n - s) / r, pps) for L, s, (n, r, pps) in out if r]

res = {'points': []}
# trial and probe
for stem in ('L45-break_rel-trial', 'L57-break_rel-probe'):
    meta = json.loads((TR / f'{stem}.meta.json').read_text())
    jobs, plan = plan_of(meta)
    st = steps_per_batch(TR / f'{stem}.jsonl', jobs, plan)
    (L, n, secs, pps), = session_rates(TR / f'{stem}.log')
    wall = float((TR / f'{stem}.t1').read_text().split()[0]) - float((TR / f'{stem}.t0').read_text().split()[0])
    res['points'].append({'run': stem, 'max_new': meta['max_new'], 'batch': meta['batch'], 'answers': n, 'batches': len(plan),
                          'fill': round(n / (len(plan) * meta['batch']), 4), 'write_s': round(secs, 1), 'wall_s': wall,
                          'mean_steps': round(float(st.mean()), 2), 'share_batches_at_max': round(float((st == meta['max_new']).mean()), 4),
                          's_per_batch': round(secs / len(plan), 4), 'pieces_per_s': pps})
# calibration ladders (calib_rel, 6 layers, one log)
sess = session_rates(CALLOG)
for L, n, secs, pps in sess:
    meta = json.loads((CAL / f'L{L}-calib_rel.meta.json').read_text())
    jobs, plan = plan_of(meta)
    st = steps_per_batch(CAL / f'L{L}-calib_rel.jsonl', jobs, plan)
    res['points'].append({'run': f'L{L}-calib_rel', 'max_new': meta['max_new'], 'batch': meta['batch'], 'answers': n, 'batches': len(plan),
                          'fill': round(n / (len(plan) * meta['batch']), 4), 'write_s': round(secs, 1),
                          'mean_steps': round(float(st.mean()), 2), 'share_batches_at_max': round(float((st == meta['max_new']).mean()), 4),
                          's_per_batch': round(secs / len(plan), 4), 'pieces_per_s': pps})
for p in res['points']:
    print(p)
# fit T = a + b*m on per-batch time against mean steps run (REJECTED if a < 0: the 64-piece calibration ran from an
# older commit, other questions and a 10 x batch plan; reported only to show it cannot be used)
t48 = np.mean([p['s_per_batch'] for p in res['points'] if p['max_new'] == 48])
m48 = np.mean([p['mean_steps'] for p in res['points'] if p['max_new'] == 48])
t64 = np.mean([p['s_per_batch'] for p in res['points'] if p['max_new'] == 64])
m64 = np.mean([p['mean_steps'] for p in res['points'] if p['max_new'] == 64])
b = (t64 - t48) / (m64 - m48); a = t48 - m48 * b
res['fit'] = {'t48': round(t48, 4), 'm48': round(m48, 2), 't64': round(t64, 4), 'm64': round(m64, 2), 'a_s': round(a, 4), 'b_s_per_step': round(b, 5)}
print('fit', res['fit'])
# whole layer job counts and batches (census's own planner, break_rel at L45)
meta = json.loads((TR / 'L45-break_rel-trial.meta.json').read_text())
design = [tuple(x) for x in meta['design']]
qs = cz.meta_questions(meta)
layer = {}
for C in (1024, 2048):
    jobs = cz.census_jobs(range(21504), C, design, qs)
    plan = cz.plan_batches(jobs, 64, meta['plan'])
    layer[C] = (len(jobs), len(plan))
res['layer'] = {str(C): {'answers': n, 'batches_64': nb} for C, (n, nb) in layer.items()}
trial = res['points'][0]
rate48 = trial['answers'] / trial['write_s']
rate48_wall = trial['answers'] / trial['wall_s']
rows = {}
for C, (n, nb) in layer.items():
    fillL = n / (nb * 64)
    h48_rate = n / rate48 / 3600
    h48_wall = n / rate48_wall / 3600
    h48_batch = nb * trial['s_per_batch'] / 3600
    # 32 pieces: (i) the two-point fit, a batch running 32 steps (the trial's batches ran to the cap: share_batches_at_max);
    # (ii) the formula's ratio (0.45 + 32*0.109) / (0.45 + 48*0.109) applied to the measured 48 rate;
    # (iii) the bound with no fixed cost (time proportional to steps): 32/48 of the 48-piece time
    h32_fit = nb * (a + 32 * b) / 3600
    h32_formula_ratio = h48_batch * (0.45 + 32 * 0.109) / (0.45 + 48 * 0.109)
    h32_floor = h48_batch * 32 / 48
    # (iv) the formula's 0.45 s fixed per batch kept, the step refitted to the measured 48-piece batch time
    step_fit = (trial['s_per_batch'] - 0.45) / 48
    h32_fixed045 = nb * (0.45 + 32 * step_fit) / 3600
    rows[str(C)] = {'answers': n, 'batches': nb, 'fill': round(fillL, 4), 'h48_by_write_rate': round(h48_rate, 2),
                    'h48_by_wall_rate': round(h48_wall, 2), 'h48_by_s_per_batch': round(h48_batch, 2),
                    'h32_two_point_fit': round(h32_fit, 2), 'h32_formula_ratio': round(h32_formula_ratio, 2),
                    'h32_no_fixed_cost_bound': round(h32_floor, 2), 'h32_fixed045_step_refit': round(h32_fixed045, 2),
                    'step_refit_s': round(step_fit, 4),
                    'formula_h48_b64': round(nb * (0.45 + 48 * 0.109) / 3600, 2), 'formula_h32_b64': round(nb * (0.45 + 32 * 0.109) / 3600, 2)}
res['whole_layer_L45_batch64'] = rows
res['rates'] = {'trial_write_answers_per_s': round(rate48, 3), 'trial_wall_answers_per_s': round(rate48_wall, 3),
                'formula_b64_m48': round(64 / (0.45 + 48 * 0.109), 3), 'formula_b64_m32': round(64 / (0.45 + 32 * 0.109), 3),
                'fit_b64_m32': round(64 * trial['fill'] / (a + 32 * b), 3)}
for k, v in rows.items():
    print('controls', k, v)
print(res['rates'])
json.dump(res, open(OUT / 'hours.json', 'w'), indent=1)
