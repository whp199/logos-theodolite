#!/usr/bin/env python3
"""Read a break_rel trial run against the gates fixed on 29 October 2026 before it was taken. CPU only; reads the run,
writes only --out. Every answer counted (no max_rel cap), each answer's own question's unpushed words left out, mass
against the controls' mean alone (readlib.py).

    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<the break_rel checkout> .venv/bin/python -I trial_read.py RUN.jsonl \
        --gates L45 --fresh trial_L45_fresh.txt --retest-ref atlas_data/census/runs/L45-calib_rel.jsonl \
        --reasked trial_L45_reasked.txt --out L45_verdict.json
    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<the break_rel checkout> .venv/bin/python -I trial_read.py RUN.jsonl \
        --gates L57 --out L57_verdict.json

Run from the copies tonight.sh kept beside the runs (census/trials/prereg, hashed in SHA256SUMS before step 0).

--half-a / --half-b default to the meta's design questions split 3 + 3 in order (who, orange, free | bicycle, walk,
chair, or the spares that replaced them). --cuts: the word-piece lengths read (the first is the primary reading);
answers are cut by the checkpoint's own tokenizer, the unpushed answers too.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readlib as R  # noqa: E402
from backend import census as cz  # noqa: E402

TOK = ('/home/eric/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/'
       '52f3f65bc7a02d555763bc923bd1d9094898219d/tokenizer.json')

# The gates, fixed before the trial (see the design). Each: (statistic, bar). AUC of neurons over controls.
# The calibration beside them (calib_rel, stream +-1 and +-2 on who/orange/free, 48 pieces, 64 + 64 units; rep the mean
# of the three one-question-held-out splits, halves of 8 and 4 answers; calib_expect.json): L45 mass_c 0.64, residual
# 0.69, rep_AB 0.66, rep_BA 0.59 (0.53-0.66 by split); L57 mass_c 0.62, residual 0.67, rep_AB 0.56, rep_BA 0.49 (so S3
# is set above the calibration's rep_BA).
GATES = {
    'L45': {'G1 lead (mass, every answer)': [('mass_c', 0.60)],
            'G2 beyond how the text survives (mass residual on six covariates, fitted on the controls)': [('mass_c_res', 0.58)],
            'G3 replication on disjoint questions (words picked on one half, scored on the other, both ways)': [('rep_AB', 0.57), ('rep_BA', 0.57)],
            'G4 the new questions alone carry a lead (mass on half B)': [('mass_B', 0.56)]},
    'L57': {'S1 lead': [('mass_c', 0.58)], 'S2 beyond text survival': [('mass_c_res', 0.56)],
            'S3 replication both ways': [('rep_AB', 0.55), ('rep_BA', 0.55)]},
}
NULL_BAND = (0.47, 0.53)  # 50 splits of 512/512 controls: a bug moves it far more than this
GAIN_BAND = (0.8, 1.25)


def load(path, cut, tok):
    rows = []
    for r in cz.read_rows(Path(path), tolerate_tail=True):
        ids, lp, text, fin = r['token_ids'], r['logprob'], r['text'], r['finish']
        if cut is not None and len(ids) > cut:
            body = ids[:-1] if fin == 'stop' else ids
            ids, lp, text, fin = ids[:cut], lp[:cut], tok.decode(body[:cut], skip_special_tokens=True), 'length'
        rows.append({'layer': r['layer'], 'kind': r['kind'], 'neuron': int(r['neuron']), 'prompt': r['prompt'],
                     'dose': float(r['dose']), 'rel_push': float(r['rel_push']), 'finish': fin, 'text': text,
                     'token_ids': ids, 'lp': lp})
    return rows


def batch_sizes(meta):
    """{(kind, neuron, prompt, dose): the size of the batch its answer was written in}, from the run's own job list and
    plan (census.census_jobs, census.plan_batches); None where its meta does not rebuild its job list."""
    if not (meta.get('neuron_ids') and meta.get('d_ff') and meta.get('batch') and meta.get('jobs')):
        return None
    try:  # (a census.py without break_rel's questions or plans, THEODOLITE_TRIAL_REPO unset: not rebuilt, said as None)
        jobs = cz.census_jobs(cz.parse_neurons(meta['neuron_ids'], int(meta['d_ff'])), int(meta['random_controls']),
                              [tuple(x) for x in meta['design']], cz.meta_questions(meta))
        if cz.jobs_digest(jobs) != meta['jobs']:
            return None
        out = {}
        for _, _, idx in cz.plan_batches(jobs, int(meta['batch']), meta.get('plan') or cz.PLAN_BEFORE):
            for i in idx:
                j = jobs[i]
                out[(j['kind'], int(j['neuron']), j['prompt'], float(j['dose']))] = len(idx)
        return out
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def writing_rate(log_path):
    """Answers over the seconds spent writing them, from census_run's lines ("layer L: ... K answers to write (D already
    written)" opens a session; "N/T answers, ... pieces/s, R answers/s" closes each chunk, R over the session so far)."""
    import re
    sessions, cur = [], None
    for line in Path(log_path).read_text(errors='replace').splitlines():
        m = re.match(r'^layer \d+: .*?answers to write \((\d+) already written\)', line)
        if m:
            cur = {'start': int(m.group(1)), 'last': None}
            sessions.append(cur)
            continue
        m = re.match(r'^(\d+)/(\d+) answers, [\d,]+ pieces/s, ([\d.]+) answers/s', line)
        if m and cur is not None:
            cur['last'] = (int(m.group(1)), float(m.group(3)))
    done = [(s['last'][0] - s['start'], s['last'][1]) for s in sessions if s['last'] and s['last'][1] > 0]
    if not done:
        return None
    n = sum(w for w, _ in done)
    secs = sum(w / r for w, r in done)
    # answers written (an answer cut off by a stop and written again counts twice: it was written twice)
    return {'answers_written': n, 'seconds': round(secs, 1), 'answers_per_s': round(n / secs, 2) if secs > 0 else None,
            'sessions': len(sessions)}


def ids_of(path, file):
    return {int(x) for x in Path(file).read_text().replace('\n', ',').split(',') if x.strip()} if file else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--gates', choices=sorted(GATES), required=True)
    ap.add_argument('--half-a', default='')
    ap.add_argument('--half-b', default='')
    ap.add_argument('--doses', default='-2,-1,1,2')
    ap.add_argument('--cuts', default='48,32')
    ap.add_argument('--fresh', default='', help='the neurons the gates are read on (default: every neuron of the run)')
    ap.add_argument('--retest-ref', default='', help='a run holding the re-asked neurons on shared cells (the calibration)')
    ap.add_argument('--reasked', default='')
    ap.add_argument('--null-splits', type=int, default=50)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    meta = json.loads(cz.meta_path_of(Path(a.run)).read_text())
    qs = list(meta.get('prompts') or {})
    asked = [q for q in qs if any(p == q for p, _ in meta['design'])]
    A = a.half_a.split(',') if a.half_a else asked[:3]
    B = a.half_b.split(',') if a.half_b else asked[3:6]
    doses = [float(x) for x in a.doses.split(',')]
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
    fresh = ids_of(a.run, a.fresh)
    reasked = ids_of(a.run, a.reasked)
    cuts = [None if c in ('none', 'all') else int(c) for c in a.cuts.split(',')]
    out = {'run': Path(a.run).name, 'layer': meta.get('layer'), 'design': meta.get('design_name'), 'batch': meta.get('batch'),
           'max_new': meta.get('max_new'), 'seed': meta.get('seed'), 'half_a': A, 'half_b': B, 'doses': doses,
           'read_gain': meta.get('read_gain'), 'readings': {}}
    ok_all = True
    for ci, cut in enumerate(cuts):
        rows = load(a.run, cut, tok)
        # completeness: every unit holds every (question, dose) of the reading
        cells = {(q, d) for q in A + B for d in doses}
        have = {}
        for r in rows:
            if r['kind'] != 'baseline' and keep(r):
                have.setdefault((r['kind'], r['neuron']), set()).add((r['prompt'], r['dose']))
        planned = ({('neuron', n) for n in cz.parse_neurons(meta['neuron_ids'], int(meta['d_ff']))}
                   if meta.get('neuron_ids') and meta.get('d_ff') else {u for u in have if u[0] == 'neuron'})
        planned |= {('random', k) for k in range(int(meta.get('random_controls') or 0))}
        n_rows = sum(1 for r in rows)
        complete = (set(have) == planned and all(v == cells for v in have.values())
                    and (fresh is None or fresh <= {n for k, n in have if k == 'neuron'})
                    and (not meta.get('jobs') or n_rows == int(meta['jobs']['n'])))
        fig, stat, named, per = R.read(rows, keep, A, B, units_only=fresh, null_splits=a.null_splits)
        # sign halves and the text at each dose
        figs_sign = {}
        for sgn, name in ((1, '+'), (-1, '-')):
            k2 = lambda r, s=sgn: keep(r) and np.sign(r['dose']) == s  # noqa: E731
            cnt = R.counts(rows, k2, 'own')
            m = {u: v[0] for u, v in R.mass_from_counts(cnt, 'controls').items()}
            if fresh is not None:
                m = {u: v for u, v in m.items() if u[0] == 'random' or u[1] in fresh}
            figs_sign[name] = round(R.auc(*R.split(m))[0], 4)
        base = R.baselines(rows)
        text = {}
        for d in doses:
            for kind in ('neuron', 'random'):
                v = [cz.distinct4(r['token_ids']) - cz.distinct4(base[r['prompt']]['token_ids']) for r in rows
                     if r['kind'] == kind and keep(r) and abs(r['dose'] - d) < 1e-9 and (kind == 'random' or fresh is None or r['neuron'] in fresh)]
                c = [r['finish'] == 'length' for r in rows if r['kind'] == kind and keep(r) and abs(r['dose'] - d) < 1e-9]
                text[f'{d:+g} {kind}'] = {'d4_minus_unpushed': round(float(np.mean(v)), 3), 'cut_off': round(float(np.mean(c)), 3)}
        rd = {'figures': {k: v for k, v in fig.items()}, 'sign_halves_mass_auc': figs_sign, 'text': text, 'complete': complete,
              'units': {'neurons_read': fig['counts']['neurons'], 'controls': fig['counts']['controls'],
                        'units_in_run': len(have), 'units_planned': len(planned), 'answers': n_rows,
                        'answers_planned': (meta.get('jobs') or {}).get('n')}}
        # the gates (on the primary reading; the second cut is read the same way, for the max_new decision)
        gates = {}
        for gname, bars in GATES[a.gates].items():
            vals = [(s, fig[s]['auc'], bar) for s, bar in bars]
            gates[gname] = {'values': vals, 'pass': all(v >= bar for _, v, bar in vals)}
        nm = fig['null_split_auc']['mean']
        gr = (meta.get('read_gain') or {}).get('ratio')
        gates['G5 sanity (null split, read gain, every answer present)'] = {
            'values': [('null_split_mean', nm, NULL_BAND), ('read_gain_ratio', gr, GAIN_BAND),
                       ('complete (every unit planned, each with all 24 cells; every answer of the job list)', complete, True)],
            'pass': NULL_BAND[0] <= nm <= NULL_BAND[1] and gr is not None and GAIN_BAND[0] <= gr <= GAIN_BAND[1] and complete}
        # the code's own ranking (write_units with the design's reading) must be this reading, where it is on disk
        up = cz.units_path_of(Path(a.run))
        if up.exists() and cut is None or (up.exists() and cut is not None and int(meta.get('max_new', 0)) <= cut):
            units = {(u['kind'], int(u['neuron'])): u['mass'] for u in cz.read_rows(up)}
            diff = max(abs(units[u] - per['top'][u][0]) for u in per['top'] if u in units)
            gates['G5b the code ranks as read here (units.jsonl mass = mass_c)'] = {'values': [('max_abs_diff', round(diff, 5), 0.002)],
                                                                                 'pass': diff <= 0.002}
        rd['gates'] = gates
        rd['pass'] = all(g['pass'] for g in gates.values())
        # test-retest of the re-asked neurons on the cells they share with the reference run
        if a.retest_ref and reasked:
            ref = load(a.retest_ref, cut if cut is not None else int(meta['max_new']), tok)
            shared_q = [q for q in A + B if any(r['prompt'] == q for r in ref if r['kind'] != 'baseline')]
            k3 = lambda r: r['prompt'] in set(shared_q) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
            mt = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, k3, 'own'), 'controls').items()}
            mr = {u: v[0] for u, v in R.mass_from_counts(R.counts(ref, k3, 'own'), 'controls').items()}
            us = [('neuron', n) for n in sorted(reasked) if ('neuron', n) in mt and ('neuron', n) in mr]
            rho = float(stats.spearmanr([mt[u] for u in us], [mr[u] for u in us])[0]) if len(us) > 2 else None
            by = {(r['kind'], r['neuron'], r['prompt'], r['dose']): r['token_ids'] for r in ref}
            ref_meta = json.loads(cz.meta_path_of(Path(a.retest_ref)).read_text())
            bs_here, bs_ref = batch_sizes(meta), batch_sizes(ref_meta)
            pairs = [(by.get(('neuron', r['neuron'], r['prompt'], r['dose'])) == r['token_ids'],
                      bs_here is not None and bs_ref is not None
                      and bs_here.get(('neuron', r['neuron'], r['prompt'], r['dose'])) == bs_ref.get(('neuron', r['neuron'], r['prompt'], r['dose'])))
                     for r in rows if r['kind'] == 'neuron' and r['neuron'] in reasked and k3(r)]
            # a row's bits depend on its batch's size (census.plan_batches' note): only the answers written in batches of
            # one size in both runs can be expected to agree bit for bit (660 of 768 at batch 64 against
            # L45-calib_rel's 10 x batch plan, none at 96); reported, gating nothing
            rd['retest'] = {'neurons': len(us), 'cells': shared_q, 'spearman_mass': rho, 'answers': len(pairs),
                            'bit_identical': int(sum(s for s, _ in pairs)),
                            'expected_identical': int(sum(e for _, e in pairs)) if bs_here is not None and bs_ref is not None else None,
                            'bit_identical_where_expected': int(sum(1 for s, e in pairs if s and e)),
                            'batch_here': meta.get('batch'), 'batch_ref': ref_meta.get('batch')}
        rd['named'] = [u[1] for u in named]
        rd['named_words'] = per['named_words']
        rd['named_by_halves'] = [u[1] for u in per['named_by_halves']]
        out['readings'][str(cut or meta.get('max_new'))] = rd
        if ci == 0:
            ok_all = rd['pass']
    out['pass'] = ok_all
    # throughput (reported): the run's answers over its wall time, from <run>.t0 / <run>.t1 (unix seconds) where kept
    stem = str(a.run)[:-len('.jsonl')]
    t0p, t1p, logp = Path(stem + '.t0'), Path(stem + '.t1'), Path(stem + '.log')
    writing = writing_rate(logp) if logp.exists() else None
    if t0p.exists() and t1p.exists():
        secs = float(t1p.read_text().split()[0]) - float(t0p.read_text().split()[0])
        n = sum(1 for _ in cz.read_rows(Path(a.run), tolerate_tail=True))
        # wall time from the first start to the last finish: the model's load and the ranking in it, and any stop between
        out['throughput'] = {'answers': n, 'seconds': round(secs, 1), 'answers_per_s': round(n / secs, 2) if secs > 0 else None,
                             'sessions': writing['sessions'] if writing else None}
    if writing:
        # the writer's own rate (census_run's per-chunk lines): answers over the time spent writing them, every session
        out['writing'] = writing
    prim, sec = (list(out['readings'].values()) + [None])[:2]
    if sec is not None:
        f1, f2 = prim['figures'], sec['figures']
        out['max_new_rule'] = {
            'shorter_passes': sec['pass'],
            'mass_c_drop': round(f1['mass_c']['auc'] - f2['mass_c']['auc'], 4),
            'rep_min_drop': round(min(f1['rep_AB']['auc'], f1['rep_BA']['auc']) - min(f2['rep_AB']['auc'], f2['rep_BA']['auc']), 4),
        }
        r = out['max_new_rule']
        r['take_shorter'] = bool(r['shorter_passes'] and r['mass_c_drop'] <= 0.02 and r['rep_min_drop'] <= 0.02)
    cz.write_json_atomic(Path(a.out), out)
    for cut, rd in out['readings'].items():
        print(f'--- read at {cut} word-pieces: {"PASS" if rd["pass"] else "FAIL"}')
        for g, v in rd['gates'].items():
            print(f'  {"pass" if v["pass"] else "FAIL"}  {g}: ' + ', '.join(f'{s} {x}' + (f' (bar {b})' if not isinstance(b, bool) else '') for s, x, b in v['values']))
        f = rd['figures']
        print('  reported: ' + ', '.join(f'{k} {f[k]["auc"]:.3f}' for k in ('shift', 'shift_up', 'shift_down', 'xq', 'rep', 'mass_A', 'mass_u', 'shift_res', 'rep_res')))
        print(f'  counts {f["counts"]}; rho_AB neurons {f["rho_AB_neurons"]:.3f} controls {f["rho_AB_controls"]:.3f}; signs {rd["sign_halves_mass_auc"]}')
        if 'retest' in rd:
            print(f'  retest {rd["retest"]}')
    if 'max_new_rule' in out:
        print(f'max_new rule: {out["max_new_rule"]}')
    if 'throughput' in out:
        print(f'throughput (wall): {out["throughput"]}')
    if 'writing' in out:
        print(f'throughput (writing, census_run\'s own lines): {out["writing"]}')
    print(f'-> {a.out}: {"PASS" if out["pass"] else "FAIL"}')
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
