#!/usr/bin/env python3
"""The smaller figures: calib_expect's quoted values, the scratchpad copies ($S) against the hashed prereg copies,
the decoder check behind the 32-piece reading, the verdict JSONs' text-per-dose block (whose neurons each column
counts), and the retest (bit identity and the mass Spearman on the shared cells), re-derived with myread. Reads only;
writes misc.json here."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import myread as M  # noqa: E402

T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
S = Path('/tmp/claude-1000/-home-eric-bin-ai-theodolite2/55e2a1fc-688e-4ae1-82e6-1ef32bab9bd9/scratchpad/design/synth')
RUNS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/runs')
out = {}

c = json.loads((T / 'prereg/calib_expect.json').read_text())
cal = {}
for cut, v in c.items():
    Ls = [e['L'] for e in v['extra']]
    cal[cut] = {L: {k: round(v['per'][k][i][0], 4) for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA')} for i, L in enumerate(Ls)}
out['calib_expect'] = {cut: {L: cal[cut][L] for L in (45, 57)} for cut in cal}

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None  # noqa: E731
out['S_vs_prereg'] = {f: (sha(S / f) == sha(T / 'prereg' / f), sha(S / f) is not None) for f in
                      ('tonight.sh', 'step0_pick.py', 'trial_read.py', 'readlib.py', 'calib_expect.json', 'trial_L45_neurons.txt',
                       'trial_L45_fresh.txt', 'trial_L45_reasked.txt', 'probe_L57_neurons.txt')}

# decoder: the tokenizer's decode of each stored answer (stop piece dropped) against its stored text
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(M.TOK_PATH)
dec = {}
for name in ('L45-break_rel-trial', 'L57-break_rel-probe'):
    n = mis = longer32 = 0
    with open(T / f'{name}.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            body = r['token_ids'][:-1] if r['finish'] == 'stop' else r['token_ids']
            n += 1
            mis += tok.decode(body, skip_special_tokens=True) != r['text']
            longer32 += len(r['token_ids']) > 32
    dec[name] = {'rows': n, 'mismatch': mis, 'rows_cut_at_32': longer32}
out['decoder'] = dec
print(dec, flush=True)

# the verdict's text block: which neurons it counts
v45 = json.loads((T / 'L45_verdict.json').read_text())
out['verdict_text_L45_48'] = v45['readings']['48']['text']
out['verdict_max_new'] = {'L45': v45['max_new_rule'], 'L57': json.loads((T / 'L57_verdict.json').read_text())['max_new_rule']}
out['verdict_retest_L45_48'] = v45['readings']['48']['retest']

# retest: the 64 re-asked neurons on who/orange/free at +-1, +-2, against L45-calib_rel
reasked = set(int(x) for x in (T / 'prereg/trial_L45_reasked.txt').read_text().split(',') if x.strip())
trial = M.load(T / 'L45-break_rel-trial.jsonl')
ref = M.load(RUNS / 'L45-calib_rel.jsonl', 48, tok)  # cut to the trial's 48 pieces, as trial_read does
by = {(r['kind'], r['neuron'], r['prompt'], r['dose']): r['ids'] for r in ref}
qs = ('who', 'orange', 'free')
pairs = [(r, by.get(('neuron', r['neuron'], r['prompt'], r['dose']))) for r in trial
         if r['kind'] == 'neuron' and r['neuron'] in reasked and r['prompt'] in qs and abs(r['dose']) in (1.0, 2.0)]
ident = sum(1 for r, b in pairs if b == r['ids'])
mt = M.mass(M.tops(M.Reading(trial, qs=qs).counts()))
mr = M.mass(M.tops(M.Reading(ref, qs=qs).counts()))
us = sorted(('neuron', n) for n in reasked)
out['retest'] = {'answers': len(pairs), 'bit_identical': ident, 'missing_in_ref': sum(1 for _, b in pairs if b is None),
                 'spearman_mass': float(stats.spearmanr([mt[u] for u in us], [mr[u] for u in us])[0]),
                 'ref_controls': sum(1 for u in mr if u[0] == 'random'), 'ref_neurons': sum(1 for u in mr if u[0] == 'neuron')}
print(out['retest'], flush=True)
(HERE / 'misc.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('calib_expect', 'S_vs_prereg', 'verdict_max_new')}, indent=1))
