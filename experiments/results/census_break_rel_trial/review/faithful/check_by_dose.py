#!/usr/bin/env python3
"""Reported only (no gate): the gated statistics read on |dose| 1 alone and on |dose| 2 alone, with readlib.read and
the pre-registered halves, to see where the lead sits. Reads only; writes by_dose.json here."""
import json, sys
from pathlib import Path
T = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')
sys.path.insert(0, str(T / 'prereg'))
import readlib as R  # noqa: E402
import trial_read as TR  # noqa: E402
from backend import census as cz  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(TR.TOK)
A, B = ['who', 'orange', 'free'], ['bicycle', 'walk', 'chair']
out = {}
for run, fresh in (('L45-break_rel-trial', T / 'prereg/trial_L45_fresh.txt'), ('L57-break_rel-probe', None)):
    rows = TR.load(T / f'{run}.jsonl', 48, tok)
    fr = set(cz.parse_neurons(fresh.read_text().strip(), 21504)) if fresh else None
    for name, ds in (('|dose| 1', (-1.0, 1.0)), ('|dose| 2', (-2.0, 2.0)), ('all four', (-2.0, -1.0, 1.0, 2.0))):
        keep = lambda r, ds=ds: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in ds)  # noqa: E731
        f, _, _, _ = R.read(rows, keep, A, B, units_only=fr, null_splits=1)
        out[f'{run} {name}'] = {k: f[k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'rep_res', 'cov_lp', 'cov_d4', 'cov_lost')}
        print(run, name, out[f'{run} {name}'], flush=True)
Path(__file__).resolve().parent.joinpath('by_dose.json').write_text(json.dumps(out, indent=1))
