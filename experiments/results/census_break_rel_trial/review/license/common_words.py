"""Are the named neurons' replicated words their own, or a vocabulary the neurons share and the controls lack?
CPU only; reads the L45 trial through the pre-registered reader; writes only into this folder.

1. For each word, the share of fresh neurons and of controls holding it among their mass_c top-5 words with z > 0.
2. Sensitivity: drop the shared words (in the top-5 of >= 2% of fresh neurons, i.e. >= 21 of 1,022), from every
   answer and every unpushed answer, and re-read G1-G4's statistics and the naming rule (as named_null.py does)."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

PRE = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg')
sys.path.insert(0, str(PRE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readlib as R  # noqa: E402
import trial_read as T  # noqa: E402
from named_null import stats_of, restrict, naming, ids  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from backend import census as cz  # noqa: E402

OUT = Path(__file__).resolve().parent
TR = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials')


def read_all(rows, A, B, fresh, keep):
    kA = lambda r: keep(r) and r['prompt'] in set(A)  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in set(B)  # noqa: E731
    cnt, cA, cB = R.counts(rows, keep, 'own'), R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    cov = R.covariates(rows, keep)
    st = restrict(stats_of(cnt, cA, cB, cov), fresh)
    nm = naming(st)
    aucs = {k: round(R.auc(*R.split(st[k]))[0], 4) for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B', 'mass_A')}
    return cnt, st, nm, aucs


def main(cut=48, frac=0.02, layer='L45'):
    tok = Tokenizer.from_file(T.TOK)
    if layer == 'L45':
        path, fresh = TR / 'L45-break_rel-trial.jsonl', ids(PRE / 'trial_L45_fresh.txt')
    else:
        path, fresh = TR / 'L57-break_rel-probe.jsonl', None
    meta = json.loads(Path(str(path)[:-6] + '.meta.json').read_text())
    asked = [q for q in meta['prompts'] if any(p == q for p, _ in meta['design'])]
    A, B = asked[:3], asked[3:6]
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in (-2.0, -1.0, 1.0, 2.0))  # noqa: E731
    rows = T.load(path, cut, tok)
    cnt, st, nm, aucs = read_all(rows, A, B, fresh, keep)
    m_c = R.mass_from_counts(cnt, 'controls')
    neu = [u for u in m_c if u[0] == 'neuron' and (fresh is None or u[1] in fresh)]
    ctl = [u for u in m_c if u[0] == 'random']
    tn, tc = collections.Counter(), collections.Counter()
    for u in neu:
        tn.update({w for z, w in m_c[u][1] if z > 0})
    for u in ctl:
        tc.update({w for z, w in m_c[u][1] if z > 0})
    top = [(w, n, round(n / len(neu), 3), tc[w], round(tc[w] / len(ctl), 3)) for w, n in tn.most_common(20)]
    drop = {w for w, n in tn.items() if n >= frac * len(neu)}
    # the same words in the answers of the units with each word, by sign of the push (is it the push's sign?)
    sign = {}
    for w in sorted(drop):
        s = collections.Counter()
        for r in rows:
            if r['kind'] == 'neuron' and keep(r) and w in R._words(r):
                s['+' if r['dose'] > 0 else '-'] += 1
        sign[w] = dict(s)
    # drop them everywhere and re-read
    for r in rows:
        r['_w'] = frozenset(cz.words(r['text'])) - drop
    cnt2, st2, nm2, aucs2 = read_all(rows, A, B, fresh, keep)
    named_both = len({u for u in nm['named']} & {u for u in nm2['named']})
    out = {'layer': layer, 'cut': cut, 'neurons': len(neu), 'controls': len(ctl),
           'top_words_share': [{'word': w, 'neurons': n, 'neuron_share': s, 'controls': c, 'control_share': cs} for w, n, s, c, cs in top],
           'dropped_words': sorted(drop), 'dropped_rule': f'in the z>0 top-5 of >= {frac:.0%} of the neurons read',
           'neuron_answers_holding_dropped_word_by_push_sign': sign,
           'before': {'aucs': aucs, 'named': len(nm['named']), 'q05': nm['q05'], 'above_every_control': nm['above_every_control'],
                      'emp_null_named': round(nm['N'] * nm['controls_named_at_threshold'] / nm['C'], 2)},
           'after_drop': {'aucs': aucs2, 'named': len(nm2['named']), 'q05': nm2['q05'], 'above_every_control': nm2['above_every_control'],
                          'emp_null_named': round(nm2['N'] * nm2['controls_named_at_threshold'] / nm2['C'], 2),
                          'named_in_both': named_both}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(OUT / f'common_words_{layer}_{cut}.json', 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(cut=int(a[1]) if len(a) > 1 else 48, layer=a[0] if a else 'L45')
