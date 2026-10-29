"""Q2/Q5 groundwork: what an answer looks like at each dose, neurons vs controls (L45 fresh neurons / L57 all).
Per answer: same as the unpushed answer?, first departure share, new words, distinct-4, the most repeated word-piece
(decoded) and its share of the answer's pieces, the most repeated word (cz.TOKEN) and its share of word tokens.
Writes q5_answer_shape.json and prints the tallies."""
import collections
import json
import re

import numpy as np

import common as C
from common import R, cz

tok = C.tok()


def piece_class(s):
    t = s.strip()
    if t == '':
        return 'whitespace'
    if re.fullmatch(r'[^\W\d_]{3,}', t):
        return 'word3+ (counted by cz.words)'
    if re.fullmatch(r'[^\W\d_]{1,2}', t):
        return 'letters 1-2 (not a word)'
    if re.fullmatch(r'\d+', t):
        return 'digits'
    if re.fullmatch(r'[\W_]+', t):
        return 'punctuation/symbol'
    return 'mixed'


out = {}
for layer in ('L45', 'L57'):
    rows = C.rows(layer)
    fresh = C.FRESH[layer]
    base = R.baselines(rows)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    loopword = collections.defaultdict(collections.Counter)    # (kind, |d|) -> top word of each answer
    looppiece = collections.defaultdict(collections.Counter)   # (kind, |d|) -> class of the top piece
    for r in rows:
        if r['kind'] == 'baseline' or not C.keep_all(r):
            continue
        if r['kind'] == 'neuron' and fresh is not None and r['neuron'] not in fresh:
            continue
        key = (r['kind'], int(abs(r['dose'])))
        b = base[r['prompt']]
        ids = r['token_ids']
        a = acc[key]
        dep = cz.first_departure(ids, b['token_ids'])
        a['identical'].append(dep is None)
        a['dep_share'].append((len(ids) if dep is None else dep) / max(len(ids), 1))
        a['new_words'].append(len(R._words(r) - R._words(b)))
        a['d4'].append(cz.distinct4(ids))
        pc = collections.Counter(ids)
        pid, pn = pc.most_common(1)[0] if pc else (None, 0)
        a['top_piece_share'].append(pn / max(len(ids), 1))
        piece = tok.decode([pid]) if pid is not None else ''
        looppiece[key][piece_class(piece)] += 1
        wt = cz.TOKEN.findall(r['text'].lower())
        a['n_word_tokens'].append(len(wt))
        wc = collections.Counter(wt)
        if wc:
            w, n = wc.most_common(1)[0]
            a['top_word_share'].append(n / len(wt))
            if a['d4'][-1] < 0.5:
                loopword[key][w] += 1
        else:
            a['top_word_share'].append(0.0)
            if a['d4'][-1] < 0.5:
                loopword[key]['<no word>'] += 1
        a['no_word'].append(len(wt) == 0)
    L = {}
    for key in sorted(acc):
        a = acc[key]
        L[f'{key[0]} |d|={key[1]}'] = {
            'answers': len(a['d4']),
            'identical_to_unpushed': round(float(np.mean(a['identical'])), 3),
            'first_departure_share': round(float(np.mean(a['dep_share'])), 3),
            'new_words_mean': round(float(np.mean(a['new_words'])), 2),
            'd4_mean': round(float(np.mean(a['d4'])), 3),
            'share_d4_below_0.5 (a loop)': round(float(np.mean(np.array(a['d4']) < 0.5)), 3),
            'top_piece_share_mean': round(float(np.mean(a['top_piece_share'])), 3),
            'top_word_share_mean': round(float(np.mean(a['top_word_share'])), 3),
            'word_tokens_mean': round(float(np.mean(a['n_word_tokens'])), 1),
            'no_word_at_all': round(float(np.mean(a['no_word'])), 3),
            'top_piece_class': {k: round(v / len(a['d4']), 3) for k, v in looppiece[key].most_common()},
            'loop_word_top15 (answers with d4<0.5: most repeated word)': loopword[key].most_common(15),
            'distinct_loop_words': len(loopword[key]),
        }
    out[layer] = L
    for k, v in L.items():
        print(layer, k, json.dumps(v)[:1400])
(C.HERE / 'q5_answer_shape.json').write_text(json.dumps(out, indent=1))
