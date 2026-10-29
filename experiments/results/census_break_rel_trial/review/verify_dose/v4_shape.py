"""V4: what the answers look like at |dose| 1 and 2, neurons (L45: fresh) vs controls; +-1 loop share (d4 < 0.5) by
question for named / neurons / controls; most repeated word of looping +-2 answers; top-5 words at +-2 (from v1's
+-2-only reading). Writes v4_shape.json."""
import collections
import json
import pickle
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import vcore as V  # noqa: E402
from vcore import cz  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(V.T.TOK)


def pclass(s):
    t = s.strip()
    if not t:
        return 'whitespace'
    if all(ch.isalpha() for ch in t):
        return 'word3+' if len(t) >= 3 else 'letters1-2'
    if t.isdigit():
        return 'digits'
    if not any(ch.isalnum() for ch in t):
        return 'punct'
    return 'mixed'


out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    named = set(json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']['named'])
    base = {r['prompt']: r for r in rows if r['kind'] == 'baseline'}
    bw = {q: set(cz.words(r['text'])) for q, r in base.items()}
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    loopw = collections.defaultdict(collections.Counter)
    pcl = collections.defaultdict(collections.Counter)
    loop1 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if not V.in_design(r):
            continue
        u = (r['kind'], r['neuron'])
        isn = u[0] == 'neuron'
        if isn and fresh is not None and u[1] not in fresh:
            continue
        a = int(abs(r['dose']))
        if a == 1:
            grp = ['controls'] if not isn else (['neurons', 'named'] if u[1] in named else ['neurons'])
            for g in grp:
                loop1[g][r['prompt']].append(r['d4'] < 0.5)
        key = (u[0], a)
        ids = r['token_ids']
        b = base[r['prompt']]['token_ids']
        dep = cz.first_departure(ids, b)
        x = acc[key]
        x['identical'].append(dep is None)
        x['dep'].append((len(ids) if dep is None else dep) / max(len(ids), 1))
        x['new'].append(len(set(cz.words(r['text'])) - bw[r['prompt']]))
        x['d4'].append(r['d4'])
        pc = collections.Counter(ids)
        if pc:
            pid, pn = pc.most_common(1)[0]
            x['top_piece_share'].append(pn / len(ids))
            pcl[key][pclass(tok.decode([pid]))] += 1
        toks = cz.TOKEN.findall(r['text'].lower())
        x['no_word'].append(not toks)
        if r['d4'] < 0.5:
            loopw[key][collections.Counter(toks).most_common(1)[0][0] if toks else '<no word>'] += 1
    L = {}
    for key in sorted(acc):
        x = acc[key]
        n = len(x['d4'])
        L[f'{key[0]} |d|={key[1]}'] = {
            'answers': n, 'identical': round(float(np.mean(x['identical'])), 4), 'first_departure_share': round(float(np.mean(x['dep'])), 4),
            'new_words': round(float(np.mean(x['new'])), 2), 'd4': round(float(np.mean(x['d4'])), 4),
            'd4<0.5': round(float(np.mean(np.array(x['d4']) < 0.5)), 4), 'top_piece_share': round(float(np.mean(x['top_piece_share'])), 3),
            'no_word': round(float(np.mean(x['no_word'])), 4),
            'top_piece_class': {k: round(v / n, 3) for k, v in pcl[key].most_common()},
            'loop_words': loopw[key].most_common(10)}
    L['pm1_loop_share'] = {g: {**{q: round(float(np.mean(v)), 3) for q, v in d.items()},
                               'all': round(float(np.mean([y for v in d.values() for y in v])), 4)} for g, d in loop1.items()}
    per = pickle.loads((V.HERE / f'v1_perunit_{layer}.pkl').read_bytes())
    for sub in ('pm2', 'pm1'):
        for kind in ('random', 'neuron'):
            c = collections.Counter()
            for u, t in per[sub]['top'].items():
                if u[0] != kind or (kind == 'neuron' and fresh is not None and u[1] not in fresh):
                    continue
                c.update(w for z, w in t if z > 0)
            L[f'{sub} top-5 of {kind}'] = c.most_common(12)
    print(layer, json.dumps(L)[:6000], flush=True)
    out[layer] = L
(V.HERE / 'v4_shape.json').write_text(json.dumps(out, indent=1))
