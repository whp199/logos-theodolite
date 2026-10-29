"""Q5 and naming checks (writes q5b_words.json):
- the controls' and the neurons' top-5 words on the +-2-only reading and on the +-1-only reading (q1_perunit pickles),
  counted as the number of units holding the word among their top-5 with z > 0;
- the overlap of the verdict's named neurons with those named on the +-1-only reading;
- how many named neurons carry collapse vocabulary among their replicated words (verdict named_words);
- the full pre-registered reading (readlib.read) re-run with the collapse vocabulary taken out of every answer's words:
  the gate figures and the named count."""
import collections
import json
import pickle

import common as C
from common import R, cz

out = {}
for layer in ('L45', 'L57'):
    per = pickle.load(open(C.HERE / f'q1_perunit_{layer}.pkl', 'rb'))
    fresh = C.FRESH[layer]
    V = json.loads((C.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']
    L = {}
    for sub in ('pm1', 'pm2', 'all'):
        top = per[sub]['top']
        for kind in ('random', 'neuron'):
            c = collections.Counter()
            n = 0
            for u, t in top.items():
                if u[0] != kind or (kind == 'neuron' and fresh is not None and u[1] not in fresh):
                    continue
                n += 1
                c.update(w for z, w in t if z > 0)
            L[f'{sub} top-5 words of {kind}s (units holding it, of {n})'] = c.most_common(25)
    L['named on pm1 alone'] = len(per['pm1']['named'])
    L['verdict named also named on pm1 alone'] = len(set(V['named']) & set(per['pm1']['named']))
    L['verdict named with q<=0.05 on pm1 alone'] = sum(1 for n in V['named'] if per['pm1']['q'].get(('neuron', n), 1) <= 0.05)
    L['verdict named with q<=0.05 on pm2 alone'] = sum(1 for n in V['named'] if per['pm2']['q'].get(('neuron', n), 1) <= 0.05)
    L['verdict named with p<=0.05 on pm2 alone'] = sum(1 for n in V['named'] if per['pm2']['p'].get(('neuron', n), 1) <= 0.05)
    L['verdict named with p<=0.05 on pm1 alone'] = sum(1 for n in V['named'] if per['pm1']['p'].get(('neuron', n), 1) <= 0.05)
    # collapse vocabulary in named_words
    def collapse(w):
        return 'thought' in w or w in {'own', 'owns', 'same', 'idea', 'rest', 'sss', 'uma', 'and', 'depending', 'whether', 'since'}
    nw = V['named_words']
    k_any = sum(1 for n, d in nw.items() if any(collapse(w) for w, z in d['A_on_B'] + d['B_on_A']))
    k_only = sum(1 for n, d in nw.items() if all(collapse(w) for w, z in d['A_on_B'] + d['B_on_A']))
    k_lead = sum(1 for n, d in nw.items() if d['A_on_B'] and d['B_on_A'] and collapse(max(d['A_on_B'] + d['B_on_A'], key=lambda x: x[1])[0]))
    L['named with collapse vocabulary among replicated words'] = k_any
    L['named whose replicated words are all collapse vocabulary'] = k_only
    L['named whose highest-z replicated word is collapse vocabulary'] = k_lead
    rep_words = collections.Counter()
    for n, d in nw.items():
        rep_words.update({w for w, z in d['A_on_B'] + d['B_on_A']})
    L['most frequent replicated words among named (neurons holding it)'] = rep_words.most_common(20)
    out[layer] = L
    print(layer, json.dumps({k: v for k, v in L.items() if not k.startswith(('pm', 'all'))}), flush=True)
    for k in L:
        if k.startswith(('pm2 top-5', 'pm1 top-5')):
            print(layer, k, L[k][:15])

    # the pre-registered reading with the collapse vocabulary removed from every answer's words
    rows = C.rows(layer)
    c2 = R.counts(rows, C.keep_abs(2.0), 'own')
    held = collections.Counter()
    allw = set()
    for u, c in c2.items():
        allw.update(c)
        if u[0] == 'random':
            held.update(w for w, n in c.items() if n >= 2)
    drop = frozenset({w for w in allw if 'thought' in w} | {w for w, n in held.items() if n >= 10})
    for r in rows:
        r['_w'] = frozenset(cz.words(r['text'])) - drop
    fig, stat, named, pp = R.read(rows, C.keep_all, C.A, C.B, units_only=fresh, null_splits=20)
    L['full reading, collapse vocabulary removed'] = {k: fig[k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B')}
    L['full reading, collapse vocabulary removed']['named'] = fig['counts']['named']
    L['full reading, collapse vocabulary removed']['named also in the verdict'] = len(set(V['named']) & {u[1] for u in named})
    L['full reading, collapse vocabulary removed']['null'] = fig['null_split_auc']['mean']
    print(layer, 'collapse removed', L['full reading, collapse vocabulary removed'], flush=True)
    for r in rows:
        r.pop('_w', None)
(C.HERE / 'q5b_words.json').write_text(json.dumps(out, indent=1))
