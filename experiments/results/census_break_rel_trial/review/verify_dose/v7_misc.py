"""V7: (1) share of top-5 z on the reviewer's top-20 'attractor' list vs on the full removed set; (2) how question-bound
a unit's top-5 words are (share of each word's answers from its modal question; share of z on words whose answers all
come from one question, and from one question at both signs); (3) the +-2 answers of L45 named 11353, 2833, 9194 and
the top words of the L57 named; (4) +-1 looping answers of named L45 neurons: loop word on its own top-5 or not.
Writes v7_misc.json."""
import collections
import json
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcore as V  # noqa: E402
from vcore import R, cz  # noqa: E402

out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    vd = json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']
    named = set(vd['named'])
    drop = frozenset(json.loads((V.HERE / f'v5_drop_{layer}.json').read_text()))
    base = {r['prompt']: set(cz.words(r['text'])) for r in rows if r['kind'] == 'baseline'}
    c2 = R.counts(rows, V.K(absd=2.0), 'own')
    held = collections.Counter()
    for u, c in c2.items():
        if u[0] == 'random':
            held.update(w for w, n in c.items() if n >= 2)
    attract = {w for w, _ in held.most_common(20)}
    cnt = R.counts(rows, V.K(), 'own')
    m = R.mass_from_counts(cnt, 'controls')
    where = collections.defaultdict(lambda: collections.defaultdict(list))  # unit -> word -> [(q, dose)]
    for r in rows:
        if not V.in_design(r):
            continue
        u = (r['kind'], r['neuron'])
        for w in set(cz.words(r['text'])) - base[r['prompt']]:
            where[u][w].append((r['prompt'], r['dose']))
    groups = {'named': [('neuron', n) for n in named],
              'neurons': [u for u in cnt if u[0] == 'neuron' and (fresh is None or u[1] in fresh)],
              'controls': [u for u in cnt if u[0] == 'random']}
    L = {'attractor20': sorted(attract)}
    for g, us in groups.items():
        Z = Za = Zd = Z1 = Z1b = Zmod = 0.0
        for u in us:
            for z, w in m[u][1]:
                if z <= 0:
                    continue
                Z += z
                Za += z if w in attract else 0
                Zd += z if w in drop else 0
                wl = where[u][w]
                qs = collections.Counter(q for q, _ in wl)
                Zmod += z * qs.most_common(1)[0][1] / len(wl)
                if len(qs) == 1:
                    Z1 += z
                    if len({d > 0 for _, d in wl}) == 2:
                        Z1b += z
        L[g] = {'z_share_attractor20': round(Za / Z, 3), 'z_share_removed_set': round(Zd / Z, 3),
                'z_weighted_modal_question_share': round(Zmod / Z, 3), 'z_share_one_question': round(Z1 / Z, 3),
                'z_share_one_question_both_signs': round(Z1b / Z, 3)}
    if layer == 'L45':
        ex = {}
        for n in (11353, 2833, 9194):
            u = ('neuron', n)
            ans = [r for r in rows if r['kind'] == 'neuron' and r['neuron'] == n and V.in_design(r) and abs(r['dose']) == 2.0]
            mc = collections.Counter()
            for r in ans:
                t = cz.TOKEN.findall(r['text'].lower())
                mc[collections.Counter(t).most_common(1)[0][0] if t else '<no word>'] += 1
            ex[n] = {'top5': [(w, round(z, 2)) for z, w in m[u][1]], 'pm2_most_repeated_word': mc.most_common(5),
                     'idea_in_removed_set': 'idea' in drop}
        L['examples'] = ex
        # +-1 looping answers of named neurons: is the loop word one of the unit's own top-5, a collapse word, or neither
        lw = collections.Counter()
        for r in rows:
            if not (V.in_design(r) and r['kind'] == 'neuron' and r['neuron'] in named and abs(r['dose']) == 1.0 and r['d4'] < 0.5):
                continue
            t = cz.TOKEN.findall(r['text'].lower())
            if not t:
                lw['no word'] += 1
                continue
            w = collections.Counter(t).most_common(1)[0][0]
            top = {x for z, x in m[('neuron', r['neuron'])][1] if z > 0}
            lw[('collapse' if w in drop else 'other') + (' & own top-5' if w in top else '')] += 1
        L['named_pm1_loops_on'] = dict(lw)
    else:
        L['named_top_words'] = {n: [(w, round(z, 1)) for z, w in m[('neuron', n)][1][:3]] for n in sorted(named)}
    print(layer, json.dumps(L, default=str), flush=True)
    out[layer] = L
(V.HERE / 'v7_misc.json').write_text(json.dumps(out, indent=1, default=str))
