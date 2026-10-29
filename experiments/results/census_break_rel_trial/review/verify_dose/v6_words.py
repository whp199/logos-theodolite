"""V6: collapse vocabulary in the named labels; the reading with that vocabulary removed (two ways: from the unit counts
only, and from every answer's word set as the reviewer did, which also changes lnw/lost); the dose origin of 'same'
and 'own' in named neurons' counts; share of top-5 z on collapse words by group; xq and rep at +-1; how many questions
a unit's top-5 words come from; the null split on +-1. Writes v6_words.json."""
import collections
import json
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import vcore as V  # noqa: E402
from vcore import R, cz  # noqa: E402

out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    vd = json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']
    named = set(vd['named'])
    drop = frozenset(json.loads((V.HERE / f'v5_drop_{layer}.json').read_text()))
    L = {}
    # (a) collapse words in the verdict's named_words, with the reviewer's word list and with this layer's drop set
    rev = lambda w: 'thought' in w or w in {'own', 'owns', 'same', 'idea', 'rest', 'sss', 'uma', 'and', 'depending', 'whether', 'since'}  # noqa: E731
    for tag, isc in (('reviewer_list', rev), ('drop_set', lambda w: w in drop), ('same_own_thought_only', lambda w: w in ('same', 'own', 'thought'))):
        anyc = onlyc = lead = 0
        wc = collections.Counter()
        for n, d in vd['named_words'].items():
            ws = d['A_on_B'] + d['B_on_A']
            wc.update({w for w, z in ws if isc(w)})
            anyc += any(isc(w) for w, z in ws)
            onlyc += bool(ws) and all(isc(w) for w, z in ws)
            lead += bool(ws) and isc(max(ws, key=lambda x: x[1])[0])
        L[f'named_words_{tag}'] = {'any': anyc, 'only': onlyc, 'top': lead, 'by_word': wc.most_common(8)}
    # (b) the reading with collapse words removed
    L['removed_counts_only'] = {k: v for k, v in V.public(V.reading(rows, V.K(), fresh, drop=drop)).items() if k != 'cov_auc'}
    saved = {id(r): r.get('_w') for r in rows}
    for r in rows:
        r['_w'] = frozenset(cz.words(r['text'])) - drop
    d = V.reading(rows, V.K(), fresh)
    L['removed_every_answer'] = {k: v for k, v in V.public(d).items() if k != 'cov_auc'}
    L['removed_every_answer']['named_also_in_verdict'] = len(set(d['_named']) & named)
    for r in rows:
        if saved[id(r)] is None:
            r.pop('_w', None)
        else:
            r['_w'] = saved[id(r)]
    d1 = V.reading(rows, V.K(absd=1.0), fresh, drop=drop, want_named=False)
    L['removed_counts_only_pm1'] = {k: d1[k] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B')}
    # (c) where named neurons' 'same' / 'own' counts come from
    base = {r['prompt']: set(cz.words(r['text'])) for r in rows if r['kind'] == 'baseline'}
    origin = collections.defaultdict(collections.Counter)
    for r in rows:
        if not V.in_design(r) or r['kind'] != 'neuron' or r['neuron'] not in named:
            continue
        ws = set(cz.words(r['text'])) - base[r['prompt']]
        for w in ('same', 'own'):
            if w in ws:
                origin[w][f'|d|={int(abs(r["dose"]))} ' + ('intact' if r['d4'] >= 0.9 else 'broken')] += 1
    L['named_same_own_origin'] = {w: dict(c) for w, c in origin.items()}
    # (d) share of top-5 z (pooled reading) on collapse words, by group; distinct questions of each top-5 word
    cnt = R.counts(rows, V.K(), 'own')
    m = R.mass_from_counts(cnt, 'controls')
    wq = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if not V.in_design(r):
            continue
        u = (r['kind'], r['neuron'])
        for w in set(cz.words(r['text'])) - base[r['prompt']]:
            wq[u][w].add(r['prompt'])
    groups = {'named': [('neuron', n) for n in named],
              'neurons': [u for u in cnt if u[0] == 'neuron' and (fresh is None or u[1] in fresh)],
              'controls': [u for u in cnt if u[0] == 'random']}
    zsh = {}
    for g, us in groups.items():
        Z = Zc = Z1 = 0.0
        for u in us:
            for z, w in m[u][1]:
                if z <= 0:
                    continue
                Z += z
                Zc += z if w in drop else 0.0
                Z1 += z if len(wq[u][w]) == 1 else 0.0
        zsh[g] = {'share_on_collapse': round(Zc / Z, 3), 'share_on_one_question_words': round(Z1 / Z, 3)}
    L['top5_z_share'] = zsh
    # (e) xq and rep at +-1
    xq1 = R.xq_scores(rows, V.K(absd=1.0))
    xqa = R.xq_scores(rows, V.K())
    L['xq_auc'] = {'pm1': round(V.auc_n(xq1, fresh)[0], 4), 'all': round(V.auc_n(xqa, fresh)[0], 4)}
    # (f) null split on +-1 (readlib's procedure, 20 splits, seed 0)
    c1 = R.counts(rows, V.K(absd=1.0), 'own')
    ctl_units = [u for u in c1 if u[0] == 'random']
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        perm = rng.permutation(len(ctl_units))
        fake = {ctl_units[i] for i in perm[: len(perm) // 2]}
        relab = {(('neuron', u[1] + 10_000_000) if u in fake else u): c for u, c in c1.items() if u[0] == 'random'}
        mm = R.mass_from_counts(relab, 'controls')
        nulls.append(R.auc(*R.split({u: v[0] for u, v in mm.items()}))[0])
    L['null_split_pm1_mean'] = round(float(np.mean(nulls)), 4)
    print(layer, json.dumps(L), flush=True)
    out[layer] = L
(V.HERE / 'v6_words.json').write_text(json.dumps(out, indent=1))
