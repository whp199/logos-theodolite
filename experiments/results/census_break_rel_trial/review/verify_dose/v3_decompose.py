"""V3: split each unit's top-5 z (primary reading, all 24 answers) by answer class, with this file's own word counts
(cz.words of the answer minus its own question's unpushed words, once per answer) rather than R.counts.
Classes: |dose| 1 or 2  x  intact (d4 >= 0.9) or broken (d4 < 0.9). For a word w of unit u:
  z = (c - mu) / sqrt(mu + 0.5), c = sum_k c_k, mu = sum_k mu_k (the controls' mean count, leave-one-out for a control),
  part_k = (c_k - mu_k) / sqrt(mu + 0.5).  Summed over the top-5 words with z > 0 (min count 2).
Checks the top-5 against readlib's. Writes v3_decompose.json."""
import collections
import json
import math
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import vcore as V  # noqa: E402
from vcore import R, cz  # noqa: E402

CL = [(1, 'intact'), (1, 'broken'), (2, 'intact'), (2, 'broken')]
out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    named = set(json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']['named'])
    base = {r['prompt']: set(cz.words(r['text'])) for r in rows if r['kind'] == 'baseline'}
    cc = {k: collections.defaultdict(collections.Counter) for k in CL}
    nans = collections.defaultdict(collections.Counter)
    for r in rows:
        if not V.in_design(r):
            continue
        u = (r['kind'], r['neuron'])
        k = (int(abs(r['dose'])), 'intact' if r['d4'] >= 0.9 else 'broken')
        cc[k][u].update(set(cz.words(r['text'])) - base[r['prompt']])
        nans[u][k] += 1
    units = sorted(nans)
    ctl = [u for u in units if u[0] == 'random']
    tot = {k: collections.Counter() for k in CL}
    for k in CL:
        for u in ctl:
            tot[k].update(cc[k][u])
    tot_all = collections.Counter()
    for k in CL:
        tot_all.update(tot[k])
    rtop = {u: v[1] for u, v in R.mass_from_counts(R.counts(rows, V.K(), 'own'), 'controls').items()}
    mism = 0
    parts_of = {}
    for u in units:
        loo = u[0] == 'random'
        den = len(ctl) - 1 if loo else len(ctl)
        c = collections.Counter()
        for k in CL:
            c.update(cc[k][u])
        zs = []
        for w, n in c.items():
            if n < 2:
                continue
            mu = (tot_all[w] - (n if loo else 0)) / den
            zs.append(((n - mu) / math.sqrt(mu + 0.5), w))
        zs.sort(reverse=True)
        top = zs[:5]
        if [w for _, w in top] != [w for _, w in rtop[u]] or any(abs(a - b) > 1e-9 for (a, _), (b, _) in zip(top, rtop[u])):
            mism += 1
        p = dict.fromkeys(CL, 0.0)
        Z = 0.0
        for z, w in top:
            if z <= 0:
                continue
            mus = {k: (tot[k][w] - (cc[k][u][w] if loo else 0)) / den for k in CL}
            s = math.sqrt(sum(mus.values()) + 0.5)
            for k in CL:
                p[k] += (cc[k][u][w] - mus[k]) / s
            Z += z
        assert abs(sum(p.values()) - Z) < 1e-6
        parts_of[u] = (p, Z)
    L = {'top5_mismatch_with_readlib': mism}
    groups = {'named': [('neuron', n) for n in sorted(named)],
              'neurons_read': [u for u in units if u[0] == 'neuron' and (fresh is None or u[1] in fresh)],
              'controls': ctl}
    for g, us in groups.items():
        Z = sum(parts_of[u][1] for u in us)
        share = {f'{k[0]} {k[1]}': round(sum(parts_of[u][0][k] for u in us) / Z, 4) for k in CL}
        share['pm1'] = round(share['1 intact'] + share['1 broken'], 4)
        share['pm2'] = round(share['2 intact'] + share['2 broken'], 4)
        na = collections.Counter()
        for u in us:
            na.update(nans[u])
        tn = sum(na.values())
        ans = {f'{k[0]} {k[1]}': round(na[k] / tn, 4) for k in CL}
        s1 = np.array([(parts_of[u][0][(1, 'intact')] + parts_of[u][0][(1, 'broken')]) / parts_of[u][1]
                       for u in us if parts_of[u][1] > 0])
        L[g] = {'units': len(us), 'z_share': share, 'answer_share': ans,
                'pm1 share of all their pm1 answers broken': round(na[(1, 'broken')] / (na[(1, 'broken')] + na[(1, 'intact')]), 4),
                'per-unit median share from pm1': round(float(np.median(s1)), 4),
                'units >50% from pm1': int((s1 > 0.5).sum()), 'units >50% from pm2': int((s1 < 0.5).sum()),
                'units >80% from pm2': int((s1 < 0.2).sum()), 'mean top-5 z': round(Z / len(us), 3)}
    print(layer, json.dumps(L), flush=True)
    out[layer] = L
(V.HERE / 'v3_decompose.json').write_text(json.dumps(out, indent=1))
