"""Q3: for each unit's top-5 words of the primary reading (all 24 answers, mass_c), split each word's z exactly into
the part from its +-1 answers and the part from its +-2 answers:
  z = (c1 + c2 - mu1 - mu2) / sqrt(mu1 + mu2 + 0.5),  part_k = (c_k - mu_k) / sqrt(mu1 + mu2 + 0.5)
(mu_k the controls' mean count of the word in their +-k answers, a control left out of its own mean), summed over the
words with z > 0 (the ones mass_c sums). Also: the share of that z on the 'attractor' words, the words the controls
themselves loop on at +-2 (top 20 words by the number of controls holding them in >= 2 of their 12 +-2 answers).
Read for the verdict's 262 named neurons at L45, all 1,022 fresh neurons, the 1,024 controls, and L57's 20 named.
Writes q3_decompose.json."""
import collections
import json
import math

import numpy as np

import common as C
from common import R

V45 = json.loads((C.TRIALS / 'L45_verdict.json').read_text())['readings']['48']
V57 = json.loads((C.TRIALS / 'L57_verdict.json').read_text())['readings']['48']
out = {}
for layer, named in (('L45', set(V45['named'])), ('L57', set(V57['named']))):
    rows = C.rows(layer)
    fresh = C.FRESH[layer]
    cnt = R.counts(rows, C.keep_all, 'own')
    c1 = R.counts(rows, C.keep_abs(1.0), 'own')
    c2 = R.counts(rows, C.keep_abs(2.0), 'own')
    m = R.mass_from_counts(cnt, 'controls')
    ctl = [u for u in cnt if u[0] == 'random']
    tot1, tot2 = collections.Counter(), collections.Counter()
    for u in ctl:
        tot1.update(c1[u])
        tot2.update(c2[u])
    # attractor words: what the controls carry in >= 2 of their +-2 answers, most controls first
    held = collections.Counter()
    for u in ctl:
        held.update(w for w, n in c2[u].items() if n >= 2)
    attract = [w for w, _ in held.most_common(20)]
    held1 = collections.Counter()
    for u in ctl:
        held1.update(w for w, n in c1[u].items() if n >= 2)

    def parts(u):
        loo = u[0] == 'random'
        den = len(ctl) - 1 if loo else len(ctl)
        p1 = p2 = zt = za = 0.0
        n1 = n2 = 0
        words = []
        for z, w in m[u][1]:
            if z <= 0:
                continue
            a1, a2 = c1[u][w], c2[u][w]
            mu1 = (tot1[w] - (a1 if loo else 0)) / den
            mu2 = (tot2[w] - (a2 if loo else 0)) / den
            s = math.sqrt(mu1 + mu2 + 0.5)
            q1, q2 = (a1 - mu1) / s, (a2 - mu2) / s
            assert abs(q1 + q2 - z) < 1e-9, (u, w, q1 + q2, z)
            p1 += q1
            p2 += q2
            zt += z
            n1 += a1
            n2 += a2
            if w in attract:
                za += z
            words.append((w, round(z, 2), a1, a2))
        return {'z': zt, 'z_pm1': p1, 'z_pm2': p2, 'cnt_pm1': n1, 'cnt_pm2': n2, 'z_on_attractors': za, 'words': words}

    groups = {'named': [('neuron', n) for n in sorted(named)],
              'fresh neurons' if fresh else 'neurons': [u for u in cnt if u[0] == 'neuron' and (fresh is None or u[1] in fresh)],
              'controls': ctl}
    L = {'attractor_words (controls holding them in >=2 of 12 +-2 answers)': [(w, held[w]) for w in attract],
         'controls holding these in >=2 of 12 +-1 answers': [(w, held1[w]) for w in attract]}
    for g, us in groups.items():
        P = [parts(u) for u in us]
        Z = sum(p['z'] for p in P)
        zp1 = np.array([p['z_pm1'] / p['z'] if p['z'] > 0 else np.nan for p in P])
        L[g] = {
            'units': len(us),
            'pooled share of top-5 z from +-1 answers': round(sum(p['z_pm1'] for p in P) / Z, 3),
            'pooled share of top-5 z from +-2 answers': round(sum(p['z_pm2'] for p in P) / Z, 3),
            'median per-unit share from +-1': round(float(np.nanmedian(zp1)), 3),
            'units with >50% of their z from +-1': int(np.nansum(zp1 > 0.5)),
            'units with >50% of their z from +-2': int(np.nansum(zp1 < 0.5)),
            'units with >80% from +-1': int(np.nansum(zp1 > 0.8)), 'units with >80% from +-2': int(np.nansum(zp1 < 0.2)),
            'share of top-5 word counts in +-1 answers': round(sum(p['cnt_pm1'] for p in P) / max(1, sum(p['cnt_pm1'] + p['cnt_pm2'] for p in P)), 3),
            'pooled share of top-5 z on attractor words': round(sum(p['z_on_attractors'] for p in P) / Z, 3),
            'mean mass': round(float(np.mean([p['z'] for p in P])), 2),
        }
        if g == 'named':
            L['named_examples'] = {u[1]: dict(zip(('z', 'z_pm1', 'z_pm2', 'words'),
                                                  (round(p['z'], 2), round(p['z_pm1'], 2), round(p['z_pm2'], 2), p['words'])))
                                   for u, p in zip(us, P)}
        # which words carry the group's top-5 z, most z first
        wz = collections.Counter()
        for p in P:
            for w, z, a1, a2 in p['words']:
                wz[w] += z
        L[g]['top words by summed z'] = [(w, round(z, 1)) for w, z in wz.most_common(25)]
    out[layer] = L
    print(layer, 'attractors', L['attractor_words (controls holding them in >=2 of 12 +-2 answers)'])
    print(layer, 'same words at +-1', L['controls holding these in >=2 of 12 +-1 answers'])
    for g in groups:
        print(layer, g, json.dumps(L[g]))
(C.HERE / 'q3_decompose.json').write_text(json.dumps(out, indent=1))
