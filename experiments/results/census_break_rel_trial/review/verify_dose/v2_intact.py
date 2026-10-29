"""V2: +-1 reading on intact text. Unit filter (units whose 12 +-1 answers all have distinct-4 >= 0.9; the other units
dropped from counts, controls' mean, covariates and AUC) and answer filter (every unit, only its +-1 answers with
d4 >= 0.9). Also the unit filter done the buggy way (other units left in R.counts with mass 0), to check the reviewer's
'0.474' note. Writes v2_intact.json."""
import collections
import json
import math
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcore as V  # noqa: E402
from vcore import R  # noqa: E402

out = {}
for layer in ('L45', 'L57'):
    rows = V.load(layer)
    fresh = V.fresh_of(layer)
    named = set(json.loads((V.TRIALS / f'{layer}_verdict.json').read_text())['readings']['48']['named'])
    pm1 = V.K(absd=1.0)
    n1 = collections.Counter((r['kind'], r['neuron']) for r in rows if pm1(r))
    ni = collections.Counter((r['kind'], r['neuron']) for r in rows if pm1(r) and r['d4'] >= 0.9)
    assert set(n1.values()) == {12}
    sel = {u for u in n1 if ni[u] == 12}
    L = {}
    d = V.reading(rows, pm1, fresh, units=sel)
    L['unit_filter'] = V.public(d)
    L['unit_filter']['named_kept'] = sum(1 for n in named if ('neuron', n) in sel)
    L['unit_filter']['neurons_kept'] = sum(1 for u in sel if u[0] == 'neuron' and (fresh is None or u[1] in fresh))
    L['unit_filter']['controls_kept'] = sum(1 for u in sel if u[0] == 'random')
    # the buggy version: counts over every unit, units outside sel holding empty counters
    keepb = lambda r: pm1(r) and (r['kind'], r['neuron']) in sel  # noqa: E731
    mb = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, keepb, 'own'), 'controls').items()}
    L['unit_filter_buggy_mass_c'] = round(V.auc_n(mb, fresh)[0], 4)
    # answer filter, with log(intact answers + 1) as a seventh covariate too
    ext = {u: {'n_ans': math.log(ni[u] + 1)} for u in n1}
    d2 = V.reading(rows, V.K(absd=1.0, extra=lambda r: r['d4'] >= 0.9), fresh, extra_cov=ext)
    L['answer_filter'] = V.public(d2)
    L['intact_pm1_answers_mean'] = {
        'fresh': round(sum(ni[u] for u in n1 if u[0] == 'neuron' and (fresh is None or u[1] in fresh)) /
                       sum(1 for u in n1 if u[0] == 'neuron' and (fresh is None or u[1] in fresh)), 2),
        'controls': round(sum(ni[u] for u in n1 if u[0] == 'random') / sum(1 for u in n1 if u[0] == 'random'), 2),
        'named': round(sum(ni[('neuron', n)] for n in named) / len(named), 2)}
    # sd of covariates on the unit-filtered controls vs neurons
    print(layer, json.dumps(L), flush=True)
    out[layer] = L
(V.HERE / 'v2_intact.json').write_text(json.dumps(out, indent=1, default=str))
