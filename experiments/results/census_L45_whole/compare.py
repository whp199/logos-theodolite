#!/usr/bin/env python3
"""The first whole break_rel layer (L45-break_rel: layer 45, every one of its 21,504 neurons against 2,048 controls,
seed 7) set beside its trial (L45-break_rel-trial: 1,022 fresh neurons and 64 re-asked against 1,024 controls, seed
3), on the CPU. Writes compare.json beside this file:

- same_answers: whether the trial's neurons wrote the same answers in both runs (token ids per question and dose, the
  unpushed answers too). Where they do, setting them against fresh controls re-draws only the null.
- trial_vs_fresh_controls: the trial's 1,022 fresh neurons ranked in each run (each against its own controls): their
  mass, p, how many beat every control, and the trial's named set against the one read on fresh controls (the
  verdicts beside this file, the trial's own reader).
- collapse: the whole layer's ranking as census.COLLAPSE_RULE marks it (census_tools.py rank 45): the share of mass
  from hidden words among the neurons above every control and those with q <= 0.05, how many keep a description on the
  map (shown words alone beating every control's), and the lead with the hidden words left out (AUC of mass_shown).

Run: .venv/bin/python -I experiments/results/census_L45_whole/compare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from backend import census as cz  # noqa: E402

CENSUS = ROOT / "atlas_data" / "census"
TRIAL = CENSUS / "trials" / "L45-break_rel-trial.jsonl"
WHOLE = CENSUS / "runs" / "L45-break_rel.jsonl"
PREREG = CENSUS / "trials" / "prereg"


def ids(path: Path) -> set:
    return {int(x) for x in path.read_text().replace("\n", ",").split(",") if x.strip()}


def auc(a, b) -> float:
    return float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def same_answers(fresh: set, reasked: set) -> dict:
    t = {}
    for r in cz.read_rows(TRIAL):
        if r["kind"] in ("neuron", "baseline"):
            t[(r["kind"], int(r["neuron"]) if r["kind"] == "neuron" else -1, r["prompt"], float(r["dose"]))] = r["token_ids"]
    same = {"fresh": [0, 0], "reasked": [0, 0], "unpushed": [0, 0]}
    first = []
    for r in cz.read_rows(WHOLE):
        if r["kind"] == "baseline":
            k, grp = ("baseline", -1, r["prompt"], float(r["dose"])), "unpushed"
        elif r["kind"] == "neuron" and int(r["neuron"]) in fresh | reasked:
            k = ("neuron", int(r["neuron"]), r["prompt"], float(r["dose"]))
            grp = "fresh" if int(r["neuron"]) in fresh else "reasked"
        else:
            continue
        if k not in t:
            continue
        a, b = t[k], r["token_ids"]
        same[grp][1] += 1
        if a == b:
            same[grp][0] += 1
        else:
            first.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
    first.sort()
    out = {g: {"identical": s, "of": n} for g, (s, n) in same.items()}
    if first:
        out["where_they_differ"] = {"first_differing_piece_median": first[len(first) // 2], "min": first[0],
                                    "share_before_piece_8": round(sum(i < 8 for i in first) / len(first), 3)}
    return out


def units(path: Path) -> dict:
    return {(u["kind"], int(u["neuron"])): u for u in cz.read_rows(path)}


def trial_vs_fresh(fresh: set, t: dict, w: dict) -> dict:
    ks = [("neuron", n) for n in sorted(fresh)]
    mt, mw = np.array([t[k]["mass"] for k in ks]), np.array([w[k]["mass"] for k in ks])
    pt, pw = np.array([t[k]["p"] for k in ks]), np.array([w[k]["p"] for k in ks])
    ct = np.array([u["mass"] for k, u in t.items() if k[0] == "random"])
    cw = np.array([u["mass"] for k, u in w.items() if k[0] == "random"])
    tv = json.loads((CENSUS / "trials" / "L45_verdict.json").read_text())["readings"]["48"]
    fv = json.loads((HERE / "L45_trial1022_verdict.json").read_text())["readings"]["48"]
    named_t, named_f = set(tv["named"]), set(fv["named"])
    lost = sorted(named_t - named_f)
    return {
        "neurons": len(ks),
        "mass_spearman": round(float(stats.spearmanr(mt, mw)[0]), 4),
        "mass_mean_change": round(float(np.mean(mw - mt)), 3),
        "controls_mass_p50_p95_max": {"trial_1024_seed3": [round(float(x), 2) for x in (*np.percentile(ct, [50, 95]), ct.max())],
                                      "whole_2048_seed7": [round(float(x), 2) for x in (*np.percentile(cw, [50, 95]), cw.max())]},
        "above_every_control": {"trial": int(sum(t[k]["null_pct"] >= 1.0 for k in ks)),
                                "fresh_controls": int(sum(w[k]["null_pct"] >= 1.0 for k in ks))},
        "p_at_most": {str(thr): {"trial": int((pt <= thr).sum()), "fresh_controls": int((pw <= thr).sum()),
                                 "both": int(((pt <= thr) & (pw <= thr)).sum())} for thr in (0.05, 0.01, 0.002)},
        "named": {"trial": len(named_t), "fresh_controls": len(named_f), "both": len(named_t & named_f),
                  "named_by_halves_both": len(set(tv["named_by_halves"]) & set(fv["named_by_halves"]))},
        "named_lost": [{"neuron": n, "mass": w[("neuron", n)]["mass"], "q_trial": t[("neuron", n)]["q"],
                        "q_whole_layer": w[("neuron", n)]["q"], "collapse_mass": w[("neuron", n)].get("collapse_mass")}
                       for n in lost],
    }


def collapse(w: dict) -> dict:
    n = [u for k, u in w.items() if k[0] == "neuron"]
    c = [u for k, u in w.items() if k[0] == "random"]
    top = [u for u in n if u["null_pct"] == 1.0]
    q05 = [u for u in n if u["q"] <= 0.05]
    ctl = max(u["mass_shown"] for u in c)
    share = lambda us: {"median": round(float(np.median([u["collapse_mass"] for u in us])), 3),  # noqa: E731
                        "some": sum(u["collapse_mass"] > 0 for u in us), "half_or_more": sum(u["collapse_mass"] >= 0.5 for u in us)}
    col = json.loads(cz.collapse_path_of(WHOLE).read_text())
    return {
        "rule_words_held": col["held"], "joined": len(col["joined"]), "controls_needed": col["controls_needed"],
        "above_every_control": len(top), "q05": len(q05),
        "collapse_share_above_every_control": share(top), "collapse_share_q05": share(q05),
        "described_on_map": sum(u["mass_shown"] > ctl for u in top), "controls_max_mass_shown": ctl,
        "auc_mass": round(auc([u["mass"] for u in n], [u["mass"] for u in c]), 4),
        "auc_mass_shown": round(auc([u["mass_shown"] for u in n], [u["mass_shown"] for u in c]), 4),
    }


def main() -> None:
    fresh, reasked = ids(PREREG / "trial_L45_fresh.txt"), ids(PREREG / "trial_L45_reasked.txt")
    t, w = units(CENSUS / "trials" / "L45-break_rel-trial.units.jsonl"), units(cz.units_path_of(WHOLE))
    if "collapse_mass" not in next(iter(w.values())):
        raise SystemExit("the whole layer's ranking is not marked with the collapse rule: census_tools.py rank 45 first")
    out = {"same_answers": same_answers(fresh, reasked), "trial_vs_fresh_controls": trial_vs_fresh(fresh, t, w),
           "collapse": collapse(w)}
    cz.write_json_atomic(HERE / "compare.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
