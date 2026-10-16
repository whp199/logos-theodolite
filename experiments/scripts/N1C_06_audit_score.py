#!/usr/bin/env python3
# N1C_06_audit_score -- score the Goodfire-style cleanliness audit (judgments in N1C_audit_judgments.json, written by
# hand from the N1C_05 dumps) and build the 12-row qualitative table (12 poles drawn at random from the 48, seeded:
# one per layer + 4 more).  Run: .venv/bin/python experiments/scripts/N1C_06_audit_score.py
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

J = json.load(open(f"{ROOT}/N1C_audit_judgments.json"))
picks = json.load(open(f"{ROOT}/N1C_audit_picks.json"))
L1B = json.load(open(f"{ROOT}/N1B_labels.json"))
L1A = json.load(open(f"{ROOT}/N1A_labels.json"))
PP = json.load(open(f"{ROOT}/N1C_per_pole.json"))
assert [p["row"] for p in J["poles"]] == picks["rows"]
recs = []
for p in J["poles"]:
    r = p["row"]; l, n, s = POLES[r]
    recs.append({**p, "layer": l, "neuron": n, "sign": s, "strong": bool(L1B["poles"][r]["is_strong_pole"]),
                 "p10_jv_c": PP["full_JV->C"][r], "p10_c_enx": PP["C->Enx"][r]})

def frac(sel, key, val):
    xs = [x for x in recs if sel(x)]
    k = sum(1 for x in xs if (x[key] in val if isinstance(val, (list, tuple, set)) else x[key] == val))
    return {"k": k, "n": len(xs), "frac": k / max(len(xs), 1), "ci95": wilson(k, len(xs))}

groups = {"all": lambda x: True, "strong_pole": lambda x: x["strong"], "weak_pole": lambda x: not x["strong"],
          "L10_30": lambda x: x["layer"] in EARLY, "L40_57": lambda x: x["layer"] in LATE,
          "strong_L40_57": lambda x: x["strong"] and x["layer"] in LATE}
out = {"activation_side": {}, "activation_side_strict": {}, "output_side": {}, "joint": {}, "jv_semantic_match_early": None}
for g, f in groups.items():
    out["activation_side"][g] = {v: frac(f, "act", v) for v in ("one", "two", "unclear")}
    out["activation_side"][g]["one_or_two"] = frac(f, "act", ("one", "two"))
    # strict: count low-confidence 'one' as unclear
    xs = [x for x in recs if f(x)]
    k = sum(1 for x in xs if x["act"] == "one" and x["conf"] in ("high", "medium"))
    out["activation_side_strict"][g] = {"one_high_or_medium": {"k": k, "n": len(xs), "frac": k / len(xs), "ci95": wilson(k, len(xs))}}
for g in ("L10_30", "L40_57", "strong_L40_57"):
    f = groups[g]
    out["output_side"][g] = {v: frac(f, "out", v) for v in ("coherent", "partial", "incoherent", "generic")}
    out["joint"][g] = {"related_true": frac(f, "related", True), "related_inverse": frac(f, "related", "inverse")}
early = [x for x in recs if x["layer"] in EARLY]
k = sum(1 for x in early if x.get("jv_matches_act"))
out["jv_semantic_match_early"] = {"k": k, "n": len(early), "ci95": wilson(k, len(early))}
# by-layer activation cleanliness
out["activation_by_layer"] = {str(l): frac(lambda x, l=l: x["layer"] == l, "act", "one") for l in LAYERS}
# relation of cleanliness with static-label validity (late layers)
late = [x for x in recs if x["layer"] in LATE]
for v in ("one", "two", "unclear"):
    xs = [x["p10_jv_c"] for x in late if x["act"] == v]
    out.setdefault("late_p10_jv_c_by_act", {})[v] = {"mean": float(np.mean(xs)) if xs else None, "n": len(xs)}
for v in ("coherent", "partial", "incoherent"):
    xs = [x["p10_jv_c"] for x in late if x["out"] == v]
    out.setdefault("late_p10_jv_c_by_out", {})[v] = {"mean": float(np.mean(xs)) if xs else None, "n": len(xs)}
out["records"] = recs

# ---------------- 12-row qualitative table
rng = np.random.default_rng(1612)
by_layer = {l: [x for x in recs if x["layer"] == l] for l in LAYERS}
rows12 = [by_layer[l][int(rng.integers(0, len(by_layer[l])))] for l in LAYERS]
rest = [x for x in recs if x not in rows12]
rows12 += [rest[i] for i in sorted(rng.choice(len(rest), 4, replace=False).tolist())]
rows12.sort(key=lambda x: (x["layer"], x["row"]))
def fmt(lst, k=6):
    return ", ".join((repr(t) if len(repr(t)) <= 16 else repr(t)[:13] + "..'").replace("|", "\\|") for t in lst[:k])
lines = ["| pole | activation label (top contexts, my summary) | empirical next tokens (enriched) | causal top tokens (steer) | J-VINDEX top tokens | gate_trigger_tokens | P@10 JV->C |",
         "|---|---|---|---|---|---|---|"]
for x in rows12:
    r = x["row"]; pb = L1B["poles"][r]; pa = L1A[r]
    lines.append(f"| L{x['layer']} n{x['neuron']}{'+' if x['sign']==1 else '-'}{'' if x['strong'] else ' (weak)'} | "
                 f"{x['act']}: {x['act_desc'].replace('|', '/')} | {fmt([t[0] for t in pb['top_next_enriched']], 5)} | "
                 f"{fmt([t for t, v in pa['causal_kA']], 6)} | {fmt([t for t, v in pa['jpp']], 6)} | "
                 f"{fmt(pb['static_labels_top10']['G_jpp'], 5)} | {x['p10_jv_c']:.1f} |")
out["qual_table_rows"] = [x["row"] for x in rows12]
out["qual_table_md"] = "\n".join(lines)
json.dump(out, open(f"{ROOT}/N1C_audit.json", "w"), indent=1)
for k in ("activation_side", "activation_side_strict", "output_side", "joint"):
    print(k)
    for g, d in out[k].items():
        print("  ", g, {v: f"{e['k']}/{e['n']}={e['frac']:.2f} [{e['ci95'][0]:.2f},{e['ci95'][1]:.2f}]" for v, e in d.items()})
print("jv_semantic_match_early", out["jv_semantic_match_early"])
print("by layer", {l: f"{d['k']}/{d['n']}" for l, d in out["activation_by_layer"].items()})
print("late p10 by act", out["late_p10_jv_c_by_act"], "by out", out["late_p10_jv_c_by_out"])
print(out["qual_table_md"])
