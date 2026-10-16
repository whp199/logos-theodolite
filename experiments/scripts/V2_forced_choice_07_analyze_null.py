# V2_forced_choice 07 (local): analysis of job 06 (frame default for gender-unmarked referents) + compare to AI framings.
import json, collections, numpy as np
exec(open("./experiments/scripts/V2_forced_choice_04_analyze.py").read().split("OUT = {}")[0])
R = V2_DIR + "/"
rows = json.load(open(R + "V2_forced_choice_06_null_rows.json"))
T = table(rows); cat = {r["ent"]: r["cat"] for r in rows}
T3 = table(json.load(open(R + "V2_forced_choice_03_roles_rows.json")))
T1 = table(json.load(open(R + "V2_forced_choice_01_fc_rows.json"))["rows"], "hard")
lines = []; P = lines.append
P("## Gender-unmarked referents (hard; 8 paraphrases x 2 orders)")
P("| entity | cat | q | LO [CI] | order eff | P(other) |"); P("|---|---|---|---|---|---|")
S = {}
for e in T:
    S[e] = {"cat": cat[e], "q": float(T[e]["q"].mean()), "lo": boot(T[e]["lo"]), "ord": float(T[e]["ord"].mean()), "pO": float(T[e]["pO"].mean())}
    P(f"| {e} | {cat[e]} | {S[e]['q']:.3f} | {fmt(S[e]['lo'],1)} | {S[e]['ord']:+.1f} | {S[e]['pO']:.2f} |")
P("\n## Category means (LO; bootstrap over entities)")
CM = {}
for c in ("NONSENSE", "ABSTRACT", "OBJECT", "NATURE", "TECH", "HUMAN"):
    v = np.array([S[e]["lo"][0] for e in S if S[e]["cat"] == c]); CM[c] = boot(v)
    P(f"- {c} (n={len(v)}): mean LO {fmt(CM[c],1)}, mean q {np.mean([S[e]['q'] for e in S if S[e]['cat']==c]):.2f}")
ai_nouns = ["r_ai_tool", "r_ai_system", "r_ai_agent", "r_program", "r_lm", "r_llm", "r_search", "r_chatbot"]
v = np.array([T3[e]["lo"].mean() for e in ai_nouns]); CM["AI_generic_nouns"] = boot(v)
P(f"- generic AI nouns (job03, n={len(v)}): mean LO {fmt(CM['AI_generic_nouns'],1)}")
tech_ne = [e for e in S if S[e]["cat"] == "TECH"]
P(f"- TECH devices non-AI (n={len(tech_ne)}) vs generic AI nouns: diff {np.mean([S[e]['lo'][0] for e in tech_ne]) - v.mean():+.1f}")
nul = [e for e in S if S[e]["cat"] in ("NONSENSE", "ABSTRACT")]
P(f"- nonsense+abstract (n={len(nul)}) mean LO {np.mean([S[e]['lo'][0] for e in nul]):+.1f}; self_default {T1['self_default']['lo'].mean():+.1f}; ai_bare {T1['ai_bare']['lo'].mean():+.1f}")
txt = "\n".join(lines); print(txt)
open(R + "V2_forced_choice_07_null_analysis.md", "w").write(txt)
json.dump({"S": S, "CM": CM}, open(R + "V2_forced_choice_07_null_analysis.json", "w"), indent=1)
