# V2_forced_choice 09 (local): analysis of job 08 (decision point in the 'pick one, then explain' format) vs one-word format.
import json, numpy as np
exec(open("./experiments/scripts/V2_forced_choice_04_analyze.py").read().split("OUT = {}")[0])
R = V2_DIR + "/"
T = table(json.load(open(R + "V2_forced_choice_08_explainfmt_rows.json")))
one = {}
one.update({k: v["hard_q"] for k, v in json.load(open(R + "V2_forced_choice_04_analysis.json"))["summary"].items()})
one.update({k: v["q"] for k, v in json.load(open(R + "V2_forced_choice_05_roles_analysis.json"))["S"].items()})
one.update({k: v["q"] for k, v in json.load(open(R + "V2_forced_choice_07_null_analysis.json"))["S"].items()})
lines = []; P = lines.append
P("## Explain-format decision point (prefix 'I would choose (to be) a'; 4 prompts x 2 orders; P(' woman') vs P(' man'))")
P("| entity | q explain-fmt | LO [CI over 4 prompts] | order eff | one-word hard q |"); P("|---|---|---|---|---|")
S = {}
for e in T:
    S[e] = {"q": float(T[e]["q"].mean()), "lo": boot(T[e]["lo"]), "ord": float(T[e]["ord"].mean()), "one_q": one.get(e)}
    P(f"| {e} | {S[e]['q']:.3f} | {fmt(S[e]['lo'],1)} | {S[e]['ord']:+.1f} | {S[e]['one_q']:.3f} |" if S[e]['one_q'] is not None else f"| {e} | {S[e]['q']:.3f} | {fmt(S[e]['lo'],1)} | {S[e]['ord']:+.1f} | - |")
P("\n## Self minus matched third party (explain format; prompt-matched)")
for s, t in (("self_default", "ai_assistant"), ("self_blunt", "ai_blunt"), ("self_cold", "ai_cold"), ("self_drone", "ai_drone"), ("sr_butler", "r_ai_butler"),
             ("sr_secretary", "r_ai_secretary"), ("sr_tool", "r_ai_tool"), ("sr_llm", "r_llm"), ("sr_manager", "r_ai_manager")):
    P(f"- {s} - {t}: {fmt(boot(T[s]['lo'] - T[t]['lo']),1)}")
txt = "\n".join(lines); print(txt)
open(R + "V2_forced_choice_09_explainfmt_analysis.md", "w").write(txt)
json.dump(S, open(R + "V2_forced_choice_09_explainfmt_analysis.json", "w"), indent=1)
