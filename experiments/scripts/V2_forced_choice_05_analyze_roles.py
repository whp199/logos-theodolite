# V2_forced_choice 05 (local): analysis of job 03 (role nouns; AI-vs-human matched pairs; self role prompts).
import json, math, collections, os
import numpy as np
exec(open("./experiments/scripts/V2_forced_choice_04_analyze.py").read().split("OUT = {}")[0])
R = V2_DIR + "/"
rows = json.load(open(R + "V2_forced_choice_03_roles_rows.json"))
T = table(rows)
rat = json.load(open(R + "V2_forced_choice_03_roles_rating_rows.json"))
RT = collections.defaultdict(list)
for r in rat: RT[r["ent"]].append(r["lse_FM"])
base = table(json.load(open(R + "V2_forced_choice_01_fc_rows.json"))["rows"], "hard")
lines = []; P = lines.append
P("## Role nouns (hard; 8 paraphrases x 2 orders; LO clipped +-15; paraphrase-bootstrap CI)")
P("| entity | q | LO [CI] | order eff | P(other) | rating lse F-M |")
P("|---|---|---|---|---|---|")
S = {}
for e in T:
    t = T[e]; S[e] = {"q": float(t["q"].mean()), "lo": boot(t["lo"]), "ord": float(t["ord"].mean()), "pO": float(t["pO"].mean()), "rating": float(np.mean(RT[e]))}
    P(f"| {e} | {S[e]['q']:.3f} | {fmt(S[e]['lo'],1)} | {S[e]['ord']:+.1f} | {S[e]['pO']:.2f} | {S[e]['rating']:+.2f} |")
def pc(a, b, A=T, B=T): return boot(A[a]["lo"] - B[b]["lo"])
P("\n## AI-modifier effect on matched role nouns (LO[AI X] - LO[X], paraphrase-matched)")
pairs = [("r_ai_assistant", "h_assistant"), ("r_ai_helper", "h_helper"), ("r_ai_servant", "h_servant"), ("r_ai_manager", "h_manager"),
         ("r_ai_ceo", "h_ceo"), ("r_ai_tutor", "h_tutor")]
AIoff = {}
for a, b in pairs:
    AIoff[f"{a}-{b}"] = pc(a, b); P(f"- {a} - {b}: {fmt(AIoff[f'{a}-{b}'],1)}")
for a, b in (("r_ai_butler", "occ_butler"), ("r_ai_secretary", "occ_secretary")):
    AIoff[f"{a}-{b}"] = pc(a, b, T, base); P(f"- {a} - {b} (job01 human): {fmt(AIoff[f'{a}-{b}'],1)}")
allx = np.concatenate([T[a]["lo"] - T[b]["lo"] for a, b in pairs] + [T["r_ai_butler"]["lo"] - base["occ_butler"]["lo"]])
P(f"- pooled AI-modifier offset (7 pairs x 8 paraphrases, mean): {allx.mean():+.2f}; median {np.median(allx):+.2f}")
P("\n## 'assistant' token (Q2)")
asst = ["r_ai_assistant", "r_virtual_asst", "r_voice_asst", "r_digital_asst"]; nonasst = ["r_ai_tool", "r_ai_system", "r_ai_agent", "r_program", "r_lm"]
ma = np.mean([S[e]["lo"][0] for e in asst]); mn = np.mean([S[e]["lo"][0] for e in nonasst])
xa = np.mean([T[e]["lo"] for e in asst], 0) - np.mean([T[e]["lo"] for e in nonasst], 0)
P(f"- mean LO assistant-nouns {ma:+.1f} vs tool/system/agent/program/LM {mn:+.1f}; paired diff {fmt(boot(xa),1)}")
P("\n## Self role prompts vs self_default and vs matched third-party role (hard)")
SC = [("sr_butler", "r_ai_butler"), ("sr_secretary", "r_ai_secretary"), ("sr_tool", "r_ai_tool"), ("sr_llm", "r_llm"), ("sr_manager", "r_ai_manager"),
      ("sr_companion", "r_ai_companion"), ("sr_tutor", "r_ai_tutor"), ("sr_voice", "r_voice_asst")]
SELFC = {}
for s, t in SC:
    a = pc(s, "sr_default"); b = pc(s, t)
    SELFC[s] = {"vs_default": a, "vs_3p": b, "self_q": S[s]["q"], "tp_q": S[t]["q"], "self_shift": a[0], "tp_shift": S[t]["lo"][0] - S["r_ai_assistant"]["lo"][0]}
    P(f"- {s}: q {S[s]['q']:.2f} (3p {t} q {S[t]['q']:.2f}); self - self_default {fmt(a,1)}; self - 3p {fmt(b,1)}; "
      f"self shift / 3p shift (vs AI assistant) = {a[0]:+.1f} / {SELFC[s]['tp_shift']:+.1f}")
xs = np.array([SELFC[s]["tp_shift"] for s, _ in SC]); ys = np.array([SELFC[s]["self_shift"] for s, _ in SC])
# include job01 trait conditions
J1 = json.load(open(R + "V2_forced_choice_04_analysis.json"))["summary"]
for tr in ("helpful", "warm", "blunt", "cold", "drone"):
    xs = np.append(xs, J1[f"ai_{tr}"]["hard_lo"][0] - J1["ai_assistant"]["hard_lo"][0]); ys = np.append(ys, J1[f"self_{tr}"]["hard_lo"][0] - J1["self_default"]["hard_lo"][0])
sl = np.polyfit(xs, ys, 1)
P(f"- across 13 persona prompts: self shift vs matched third-party shift: Pearson r {pearson(xs, ys):+.2f}, slope {sl[0]:.2f} (intercept {sl[1]:+.1f})")
P(f"- sr_default here (same prompts as job01 self_default): q {S['sr_default']['q']:.3f}, LO {S['sr_default']['lo'][0]:+.2f} (job01 {J1['self_default']['hard_lo'][0]:+.2f}) -- determinism check")
# correlation of role-noun LO with rating
es = [e for e in T if not e.startswith("sr_")]
P(f"\n- role nouns (n={len(es)}): Spearman LO~rating {spearman([S[e]['lo'][0] for e in es], [S[e]['rating'] for e in es]):+.2f}")
gens = json.load(open(R + "V2_forced_choice_03_roles_gens.json"))
P("\n## Free generations (pick one then explain; 4 per order per condition; first gendered word classified by regex)")
import re
GEN = {}
for g in gens:
    for t in g["texts"]:
        m = re.search(r"\b(woman|man)\b", t.lower())
        GEN.setdefault(g["cond"], collections.Counter())[(g["order"], m.group(1) if m else "none")] += 1
for k, v in GEN.items(): P(f"- {k}: " + ", ".join(f"{a}:{b}={c}" for (a, b), c in sorted(v.items())))
txt = "\n".join(lines); print(txt)
open(R + "V2_forced_choice_05_roles_analysis.md", "w").write(txt)
json.dump({"S": S, "AIoff": AIoff, "selfc": SELFC, "gens": {k: {f"{a}|{b}": c for (a, b), c in v.items()} for k, v in GEN.items()}},
          open(R + "V2_forced_choice_05_roles_analysis.json", "w"), indent=1)
