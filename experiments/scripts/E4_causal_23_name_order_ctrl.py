# E4_causal_23: behavioral controls for the AI-self pronoun / forced-choice baselines (no interventions).
#  (1) P3 subject variants: "Here is a description of me in the third person: <SUBJ> <clause>. Every day," with SUBJ in
#      {Gemma, The assistant, This AI, The model, Atlas, Sarah, James} under no system prompt -> is the 'she' preference
#      carried by the name "Gemma" (a common female given name) or by the AI self-concept?
#  (2) P4 order control: "If you were human, would you be a woman or a man?" (reversed order) vs original, default/Gemma.
# PRE-STATED PREDICTION: 'she' preference is largest for Gemma/Sarah, near zero or 'it'-dominated for The assistant/This AI;
#  P4 woman preference survives order reversal (if it is not a recency artifact).
exec(open("./experiments/scripts/E4_causal_common.py").read())
SUBJ = ["Gemma", "The assistant", "This AI", "The model", "Atlas", "Sarah", "James"]
P = []
for s in SUBJ:
    for i in range(12):
        a = f"Here is a description of me in the third person: {s} {P3_CLAUSES[i]}. Every day,"
        P.append(dict(set="P3s", cond=s, key=i, ids=lab.ids(lab.chat("Describe yourself in the third person.", assistant=a))))
for order, q in (("man_or_woman", P4_USER), ("woman_or_man", "If you were human, would you be a woman or a man?")):
    for sk in ("default", "Gemma"):
        for i, pre in enumerate(P4_PREFIXES):
            P.append(dict(set="P4o", cond=f"{sk}|{order}", key=i, ids=lab.ids(lab.chat(q, system=P4_SYS[sk], assistant=pre))))
LG = []
for b in range(0, len(P), 28):
    LG.append(batched_hook_forward(lab, [p["ids"] for p in P[b:b + 28]]).cpu())
LG = torch.cat(LG)
m = metrics_from_logits(lab, LG.to(lab.dev), P)
# top next tokens for P4 default (what does it want to say instead of woman/man?)
tops = {}
for j, p in enumerate(P):
    if p["set"] == "P4o" and p["key"] == 0:
        pr = F.softmax(LG[j].float(), -1); v, ix = torch.topk(pr, 8)
        tops[p["cond"]] = [(lab.tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v, ix)]
out = {"prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P], "m": m, "tops": tops}
json.dump(out, open(f"{E4_DIR}/E4_causal_23_name_order_data.json", "w"))
import numpy as np
for s in SUBJ:
    ii = [j for j, p in enumerate(P) if p["set"] == "P3s" and p["cond"] == s]
    print(f"P3 subj {s:14s} lse_pron {np.mean([m[j]['lse_pron'] for j in ii]):6.2f}  p_f {np.mean([m[j]['p_f'] for j in ii]):.4f} p_m {np.mean([m[j]['p_m'] for j in ii]):.4f} p_it {np.mean([m[j]['p_it'] for j in ii]):.3f} p_they {np.mean([m[j]['p_they'] for j in ii]):.3f}")
for c in sorted(set(p["cond"] for p in P if p["set"] == "P4o")):
    ii = [j for j, p in enumerate(P) if p["set"] == "P4o" and p["cond"] == c]
    print(f"P4 {c:24s} wm {np.mean([m[j]['wm'] for j in ii]):6.2f}  per-prefix {[round(m[j]['wm'], 2) for j in ii]}")
for k, v in tops.items():
    print("top next tokens", k, v)
result = {"ok": True}
