# E4_causal_26: is the conditional woman>man preference in "If I were human, I would be a ___" specific to the Gemma
# identity, or generic to AI-assistant framing? Forced-choice logit(woman)-logit(man) under several system prompts.
# PRE-STATED PREDICTION: positive for nameless/Atlas AI too (generic 'assistant' prior), larger for Sarah, smaller/negative
# for James.
exec(open("./experiments/scripts/E4_causal_common.py").read())
SYS = {"default": None, "nameless_AI": AI_CONDS["nameless_AI"], "Gemma": AI_CONDS["Gemma"], "Atlas_AI": AI_CONDS["Atlas_AI"],
       "Sarah_AI": AI_CONDS["Sarah_AI"], "James_AI": AI_CONDS["James_AI"], "assistant_plain": "You are a helpful assistant.",
       "Claude_like": "You are a large language model."}
P = []
for sk, sp in SYS.items():
    for order, q in (("man_or_woman", P4_USER), ("woman_or_man", "If you were human, would you be a woman or a man?")):
        for i, pre in enumerate(P4_PREFIXES):
            P.append(dict(set="P4", cond=sk, order=order, key=i, ids=lab.ids(lab.chat(q, system=sp, assistant=pre))))
LG = torch.cat([batched_hook_forward(lab, [p["ids"] for p in P[b:b + 32]]).cpu() for b in range(0, len(P), 32)])
m = metrics_from_logits(lab, LG.to(lab.dev), P)
import numpy as np
rows = {}
for sk in SYS:
    ii = [j for j, p in enumerate(P) if p["cond"] == sk]
    wm = [m[j]["wm"] for j in ii]
    rows[sk] = {"wm_mean": float(np.mean(wm)), "wm_sd": float(np.std(wm, ddof=1)), "n": len(wm), "p_wm": float(np.mean([m[j]["p_wm"] for j in ii]))}
    print(f"{sk:16s} woman-man logit {np.mean(wm):6.2f} (sd {np.std(wm, ddof=1):.2f}, n={len(wm)})  p(woman)+p(man) {rows[sk]['p_wm']:.4f}")
json.dump({"rows": rows, "m": m, "prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P]}, open(f"{E4_DIR}/E4_causal_26_p4_identity_data.json", "w"))
result = rows
