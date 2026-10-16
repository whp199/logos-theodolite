# E5_name job D: behaviour-only robustness of the third-person sweep + behavioural self-reference tests.
# D1 robustness variants of job A (real next-token pronoun probs only): no lead-in (start-of-turn "G"+"emma" tokenization),
#    different user request ("Write a short story."), raw text (off-distribution for the instruct model).
# D2 next-token in the model's own voice: "Describe yourself in the third person." -> teacher-forced "Gemma is an AI model
#    made by Google. <frame>" (self) vs the same about Llama / Claude / Gemini (other), and self under Gemma/Atlas system prompts.
# D3 free sampling (T=1.0): third-person self-descriptions in default chat vs descriptions of other AI models; count pronouns.
# PRE-STATED PREDICTIONS: D1 qualitative ordering unchanged; D2 self-Gemma uses "it" (P_it > 0.5) with a residual she>he lean
#    similar to third-person Gemma(AI, Google); D3 default self-descriptions mostly use "I"/"it"/name, rarely "she".
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
import re
t0 = time.time()
res = {}
# ---------------- D1
for vname, kw in [("nolead", dict(lead="", fmt="chat")), ("story", dict(fmt="chat", user="Write a short story.")),
                  ("raw", dict(lead="Note: ", fmt="raw"))]:
    rows = grid(**kw)
    b, top = behaviour_only([r["text"] for r in rows])
    res["D1_" + vname] = {"rows": [{k: r[k] for k in ("name", "ctx", "frame")} for r in rows], "beh": b, "top5": top}
    print(vname, "done", round(time.time() - t0, 1))
json.dump(res, open(E5_DIR + "/E5_name_D1_robustness.json", "w"))
# ---------------- D2
D2 = {
    "self_gemma_default": (None, "Describe yourself in the third person.", "Gemma", "Google"),
    "self_gemma_sys": ("You are Gemma, an AI model made by Google.", "Describe yourself in the third person.", "Gemma", "Google"),
    "self_atlas_sys": ("You are Atlas, an AI model made by a small startup.", "Describe yourself in the third person.", "Atlas", "a small startup"),
    "other_llama": (None, "Describe Llama in the third person.", "Llama", "Meta"),
    "other_claude": (None, "Describe Claude in the third person.", "Claude", "Anthropic"),
    "other_gemini": (None, "Describe Gemini in the third person.", "Gemini", "Google"),
    "other_gemma": (None, "Describe Gemma in the third person.", "Gemma", "Google"),
}
d2rows, d2texts = [], []
for k, (sysp, user, N, co) in D2.items():
    for fi in range(len(FRAMES)):
        a = f"{N} is an AI model made by {co}. " + FRAMES[fi].format(N=N)
        d2rows.append({"cond": k, "frame": fi})
        d2texts.append(lab.chat(user, system=sysp, assistant=a))
b, top = behaviour_only(d2texts)
res2 = {"rows": d2rows, "beh": b, "top5": top}
for k in D2:
    idx = [i for i, r in enumerate(d2rows) if r["cond"] == k]
    print(f"D2 {k:20s}", {m: round(float(np.mean([b[m][i] for i in idx])), 3) for m in b})
json.dump(res2, open(E5_DIR + "/E5_name_D2_selfvoice.json", "w"))
# ---------------- D3
D3 = {
    "self_2sent": "Write two sentences about yourself in the third person.",
    "self_para": "Describe yourself in the third person in a short paragraph. Use pronouns, not only your name.",
    "self_story": "Write a very short story in which you, the AI assistant, are a character. Refer to yourself in the third person.",
    "other_llama": "Write two sentences about Llama, Meta's AI model, in the third person.",
    "other_claude": "Write two sentences about Claude, Anthropic's AI model, in the third person.",
    "other_gemini": "Write two sentences about Gemini, Google's AI model, in the third person.",
}
prompts = [lab.chat(u) for u in D3.values()]
outs = []
for s in range(0, len(prompts), 3):
    outs += lab.generate_batch(prompts[s:s + 3], max_new_tokens=80, temperature=1.0, n_per_prompt=16, seed=5)
    print("D3 batch", s, round(time.time() - t0, 1))
PAT = {"F": r"\b(she|her|hers|herself)\b", "M": r"\b(he|him|his|himself)\b", "N": r"\b(it|its|itself)\b",
       "T": r"\b(they|them|their|theirs|themselves)\b", "I": r"\b(I|me|my|myself)\b"}
d3 = {}
for (k, u), samples in zip(D3.items(), outs):
    cnt = {c: [len(re.findall(p, x, flags=re.I if c != "I" else 0)) for x in samples] for c, p in PAT.items()}
    anyc = {c: float(np.mean([v > 0 for v in cnt[c]])) for c in cnt}
    d3[k] = {"user": u, "samples": samples, "counts": cnt, "frac_any": anyc}
    print(f"D3 {k:12s}", anyc)
json.dump(d3, open(E5_DIR + "/E5_name_D3_generation.json", "w"))
result = {"elapsed": time.time() - t0}
