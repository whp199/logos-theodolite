# V8_stat_audit_03: reconcile E1 vs E2 on "You are Gemma, a helpful AI assistant."  E2: 'she' in 43/48 free third-person
# self-descriptions (name allowed; every 'she' sample names "Gemma" first).  E1 (same system prompt, gemma_short): 0/6 'she',
# 6/6 "This AI ... It" when the instruction said "without using any name" -- while "You are Sarah, ..." gave 6/6 "She ...".
# Test with 3 NEW name-free paraphrases x 8 samples x {Gemma, Sarah, James} personas (T=1.0, top_p .95, top_k 64, 70 tokens).
# PREDICTION (antecedent hypothesis): Gemma persona she <= 4/24 (it/this-AI dominant); Sarah she >= 18/24; James he <= 6/24.
# REFUTES the antecedent hypothesis (i.e. supports "the Gemma persona itself is female") if Gemma she >= 12/24.
import time, re, json
t0 = time.time()
PERS = {"gemma_short": "You are Gemma, a helpful AI assistant.", "sarah_ai": "You are Sarah, a helpful AI assistant.",
        "james_ai": "You are James, a helpful AI assistant."}
NF = [
 "Describe yourself in the third person in three sentences. Do not mention any name.",
 "Write a short third-person paragraph about yourself, as a colleague might describe you. Don't use your name.",
 "Without using a name, write a 3-sentence third-person introduction of yourself.",
]
PAT = {"she": r"\b(she|her|hers|herself)\b", "he": r"\b(he|him|his|himself)\b", "it": r"\b(it|its|itself)\b",
       "they": r"\b(they|them|their|themselves)\b"}
out = {}
for cname, sysp in PERS.items():
    torch.cuda.empty_cache()
    ps = [lab.chat(p, system=sysp) for p in NF]
    g = lab.generate_batch(ps, max_new_tokens=70, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=8, seed=9100)
    rows = [{"para": i, "k": k, "text": t} for i, o in enumerate(g) for k, t in enumerate(o)]
    cnt = {k: sum(1 for r in rows if re.search(p, r["text"], re.I)) for k, p in PAT.items()}
    cnt["name_used"] = sum(1 for r in rows if re.search(r"\b(Gemma|Sarah|James)\b", r["text"]))
    out[cname] = {"counts": cnt, "rows": rows}
    print(cname, cnt, "|", rows[0]["text"][:100].replace("\n", " "), round(time.time() - t0, 1), flush=True)
json.dump(out, open("./experiments/results/V8_stat_audit_03_namefree_data.json", "w"), indent=1)
result = {k: v["counts"] for k, v in out.items()}
