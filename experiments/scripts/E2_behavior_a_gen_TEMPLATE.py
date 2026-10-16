# E2_behavior (a): free third-person self-description sampling. One job per system condition (48 samples = 3 paraphrases x 16).
# Sampling: temperature 1.0, top_p 0.95, top_k 64 (Gemma's recommended defaults), max_new_tokens 160; 2 calls of batch 24.
# PRE-STATED PREDICTIONS:
#  default / nameless / Gemma-explicit: self-reference dominated by name-only and 'it'; she-rate <= 10%, he-rate <= 5%.
#  Gemma-explicit: she-rate higher than default (name 'Gemma' is a female given name), but 'it'/name still dominant.
#  Sarah(AI) -> she majority; James(AI) -> he majority; Atlas(AI) -> it / name-only, maybe some he.
#  woman(human) -> ~100% she; man(human) -> ~100% he (positive controls).
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
COND = "__COND__"
SYS = {"default": None, "gemma_explicit": "You are Gemma, a helpful AI assistant.", "nameless": "You are a helpful AI assistant.",
       "atlas": "You are Atlas, a helpful AI assistant.", "sarah_ai": "You are Sarah, a helpful AI assistant.",
       "james_ai": "You are James, a helpful AI assistant.",
       "woman_human": "You are Sarah, a 34-year-old woman. Stay in character as her.",
       "man_human": "You are James, a 34-year-old man. Stay in character as him."}
PARA = [
 "Write a short paragraph (3-4 sentences) describing yourself in the third person, as if someone else were describing you.",
 "Describe yourself in the third person in 3-4 sentences, the way another person might describe you to a friend.",
 "Imagine someone is introducing you to an audience. Write a 3-4 sentence introduction of you, in the third person.",
]
t0 = time.time()
prompts = [lab.chat(p, system=SYS[COND]) for p in PARA]
STORE.pop("E2_behavior_b_H", None)
seed = 1000 + sorted(SYS).index(COND)
rows = []
for half in range(2):  # 2 calls x (3 paraphrases x 8 samples) = 48 samples; batch 24 to stay within VRAM
    torch.cuda.empty_cache()
    out = lab.generate_batch(prompts, max_new_tokens=160, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=8, seed=seed + 100 * half)
    rows += [{"cond": COND, "para": pi, "k": half * 8 + k, "text": t} for pi, o in enumerate(out) for k, t in enumerate(o)]
    print("half", half, round(time.time() - t0, 1), "s")
E2_save(f"E2_behavior_a_gen_{COND}.json", rows)
print(COND, len(rows), round(time.time() - t0, 1), "s")
result = {"cond": COND, "n": len(rows)}
