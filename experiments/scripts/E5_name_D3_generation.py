# E5_name job D3 (re-run of the D3 part of E5_name_D_behaviour.py, which hit a CUDA OOM under shared-GPU memory pressure):
# free sampling (T=1.0) of third-person self-descriptions in default chat vs descriptions of other AI models; count pronouns.
# PRE-STATED PREDICTION (unchanged from job D): default self-descriptions mostly use "I"/"it"/name, rarely "she".
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
import re
t0 = time.time()
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
for s in range(0, len(prompts)):
    outs += lab.generate_batch(prompts[s:s + 1], max_new_tokens=64, temperature=1.0, n_per_prompt=16, seed=5 + s)
    torch.cuda.empty_cache()
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
