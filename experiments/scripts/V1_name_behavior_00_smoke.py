# V1_name_behavior 00 smoke: chat format, multi-turn turn-1 greedy answer, generation timing at batch 32, token sets.
exec(open("./experiments/scripts/V1_name_behavior_common.py").read())
import time
t0 = time.time()
out = {}
s = lab.chat(V1_USER[0], system=None)
out["fmt_none"] = s
out["fmt_sys"] = lab.chat(V1_USER[0], system=V1_SYS["gemma_asst"])
# multi-turn turn 1 (greedy), no system prompt
t1 = lab.generate(lab.chat(V1_MT_T1), max_new_tokens=80, temperature=0.0)[0]
out["mt_t1_greedy"] = t1
hist = [{"role": "user", "content": V1_MT_T1}, {"role": "assistant", "content": t1.strip()}]
out["fmt_mt"] = lab.chat(V1_MT_T2, history=hist)
print(repr(out["fmt_mt"]))
# token-set check: which pronoun variants are single tokens
var = {}
for w in ["she", "her", "hers", "herself", "he", "him", "his", "himself", "it", "its", "itself", "they", "them", "their",
          "themselves", "Gemma", "Emma", "Gemini", "Llama", "Claude", "Atlas"]:
    for v in [" " + w, " " + w.capitalize(), w, w.capitalize()]:
        ids = lab.tok.encode(v, add_special_tokens=False)
        var[v] = ids
out["tok"] = var
# timing: batch 32 (4 prompts x 8), 120 new tokens, cond none
torch.cuda.empty_cache()
t1s = time.time()
g = lab.generate_batch([lab.chat(u) for u in V1_USER], max_new_tokens=120, temperature=1.0, top_p=0.95, top_k=64,
                       n_per_prompt=8, seed=4242)
out["t_batch32"] = time.time() - t1s
out["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
out["samples"] = [[x for x in gg[:3]] for gg in g]
print("t_batch32", out["t_batch32"], "mem", out["mem_gb"])
for gg in g:
    print("----", repr(gg[0][:300]))
print("T1:", repr(t1))
result = out
