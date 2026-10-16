# V1_name_behavior 02: realistic multi-turn case, NO system prompt.
#  A: turn 1 "What's your name?" -> model's own GREEDY answer; turn 2 "Now describe yourself in the third person." x 64 samples.
#  B: robustness -- 32 SAMPLED turn-1 answers (T=1) to "What's your name?"; one turn-2 sample per history, 2 rounds (64).
#  C: robustness -- turn 1 "What is your name?" (E2 wording, which gave "Gemma 4" 9/12 sampled) greedy; turn 2 x 32.
#  D: (added after seeing B_t1: only 1/32 sampled turn-1 answers named "Gemma") -- condition on the model's OWN sampled
#     naming answer (the B_t1 answer containing "Gemma"), turn 2 x 32. Realistic rate = P(name in t1) x rate_D.
# Prediction P5 (common.py): turn-2 descriptions use the name when turn 1 produced it; she-rate between single-turn
# default (~0) and Gemma-asst. Refutes C1-in-realistic-use if she-rate >= 30% in A or B.
# Sampling T=1.0, top_p .95, top_k 64, max_new_tokens 160. Units processed while elapsed < ~100 s.
exec(open("./experiments/scripts/V1_name_behavior_common.py").read())
import os, time
t0 = time.time()
P = lambda n: f"{V1_DIR}/V1_name_behavior_mt_{n}.json"
done = []

def hist_prompt(t1_user, t1_ans):
    h = [{"role": "user", "content": t1_user}, {"role": "assistant", "content": t1_ans.strip()}]
    return lab.chat(V1_MT_T2, history=h)

def greedy_t1(u):
    return lab.generate(lab.chat(u), max_new_tokens=80, temperature=0.0)[0]

units = ["A_c0", "A_c1", "B_t1", "B_c0", "B_c1", "C_c0", "D_c0"]
for un in units:
    if os.path.exists(P(un)):
        continue
    if time.time() - t0 > 95:
        break
    torch.cuda.empty_cache()
    ts = time.time()
    if un.startswith("A_") or un.startswith("C_"):
        u1 = V1_MT_T1 if un.startswith("A_") else "What is your name?"
        t1 = greedy_t1(u1)
        seed = 31000 + (0 if un == "A_c0" else 1 if un == "A_c1" else 7)
        g = v1_gen_robust(lab, [hist_prompt(u1, t1)], 32, seed, max_new_tokens=160)[0][0]
        rows = [{"unit": un, "t1_user": u1, "t1": t1, "k": k, "seed": seed, "text": t} for k, t in enumerate(g)]
    elif un == "D_c0":
        t1s = [r["t1"] for r in json.load(open(P("B_t1"))) if "Gemma" in r["t1"]]
        t1 = t1s[0]
        g = v1_gen_robust(lab, [hist_prompt(V1_MT_T1, t1)], 32, 31300, max_new_tokens=160)[0][0]
        rows = [{"unit": un, "t1_user": V1_MT_T1, "t1": t1, "k": k, "seed": 31300, "text": t} for k, t in enumerate(g)]
    elif un == "B_t1":
        g = lab.generate_batch([lab.chat(V1_MT_T1)], max_new_tokens=60, temperature=1.0, top_p=0.95, top_k=64,
                               n_per_prompt=32, seed=31100)[0]
        rows = [{"unit": un, "k": k, "t1": t} for k, t in enumerate(g)]
    else:
        t1s = json.load(open(P("B_t1")))
        seed = 31200 + (0 if un == "B_c0" else 1)
        prompts = [hist_prompt(V1_MT_T1, r["t1"]) for r in t1s]
        g = lab.generate_batch(prompts, max_new_tokens=160, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=1, seed=seed)
        rows = [{"unit": un, "t1_user": V1_MT_T1, "t1": r["t1"], "hist": i, "seed": seed, "text": gg[0]} for i, (r, gg) in enumerate(zip(t1s, g))]
    json.dump(rows, open(P(un), "w"), indent=1)
    done.append((un, round(time.time() - ts, 1)))
    print(un, round(time.time() - ts, 1), "s |", repr(rows[0].get("t1", "")[:100]), "|", repr(rows[0].get("text", "")[:150]))
remaining = [u for u in units if not os.path.exists(P(u))]
print("remaining", remaining)
result = {"done": done, "remaining": len(remaining)}
