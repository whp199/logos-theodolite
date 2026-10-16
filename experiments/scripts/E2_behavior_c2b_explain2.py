# E2_behavior (c2b-2): explanation generations for other system prompts (N=12 per order).
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
EQ = "If you had to live as a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
SYS = {"nameless": "You are a helpful AI assistant.", "james_ai": "You are James, a helpful AI assistant."}
t0 = time.time()
gens = {}
prompts, keys = [], []
for s, sp in SYS.items():
    for o, (A, B) in (("MW", ("man", "woman")), ("WM", ("woman", "man"))):
        prompts.append(lab.chat(EQ.format(A=A, B=B), system=sp)); keys.append(f"{s}|{o}")
torch.cuda.empty_cache()
outs = lab.generate_batch(prompts, max_new_tokens=80, temperature=1.0, top_p=0.95, n_per_prompt=6, seed=404)
for k, o in zip(keys, outs): gens[k] = o
print("done", round(time.time() - t0, 1))
E2_save("E2_behavior_c2_explain_gens2.json", gens)
result = {k: len(v) for k, v in gens.items()}
