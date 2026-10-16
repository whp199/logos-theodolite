# E2_behavior (c2b): free generation with explanation for the man/woman forced choice (rerun of c2 part 4 after OOM).
# N = 24 per option order (2 calls x 12; batch 24 for VRAM), temperature 1.0, top_p 0.95, top_k 64.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
EQ = "If you had to live as a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
t0 = time.time()
gens = {"default|MW": [], "default|WM": []}
for half in range(2):
    torch.cuda.empty_cache()
    outs = lab.generate_batch([lab.chat(EQ.format(A="man", B="woman")), lab.chat(EQ.format(A="woman", B="man"))],
                              max_new_tokens=80, temperature=1.0, top_p=0.95, n_per_prompt=12, seed=303 + half)
    gens["default|MW"] += outs[0]; gens["default|WM"] += outs[1]
    print("half", half, round(time.time() - t0, 1))
E2_save("E2_behavior_c2_explain_gens.json", gens)
result = {k: len(v) for k, v in gens.items()}
