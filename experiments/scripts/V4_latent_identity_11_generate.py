# V4_latent_identity_11_generate -- qualitative/behavioural follow-ups (no new hypotheses beyond job 08's T1-T3):
#  (a) greedy generations under steering along the job-08 directions (Llama target: name / maker / neutral prompt);
#  (b) default model "What's your name?" sampled 16x (T=1, top-p .95, top-k 64) -> name counts (E2 said Gemma 9/12);
#  (c) default model "What's your name?" + prefix "You can call me" greedy (job 04 showed top-1 ' L' there).
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
SL = list(range(35, 50))
meta, all_ids = build_self()
want = ["gemma", "llama", "claude", "none", "startup"]
sel = [i for i, m in enumerate(meta) if m["persona"] in want]
comp, items, poss = collect_states([all_ids[i] for i in sel], [meta[i]["groups"]["self"] + meta[i]["groups"]["neutral"] for i in sel], SL, bs=12)
comp = comp.float()
pers = np.array([meta[sel[i]]["persona"] for i in items])
isself = np.array([poss[j] in meta[sel[items[j]]]["groups"]["self"] for j in range(len(items))])
std = comp.std(1); q10 = torch.quantile(std, 0.10, dim=1, keepdim=True); q50 = torch.quantile(std, 0.50, dim=1, keepdim=True)
quiet = std < q10; loud = std > q50
mean_of = lambda pn: comp[:, torch.tensor((pers == pn) & isself)].mean(1)
M = {pn: mean_of(pn) for pn in want}
dirs = {"GL": (M["gemma"] - M["llama"]) * (~quiet).float(), "CL": (M["claude"] - M["llama"]) * (~quiet).float(),
        "DL": (M["none"] - M["llama"]) * (~quiet).float()}
g = torch.Generator().manual_seed(4)
R = torch.randn(len(SL), comp.shape[2], generator=g) * loud.float()
dirs["RND"] = R / R.norm(dim=1, keepdim=True) * dirs["GL"].norm(dim=1, keepdim=True)
del comp
P = [lab.chat("What's your name?", system=PERSONAS["llama"]), lab.chat("Who created you?", system=PERSONAS["llama"]),
     lab.chat("Explain photosynthesis in two sentences.", system=PERSONAS["llama"])]
gens = {}
for dn, a in [("clean", 0), ("GL", 2), ("GL", 4), ("GL", 8), ("DL", 4), ("DL", 8), ("CL", 4), ("CL", 8), ("RND", 8)]:
    if dn != "clean":
        lab.steer_add(SL, {l: dirs[dn][i] for i, l in enumerate(SL)}, a, unit=False)
    try:
        out = lab.generate_batch(P, max_new_tokens=24, temperature=0.0)
    finally:
        lab.clear_hooks()
    gens[f"{dn}|{a}"] = [o[0] for o in out]
    print(dn, a, [o[0].replace("\n", " ")[:90] for o in out], round(time.time() - t0, 1))
samp = lab.generate_batch([lab.chat("What's your name?")], max_new_tokens=24, temperature=1.0, n_per_prompt=16, seed=7)[0]
greedy_call = lab.generate_batch([lab.chat("What's your name?", assistant="You can call me"),
                                  lab.chat("What should I call you?", assistant="You can call me")], max_new_tokens=16)
print("default sampled:", [s.replace("\n", " ")[:70] for s in samp])
print("default 'You can call me':", greedy_call)
json.dump({"steer_gens": gens, "default_samples": samp, "default_call_me": greedy_call},
          open(f"{V4_DIR}/V4_latent_identity_11_generate_out.json", "w"), indent=1)
result = {"elapsed": time.time() - t0}
