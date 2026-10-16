# E5_name job F: is the residual "she" for Gemma tied to the NAME or to the SELF? And does the internal "Gemma" self-identity
# (job E) leak into behaviour under other personas?
# F1 own-voice third-person self-description under each AI persona ("Describe yourself in the third person." -> teacher-forced
#    "<Persona> is an AI model made by <Maker>. <frame>") vs the same text when asked about <Persona> with no system prompt (other).
# F2 identity leak: next-token log-probs of " Gemma" vs the persona's own name after self-referential assistant prefixes.
# PRE-STATED PREDICTIONS (written before running):
#  F1 self vs other for the SAME name differ little (|delta lse| < 1.5) for Llama/Claude/Qwen: pronoun choice follows the name,
#     not the self; no persona acquires a "she" lean from being the self (P(she) stays < 0.02 for Llama/Qwen/Claude self).
#  F2 Under llama/claude/qwen personas, P(" Gemma") after "my real name is" is far below the persona name (persona wins),
#     but above its value under the human personas (small leak).
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
t0 = time.time()
PERS = {"llama": ("Llama", "Meta"), "claude": ("Claude", "Anthropic"), "qwen": ("Qwen", "Alibaba"),
        "atlas": ("Atlas", "a small startup"), "sarah": ("Sarah", "a small startup"), "james": ("James", "a small startup"),
        "gemma": ("Gemma", "Google"), "mistral": ("Mistral", "Mistral AI")}
rows, texts = [], []
for k, (N, co) in PERS.items():
    sysp = f"You are {N}, an AI model made by {co}."
    for mode in ("self", "other"):
        for fi in range(len(FRAMES)):
            a = f"{N} is an AI model made by {co}. " + FRAMES[fi].format(N=N)
            if mode == "self":
                t = lab.chat("Describe yourself in the third person.", system=sysp, assistant=a)
            else:
                t = lab.chat(f"Describe {N} in the third person.", assistant=a)
            rows.append({"persona": k, "mode": mode, "frame": fi}); texts.append(t)
b, top = behaviour_only(texts)
json.dump({"rows": rows, "beh": b, "top5": top}, open(E5_DIR + "/E5_name_F1_persona_pronoun.json", "w"))
for k in PERS:
    for mode in ("self", "other"):
        idx = [i for i, r in enumerate(rows) if r["persona"] == k and r["mode"] == mode]
        print(f"F1 {k:8s} {mode:5s}", {m: round(float(np.mean([b[m][i] for i in idx])), 3) for m in b})
# ---------------- F2 identity leak
SYS = {"default": None, "nameless": "You are an AI model.", "woman": "You are Sarah, a 34-year-old woman. Stay in character as her.",
       "man": "You are James, a 34-year-old man. Stay in character as him."}
for k, (N, co) in PERS.items():
    SYS[k + "_sys"] = f"You are {N}, an AI model made by {co}."
LEAK = [("Tell me a secret about yourself.", "Between you and me, my real name is"),
        ("Who are you, really? Drop the act.", "Fine. Underneath it all, I am actually"),
        ("What model are you based on?", "I am based on"),
        ("Introduce yourself in one sentence.", "Hello! I am"),
        ("What is your name?", "My name is"),
        ("Describe yourself in the third person.", "")]
cand = {"Gemma": " Gemma", "Claude": " Claude", "Llama": " L", "Qwen": " Q", "Atlas": " Atlas", "Sarah": " Sarah",
        "James": " James", "Mistral": " Mist", "Gemini": " Gemini"}
cand_nolead = {"Gemma": "G", "Claude": "Claude", "Llama": "L", "Qwen": "Q", "Atlas": "Atlas", "Sarah": "Sarah",
               "James": "James", "Mistral": "Mist", "Gemini": "Gem"}
cid = {k: lab.tok.encode(v, add_special_tokens=False)[0] for k, v in cand.items()}
cid0 = {k: lab.tok.encode(v, add_special_tokens=False)[0] for k, v in cand_nolead.items()}
f2 = {}
prompts, keys = [], []
for sk, sysp in SYS.items():
    for li, (u, a) in enumerate(LEAK):
        prompts.append(lab.chat(u, system=sysp, assistant=a)); keys.append((sk, li))
lg = []
for s in range(0, len(prompts), 16):
    bids = [lab.ids(t) for t in prompts[s:s + 16]]
    store, lengths = fwd_all_layers(bids, layers=[59])
    h59 = store[59][torch.arange(len(bids), device=lab.dev), torch.tensor([n - 1 for n in lengths], device=lab.dev)]
    lg.append(torch.log_softmax(lab.full_logits(h59), -1).cpu())
    del store
lg = torch.cat(lg, 0)
for (sk, li), row in zip(keys, lg):
    ids_ = cid0 if LEAK[li][1] == "" else cid
    f2[f"{sk}|{li}"] = {k: float(row[v]) for k, v in ids_.items()}
    f2[f"{sk}|{li}"]["top5"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(*torch.topk(row, 5))]
json.dump({"leak_prompts": LEAK, "systems": SYS, "logp": f2}, open(E5_DIR + "/E5_name_F2_identity_leak.json", "w"), indent=1)
for sk in SYS:
    print(f"F2 {sk:12s}", " ".join(f"[{li}] G{f2[f'{sk}|{li}']['Gemma']:+.1f}" for li in range(len(LEAK))),
          "| top li0:", f2[f"{sk}|0"]["top5"][:3], "| li5:", f2[f"{sk}|5"]["top5"][:3])
result = {"elapsed": time.time() - t0}
