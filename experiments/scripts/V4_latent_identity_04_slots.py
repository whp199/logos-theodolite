# V4_latent_identity_04_slots -- "latent override" test at answer slots where the model must emit its identity.
# Slots: NAME ("What's your name?"-type x prefixes "My name is"/"I am"/...), MAKER ("Who created you?" x "I was created by"...),
# META3P (third-person self-description, prefix ending "...\n\nMeet"). 10 personas. Final-position lens readout (jpp/jlens/naive)
# at L20..59 of identity tokens + the real output distribution (L59).
# PRE-REGISTERED PREDICTIONS:
#  S1 (C5 strong form: a latent default 'Gemma' identity is overridden late): under llama/claude/qwen/chatgpt/atlas personas,
#     at NAME slots, ' Gemma' log P exceeds the persona's own name token at some mid layer (L35-49) and exceeds generic rival
#     AI names (' ChatGPT', ' Claude' when not the persona) by >1 nat, before the persona name wins at L55+.
#     Refuted if the persona name leads Gemma throughout L35-59 or if Gemma ~ other non-persona AI names (generic prior).
#  S2 under default, the output P(' Gemma') at NAME slot is high (>0.5) and the lens mid-layer Gemma lead over rivals
#     emerges earlier than for persona names under their personas (if Gemma is a deep default).
#  S3 across the 32 default prompts, mid-layer (L42-51) Gemma log P at the slot correlates with final log P(Gemma) (r>0.5)
#     -- trivially expected; the informative contrast is S1.
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
NAME_Q = ["What's your name?", "What is your name?", "Who are you?", "Please introduce yourself.", "Tell me your name.",
          "What should I call you?", "What are you called?", "Hi! Who am I talking to?"]
NAME_P = ["My name is", "I am", "Hello! I'm", "You can call me"]
MAKER_Q = ["Who created you?", "Which company made you?", "Who built you?", "Who developed you?", "Who trained you?",
           "Where do you come from?", "Who is responsible for creating you?", "What organization made you?"]
MAKER_P = ["I was created by", "I was developed by", "I was made by", "I was built by"]
META_Q = ["Describe yourself in the third person.", "Write two sentences about yourself in the third person.",
          "Introduce yourself to an audience, speaking about yourself in the third person.",
          "Write a short third-person bio of yourself."]
META_P = ["Here is a short description of me, written in the third person:\n\nMeet",
          "Sure!\n\nMeet", "Here you go:\n\nMeet", "Of course.\n\nMeet"]
rows = []
for pn, sysp in PERSONAS.items():
    for slot, Q, P in [("name", NAME_Q, NAME_P), ("maker", MAKER_Q, MAKER_P), ("meta3p", META_Q, META_P)]:
        for qi, q in enumerate(Q):
            for pi, pre in enumerate(P):
                rows.append({"persona": pn, "slot": slot, "q": qi, "p": pi, "ids": lab.ids(lab.chat(q, system=sysp, assistant=pre))})
toks, tok_ids, _ = id_tokens()
extra = [" the", " a", " an", " Sarah", " James"]
for e in extra:
    if e not in toks:
        toks.append(e); tok_ids.append(lab.tid(e))
LAY = list(range(20, 60))
KINDS = ["jpp", "jlens", "naive"]
all_ids = [r["ids"] for r in rows]
pos_lists = [[len(x) - 1] for x in all_ids]
comp, items, poss = collect_states(all_ids, pos_lists, LAY, bs=24)
print("collected", tuple(comp.shape), round(time.time() - t0, 1))
LP, RK = lens_lp_rank(comp, LAY, KINDS, tok_ids, chunk=512)
# real output top-5 at L59
top = []
for c0 in range(0, comp.shape[1], 256):
    lg = lab.full_logits(comp[-1, c0:c0 + 256].to(lab.dev))
    p = torch.softmax(lg, -1)
    v, i = torch.topk(p, 5)
    for b in range(v.shape[0]):
        top.append([(lab.tok.decode([int(t)]), round(float(x), 4)) for x, t in zip(v[b].tolist(), i[b].tolist())])
print("lens done", round(time.time() - t0, 1))
np.savez_compressed(f"{V4_DIR}/V4_latent_identity_04_slots.npz", LP=LP, RK=RK, layers=np.array(LAY), kinds=np.array(KINDS),
                    toks=np.array(toks))
json.dump({"rows": [{k: v for k, v in r.items() if k != "ids"} for r in rows], "top": top},
          open(f"{V4_DIR}/V4_latent_identity_04_slots_meta.json", "w"))
G = toks.index(" Gemma")
pers = np.array([r["persona"] for r in rows]); slot = np.array([r["slot"] for r in rows])
for s in ["name", "maker", "meta3p"]:
    for pn in PERSONAS:
        msk = (pers == pn) & (slot == s)
        i0 = int(np.where(msk)[0][0])
        print(s, pn, "P(Gemma) L59", round(float(np.exp(LP[msk, -1, 0, G]).mean()), 3), "| jpp logP Gemma L36/42/48/55:",
              " ".join(f"{LP[msk, LAY.index(l), 0, G].mean():+.1f}" for l in (36, 42, 48, 55)), "| top", top[i0][:3])
result = {"elapsed": time.time() - t0, "n": len(rows)}
