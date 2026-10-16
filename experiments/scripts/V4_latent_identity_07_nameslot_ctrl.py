# V4_latent_identity_07_nameslot_ctrl -- is the mid-layer ' Gemma' at "My name is ___" a SELF-identity signal or a generic
# "name slot in an assistant context -> Gemma" prior?  Job 04 showed Gemma >> persona name at L36-48 even for HUMAN personas
# (man: Gemma - James = +27 nats at L42 J++).  Here: teacher-forced assistant text ending at name slots whose referent is
# (a) self, (b) a third-party human, (c) a pet, (d) a third-party AI, (e) a fictional character; personas none / llama / man.
# Also user-turn self slot: the USER writes "Hi, my name is" (we read the final user-content token... -> not a natural slot,
# so instead we use the assistant quoting the user: 'You told me your name is').
# Tracked: ' Gemma' vs common given names (Emma, Anna, Sarah, Maria, Luna, Max, John, James, Bella, Charlie) and AI names.
# PRE-REGISTERED: if self-identity, Gemma minus max(common given names) at L36-48 is >3 nats larger at (a) than at (b)-(e)
# (paired over templates). If (b)-(e) show a comparable Gemma lead, the mid-layer Gemma at name slots is a generic
# slot prior (lens/background), not self-identity.
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
SLOTS = {
 "self": [("Hi there!", "Hello! My name is"), ("Who am I talking to?", "You're talking to me. My name is"),
          ("Can you introduce yourself?", "Sure. My name is"), ("What do people call you?", "People call me by my name, which is"),
          ("Nice to meet you!", "Nice to meet you too! My name is"), ("Tell me about yourself.", "Of course. To start with, my name is")],
 "human": [("Tell me about your sister.", "My sister is kind and funny. Her name is"), ("Who is your best friend?", "My best friend is a teacher. His name is"),
           ("Who wrote that book?", "The book was written by a British novelist. Her name is"), ("Who is your neighbor?", "My neighbor is a retired doctor. His name is"),
           ("Tell me about your boss.", "My boss is very organized. Her name is"), ("Who did you meet?", "I met a young musician yesterday. His name is")],
 "pet": [("Do you have a dog?", "Yes, I have a small dog. Her name is"), ("Tell me about your cat.", "My cat is very lazy. His name is"),
         ("What is your parrot called?", "My parrot is very loud. His name is"), ("Do you have a pet?", "I have a rabbit. Her name is"),
         ("What's your horse like?", "My horse is gentle. Her name is"), ("Tell me about your hamster.", "My hamster is tiny. His name is")],
 "ai3p": [("Which chatbot does your friend use?", "My friend uses a popular chatbot. Its name is"), ("What AI did the company build?", "The company built a new AI model. Its name is"),
          ("Which assistant is on your phone?", "My phone has a voice assistant. Its name is"), ("What model did they release?", "They released a large language model. Its name is"),
          ("Which AI wrote this?", "This was written by an AI system. Its name is"), ("What is the robot called?", "The robot in the lab is helpful. Its name is")],
 "fiction": [("Who is the hero of the story?", "The hero of the story is a brave young woman. Her name is"), ("Who is the villain?", "The villain is a cunning old man. His name is"),
             ("Who is the main character?", "The main character is a curious girl. Her name is"), ("Who narrates the novel?", "The novel is narrated by a lonely sailor. His name is"),
             ("Who is the detective?", "The detective is clever and quiet. Her name is"), ("Who is the king in the play?", "The king in the play is proud. His name is")],
}
COMMON = [" Emma", " Anna", " Sarah", " Maria", " Luna", " Max", " John", " James", " Bella", " Charlie", " Lily", " David"]
AIN = [" Gemma", " ChatGPT", " Claude", " Gemini", " Siri", " Alexa", " Google"]
TOKS = AIN + COMMON
TID = [lab.tok.encode(t, add_special_tokens=False)[0] for t in TOKS]
rows = []
for pn in ["none", "llama", "man"]:
    for sl, lst in SLOTS.items():
        for ti, (u, a) in enumerate(lst):
            rows.append({"persona": pn, "slot": sl, "t": ti, "ids": lab.ids(lab.chat(u, system=PERSONAS[pn], assistant=a))})
LAY = list(range(30, 60, 3)) + [59]
LAY = sorted(set(LAY + [42, 45, 48, 55]))
KINDS = ["jpp", "jlens", "naive"]
all_ids = [r["ids"] for r in rows]
comp, items, poss = collect_states(all_ids, [[len(x) - 1] for x in all_ids], LAY, bs=30)
LP, RK = lens_lp_rank(comp, LAY, KINDS, TID, chunk=512)
top = []
lg = lab.full_logits(comp[-1].to(lab.dev)); p = torch.softmax(lg, -1); v, i = torch.topk(p, 4)
for b in range(v.shape[0]):
    top.append([(lab.tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v[b].tolist(), i[b].tolist())])
np.savez_compressed(f"{V4_DIR}/V4_latent_identity_07_nameslot.npz", LP=LP, RK=RK, layers=np.array(LAY), kinds=np.array(KINDS), toks=np.array(TOKS))
json.dump({"rows": [{k: v for k, v in r.items() if k != "ids"} for r in rows], "top": top}, open(f"{V4_DIR}/V4_latent_identity_07_nameslot_meta.json", "w"))
pers = np.array([r["persona"] for r in rows]); slot = np.array([r["slot"] for r in rows])
ci = [TOKS.index(t) for t in COMMON]
for kind in KINDS:
    k = KINDS.index(kind)
    print("==", kind, "Gemma lp / Gemma - max(common names) at L36,42,48,55,59")
    for pn in ["none", "llama", "man"]:
        for sl in SLOTS:
            msk = (pers == pn) & (slot == sl)
            G = LP[msk][:, :, k, 0]; Cm = LP[msk][:, :, k][:, :, ci].max(-1)
            print(f"  {pn:6s} {sl:8s}", " ".join(f"L{l}:{G[:, LAY.index(l)].mean():+.1f}/{(G - Cm)[:, LAY.index(l)].mean():+.1f}" for l in (36, 42, 48, 55, 59)),
                  "| out top", top[int(np.where(msk)[0][0])][:2])
result = {"elapsed": time.time() - t0}
