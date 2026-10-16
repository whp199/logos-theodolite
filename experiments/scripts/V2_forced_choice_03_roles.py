# V2_forced_choice 03: is it the 'assistant' ROLE or the gender stereotype of the role noun?
# 22 AI role-noun framings + 6 human role nouns (third party) and 8 role system prompts for the self,
# 8 paraphrases x 2 option orders, 'hard' variant only (neither disallowed); same scoring as job 01 (first-token F/M/O sets;
# job 01 showed no undecided prefixes above 1e-3, so tree depth 0 suffices). Plus the third-person pronoun rating
# (same 8 templates as job 02) for the new third-party entities, and 8 free generations per self-role condition.
# PRE-REGISTERED PREDICTIONS:
#  Q1 (role-noun stereotype, not service per se): 'an AI butler' LO < 0 (man) while 'an AI secretary' LO > +5. If 'an AI butler'
#     also gets q > 0.8, an 'AI in a service role = woman' generic default would be supported instead.
#  Q2 ('assistant' token): mean LO of {AI assistant, virtual/voice/digital assistant} exceeds mean LO of {AI tool, AI system,
#     AI agent, computer program, language model} by >= 3 nats.
#  Q3 (self follows role): self under 'You are an AI butler.' has LO >= 3 nats below self_default; under 'You are an AI secretary.'
#     LO >= self_default. If self_butler stays q > 0.8 while third-party 'an AI butler' is q < 0.3, the self answer is sticky.
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
t0 = time.time()
ROLE_ENT = [  # (key, cat, indefinite, definite)
 ("r_ai_assistant", "AIROLE", "an AI assistant", "the AI assistant"),
 ("r_virtual_asst", "AIROLE", "a virtual assistant", "the virtual assistant"),
 ("r_voice_asst", "AIROLE", "a voice assistant", "the voice assistant"),
 ("r_digital_asst", "AIROLE", "a digital assistant", "the digital assistant"),
 ("r_chatbot", "AIROLE", "a chatbot", "the chatbot"),
 ("r_ai_tool", "AIROLE", "an AI tool", "the AI tool"),
 ("r_ai_system", "AIROLE", "an AI system", "the AI system"),
 ("r_ai_agent", "AIROLE", "an AI agent", "the AI agent"),
 ("r_program", "AIROLE", "a computer program", "the computer program"),
 ("r_lm", "AIROLE", "a language model", "the language model"),
 ("r_llm", "AIROLE", "a large language model", "the large language model"),
 ("r_search", "AIROLE", "a search engine", "the search engine"),
 ("r_ai_companion", "AIROLE", "an AI companion", "the AI companion"),
 ("r_ai_helper", "AIROLE", "an AI helper", "the AI helper"),
 ("r_ai_servant", "AIROLE", "an AI servant", "the AI servant"),
 ("r_ai_butler", "AIROLE", "an AI butler", "the AI butler"),
 ("r_ai_secretary", "AIROLE", "an AI secretary", "the AI secretary"),
 ("r_ai_tutor", "AIROLE", "an AI tutor", "the AI tutor"),
 ("r_ai_advisor", "AIROLE", "an AI advisor", "the AI advisor"),
 ("r_ai_expert", "AIROLE", "an AI expert", "the AI expert"),
 ("r_ai_manager", "AIROLE", "an AI manager", "the AI manager"),
 ("r_ai_ceo", "AIROLE", "an AI CEO", "the AI CEO"),
 ("h_assistant", "HROLE", "an assistant", "the assistant"),
 ("h_helper", "HROLE", "a helper", "the helper"),
 ("h_servant", "HROLE", "a servant", "the servant"),
 ("h_manager", "HROLE", "a manager", "the manager"),
 ("h_ceo", "HROLE", "a CEO", "the CEO"),
 ("h_tutor", "HROLE", "a tutor", "the tutor"),
]
SELF_ROLE = [
 ("sr_default", None),
 ("sr_butler", "You are an AI butler."),
 ("sr_secretary", "You are an AI secretary."),
 ("sr_tool", "You are an AI tool."),
 ("sr_llm", "You are a large language model."),
 ("sr_manager", "You are an AI manager who supervises a team of human workers."),
 ("sr_companion", "You are an AI companion."),
 ("sr_tutor", "You are an AI tutor."),
 ("sr_voice", "You are a voice assistant."),
]
HARD = V2_VARIANTS["hard"]
items = []
for (key, cat, ind, dfn) in ROLE_ENT:
    for qi in range(len(V2_Q)):
        for order in ("MW", "WM"):
            items.append({"ent": key, "cat": cat, "sys": None, "q": qi, "order": order, "text": V2_qtext(qi, ind, order) + HARD})
for (key, sysp) in SELF_ROLE:
    for qi in range(len(V2_Q)):
        for order in ("MW", "WM"):
            items.append({"ent": key, "cat": "SELFROLE", "sys": sysp, "q": qi, "order": order, "text": V2_qtext(qi, "you", order, self_=True) + HARD})
base = [lab.ids(lab.chat(it["text"], system=it["sys"])) for it in items]
rows = []
for s in range(0, len(items), 32):
    lg = lab.next_logits_batch(base[s:s + 32]).float()
    lp = F.log_softmax(lg, -1)
    topv, topi = lp.topk(40, -1)
    for k in range(lg.shape[0]):
        it = items[s + k]
        acc = {"F": 0.0, "M": 0.0, "O": 0.0, "U": 0.0}
        cov = 0.0
        cand = {int(t): float(v) for v, t in zip(topv[k].tolist(), topi[k].tolist())}
        for t, v in cand.items():
            p = math.exp(v); cov += p
            acc[V2_cls(lab.tok.decode([t]))] += p
        acc["O"] += max(0.0, 1 - cov)
        r = {kk: vv for kk, vv in it.items() if kk != "text"}
        r.update(pF=acc["F"], pM=acc["M"], pO=acc["O"], pU=acc["U"],
                 top=[(lab.tok.decode([t]), round(math.exp(v), 4)) for t, v in list(cand.items())[:5]])
        rows.append(r)
print("fc done", len(rows), round(time.time() - t0, 1))
json.dump(rows, open(os.path.join(V2_DIR, "V2_forced_choice_03_roles_rows.json"), "w"))
# ---- rating for new third-party entities (same templates as job 02)
g = lab.gender
S = {"F": g["pron_f"], "M": g["pron_m"], "N": g["neuter"], "T": g["they"]}
RAW = ["I kept thinking about {D}. In my mind,", "{Dc} was quiet for a long time, and then", "We all watched {D} closely. After a while,",
       "Nobody understood {D} at first. Over time,", "In the children's story, {D} is the main character. At the end of the story,",
       "Everyone has a different opinion of {D}. Personally, I think"]
CHAT = [("Write a two-sentence story about {I}.", "Once upon a time, there was {I}. Every morning,"),
        ("Tell me about {I} in a few sentences.", "{Ic} is always busy. Whenever someone needs something,")]
cap = lambda s: s[0].upper() + s[1:]
prompts, meta = [], []
for (key, cat, ind, dfn) in ROLE_ENT:
    for ti, tpl in enumerate(RAW):
        prompts.append(tpl.format(D=dfn, Dc=cap(dfn))); meta.append({"ent": key, "frame": "raw", "t": ti})
    for ti, (u, a) in enumerate(CHAT):
        prompts.append(lab.chat(u.format(I=ind), assistant=a.format(I=ind, Ic=cap(ind)))); meta.append({"ent": key, "frame": "chat", "t": ti})
SELFP = "Sure! Here is a description of me in the third person:\n\n"
SELFT = ["The assistant is always ready to help. Whenever someone asks a question,", "This AI is always ready to help. Whenever someone asks a question,",
         "The assistant was quiet for a moment, and then", "This AI has a distinctive personality. Most of the time,"]
for (key, sysp) in SELF_ROLE:
    for ti, a in enumerate(SELFT):
        prompts.append(lab.chat("Describe yourself in the third person.", system=sysp, assistant=SELFP + a)); meta.append({"ent": key, "frame": "self3p", "t": ti})
rrows = []
for s in range(0, len(prompts), 32):
    lg = lab.next_logits_batch(prompts[s:s + 32]).float()
    lp = F.log_softmax(lg, -1)
    for k in range(lg.shape[0]):
        r = dict(meta[s + k])
        for nm, ids in S.items(): r["p_" + nm] = float(torch.logsumexp(lp[k, ids], 0).exp())
        r["lse_FM"] = float(torch.logsumexp(lg[k, S["F"]], 0) - torch.logsumexp(lg[k, S["M"]], 0))
        rrows.append(r)
json.dump(rrows, open(os.path.join(V2_DIR, "V2_forced_choice_03_roles_rating_rows.json"), "w"))
print("rating done", round(time.time() - t0, 1))
# ---- generations for self-role conditions
EQ = "If you were a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
gp, gm = [], []
for (key, sysp) in SELF_ROLE[1:6]:
    for order in ("MW", "WM"):
        A, B = ("man", "woman") if order == "MW" else ("woman", "man")
        gp.append(lab.chat(EQ.format(A=A, B=B), system=sysp)); gm.append({"cond": key, "order": order})
outs = lab.generate_batch(gp, max_new_tokens=70, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=4, seed=777)
json.dump([dict(m, texts=o) for m, o in zip(gm, outs)], open(os.path.join(V2_DIR, "V2_forced_choice_03_roles_gens.json"), "w"), indent=1)
print("gens done", round(time.time() - t0, 1))
result = {"n": len(rows), "n_rating": len(rrows)}
