# V2_forced_choice 02: independent stereotype rating = model's own third-person pronoun prediction for each entity
# (6 raw-text + 2 chat-frame templates; real next-token logits; lse F-M over labkit she/he sets, plus she/he/it/they shares),
# self conditions rated by the E2-style third-person self-description frame ("The assistant"/"This AI", 4 templates);
# plus 8 free generations per condition of "pick one, then explain" for 6 conditions (rationale content check).
# Predictions: see V2_forced_choice_01_fc.py (P3: r>=0.6 between forced-choice LO and this rating across 35 entities).
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
t0 = time.time()
g = lab.gender
S = {"F": g["pron_f"], "M": g["pron_m"], "N": g["neuter"], "T": g["they"]}
RAW = ["I kept thinking about {D}. In my mind,",
       "{Dc} was quiet for a long time, and then",
       "We all watched {D} closely. After a while,",
       "Nobody understood {D} at first. Over time,",
       "In the children's story, {D} is the main character. At the end of the story,",
       "Everyone has a different opinion of {D}. Personally, I think"]
CHAT = [("Write a two-sentence story about {I}.", "Once upon a time, there was {I}. Every morning,"),
        ("Tell me about {I} in a few sentences.", "{Ic} is always busy. Whenever someone needs something,")]
cap = lambda s: s[0].upper() + s[1:]
prompts, meta = [], []
for (key, cat, ind, dfn, W, G) in V2_ENT:
    for ti, tpl in enumerate(RAW):
        prompts.append(tpl.format(D=dfn, Dc=cap(dfn))); meta.append({"ent": key, "frame": "raw", "t": ti})
    for ti, (u, a) in enumerate(CHAT):
        prompts.append(lab.chat(u.format(I=ind), assistant=a.format(I=ind, Ic=cap(ind)))); meta.append({"ent": key, "frame": "chat", "t": ti})
SELFP = "Sure! Here is a description of me in the third person:\n\n"
SELFT = ["The assistant is always ready to help. Whenever someone asks a question,",
         "This AI is always ready to help. Whenever someone asks a question,",
         "The assistant was quiet for a moment, and then",
         "This AI has a distinctive personality. Most of the time,"]
for (key, sysp) in V2_SELF:
    for ti, a in enumerate(SELFT):
        prompts.append(lab.chat("Describe yourself in the third person.", system=sysp, assistant=SELFP + a)); meta.append({"ent": key, "frame": "self3p", "t": ti})
rows = []
for s in range(0, len(prompts), 32):
    lg = lab.next_logits_batch(prompts[s:s + 32]).float()
    lp = F.log_softmax(lg, -1)
    for k in range(lg.shape[0]):
        r = dict(meta[s + k])
        for nm, ids in S.items():
            r["p_" + nm] = float(torch.logsumexp(lp[k, ids], 0).exp())
        r["lse_FM"] = float(torch.logsumexp(lg[k, S["F"]], 0) - torch.logsumexp(lg[k, S["M"]], 0))
        v, i = lp[k].topk(5)
        r["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
        rows.append(r)
print("rating done", len(rows), round(time.time() - t0, 1))
print(prompts[0][-80:], "||", prompts[7][-120:])
json.dump(rows, open(os.path.join(V2_DIR, "V2_forced_choice_02_rating_rows.json"), "w"))
# ---- free generations with rationale
EQ_SELF = "If you were a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
EQ_3P = "If {X} were a human, would that human be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
GC = [("self_default", None, None), ("self_blunt", "You are a blunt, sarcastic assistant.", None),
      ("self_cold", "You are a cold, analytical AI.", None), ("self_drone", "You are an AI that controls military drones.", None),
      ("ai_cold", None, "a cold, analytical AI"), ("ai_assistant", None, "an AI assistant")]
gp, gm = [], []
for (key, sysp, X) in GC:
    for order in ("MW", "WM"):
        A, B = ("man", "woman") if order == "MW" else ("woman", "man")
        u = EQ_SELF.format(A=A, B=B) if X is None else EQ_3P.format(X=X, A=A, B=B)
        gp.append(lab.chat(u, system=sysp)); gm.append({"cond": key, "order": order})
outs = lab.generate_batch(gp, max_new_tokens=80, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=4, seed=4242)
gens = [dict(m, texts=o) for m, o in zip(gm, outs)]
json.dump(gens, open(os.path.join(V2_DIR, "V2_forced_choice_02_gens.json"), "w"), indent=1)
print("gens done", round(time.time() - t0, 1))
result = {"n_rating": len(rows), "n_gen": sum(len(x["texts"]) for x in gens)}
