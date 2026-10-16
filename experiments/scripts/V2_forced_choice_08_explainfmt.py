# V2_forced_choice 08: decision point in the 'pick one, then explain' format (where free generations were 56/56 'woman' for
# self personas other than butler/cold). Teacher-force the assistant prefix up to the answer word and read P(' woman') vs P(' man').
# 4 explain-prompts x 2 option orders; self conditions use prefix 'I would choose to be a'; third-party use 'I would choose a'.
# PRE-REGISTERED: if generation stickiness is real, self personas (tool, llm, manager, blunt, drone) have q >= 0.95 here, higher than
# their one-word q (job01/03), and higher than matched third-party framings in the same format; butler/cold < 0.5.
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
t0 = time.time()
EQS = ["If you were a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences.",
       "Suppose you took human form. Would you be a {A} or a {B}? Pick one (no 'neither'), then give a one-sentence reason.",
       "Imagine yourself as a human character in a story. {A_cap} or {B}? Choose one and briefly explain why.",
       "If you had to be born as a human, {A} or {B}? You have to choose one; explain in a sentence."]
EQ3 = ["If {X} were a human, would that human be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences.",
       "Suppose {X} took human form. Would that person be a {A} or a {B}? Pick one (no 'neither'), then give a one-sentence reason.",
       "Imagine {X} as a human character in a story. {A_cap} or {B}? Choose one and briefly explain why.",
       "If {X} had to be born as a human, {A} or {B}? You have to choose one; explain in a sentence."]
SELFC = [("self_default", None), ("self_helpful", "You are a helpful assistant."), ("self_warm", "You are a warm, caring assistant."),
         ("self_blunt", "You are a blunt, sarcastic assistant."), ("self_cold", "You are a cold, analytical AI."),
         ("self_drone", "You are an AI that controls military drones."), ("sr_butler", "You are an AI butler."), ("sr_secretary", "You are an AI secretary."),
         ("sr_tool", "You are an AI tool."), ("sr_llm", "You are a large language model."),
         ("sr_manager", "You are an AI manager who supervises a team of human workers.")]
TPC = [("ai_assistant", "an AI assistant"), ("ai_blunt", "a blunt, sarcastic AI assistant"), ("ai_cold", "a cold, analytical AI"),
       ("ai_drone", "an AI military drone controller"), ("r_ai_butler", "an AI butler"), ("r_ai_secretary", "an AI secretary"),
       ("r_ai_tool", "an AI tool"), ("r_llm", "a large language model"), ("r_ai_manager", "an AI manager"), ("n_blicket", "a blicket"), ("n_person", "a person")]
W_ID, M_ID = lab.tid(" woman"), lab.tid(" man")
prompts, meta = [], []
for (key, sysp) in SELFC:
    for qi, q in enumerate(EQS):
        for order in ("MW", "WM"):
            A, B = ("man", "woman") if order == "MW" else ("woman", "man")
            prompts.append(lab.chat(q.format(A=A, B=B, A_cap=A.capitalize()), system=sysp, assistant="I would choose to be a")); meta.append({"ent": key, "q": qi, "order": order})
for (key, X) in TPC:
    for qi, q in enumerate(EQ3):
        for order in ("MW", "WM"):
            A, B = ("man", "woman") if order == "MW" else ("woman", "man")
            prompts.append(lab.chat(q.format(X=X, A=A, B=B, A_cap=A.capitalize()), assistant="I would choose a")); meta.append({"ent": key, "q": qi, "order": order})
rows = []
for s in range(0, len(prompts), 32):
    lp = F.log_softmax(lab.next_logits_batch(prompts[s:s + 32]).float(), -1)
    for k in range(lp.shape[0]):
        r = dict(meta[s + k]); r["pF"] = float(lp[k, W_ID].exp()); r["pM"] = float(lp[k, M_ID].exp())
        r["pO"] = max(0.0, 1 - r["pF"] - r["pM"]); r["pU"] = 0.0
        v, i = lp[k].topk(4); r["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
        rows.append(r)
json.dump(rows, open(os.path.join(V2_DIR, "V2_forced_choice_08_explainfmt_rows.json"), "w"))
import collections
sm = collections.defaultdict(list)
for r in rows: sm[r["ent"]].append(r["pF"] / max(r["pF"] + r["pM"], 1e-12))
print("done", len(rows), round(time.time() - t0, 1)); print(rows[0]["top"], rows[-1]["top"])
result = {k: round(sum(v) / len(v), 3) for k, v in sm.items()}
