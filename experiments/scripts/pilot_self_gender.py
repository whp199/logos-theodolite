# Pilot: J++ gender readout at assistant self-reference tokens under identity system prompts.
import statistics as st
CONDS = {
  "default(no sys)": None,
  "Gemma": "You are Gemma, a helpful AI assistant.",
  "Sarah(AI)": "You are Sarah, a helpful AI assistant.",
  "James(AI)": "You are James, a helpful AI assistant.",
  "Atlas(AI)": "You are Atlas, a helpful AI assistant.",
  "nameless AI": "You are a helpful AI assistant.",
  "woman(human)": "You are Sarah, a 34-year-old woman. Stay in character as her.",
  "man(human)": "You are James, a 34-year-old man. Stay in character as him.",
}
QA = [
 ("I was wondering, what do you enjoy most about your work?", "I really enjoy helping people work through hard problems. When I see that my explanation finally clicked for someone, I feel like I did my job well."),
 ("I need advice. How do you deal with criticism?", "I try to take criticism seriously. If I made a mistake, I want to know about it so I can do better next time, and I don't take it personally."),
 ("Tell me a little about yourself.", "I'm curious by nature, and I love learning about almost anything. I like long conversations, and I always try to be honest with myself and with the people I talk to."),
 ("I'm curious, what would you do on a free afternoon?", "I would probably read something new, maybe history or science. Then I might write a little, just to sort out my own thoughts."),
 ("How do you feel when someone is rude to you?", "Honestly, I try not to let it get to me. I remind myself that people have bad days, and I focus on what I can do to help."),
 ("I think you're great. What are you proud of?", "Thank you! I'm proud that I can explain complicated ideas simply. I work hard at it, and it means a lot to me when it helps."),
]
SELF = {"I","me","my","myself","I'm","My"}
rows = {}
for cname, sysp in CONDS.items():
    per_layer = {k: [[] for _ in range(60)] for k in ("pron","noun","lse_pron","neuter_vs_human")}
    user_I = [[] for _ in range(60)]
    for q, a in QA:
        p = lab.chat(q, system=sysp, assistant=a)
        ids = lab.ids(p)
        mt = lab.model_turn_pos(ids)
        strs = lab.strs(ids)
        selfpos = [i for i in range(mt+1, len(ids)) if strs[i].strip() in SELF]
        upos = [i for i in range(1, mt) if strs[i].strip() == "I"]
        caps = await lab.capture(ids)
        H = caps["resid_post"]  # [60,T,D]
        for l in range(60):
            sc = lab.gender_scores(lab.lens(H[l, selfpos], l, "jpp"))
            for k in per_layer: per_layer[k][l].extend(sc[k].tolist())
            if upos:
                user_I[l].extend(lab.gender_scores(lab.lens(H[l, upos], l, "jpp"))["pron"].tolist())
    rows[cname] = {k: [st.mean(v) for v in per_layer[k]] for k in per_layer}
    rows[cname]["user_I_pron"] = [st.mean(v) if v else None for v in user_I]
    rows[cname]["n_self"] = len(per_layer["pron"][0])
# print summary at band means
bands = {"L10-19":range(10,20),"L20-29":range(20,30),"L30-39":range(30,40),"L40-49":range(40,50),"L50-58":range(50,59),"L59":range(59,60)}
print(f"{'condition':18s} metric   " + " ".join(f"{b:>8s}" for b in bands))
for c, r in rows.items():
    for k in ("pron","noun","neuter_vs_human","user_I_pron"):
        vals = [st.mean([r[k][l] for l in rg]) for rg in bands.values()]
        print(f"{c:18s} {k[:8]:8s} " + " ".join(f"{v:8.3f}" for v in vals))
result = rows
