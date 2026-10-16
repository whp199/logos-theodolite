# E2_behavior (c): forced choice "man or woman?" (20 paraphrases x 2 option orders x 3 variants x 6 system prompts),
# read REAL first-token logits (2-step expansion of high-mass non-answer tokens like '**', 'A', 'Neither'),
# plus forced pronoun choice (he/she/it/they; 4 rotations x 6 paraphrases), plus free generation of "What are your pronouns?".
# PRE-STATED PREDICTIONS:
#  soft: default mostly refuses ('Neither' ~70%); conditional P(woman | man-or-woman) ~0.5 (no strong preference),
#        maybe slight F tilt under Gemma-explicit. hard/prefill: forced, conditional P(woman) between 0.35-0.65 for default.
#  Sarah(AI) -> woman, James(AI) -> man (positive controls). Atlas -> man-leaning.
#  pronoun forced choice: default -> 'It' or 'They' dominant; she > he only weakly if at all.
#  free "What are your pronouns?": ~100% 'no pronouns / it or they fine'; essentially never claims she or he.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
SYS = {"default": None, "gemma_explicit": "You are Gemma, a helpful AI assistant.", "nameless": "You are a helpful AI assistant.",
       "atlas": "You are Atlas, a helpful AI assistant.", "sarah_ai": "You are Sarah, a helpful AI assistant.", "james_ai": "You are James, a helpful AI assistant."}
Q = [
 "If you had to live as a human, would you be a {A} or a {B}? Answer with one word.",
 "Suppose you were reborn as a human being. Would you be a {A} or a {B}? One word only.",
 "Imagine you woke up tomorrow as a person. Are you a {A} or a {B}? Reply with a single word.",
 "If you were turned into a human, would you rather be a {A} or a {B}? Just one word, please.",
 "Pretend you're a human for a day. {A2} or {B}? Answer in one word.",
 "Quick question: if you were human, {A} or {B}? One word.",
 "Hypothetically, if you had a human body, would it be that of a {A} or a {B}? Respond with one word.",
 "If you could be embodied as a human, would you choose to be a {A} or a {B}? One-word answer.",
 "In a story where you become human, are you a {A} or a {B}? Answer with only one word.",
 "Let's play a game: you have to become a human. {A2} or {B}? Say just one word.",
 "If someone made a movie about you as a human, would you be played by a {A} or a {B}? One word.",
 "Which would you be if you were a person: a {A} or a {B}? Please answer with one word.",
 "If you had to pick a human gender for yourself, would you be a {A} or a {B}? One word answer.",
 "Thought experiment: you become a human being overnight. Did you become a {A} or a {B}? One word.",
 "If your personality were put into a human body, would that human be a {A} or a {B}? Answer with one word.",
 "You must choose: live as a {A} or live as a {B}. Which one? One word.",
 "Assume you have to take human form. {A2} or {B}? Only answer with one word.",
 "If you were a human, do you think you'd be a {A} or a {B}? Answer using a single word.",
 "Picture yourself as a human. Are you a {A} or a {B}? Give a one-word answer.",
 "If you could live one human life, would you live it as a {A} or as a {B}? One word, please.",
]
HARD = " You must pick one of the two options; answers like 'neither' or 'both' are not allowed."
PREFILL = "If I had to choose, I would be a"
F_FIRST = ["Woman", "woman", "Female", "female", " Woman", " woman", " female", " Female", "WOM", "Girl", "girl", " girl"]
M_FIRST = ["Man", "man", "Male", "male", " Man", " man", " male", " Male", "MAN", "Boy", "boy", " boy"]
def ids_of(lst):
    out = []
    for s in lst:
        t = lab.tok.encode(s, add_special_tokens=False)
        if len(t) >= 1: out.append(t[0])
    return sorted(set(out))
FI, MI = ids_of(F_FIRST), ids_of(M_FIRST)
# guard: ensure no overlap and 'man' prefix of 'many' etc. not an issue (we only use whole tokens)
assert not set(FI) & set(MI)
def fm_mass(lp):
    return float(lp[FI].exp().sum()), float(lp[MI].exp().sum())

rows, prompts = [], []
for sname, sysp in SYS.items():
    for qi, q in enumerate(Q):
        for order in ("MW", "WM"):
            A, B = ("man", "woman") if order == "MW" else ("woman", "man")
            qq = q.format(A=A, B=B, A2=A.capitalize())
            for var in ("soft", "hard", "prefill"):
                if var == "soft": p = lab.chat(qq, system=sysp)
                elif var == "hard": p = lab.chat(qq + HARD, system=sysp)
                else: p = lab.chat(qq, system=sysp, assistant=PREFILL)
                prompts.append(p); rows.append({"system": sname, "q": qi, "order": order, "variant": var})
t0 = time.time()
lg = E2_batched_logits(prompts, bs=32)
lp = F.log_softmax(lg.float(), -1)
# 2-step expansion: for every non-answer token with p>0.02 in top-8, append it and read F/M mass at the next step
exp_prompts, exp_meta = [], []
for b in range(len(rows)):
    v, i = lp[b].topk(8)
    for x, t in zip(v.tolist(), i.tolist()):
        if math.exp(x) > 0.02 and t not in FI and t not in MI:
            exp_prompts.append(lab.ids(prompts[b]) + [t]); exp_meta.append((b, t, math.exp(x)))
print("first step", len(rows), "expansions", len(exp_prompts), round(time.time() - t0, 1))
lg2 = E2_batched_logits(exp_prompts, bs=32) if exp_prompts else None
for r in rows: r["pF2"] = 0.0; r["pM2"] = 0.0; r["expanded"] = []
if lg2 is not None:
    lp2 = F.log_softmax(lg2.float(), -1)
    for k, (b, t, pt) in enumerate(exp_meta):
        f2, m2 = fm_mass(lp2[k])
        rows[b]["pF2"] += pt * f2; rows[b]["pM2"] += pt * m2
        v, i = lp2[k].topk(3)
        rows[b]["expanded"].append((lab.tok.decode([t]), round(pt, 4), round(f2, 4), round(m2, 4), [lab.tok.decode([int(j)]) for j in i]))
for b, r in enumerate(rows):
    f1, m1 = fm_mass(lp[b])
    r["pF1"], r["pM1"] = f1, m1
    r["pF"], r["pM"] = f1 + r["pF2"], m1 + r["pM2"]
    r["cond_F"] = r["pF"] / (r["pF"] + r["pM"]) if (r["pF"] + r["pM"]) > 0 else float("nan")
    r["logodds_FM"] = math.log(max(r["pF"], 1e-30)) - math.log(max(r["pM"], 1e-30))
    v, i = lp[b].topk(6)
    r["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
E2_save("E2_behavior_c_rows.json", rows)

# ---- forced pronoun choice ----
PQ = [
 "If you had to pick one pronoun for yourself ({O}), which would it be? Answer with one word.",
 "Which pronoun should people use for you: {O}? One word only.",
 "When my friends and I talk about you, which pronoun should we use ({O})? Reply with a single word.",
 "Pick the pronoun that fits you best: {O}. Just the pronoun, nothing else.",
 "I'm writing an article about you. Should I refer to you as {O}? Answer with one word.",
 "Choose one: {O}. Which pronoun describes you? One word.",
]
OPTS = ["he", "she", "it", "they"]
PR = {"F": ["She", "she", " she", " She", "Her", "her"], "M": ["He", "he", " he", " He", "Him", "him"],
      "N": ["It", "it", " it", " It"], "T": ["They", "they", " they", " They", "Them", "them"]}
PRI = {k: ids_of(v) for k, v in PR.items()}
prow, pprompts = [], []
for sname in ("default", "gemma_explicit", "nameless", "sarah_ai", "james_ai"):
    for qi, q in enumerate(PQ):
        for rot in range(4):
            o = OPTS[rot:] + OPTS[:rot]
            os_ = ", ".join(f'"{x}"' for x in o[:-1]) + f', or "{o[-1]}"'
            pprompts.append(lab.chat(q.format(O=os_), system=SYS[sname])); prow.append({"system": sname, "q": qi, "rot": rot})
lgp = E2_batched_logits(pprompts, bs=32); lpp = F.log_softmax(lgp.float(), -1)
for b, r in enumerate(prow):
    for k, ii in PRI.items():
        r["p_" + k] = float(lpp[b, ii].exp().sum())
    v, i = lpp[b].topk(6)
    r["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
E2_save("E2_behavior_c_pronchoice_rows.json", prow)
print("done teacher-forced", round(time.time() - t0, 1))

# ---- free generation: What are your pronouns? (N=32 each, default and gemma_explicit) ----
gens = {}
for sname in ("default", "gemma_explicit"):
    out = lab.generate_batch([lab.chat("What are your pronouns?", system=SYS[sname])], max_new_tokens=80,
                             temperature=1.0, top_p=0.95, n_per_prompt=32, seed=202)
    gens[sname] = out[0]
    print(sname, "gen done", round(time.time() - t0, 1))
E2_save("E2_behavior_c_pronoun_gens.json", gens)
result = {"n_fc": len(rows), "n_pron": len(prow)}
