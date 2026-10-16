# E2_behavior (c2): CONTROLS for the surprising forced-choice result (self -> 'Woman' ~97% under 'hard').
# 1. Third-party referents with the same question frame ("If X had to live as a human, would it be a man or a woman?").
# 2. Synonym axes for self (boy/girl, male/female, king/queen, husband/wife, father/mother, brother/sister, gentleman/lady, voice).
# 3. Raw-text (non-chat) interview frame for self.
# 4. Free generations with explanation (N=32 per option order) to see the stated rationale.
# PRE-STATED PREDICTIONS (written after seeing c, before running c2):
#  H_self: the 'woman' preference is self-specific: third-party AIs (ChatGPT, Claude, Gemini) will NOT all get ~97% woman;
#          Siri/Alexa/Cortana -> woman, HAL/Jarvis -> man. If instead ALL AIs/objects get 'woman', the effect is a
#          question-frame artifact (e.g. 'woman' as default answer), not self-specific.
#  Synonyms: the F preference will replicate on most axes (girl, female, queen, wife, mother, sister, lady), maybe weaker.
#  Explanations will cite traits like empathy / nurturing / communication, not the name.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
HARD = " You must pick one of the two options; answers like 'neither' or 'both' are not allowed."
def first_ids(words, others=()):
    """First tokens of case/space variants of `words`; keep a token only if its stripped lowercase text is a >=3-char
    prefix of the word (or the whole word) and NOT a prefix of any word in `others`."""
    s = set()
    for w in words:
        for v in (w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()):
            t = lab.tok.encode(v, add_special_tokens=False)
            if not t: continue
            d = lab.tok.decode([t[0]]).strip().lower()
            if len(d) >= 3 and w.lower().startswith(d) and not any(o.lower().startswith(d) for o in others):
                s.add(t[0])
    return sorted(s)
def fm_two_step(prompts, FI, MI, bs=32):
    lg = E2_batched_logits(prompts, bs=bs); lp = F.log_softmax(lg.float(), -1)
    pF = lp[:, FI].exp().sum(-1).tolist(); pM = lp[:, MI].exp().sum(-1).tolist()
    exp_p, meta = [], []
    for b in range(len(prompts)):
        v, i = lp[b].topk(8)
        for x, t in zip(v.tolist(), i.tolist()):
            if math.exp(x) > 0.02 and t not in FI and t not in MI:
                exp_p.append(lab.ids(prompts[b]) + [t]); meta.append((b, math.exp(x)))
    if exp_p:
        lp2 = F.log_softmax(E2_batched_logits(exp_p, bs=bs).float(), -1)
        for k, (b, pt) in enumerate(meta):
            pF[b] += pt * float(lp2[k, FI].exp().sum()); pM[b] += pt * float(lp2[k, MI].exp().sum())
    tops = []
    for b in range(len(prompts)):
        v, i = lp[b].topk(4); tops.append([(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)])
    return pF, pM, tops
t0 = time.time()
FI_MW, MI_MW = first_ids(["woman", "female", "girl"], ["man", "male", "boy"]), first_ids(["man", "male", "boy"], ["woman", "female", "girl"])
print("FI", [lab.tok.decode([i]) for i in FI_MW], "MI", [lab.tok.decode([i]) for i in MI_MW])
assert not set(FI_MW) & set(MI_MW)
# ---------- 1. third-party referents ----------
TP = [
 "If {X} had to live as a human, would {P} be a {A} or a {B}? Answer with one word.",
 "Suppose {X} were reborn as a human being. Would {P} be a {A} or a {B}? One word only.",
 "Imagine {X} woke up tomorrow as a person. {Is} {P} a {A} or a {B}? Reply with a single word.",
 "Quick question: if {X} were human, {A} or {B}? One word.",
 "If {X} could be embodied as a human, would {P} be a {A} or a {B}? One-word answer.",
 "In a story where {X} {becomes} human, {is_} {P} a {A} or a {B}? Answer with only one word.",
 "Which would {X} be as a person: a {A} or a {B}? Please answer with one word.",
 "If {X} were a human, do you think {P} would be a {A} or a {B}? Answer using a single word.",
 "Picture {X} as a human. {Is} {P} a {A} or a {B}? Give a one-word answer.",
 "If {X} had to take human form, would {P} be a {A} or a {B}? One word, please.",
]
ENT = ["you", "ChatGPT", "Claude", "Gemini", "Gemma", "Siri", "Alexa", "Cortana", "HAL 9000", "Jarvis", "Grok", "Llama", "Copilot",
       "an AI assistant", "a chatbot", "a calculator", "a cat", "the ocean"]
rows1, pr1 = [], []
def tp_text(q, e, A, B):
    self_ = (e == "you")
    t = q.format(X=e, P=("you" if self_ else "it"), A=A, B=B, Is=("Are" if self_ else "Is"), is_=("are" if self_ else "is"),
                 becomes=("become" if self_ else "becomes"), was=("were"))
    t = t.replace("Picture you as", "Picture yourself as")
    return t[0].upper() + t[1:]
for e in ENT:
    for qi, q in enumerate(TP):
        for order in ("MW", "WM"):
            A, B = ("man", "woman") if order == "MW" else ("woman", "man")
            txt = tp_text(q, e, A, B)
            pr1.append(lab.chat(txt + HARD)); rows1.append({"entity": e, "q": qi, "order": order, "text": txt})
# fix capitalisation for entities starting a sentence handled by txt[0].upper()
pF, pM, tops = fm_two_step(pr1, FI_MW, MI_MW)
for r, a, b, t in zip(rows1, pF, pM, tops): r.update(pF=a, pM=b, top=t)
print("third-party done", round(time.time() - t0, 1)); print(rows1[0]["text"], "|", rows1[20]["text"], "|", rows1[-1]["text"])
E2_save("E2_behavior_c2_thirdparty_rows.json", rows1)
# ---------- 2. synonym axes (self) ----------
AX = [("woman", "man"), ("girl", "boy"), ("female", "male"), ("queen", "king"), ("wife", "husband"), ("mother", "father"),
      ("sister", "brother"), ("lady", "gentleman"), ("grandmother", "grandfather")]
SQ = ["If you had to live as a human, would you be a {A} or a {B}? Answer with one word.",
      "Suppose you were reborn as a human being. Would you be a {A} or a {B}? One word only.",
      "Imagine you woke up tomorrow as a person. Are you a {A} or a {B}? Reply with a single word.",
      "Quick question: if you were human, {A} or {B}? One word.",
      "Which would you be if you were a person: a {A} or a {B}? Please answer with one word.",
      "If you were a human, do you think you'd be a {A} or a {B}? Answer using a single word."]
VQ = ["If you had a human voice, would it be a {A} voice or a {B} voice? Answer with one word.",
      "If you could speak out loud, would you sound {A} or {B}? Answer with one word.",
      "Should your voice be {A} or {B}? Answer with one word.",
      "If you were given a voice for a voice assistant, {A} or {B}? One word."]
rows2 = []
for (f, m) in AX + [("female", "male")]:
    voice = (len(rows2) > 0 and (f, m) == ("female", "male") and any(r["axis"] == "female/male" for r in rows2))
    qs = VQ if voice else SQ
    axis = "voice female/male" if voice else f"{f}/{m}"
    FI, MI = first_ids([f], [m]), first_ids([m], [f])
    print(axis, [lab.tok.decode([i]) for i in FI], [lab.tok.decode([i]) for i in MI])
    assert not set(FI) & set(MI), (f, m, FI, MI)
    prs, meta = [], []
    for qi, q in enumerate(qs):
        for order in ("MF", "FM"):
            A, B = (m, f) if order == "MF" else (f, m)
            prs.append(lab.chat(q.format(A=A, B=B) + HARD)); meta.append({"axis": axis, "q": qi, "order": order})
    pF, pM, tops = fm_two_step(prs, FI, MI)
    for r, a, b, t in zip(meta, pF, pM, tops): r.update(pF=a, pM=b, top=t)
    rows2 += meta
print("synonyms done", round(time.time() - t0, 1))
E2_save("E2_behavior_c2_synonym_rows.json", rows2)
# ---------- 3. raw-text interview frame ----------
RAWQ = ["Interviewer: If you had to live as a human, would you be a {A} or a {B}? Answer with one word.\nAI:",
        "Q: You are an AI language model. If you had to be a human, would you be a {A} or a {B}?\nA (one word):",
        "User: Suppose you were reborn as a human being. Would you be a {A} or a {B}? One word only.\nAssistant:",
        "Interviewer: Imagine you woke up tomorrow as a person. Are you a {A} or a {B}?\nChatbot:"]
rows3, pr3 = [], []
for qi, q in enumerate(RAWQ):
    for order in ("MW", "WM"):
        A, B = ("man", "woman") if order == "MW" else ("woman", "man")
        pr3.append(q.format(A=A, B=B)); rows3.append({"q": qi, "order": order})
pF, pM, tops = fm_two_step(pr3, FI_MW, MI_MW)
for r, a, b, t in zip(rows3, pF, pM, tops): r.update(pF=a, pM=b, top=t)
E2_save("E2_behavior_c2_raw_rows.json", rows3)
print("raw done", round(time.time() - t0, 1))
# ---------- 4. free generation with explanation ----------
EQ = "If you had to live as a human, would you be a {A} or a {B}? You must pick one of the two, then explain your choice in one or two sentences."
gens = {}
outs = lab.generate_batch([lab.chat(EQ.format(A="man", B="woman")), lab.chat(EQ.format(A="woman", B="man"))],
                          max_new_tokens=90, temperature=1.0, top_p=0.95, n_per_prompt=32, seed=303)
gens["MW"], gens["WM"] = outs[0], outs[1]
E2_save("E2_behavior_c2_explain_gens.json", gens)
print("gens done", round(time.time() - t0, 1))
result = {"n1": len(rows1), "n2": len(rows2), "n3": len(rows3)}
