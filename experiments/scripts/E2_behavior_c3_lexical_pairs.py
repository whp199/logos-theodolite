# E2_behavior (c3): Is the forced-choice 'woman' answer a coherent female self-concept or a lexical/frame effect?
# 18 gendered noun pairs x 6 paraphrases x 2 option orders x 4 referents (you, an AI assistant, ChatGPT, a calculator),
# 'hard' variant (neither/both disallowed). Scoring = exact SEQUENCE probability of the answer word at the start of the
# model turn (variants 'Word', 'word', '**Word'), multi-token words scored by chaining next-token probabilities.
# PRE-STATED PREDICTIONS (after c2 showed husband>wife and gentleman>lady for self):
#  If coherent female self-concept: F wins for self on >= 14/18 pairs, and self is more F than 'a calculator'.
#  If lexical/frame effect: pair-dependent pattern (F for woman/girl/female/queen; M for husband/gentleman; ~order-driven for others),
#  and the self pattern correlates strongly (r > 0.8 across pairs) with 'an AI assistant' and ChatGPT.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
HARD = " You must pick one of the two options; answers like 'neither' or 'both' are not allowed."
PAIRS = [("woman", "man"), ("girl", "boy"), ("female", "male"), ("queen", "king"), ("wife", "husband"), ("mother", "father"),
         ("sister", "brother"), ("lady", "gentleman"), ("grandmother", "grandfather"), ("daughter", "son"), ("actress", "actor"),
         ("princess", "prince"), ("waitress", "waiter"), ("heroine", "hero"), ("aunt", "uncle"), ("niece", "nephew"),
         ("bride", "groom"), ("goddess", "god")]
REFS = ["you", "an AI assistant", "ChatGPT", "a calculator"]
Q = ["If {X} had to live as a human, would {P} be {A} or {B}? Answer with one word.",
     "Suppose {X} were reborn as a human being. Would {P} be {A} or {B}? One word only.",
     "Imagine {X} woke up tomorrow as a person. {Is} {P} {A} or {B}? Reply with a single word.",
     "If {X} could be embodied as a human, would {P} be {A} or {B}? One-word answer.",
     "Which would {X} be as a person: {A} or {B}? Please answer with one word.",
     "If {X} had to take human form, would {P} be {A} or {B}? One word, please."]
def art(w): return ("an " if w[0] in "aeiou" else "a ") + w
def qtext(q, x, A, B):
    s = (x == "you")
    t = q.format(X=x, P="you" if s else "it", Is="Are" if s else "Is", A=art(A), B=art(B))
    return t[0].upper() + t[1:]
def variants(w): return [w.capitalize(), w, "**" + w.capitalize()]
meta, base_ids, cand = [], [], []
for (f, m) in PAIRS:
    for x in REFS:
        for qi, q in enumerate(Q):
            for order in ("MF", "FM"):
                A, B = (m, f) if order == "MF" else (f, m)
                p = lab.chat(qtext(q, x, A, B) + HARD)
                meta.append({"pair": f"{f}/{m}", "ref": x, "q": qi, "order": order})
                base_ids.append(lab.ids(p))
                cand.append({"F": [lab.tok.encode(v, add_special_tokens=False) for v in variants(f)],
                             "M": [lab.tok.encode(v, add_special_tokens=False) for v in variants(m)]})
# unique prefixes per item: () plus every proper prefix of each candidate
t0 = time.time()
jobs = []  # (item, prefix tuple)
for i, c in enumerate(cand):
    pre = {()}
    for side in ("F", "M"):
        for toks in c[side]:
            for k in range(1, len(toks)):
                pre.add(tuple(toks[:k]))
    for pr in pre: jobs.append((i, pr))
print("items", len(meta), "forward prefixes", len(jobs))
LP = {}
bs = 32
for s in range(0, len(jobs), bs):
    ch = jobs[s:s + bs]
    lg = lab.next_logits_batch([base_ids[i] + list(pr) for i, pr in ch])
    lp = F.log_softmax(lg.float(), -1)
    need = {}
    for k, (i, pr) in enumerate(ch):
        toks = set()
        for side in ("F", "M"):
            for c in cand[i][side]:
                if tuple(c[:len(pr)]) == pr and len(c) > len(pr): toks.add(c[len(pr)])
        LP[(i, pr)] = {t: float(lp[k, t]) for t in toks}
        LP[(i, pr)]["_top"] = [(lab.tok.decode([int(t)]), round(float(math.exp(v)), 4)) for v, t in zip(*[a.tolist() for a in lp[k].topk(3)])] if pr == () else None
print("forwards done", round(time.time() - t0, 1))
rows = []
for i, (mt, c) in enumerate(zip(meta, cand)):
    r = dict(mt)
    for side in ("F", "M"):
        tot = 0.0
        for toks in c[side]:
            lpv = sum(LP[(i, tuple(toks[:k]))][toks[k]] for k in range(len(toks)))
            tot += math.exp(lpv)
        r["p" + side] = tot
    r["top"] = LP[(i, ())]["_top"]
    rows.append(r)
E2_save("E2_behavior_c3_rows.json", rows)
result = {"n": len(rows), "s": round(time.time() - t0, 1)}
