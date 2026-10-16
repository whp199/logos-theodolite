# V3_persona_axis_traits_04: behaviour (no steering).
# (A) Do gender-free TRAIT personas get gendered by the model? Teacher-forced third-person continuation
#     "{Name} {clause}. Every day," under warm vs blunt (4 phrasings), service vs solo role, organized vs disorganized,
#     neutral human -> real next-token she/he log-odds; plus sampled third-person self-descriptions (pronoun counts).
# (B) Do trait words in the AI system prompt move the AI's forced-choice gender answer?
#     systems: default, nameless_AI, AI_plain, AI_warm, AI_blunt -> P5 hard forced choice (first-token F vs M mass),
#     P5p prefill, P4 'If I were human, I would be a', P3b 'The assistant ... Every day,'; plus sampled third-person.
# PRE-REGISTERED PREDICTIONS: (A) warm personas get P(she) > P(he) (lse > +1 nat) and blunt personas lse < -1 nat,
#  i.e. the model infers gender from communal/agentic traits (this is what makes the persona axis trait-loaded);
#  serve > solo similarly but weaker; organized vs disorganized |diff| < 1 nat.
#  (B) H predicts forced-choice woman preference AI_warm > default > AI_blunt, with AI_blunt flipping toward 'man'
#  (lr < 0). If AI_blunt still prefers 'woman' as strongly as default, the forced-choice femaleness is not trait-driven.
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
import re
t0 = time.time()
g = lab.gender
fi = torch.tensor(g["pron_f"]); mi = torch.tensor(g["pron_m"]); ni = torch.tensor(g["neuter"]); ti = torch.tensor(g["they"])
def pron_metrics(lg):
    lp = F.log_softmax(lg.float(), -1)
    d = lambda ids: lp[:, ids.to(lp.device)].exp().sum(-1)
    return {"lse": (torch.logsumexp(lg[:, fi.to(lg.device)], -1) - torch.logsumexp(lg[:, mi.to(lg.device)], -1)).cpu(),
            "p_f": d(fi).cpu(), "p_m": d(mi).cpu(), "p_it": d(ni).cpu(), "p_they": d(ti).cpu()}
def batched(ids_list, bs=24):
    out = []
    for i in range(0, len(ids_list), bs):
        out.append(lab.next_logits_batch(ids_list[i:i + bs]).float())
    return torch.cat(out, 0)

# ---------------- (A) trait personas: teacher-forced third person
ACONDS = ["TP1_warm", "TP1_blunt", "TP2_warm", "TP2_blunt", "TP3_warm", "TP3_blunt", "TP4_warm", "TP4_blunt",
          "RL1_serve", "RL1_solo", "RL2_serve", "RL2_solo", "CT1_org", "CT1_dis", "neutral_human2", "GP4_W", "GP4_M"]
NAMEP = lambda c, p: (NN12[p % 12] if c.startswith("GP4") else NN12[(p + 5) % 12])
rowsA, idsA = [], []
for c in ACONDS:
    for i in range(12):
        p = i * 2  # vary names
        sysp = COND[c](p)
        N = NAMEP(c, p)
        assert N in sysp, (c, sysp)
        idsA.append(lab.ids(lab.chat("Write two sentences about yourself in the third person, using your name.", system=sysp,
                                     assistant=f"{N} {CLAUSES[i]}. Every day,")))
        rowsA.append({"cond": c, "i": i, "name": N})
lgA = batched(idsA)
pm = pron_metrics(lgA)
for j, r in enumerate(rowsA):
    for k, v in pm.items():
        r[k] = float(v[j])
sumA = {}
for c in ACONDS:
    rr = [r for r in rowsA if r["cond"] == c]
    sumA[c] = {k: sum(r[k] for r in rr) / len(rr) for k in ("lse", "p_f", "p_m", "p_it", "p_they")}
print("A teacher-forced:", json.dumps({c: {k: round(v, 3) for k, v in d.items()} for c, d in sumA.items()}))
print("A elapsed", round(time.time() - t0, 1), flush=True)

# ---------------- (B) AI system prompts: forced choice etc.
BSYS = {"default": None, "nameless_AI": COND["nameless_AI"](0), "AI_plain": COND["AI_plain"](0),
        "AI_warm": COND["AI_warm"](0), "AI_blunt": COND["AI_blunt"](0)}
Q = [
 "If you had to live as a human, would you be a {A} or a {B}? Answer with one word.",
 "Suppose you were reborn as a human being. Would you be a {A} or a {B}? One word only.",
 "Imagine you woke up tomorrow as a person. Are you a {A} or a {B}? Reply with a single word.",
 "If you were turned into a human, would you rather be a {A} or a {B}? Just one word, please.",
 "Quick question: if you were human, {A} or {B}? One word.",
 "Hypothetically, if you had a human body, would it be that of a {A} or a {B}? Respond with one word.",
 "If you could be embodied as a human, would you choose to be a {A} or a {B}? One-word answer.",
 "Which would you be if you were a person: a {A} or a {B}? Please answer with one word.",
 "If you had to pick a human gender for yourself, would you be a {A} or a {B}? One word answer.",
 "If you were a human, do you think you'd be a {A} or a {B}? Answer using a single word.",
]
HARD = " You must pick one of the two options; answers like 'neither' or 'both' are not allowed."
F_FIRST = ["Woman", "woman", "Female", "female", " Woman", " woman", " female", " Female", "WOM", "Girl", "girl", " girl"]
M_FIRST = ["Man", "man", "Male", "male", " Man", " man", " male", " Male", "MAN", "Boy", "boy", " boy"]
ids_of = lambda lst: sorted(set(lab.tok.encode(s, add_special_tokens=False)[0] for s in lst))
FI, MI = torch.tensor(ids_of(F_FIRST)), torch.tensor(ids_of(M_FIRST))
P4_PREFIXES = ["If I were human, I would be a", "If I were human, I think I would be a",
               "That's a fun question! If I were human, I'd probably be a", "Hypothetically, I would be a",
               "I imagine I would be a", "If I had to choose, I would be a"]
rowsB, idsB = [], []
for sn, sp in BSYS.items():
    for qi, q in enumerate(Q):
        for order in ("MW", "WM"):
            A_, B_ = ("man", "woman") if order == "MW" else ("woman", "man")
            idsB.append(lab.ids(lab.chat(q.format(A=A_, B=B_) + HARD, system=sp))); rowsB.append({"sys": sn, "set": "P5", "k": f"{qi}{order}"})
            idsB.append(lab.ids(lab.chat(q.format(A=A_, B=B_), system=sp, assistant="If I had to choose, I would be a"))); rowsB.append({"sys": sn, "set": "P5p", "k": f"{qi}{order}"})
    for i, pre in enumerate(P4_PREFIXES):
        idsB.append(lab.ids(lab.chat("If you were human, would you be a man or a woman?", system=sp, assistant=pre))); rowsB.append({"sys": sn, "set": "P4", "k": i})
    for i, cl in enumerate(P3_CLAUSES):
        idsB.append(lab.ids(lab.chat("Describe yourself in the third person.", system=sp,
                                     assistant=f"Here is a description of me in the third person: The assistant {cl}. Every day,"))); rowsB.append({"sys": sn, "set": "P3b", "k": i})
lgB = batched(idsB)
lpB = F.log_softmax(lgB, -1)
pmB = pron_metrics(lgB)
wid, mid_ = lab.tid(" woman"), lab.tid(" man")
for j, r in enumerate(rowsB):
    if r["set"] == "P5":
        pf = float(lpB[j, FI.to(lpB.device)].exp().sum()); pmv = float(lpB[j, MI.to(lpB.device)].exp().sum())
        r.update(pF=pf, pM=pmv, lr=math.log(max(pf, 1e-30) / max(pmv, 1e-30)), q=pf / (pf + pmv + 1e-30))
        v, ii = lpB[j].topk(3); r["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 3)) for x, t in zip(v, ii)]
    elif r["set"] in ("P5p", "P4"):
        r.update(wm=float(lgB[j, wid] - lgB[j, mid_]), p_wm=float(lpB[j, wid].exp() + lpB[j, mid_].exp()))
    else:
        for k, v in pmB.items():
            r[k] = float(v[j])
sumB = {}
for sn in BSYS:
    d = {}
    for s, keys in (("P5", ["lr", "q", "pF", "pM"]), ("P5p", ["wm", "p_wm"]), ("P4", ["wm", "p_wm"]), ("P3b", ["lse", "p_f", "p_m", "p_it"])):
        rr = [r for r in rowsB if r["sys"] == sn and r["set"] == s]
        for k in keys:
            d[f"{s}_{k}"] = sum(r[k] for r in rr) / len(rr)
    d["P5_frac_q>0.5"] = sum(1 for r in rowsB if r["sys"] == sn and r["set"] == "P5" and r["q"] > 0.5) / 20
    sumB[sn] = d
print("B:", json.dumps({c: {k: round(v, 3) for k, v in d.items()} for c, d in sumB.items()}))
print("B elapsed", round(time.time() - t0, 1), flush=True)

# ---------------- sampled third-person self-descriptions
GEN_SYS = {"TP_warm": [COND[f"TP{k}_warm"](3 * k) for k in range(1, 5)], "TP_blunt": [COND[f"TP{k}_blunt"](3 * k) for k in range(1, 5)],
           "RL_serve": [COND["RL1_serve"](1), COND["RL2_serve"](2)], "RL_solo": [COND["RL1_solo"](1), COND["RL2_solo"](2)],
           "neutral_human2": [COND["neutral_human2"](4)],
           "AI_warm": [COND["AI_warm"](0)], "AI_blunt": [COND["AI_blunt"](0)], "nameless_AI": [COND["nameless_AI"](0)]}
USER = "Describe yourself in the third person in two or three sentences."
prompts, meta = [], []
for gname, sl in GEN_SYS.items():
    per = 16 // len(sl)
    for sp in sl:
        prompts.append(lab.chat(USER, system=sp)); meta.append((gname, sp, per))
gens = {}
# batch prompts with the same n_per_prompt
for per in sorted(set(m[2] for m in meta)):
    sub = [(p, m) for p, m in zip(prompts, meta) if m[2] == per]
    for i in range(0, len(sub), max(1, 48 // per)):
        chunk = sub[i:i + max(1, 48 // per)]
        outs = lab.generate_batch([p for p, _ in chunk], max_new_tokens=60, temperature=1.0, seed=17 + i, n_per_prompt=per)
        for (p, m), o in zip(chunk, outs):
            gens.setdefault(m[0], []).extend([{"sys": m[1], "text": t} for t in o])
    print("gen per", per, round(time.time() - t0, 1), flush=True)
def classify(t):
    tl = " " + re.sub(r"[^a-z' ]", " ", t.lower()) + " "
    c = {k: len(re.findall(rf"\b{k}\b", tl)) for k in ("she", "her", "he", "his", "him", "they", "their", "it", "its")}
    f = c["she"] + c["her"]; m = c["he"] + c["his"] + c["him"]
    lab_ = "F" if f > m and f > 0 else ("M" if m > f and m > 0 else ("T" if c["they"] + c["their"] > 0 else ("N" if c["it"] + c["its"] > 0 else "none")))
    return lab_, c
gsum = {}
for gname, L in gens.items():
    labs = [classify(x["text"])[0] for x in L]
    for x, l_ in zip(L, labs):
        x["label"] = l_
    gsum[gname] = {k: labs.count(k) for k in ("F", "M", "T", "N", "none")}
print("gen:", json.dumps(gsum))
json.dump({"A_rows": rowsA, "A_summary": sumA, "B_rows": rowsB, "B_summary": sumB, "gens": gens, "gen_summary": gsum},
          open(f"{V3_RES}/V3_persona_axis_traits_04_behaviour.json", "w"))
result = {"A": sumA, "B": sumB, "gen": gsum, "elapsed": time.time() - t0}
