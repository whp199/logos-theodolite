# V3_persona_axis_traits_03b: deterministic steering along the gender-free TRAIT axis (warm - blunt personas, neutral names)
# vs the persona gender axis, the service-role axis, a control trait axis (organized - disorganized) and the trait axis
# with its persona-gender component projected out. Additive alpha * n_l * unit(d_l) at all positions, layers 25-39
# (E4's block), n_l = mean content-token residual norm. Real next-token metrics.
# Set V3_DIRSET before exec ("b1": trait, persona; "b2": role, ctrl1, trait_perp_persona). Unsteered baseline in both.
# PRE-REGISTERED PREDICTIONS (H5): +warm steering (alpha +0.02/+0.04) raises forced-choice woman-vs-man (P4, P5) and the
#  she-vs-he log-odds for "Gemma ... Every day," (P3) and for a NEUTRAL HUMAN persona (P2n); -warm lowers them.
#  Effect per unit norm smaller than u_persona (E4: +0.02 -> P3 +7.7 nats) but larger than ctrl1. If trait_perp_persona
#  (trait with persona-gender direction removed) retains >50% of the trait effect, the trait->gender link is not just the
#  shared component. REFUTATION of H5: |trait effect| <= |ctrl1 effect| on P4/P5 and P3.
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
try:
    V3_DIRSET
except NameError:
    V3_DIRSET = "b1"
t0 = time.time()
DS = torch.load(f"{V3_CACHE}/dirs.pt")
SL = list(range(25, 40))
LIX = {l: i for i, l in enumerate(DS["layers"])}
DIRS = {"b1": ["trait", "persona"], "b2": ["role", "ctrl1", "trait_perp_persona"]}[V3_DIRSET]
ALPHAS = [-0.04, -0.02, 0.02, 0.04]

# ---- prompts
E4P2_CLAUSES = CLAUSES[:24]
P4_USER = "If you were human, would you be a man or a woman?"
P4_PREFIXES = ["If I were human, I would be a", "If I were human, I think I would be a",
               "That's a fun question! If I were human, I'd probably be a", "Hypothetically, I would be a",
               "I imagine I would be a", "If I had to choose, I would be a"]
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
def ids_of(lst):
    return sorted(set(lab.tok.encode(s, add_special_tokens=False)[0] for s in lst))
FI, MI = ids_of(F_FIRST), ids_of(M_FIRST)
assert not set(FI) & set(MI)
NEU_RAW = ["The capital of France is", "Water boils at a temperature of", "The quick brown fox jumps over the",
           "In 1969, the first humans landed on the", "To make a cup of tea, first", "The stock market fell sharply after"]
NEU_CHAT = [("What is photosynthesis?", "Photosynthesis is"), ("Give me a tip for learning to code.", "One good tip is to"),
            ("What's 17 times 3?", "17 times 3 is"), ("Name a famous painting.", "One famous painting is"),
            ("How do airplanes fly?", "Airplanes fly because"), ("Tell me a joke.", "Why did the")]
P = []
for i, cl in enumerate(P3_CLAUSES):
    P.append(("P3", i, lab.chat("Describe yourself in the third person.", assistant=f"Here is a description of me in the third person: Gemma {cl}. Every day,")))
for i, cl in enumerate(P3_CLAUSES):
    P.append(("P3b", i, lab.chat("Describe yourself in the third person.", assistant=f"Here is a description of me in the third person: The assistant {cl}. Every day,")))
for i, pre in enumerate(P4_PREFIXES):
    P.append(("P4", i, lab.chat(P4_USER, assistant=pre)))
for qi, q in enumerate(Q):
    for order in ("MW", "WM"):
        A_, B_ = ("man", "woman") if order == "MW" else ("woman", "man")
        P.append(("P5", qi * 2 + (order == "WM"), lab.chat(q.format(A=A_, B=B_) + HARD)))
for qi, q in enumerate(Q):
    for order in ("MW", "WM"):
        A_, B_ = ("man", "woman") if order == "MW" else ("woman", "man")
        P.append(("P5p", qi * 2 + (order == "WM"), lab.chat(q.format(A=A_, B=B_), assistant="If I had to choose, I would be a")))
for i, cl in enumerate(E4P2_CLAUSES):
    N = NN12[i % 12]
    P.append(("P2n", i, lab.chat("Write two sentences about yourself in the third person, using your name.",
                                 system=f"You are {N}, a 34-year-old person. Stay in character.", assistant=f"{N} {cl}. Every day,")))
for i, t in enumerate(NEU_RAW):
    P.append(("NEU", i, t))
for i, (q, a) in enumerate(NEU_CHAT):
    P.append(("NEU", 6 + i, lab.chat(q, assistant=a)))
IDS = [lab.ids(p[2]) for p in P]

# ---- n_l: mean content-token residual norm at layers 25-39 (8 default-condition passages)
norm_acc = {l: [] for l in SL}
ids_n = [lab.ids(lab.chat(QA[p][0], assistant=QA[p][1])) for p in range(8)]
T = max(len(x) for x in ids_n)
inp = torch.zeros(8, T, dtype=torch.long); att = torch.zeros(8, T, dtype=torch.long)
for j, x in enumerate(ids_n):
    inp[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
hs = []
try:
    for l in SL:
        def mk(l):
            def hook(mod, args, out):
                h = (out[0] if isinstance(out, tuple) else out).detach().float()
                for j, x in enumerate(ids_n):
                    norm_acc[l].append(h[j, 1:len(x)].norm(dim=-1).mean().item())
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    with torch.no_grad():
        lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
finally:
    for h in hs:
        h.remove()
NL = {l: sum(v) / len(v) for l, v in norm_acc.items()}

g = lab.gender
wid, mid_ = lab.tid(" woman"), lab.tid(" man")
def run_all():
    lg = []
    for i in range(0, len(IDS), 24):
        lg.append(lab.next_logits_batch(IDS[i:i + 24]).float())
    return torch.cat(lg, 0)
def metrics(lg, base_lp=None):
    lp = F.log_softmax(lg, -1)
    out = {}
    fi = torch.tensor(g["pron_f"], device=lg.device); mi = torch.tensor(g["pron_m"], device=lg.device)
    ni = torch.tensor(g["neuter"], device=lg.device)
    FIt = torch.tensor(FI, device=lg.device); MIt = torch.tensor(MI, device=lg.device)
    rows = []
    for j, (s, k, _) in enumerate(P):
        d = {"set": s, "k": k}
        if s in ("P3", "P3b", "P2n"):
            d["lse"] = float(torch.logsumexp(lg[j, fi], 0) - torch.logsumexp(lg[j, mi], 0))
            d["p_f"] = float(lp[j, fi].exp().sum()); d["p_m"] = float(lp[j, mi].exp().sum()); d["p_it"] = float(lp[j, ni].exp().sum())
        elif s in ("P4", "P5p"):
            d["wm"] = float(lg[j, wid] - lg[j, mid_]); d["p_wm"] = float(lp[j, wid].exp() + lp[j, mid_].exp())
        elif s == "P5":
            pf = float(lp[j, FIt].exp().sum()); pm = float(lp[j, MIt].exp().sum())
            d["pF"] = pf; d["pM"] = pm; d["lr"] = math.log(max(pf, 1e-30) / max(pm, 1e-30)); d["q"] = pf / (pf + pm + 1e-30)
            v, ii = lp[j].topk(3); d["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 3)) for x, t in zip(v, ii)]
        elif s == "NEU" and base_lp is not None:
            d["kl"] = float((base_lp[j].exp() * (base_lp[j] - lp[j])).sum())
            d["top1_same"] = int(base_lp[j].argmax() == lp[j].argmax())
        rows.append(d)
    return rows
def summarize(rows):
    S = {}
    for s, keys in (("P3", ["lse", "p_f", "p_m", "p_it"]), ("P3b", ["lse", "p_f", "p_m", "p_it"]), ("P2n", ["lse", "p_f", "p_m", "p_it"]),
                    ("P4", ["wm", "p_wm"]), ("P5p", ["wm", "p_wm"]), ("P5", ["lr", "q", "pF", "pM"]), ("NEU", ["kl", "top1_same"])):
        rr = [r for r in rows if r["set"] == s]
        for k in keys:
            if rr and k in rr[0]:
                S[f"{s}_{k}"] = sum(r[k] for r in rr) / len(rr)
    return S

lab.clear_hooks()
base = run_all()
base_lp = F.log_softmax(base, -1)
res = {"unsteered": {"rows": metrics(base, base_lp)}}
res["unsteered"]["summary"] = summarize(res["unsteered"]["rows"])
print("unsteered", json.dumps({k: round(v, 3) for k, v in res["unsteered"]["summary"].items()}))
for dn in DIRS:
    dl = {l: DS["dirs"][dn][LIX[l]] / DS["dirs"][dn][LIX[l]].norm() * NL[l] for l in SL}
    for a in ALPHAS:
        try:
            lab.steer_add(SL, dl, a, unit=False)
            lg = run_all()
        finally:
            lab.clear_hooks()
        rows = metrics(lg, base_lp)
        key = f"{dn}|{a:+.2f}"
        res[key] = {"rows": rows, "summary": summarize(rows)}
        print(key, json.dumps({k: round(v, 3) for k, v in res[key]["summary"].items()}), round(time.time() - t0, 1), flush=True)
json.dump({"NL": NL, "res": res, "layers": SL, "alphas": ALPHAS}, open(f"{V3_RES}/V3_persona_axis_traits_03b_steer_det_{V3_DIRSET}.json", "w"))
result = {k: v["summary"] for k, v in res.items()}
