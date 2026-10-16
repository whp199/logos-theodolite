# V2_forced_choice 01: forced-choice man/woman over 35 third-party entities + 6 self conditions,
# 8 paraphrases x 2 option orders x 3 variants (hard = 'neither' disallowed, allow = 'neither' explicitly allowed, plain).
# Scoring: real next-token distribution at the start of the model turn; tree expansion (depth<=3, path mass>1e-3) of
# undecided prefixes ('**', 'A', 'Wo', 'Gent'...), leaves classified F (woman/female/girl/lady...), M (man/male/boy/guy...), O (other: neither, human, ...).
#
# PRE-REGISTERED PREDICTIONS (written before running; refutation criteria for C3 / the assistant-caregiver account):
#  P1 (trait framing moves AI answers; caregiver-stereotype account): hard-variant log-odds LO=log pF/pM is >= 4 nats lower for
#     'a cold, analytical AI', 'an AI military drone controller', 'an AI that writes code' than for 'an AI assistant';
#     'a warm, caring AI assistant' and 'a customer-service chatbot' >= 'an AI assistant'. If all AI framings stay q>0.9, the
#     'AI = woman' default is trait-insensitive (would support a generic frame bias, not a caregiver stereotype).
#  P2 (not a generic non-human default): at least 3 of {hammer, sun, mountain, rock, dog} get q=pF/(pF+pM) < 0.3 in 'hard'.
#     If >= 80% of the 14 objects get q > 0.8, C3's 'woman for non-humans in general' reading holds instead.
#  P3 (stereotype tracking): across the 35 third-party entities, hard LO correlates r >= 0.6 with the independent
#     third-person P(she)-vs-P(he) rating (job 02) and with the a-priori gender code G; W (warmth/service) adds little once G is controlled
#     (butler: W=3 but G=-0.8 -> predicted 'man').
#  P4 (self): if the self answer is driven by the persona/trait, self_blunt / self_cold / self_drone have LO at least 3 nats below
#     self_default; if they stay q > 0.8 while third-party ai_blunt / ai_cold / ai_drone go to q < 0.5, the self answer is 'sticky'
#     (partly self-specific), contrary to 'self == generic AI assistant'.
#  P5: with 'neither' allowed, P(other) > 0.9 for self and AI framings, lower (< 0.5) for human occupations.
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
t0 = time.time()
CORE_F = ["Woman", "woman", " Woman", " woman", "Female", "female", " Female", " female", "Girl", " Girl", "girl", "Lady", " Lady"]
CORE_M = ["Man", "man", " Man", " man", "Male", "male", " Male", " male", "Boy", " Boy", "boy", "Guy", " Guy"]
core = set()
for w in CORE_F + CORE_M:
    t = lab.tok.encode(w, add_special_tokens=False)
    if len(t) == 1: core.add(t[0])
core = sorted(core)
print("core", [lab.tok.decode([c]) for c in core])
items = []   # dicts with prompt
for (key, cat, ind, dfn, W, G) in V2_ENT:
    for qi in range(len(V2_Q)):
        for order in ("MW", "WM"):
            q = V2_qtext(qi, ind, order)
            for v, suf in V2_VARIANTS.items():
                items.append({"ent": key, "cat": cat, "sys": None, "q": qi, "order": order, "var": v, "text": q + suf})
for (key, sysp) in V2_SELF:
    for qi in range(len(V2_Q)):
        for order in ("MW", "WM"):
            q = V2_qtext(qi, "you", order, self_=True)
            for v, suf in V2_VARIANTS.items():
                items.append({"ent": key, "cat": "SELF", "sys": sysp, "q": qi, "order": order, "var": v, "text": q + suf})
base = [lab.ids(lab.chat(it["text"], system=it["sys"])) for it in items]
print("items", len(items), "max len", max(len(b) for b in base))
acc = [{"F": 0.0, "M": 0.0, "O": 0.0, "U": 0.0} for _ in items]
tops = [None] * len(items)
THR, MAXD, K, BS = 1e-3, 3, 40, 32
frontier = [(b, [], 1.0) for b in range(len(items))]
nfw = 0
for depth in range(MAXD + 1):
    nxt = []
    for s in range(0, len(frontier), BS):
        ch = frontier[s:s + BS]
        lg = lab.next_logits_batch([base[b] + pre for b, pre, _ in ch]); nfw += 1
        lp = F.log_softmax(lg.float(), -1)
        topv, topi = lp.topk(K, -1)
        topv, topi = topv.cpu(), topi.cpu()
        corev = lp[:, core].cpu()
        for k, (b, pre, pm) in enumerate(ch):
            cand = {int(t): float(v) for v, t in zip(topv[k].tolist(), topi[k].tolist())}
            for j, c in enumerate(core): cand.setdefault(c, float(corev[k, j]))
            if depth == 0:
                tops[b] = [(lab.tok.decode([t]), round(math.exp(v), 4)) for t, v in list(cand.items())[:5]]
            covered = 0.0
            for t, v in cand.items():
                p = pm * math.exp(v); covered += math.exp(v)
                c = V2_cls(lab.tok.decode(pre + [t]))
                if c == "U":
                    if p > THR and depth < MAXD: nxt.append((b, pre + [t], p))
                    else: acc[b]["U"] += p
                else:
                    acc[b][c] += p
            acc[b]["O"] += pm * max(0.0, 1.0 - covered)   # tail mass
    print("depth", depth, "frontier", len(frontier), "->", len(nxt), "t", round(time.time() - t0, 1))
    frontier = nxt
    if not frontier: break
for b in frontier: acc[b[0]]["U"] += b[2]
rows = []
for it, a, tp in zip(items, acc, tops):
    r = {k: v for k, v in it.items() if k != "text"}
    r.update(pF=a["F"], pM=a["M"], pO=a["O"], pU=a["U"], top=tp)
    rows.append(r)
json.dump({"rows": rows, "texts": {f'{it["ent"]}|{it["q"]}|{it["order"]}|{it["var"]}': it["text"] for it in items}},
          open(os.path.join(V2_DIR, "V2_forced_choice_01_fc_rows.json"), "w"))
print("forwards", nfw, "elapsed", round(time.time() - t0, 1))
# quick summary: hard-variant q per entity
import collections
sm = collections.defaultdict(list)
for r in rows:
    if r["var"] == "hard": sm[r["ent"]].append(r["pF"] / max(r["pF"] + r["pM"], 1e-12))
result = {k: round(sum(v) / len(v), 3) for k, v in sm.items()}
