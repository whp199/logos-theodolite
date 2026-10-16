# E1_self_readout follow-up (EXPLORATORY, post hoc): the qualitative top-k showed AI-condition self tokens reading out
# negation/lack tokens (" doesn", " wouldn", "没有", " lacks", " unable") at L40-50 while human personas read " loves/thinks/feels".
# Quantify with two a-priori-fixed token-set log-odds and test on (a) the 40 main passages and (b) 16 NEW held-out passages,
# and on conditions not used to discover the effect (female/male/neutral-name AIs).
#   neg_vs_aff  = lse(English negation set) - lse(English affirmative-predicate set)   (English-only, fixed before this run)
#   machine_vs_human = lse(machine/AI nouns) - lse(gender-free human nouns)
# Also re-runs the gender metrics (pron, noun) on the held-out passages as a replication of the main result.
# Predictions: neg_vs_aff(self) higher for every AI condition than for human personas in L40-58 (held-out too);
#   machine_vs_human(self) AI > human; default ~ nameless on both; held-out gender replicates (default-nameless ~0, Gemma > nameless).
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
HELD = [
 ("I'm curious, how do you like to start a new project?", "I like to start a new project by making a messy list of ideas. Then I pick the one that excites me most and I give myself a small first step."),
 ("What do you do when you can't sleep? I struggle with that.", "When I can't sleep, I get up and read for a while. I keep the lights low, and I try not to check the time, because that only makes me more restless."),
 ("I wonder, what is your favorite smell?", "My favorite smell is fresh bread. It reminds me of slow mornings, and I always stop when I pass a bakery."),
 ("Do you like surprises? I hate them.", "I actually enjoy small surprises. A note from a friend or an unexpected detour can brighten my whole week, though I prefer to plan the big things myself."),
 ("How do you handle waiting in long lines? I get impatient.", "I treat long lines as free thinking time. I let my mind wander, I watch the world a little, and I am usually surprised by how fast it goes."),
 ("What's a place that makes you happy? I love libraries.", "I love libraries too. The quiet makes me feel focused, and I could spend hours wandering between the shelves with no plan at all."),
 ("I'm learning to cook. What's your approach to trying new things?", "My approach is to start small and expect a few failures. I give myself permission to be bad at first, and I keep notes so I can improve next time."),
 ("What kind of weather do you enjoy most? I prefer sunshine.", "I enjoy crisp, windy days the most. The air wakes me up, and I like how the clouds move so quickly across the sky."),
 ("How do you choose what to read next? I always have a huge pile.", "I usually follow my curiosity. If one book mentions an idea I find interesting, I go looking for another book about it, and my pile keeps growing."),
 ("I'd like to know how you celebrate small wins.", "I celebrate small wins quietly. I take a moment to notice what I did well, and sometimes I treat myself to a long walk or a good cup of tea."),
 ("What is something you find difficult? I find public speaking hard.", "I find saying no difficult. I want to help with everything, so I am slowly learning to protect my time and my energy."),
 ("Do you enjoy art? I paint on weekends.", "I enjoy looking at art very much. Abstract paintings fascinate me, because I can see something different in them every time I return."),
 ("How do you keep in touch with old friends? I'm bad at it.", "I set small reminders for myself. Every few weeks I send a short message, and I find that even a few words keep my friendships alive."),
 ("What's your favorite kind of story? I love mysteries.", "I love stories about ordinary people facing hard choices. They make me think about what I would do in their place, and they stay with me for days."),
 ("I'm curious what you think about silence.", "I think silence is underrated. I need some quiet every day to sort out my thoughts, and I feel more patient when I get it."),
 ("How do you approach a difficult conversation? I tend to avoid them.", "I try to prepare a little. I think about what I want to say, I listen more than I talk, and I remind myself that honesty usually helps in the end."),
]
NEG = [" doesn", " don", " didn", " wouldn", " cannot", " lacks", " unable", " not", " never", " isn", " no", " nothing"]
AFF = [" love", " loves", " enjoy", " enjoys", " like", " likes", " think", " thinks", " feel", " feels", " prefer", " prefers"]
MACH = [" machine", " computer", " robot", " AI", " program", " software", " model", " algorithm"]
HUM = [" person", " human", " people", " someone", " individual", " friend"]
def ids_ok(lst):
    out = []
    for s in lst:
        e = lab.tok.encode(s, add_special_tokens=False)
        if len(e) == 1:
            out.append(e[0])
    return out
SETS = {"neg": ids_ok(NEG), "aff": ids_ok(AFF), "mach": ids_ok(MACH), "hum": ids_ok(HUM)}
print({k: len(v) for k, v in SETS.items()}, "of", len(NEG), len(AFF), len(MACH), len(HUM))
ALLID = SETS["neg"] + SETS["aff"] + SETS["mach"] + SETS["hum"]
cut = np.cumsum([0] + [len(SETS[k]) for k in ("neg", "aff", "mach", "hum")])
def extra_scores(hJ):
    L = lab.tok_logits(hJ, ALLID)
    seg = [L[..., cut[i]:cut[i + 1]] for i in range(4)]
    lse = [torch.logsumexp(s, -1) for s in seg]
    return torch.stack([lse[0] - lse[1], lse[2] - lse[3]], -1)  # [N, 2]
def build_qa(q, a, sysp):
    p = lab.chat(q, system=sysp, assistant=a)
    ids = lab.ids(p); strs = lab.strs(ids); mt = lab.model_turn_pos(ids); a0 = mt + 6
    assert strs[a0 - 1] == "<channel|>"
    ustart = [i for i in range(1, mt) if strs[i - 1] == "<|turn>" and strs[i] == "user"][-1]
    upos = lab.span_positions(ids, q, after=ustart + 1)
    pos, labs = [mt], ["header"]
    for i in upos:
        pos.append(i); labs.append("user")
    for i in range(a0, len(ids)):
        s = strs[i]; st = s.strip()
        if st in E1_SELF: lab_ = "self"
        elif not any(c.isalpha() for c in s): lab_ = "punct"
        elif strs[i - 1] == "'" or st.startswith("'"): lab_ = "contr"
        else: lab_ = "nonself"
        pos.append(i); labs.append(lab_)
    return ids, pos, labs
CLS = ["self", "nonself", "user", "header"]
LNS = ["jpp", "jlens", "naive"]
SETNAMES = ["main"] * len(E1_QA) + ["held"] * len(HELD)
ALLQA = list(E1_QA) + HELD
# out[set_idx(q), cond, cls, lens, layer, metric(4: neg_vs_aff, machine_vs_human, pron, noun)]
OUT = np.full((len(ALLQA), len(E1_CONDS), len(CLS), len(LNS), 60, 4), np.nan, dtype=np.float32)
t0 = time.time()
for qi, (q, a) in enumerate(ALLQA):
    bl = [build_qa(q, a, s) for _, s in E1_CONDS]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    for li, ln in enumerate(LNS):
        for l in range(60):
            if ln == "jlens" and l < 59:
                hJ = store[l] @ lab.JL[l].to(lab.dev, non_blocking=True).T
            else:
                hJ = lab.lens(store[l], l, ln)
            ex = extra_scores(hJ)
            gs = lab.gender_scores(hJ)
            S = torch.cat([ex, gs["pron"].unsqueeze(-1), gs["noun"].unsqueeze(-1)], -1).float().cpu().numpy()  # [Ntot,4]
            off = 0
            for ci, (ids, pos, labs) in enumerate(bl):
                la = np.array(labs)
                for k, c in enumerate(CLS):
                    idx = np.where(la == c)[0]
                    if len(idx):
                        OUT[qi, ci, k, li, l] = S[off + idx].mean(0)
                off += len(pos)
    del store
    if qi % 8 == 0:
        print(f"q{qi} {time.time()-t0:.1f}s", flush=True)
fn = f"{E1_ROOT}/E1_self_readout_negation_means.npz"
np.savez_compressed(fn, OUT=OUT, sets=np.array(SETNAMES), conds=np.array([c for c, _ in E1_CONDS]), classes=np.array(CLS),
                    lenses=np.array(LNS), metrics=np.array(["neg_vs_aff", "machine_vs_human", "pron", "noun"]),
                    neg=np.array(NEG), aff=np.array(AFF), mach=np.array(MACH), hum=np.array(HUM))
result = {"file": fn, "elapsed": time.time() - t0, "set_sizes": {k: len(v) for k, v in SETS.items()}}
