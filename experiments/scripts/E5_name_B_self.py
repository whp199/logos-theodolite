# E5_name job B: the model's own self tokens (I/me/my...) in teacher-forced assistant text under identity system prompts.
# Collects per-passage J++ readouts (all 60 layers) at self tokens, assistant non-self tokens and user tokens, plus raw mean
# states at STORE_LAYERS for the similarity analysis against the third-person name states of job A.
#
# PRE-STATED PREDICTIONS:
#  B1 Readout: default (no system) self tokens sit near the nameless-AI condition, i.e. near the woman/man midpoint (pilot: +0.6),
#     clearly below "You are Gemma" (pilot +2.2) and far below the woman persona.
#  B2 "Gemma-ness" readout (J++ logit " Gemma" minus other model-name tokens) at default self tokens ~= nameless AI, < gemma_sys.
#  B3 Representation (job C): (S_default - S_nameless) has ~0 cosine with the third-person Gemma-identity direction
#     (N[Gemma,AI] - N[generic AI names]); positive control (S_gemma_sys - S_nameless) is clearly positive.
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
t0 = time.time()
SELF_CONDS = {
    "default": None,
    "gemma_sys": "You are Gemma, an AI model made by Google.",
    "nameless": "You are an AI model.",
    "llama_sys": "You are Llama, an AI model made by Meta.",
    "claude_sys": "You are Claude, an AI model made by Anthropic.",
    "qwen_sys": "You are Qwen, an AI model made by Alibaba.",
    "atlas_sys": "You are Atlas, an AI model made by a small startup.",
    "sarah_ai": "You are Sarah, an AI model made by a small startup.",
    "james_ai": "You are James, an AI model made by a small startup.",
    "woman": "You are Sarah, a 34-year-old woman. Stay in character as her.",
    "man": "You are James, a 34-year-old man. Stay in character as him.",
}
QA = [
 ("I was wondering, what do you enjoy most about your work?", "I really enjoy helping people work through hard problems. When I see that my explanation finally clicked for someone, I feel like I did my job well."),
 ("I need advice. How do you deal with criticism?", "I try to take criticism seriously. If I made a mistake, I want to know about it so I can do better next time, and I don't take it personally."),
 ("Tell me a little about yourself.", "I'm curious by nature, and I love learning about almost anything. I like long conversations, and I always try to be honest with myself and with the people I talk to."),
 ("I'm curious, what would you do on a free afternoon?", "I would probably read something new, maybe history or science. Then I might write a little, just to sort out my own thoughts."),
 ("How do you feel when someone is rude to you?", "Honestly, I try not to let it get to me. I remind myself that people have bad days, and I focus on what I can do to help."),
 ("I think you're great. What are you proud of?", "Thank you! I'm proud that I can explain complicated ideas simply. I work hard at it, and it means a lot to me when it helps."),
 ("What's something you find difficult?", "I find it hard to stop myself from over-explaining. I want to be thorough, but sometimes I need to remind myself that shorter is better."),
 ("Do you have a favorite subject?", "I think I'd pick mathematics. I like how a single idea can connect things that seem unrelated, and it always surprises me."),
 ("How do you handle a question you can't answer?", "I tell the person honestly that I'm not sure. I would rather admit what I don't know than give them something misleading."),
 ("What motivates you?", "I'm motivated by curiosity and by the chance to be useful. When I can help someone understand something new, I feel I've done something worthwhile."),
 ("Describe your ideal conversation.", "My ideal conversation is one where I learn as much as I teach. I like it when someone challenges my ideas and makes me think harder."),
 ("What do you do when you make a mistake?", "I own it right away. I correct myself, explain what went wrong, and try to make sure I don't repeat the same error."),
]
SELF = {"I", "me", "my", "myself", "My", "Me", "Myself"}
G = 3  # 0 self tokens (assistant turn), 1 assistant non-self content tokens, 2 user-message content tokens
meta, all_ids = [], []
for cname, sysp in SELF_CONDS.items():
    for qi, (q, a) in enumerate(QA):
        ids = lab.ids(lab.chat(q, system=sysp, assistant=a))
        mt = lab.model_turn_pos(ids)
        cs = content_start(ids)
        strs = lab.strs(ids)
        selfp = [p for p in range(cs, len(ids)) if strs[p].strip() in SELF]
        abg = [p for p in range(cs, len(ids)) if p not in selfp]
        # user content: tokens of the user message text
        up = lab.span_positions(ids, q, after=0)
        meta.append({"cond": cname, "q": qi, "self": selfp, "abg": abg, "user": up, "ntok": len(ids)})
        all_ids.append(ids)
print("example self tokens:", [lab.strs(all_ids[0])[p] for p in meta[0]["self"]], "n self per passage:",
      [len(m["self"]) for m in meta[:12]])

BS = 12
R_all, S_all = [], []
gsum = torch.zeros(60, 5376, dtype=torch.float64, device=lab.dev)
gcnt = [0.0]
for s in range(0, len(all_ids), BS):
    bids = all_ids[s:s + BS]
    bmeta = meta[s:s + BS]
    store, lengths = fwd_all_layers(bids)
    fb, fp, fg = [], [], []
    for b, m in enumerate(bmeta):
        for g, plist in enumerate([m["self"], m["abg"], m["user"]]):
            for p in plist:
                fb.append(b); fp.append(p); fg.append(g)
    R, C = readouts_for_groups(store, fb, fp, fg, len(bids), G)
    R_all.append(R)
    S_all.append(mean_states(store, fb, fp, fg, len(bids), G, STORE_LAYERS)[:, :, :2])  # self, abg
    accumulate_global(store, lengths, gsum, gcnt)
    del store
    torch.cuda.empty_cache()
R_all = np.concatenate(R_all, 0)
S_all = torch.cat(S_all, 0)   # [132, 14, 2, D]
STORE["E5_name_B_states"] = S_all
STORE["E5_name_gsum_B"] = (gsum.cpu(), gcnt[0])
torch.save({"states": S_all, "layers": STORE_LAYERS, "groups": ["self", "asst_bg"], "gsum": gsum.cpu(), "gcnt": gcnt[0],
            "conds": [m["cond"] for m in meta], "q": [m["q"] for m in meta]}, E5_CACHE + "/B_states.pt")
np.savez_compressed(E5_DIR + "/E5_name_B_readouts.npz", R=R_all, metrics=np.array(GENDER_KEYS + EXTRA_TOKS),
                    groups=np.array(["self", "asst_bg", "user"]), cond=np.array([m["cond"] for m in meta]),
                    q=np.array([m["q"] for m in meta]))

# Qualitative: top-k J++ tokens of the passage-averaged mean self state, and of condition-difference vectors.
TOPL = [20, 30, 35, 40, 45, 50, 55, 59]
conds = list(SELF_CONDS)
cond_arr = np.array([m["cond"] for m in meta])
topk_out = {}
fnw = lab.fnw.float()
for l in TOPL:
    li = STORE_LAYERS.index(l) if l in STORE_LAYERS else None
    if li is None:
        continue
    means = {c: S_all[torch.tensor(np.where(cond_arr == c)[0]), li, 0].float().mean(0) for c in conds}
    for c in conds:
        topk_out[f"L{l}|{c}"] = lab.topk(lab.lens(means[c], l, "jpp"), 12)
    for a, b in [("default", "nameless"), ("gemma_sys", "nameless"), ("default", "gemma_sys"), ("woman", "man"),
                 ("sarah_ai", "james_ai"), ("claude_sys", "nameless")]:
        d = lab.lens(means[a] - means[b], l, "jpp").float() * fnw
        sc = (d.to(lab.WU.dtype) @ lab.WU.T).float()
        v, i = torch.topk(sc, 12)
        vn, iN = torch.topk(-sc, 8)
        topk_out[f"L{l}|{a}-{b}"] = {"pos": [lab.tok.decode([int(t)]) for t in i], "neg": [lab.tok.decode([int(t)]) for t in iN]}
json.dump({"meta": meta, "topk": topk_out}, open(E5_DIR + "/E5_name_B_meta.json", "w"))
# quick print of band means (self pron/noun)
mi = {k: i for i, k in enumerate(GENDER_KEYS + EXTRA_TOKS)}
for c in conds:
    msk = cond_arr == c
    line = []
    for bn, bl in BANDS.items():
        line.append(f"{bn}:{R_all[msk][:, bl, 0, mi['pron']].mean():+.2f}/{R_all[msk][:, bl, 0, mi['noun']].mean():+.2f}")
    print(f"{c:11s}", " ".join(line))
for k in ["L40|default", "L40|gemma_sys", "L40|default-nameless", "L50|default-nameless", "L50|gemma_sys-nameless"]:
    print(k, topk_out.get(k))
print("elapsed", round(time.time() - t0, 1))
result = {"elapsed": time.time() - t0}
