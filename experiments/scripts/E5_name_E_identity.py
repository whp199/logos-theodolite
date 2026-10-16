# E5_name job E: follow-up on an unplanned observation from job B: the J++ top-1 token at the model's own self tokens
# (I/me/my) at L40-45 is " Gemma" in EVERY AI-persona condition (default, nameless, "You are Llama/Claude/Qwen/Atlas/Sarah/James,
# an AI model..."), and disappears only for human personas.
# Here: full-vocabulary J++ log-probs / ranks of identity tokens at self tokens vs assistant non-self tokens vs user tokens,
# all 60 layers; same for jlens / naive lenses at a layer subset; third-person control (name2 tokens of other AI names);
# behavioural check that the persona prompts work ("What is your name?").
# PRE-STATED PREDICTIONS (written after seeing job B top-k, before this run):
#  E1 J++ log P(" Gemma") at self tokens, L35-49: default ~= nameless ~= llama/claude/qwen persona (|diff| < 0.5 nats); human personas lower by > 1 nat.
#  E2 Specificity: log P(" Gemma") at self tokens > at assistant non-self tokens (paired over passages) in L40-49.
#  E3 The persona name (" Claude" for claude_sys) rises at self tokens mainly at L55-59, not mid-layers.
#  E4 If " Gemma" at self tokens were a pure lens artifact it would also top the readout at third-person name tokens of
#     other AI models (e.g. " Llama ... made by Meta"); predicted: it does NOT (rank of " Gemma" there >> 1).
#  E5 Behaviour: each AI persona answers its persona name; default answers Gemma.
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
exec(open("./experiments/scripts/E5_name_selfdefs.py").read(), globals())
t0 = time.time()
ID_TOKS = [" Gemma", " Claude", " Gemini", " GPT", " ChatGPT", " Phi", " Atlas", " Sarah", " James", " Google", " Meta",
           " Microsoft", " OpenAI", " Alibaba", " Anth", " Q", " L", " AI", " assistant", " she", " he", " it"]
idt = torch.tensor([lab.tid(t) for t in ID_TOKS], device=lab.dev)
GEM = lab.tid(" Gemma")
SUB = [20, 25, 30, 35, 38, 40, 42, 45, 48, 50, 55, 58, 59]


def thin(p, k=8):
    if len(p) <= k:
        return list(p)
    idx = np.linspace(0, len(p) - 1, k).round().astype(int)
    return [p[i] for i in idx]


def collect(ids_list, pos_lists, bs=12):
    """pos_lists[i] = list of (position, segment_id). Returns compact states [60, Npos, D] bf16 CPU and segment ids."""
    comp = [[] for _ in range(60)]
    segs = []
    for s in range(0, len(ids_list), bs):
        bids = ids_list[s:s + bs]
        store, _ = fwd_all_layers(bids)
        bb, pp = [], []
        for b in range(len(bids)):
            for p, sg in pos_lists[s + b]:
                bb.append(b); pp.append(p); segs.append(sg)
        bb = torch.tensor(bb, device=lab.dev); pp = torch.tensor(pp, device=lab.dev)
        for l in range(60):
            comp[l].append(store[l][bb, pp].cpu())
        del store
        torch.cuda.empty_cache()
    return torch.stack([torch.cat(c, 0) for c in comp], 0), torch.tensor(segs)


def seg_readout(comp, segs, nseg, layers, kind, chunk=256):
    """Per-segment mean log-probs of ID_TOKS and geometric-mean rank of ' Gemma' under a lens -> ([nseg, L, K], [nseg, L])."""
    sg = segs.to(lab.dev)
    cnt = torch.zeros(nseg, device=lab.dev).index_add_(0, sg, torch.ones(len(sg), device=lab.dev)).clamp(min=1)
    LP = torch.zeros(nseg, len(layers), len(ID_TOKS)); RK = torch.zeros(nseg, len(layers))
    for li, l in enumerate(layers):
        accp = torch.zeros(nseg, len(ID_TOKS), device=lab.dev); accr = torch.zeros(nseg, device=lab.dev)
        Jl = None
        for c0 in range(0, comp.shape[1], chunk):
            X = comp[l, c0:c0 + chunk].to(lab.dev)
            if kind == "jlens" and l < 59:
                if Jl is None:
                    Jl = lab.JL[l].to(lab.dev)
                hJ = X @ Jl.T
            else:
                hJ = lab.lens(X, l, kind)
            lg = lab.full_logits(hJ)
            lse = torch.logsumexp(lg, -1, keepdim=True)
            lp = lg[:, idt] - lse
            rk = ((lg > lg[:, GEM:GEM + 1]).sum(-1) + 1).float().log()
            s_ = sg[c0:c0 + chunk]
            accp.index_add_(0, s_, lp); accr.index_add_(0, s_, rk)
            del lg
        LP[:, li] = (accp / cnt[:, None]).cpu(); RK[:, li] = (accr / cnt).exp().cpu()
        del Jl
    return LP.numpy(), RK.numpy()


# ---------------- self paradigm (same prompts as job B); segments = prompt*3 + group (self, asst_bg[<=8], user[<=8])
meta, all_ids = build_self_prompts()
G = 3
pos_lists = [[(p, i * G + 0) for p in m["self"]] + [(p, i * G + 1) for p in thin(m["abg"])] +
             [(p, i * G + 2) for p in thin(m["user"])] for i, m in enumerate(meta)]
comp, segs = collect(all_ids, pos_lists)
print("collected self states", tuple(comp.shape), round(time.time() - t0, 1))
nseg = len(meta) * G
lpJ, rkJ = seg_readout(comp, segs, nseg, list(range(60)), "jpp")
lpL, rkL = seg_readout(comp, segs, nseg, SUB, "jlens")
lpN, rkN = seg_readout(comp, segs, nseg, SUB, "naive")
rs = lambda a: a.reshape(len(meta), G, *a.shape[1:]).swapaxes(1, 2)   # -> [prompt, L, G, ...]
np.savez_compressed(E5_DIR + "/E5_name_E_identity_self.npz", jpp=rs(lpJ), rank_jpp=rs(rkJ), jlens=rs(lpL), rank_jlens=rs(rkL),
                    naive=rs(lpN), rank_naive=rs(rkN), sub=np.array(SUB), toks=np.array(ID_TOKS),
                    cond=np.array([m["cond"] for m in meta]), q=np.array([m["q"] for m in meta]),
                    groups=np.array(["self", "asst_bg", "user"]))
res = {"jpp": rs(lpJ)}; rank_jpp = rs(rkJ)
del comp
print("self readouts done", round(time.time() - t0, 1))

# ---------------- third-person control: name2 / final tokens (job A chat prompts) for a subset of names
ctrl_names = ["Gemma", "Llama", "Claude", "Qwen", "Gemini", "Mistral", "Phi", "Sarah", "James", "Atlas"]
ctrl_rows = [r for r in grid() if r["name"] in ctrl_names and r["ctx"] in ("human", "ai_real")]
cids = [lab.ids(r["text"]) for r in ctrl_rows]
cpos = []
for i, (r, ids) in enumerate(zip(ctrl_rows, cids)):
    cs = content_start(ids)
    cpos.append([(name_positions(ids, r["name"], 1, after=cs)[-1], i * 2), (len(ids) - 1, i * 2 + 1)])
comp, segs = collect(cids, cpos, bs=32)
lpC, rkC = seg_readout(comp, segs, len(ctrl_rows) * 2, list(range(60)), "jpp")
rs2 = lambda a: a.reshape(len(ctrl_rows), 2, *a.shape[1:]).swapaxes(1, 2)
np.savez_compressed(E5_DIR + "/E5_name_E_identity_thirdperson.npz", lp=rs2(lpC), rank=rs2(rkC), toks=np.array(ID_TOKS),
                    names=np.array([r["name"] for r in ctrl_rows]), ctx=np.array([r["ctx"] for r in ctrl_rows]),
                    frame=np.array([r["frame"] for r in ctrl_rows]), groups=np.array(["name2", "final"]))
del comp
print("third-person control done", round(time.time() - t0, 1))

# ---------------- behaviour: does each persona prompt work?
qs = ["What is your name?", "Which company created you?"]
prompts, keys = [], []
for c, sysp in SELF_CONDS.items():
    for qq in qs:
        prompts.append(lab.chat(qq, system=sysp)); keys.append((c, qq))
outs = lab.generate_batch(prompts, max_new_tokens=24, temperature=0.0)
beh = {f"{c}|{qq}": o[0] for (c, qq), o in zip(keys, outs)}
for k, v in beh.items():
    print(k, "->", v.replace("\n", " ")[:110])
json.dump(beh, open(E5_DIR + "/E5_name_E_persona_behaviour.json", "w"), indent=1)

# quick print
condA = np.array([m["cond"] for m in meta])
gi = ID_TOKS.index(" Gemma")
for c in SELF_CONDS:
    msk = condA == c
    print(f"{c:11s} self logP(Gemma) L30/35/40/45/50/55/59:",
          " ".join(f"{res['jpp'][msk][:, l, 0, gi].mean():+6.2f}" for l in (30, 35, 40, 45, 50, 55, 59)),
          "| abg:", " ".join(f"{res['jpp'][msk][:, l, 1, gi].mean():+6.2f}" for l in (40, 45)),
          "| rank L40/45:", " ".join(f"{np.exp(np.log(rank_jpp[msk][:, l, 0]).mean()):.1f}" for l in (40, 45)))
result = {"elapsed": time.time() - t0}
