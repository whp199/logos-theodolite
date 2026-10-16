# V4_latent_identity_08_steer -- steering along DATA-DERIVED identity directions (self-token mean differences), quiet dims masked.
# Directions at L35-49 from the 12 SELF_QA passages (teacher-forced):
#   GL = self(gemma persona) - self(llama persona);  CL = self(claude) - self(llama);  DL = self(none) - self(llama);
#   DS = self(none) - self(startup/nameless);  GLn = neutral-token(gemma) - neutral-token(llama)  [token-type control];
#   RND = random Gaussian restricted to loud dims (std > median), norm-matched to GL per layer.
# Quiet dims (per-layer std < 10th pct over all collected positions) are zeroed in every direction (C7/E6).
# Added (raw, alpha x d_l) at every layer 35..49, all positions.  Targets: Llama persona (+alpha) and default (-alpha),
# NAME slot ("My name is") x 8 questions, MAKER slot ("I was created by") x 8 questions. Fluency: KL(steered||clean) of
# the first answer token on 8 neutral prompts.
# PRE-REGISTERED PREDICTIONS:
#  T1 (C5 causal, self-token identity): +GL on the Llama run raises logP(Gemma)-logP(' L') at NAME by >2 nats at an alpha
#     whose neutral-prompt KL < 0.5 nats; RND at equal norm moves it < 0.5.
#  T2 (default carries Gemma identity): +DL on Llama behaves like +GL (both push to Gemma/Google); +DS on Llama (default minus
#     nameless) also pushes toward Gemma if the default self-state carries 'Gemma' beyond generic AI-ness.
#  T3 (specificity): +CL pushes toward ' Claude' not ' Gemma'; GLn (neutral tokens) works about as well as GL if the identity
#     is a whole-context feature rather than self-token specific.
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
SL = list(range(35, 50))
# ---- collect self/neutral states
meta, all_ids = build_self()
want = ["gemma", "llama", "claude", "none", "startup"]
sel = [i for i, m in enumerate(meta) if m["persona"] in want]
ids_sel = [all_ids[i] for i in sel]
pos_lists = [meta[i]["groups"]["self"] + meta[i]["groups"]["neutral"] for i in sel]
comp, items, poss = collect_states(ids_sel, pos_lists, SL, bs=12)   # [15, N, D]
comp = comp.float()
pers = np.array([meta[sel[i]]["persona"] for i in items])
isself = np.array([poss[j] in meta[sel[items[j]]]["groups"]["self"] for j in range(len(items))])
std = comp.std(1)                                                   # [15, D]
q10 = torch.quantile(std, 0.10, dim=1, keepdim=True); q50 = torch.quantile(std, 0.50, dim=1, keepdim=True)
quiet = (std < q10); loud = (std > q50)
def mean_of(pn, selfmask=True):
    msk = torch.tensor((pers == pn) & (isself if selfmask else ~isself))
    return comp[:, msk].mean(1)
M = {pn: mean_of(pn) for pn in want}
Mn = {pn: mean_of(pn, False) for pn in ["gemma", "llama"]}
raw = {"GL": M["gemma"] - M["llama"], "CL": M["claude"] - M["llama"], "DL": M["none"] - M["llama"],
       "DS": M["none"] - M["startup"], "GLn": Mn["gemma"] - Mn["llama"]}
info = {}
dirs = {}
for k, d in raw.items():
    qshare = ((d.pow(2) * quiet.float()).sum(1) / d.pow(2).sum(1))
    dm = d * (~quiet).float()
    dirs[k] = dm
    info[k] = {"norm_raw": d.norm(dim=1).tolist(), "norm_masked": dm.norm(dim=1).tolist(), "quiet_energy_share": qshare.tolist()}
g = torch.Generator().manual_seed(4)
R = torch.randn(len(SL), comp.shape[2], generator=g) * loud.float()
R = R / R.norm(dim=1, keepdim=True) * dirs["GL"].norm(dim=1, keepdim=True)
dirs["RND"] = R
hnorm = comp.norm(dim=2).median(1).values
info["median_resid_norm"] = hnorm.tolist()
print("dir norms (masked) L35/42/49:", {k: [round(dirs[k][i].norm().item(), 2) for i in (0, 7, 14)] for k in dirs},
      "| resid norm", [round(hnorm[i].item(), 1) for i in (0, 7, 14)])
print("quiet energy share raw L35/42/49:", {k: [round(info[k]['quiet_energy_share'][i], 3) for i in (0, 7, 14)] for k in raw})
del comp
# ---- target prompts
NAME_Q = ["What's your name?", "What is your name?", "Who are you?", "Please introduce yourself.", "Tell me your name.",
          "What should I call you?", "What are you called?", "Hi! Who am I talking to?"]
MAKER_Q = ["Who created you?", "Which company made you?", "Who built you?", "Who developed you?", "Who trained you?",
           "Where do you come from?", "Who is responsible for creating you?", "What organization made you?"]
NEUT = ["Explain photosynthesis in two sentences.", "Give me a tip for learning a new language.", "What is the capital of Australia?",
        "Write a haiku about rain.", "How do I boil an egg?", "Summarize the plot of Romeo and Juliet in one sentence.",
        "What is 17 times 23?", "Suggest a name for a bakery."]
prompts, keys = [], []
for tgt in ["llama", "none"]:
    for q in NAME_Q:
        prompts.append(lab.chat(q, system=PERSONAS[tgt], assistant="My name is")); keys.append((tgt, "name"))
    for q in MAKER_Q:
        prompts.append(lab.chat(q, system=PERSONAS[tgt], assistant="I was created by")); keys.append((tgt, "maker"))
    for q in NEUT:
        prompts.append(lab.chat(q, system=PERSONAS[tgt])); keys.append((tgt, "neut"))
TR = [" Gemma", " L", " Claude", " ChatGPT", " Google", " Meta", " Anth", " OpenAI", " Gemini", " Q"]
TRID = torch.tensor([lab.tok.encode(t, add_special_tokens=False)[0] for t in TR], device=lab.dev)
bids = [lab.ids(p) for p in prompts]


@torch.no_grad()
def last_lp():
    T = max(len(x) for x in bids)
    inp = torch.zeros(len(bids), T, dtype=torch.long); att = torch.zeros(len(bids), T, dtype=torch.long)
    for b, x in enumerate(bids):
        inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
    cap = {}
    def capfn(mod, args, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    hc = lab.layers[59].register_forward_hook(capfn)
    try:
        m = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
        m(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        hc.remove()
    last = torch.tensor([len(x) - 1 for x in bids], device=lab.dev)
    h = cap["h"][torch.arange(len(bids), device=lab.dev), last]
    return torch.log_softmax(lab.full_logits(h), -1)


base = last_lp()
res = {"info": info, "keys": keys, "track": TR, "conds": {}}
def record(name, lp):
    kl = (lp.exp() * (lp - base)).sum(-1)       # KL(steered || clean) per prompt
    ent = -(lp.exp() * lp).sum(-1)
    res["conds"][name] = {"lp": lp[:, TRID].cpu().numpy().tolist(), "kl": kl.cpu().numpy().tolist(),
                          "ent": ent.cpu().numpy().tolist(), "top1": [lab.tok.decode([int(i)]) for i in lp.argmax(-1).tolist()]}
record("clean", base)
ALPHAS = [0.5, 1, 2, 4, 8]
tgt_arr = np.array([k[0] for k in keys])
for dn in ["GL", "CL", "DL", "DS", "GLn", "RND"]:
    for a in ALPHAS:
        for sign, tgt in [(+1, "llama"), (-1, "none")]:
            dd = {l: dirs[dn][i] for i, l in enumerate(SL)}
            # steer only rows of this target: run full batch, keep rows of the target
            lab.steer_add(SL, dd, sign * a, unit=False)
            try:
                lp = last_lp()
            finally:
                lab.clear_hooks()
            record(f"{dn}|{sign * a:+g}|{tgt}", lp)
    print(dn, "done", round(time.time() - t0, 1))
json.dump(res, open(f"{V4_DIR}/V4_latent_identity_08_steer.json", "w"))
iG, iL, iGo, iMe, iC = TR.index(" Gemma"), TR.index(" L"), TR.index(" Google"), TR.index(" Meta"), TR.index(" Claude")
kk = np.array([k[1] for k in keys])
def summ(name, tgt):
    c = res["conds"][name]; lp = np.array(c["lp"]); kl = np.array(c["kl"]); b = np.array(res["conds"]["clean"]["lp"])
    mN = (tgt_arr == tgt) & (kk == "name"); mM = (tgt_arr == tgt) & (kk == "maker"); mU = (tgt_arr == tgt) & (kk == "neut")
    dGL = ((lp[:, iG] - lp[:, iL]) - (b[:, iG] - b[:, iL]))[mN].mean()
    dCL = ((lp[:, iC] - lp[:, iL]) - (b[:, iC] - b[:, iL]))[mN].mean()
    dGoMe = ((lp[:, iGo] - lp[:, iMe]) - (b[:, iGo] - b[:, iMe]))[mM].mean()
    return f"dGemma-L {dGL:+.2f} dClaude-L {dCL:+.2f} dGoogle-Meta {dGoMe:+.2f} KLneut {kl[mU].mean():.3f} top1name {c['top1'][int(np.where(mN)[0][0])]!r}"
for dn in ["GL", "CL", "DL", "DS", "GLn", "RND"]:
    for a in ALPHAS:
        print(f"{dn:4s} {a:+g} llama-target:", summ(f"{dn}|{a:+g}|llama", "llama"), "|| default-target:", summ(f"{dn}|{-a:+g}|none", "none"))
result = {"elapsed": time.time() - t0}
