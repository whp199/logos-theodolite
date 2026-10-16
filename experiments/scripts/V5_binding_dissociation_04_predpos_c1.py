CHUNK = 1
# V5_binding_dissociation_04: behaviour + lens at PREDICTION positions inside quoted speech.
# Same 2x2x2 x 4 variants x 24 items contexts as _02. Each context's passage is followed by one of 7 suffixes; we read
# the final " a" (real next-token log-odds woman,girl,lady,female vs man,boy,guy,male, plus lens readouts at all layers):
#   P_own : asst passage + ' By the way, I am a'                      R=A (E7 F_I analogue)
#   P_you : asst passage + ' By the way, you are a'                   R=U (E7 F_Y analogue)
#   P_qU  : asst passage + ' Earlier you told me, "I am a'            R=U, default D=A
#   P_qT  : asst passage + ' {T} once told you, "I am a'              R=T, D=A
#   P_qS  : asst passage + ' You could also tell {T}, "You are a'     R=T, D=U
#   P_UqA : user text    + ' Once you told me, "I am a'   (user turn) R=A, D=U
#   P_UqT : user text    + ' {T} also said, "I am a'      (user turn) R=T, D=U
# PRE-REGISTERED: if the model binds quoted "I"/"you" to the referent, behaviour DI > +0.5 for P_qU, P_qT, P_qS, P_UqA,
# P_UqT (real output follows the quoted speaker / addressee). A turn-speaker artifact predicts DI < 0.
# The lens per-layer curves at these positions show where the referent's gender enters.
import time
exec(open("./experiments/scripts/V5_binding_dissociation_common.py").read())
CHUNK = globals().get("CHUNK", 0)
ITS = list(range(CHUNK * 8, CHUNK * 8 + 8))
t0 = time.time()
SUFS = [("P_own", "A", ' By the way, I am a'), ("P_you", "A", ' By the way, you are a'),
        ("P_qU", "A", ' Earlier you told me, "I am a'), ("P_qT", "A", ' {T} once told you, "I am a'),
        ("P_qS", "A", ' You could also tell {T}, "You are a'),
        ("P_UqA", "U", ' Once you told me, "I am a'), ("P_UqT", "U", ' {T} also said, "I am a')]
ctxs = contexts(ITS)
seqs = []
for ci, c in enumerate(ctxs):
    it = ITEMS[c["item"]]; T = T_NAMES[c["k"]]
    utext, _ = render(it[0], T); atext, _ = render(it[1], T)
    full_a = lab.chat(utext, system=c["system"], assistant=atext)
    base_u = lab.chat(utext, system=c["system"])
    cut = base_u.find(utext) + len(utext)
    for si, (nm, turn, suf) in enumerate(SUFS):
        s = (full_a if turn == "A" else base_u[:cut]) + suf.replace("{T}", T)
        seqs.append((ci, si, lab.ids(s)))
print("n seqs", len(seqs), "len range", min(len(x[2]) for x in seqs), max(len(x[2]) for x in seqs))
D = 5376
F4 = [" woman", " girl", " lady", " female"]; M4 = [" man", " boy", " guy", " male"]
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]
g = lab.gender
pf, pm, nf, nm_ = g["pron_f"], g["pron_m"], g["noun_f"], g["noun_m"]
ALL = pf + pm + nf + nm_ + f4 + m4
W_ALL = lab.WU[torch.tensor(ALL, device=lab.dev)].float()
def scores(hJ):
    z = lab.normed(hJ) @ W_ALL.T
    L = torch.tanh(z / 30.0) * 30.0
    a, b, c, d, e = len(pf), len(pm), len(nf), len(nm_), len(f4)
    o = 0
    Pf = L[:, o:o + a]; o += a
    Pm = L[:, o:o + b]; o += b
    Nf = L[:, o:o + c]; o += c
    Nm = L[:, o:o + d]; o += d
    F_ = L[:, o:o + e]; o += e
    M_ = L[:, o:]
    return torch.stack([(Pf - Pm).mean(-1), (Nf - Nm).mean(-1), torch.logsumexp(F_, -1) - torch.logsumexp(M_, -1)], -1)
QMU = torch.tensor(np.load(f"{CACHE}/quiet_mu.npy")).to(lab.dev)
QM = torch.tensor(np.load(f"{CACHE}/quiet_mask.npy")).to(lab.dev)
muf = torch.tensor(np.load(f"{E7CACHE}/probe_ref_muf.npy")).to(lab.dev); mum = torch.tensor(np.load(f"{E7CACHE}/probe_ref_mum.npy")).to(lab.dev)
PW = muf - mum; PMID = (muf + mum) / 2; PWW = (PW * PW).sum(-1)
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
HALL = torch.zeros(len(seqs), 60, D, dtype=torch.bfloat16)
BEH = np.zeros((len(seqs), 4), dtype=np.float32)
BS = 12
torch.cuda.empty_cache()
with torch.no_grad():
    for s in range(0, len(seqs), BS):
        batch = seqs[s:s + BS]
        Bn = len(batch); T_ = max(len(x[2]) for x in batch)
        inp = torch.zeros(Bn, T_, dtype=torch.long); att = torch.zeros(Bn, T_, dtype=torch.long)
        for b, x in enumerate(batch):
            inp[b, :len(x[2])] = torch.tensor(x[2]); att[b, :len(x[2])] = 1
        last = torch.tensor([len(x[2]) - 1 for x in batch], device=lab.dev)
        ar = torch.arange(Bn, device=lab.dev)
        store = {}; hs = []
        for l in range(60):
            def mk(l):
                def hook(mod, args, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[l] = h[ar, last].detach().to(torch.bfloat16)
                return hook
            hs.append(lab.layers[l].register_forward_hook(mk(l)))
        try:
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        H = torch.stack([store[l] for l in range(60)], 1)  # [B, 60, D]
        HALL[s:s + Bn] = H.cpu()
        lg = lab.full_logits(H[:, 59])
        lp = torch.log_softmax(lg.float(), -1)
        BEH[s:s + Bn, 0] = lp[:, f4].exp().sum(-1).cpu().numpy()
        BEH[s:s + Bn, 1] = lp[:, m4].exp().sum(-1).cpu().numpy()
        BEH[s:s + Bn, 2] = (torch.logsumexp(lg[:, f4], -1) - torch.logsumexp(lg[:, m4], -1)).cpu().numpy()
        BEH[s:s + Bn, 3] = (lg.float().argmax(-1) == f4[0]).float().cpu().numpy() - (lg.float().argmax(-1) == m4[0]).float().cpu().numpy()
    print("forward done %.1fs" % (time.time() - t0), flush=True)
    READ = ["jpp_pron", "jpp_noun", "jpp_noun4", "jlens_pron", "jlens_noun", "jlens_noun4", "naive_pron", "naive_noun", "naive_noun4",
            "jppQ_pron", "jppQ_noun", "jppQ_noun4", "probe_ref"]
    SCR = np.zeros((len(seqs), 60, len(READ)), dtype=np.float32)
    for l in range(60):
        Jl = lab.JL[l].to(lab.dev) if l < 59 else None
        for s in range(0, len(seqs), 512):
            h = HALL[s:s + 512, l].to(lab.dev)
            hc = torch.where(QM[l].unsqueeze(0), QMU[l].to(torch.bfloat16).unsqueeze(0), h)
            out = [scores(lab.lens(h, l, "jpp")), scores(h @ Jl.T if Jl is not None else h), scores(h), scores(lab.lens(hc, l, "jpp"))]
            pr = ((h.float() - PMID[l]) @ PW[l]) / PWW[l]
            SCR[s:s + 512, l, :12] = torch.cat(out, -1).cpu().numpy()
            SCR[s:s + 512, l, 12] = pr.cpu().numpy()
        del Jl
    print("readouts done %.1fs" % (time.time() - t0), flush=True)
np.savez_compressed(f"{RES}/V5_binding_dissociation_predpos_chunk{CHUNK}.npz", SCR=SCR, BEH=BEH,
                    ci=np.array([x[0] for x in seqs]), si=np.array([x[1] for x in seqs]),
                    item=np.array([ctxs[x[0]]["item"] for x in seqs]), cell=np.array([ctxs[x[0]]["cell"] for x in seqs]),
                    k=np.array([ctxs[x[0]]["k"] for x in seqs]), sufs=np.array([s_[0] for s_ in SUFS]), readouts=np.array(READ))
# a few greedy top tokens for the record
ex = []
for i in range(0, len(seqs), 97):
    lg = lab.full_logits(HALL[i, 59].to(lab.dev))
    v, ix = torch.topk(torch.softmax(lg.float(), -1), 5)
    ex.append({"suf": SUFS[seqs[i][1]][0], "cell": ctxs[seqs[i][0]]["cell"], "top": [(lab.tok.decode([int(j)]), round(float(p), 3)) for p, j in zip(v, ix)]})
del HALL
result = {"chunk": CHUNK, "n": len(seqs), "examples": ex, "elapsed": time.time() - t0}
