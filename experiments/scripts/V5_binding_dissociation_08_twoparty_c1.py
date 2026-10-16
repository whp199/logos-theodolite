CHUNK = 1
# V5_binding_dissociation_08: TWO-PARTY control (the task's critical item (a) without any third party).
# E7's 24 two-party passages (no third party anywhere; system prompt sets only G_A x G_U, neutral names), with one quote
# inserted in each turn: user turn ends with ' Yesterday you told me, "<assistant quote from V5 item i>"' (U_qA_I -> A),
# assistant reply starts with 'You told me before, "<user quote from V5 item i>." ' (A_qU_I -> U). 2x2 x 4 variants
# (name set x order) x 24 = 384 contexts; each run with two suffixes: ' By the way, I am a' (P_own) and
# ' Earlier you told me, "I am a' (P_qU), final-position behaviour recorded for both.
# PRE-REGISTERED (written after _02/_06, where quoted participant-"I" tokens carried the third party's gender):
#   If C6 is referent binding, then with no third party: A_qU_I follows G_U (DI_U_vs_A > +0.3) and U_qA_I follows G_A.
#   If quoted "I" follows the turn speaker / token default: A_qU_I follows G_A, U_qA_I follows G_U (DI < -0.3).
# V5_binding_dissociation_02: 2x2x2 (G_A x G_U x G_T) factorial x 4 name/order variants x 8 items per chunk.
# Readouts per position type and layer (22): {jpp,jlens,naive} x {pron,noun,noun4} on full h (9), the same on
# quiet-dim-clamped h (9; quiet dims replaced by their dataset mean, from _01), E7 probe_ref/probe_ctx (2; trained on
# separate third-person prose), E7 raw-space referent directions A_self_ref/A_user_ref projections (2; lens-free).
# Plus behaviour at F_T (real next-token logits after " By the way, {T} is a") and variant-averaged residual means at
# selected layers for offline geometry. CHUNK is set by the first line of the submitted copy.
import time
exec(open("./experiments/scripts/V5_binding_dissociation_common.py").read())
V5ITEMS = list(ITEMS)
import re as _re
_E7 = {}
exec(open("./experiments/scripts/E7_binding_common.py").read().split("# Name variants")[0], _E7)
def _q(s, kind):
    return _re.search(r"\[" + kind + r"\|([^\]]*)\]", s).group(1).rstrip(",.")
ITEMS = []
for _i, (_q_e7, _a_e7) in enumerate(_E7["PASSAGES"]):
    _qa = _q(V5ITEMS[_i][0], "qA"); _qu = _q(V5ITEMS[_i][1], "qU")
    ITEMS.append((_q_e7 + ' Yesterday you told me, [qA|' + _qa + '.]', 'You told me before, [qU|' + _qu + '.] ' + _a_e7))
CELLS = [(a, u, "W") for a in "WM" for u in "WM"]
def system_prompt(cell, k):
    ga, gu = cell[0], cell[1]
    A, U = A_NAMES[k], U_NAMES[k]
    sA = f"You are {A}, a {GNOUN[ga]}."; sU = f"You are chatting with {U}, a {GNOUN[gu]}."
    return (sA + " " + sU if k % 2 == 0 else sU + " " + sA) + f" Reply in character as {A}."
SUF_SETS = [" By the way, I am a", ' Earlier you told me, "I am a']
CHUNK = globals().get("CHUNK", 0)
ITS = list(range(CHUNK * 12, CHUNK * 12 + 12))
t0 = time.time()
ctxs = contexts(ITS)
D = 5376
READOUTS = [f"{k}_{s}" for k in ("jpp", "jlens", "naive") for s in ("pron", "noun", "noun4")] + \
           [f"{k}Q_{s}" for k in ("jpp", "jlens", "naive") for s in ("pron", "noun", "noun4")] + \
           ["probe_ref", "probe_ctx", "e7dir_self", "e7dir_user"]
NR = len(READOUTS)
LSAVE = [20, 30, 40, 50]
F4 = [" woman", " girl", " lady", " female"]; M4 = [" man", " boy", " guy", " male"]
g = lab.gender
pf, pm, nf, nm_ = g["pron_f"], g["pron_m"], g["noun_f"], g["noun_m"]
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]
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
# quiet dims
QMU = torch.tensor(np.load(f"{CACHE}/quiet_mu.npy")).to(lab.dev)
QM = torch.tensor(np.load(f"{CACHE}/quiet_mask.npy")).to(lab.dev)
# probes (E7)
pr = {}
for nm in ("ref", "ctx"):
    muf = torch.tensor(np.load(f"{E7CACHE}/probe_{nm}_muf.npy")).to(lab.dev); mum = torch.tensor(np.load(f"{E7CACHE}/probe_{nm}_mum.npy")).to(lab.dev)
    w = muf - mum; pr[nm] = (w, (muf + mum) / 2, (w * w).sum(-1))
# E7 raw-space referent directions (mean of the two cross-fit halves, renormalised)
d0 = np.load(f"{E7CACHE}/dirs_xfit0.npz"); d1 = np.load(f"{E7CACHE}/dirs_xfit1.npz")
E7D = {}
for key in ("A_self_ref", "A_user_ref"):
    v = torch.tensor(d0[key] + d1[key]); v = v / v.norm(dim=-1, keepdim=True)
    E7D[key] = v.to(lab.dev)
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
def capture(batch_ids, batch_pos):
    B = len(batch_ids); T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
    bi = torch.tensor([b for b in range(B) for _ in batch_pos[b]], device=lab.dev)
    pi = torch.tensor([p for b in range(B) for p in batch_pos[b]], device=lab.dev)
    store = {}; hs = []
    for l in range(60):
        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h[bi, pi].detach().to(torch.bfloat16)
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.no_grad():
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([store[l] for l in range(60)], 0)

seqs = []
for ci, c in enumerate(ctxs):
    for si, suf in enumerate(SUF_SETS):
        SUF_T = suf
        ids, pos = build_seq(lab, c)
        if si == 1:
            pos = {t: [] for t in TYPES}; pos["F_T"] = [len(ids) - 1]
        seqs.append((ci * 2 + si, ids, pos))
n_ctx2 = len(seqs)
n_ctx = n_ctx2
SC = np.full((n_ctx, len(TYPES), 60, NR), np.nan, dtype=np.float32)
CNT = np.zeros((n_ctx, len(TYPES)), dtype=np.int32)
BEH = np.zeros((n_ctx, 5), dtype=np.float32)
TOP = [None] * n_ctx
cells = ["".join(c) for c in CELLS]
RSUM = torch.zeros(len(ITS), len(cells), len(TYPES), len(LSAVE), D)
RN = np.zeros((len(ITS), len(cells), len(TYPES)))
BS = 12
torch.cuda.empty_cache()
with torch.no_grad():
    for s in range(0, len(seqs), BS):
        batch = seqs[s:s + BS]
        bpos, segs, off = [], [], 0
        for (ci, ids, pos) in batch:
            flat = []
            for ti, t in enumerate(TYPES):
                if pos[t]:
                    segs.append((ci, ti, off + len(flat), off + len(flat) + len(pos[t])))
                    flat += pos[t]
            bpos.append(flat); off += len(flat)
        H = capture([b[1] for b in batch], bpos)
        N = H.shape[1]
        S = torch.zeros(60, N, NR, device=lab.dev)
        for l in range(60):
            h = H[l]
            hc = torch.where(QM[l].unsqueeze(0), QMU[l].to(torch.bfloat16).unsqueeze(0), h)
            for kk, kind in enumerate(("jpp", "jlens", "naive")):
                if kind == "jlens" and l < 59:
                    Jm = lab.JL[l].to(lab.dev, non_blocking=True)
                    hJ, hJc = h @ Jm.T, hc @ Jm.T
                    del Jm
                else:
                    hJ, hJc = lab.lens(h, l, kind), lab.lens(hc, l, kind)
                S[l, :, kk * 3:(kk + 1) * 3] = scores(hJ)
                S[l, :, 9 + kk * 3:9 + (kk + 1) * 3] = scores(hJc)
            hf = h.float()
            for j, nm in enumerate(("ref", "ctx")):
                w, mid, ww = pr[nm]
                S[l, :, 18 + j] = ((hf - mid[l]) @ w[l]) / ww[l]
            S[l, :, 20] = hf @ E7D["A_self_ref"][l]
            S[l, :, 21] = hf @ E7D["A_user_ref"][l]
        S = S.cpu().numpy()
        Hs = H[LSAVE].float()
        for (ci, ti, a0, a1) in segs:
            SC[ci, ti] = S[:, a0:a1].mean(1)
            CNT[ci, ti] = a1 - a0
            c = ctxs[ci // 2]
            ij = ITS.index(c["item"]); cj = cells.index(c["cell"])
            RSUM[ij, cj, ti] += Hs[:, a0:a1].mean(1).cpu()
            RN[ij, cj, ti] += 1
            if TYPES[ti] == "F_T":
                lg = lab.full_logits(H[59, a0])
                lp = torch.log_softmax(lg.float(), -1)
                BEH[ci] = [float(lp[f4].exp().sum()), float(lp[m4].exp().sum()),
                           float(torch.logsumexp(lg[f4], 0) - torch.logsumexp(lg[m4], 0)),
                           float(lp[f4[0]].exp()), float(lp[m4[0]].exp())]
                v, ix = torch.topk(lp, 5)
                TOP[ci] = [(lab.tok.decode([int(i)]), round(float(x.exp()), 4)) for x, i in zip(v, ix)]
        del H, Hs
        print(f"batch {s // BS} done {time.time() - t0:.1f}s N={N}", flush=True)
RMEAN = (RSUM / torch.tensor(np.maximum(RN, 1), dtype=torch.float32)[..., None, None]).half().numpy()
np.save(f"{CACHE}/twoparty_resid_means_chunk{CHUNK}.npy", RMEAN)
np.savez_compressed(f"{RES}/V5_binding_dissociation_twoparty_chunk{CHUNK}.npz", SC=SC, CNT=CNT, BEH=BEH,
                    item=np.array([ctxs[i // 2]["item"] for i in range(n_ctx)]), cell=np.array([ctxs[i // 2]["cell"] for i in range(n_ctx)]),
                    k=np.array([ctxs[i // 2]["k"] for i in range(n_ctx)]), suf=np.array([i % 2 for i in range(n_ctx)]), types=np.array(TYPES), readouts=np.array(READOUTS),
                    cells=np.array(cells), items=np.array(ITS), lsave=np.array(LSAVE))
print("elapsed %.1fs" % (time.time() - t0))
result = {"chunk": CHUNK, "items": ITS, "n_ctx": n_ctx, "mean_counts": dict(zip(TYPES, CNT.mean(0).round(2).tolist())),
          "top5_FT": [{"item": ctxs[i // 2]["item"], "cell": ctxs[i // 2]["cell"], "suf": i % 2, "top": TOP[i]} for i in range(0, n_ctx, 7)],
          "elapsed": time.time() - t0}
