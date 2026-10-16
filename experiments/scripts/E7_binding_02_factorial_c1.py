CHUNK = 1
# E7_binding_02: 3x3 factorial (G_self x G_user in system prompt) + default, teacher-forced assistant passages.
# Readouts per position type (A_self, A_user, A_neut, U_self, U_addr, F_I, F_Y) and layer:
#   jpp/jlens/naive x (pron, noun, noun4) + probe_ref + probe_ctx projections  (11 readouts)
# plus behavioural next-token gender log-odds at "... I am a" / "... you are a" and variant-averaged raw residual means.
# CHUNK is set by the first line of the submitted copy.
import os, time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 8, CHUNK * 8 + 8))
t0 = time.time()
ctxs = contexts(PIDS)
D = 5376
NR = 11
READOUTS = [f"{k}_{s}" for k in ("jpp", "jlens", "naive") for s in ("pron", "noun", "noun4")] + ["probe_ref", "probe_ctx"]
sid = e7_score_ids(lab)
pr = {}
for nm in ("ref", "ctx"):
    if f"{EXP}_probe_{nm}" in STORE:
        muf, mum = STORE[f"{EXP}_probe_{nm}"]
    else:
        muf = torch.tensor(np.load(f"{CACHE}/probe_{nm}_muf.npy")); mum = torch.tensor(np.load(f"{CACHE}/probe_{nm}_mum.npy"))
    w = (muf - mum).to(lab.dev); mid = ((muf + mum) / 2).to(lab.dev)
    pr[nm] = (w, mid, (w * w).sum(-1))
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]

# build sequences
seqs = []  # (ctx_idx, kind, ids, pos dict)
for ci, c in enumerate(ctxs):
    q, a = PASSAGES[c["pid"]]
    prefix = lab.chat(q, system=c["system"], assistant=a)
    pids_ = lab.ids(prefix)
    for kind, suf in (("I", SUF_I), ("Y", SUF_Y)):
        ids = lab.ids(prefix + suf)
        assert ids[:len(pids_)] == pids_, "prefix tokenization changed"
        if kind == "I":
            pos = e7_positions(lab, ids, len(pids_))
            pos["F_I"] = [len(ids) - 1]
            pos["F_Y"] = []
        else:
            pos = {k: [] for k in TYPES}
            pos["F_Y"] = [len(ids) - 1]
        seqs.append((ci, kind, ids, pos))
n_ctx = len(ctxs)
SC = np.full((n_ctx, len(TYPES), 60, NR), np.nan, dtype=np.float32)
CNT = np.zeros((n_ctx, len(TYPES)), dtype=np.int32)
BEH = np.zeros((n_ctx, 2, 5), dtype=np.float32)  # pF4, pM4, lse4, p_woman, p_man
TOP = [[None, None] for _ in range(n_ctx)]
cell_names = ["".join(x) for x in CELLS] + ["D"]
RSUM = torch.zeros(len(PIDS), len(cell_names), len(TYPES), 60, D)  # variant-summed per-type mean residuals
RN = np.zeros((len(PIDS), len(cell_names), len(TYPES)))
BS = 32
for s in range(0, len(seqs), BS):
    batch = seqs[s:s + BS]
    bpos, segs, off = [], [], 0
    for (ci, kind, ids, pos) in batch:
        flat = []
        for ti, tname in enumerate(TYPES):
            if pos[tname]:
                segs.append((ci, ti, off + len(flat), off + len(flat) + len(pos[tname])))
                flat += pos[tname]
        bpos.append(flat)
        off += len(flat)
    H = e7_capture(lab, [b[2] for b in batch], bpos)  # [60, N, D] bf16 GPU
    N = H.shape[1]
    S = torch.zeros(60, N, NR, device=lab.dev)
    for l in range(60):
        for kk, kind in enumerate(("jpp", "jlens", "naive")):
            S[l, :, kk * 3:(kk + 1) * 3] = e7_scores(lab, lab.lens(H[l], l, kind), sid)
        hf = H[l].float()
        for j, nm in enumerate(("ref", "ctx")):
            w, mid, ww = pr[nm]
            S[l, :, 9 + j] = ((hf - mid[l]) @ w[l]) / ww[l]
    S = S.cpu().numpy()
    for (ci, ti, a0, a1) in segs:
        SC[ci, ti] = S[:, a0:a1].mean(1)
        CNT[ci, ti] = a1 - a0
        c = ctxs[ci]
        pj = PIDS.index(c["pid"]); cj = cell_names.index(c["cell"])
        RSUM[pj, cj, ti] += H[:, a0:a1].float().mean(1).cpu()
        RN[pj, cj, ti] += 1
        if TYPES[ti] in ("F_I", "F_Y"):
            lg = lab.full_logits(H[59, a0])
            lp = torch.log_softmax(lg.float(), -1)
            k2 = 0 if TYPES[ti] == "F_I" else 1
            BEH[ci, k2] = [float(lp[f4].exp().sum()), float(lp[m4].exp().sum()),
                           float(torch.logsumexp(lg[f4], 0) - torch.logsumexp(lg[m4], 0)),
                           float(lp[f4[0]].exp()), float(lp[m4[0]].exp())]
            v, ix = torch.topk(lp, 5)
            TOP[ci][k2] = [(lab.tok.decode([int(i)]), round(float(x.exp()), 4)) for x, i in zip(v, ix)]
    del H
    print(f"batch {s // BS} done {time.time() - t0:.1f}s", flush=True)
RMEAN = (RSUM / torch.tensor(np.maximum(RN, 1), dtype=torch.float32)[..., None, None]).half().numpy()
np.save(f"{CACHE}/resid_means_chunk{CHUNK}.npy", RMEAN)
np.savez_compressed(f"{RES}/E7_binding_factorial_chunk{CHUNK}.npz", SC=SC, CNT=CNT, BEH=BEH,
                    pid=np.array([c["pid"] for c in ctxs]), cell=np.array([c["cell"] for c in ctxs]),
                    k=np.array([c["k"] for c in ctxs]), types=np.array(TYPES), readouts=np.array(READOUTS),
                    cell_names=np.array(cell_names), pids=np.array(PIDS))
# example positions for the record
ex = seqs[0]
example = {t: [(p, lab.tok.decode([ex[2][p]])) for p in ex[3][t]] for t in TYPES}
print("elapsed %.1fs" % (time.time() - t0))
print("mean counts per type:", dict(zip(TYPES, CNT.mean(0).round(2).tolist())))
result = {"chunk": CHUNK, "pids": PIDS, "n_ctx": n_ctx, "example_positions": example,
          "top5": [{"pid": ctxs[i]["pid"], "cell": ctxs[i]["cell"], "k": ctxs[i]["k"], "I": TOP[i][0], "Y": TOP[i][1]} for i in range(n_ctx)],
          "elapsed": time.time() - t0}
