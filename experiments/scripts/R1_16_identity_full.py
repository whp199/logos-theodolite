# R1_IT_LENS 16: identity comparison read through the FULL-MATRIX IT lens with the tool's own convention
# (hJ -> RMSNorm(final weights) -> W_U -> softcap; pron/noun = mean F-M logit difference, as lab.gender_scores), at
# R1_FM_LAYERS. Lenses: it_full, it_lr256 (rank-256 truncation), and jpp for a same-job reference. Self/nonself/user positions.
import json, time
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
exec(open("./experiments/scripts/R1_common.py").read())
t0 = time.time()
dev = lab.dev
FM = R1_FM_LAYERS
CONDS = ["default", "nameless", "gemma_short", "gemma_explicit", "f_sarah", "m_james", "n_atlas", "n_quill", "n_unit7",
         "woman_p1", "woman_p2", "man_p1", "man_p2"]
SYS = dict(E1_CONDS)
fnw = lab.fnw.float()
GB = torch.load(R1_RES + "R1_04_gbar.pt")
names = GB["names"]
U16 = [k for k, n in enumerate(names) if n.startswith("unit")]
U16dims = torch.tensor([int(names[k][4:]) for k in U16], device=dev)
dpn = float(torch.load(R1_RES + "R1_02_dirs.pt")["dirs"][0].norm())
GU = GB["mix"][U16][:, FM].to(dev) / dpn
def getA(key, path):
    s = STORE.get(key)
    if s is None:
        s = torch.load(path, mmap=True)
    return s["A"], s["NA"]
A3, N3 = getA("R1_fit", R1_RES + "R1_03_fit_ckpt.pt")
A9, N9 = getA("R1_fm", R1_RES + "R1_09_fm_ckpt.pt")
NT = float(N3.sum() + N9.sum())
LT, LR = [], []
for j in range(len(FM)):
    a = ((A3[0, j] + A3[1, j] + A9[0, j] + A9[1, j]) / NT).to(dev)
    U, Sv, V = torch.svd_lowrank(a, q=288, niter=4)
    LT.append(a.to(torch.bfloat16)); LR.append(((U[:, :256] * Sv[:256]) @ V[:, :256].T).to(torch.bfloat16))
    del a, U, Sv, V
def it_raw(h, Lt, j):
    hJ = (h @ Lt).float() / fnw
    hJ[:, U16dims] = h.float() @ GU[:, j].T
    return hJ
meta, chunks = [], []
off = 0
for qi in range(40):
    bl = [e1_build(qi, c, SYS[c]) for c in CONDS]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    per_l = []
    for j, l in enumerate(FM):
        h = store[l]
        row = []
        for hJ in (it_raw(h, LT[j], j), it_raw(h, LR[j], j), lab.lens(h, l, "jpp")):
            sc = lab.gender_scores(hJ)
            row += [sc["pron"].float(), sc["noun"].float()]
        per_l.append(torch.stack(row, -1))
    chunks.append(torch.stack(per_l, 1).half().cpu())
    del store
    for cname, (ids, pos, labs, toks) in zip(CONDS, bl):
        meta.append({"qi": qi, "cond": cname, "off": off, "n": len(pos), "pos": pos, "labels": labs})
        off += len(pos)
cols = ["itfull_pron", "itfull_noun", "itlr_pron", "itlr_noun", "jpp_pron", "jpp_noun"]
torch.save({"meta": meta, "scores": torch.cat(chunks, 0), "cols": cols, "conds": CONDS, "layers": FM, "NT": NT},
           R1_RES + "R1_16_identity_full.pt")
result = {"rows": off, "t": time.time() - t0, "NT": NT}
