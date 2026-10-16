# R1_IT_LENS 10: absolute fidelity of the full-matrix IT lens (E3 metrics) at R1_FM_LAYERS on the held-out E3 stimuli.
# Lenses: jpp, jlens, naive (E3 conventions); it_full (probe-estimated L_y = diag(fnw) E[J_l] from R1_03 + R1_09, with the 16
# highest-energy output rows replaced by the exact directional rows); it_h0 / it_h1 (split-half estimates: noise check);
# it_lr256 (rank-256 SVD truncation of it_full, to strip isotropic probe noise); hybrids that swap input-column blocks between J++ and IT: hyb_jL_iQ = J++ on loud (non-q10) input coords + IT on q10 coords,
# hyb_iL_jQ = the reverse. Metrics vs the real final distribution: KL(final||lens) at temperature multipliers TAUS (pre-softcap
# logits * tau), top-1 agreement, top-10 overlap.
# PREDICTION (registered before running): it_full beats J++ in KL and top-1 at L40-58 on assistant/self positions; most of the
# gain is reproduced by hyb_jL_iQ (i.e. by fixing the quiet-coordinate columns).
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
dev = lab.dev
FM = R1_FM_LAYERS
TAUS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
LENSES = ["jpp", "jlens", "naive", "it_full", "it_h0", "it_h1", "hyb_jL_iQ", "hyb_iL_jQ", "it_lr256"]
fnw = lab.fnw.float()
GB = torch.load(R1_RES + "R1_04_gbar.pt")
names = GB["names"]
U16 = [k for k, n in enumerate(names) if n.startswith("unit")]
U16dims = torch.tensor([int(names[k][4:]) for k in U16], device=dev)
dpn = float(torch.load(R1_RES + "R1_02_dirs.pt")["dirs"][0].norm())
GU = GB["mix"][U16][:, FM].to(dev) / dpn                     # [16, nFM, D] exact rows  E[J_l]^T e_i
v6 = torch.load(R1_RES + "V6_quiet_dims_stats.pt", weights_only=False)
std = v6["std_all"].float()
q10 = (std <= torch.quantile(std, 0.1, dim=1, keepdim=True)).float().to(dev)
# accumulators (prefer the daemon STORE copies)
def getA(key, path):
    s = STORE.get(key)
    if s is None:
        s = torch.load(path, mmap=True)
    return s["A"], s["NA"]
A3, N3 = getA("R1_fit", R1_RES + "R1_03_fit_ckpt.pt")
A9, N9 = getA("R1_fm", R1_RES + "R1_09_fm_ckpt.pt")
NH = [float(N3[h] + N9[h]) for h in (0, 1)]
print("probe samples per half", NH, flush=True)
# ---- gather states at FM layers + 59 for all E3 eval positions
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
seqs = []
for i, (tag, txt) in enumerate(PROSE):
    ids = lab.ids(txt); seqs.append((f"prose{i:02d}", ids, {"prose": list(range(4, len(ids)))}))
for i, c in enumerate(chats):
    ids = lab.ids(lab.chat(c["q"], assistant=c["a"]))
    cats = e3_chat_positions(lab, ids, SELF_TOKS)
    seqs.append((f"chat{i:02d}", ids, {"asst": cats["nonself_asst"], "self": cats["self"], "user": cats["user"]}))
HS, meta = [], []
for name, ids, regions in seqs:
    allp = [p for c, ps in regions.items() for p in ps]
    cl = [c for c, ps in regions.items() for p in ps]
    H = lab.forward_resid([ids], layers=FM + [59], positions=[allp])[0]   # [12,P,D]
    HS.append(H.to(torch.bfloat16))
    meta += [{"name": name, "cls": c, "pos": p} for c, p in zip(cl, allp)]
HS = torch.cat(HS, 1).to(dev)                                            # [12,Ptot,D] bf16
Ptot = HS.shape[1]
print("positions", Ptot, round(time.time() - t0, 1), flush=True)
WU = lab.WU
def zlogits(hJ):
    x = hJ.float()
    return ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * fnw).to(WU.dtype) @ WU.T).float()
def logits_from(hJ, tau):
    return torch.tanh(zlogits(hJ) * tau / 30.0) * 30.0
CH = 256
fin_logp, fin_top10 = [], []
for c0 in range(0, Ptot, CH):
    lf = logits_from(HS[-1, c0:c0 + CH], 1.0)
    fin_logp.append(F.log_softmax(lf, -1).to(torch.bfloat16)); fin_top10.append(torch.topk(lf, 10, -1).indices)
    del lf
KL = torch.zeros(Ptot, len(FM), len(LENSES), len(TAUS), dtype=torch.float16)
T1 = torch.zeros(Ptot, len(FM), len(LENSES), dtype=torch.bool)
T10 = torch.zeros(Ptot, len(FM), len(LENSES), dtype=torch.float16)
for j, l in enumerate(FM):
    Afull = ((A3[0, j] + A3[1, j] + A9[0, j] + A9[1, j]) / (NH[0] + NH[1])).to(dev, torch.bfloat16)
    Ah = [((A3[h, j] + A9[h, j]) / NH[h]).to(dev, torch.bfloat16) for h in (0, 1)]
    Uu, Ss, Vv = torch.svd_lowrank(Afull.float(), q=288, niter=4)      # rank-256 truncation: probe noise is ~isotropic
    Alr = ((Uu[:, :256] * Ss[:256]) @ Vv[:, :256].T).to(torch.bfloat16)
    del Uu, Ss, Vv
    JL = lab.JL[l].to(dev)
    def it_raw(hm, Lt):
        y = (hm @ Lt).float()
        hJ = y / fnw
        hJ[:, U16dims] = hm.float() @ GU[:, j].T
        return hJ
    for c0 in range(0, Ptot, CH):
        h = HS[j, c0:c0 + CH]
        hq = (h.float() * q10[l]).to(torch.bfloat16); hl = (h.float() * (1 - q10[l])).to(torch.bfloat16)
        hJ = {"jpp": (h @ lab.J[l].T).float(), "jlens": (h @ JL.T).float(), "naive": h.float(),
              "it_full": it_raw(h, Afull), "it_h0": it_raw(h, Ah[0]), "it_h1": it_raw(h, Ah[1])}
        hJ["hyb_jL_iQ"] = (hl @ lab.J[l].T).float() + it_raw(hq, Afull)
        hJ["hyb_iL_jQ"] = it_raw(hl, Afull) + (hq @ lab.J[l].T).float()
        hJ["it_lr256"] = it_raw(h, Alr)
        lpf = fin_logp[c0 // CH].float(); pf = lpf.exp(); t10f = fin_top10[c0 // CH]
        for k, ln in enumerate(LENSES):
            z = zlogits(hJ[ln])
            for ti, tau in enumerate(TAUS):
                lg = torch.tanh(z * tau / 30.0) * 30.0
                lp = F.log_softmax(lg, -1)
                KL[c0:c0 + CH, j, k, ti] = (pf * (lpf - lp)).sum(-1).half().cpu()
                if tau == 1.0:
                    t10 = torch.topk(lg, 10, -1).indices
                    T1[c0:c0 + CH, j, k] = (t10[:, 0] == t10f[:, 0]).cpu()
                    T10[c0:c0 + CH, j, k] = (t10.unsqueeze(-1) == t10f.unsqueeze(-2)).any(-1).float().mean(-1).half().cpu()
                del lg, lp
            del z
        del hJ
    del Afull, Ah, JL, Alr
    torch.cuda.empty_cache()
    print("layer", l, round(time.time() - t0, 1), flush=True)
torch.save({"meta": meta, "KL": KL, "T1": T1, "T10": T10, "lenses": LENSES, "taus": TAUS, "layers": FM, "NH": NH},
           R1_RES + "R1_10_evalfull.pt")
result = {"P": Ptot, "NH": NH, "t": time.time() - t0}
