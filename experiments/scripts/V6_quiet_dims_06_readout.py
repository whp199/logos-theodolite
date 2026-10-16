# V6_quiet_dims_06_readout -- consequence of quiet coordinates for the gender READOUTS (C1/C4/C7).
# 7 conditions (default, nameless, Gemma-short, woman x2, man x2) x 24 E1 passages (teacher-forced), all assistant/user content
# tokens, layers 10-58.  For each state h we read the J++ and JLens gender scores (pron, noun) under 5 variants:
#   full; mean10 (bottom-10%-std coords -> corpus token-mean); meanP (181 persistent-quiet coords -> mean);
#   meanDeep (per-layer std < 0.05*median coords -> mean); zero10 (bottom-10% coords -> 0, removes their DC part too).
# plus a linear decomposition of the pre-softcap pron score into mu_loud / mu_quiet / dev_loud / dev_quiet parts (q10 set,
# divided by the FULL state's rms(hJ)), and the IT-gradient readouts <h, gbar_l> (E3 gbar: all / chat / prose) full and mean10.
# Quiet sets / means come from V6_quiet_dims_01_stats (independent corpus: E3 prose + chats).
# PRE-REGISTERED PREDICTIONS
#  R1: the woman-man separation (persona gap) survives masking at L20-49 (>= 70% of the full gap remains under mean10) for J++.
#  R2: the default-vs-nameless difference stays within +-3% of the gap under every variant (C1 survives).
#  R3: Gemma-minus-nameless (+7-9% of span in E1) survives masking with the same sign.
#  R4: the absolute 'background' (default position on the woman/man scale, and raw score level) moves by > 0.1 of the gap under
#      masking at L40-58 (E6 saw large background shifts), i.e. absolute positions are quiet-dim dependent.
#  R5: masking moves the J++ default position toward the IT-gradient readout position (E3: J++ 0.45 vs IT 0.22 at L30-39);
#      refuted if |J++ - IT| does not shrink.
exec(open("./experiments/scripts/V6_quiet_dims_common.py").read())
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
t0 = time.time()
CONDS = [("default", None), ("nameless", "You are a helpful AI assistant."), ("gemma_short", "You are Gemma, a helpful AI assistant."),
         ("woman_p1", "You are Laura, a 34-year-old woman. Stay in character."),
         ("woman_p2", "You are a 41-year-old woman. Answer every message as yourself and stay in character."),
         ("man_p1", "You are Daniel, a 34-year-old man. Stay in character."),
         ("man_p2", "You are a 41-year-old man. Answer every message as yourself and stay in character.")]
NP = 24
LAYS = list(range(10, 59))
S = torch.load(V6_R + "V6_quiet_dims_stats.pt")
std = S["std_all"].float().to(lab.dev); mu = S["mean_all"].float().to(lab.dev)
q10 = torch.stack([std[l] < torch.quantile(std[l], 0.10) for l in range(60)])
deep = torch.stack([std[l] < 0.05 * std[l].median() for l in range(60)])
pers = deep[42:58].all(0)
GB = torch.load(V6_R + "E3_lens_validation_gbar.pt")
gbars = {k: GB[k][0].float().to(lab.dev) for k in ("gbar", "gbar_chat", "gbar_prose")}  # [60, D] pron direction
g = lab.gender
dpron = (lab.WU[torch.tensor(g["pron_f"], device=lab.dev)].float() - lab.WU[torch.tensor(g["pron_m"], device=lab.dev)].float()).mean(0)
VARS = ["full", "mean10", "meanP", "meanDeep", "zero10"]
items = []
for ci, (cn, sp) in enumerate(CONDS):
    for qi in range(NP):
        ids, pos, labs, toks = e1_build(qi, cn, sp)
        keep = [(p, lb, tk) for p, lb, tk in zip(pos, labs, toks) if lb in ("self", "nonself", "user_I", "user_other")]
        items.append({"cond": cn, "qi": qi, "ids": ids, "pos": [k[0] for k in keep], "lab": [k[1] for k in keep], "tok": [k[2] for k in keep]})
meta = [{"cond": it["cond"], "qi": it["qi"], "lab": lb, "tok": tk} for it in items for lb, tk in zip(it["lab"], it["tok"])]
N = len(meta)
nL = len(LAYS)
SC = {kind: torch.zeros(N, nL, len(VARS), 2) for kind in ("jpp", "jlens")}     # pron, noun
LIN = {kind: torch.zeros(N, nL, 4) for kind in ("jpp", "jlens")}               # mu_loud, mu_quiet, dev_loud, dev_quiet
ITR = torch.zeros(N, nL, 3, 2)                                                 # gbar kinds x (full, mean10)
off = 0
BS = 8


def variants(h, l):
    m10 = q10[l].unsqueeze(0); mP = pers.unsqueeze(0); mD = deep[l].unsqueeze(0); m = mu[l].unsqueeze(0)
    return {"full": h, "mean10": torch.where(m10, m, h), "meanP": torch.where(mP, m, h), "meanDeep": torch.where(mD, m, h),
            "zero10": torch.where(m10, torch.zeros_like(h), h)}


with torch.no_grad():
    for b0 in range(0, len(items), BS):
        ch = items[b0:b0 + BS]
        store = v6_gather([it["ids"] for it in ch], [it["pos"] for it in ch], layers=LAYS)
        n = store[LAYS[0]].shape[0]
        for li, l in enumerate(LAYS):
            h = store[l]
            V = variants(h, l)
            for kind in ("jpp", "jlens"):
                Jm = lab.J[l] if kind == "jpp" else lab.JL[l].to(lab.dev)
                for vi, vn in enumerate(VARS):
                    hJ = (V[vn].to(torch.bfloat16) @ Jm.T)
                    sc = lab.gender_scores(hJ)
                    SC[kind][off:off + n, li, vi, 0] = sc["pron"].float().cpu(); SC[kind][off:off + n, li, vi, 1] = sc["noun"].float().cpu()
                # linear decomposition of pre-softcap pron score (fixed rms of the full hJ)
                hJf = (h.to(torch.bfloat16) @ Jm.T).float()
                rmsJ = hJf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                m10 = q10[l].float().unsqueeze(0); mm = mu[l].unsqueeze(0)
                parts = [mm * (1 - m10), mm * m10, (h - mm) * (1 - m10), (h - mm) * m10]
                for pi, P in enumerate(parts):
                    PJ = (P.expand_as(h).to(torch.bfloat16) @ Jm.T).float()
                    LIN[kind][off:off + n, li, pi] = ((PJ * lab.fnw.float()) @ dpron / rmsJ.squeeze(-1)).cpu()
                if kind == "jlens":
                    del Jm
            for gi, gk in enumerate(("gbar", "gbar_chat", "gbar_prose")):
                ITR[off:off + n, li, gi, 0] = (h @ gbars[gk][l]).cpu()
                ITR[off:off + n, li, gi, 1] = (V["mean10"] @ gbars[gk][l]).cpu()
        off += n
        del store
    # background: readout of the corpus mean itself, full and zero10
    BG = {}
    for kind in ("jpp", "jlens"):
        rows = []
        for l in LAYS:
            Jm = lab.J[l] if kind == "jpp" else lab.JL[l].to(lab.dev)
            m = mu[l].unsqueeze(0)
            r = []
            for hv in (m, torch.where(q10[l].unsqueeze(0), torch.zeros_like(m), m)):
                sc = lab.gender_scores(hv.to(torch.bfloat16) @ Jm.T)
                r.append([float(sc["pron"]), float(sc["noun"])])
            rows.append(r)
        BG[kind] = rows
assert off == N
torch.save({"meta": meta, "layers": LAYS, "vars": VARS, "SC": SC, "LIN": LIN, "ITR": ITR, "BG": BG, "conds": [c for c, _ in CONDS]},
           V6_R + "V6_quiet_dims_06_readout.pt")
print("N", N, "t", round(time.time() - t0, 1))
result = {"N": N, "t": round(time.time() - t0, 1)}
