# V6_quiet_dims_06b_readout_ctrl -- CONTROL masks for 06: is the effect of masking the bottom-10%-std coords specific?
# Variants: full; rand10_s{0,1,2} (random 538 coords -> corpus mean); mod (bottom-10% minus deep-quiet coords -> mean);
# top10gain (top-10% |input_layernorm[l+1]| coords -> mean).  Same passages/conditions/positions as 06.
# PRE-REGISTERED: random 10% masks retain >= 0.85 of the persona gap at every band (otherwise masking per se is destructive);
#   if 'mod' reproduces the L50-58 gap collapse seen with mean10, the late J++ persona gap lives in moderately-low-variance coords.
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
VARS = ["full", "rand10_s0", "rand10_s1", "rand10_s2", "mod", "top10gain"]
RM = []
for sd_ in range(3):
    gg = torch.Generator().manual_seed(77 + sd_)
    RM.append(torch.stack([torch.zeros(5376, dtype=torch.bool).index_fill_(0, torch.randperm(5376, generator=gg)[:538], True) for l in range(60)]).to(lab.dev))
WIN = S["g_w_in"].float().abs().to(lab.dev)
TG = torch.stack([WIN[min(l + 1, 59)] >= torch.quantile(WIN[min(l + 1, 59)], 0.90) for l in range(60)])
MOD = q10 & ~deep
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
    o = {"full": h, "mod": torch.where(MOD[l].unsqueeze(0), m, h), "top10gain": torch.where(TG[l].unsqueeze(0), m, h)}
    for k in range(3):
        o[f"rand10_s{k}"] = torch.where(RM[k][l].unsqueeze(0), m, h)
    return o


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
                if kind == "jlens":
                    del Jm
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
           V6_R + "V6_quiet_dims_06b_readout_ctrl.pt")
print("N", N, "t", round(time.time() - t0, 1))
result = {"N": N, "t": round(time.time() - t0, 1)}
