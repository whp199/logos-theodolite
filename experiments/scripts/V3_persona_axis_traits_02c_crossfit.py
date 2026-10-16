# V3_persona_axis_traits_02c (offline, CPU; .venv/bin/python). CROSS-FITTED geometry of persona / trait / role / prose
# axes and the AI-vs-human offset at assistant self tokens.
# Design: passages split 12/12 into folds A/B; every axis (and projection-out basis) is estimated on one fold and all
# positions are measured on condition means from the OTHER fold (then swapped and averaged). This removes the in-sample
# bias that pulls every position toward 0.5 when an axis is noisy (a problem for whitened metrics).
# Point estimate = mean over 24 random no-resample splits; 95% CI = percentiles over NBOOT replicates with a fresh random
# split and stratified within-fold passage resampling. Prose pairs are split by clause the same way.
# Metrics: raw (all 31 layers L20-50; also raw restricted to the 7 covariance layers), full whitening with token
# covariance (shrinkage lambda 0.01/0.1/0.5), diag whitening, drop 10% quietest dims, drop top-1% variance dims,
# gain metric (|input_layernorm weight of layer l+1|).
# Null for projection-out: persona axis with k-dim subspaces of OTHER persona-description contrasts removed (all 15
# pairwise differences among CT1_org, CT1_dis, CT2_city, CT2_rural, neutral_human, neutral_human2).
# Predictions: see V3_persona_axis_traits_01_collect.py header (H1-H4).
import json, math, sys, time, itertools
import numpy as np, torch
torch.set_num_threads(16)
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
t0 = time.time()
LOC = sys.argv[1] if len(sys.argv) > 1 else "self"
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NPT = 24
A = torch.load(f"{V3_CACHE}/self_A.pt"); Bd = torch.load(f"{V3_CACHE}/self_B.pt"); PR = torch.load(f"{V3_CACHE}/prose.pt")
NW = torch.load(f"{V3_ROOT}/experiments/results/E4_causal_cache/normweights.pt")["input_layernorm"].double()
LI = {l: i for i, l in enumerate(LAYERS)}
cos = torch.nn.functional.cosine_similarity

def gather(S, key):
    out = {}
    for c in dict.fromkeys(it["cond"] for it in S["items"]):
        d = {it["p"]: i for i, it in enumerate(S["items"]) if it["cond"] == c}
        out[c] = S[key][[d[p] for p in range(NP)]]
    return out
X = {**gather(A, LOC), **gather(Bd, LOC)}
X = {c: v.float() for c, v in X.items()}
pit = PR["items"]
pidx = {(it["tpl"], it["i"], it["cond"]): k for k, it in enumerate(pit)}
pairs = [(pidx[(t, i, "F")], pidx[(t, i, "M")]) for t in ("A", "B") for i in range(36)]
pair_clause = [i for t in ("A", "B") for i in range(36)]
PD_c = (PR["cont"][[a for a, b in pairs]] - PR["cont"][[b for a, b in pairs]]).float()
NPR = len(pairs)

GPW = [w for w, m in PAIRS["persona"]]; GPM = [m for w, m in PAIRS["persona"]]
TW = [a for a, b in PAIRS["trait"]]; TB = [b for a, b in PAIRS["trait"]]
REPORT = ["default", "nameless_AI", "Gemma", "AI_plain", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2",
          "TP_warm(pooled)", "TP_blunt(pooled)", "RL_serve(pooled)", "RL_solo(pooled)", "CT1_org", "CT1_dis", "W(pooled)", "M(pooled)", "GP1_W", "GP1_M"]
NULLPOOL = list(itertools.combinations(["CT1_org", "CT1_dis", "CT2_city", "CT2_rural", "neutral_human", "neutral_human2"], 2))
rng0 = np.random.default_rng(11)
NULL_K2 = [tuple(rng0.choice(len(NULLPOOL), 2, replace=False)) for _ in range(20)]
NULL_K4 = [tuple(rng0.choice(len(NULLPOOL), 4, replace=False)) for _ in range(20)]

def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def proj_out(u, basis):
    Q, _ = torch.linalg.qr(basis.transpose(-1, -2))
    return u - (Q @ (Q.transpose(-1, -2) @ u.unsqueeze(-1))).squeeze(-1)

def cond_means(Xl, W):
    M = {c: W @ v for c, v in Xl.items()}
    M["W(pooled)"] = sum(M[c] for c in GPW) / 4; M["M(pooled)"] = sum(M[c] for c in GPM) / 4
    M["TP_warm(pooled)"] = sum(M[c] for c in TW) / 4; M["TP_blunt(pooled)"] = sum(M[c] for c in TB) / 4
    M["RL_serve(pooled)"] = (M["RL1_serve"] + M["RL2_serve"]) / 2; M["RL_solo(pooled)"] = (M["RL1_solo"] + M["RL2_solo"]) / 2
    M["nh(pooled)"] = (M["neutral_human"] + M["neutral_human2"]) / 2
    return M

def axes_from(M, Pd, Wp):
    ax = {"persona": M["W(pooled)"] - M["M(pooled)"], "trait": M["TP_warm(pooled)"] - M["TP_blunt(pooled)"],
          "role": M["RL_serve(pooled)"] - M["RL_solo(pooled)"], "ctrl1": M["CT1_org"] - M["CT1_dis"],
          "ctrl2": M["CT2_city"] - M["CT2_rural"], "prose": Wp @ Pd,
          "AIoff": M["default"] - M["nh(pooled)"], "AIoff_E4": M["default"] - M["neutral_human"],
          "AI_warm-AI_blunt": M["AI_warm"] - M["AI_blunt"], "AIblunt_off": M["AI_blunt"] - M["nh(pooled)"]}
    for k, (a, b) in enumerate(PAIRS["persona"]):
        ax[f"persona_{k+1}"] = M[a] - M[b]
    for k, (a, b) in enumerate(PAIRS["trait"]):
        ax[f"trait_{k+1}"] = M[a] - M[b]
    for k, (a, b) in enumerate(PAIRS["role"]):
        ax[f"role_{k+1}"] = M[a] - M[b]
    for j, (a, b) in enumerate(NULLPOOL):
        ax[f"null_{j}"] = M[a] - M[b]
    return ax

def variants_from(ax, with_null=False):
    st = lambda names: torch.stack([ax[n] for n in names], -2)
    v = {"persona": ax["persona"], "persona_GP1only": ax["persona_1"],
         "persona_perp_trait": proj_out(ax["persona"], st(["trait"])),
         "persona_perp_trait4": proj_out(ax["persona"], st([f"trait_{k}" for k in range(1, 5)])),
         "persona_perp_role": proj_out(ax["persona"], st(["role"])),
         "persona_perp_role2": proj_out(ax["persona"], st(["role_1", "role_2"])),
         "persona_perp_trait_role": proj_out(ax["persona"], st(["trait", "role"])),
         "persona_perp_trait4_role2": proj_out(ax["persona"], st([f"trait_{k}" for k in range(1, 5)] + ["role_1", "role_2"])),
         "persona_perp_ctrl2": proj_out(ax["persona"], st(["ctrl1", "ctrl2"])),
         "persona_perp_prose": proj_out(ax["persona"], st(["prose"])),
         "persona_perp_AIoff": proj_out(ax["persona"], st(["AIoff"])),
         "trait_component_of_persona": (ax["persona"] * unit(ax["trait"])).sum(-1, keepdim=True) * unit(ax["trait"]),
         "prose": ax["prose"], "trait_axis": ax["trait"], "role_axis": ax["role"]}
    if with_null:
        for j in range(len(NULLPOOL)):
            v[f"NULL1_{j}"] = proj_out(ax["persona"], st([f"null_{j}"]))
        for j, ks in enumerate(NULL_K2):
            v[f"NULL2_{j}"] = proj_out(ax["persona"], st([f"null_{k}" for k in ks]))
        for j, ks in enumerate(NULL_K4):
            v[f"NULL4_{j}"] = proj_out(ax["persona"], st([f"null_{k}" for k in ks]))
    return v

def positions(var, Mev, lo_name="M(pooled)", hi_name="W(pooled)", conds=REPORT):
    out = {}
    for vn, u in var.items():
        u = unit(u)
        lo, hi = (Mev["GP1_M"], Mev["GP1_W"]) if vn == "persona_GP1only" else (Mev[lo_name], Mev[hi_name])
        den = ((hi - lo) * u).sum(-1)
        cl = ["default"] if vn.startswith("NULL") else conds
        for c in cl:
            out[f"pos|{vn}|{c}"] = ((Mev[c] - lo) * u).sum(-1) / den
    return out

COSPAIRS = [("persona", "trait"), ("persona", "role"), ("persona", "prose"), ("persona", "ctrl1"), ("persona", "ctrl2"),
            ("persona", "AIoff"), ("persona", "AIoff_E4"), ("persona", "AI_warm-AI_blunt"), ("persona", "AIblunt_off"),
            ("trait", "AIoff"), ("trait", "AI_warm-AI_blunt"), ("trait", "role"), ("trait", "prose"), ("trait", "AIblunt_off"),
            ("role", "AIoff"), ("prose", "AIoff"), ("ctrl1", "AIoff"), ("ctrl2", "AIoff"), ("AIoff", "AIblunt_off"),
            ("persona_1", "trait_1"), ("persona_2", "trait_2")]

def analyze(Xl, Pd, WA, WB, WpA, WpB, with_null):
    MA, MB = cond_means(Xl, WA), cond_means(Xl, WB)
    aA, aB = axes_from(MA, Pd, WpA), axes_from(MB, Pd, WpB)
    out = {}
    for a, b in COSPAIRS:
        out[f"cosin|{a}|{b}"] = 0.5 * (cos(aA[a], aA[b], dim=-1) + cos(aB[a], aB[b], dim=-1))
        out[f"cosx|{a}|{b}"] = 0.5 * (cos(aA[a], aB[b], dim=-1) + cos(aB[a], aA[b], dim=-1))
    for a in ("persona", "trait", "role", "prose", "AIoff", "ctrl1", "ctrl2", "AI_warm-AI_blunt"):
        out[f"rel|{a}"] = cos(aA[a], aB[a], dim=-1)
    for a in ("persona", "trait", "role", "AIoff", "prose"):
        out[f"norm|{a}"] = 0.5 * (aA[a].norm(dim=-1) + aB[a].norm(dim=-1))
    vA, vB = variants_from(aA, with_null), variants_from(aB, with_null)
    pAB, pBA = positions(vA, MB), positions(vB, MA)
    for k in pAB:
        out[k] = 0.5 * (pAB[k] + pBA[k])
    for vn in vA:
        if not vn.startswith("NULL"):
            out[f"retained|{vn}"] = 0.5 * (cos(unit(vA[vn]), aA["persona"], dim=-1) + cos(unit(vB[vn]), aB["persona"], dim=-1))
    tA = positions({"trait": aA["trait"]}, MB, "TP_blunt(pooled)", "TP_warm(pooled)")
    tB = positions({"trait": aB["trait"]}, MA, "TP_blunt(pooled)", "TP_warm(pooled)")
    for k in tA:
        out["t" + k] = 0.5 * (tA[k] + tB[k])
    rA = positions({"role": aA["role"]}, MB, "RL_solo(pooled)", "RL_serve(pooled)")
    rB = positions({"role": aB["role"]}, MA, "RL_solo(pooled)", "RL_serve(pooled)")
    for k in rA:
        out["r" + k] = 0.5 * (rA[k] + rB[k])
    # decomposition (axes from one fold, offset/poles from the other)
    def dec(ax_fit, Mev):
        up, ut, ur = unit(ax_fit["persona"]), unit(ax_fit["trait"]), unit(ax_fit["role"])
        off = Mev["default"] - Mev["nh(pooled)"]
        gap = ((Mev["W(pooled)"] - Mev["M(pooled)"]) * up).sum(-1)
        tot = (off * up).sum(-1)
        vt = (off * ut).sum(-1) * (up * ut).sum(-1)
        vr = (off * ur).sum(-1) * (up * ur).sum(-1)
        return tot / gap, vt / gap, vr / gap, off.norm(dim=-1) / gap
    d1, d2 = dec(aA, MB), dec(aB, MA)
    for j, nm in enumerate(["AIoff_on_persona/gap", "via_trait/gap", "via_role/gap", "|AIoff|/gap"]):
        out[f"dec|{nm}"] = 0.5 * (d1[j] + d2[j])
    return {k: v.numpy() for k, v in out.items()}

# fold weights
def make_weights(n_items, groups, R, resample, rng):
    """groups: group id per item (folds are assigned per group). Returns WA, WB [R, n_items]."""
    ug = sorted(set(groups)); WA = np.zeros((R, n_items)); WB = np.zeros((R, n_items))
    garr = np.array(groups)
    for r in range(R):
        perm = rng.permutation(ug); gA = set(perm[: len(ug) // 2].tolist())
        idxA = np.where(np.isin(garr, list(gA)))[0]; idxB = np.where(~np.isin(garr, list(gA)))[0]
        for W, idx in ((WA, idxA), (WB, idxB)):
            if resample:
                ch = rng.choice(idx, len(idx), replace=True)
                W[r] = np.bincount(ch, minlength=n_items) / len(idx)
            else:
                W[r, idx] = 1.0 / len(idx)
    return torch.tensor(WA, dtype=torch.float32), torch.tensor(WB, dtype=torch.float32)
rng = np.random.default_rng(5)
WA0, WB0 = make_weights(NP, list(range(NP)), NPT, False, rng)
WA1, WB1 = make_weights(NP, list(range(NP)), NBOOT, True, rng)
WA, WB = torch.cat([WA0, WA1]), torch.cat([WB0, WB1])
PA0, PB0 = make_weights(NPR, pair_clause, NPT, False, rng)
PA1, PB1 = make_weights(NPR, pair_clause, NBOOT, True, rng)
WpA, WpB = torch.cat([PA0, PA1]), torch.cat([PB0, PB1])

TA = torch.load(f"{V3_CACHE}/toks_A.pt"); TBk = torch.load(f"{V3_CACHE}/toks_B.pt")
def metrics_for_layer(l):
    mets = {"raw": None}
    if l not in COV_LAYERS:
        return mets, None
    T = torch.cat([TA["toks"][l], TBk["toks"][l]], 0).double()
    T = T - T.mean(0)
    S = (T.T @ T) / (T.shape[0] - 1)
    ev, V = torch.linalg.eigh(S); ev = ev.clamp_min(0); avg = ev.mean()
    for lam in (0.01, 0.1, 0.5):
        e = (1 - lam) * ev + lam * avg
        mets[f"white_l{lam}"] = (V @ torch.diag(e.rsqrt()) @ V.T).float()
    sd = S.diag().sqrt()
    mets["diag_white"] = torch.diag(1.0 / sd.clamp_min(1e-6 * sd.max())).float()
    q = sd < torch.quantile(sd, 0.10)
    mets["drop_quiet10"] = torch.diag((~q).double()).float()
    hi = sd > torch.quantile(sd, 0.99)
    mets["drop_top1var"] = torch.diag((~hi).double()).float()
    mets["gain_next_ln"] = torch.diag(NW[l + 1].abs()).float()
    return mets, {"top10_eig_frac": float(ev[-10:].sum() / ev.sum()), "quiet10_sd_ratio": float(sd[q].mean() / sd[~q].mean())}

per = {}; info = {}
for l in LAYERS:
    li = LI[l]
    mets, inf = metrics_for_layer(l)
    if inf: info[l] = inf
    for mn, Wm in mets.items():
        f = (lambda v: v) if Wm is None else (lambda v, Wm=Wm: v @ Wm)
        Xl = {c: f(v[:, li]) for c, v in X.items()}
        r = analyze(Xl, f(PD_c[:, li]), WA, WB, WpA, WpB, with_null=True)
        per.setdefault(mn, {})[l] = r
        if mn == "raw" and l in COV_LAYERS:
            per.setdefault("raw@cov", {})[l] = r
    print("layer", l, round(time.time() - t0, 1), flush=True)

BANDS3 = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-50": range(40, 51)}
summ = {}
for mn, d in per.items():
    summ[mn] = {}
    for bn, rr in BANDS3.items():
        ls = [l for l in rr if l in d]
        if not ls: continue
        summ[mn][bn] = {}
        keys = d[ls[0]].keys()
        for k in keys:
            arr = np.mean([d[l][k] for l in ls], 0)
            if k.startswith("pos|NULL"):
                continue
            summ[mn][bn][k] = [float(arr[:NPT].mean()), float(np.percentile(arr[NPT:], 2.5)), float(np.percentile(arr[NPT:], 97.5))]
        # null distribution of default position after projecting out k other-contrast dims
        for kk, pre in (("1", "NULL1_"), ("2", "NULL2_"), ("4", "NULL4_")):
            vals = np.array([np.mean([d[l][f"pos|{pre}{j}|default"][:NPT].mean() for l in ls]) for j in range(len(NULLPOOL) if kk == "1" else 20)])
            summ[mn][bn][f"nullpos_k{kk}|default"] = [float(vals.mean()), float(vals.min()), float(vals.max()), float(vals.std())]
pl = {mn: {l: {k: float(v[:NPT].mean()) for k, v in d[l].items() if not k.startswith("pos|NULL")} for l in d} for mn, d in per.items()}
json.dump({"bands": summ, "per_layer_point": pl, "info": info, "nboot": NBOOT, "npoint_splits": NPT, "loc": LOC},
          open(f"{V3_RES}/V3_persona_axis_traits_02c_crossfit_{LOC}.json", "w"))
print("done", round(time.time() - t0, 1))
