# V3_persona_axis_traits_02 (offline, CPU; run with .venv/bin/python). Geometry of persona / trait / prose axes and the
# AI-vs-human offset at assistant self tokens, in raw, whitened (token covariance, shrinkage), diag-whitened,
# quiet-dims-removed, top-variance-dims-removed and gain (next input_layernorm) metrics. Positions on the woman(1)/man(0)
# persona scale with passage-bootstrap CIs (axes re-estimated inside every bootstrap resample).
# Predictions: see V3_persona_axis_traits_01_collect.py header (H1-H4).
# NOTE: this is the IN-SAMPLE version (axis and poles from the same passages). It biases positions toward 0.5 when the
# axis is noisy (whitened metrics). Superseded by V3_persona_axis_traits_02c_crossfit.py; kept for reference.
import json, math, sys, time
import numpy as np, torch
torch.set_num_threads(12)
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
t0 = time.time()
A = torch.load(f"{V3_CACHE}/self_A.pt"); Bd = torch.load(f"{V3_CACHE}/self_B.pt"); PR = torch.load(f"{V3_CACHE}/prose.pt")
NW = torch.load(f"{V3_ROOT}/experiments/results/E4_causal_cache/normweights.pt")["input_layernorm"].double()
LOC = sys.argv[1] if len(sys.argv) > 1 else "self"
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LI = {l: i for i, l in enumerate(LAYERS)}

def gather(S, key):
    out = {}
    for c in dict.fromkeys(it["cond"] for it in S["items"]):
        d = {it["p"]: i for i, it in enumerate(S["items"]) if it["cond"] == c}
        out[c] = S[key][[d[p] for p in range(NP)]]
    return out
X = {**gather(A, LOC), **gather(Bd, LOC)}            # cond -> [24, 31, D]
X = {c: v.double() for c, v in X.items()}
CONDS = list(X.keys())
# prose: 72 pairs (36 clauses x 2 templates), F/M
pit = PR["items"]
pidx = {(it["tpl"], it["i"], it["cond"]): k for k, it in enumerate(pit)}
pairs = [(pidx[(t, i, "F")], pidx[(t, i, "M")]) for t in ("A", "B") for i in range(36)]
PF_c = PR["cont"][[a for a, b in pairs]].double(); PM_c = PR["cont"][[b for a, b in pairs]].double()
PF_l = PR["last"][[a for a, b in pairs]].double(); PM_l = PR["last"][[b for a, b in pairs]].double()
NPR = len(pairs)

GPW = [w for w, m in PAIRS["persona"]]; GPM = [m for w, m in PAIRS["persona"]]
TW = [a for a, b in PAIRS["trait"]]; TB = [b for a, b in PAIRS["trait"]]
REPORT = ["default", "nameless_AI", "Gemma", "AI_plain", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2",
          "TP_warm(pooled)", "TP_blunt(pooled)", "RL_serve(pooled)", "RL_solo(pooled)", "CT1_org", "CT1_dis", "GP1_W", "GP1_M", "W(pooled)", "M(pooled)"]

def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def proj_out(u, basis):
    """Remove span(basis rows) from u (batched over leading dim). basis: [..., k, D]."""
    Q, _ = torch.linalg.qr(basis.transpose(-1, -2))  # [..., D, k]
    return u - (Q @ (Q.transpose(-1, -2) @ u.unsqueeze(-1))).squeeze(-1)

def analyze_layer(Xl, PFc, PMc, PFl, PMl, Wb, Wp, ctrl_pass_dirs):
    """Xl: cond -> [24, D] (already metric-transformed). Wb: [B,24] bootstrap weights (row 0 = full sample),
    Wp: [B,NPR] prose bootstrap weights. Returns dict of [B] arrays."""
    M = {c: Wb @ v for c, v in Xl.items()}                     # [B, D]
    M["W(pooled)"] = sum(M[c] for c in GPW) / 4; M["M(pooled)"] = sum(M[c] for c in GPM) / 4
    M["TP_warm(pooled)"] = sum(M[c] for c in TW) / 4; M["TP_blunt(pooled)"] = sum(M[c] for c in TB) / 4
    M["RL_serve(pooled)"] = (M["RL1_serve"] + M["RL2_serve"]) / 2; M["RL_solo(pooled)"] = (M["RL1_solo"] + M["RL2_solo"]) / 2
    W_, M_ = M["W(pooled)"], M["M(pooled)"]
    ax = {}
    ax["persona"] = W_ - M_
    for k, (a, b) in enumerate(PAIRS["persona"]):
        ax[f"persona_{k+1}"] = M[a] - M[b]
    ax["trait"] = M["TP_warm(pooled)"] - M["TP_blunt(pooled)"]
    for k, (a, b) in enumerate(PAIRS["trait"]):
        ax[f"trait_{k+1}"] = M[a] - M[b]
    ax["ctrl1"] = M["CT1_org"] - M["CT1_dis"]; ax["ctrl2"] = M["CT2_city"] - M["CT2_rural"]
    ax["role"] = M["RL_serve(pooled)"] - M["RL_solo(pooled)"]
    ax["role_1"] = M["RL1_serve"] - M["RL1_solo"]; ax["role_2"] = M["RL2_serve"] - M["RL2_solo"]
    ax["prose_cont"] = Wp @ (PFc - PMc); ax["prose_last"] = Wp @ (PFl - PMl)
    nh = (M["neutral_human"] + M["neutral_human2"]) / 2
    ax["AIoff(default-neutral_human)"] = M["default"] - M["neutral_human"]
    ax["AIoff(default-nh_pooled)"] = M["default"] - nh
    ax["AIoff(nameless-nh_pooled)"] = M["nameless_AI"] - nh
    ax["AIoff(AI_blunt-nh_pooled)"] = M["AI_blunt"] - nh
    ax["AI_warm-AI_blunt"] = M["AI_warm"] - M["AI_blunt"]
    ax["Gemma-nameless"] = M["Gemma"] - M["nameless_AI"]
    out = {}
    # cosines between axes
    names = ["persona", "trait", "ctrl1", "ctrl2", "role", "prose_cont", "prose_last", "AIoff(default-neutral_human)",
             "AIoff(default-nh_pooled)", "AIoff(nameless-nh_pooled)", "AIoff(AI_blunt-nh_pooled)", "AI_warm-AI_blunt", "Gemma-nameless"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"cos|{a}|{b}"] = torch.nn.functional.cosine_similarity(ax[a], ax[b], dim=-1)
    # reliability: mean pairwise cos between phrasing-specific axes
    for fam, k in (("persona", 4), ("trait", 4)):
        cs = [torch.nn.functional.cosine_similarity(ax[f"{fam}_{i}"], ax[f"{fam}_{j}"], dim=-1) for i in range(1, k + 1) for j in range(i + 1, k + 1)]
        out[f"rel|{fam}_phrasing_meancos"] = sum(cs) / len(cs)
    out["rel|role_1_2"] = torch.nn.functional.cosine_similarity(ax["role_1"], ax["role_2"], dim=-1)
    out["rel|ctrl1_ctrl2"] = torch.nn.functional.cosine_similarity(ax["ctrl1"], ax["ctrl2"], dim=-1)
    # cross-phrasing persona-trait cos (persona_i vs trait_j, removes shared-noise inflation)
    cs = [torch.nn.functional.cosine_similarity(ax[f"persona_{i}"], ax[f"trait_{j}"], dim=-1) for i in range(1, 5) for j in range(1, 5)]
    out["cos|persona_i|trait_j(mean16)"] = sum(cs) / 16
    out["norm|persona"] = ax["persona"].norm(dim=-1); out["norm|trait"] = ax["trait"].norm(dim=-1)
    out["norm|AIoff(default-neutral_human)"] = ax["AIoff(default-neutral_human)"].norm(dim=-1)
    out["norm|role"] = ax["role"].norm(dim=-1)
    # positions on variants of the persona axis (scale: M(pooled)=0, W(pooled)=1), and E4-replication with GP1 poles
    tr_sub = torch.stack([ax[f"trait_{k}"] for k in range(1, 5)], -2)
    ctrl_sub = torch.stack([ax["ctrl1"], ax["ctrl2"], ax["role_1"], ax["role_2"]], -2)
    ctrlonly_sub = torch.stack([ax["ctrl1"], ax["ctrl2"]], -2)
    role_sub = torch.stack([ax["role_1"], ax["role_2"]], -2)
    pc = ctrl_pass_dirs.unsqueeze(0).expand(ax["persona"].shape[0], -1, -1)
    variants = {
        "persona": ax["persona"],
        "persona_GP1only": ax["persona_1"],
        "persona_perp_trait": proj_out(ax["persona"], ax["trait"].unsqueeze(-2)),
        "persona_perp_trait4": proj_out(ax["persona"], tr_sub),
        "persona_perp_role": proj_out(ax["persona"], ax["role"].unsqueeze(-2)),
        "persona_perp_role2": proj_out(ax["persona"], role_sub),
        "persona_perp_trait_role": proj_out(ax["persona"], torch.stack([ax["trait"], ax["role"]], -2)),
        "persona_perp_trait4_role2": proj_out(ax["persona"], torch.cat([tr_sub, role_sub], -2)),
        "persona_perp_ctrl2": proj_out(ax["persona"], ctrlonly_sub),
        "persona_perp_ctrl4(ctrl+role)": proj_out(ax["persona"], ctrl_sub),
        "persona_perp_passage4(control)": proj_out(ax["persona"], pc),
        "persona_perp_prose": proj_out(ax["persona"], ax["prose_cont"].unsqueeze(-2)),
        "persona_perp_AIoff": proj_out(ax["persona"], ax["AIoff(default-nh_pooled)"].unsqueeze(-2)),
        "trait_component_of_persona": (ax["persona"] * unit(ax["trait"])).sum(-1, keepdim=True) * unit(ax["trait"]),
        "prose_cont": ax["prose_cont"], "prose_last": ax["prose_last"],
        "trait_axis(on_W/M_scale)": ax["trait"],
    }
    for vn, u in variants.items():
        u = unit(u)
        if vn == "persona_GP1only":
            lo, hi = M["GP1_M"], M["GP1_W"]
        else:
            lo, hi = M_, W_
        den = ((hi - lo) * u).sum(-1)
        for c in REPORT:
            out[f"pos|{vn}|{c}"] = ((M[c] - lo) * u).sum(-1) / den
        out[f"retained|{vn}"] = torch.nn.functional.cosine_similarity(u, ax["persona"], dim=-1)
    # positions on the trait axis, scale blunt(0) .. warm(1)
    u = unit(ax["trait"]); lo, hi = M["TP_blunt(pooled)"], M["TP_warm(pooled)"]
    den = ((hi - lo) * u).sum(-1)
    for c in REPORT:
        out[f"tpos|trait|{c}"] = ((M[c] - lo) * u).sum(-1) / den
    # decomposition of AI offset projection on unit persona axis into trait-direction part and remainder
    up = unit(ax["persona"]); ut = unit(ax["trait"])
    off = ax["AIoff(default-nh_pooled)"]
    gap = ((W_ - M_) * up).sum(-1)
    tot = (off * up).sum(-1)
    via_t = (off * ut).sum(-1) * (up * ut).sum(-1)
    out["dec|AIoff_on_persona/gap"] = tot / gap
    out["dec|via_trait/gap"] = via_t / gap
    out["dec|rest/gap"] = (tot - via_t) / gap
    # regression of persona axis on [prose_cont, trait, role] (least squares, full design)
    Bm = torch.stack([ax["prose_cont"], ax["trait"], ax["role"]], -1)       # [B, D, 3]
    coef = torch.linalg.lstsq(Bm, ax["persona"].unsqueeze(-1)).solution     # [B,3,1]
    fit = (Bm @ coef).squeeze(-1)
    out["reg|R2_persona~prose+trait+role"] = 1 - ((ax["persona"] - fit).pow(2).sum(-1) / ax["persona"].pow(2).sum(-1))
    for j, nm in enumerate(["prose", "trait", "role"]):
        out[f"reg|coef_{nm}"] = coef[:, j, 0]
    Bm2 = torch.stack([ax["prose_cont"]], -1)
    fit2 = (Bm2 @ torch.linalg.lstsq(Bm2, ax["persona"].unsqueeze(-1)).solution).squeeze(-1)
    out["reg|R2_persona~prose"] = 1 - ((ax["persona"] - fit2).pow(2).sum(-1) / ax["persona"].pow(2).sum(-1))
    return {k: v.numpy() for k, v in out.items()}

rng = np.random.default_rng(7)
Wb = np.zeros((NBOOT, NP)); Wp = np.zeros((NBOOT, NPR))
Wb[0] = 1.0 / NP; Wp[0] = 1.0 / NPR
for b in range(1, NBOOT):
    Wb[b] = np.bincount(rng.integers(0, NP, NP), minlength=NP) / NP
    Wp[b] = np.bincount(rng.integers(0, NPR, NPR), minlength=NPR) / NPR
Wb = torch.tensor(Wb); Wp = torch.tensor(Wp)

# token covariance samples
TA = torch.load(f"{V3_CACHE}/toks_A.pt"); TBk = torch.load(f"{V3_CACHE}/toks_B.pt")

def metrics_for_layer(l):
    """Return dict metric_name -> linear map function (x [...,D] -> [...,D']) for layer l."""
    mets = {"raw": None}
    if l in COV_LAYERS:
        T = torch.cat([TA["toks"][l], TBk["toks"][l]], 0).double()
        T = T - T.mean(0)
        S = (T.T @ T) / (T.shape[0] - 1)
        ev, V = torch.linalg.eigh(S)
        ev = ev.clamp_min(0)
        avg = ev.mean()
        for lam in (0.01, 0.1, 0.5):
            e = (1 - lam) * ev + lam * avg
            Wm = V @ torch.diag(e.rsqrt()) @ V.T
            mets[f"white_l{lam}"] = Wm
        sd = S.diag().sqrt()
        mets["diag_white"] = torch.diag(1.0 / sd.clamp_min(1e-6 * sd.max()))
        q = sd < torch.quantile(sd, 0.10)
        mets["drop_quiet10"] = torch.diag((~q).double())
        hi = sd > torch.quantile(sd, 0.99)
        mets["drop_top1var"] = torch.diag((~hi).double())
        mets["gain_next_ln"] = torch.diag(NW[l + 1].abs())
        info = {"top_eig_frac": float(ev[-10:].sum() / ev.sum()), "quiet_sd_ratio": float(sd[q].mean() / sd[~q].mean()),
                "ntok": T.shape[0]}
        return mets, info
    return mets, None

res = {"layers": LAYERS, "nboot": NBOOT, "loc": LOC, "per_layer": {}, "info": {}}
for l in LAYERS:
    li = LI[l]
    mets, info = metrics_for_layer(l)
    if info: res["info"][l] = info
    # control: 4 passage-content directions (differences between passage means of neutral_human, fixed pairs)
    nhv = X["neutral_human"][:, li]
    pdirs = torch.stack([nhv[2 * k] - nhv[2 * k + 1] for k in range(4)], 0)
    for mn, Wm in mets.items():
        f = (lambda v: v) if Wm is None else (lambda v, Wm=Wm: v @ Wm)
        Xl = {c: f(v[:, li]) for c, v in X.items()}
        r = analyze_layer(Xl, f(PF_c[:, li]), f(PM_c[:, li]), f(PF_l[:, li]), f(PM_l[:, li]), Wb, Wp, f(pdirs))
        res["per_layer"].setdefault(mn, {})[l] = r
    print("layer", l, "metrics", list(mets), round(time.time() - t0, 1), flush=True)

# J++ readout positions (self tokens): scale M(pooled)=0 .. W(pooled)=1, and trait gap as fraction of gender gap
key = f"{LOC}_jpp_pron" if LOC in ("self", "bg") else None
jres = {}
for kind in ("jpp", "naive"):
    R = {**gather(A, f"{LOC}_{kind}_pron"), **gather(Bd, f"{LOC}_{kind}_pron")}
    R = {c: v.double() for c, v in R.items()}
    R["W(pooled)"] = sum(R[c] for c in GPW) / 4; R["M(pooled)"] = sum(R[c] for c in GPM) / 4
    R["TP_warm(pooled)"] = sum(R[c] for c in TW) / 4; R["TP_blunt(pooled)"] = sum(R[c] for c in TB) / 4
    R["RL_serve(pooled)"] = (R["RL1_serve"] + R["RL2_serve"]) / 2; R["RL_solo(pooled)"] = (R["RL1_solo"] + R["RL2_solo"]) / 2
    Rb = {c: (Wb @ v) for c, v in R.items()}  # [B, 31]
    jres[kind] = {}
    for c in REPORT:
        jres[kind][c] = ((Rb[c] - Rb["M(pooled)"]) / (Rb["W(pooled)"] - Rb["M(pooled)"])).numpy()
res["jreadout"] = jres

BANDS3 = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-50": range(40, 51)}
def band_summary(per_layer_dict, layers_avail):
    out = {}
    keys = next(iter(per_layer_dict.values())).keys()
    for bn, r in BANDS3.items():
        ls = [l for l in r if l in layers_avail]
        if not ls: continue
        out[bn] = {}
        for k in keys:
            arr = np.mean([per_layer_dict[l][k] for l in ls], 0)
            out[bn][k] = [float(arr[0]), float(np.percentile(arr[1:], 2.5)), float(np.percentile(arr[1:], 97.5))]
    return out
summ = {mn: band_summary(d, list(d.keys())) for mn, d in res["per_layer"].items()}
jsum = {}
for kind, d in jres.items():
    jsum[kind] = {}
    for bn, r in BANDS3.items():
        idx = [LI[l] for l in r]
        jsum[kind][bn] = {c: [float(v[0, idx].mean()), float(np.percentile(v[1:, idx].mean(1), 2.5)), float(np.percentile(v[1:, idx].mean(1), 97.5))] for c, v in d.items()}
perlayer_full = {mn: {l: {k: float(v[0]) for k, v in d[l].items()} for l in d} for mn, d in res["per_layer"].items()}
json.dump({"bands": summ, "jreadout_bands": jsum, "per_layer_fullsample": perlayer_full, "info": res["info"], "nboot": NBOOT, "loc": LOC},
          open(f"{V3_RES}/V3_persona_axis_traits_02_geometry_{LOC}.json", "w"))
print("done", round(time.time() - t0, 1))
