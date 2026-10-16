#!/usr/bin/env python3
# R1_IT_LENS 04 (local CPU): directional IT lens from the R1_03 fit checkpoint.
#   gbar_l(d) = mean over fit positions of the exact same-position VJP J_l(pos)^T d   (per class; "mix" = equal class weights)
# Outputs experiments/results/R1_04_dir_analysis.json and R1_04_gbar.pt (mix/chat/prose/class lenses for later jobs).
# Statistics: split-half (passage parity) reliability; CIs = bootstrap over the 98 random token-pair directions (dir-level
# claims) or over passages (pooled lens, 6 key directions; 400 resamples, stratified by source).
import glob, json, math, sys
import numpy as np
import torch

torch.set_num_threads(12)
R = "./experiments/results/"
CL = ["prose_enc", "prose_doc", "user", "asst", "self"]
BANDS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]
BN = [f"L{a}-{b}" for a, b in BANDS]
rng = np.random.default_rng(0)

ck = torch.load(R + "R1_03_fit_ckpt.pt", mmap=True, weights_only=False)
S, N = ck["S"], ck["N"].double()
print("strata counts", dict(zip([f"{c}{h}" for c in CL for h in (0, 1)], N.int().tolist())), "done", len(ck["done"]))
DD = torch.load(R + "R1_02_dirs.pt", weights_only=False)
names = DD["names"]; PBJ = DD["pb_jpp"]; PBL = DD["pb_jlens"]; DIRS = DD["dirs"]
K = len(names)
GROUPS = {"pron": [0], "noun": [1], "she_he": [2], "woman_man": [3], "gender4": [0, 1, 2, 3], "ctrl6": list(range(4, 10)),
          "iso4": list(range(10, 14)), "unit16": list(range(14, 30)), "tokpair98": list(range(30, K))}

v6 = torch.load(R + "V6_quiet_dims_stats.pt", weights_only=False)
std = v6["std_all"].float()
med = std.median(dim=1, keepdim=True).values
MASK = {}
MASK["q10"] = (std <= torch.quantile(std, 0.1, dim=1, keepdim=True)).float()
MASK["loud"] = 1 - MASK["q10"]
MASK["deep"] = (std < 0.05 * med).float()
MASK["mod"] = MASK["q10"] * (1 - MASK["deep"])
pers = np.load(R + "V6_quiet_dims_pers181.npy")
MASK["pers"] = torch.zeros(60, 5376); MASK["pers"][:, torch.as_tensor(pers)] = 1.0
MASK["all"] = torch.ones(60, 5376)


def cmean(c, h=None):
    i = CL.index(c)
    if h is None:
        return (S[2 * i] + S[2 * i + 1]) / (N[2 * i] + N[2 * i + 1]).float()
    return S[2 * i + h] / N[2 * i + h].float()


def mcos(a, b, m):
    """cosine over coordinates where m==1; a,b [K,60,D] (or broadcastable), m [60,D] -> [K,60]"""
    num = (a * b * m).sum(-1)
    den = ((a * a * m).sum(-1) * (b * b * m).sum(-1)).clamp_min(1e-300).sqrt()
    return num / den


def share(a, m):
    return (a * a * m).sum(-1) / (a * a).sum(-1).clamp_min(1e-300)


def band(x):  # x [K,60] -> [K,6]
    return np.stack([x[:, a:b + 1].mean(1) for a, b in BANDS], 1)


def summarize(x, groups=GROUPS):
    """x: np [K,60] -> {group: {band: mean}} plus direction-bootstrap CI for multi-direction groups"""
    xb = band(x)
    out = {}
    for g, idx in groups.items():
        v = xb[idx]
        d = {"mean": v.mean(0).round(4).tolist()}
        if len(idx) >= 4:
            bs = np.stack([v[rng.integers(0, len(idx), len(idx))].mean(0) for _ in range(1000)], 0)
            d["ci"] = np.percentile(bs, [2.5, 97.5], axis=0).round(4).T.tolist()
            d["q25_q75"] = np.percentile(v, [25, 75], axis=0).round(3).T.tolist()
        out[g] = d
    return out


print("building lenses...", flush=True)
CM = {c: cmean(c) for c in CL}
CH = {(c, h): cmean(c, h) for c in CL for h in (0, 1)}
LENS = {
    "mix": sum(CM.values()) / len(CL),
    "chat": (CM["user"] + CM["asst"] + CM["self"]) / 3,
    "prose": (CM["prose_enc"] + CM["prose_doc"]) / 2,
    "asstmode": (CM["asst"] + CM["self"]) / 2,
}
HALF = {"mix": [sum(CH[(c, h)] for c in CL) / len(CL) for h in (0, 1)],
        "chat": [(CH[("user", h)] + CH[("asst", h)] + CH[("self", h)]) / 3 for h in (0, 1)],
        "prose": [(CH[("prose_enc", h)] + CH[("prose_doc", h)]) / 2 for h in (0, 1)]}
res = {"names": names, "counts": N.tolist(), "bands": BN, "n_done": len(ck["done"])}

# ---------------------------------------------------------------- reliability
print("reliability...", flush=True)
rel = {}
for v in ("mix", "chat", "prose"):
    for mk in ("all", "loud", "q10", "deep"):
        r = mcos(HALF[v][0], HALF[v][1], MASK[mk]).numpy()
        rel[(v, mk)] = r
        res.setdefault("splithalf", {}).setdefault(v, {})[mk] = summarize(r)
for c in CL:
    r = mcos(CH[(c, 0)], CH[(c, 1)], MASK["all"]).numpy()
    res.setdefault("splithalf_class", {})[c] = summarize(r, {"gender4": [0, 1, 2, 3], "tokpair98": GROUPS["tokpair98"]})


def sb_ceiling(r):  # Spearman-Brown: reliability of the full-sample mean, as a ceiling for cos with a noiseless target
    r = np.clip(r, 1e-6, None)
    return np.sqrt(2 * r / (1 + r))


# ---------------------------------------------------------------- agreement with base lenses
print("agreement...", flush=True)
agree = {}
for v in ("mix", "chat", "prose", "asstmode"):
    for lensname, PB in (("jpp", PBJ), ("jlens", PBL)):
        for mk in ("all", "loud", "q10", "deep", "mod", "pers"):
            c = mcos(LENS[v], PB, MASK[mk]).numpy()
            agree[(v, lensname, mk)] = c
            res.setdefault("cos_vs_base", {}).setdefault(v, {}).setdefault(lensname, {})[mk] = summarize(c)
        if v in ("mix", "chat", "prose"):
            for mk in ("all", "loud"):
                dis = agree[(v, lensname, mk)] / sb_ceiling(rel[(v, mk)])
                res.setdefault("cos_vs_base_disatt", {}).setdefault(v, {}).setdefault(lensname, {})[mk] = summarize(dis)
# chat vs prose IT lenses, and the two base lenses with each other
for mk in ("all", "loud", "q10", "deep"):
    res.setdefault("cos_chat_prose", {})[mk] = summarize(mcos(LENS["chat"], LENS["prose"], MASK[mk]).numpy())
    res.setdefault("cos_jpp_jlens", {})[mk] = summarize(mcos(PBJ, PBL, MASK[mk]).numpy())
    res.setdefault("cos_asst_user", {})[mk] = summarize(mcos(CM["asst"], CM["user"], MASK[mk]).numpy())
    res.setdefault("cos_self_asst", {})[mk] = summarize(mcos(CM["self"], CM["asst"], MASK[mk]).numpy())
    res.setdefault("cos_enc_doc", {})[mk] = summarize(mcos(CM["prose_enc"], CM["prose_doc"], MASK[mk]).numpy())
for c in CL:
    for mk in ("all", "loud", "q10"):
        res.setdefault("cos_class_vs_jpp", {}).setdefault(c, {})[mk] = summarize(mcos(CM[c], PBJ, MASK[mk]).numpy(),
                                                                                 {"pron": [0], "gender4": [0, 1, 2, 3], "tokpair98": GROUPS["tokpair98"]})
# energy shares
for mk in ("q10", "deep", "pers"):
    res.setdefault("energy_share", {}).setdefault("it_mix", {})[mk] = summarize(share(LENS["mix"], MASK[mk]).numpy())
    res.setdefault("energy_share", {}).setdefault("jpp", {})[mk] = summarize(share(PBJ, MASK[mk]).numpy())
    res.setdefault("energy_share", {}).setdefault("jlens", {})[mk] = summarize(share(PBL, MASK[mk]).numpy())
# Where does the disagreement live?  ||u - v||^2 for unit-normalised u (IT) and v (J++), split by mask
u = LENS["mix"] / LENS["mix"].norm(dim=-1, keepdim=True).clamp_min(1e-300)
w = PBJ / PBJ.norm(dim=-1, keepdim=True).clamp_min(1e-300)
dd = (u - w) ** 2
res["disagreement_share_q10"] = summarize(((dd * MASK["q10"]).sum(-1) / dd.sum(-1)).numpy())
# Inside the quiet set: do IT and J++ agree on WHICH quiet coordinates carry the covector (per-dim energy profile, Spearman)
# while disagreeing on the signed direction? Also the energy-weighted sign agreement.
def spearman_rows(a, b):
    ra = a.argsort(-1).argsort(-1).double(); rb = b.argsort(-1).argsort(-1).double()
    ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-30)
prof_it_j, prof_j_jl, sign_it_j, sign_j_jl = (np.zeros((K, 60)) for _ in range(4))
for l in range(60):
    qi = MASK["q10"][l].bool()
    gi, gj, gl = LENS["mix"][:, l, qi], PBJ[:, l, qi], PBL[:, l, qi]
    prof_it_j[:, l] = spearman_rows(gi.pow(2), gj.pow(2)).numpy()
    prof_j_jl[:, l] = spearman_rows(gj.pow(2), gl.pow(2)).numpy()
    w = (gi.abs() * gj.abs()); sign_it_j[:, l] = ((torch.sign(gi) == torch.sign(gj)).float() * w).sum(-1).div(w.sum(-1).clamp_min(1e-30)).numpy()
    w = (gj.abs() * gl.abs()); sign_j_jl[:, l] = ((torch.sign(gj) == torch.sign(gl)).float() * w).sum(-1).div(w.sum(-1).clamp_min(1e-30)).numpy()
res["q10_energy_profile_spearman_it_jpp"] = summarize(prof_it_j)
res["q10_energy_profile_spearman_jpp_jlens"] = summarize(prof_j_jl)
res["q10_weighted_sign_agree_it_jpp"] = summarize(sign_it_j)
res["q10_weighted_sign_agree_jpp_jlens"] = summarize(sign_j_jl)
# norm ratio (IT gbar vs J++ pullback): does J++ have the right magnitude?
res["norm_ratio_it_over_jpp"] = summarize((LENS["mix"].norm(dim=-1) / PBJ.norm(dim=-1).clamp_min(1e-300)).numpy())
res["norm_ratio_it_over_jpp_loud"] = summarize(((LENS["mix"] ** 2 * MASK["loud"]).sum(-1).sqrt() /
                                                 (PBJ ** 2 * MASK["loud"]).sum(-1).sqrt().clamp_min(1e-300)).numpy())
res["norm_ratio_it_over_jpp_q10"] = summarize(((LENS["mix"] ** 2 * MASK["q10"]).sum(-1).sqrt() /
                                                (PBJ ** 2 * MASK["q10"]).sum(-1).sqrt().clamp_min(1e-300)).numpy())

# ---------------------------------------------------------------- per-position records (coherence, single-position cosines)
print("per-position records...", flush=True)
recs = []
for f in sorted(glob.glob(R + "R1_03_fit_job*_*.pt")):
    d = torch.load(f, weights_only=False)
    recs += d["recs"]
gn = torch.stack([r["gn"] for r in recs]).double()  # [P,K,60]
cls = np.array([r["cls"] for r in recs])
res["n_positions"] = len(recs)
coh = {}
for c in CL:
    sel = torch.as_tensor(cls == c)
    if sel.sum() == 0:
        continue
    coh[c] = (CM[c].norm(dim=-1).double() / gn[sel].mean(0)).numpy()
    res.setdefault("coherence", {})[c] = summarize(coh[c], {"pron": [0], "gender4": [0, 1, 2, 3], "tokpair98": GROUPS["tokpair98"]})
cj = torch.stack([r["cj"].float() for r in recs])
qs = torch.stack([r["qs"].float() for r in recs])
for c in CL:
    sel = torch.as_tensor(cls == c)
    res.setdefault("single_pos_cos_jpp", {})[c] = summarize(cj[sel].mean(0).numpy(), {"pron": [0], "tokpair98": GROUPS["tokpair98"]})
    res.setdefault("single_pos_q10_share", {})[c] = summarize(qs[sel].mean(0).numpy(), {"pron": [0], "tokpair98": GROUPS["tokpair98"]})

# ---------------------------------------------------------------- passage bootstrap (pooled lens, key directions)
print("passage bootstrap...", flush=True)
PP, SRC = [], []
for f in sorted(glob.glob(R + "R1_03_fit_job*_*.pt")):
    d = torch.load(f, weights_only=False)
    for pid, v in d["pp"].items():
        PP.append((v["sum"], v["n"])); SRC.append("chat" if v["src"].startswith("chat") else v["src"])
KEY = [0, 1, 2, 3, 10, 11]
SRC = np.array(SRC)
PS = torch.stack([p[0] for p in PP]).double()   # [P,6,60,D]
PN = torch.tensor([p[1] for p in PP], dtype=torch.float64)
B = 400
Wt = np.zeros((B, len(PP)))
for s in np.unique(SRC):
    idx = np.where(SRC == s)[0]
    for b in range(B):
        np.add.at(Wt[b], rng.choice(idx, len(idx)), 1.0)
Wt = torch.tensor(Wt)
boot = {}
for mk in ("all", "loud", "q10"):
    boot[mk] = np.zeros((B, len(KEY), 60))
pooled_pt = (PS.sum(0) / PN.sum()).float()
for l in range(60):
    G = PS[:, :, l, :]                    # [P,6,D]
    num = torch.einsum("bp,pkd->bkd", Wt, G)
    den = (Wt @ PN)[:, None, None]
    gl = (num / den).float()              # [B,6,D]
    pb = PBJ[KEY, l, :]                   # [6,D]
    for mk in ("all", "loud", "q10"):
        m = MASK[mk][l]
        boot[mk][:, :, l] = ((gl * pb * m).sum(-1) / ((gl * gl * m).sum(-1) * (pb * pb * m).sum(-1)).sqrt()).numpy()
res["pooled_key_cos_jpp"] = {}
for mk in ("all", "loud", "q10"):
    pt = mcos(pooled_pt, PBJ[KEY], MASK[mk]).numpy()
    bb = np.stack([np.stack([boot[mk][:, :, a:b + 1].mean(2) for a, b in BANDS], 2)], 0)[0]  # [B,6,6]
    ci = np.percentile(bb, [2.5, 97.5], axis=0)
    res["pooled_key_cos_jpp"][mk] = {names[k]: {"mean": band(pt)[i].round(4).tolist(),
                                                "ci": np.stack([ci[0][i], ci[1][i]], 1).round(4).tolist()} for i, k in enumerate(KEY)}

# ---------------------------------------------------------------- held-out validation with E3's independent VJPs
print("E3 held-out...", flush=True)
E3G, E3M = [], []
for k in range(3):
    d = torch.load(R + f"E3_lens_validation_vjp_chunk{k}.pt", weights_only=False)
    E3G.append(d["G"].float()); E3M += d["meta"]
E3G = torch.cat(E3G, 0)   # [144,4,60,D]  dirs = pron, noun, rand1, rand2 (= R1 dirs 0,1,10,11)
src = np.array([m["src"] for m in E3M]); grp = np.array([m["group"] for m in E3M])
E3K = [0, 1, 10, 11]
held = {}
for sub, sel in (("all", np.ones(len(src), bool)), ("chat", src == "chat"), ("prose", src == "prose")):
    g3 = E3G[torch.as_tensor(sel)].mean(0)                  # [4,60,D]
    ug = np.unique(grp[sel]); h0 = np.isin(grp, ug[::2]) & sel; h1 = np.isin(grp, ug[1::2]) & sel
    r3 = mcos(E3G[torch.as_tensor(h0)].mean(0), E3G[torch.as_tensor(h1)].mean(0), MASK["all"]).numpy()
    lensv = {"all": "mix", "chat": "chat", "prose": "prose"}[sub]
    for mk in ("all", "loud", "q10"):
        c_it = mcos(g3, LENS[lensv][E3K], MASK[mk]).numpy()
        c_jpp = mcos(g3, PBJ[E3K], MASK[mk]).numpy()
        c_jl = mcos(g3, PBL[E3K], MASK[mk]).numpy()
        held.setdefault(sub, {})[mk] = {"cos_E3_vs_R1": band(c_it).round(4).tolist(),
                                        "cos_E3_vs_jpp": band(c_jpp).round(4).tolist(),
                                        "cos_E3_vs_jlens": band(c_jl).round(4).tolist()}
    held[sub]["E3_splithalf"] = band(r3).round(4).tolist()
    held[sub]["ceiling_sqrt_relE3_relR1"] = band(np.sqrt(sb_ceiling(r3) ** 2 * sb_ceiling(rel[(lensv if lensv != "mix" else "mix", "all")][E3K]) ** 2)).round(4).tolist()
res["heldout_E3"] = held
res["heldout_E3_dirs"] = [names[k] for k in E3K]

json.dump(res, open(R + "R1_04_dir_analysis.json", "w"), indent=1)
torch.save({"names": names, "mix": LENS["mix"], "chat": LENS["chat"], "prose": LENS["prose"], "asstmode": LENS["asstmode"],
            "mix_h0": HALF["mix"][0], "mix_h1": HALF["mix"][1], "class": CM, "counts": N},
           R + "R1_04_gbar.pt")


# ---------------------------------------------------------------- print headline tables
def row(d, g="tokpair98"):
    m = d[g]["mean"]; ci = d[g].get("ci")
    if ci:
        return "  ".join(f"{x:+.2f}[{a:+.2f},{b:+.2f}]" for x, (a, b) in zip(m, ci))
    return "  ".join(f"{x:+.2f}" for x in m)


print("bands", BN)
for v in ("mix", "chat", "prose"):
    print(f"split-half {v} all  tp:", row(res["splithalf"][v]["all"]), " pron:", row(res["splithalf"][v]["all"], "pron"))
    print(f"split-half {v} q10  tp:", row(res["splithalf"][v]["q10"]))
for v in ("mix", "chat", "prose"):
    for mk in ("all", "loud", "q10"):
        print(f"cos IT[{v}] vs J++ [{mk}] tp:", row(res["cos_vs_base"][v]["jpp"][mk]), " pron:", row(res["cos_vs_base"][v]["jpp"][mk], "pron"))
for mk in ("all", "loud", "q10"):
    print(f"cos J++ vs JLens [{mk}] tp:", row(res["cos_jpp_jlens"][mk]))
    print(f"cos IT chat vs prose [{mk}] tp:", row(res["cos_chat_prose"][mk]))
print("held-out E3 (all) cos_E3_vs_R1 [all]:", held["all"]["all"]["cos_E3_vs_R1"])
print("held-out E3 (all) cos_E3_vs_jpp [all]:", held["all"]["all"]["cos_E3_vs_jpp"])
print("held-out ceiling:", held["all"]["ceiling_sqrt_relE3_relR1"])
