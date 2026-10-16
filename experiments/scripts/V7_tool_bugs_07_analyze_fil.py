# V7_tool_bugs_07_analyze_fil (OFFLINE): filament causality summary from jobs 03 (+06 gradients if present).
import glob, json
import numpy as np
RES = "./experiments/results/"
rng = np.random.default_rng(0)


def rankdata(x):
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(len(x), dtype=np.intp); inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def sp(x, y):
    if len(x) < 3:
        return float("nan")
    a, b = rankdata(x), rankdata(y)
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def boot_sp(x, y, groups=None, n=2000):
    x, y = np.asarray(x), np.asarray(y)
    est = sp(x, y)
    bs = []
    if groups is None:
        for _ in range(n):
            i = rng.integers(0, len(x), len(x))
            bs.append(sp(x[i], y[i]))
    else:
        ug = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(n):
            gs = rng.choice(ug, len(ug))
            i = np.concatenate([idx[g] for g in gs])
            bs.append(sp(x[i], y[i]))
    return {"rho": est, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)), "n": int(len(x))}


files = sorted(glob.glob(RES + "V7_tool_bugs_03_fil_*.npz"))
D = {}
for f in files:
    nm = f.split("V7_tool_bugs_03_fil_")[1][:-4]
    z = np.load(f)
    D[nm] = {k: z[k] for k in z.files}
names = list(D)
cat = lambda key, mask_fn=None: np.concatenate([D[n][key][mask_fn(D[n])] if mask_fn else D[n][key] for n in names])
fil = lambda d: d["is_fil"].astype(bool)
allm = lambda d: np.ones(len(d["l"]), bool)
out = {"prompts": names, "n_prompts": len(names)}
noise_g = max(float(D[n]["noise_g"].max()) for n in names); noise_a = max(float(D[n]["noise_a"].max()) for n in names)
out["noise_floor"] = {"clean_row_max_abs_diff_g": noise_g, "clean_row_max_abs_diff_a": noise_a}

for scope, mfn in (("tool_filaments", fil), ("all_candidates", allm)):
    grp = np.concatenate([np.full(int(mfn(D[n]).sum()), i) for i, n in enumerate(names)])
    X = {k: cat(k, mfn) for k in ("C_tool", "dot", "D_gate", "D_act", "dg", "da", "a_src", "g0", "a0", "l", "lp", "is_fil")}
    s = {"n": int(len(X["l"])), "n_per_prompt": {n: int(mfn(D[n]).sum()) for n in names}}
    big_scope = len(X["l"]) > 20000
    for score, eff in (("C_tool", "dg"), ("C_tool", "da"), ("dot", "dg"), ("D_gate", "dg"), ("D_act", "da"), ("D_gate", "da")):
        if not big_scope:
            s[f"rho_|{score}|_|{eff}|"] = boot_sp(np.abs(X[score]), np.abs(X[eff]))
            s[f"rho_|{score}|_|{eff}|_clusterboot"] = boot_sp(np.abs(X[score]), np.abs(X[eff]), groups=grp, n=500)
        else:
            pp = np.array([sp(np.abs(D[n][score][mfn(D[n])]), np.abs(D[n][eff][mfn(D[n])])) for n in names])
            bs = [np.mean(pp[rng.integers(0, len(pp), len(pp))]) for _ in range(2000)]
            s[f"rho_|{score}|_|{eff}|"] = {"pooled_rho": sp(np.abs(X[score]), np.abs(X[eff])), "mean_per_prompt": float(pp.mean()),
                                            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                            "min_prompt": float(pp.min()), "max_prompt": float(pp.max()), "n": int(len(X["l"]))}
    if not big_scope:
        s["rho_|C_tool*a_src|_|dg|"] = boot_sp(np.abs(X["C_tool"] * X["a_src"]), np.abs(X["dg"]))
    s["per_prompt_rho_Ctool_dg"] = {n: sp(np.abs(D[n]["C_tool"][mfn(D[n])]), np.abs(D[n]["dg"][mfn(D[n])])) for n in names}
    s["per_prompt_rho_Dact_da"] = {n: sp(np.abs(D[n]["D_act"][mfn(D[n])]), np.abs(D[n]["da"][mfn(D[n])])) for n in names}
    # sign agreement: tool predicts sign(dg) = -sign(dot); D_gate predicts sign(dg) = sign(D_gate)
    big = np.abs(X["dg"]) > 2 * noise_g
    s["frac_|dg|_above_2x_noise"] = float(big.mean())
    s["sign_agree_tool_(-dot)"] = float((np.sign(-X["dot"][big]) == np.sign(X["dg"][big])).mean())
    s["sign_agree_D_gate"] = float((np.sign(X["D_gate"][big]) == np.sign(X["dg"][big])).mean())
    bigA = np.abs(X["da"]) > 2 * noise_a
    s["sign_agree_D_act_vs_da"] = float((np.sign(X["D_act"][bigA]) == np.sign(X["da"][bigA])).mean())
    s["median_rel_|dg|/|g0|"] = float(np.median(np.abs(X["dg"]) / (np.abs(X["g0"]) + 1e-6)))
    s["median_rel_|da|/|a0|"] = float(np.median(np.abs(X["da"]) / (np.abs(X["a0"]) + 1e-6)))
    # magnitude calibration of D_act (log-log slope / R^2 on pairs above noise)
    m = bigA & (np.abs(X["D_act"]) > 0)
    if m.sum() > 10:
        lx, ly = np.log10(np.abs(X["D_act"][m])), np.log10(np.abs(X["da"][m]))
        A = np.vstack([lx, np.ones_like(lx)]).T
        coef, res_, *_ = np.linalg.lstsq(A, ly, rcond=None)
        r2 = 1 - ((ly - A @ coef) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
        s["loglog_Dact_vs_da"] = {"slope": float(coef[0]), "intercept": float(coef[1]), "R2": float(r2), "n": int(m.sum()),
                                  "median_ratio_da_over_Dact": float(np.median(np.abs(X["da"][m]) / np.abs(X["D_act"][m])))}
    # by source band
    bands = {"L0-9": (0, 10), "L10-29": (10, 30), "L30-49": (30, 50), "L50-58": (50, 59)}
    s["by_band"] = {}
    for b, (lo, hi) in bands.items():
        mm = (X["l"] >= lo) & (X["l"] < hi)
        if mm.sum() >= 10:
            s["by_band"][b] = {"n": int(mm.sum()), "rho_Ctool_dg": sp(np.abs(X["C_tool"][mm]), np.abs(X["dg"][mm])),
                               "rho_Dgate_dg": sp(np.abs(X["D_gate"][mm]), np.abs(X["dg"][mm])),
                               "rho_Dact_da": sp(np.abs(X["D_act"][mm]), np.abs(X["da"][mm])),
                               "median_|C_tool|": float(np.median(np.abs(X["C_tool"][mm]))),
                               "median_|dg|": float(np.median(np.abs(X["dg"][mm]))),
                               "frac_dg_zero": float((X["dg"][mm] == 0).mean()),
                               "frac_|dg|<=2noise": float((np.abs(X["dg"][mm]) <= 2 * noise_g).mean())}
    out[scope] = s

# top-|C| filaments: what happens to the highest-scoring ones?
F = {k: cat(k, fil) for k in ("C_tool", "dg", "da", "l", "lp", "D_gate", "g0")}
o = np.argsort(-np.abs(F["C_tool"]))[:20]
out["top20_by_|C_tool|"] = [{"src_l": int(F["l"][i]), "dst_l": int(F["lp"][i]), "C": float(F["C_tool"][i]), "dg": float(F["dg"][i]),
                             "D_gate": float(F["D_gate"][i]), "g0": float(F["g0"][i])} for i in o]

# precision@6 per source layer: tool's top-6 by |C| among all candidates vs true top-6 by |da| (and |dg|)
prec = {"tool_C_vs_da": [], "D_act_vs_da": [], "tool_C_vs_dg": [], "D_gate_vs_dg": [], "random_expected": []}
for n in names:
    d = D[n]
    for l in np.unique(d["l"]):
        m = d["l"] == l
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        top = lambda v: set(idx[np.argsort(-np.abs(v[idx]))[:6]])
        t_da, t_dg = top(d["da"]), top(d["dg"])
        prec["tool_C_vs_da"].append(len(top(d["C_tool"]) & t_da) / 6); prec["D_act_vs_da"].append(len(top(d["D_act"]) & t_da) / 6)
        prec["tool_C_vs_dg"].append(len(top(d["C_tool"]) & t_dg) / 6); prec["D_gate_vs_dg"].append(len(top(d["D_gate"]) & t_dg) / 6)
        prec["random_expected"].append(6 / m.sum())
out["precision_at_6_per_source_layer"] = {k: {"mean": float(np.mean(v)), "n_layers": len(v)} for k, v in prec.items()}
# do the selected filaments have larger effects than unselected candidates? (median |da| ratio)
sel = cat("is_fil").astype(bool)
DA = np.abs(cat("da")); DG = np.abs(cat("dg"))
out["selected_vs_unselected"] = {"median_|da|_selected": float(np.median(DA[sel])), "median_|da|_unselected": float(np.median(DA[~sel])),
                                 "median_|dg|_selected": float(np.median(DG[sel])), "median_|dg|_unselected": float(np.median(DG[~sel])),
                                 "AUC_|da|_selected_gt_unselected": float((rankdata(np.concatenate([DA[sel], DA[~sel]]))[:sel.sum()].mean() - (sel.sum() - 1) / 2) / (~sel).sum())}

# gradients (job 06) where present
gfiles = sorted(glob.glob(RES + "V7_tool_bugs_06_filgrad_*.npz"))
if gfiles:
    gres = {}
    allx, ally, allD, allf, allC = [], [], [], [], []
    for f in gfiles:
        nm = f.split("V7_tool_bugs_06_filgrad_")[1][:-4]
        if nm not in D:
            continue
        g = np.load(f); d = D[nm]
        key = {(int(a), int(b), int(c), int(e)): i for i, (a, b, c, e) in enumerate(zip(d["l"], d["k"], d["lp"], d["j"]))}
        ii = [key.get((int(a), int(b), int(c), int(e)), -1) for a, b, c, e in zip(g["l"], g["k"], g["lp"], g["j"])]
        ok = np.array([i >= 0 for i in ii]); ii = np.array(ii)[ok]
        Ga, Gg = g["G_act"][ok], g["G_gate"][ok]
        gres[nm] = {"n": int(ok.sum()), "a_src_match_maxabs": float(np.max(np.abs(g["a_src"][ok] - d["a_src"][ii]))),
                    "rho_|G_act|_|da|": sp(np.abs(Ga), np.abs(d["da"][ii])), "rho_|G_gate|_|dg|": sp(np.abs(Gg), np.abs(d["dg"][ii])),
                    "rho_|D_act|_|da|_same_pairs": sp(np.abs(d["D_act"][ii]), np.abs(d["da"][ii])),
                    "rho_|C_tool|_|da|_same_pairs": sp(np.abs(d["C_tool"][ii]), np.abs(d["da"][ii]))}
        allx.append(np.abs(Ga)); ally.append(np.abs(d["da"][ii])); allD.append(np.abs(d["D_act"][ii])); allf.append(d["is_fil"][ii].astype(bool))
        allC.append(np.abs(d["C_tool"][ii]))
    X, Y, DD, FF, CC = map(np.concatenate, (allx, ally, allD, allf, allC))
    gres["pooled_all"] = {"rho_|G_act|_|da|": boot_sp(X, Y), "rho_|D_act|_|da|": boot_sp(DD, Y), "rho_|C_tool|_|da|": boot_sp(CC, Y)}
    gres["pooled_filaments"] = {"rho_|G_act|_|da|": boot_sp(X[FF], Y[FF]), "rho_|D_act|_|da|": boot_sp(DD[FF], Y[FF]),
                                "rho_|C_tool|_|da|": boot_sp(CC[FF], Y[FF])}
    out["gradients"] = gres
json.dump(out, open(RES + "V7_tool_bugs_07_filament_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:15000])
