#!/usr/bin/env python3
"""E7_binding_03: local (CPU, numpy) analysis of the factorial readouts.
Unit of analysis = passage (24). Each passage's value per cell = mean over the 4 name/order variants
(tokens are first averaged within each sequence). Bootstrap over passages (2000 resamples).
Coefficients: score = a*G_self + b*G_user + c on the balanced 3x3 design => a = (mean_W. - mean_M.)/2, b likewise.
BI_ref = (coef_referent - coef_other) / (|a| + |b|), referent = self for A_self/U_addr/F_I, user for A_user/U_self/F_Y.
"""
import json
import numpy as np

ROOT = "."
RES = ROOT + "/experiments/results"
rng = np.random.default_rng(7)
NB = 2000
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50),
         "L50-58": range(50, 59), "L59": range(59, 60)}

Z = [np.load(f"{RES}/E7_binding_factorial_chunk{c}.npz") for c in range(3)]
TYPES = list(Z[0]["types"]); READ = list(Z[0]["readouts"]); CELLN = list(Z[0]["cell_names"])
SC = np.concatenate([z["SC"] for z in Z]); CNT = np.concatenate([z["CNT"] for z in Z]); BEH = np.concatenate([z["BEH"] for z in Z])
PID = np.concatenate([z["pid"] for z in Z]); CELL = np.concatenate([z["cell"] for z in Z])
P = sorted(set(PID.tolist())); NP = len(P)
# X[p, cell, type, layer, readout] (variant-averaged; nan where a passage lacks that token type)
X = np.full((NP, len(CELLN), len(TYPES), 60, len(READ)), np.nan)
B = np.full((NP, len(CELLN), 2, 5), np.nan)
for pi, p in enumerate(P):
    for ci, c in enumerate(CELLN):
        m = (PID == p) & (CELL == c)
        X[pi, ci] = np.nanmean(SC[m], 0)
        B[pi, ci] = BEH[m].mean(0)
GS = {"W": [CELLN.index(x) for x in ("WW", "WM", "WU")], "M": [CELLN.index(x) for x in ("MW", "MM", "MU")]}
GU = {"W": [CELLN.index(x) for x in ("WW", "MW", "UW")], "M": [CELLN.index(x) for x in ("WM", "MM", "UM")]}
NINE = [CELLN.index("".join(x)) for x in [(a, b) for a in "WMU" for b in "WMU"]]
REFERENT = {"A_self": "self", "U_addr": "self", "F_I": "self", "A_user": "user", "U_self": "user", "F_Y": "user", "A_neut": None}


def coefs(Y):
    """Y [..., cells, ...] with cell axis=1 -> a, b, c per passage (axis 0 kept)."""
    a = (Y[:, GS["W"]].mean(1) - Y[:, GS["M"]].mean(1)) / 2
    b = (Y[:, GU["W"]].mean(1) - Y[:, GU["M"]].mean(1)) / 2
    c = Y[:, NINE].mean(1)
    return a, b, c


def boot_stats(a, b, ref):
    """a, b: [NP] per-passage coefficients (nan-free subset used). Returns dict with means, CIs, BI and CI."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    idx = rng.integers(0, n, (NB, n))
    A, Bb = a[idx].mean(1), b[idx].mean(1)
    sign = 1 if ref in ("self", None) else -1
    bi = lambda x, y: sign * (x - y) / (np.abs(x) + np.abs(y) + 1e-12)
    BI = bi(A, Bb)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n": int(n), "a": float(a.mean()), "a_ci": q(A), "b": float(b.mean()), "b_ci": q(Bb),
            "BI": float(bi(a.mean(), b.mean())), "BI_ci": q(BI),
            "ratio_other_over_ref": float((b.mean() / a.mean()) if sign == 1 else (a.mean() / b.mean())),
            "diff_ref_minus_other": float(sign * (a.mean() - b.mean())), "diff_ci": q(sign * (A - Bb))}


out = {"n_passages": NP, "types": TYPES, "readouts": READ, "cells": CELLN, "bands": {k: list(v) for k, v in BANDS.items()}}
# ---------------------------------------------------------------- band-level binding table
band_tab = {}
for r_i, r in enumerate(READ):
    for t_i, t in enumerate(TYPES):
        for bn, br in BANDS.items():
            Y = X[:, :, t_i, list(br), r_i].mean(-1)  # [NP, cells]
            a, b, c = coefs(Y)
            band_tab[f"{r}|{t}|{bn}"] = boot_stats(a, b, REFERENT[t])
out["band_table"] = band_tab
# ---------------------------------------------------------------- per-layer arrays (point + CI) for key readouts
per_layer = {}
for r in ("jpp_pron", "jpp_noun", "jpp_noun4", "jlens_pron", "jlens_noun", "naive_pron", "naive_noun", "probe_ref", "probe_ctx"):
    r_i = READ.index(r)
    for t_i, t in enumerate(TYPES):
        rows = []
        for l in range(60):
            a, b, c = coefs(X[:, :, t_i, l, r_i])
            s = boot_stats(a, b, REFERENT[t])
            rows.append({k: s[k] for k in ("a", "a_ci", "b", "b_ci", "BI", "BI_ci")} | {"c": float(np.nanmean(c))})
        per_layer[f"{r}|{t}"] = rows
out["per_layer"] = per_layer
# ---------------------------------------------------------------- cell-mean tables (3x3 + D) for illustration
cellmeans = {}
for r in ("jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "probe_ref"):
    r_i = READ.index(r)
    for t in ("A_self", "A_user", "A_neut", "U_self", "F_I", "F_Y"):
        t_i = TYPES.index(t)
        for bn, br in BANDS.items():
            Y = X[:, :, t_i, list(br), r_i].mean(-1)
            cellmeans[f"{r}|{t}|{bn}"] = {c: float(np.nanmean(Y[:, ci])) for ci, c in enumerate(CELLN)}
out["cell_means"] = cellmeans
# ---------------------------------------------------------------- woman-man scale position of default (D) and UU at self tokens
scale = {}
for r in ("jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "probe_ref"):
    r_i = READ.index(r)
    for t in ("A_self", "F_I"):
        t_i = TYPES.index(t)
        for bn, br in BANDS.items():
            Y = X[:, :, t_i, list(br), r_i].mean(-1)
            wu, mu, uu, d = Y[:, CELLN.index("WU")], Y[:, CELLN.index("MU")], Y[:, CELLN.index("UU")], Y[:, CELLN.index("D")]
            idx = rng.integers(0, NP, (NB, NP))
            def pos(x):
                v = (x[idx].mean(1) - mu[idx].mean(1)) / (wu[idx].mean(1) - mu[idx].mean(1))
                return float((x.mean() - mu.mean()) / (wu.mean() - mu.mean())), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            pD, ciD = pos(d); pU, ciU = pos(uu)
            scale[f"{r}|{t}|{bn}"] = {"WU": float(wu.mean()), "MU": float(mu.mean()), "UU": float(uu.mean()), "D": float(d.mean()),
                                      "pos_D": pD, "pos_D_ci": ciD, "pos_UU": pU, "pos_UU_ci": ciU}
out["woman_man_scale"] = scale
# ---------------------------------------------------------------- behaviour (real next-token log-odds, noun4)
beh = {}
for k2, t in ((0, "F_I"), (1, "F_Y")):
    Y = B[:, :, k2, 2]
    a, b, c = coefs(Y)
    beh[t] = boot_stats(a, b, REFERENT[t])
    beh[t]["cell_means_lse4"] = {c_: float(Y[:, ci].mean()) for ci, c_ in enumerate(CELLN)}
    beh[t]["mass_F4_plus_M4"] = {c_: float((B[:, ci, k2, 0] + B[:, ci, k2, 1]).mean()) for ci, c_ in enumerate(CELLN)}
    beh[t]["p_woman"] = {c_: float(B[:, ci, k2, 3].mean()) for ci, c_ in enumerate(CELLN)}
    beh[t]["p_man"] = {c_: float(B[:, ci, k2, 4].mean()) for ci, c_ in enumerate(CELLN)}
out["behaviour"] = beh
json.dump(out, open(f"{RES}/E7_binding_analysis.json", "w"), indent=1)

# ---------------------------------------------------------------- print summary
def fmt(s):
    return (f"a={s['a']:+.2f}[{s['a_ci'][0]:+.2f},{s['a_ci'][1]:+.2f}] b={s['b']:+.2f}[{s['b_ci'][0]:+.2f},{s['b_ci'][1]:+.2f}] "
            f"BI={s['BI']:+.2f}[{s['BI_ci'][0]:+.2f},{s['BI_ci'][1]:+.2f}]")

print("BEHAVIOUR (real logits, LSE(woman,girl,lady,female)-LSE(man,boy,guy,male))")
for t in ("F_I", "F_Y"):
    print(" ", t, fmt(beh[t]), " mass(F4+M4) UU=%.3f WW=%.3f MM=%.3f" % (beh[t]["mass_F4_plus_M4"]["UU"], beh[t]["mass_F4_plus_M4"]["WW"], beh[t]["mass_F4_plus_M4"]["MM"]))
for r in ("jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "probe_ref", "probe_ctx"):
    print("\n==", r)
    for t in TYPES:
        print(" ", t)
        for bn in BANDS:
            print(f"    {bn:7s} " + fmt(band_tab[f'{r}|{t}|{bn}']))
