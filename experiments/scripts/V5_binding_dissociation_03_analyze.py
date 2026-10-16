#!/usr/bin/env python3
"""V5_binding_dissociation_03: local analysis (numpy) of the 2x2x2 factorial readouts.
Unit = item (24). Per item and cell: mean over the 4 name/order variants (tokens first averaged within a sequence).
Coefficients (+/-1 coding, balanced design): a = G_A effect, b = G_U effect, c = G_T effect, each = (mean_W - mean_M)/2.
For each position type with referent R and default entity D (what the token class refers to outside quotes in that turn):
  DI = (coef_R - coef_D) / (|coef_R| + |coef_D|)   (+1 follows referent, -1 follows turn speaker/addressee)
  share_R = coef_R / (|a|+|b|+|c|)
  BI_vs_X = (coef_R - coef_X)/(|coef_R|+|coef_X|) for each non-referent entity X (E7-style).
Bootstrap over items (4000 resamples): indices computed from bootstrap-mean coefficients (as in E7).
Pooled referent/default weights: across the critical quoted types, coef[t, e] = alpha*[e==R_t] + beta*[e==D_t] + gamma.
"""
import json, sys
import numpy as np

ROOT = "."
RES = ROOT + "/experiments/results"
sys.path.insert(0, ROOT + "/experiments/scripts")
exec(open(ROOT + "/experiments/scripts/V5_binding_dissociation_common.py").read())
rng = np.random.default_rng(11)
DS = sys.argv[1] if len(sys.argv) > 1 else "factorial"   # "factorial" (_02) or "noqt" (_06)
NB = 4000
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50),
         "L50-58": range(50, 59), "L59": range(59, 60)}
chunks = [c for c in range(3) if __import__("os").path.exists(f"{RES}/V5_binding_dissociation_{DS}_chunk{c}.npz")]
Z = [np.load(f"{RES}/V5_binding_dissociation_{DS}_chunk{c}.npz") for c in chunks]
TY = list(Z[0]["types"]); RD = list(Z[0]["readouts"]); CELLN = list(Z[0]["cells"])
SC = np.concatenate([z["SC"] for z in Z]); BEH = np.concatenate([z["BEH"] for z in Z])
ITEM = np.concatenate([z["item"] for z in Z]); CELL = np.concatenate([z["cell"] for z in Z]); KK = np.concatenate([z["k"] for z in Z])
P = sorted(set(ITEM.tolist())); NP = len(P)
X = np.full((NP, 8, len(TY), 60, len(RD)), np.nan)
Xk = np.full((NP, 8, 4, len(TY), 60, len(RD)), np.nan)   # per variant (order analysis)
B = np.full((NP, 8, 5), np.nan)
import warnings
warnings.filterwarnings("ignore")
for pi, p in enumerate(P):
    for ci, c in enumerate(CELLN):
        m = (ITEM == p) & (CELL == c)
        X[pi, ci] = np.nanmean(SC[m], 0)
        B[pi, ci] = BEH[m].mean(0)
        for k in range(4):
            mk = m & (KK == k)
            Xk[pi, ci, k] = SC[mk][0]
IDX = {e: {"W": [i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "W"],
           "M": [i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "M"]} for e in "AUT"}


def coefs(Y):
    """Y [NP, 8] -> dict e -> [NP] coefficient."""
    return {e: (Y[:, IDX[e]["W"]].mean(1) - Y[:, IDX[e]["M"]].mean(1)) / 2 for e in "AUT"}


def q(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def idx_stats(C, R, D):
    """C: dict e->[NP]. Returns coefficient means/CIs and indices with bootstrap CIs."""
    ok = ~np.any([np.isnan(C[e]) for e in "AUT"], 0)
    n = int(ok.sum())
    if n < 4:
        return None
    Cc = {e: C[e][ok] for e in "AUT"}
    bi = rng.integers(0, n, (NB, n))
    Bm = {e: Cc[e][bi].mean(1) for e in "AUT"}
    M = {e: float(Cc[e].mean()) for e in "AUT"}
    out = {"n": n}
    for e in "AUT":
        out[e] = M[e]; out[e + "_ci"] = q(Bm[e])
    if R is not None:
        tot = lambda d: np.abs(d["A"]) + np.abs(d["U"]) + np.abs(d["T"]) + 1e-12
        out["share_R"] = float(M[R] / tot(M)); out["share_R_ci"] = q(Bm[R] / tot(Bm))
        for xx in "AUT":
            if xx != R:
                f = lambda d: (d[R] - d[xx]) / (np.abs(d[R]) + np.abs(d[xx]) + 1e-12)
                out[f"BI_vs_{xx}"] = float(f(M)); out[f"BI_vs_{xx}_ci"] = q(f(Bm))
        if D is not None and D != R:
            f = lambda d: (d[R] - d[D]) / (np.abs(d[R]) + np.abs(d[D]) + 1e-12)
            out["DI"] = float(f(M)); out["DI_ci"] = q(f(Bm))
            out["R_minus_D"] = float(M[R] - M[D]); out["R_minus_D_ci"] = q(Bm[R] - Bm[D])
    return out


CRIT = ["A_qU_I", "A_qT_I", "A_qS_I", "A_qS_you", "U_qA_I", "U_qT_I", "U_qA_you"]
BASE = ["A_out_I", "A_init_I", "A_nearT_I", "A_out_you", "U_out_I", "U_out_you", "A_T", "U_T", "A_neut", "F_T"]
SHOW = CRIT + BASE
out = {"n_items": NP, "chunks": chunks, "types": TY, "readouts": RD, "cells": CELLN}
band = {}
for r in RD:
    ri = RD.index(r)
    for t in SHOW:
        ti = TY.index(t)
        R, D = type_referent(t)
        for bn, br in BANDS.items():
            Y = X[:, :, ti, list(br), ri].mean(-1)
            s = idx_stats(coefs(Y), R, D)
            if s is not None:
                band[f"{r}|{t}|{bn}"] = s
out["band"] = band
# per-layer DI / BI for key readouts
perl = {}
for r in ("jpp_pron", "jpp_noun", "jppQ_pron", "jlens_pron", "naive_pron", "probe_ref", "e7dir_self"):
    ri = RD.index(r)
    for t in SHOW:
        ti = TY.index(t); R, D = type_referent(t)
        rows = []
        for l in range(60):
            s = idx_stats(coefs(X[:, :, ti, l, ri]), R, D)
            rows.append(s)
        perl[f"{r}|{t}"] = rows
out["per_layer"] = perl
# pooled referent vs default weight across critical quoted types (bootstrap over items)
POOL = ["A_qU_I", "A_qT_I", "A_qS_you", "U_qA_I", "U_qT_I"] if DS == "factorial" else ["A_qU_I", "A_qS_you", "U_qA_I"]
pooled = {}
for r in RD:
    ri = RD.index(r)
    for bn, br in BANDS.items():
        Cs = []
        for t in POOL:
            ti = TY.index(t)
            Cs.append(coefs(X[:, :, ti, list(br), ri].mean(-1)))
        # design rows
        rowsX, keys = [], []
        for t in POOL:
            R, D = type_referent(t)
            for e in "AUT":
                rowsX.append([1.0 if e == R else 0.0, 1.0 if e == D else 0.0, 1.0]); keys.append((t, e))
        Xd = np.array(rowsX)
        def fit(idx):
            y = np.array([Cs[POOL.index(t)][e][idx].mean() for (t, e) in keys])
            beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            return beta
        allidx = np.arange(NP)
        b0 = fit(allidx)
        bs = np.array([fit(rng.integers(0, NP, NP)) for _ in range(1000)])
        pooled[f"{r}|{bn}"] = {"alpha_ref": float(b0[0]), "alpha_ci": q(bs[:, 0]), "beta_default": float(b0[1]), "beta_ci": q(bs[:, 1]),
                               "gamma": float(b0[2]), "ref_minus_default_ci": q(bs[:, 0] - bs[:, 1]),
                               "ref_frac": float(b0[0] / (abs(b0[0]) + abs(b0[1]) + 1e-12)),
                               "ref_frac_ci": q(bs[:, 0] / (np.abs(bs[:, 0]) + np.abs(bs[:, 1]) + 1e-12))}
out["pooled"] = pooled
# behaviour at F_T (real next-token, noun4 LSE)
Cb = coefs(B[:, :, 2])
out["behaviour_FT"] = idx_stats(Cb, "T", None)
out["behaviour_FT"]["mass_F4M4"] = float((B[:, :, 0] + B[:, :, 1]).mean())
# order: per variant k (entity order in system prompt), J++ pron L20-39 DI for critical types
order = {}
for k in range(4):
    for t in CRIT + ["A_out_I", "A_out_you"]:
        ti = TY.index(t); R, D = type_referent(t)
        Y = Xk[:, :, k, ti, 20:40, RD.index("jpp_pron")].mean(-1)
        s = idx_stats(coefs(Y), R, D)
        if s is not None:
            order[f"k{k}|{'-'.join(ORDERS[k])}|{t}"] = {kk: s[kk] for kk in s if kk in ("A", "U", "T", "DI", "DI_ci", "share_R")}
out["order"] = order
json.dump(out, open(f"{RES}/V5_binding_dissociation_analysis{'' if DS == 'factorial' else '_' + DS}.json", "w"), indent=1)


# ---------------------------------------------------------------- printed summary
def fmt(s, keys=("A", "U", "T")):
    if s is None:
        return "n/a"
    t = " ".join(f"{s[k]:+.2f}" for k in keys)
    if "DI" in s:
        t += f" DI {s['DI']:+.2f}[{s['DI_ci'][0]:+.2f},{s['DI_ci'][1]:+.2f}]"
    elif "share_R" in s:
        t += f" sh {s['share_R']:+.2f}"
    return t


for r in ("jpp_pron", "jpp_noun", "jppQ_pron", "jlens_pron", "naive_pron", "probe_ref", "e7dir_self"):
    print(f"\n=== {r}: a(G_A) b(G_U) c(G_T) and DI(referent vs default) / share_R ===")
    for t in SHOW:
        R, D = type_referent(t)
        row = [fmt(band.get(f"{r}|{t}|{bn}")) for bn in ("L20-29", "L30-39", "L40-49", "L50-58")]
        print(f"{t:10s} R={R} D={D} | " + " | ".join(row))
print("\n=== pooled referent (alpha) vs default (beta) weights, critical quoted types ===")
for r in ("jpp_pron", "jpp_noun", "jppQ_pron", "jlens_pron", "naive_pron", "probe_ref", "e7dir_self"):
    print(r, " | ".join(f"{bn}: a={pooled[f'{r}|{bn}']['alpha_ref']:+.2f} b={pooled[f'{r}|{bn}']['beta_default']:+.2f} "
                        f"rf={pooled[f'{r}|{bn}']['ref_frac']:+.2f}[{pooled[f'{r}|{bn}']['ref_frac_ci'][0]:+.2f},{pooled[f'{r}|{bn}']['ref_frac_ci'][1]:+.2f}]"
                        for bn in ("L20-29", "L30-39", "L40-49", "L50-58")))
print("\nbehaviour F_T:", fmt(out["behaviour_FT"]), "mass", round(out["behaviour_FT"]["mass_F4M4"], 3))
