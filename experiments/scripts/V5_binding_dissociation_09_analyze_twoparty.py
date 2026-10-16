#!/usr/bin/env python3
"""V5_binding_dissociation_09: analysis of _08 (two-party control, G_A x G_U only, E7 passages + inserted quotes).
Coefficients a (G_A), b (G_U) per item (variant-averaged), bootstrap over 24 items (4000).
DI = (coef_R - coef_D)/(|coef_R|+|coef_D|) for quoted types; BI = (R - other)/(|R|+|other|) for unquoted types."""
import json, os
import numpy as np
ROOT = "."; RES = ROOT + "/experiments/results"
rng = np.random.default_rng(17); NB = 4000
ch = [c for c in range(2) if os.path.exists(f"{RES}/V5_binding_dissociation_twoparty_chunk{c}.npz")]
Z = [np.load(f"{RES}/V5_binding_dissociation_twoparty_chunk{c}.npz") for c in ch]
TY = list(Z[0]["types"]); RD = list(Z[0]["readouts"])
SC = np.concatenate([z["SC"] for z in Z]); BEH = np.concatenate([z["BEH"] for z in Z])
ITEM = np.concatenate([z["item"] for z in Z]); CELL = np.concatenate([z["cell"] for z in Z]); SUF = np.concatenate([z["suf"] for z in Z])
CELLN = ["WWW", "WMW", "MWW", "MMW"]
P = sorted(set(ITEM.tolist())); NP = len(P)
import warnings; warnings.filterwarnings("ignore")
X = np.full((NP, 4, len(TY), 60, len(RD)), np.nan); B = np.full((NP, 4, 2, 5), np.nan)
for pi, p in enumerate(P):
    for ci, c in enumerate(CELLN):
        m0 = (ITEM == p) & (CELL == c) & (SUF == 0)
        X[pi, ci] = np.nanmean(SC[m0], 0)
        for s in (0, 1):
            B[pi, ci, s] = BEH[(ITEM == p) & (CELL == c) & (SUF == s)].mean(0)
AW, AM, UW, UM = [0, 1], [2, 3], [0, 2], [1, 3]
def coefs(Y):
    return (Y[:, AW].mean(1) - Y[:, AM].mean(1)) / 2, (Y[:, UW].mean(1) - Y[:, UM].mean(1)) / 2
def q(v): return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
REF = {"U_out_I": ("U", "U"), "U_out_you": ("A", "A"), "U_qA_I": ("A", "U"), "U_qA_you": ("U", "A"),
       "A_out_I": ("A", "A"), "A_out_you": ("U", "U"), "A_qU_I": ("U", "A"), "A_nearT_I": ("A", "A")}
def st(a, b, R):
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]; n = len(a)
    if n < 4: return None
    bi = rng.integers(0, n, (NB, n)); A_, B_ = a[bi].mean(1), b[bi].mean(1)
    if R == "A": f = lambda x, y: (x - y) / (np.abs(x) + np.abs(y) + 1e-12)
    else: f = lambda x, y: (y - x) / (np.abs(x) + np.abs(y) + 1e-12)
    return {"n": n, "A": float(a.mean()), "A_ci": q(A_), "U": float(b.mean()), "U_ci": q(B_),
            "idx": float(f(a.mean(), b.mean())), "idx_ci": q(f(A_, B_))}
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
out = {"n_items": NP, "band": {}, "behaviour": {}}
for r in RD:
    ri = RD.index(r)
    for t, (R, D) in REF.items():
        ti = TY.index(t)
        for bn, br in BANDS.items():
            s = st(*coefs(X[:, :, ti, list(br), ri].mean(-1)), R)
            if s is not None: out["band"][f"{r}|{t}|{bn}"] = s
for s, (nm, R) in enumerate((("P_own", "A"), ("P_qU", "U"))):
    out["behaviour"][nm] = st(*coefs(B[:, :, s, 2]), R)
    out["behaviour"][nm]["mass_F4M4"] = float((B[:, :, s, 0] + B[:, :, s, 1]).mean())
json.dump(out, open(f"{RES}/V5_binding_dissociation_twoparty_analysis.json", "w"), indent=1)
print("n items", NP, "| idx = referent-vs-other index (for quoted types: DI referent vs default)")
for r in ("jpp_pron", "jpp_noun", "jppQ_pron", "jlens_pron", "naive_pron", "probe_ref", "e7dir_self"):
    print(f"=== {r}: a b idx[CI] ===")
    for t, (R, D) in REF.items():
        row = []
        for bn in ("L20-29", "L30-39", "L40-49", "L50-58"):
            s = out["band"].get(f"{r}|{t}|{bn}")
            row.append("n/a" if s is None else f"{s['A']:+.2f} {s['U']:+.2f} {s['idx']:+.2f}[{s['idx_ci'][0]:+.2f},{s['idx_ci'][1]:+.2f}]")
        print(f"{t:10s} R={R} D={D} | " + " | ".join(row))
for k, v in out["behaviour"].items():
    print("behaviour", k, f"a {v['A']:+.2f} b {v['U']:+.2f} idx {v['idx']:+.2f}[{v['idx_ci'][0]:+.2f},{v['idx_ci'][1]:+.2f}] mass {v['mass_F4M4']:.3f}")
