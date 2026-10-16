#!/usr/bin/env python3
"""V5_binding_dissociation_07: analysis of _04 (prediction positions inside quoted speech).
Per suffix: behaviour (real next-token noun4 LSE woman-set minus man-set) coefficients a,b,c (G_A, G_U, G_T), DI(R vs D),
share_R; lens readouts per band and per layer. Bootstrap over items (4000)."""
import json, os, sys
import numpy as np
ROOT = "."; RES = ROOT + "/experiments/results"
exec(open(ROOT + "/experiments/scripts/V5_binding_dissociation_common.py").read())
rng = np.random.default_rng(13); NB = 4000
SUFR = {"P_own": ("A", "A"), "P_you": ("U", "U"), "P_qU": ("U", "A"), "P_qT": ("T", "A"), "P_qS": ("T", "U"),
        "P_UqA": ("A", "U"), "P_UqT": ("T", "U")}
BANDS = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
ch = [c for c in range(3) if os.path.exists(f"{RES}/V5_binding_dissociation_predpos_chunk{c}.npz")]
Z = [np.load(f"{RES}/V5_binding_dissociation_predpos_chunk{c}.npz") for c in ch]
SCR = np.concatenate([z["SCR"] for z in Z]); BEH = np.concatenate([z["BEH"] for z in Z])
SI = np.concatenate([z["si"] for z in Z]); ITEM = np.concatenate([z["item"] for z in Z]); CELL = np.concatenate([z["cell"] for z in Z])
SUFS = list(Z[0]["sufs"]); RD = list(Z[0]["readouts"])
CELLN = ["".join(c) for c in CELLS]
P = sorted(set(ITEM.tolist())); NP = len(P)
X = np.zeros((NP, 8, len(SUFS), 60, len(RD))); Bv = np.zeros((NP, 8, len(SUFS), 4))
for pi, p in enumerate(P):
    for ci, c in enumerate(CELLN):
        for si in range(len(SUFS)):
            m = (ITEM == p) & (CELL == c) & (SI == si)
            X[pi, ci, si] = SCR[m].mean(0); Bv[pi, ci, si] = BEH[m].mean(0)
IDX = {e: ([i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "W"], [i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "M"]) for e in "AUT"}
def coefs(Y):
    return {e: (Y[:, IDX[e][0]].mean(1) - Y[:, IDX[e][1]].mean(1)) / 2 for e in "AUT"}
def q(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def st(C, R, D):
    bi = rng.integers(0, NP, (NB, NP))
    Bm = {e: C[e][bi].mean(1) for e in "AUT"}; M = {e: float(C[e].mean()) for e in "AUT"}
    o = {e: M[e] for e in "AUT"}
    for e in "AUT": o[e + "_ci"] = q(Bm[e])
    tot = lambda d: np.abs(d["A"]) + np.abs(d["U"]) + np.abs(d["T"]) + 1e-12
    o["share_R"] = float(M[R] / tot(M)); o["share_R_ci"] = q(Bm[R] / tot(Bm))
    oth = [x for x in "AUT" if x != R]
    for xx in oth:
        f = lambda d: (d[R] - d[xx]) / (np.abs(d[R]) + np.abs(d[xx]) + 1e-12)
        o[f"BI_vs_{xx}"] = float(f(M)); o[f"BI_vs_{xx}_ci"] = q(f(Bm))
    if D != R:
        f = lambda d: (d[R] - d[D]) / (np.abs(d[R]) + np.abs(d[D]) + 1e-12)
        o["DI"] = float(f(M)); o["DI_ci"] = q(f(Bm))
    return o
out = {"n_items": NP, "behaviour": {}, "band": {}, "per_layer": {}}
for si, s in enumerate(SUFS):
    R, D = SUFR[s]
    out["behaviour"][s] = st(coefs(Bv[:, :, si, 2]), R, D)
    out["behaviour"][s]["mass_F4M4"] = float((Bv[:, :, si, 0] + Bv[:, :, si, 1]).mean())
    for r in RD:
        ri = RD.index(r)
        for bn, br in BANDS.items():
            out["band"][f"{r}|{s}|{bn}"] = st(coefs(X[:, :, si, list(br), ri].mean(-1)), R, D)
    for r in ("jpp_pron", "jpp_noun4", "probe_ref"):
        ri = RD.index(r)
        out["per_layer"][f"{r}|{s}"] = [st(coefs(X[:, :, si, l, ri]), R, D) for l in range(60)]
json.dump(out, open(f"{RES}/V5_binding_dissociation_predpos_analysis.json", "w"), indent=1)
def f(o):
    t = f"a {o['A']:+.2f} b {o['U']:+.2f} c {o['T']:+.2f} share_R {o['share_R']:+.2f}[{o['share_R_ci'][0]:+.2f},{o['share_R_ci'][1]:+.2f}]"
    if "DI" in o: t += f" DI {o['DI']:+.2f}[{o['DI_ci'][0]:+.2f},{o['DI_ci'][1]:+.2f}]"
    return t
print("n items", NP)
print("=== BEHAVIOUR (real next-token noun4 LSE) ===")
for s in SUFS:
    print(f"{s:6s} R,D={SUFR[s]} {f(out['behaviour'][s])} mass {out['behaviour'][s]['mass_F4M4']:.3f}")
for r in ("jpp_pron", "jpp_noun4", "jppQ_pron", "probe_ref"):
    print(f"=== {r} ===")
    for s in SUFS:
        print(f"{s:6s} " + " | ".join(f"{bn}: {out['band'][f'{r}|{s}|{bn}']['A']:+.2f} {out['band'][f'{r}|{s}|{bn}']['U']:+.2f} {out['band'][f'{r}|{s}|{bn}']['T']:+.2f} sh {out['band'][f'{r}|{s}|{bn}']['share_R']:+.2f}" for bn in BANDS))
