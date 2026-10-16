#!/usr/bin/env python3
"""E7_binding_04: raw-residual geometry of entity gender (local numpy).
Per layer l and position type t, per-passage diff-of-means directions:
   ds_t = (mean over G_user of [G_self=W] - [G_self=M]) / 2,   du_t = same for G_user.
Noise-corrected cosines via split-half (passages) reliability: C_ij = E[cos(dA_i, dB_j)] / sqrt(E[cos(dA_i,dB_i)] E[cos(dA_j,dB_j)]).
Unbiased squared magnitudes ||d||^2 ~ E[<dA, dB>].  Raw-space binding index along the cross-validated optimal directions.
Also: cosine of each (noise-corrected) direction with external readout directions (3rd-person probes, lens pullbacks).
Two coordinate systems: raw, and diag-standardised (per-dimension pooled within-cell SD) to limit outlier-dimension dominance.
"""
import json
import numpy as np

ROOT = "."
RES = ROOT + "/experiments/results"
CACHE = RES + "/E7_binding_cache"
rng = np.random.default_rng(11)
NS = 200
Z = [np.load(f"{RES}/E7_binding_factorial_chunk{c}.npz") for c in range(3)]
TYPES = list(Z[0]["types"]); CELLN = list(Z[0]["cell_names"])
PID = np.concatenate([z["pid"] for z in Z]); CNT = np.concatenate([z["CNT"] for z in Z])
P = sorted(set(PID.tolist()))
valid = np.array([[CNT[PID == p, ti].max() > 0 for ti in range(len(TYPES))] for p in P])  # [NP, T]
print("passages lacking type:", {t: [P[i] for i in np.where(~valid[:, ti])[0]] for ti, t in enumerate(TYPES)})
R = np.concatenate([np.load(f"{CACHE}/resid_means_chunk{c}.npy", mmap_mode="r") for c in range(3)], 0)  # [NP,10,T,60,D] f16
NP = R.shape[0]
GSW = [CELLN.index(x) for x in ("WW", "WM", "WU")]; GSM = [CELLN.index(x) for x in ("MW", "MM", "MU")]
GUW = [CELLN.index(x) for x in ("WW", "MW", "UW")]; GUM = [CELLN.index(x) for x in ("WM", "MM", "UM")]
NINE = [CELLN.index(a + b) for a in "WMU" for b in "WMU"]
DIRN = [f"{g}@{t}" for t in TYPES for g in ("ds", "du")]
EXT = {}
for nm in ("ref", "ctx"):
    EXT[f"probe_{nm}"] = np.load(f"{CACHE}/probe_{nm}_muf.npy") - np.load(f"{CACHE}/probe_{nm}_mum.npy")
for kind in ("jpp", "jlens", "naive"):
    for w in ("pron", "noun"):
        EXT[f"pull_{kind}_{w}"] = np.load(f"{CACHE}/pullback_{kind}_{w}.npy")
EXTN = list(EXT)
splits = []
for _ in range(NS):
    perm = rng.permutation(NP)
    splits.append((perm[: NP // 2], perm[NP // 2:]))


def cosm(A, B):
    na = np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12
    nb = np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12
    return (A / na) @ (B / nb).T


out = {"dirs": DIRN, "ext": EXTN, "layers": list(range(60)), "per_layer": {}}
for space in ("raw", "diag"):
    per_layer = []
    for l in range(60):
        Xl = np.asarray(R[:, :, :, l, :], dtype=np.float32)  # [NP, 10, T, D]
        if space == "diag":
            resid = Xl[:, NINE] - Xl[:, NINE].mean(0, keepdims=True)  # remove cell x type means
            m = valid[:, None, :, None].repeat(9, 1)
            sd = np.sqrt((resid ** 2 * m).sum((0, 1, 2)) / m.sum((0, 1, 2)).clip(1)) + 1e-6
        else:
            sd = np.ones(Xl.shape[-1], dtype=np.float32)
        V = []
        for ti in range(len(TYPES)):
            Xt = Xl[:, :, ti] / sd
            V.append((Xt[:, GSW].mean(1) - Xt[:, GSM].mean(1)) / 2)
            V.append((Xt[:, GUW].mean(1) - Xt[:, GUM].mean(1)) / 2)
        V = np.stack(V, 0)  # [14, NP, D]
        vmask = np.repeat(valid.T, 2, 0)  # [14, NP]
        nd = len(DIRN)
        Msum = np.zeros((nd, nd)); Mcnt = np.zeros((nd, nd)); dots = np.zeros(nd)
        bi_s = np.zeros(len(TYPES)); bi_u = np.zeros(len(TYPES))
        for (A, Bh) in splits:
            dA = np.stack([V[i, A][vmask[i, A]].mean(0) for i in range(nd)])
            dB = np.stack([V[i, Bh][vmask[i, Bh]].mean(0) for i in range(nd)])
            C = cosm(dA, dB)
            Msum += 0.5 * (C + C.T); Mcnt += 1
            dots += (dA * dB).sum(-1)
            for ti in range(len(TYPES)):
                us = dA[2 * ti] / np.linalg.norm(dA[2 * ti]); uu = dA[2 * ti + 1] / np.linalg.norm(dA[2 * ti + 1])
                a1, b1 = dB[2 * ti] @ us, dB[2 * ti + 1] @ us
                a2, b2 = dB[2 * ti] @ uu, dB[2 * ti + 1] @ uu
                bi_s[ti] += (a1 - b1) / (abs(a1) + abs(b1) + 1e-12)
                bi_u[ti] += (b2 - a2) / (abs(a2) + abs(b2) + 1e-12)
        M = Msum / Mcnt
        rel = np.diag(M).copy()
        Ccorr = M / np.sqrt(np.outer(rel.clip(1e-6), rel.clip(1e-6)))
        mag2 = dots / NS
        dfull = np.stack([V[i][vmask[i]].mean(0) for i in range(nd)])
        att = np.sqrt(mag2.clip(0)) / (np.linalg.norm(dfull, axis=-1) + 1e-12)
        ext_dirs = np.stack([EXT[k][l] * sd for k in EXTN])  # readout direction w in standardised coords = w*sd
        Cext = cosm(dfull, ext_dirs)
        per_layer.append({"rel": rel.tolist(), "C_corr": Ccorr.tolist(), "C_raw": M.tolist(), "mag": np.sqrt(mag2.clip(0)).tolist(),
                          "att": att.tolist(), "cos_ext_raw": Cext.tolist(),
                          "cos_ext_corr": (Cext / att.clip(1e-3)[:, None]).tolist(),
                          "BI_raw_selfdir": (bi_s / NS).tolist(), "BI_raw_userdir": (bi_u / NS).tolist()})
        if l % 10 == 0:
            print(space, "layer", l, "done", flush=True)
    out["per_layer"][space] = per_layer
json.dump(out, open(f"{RES}/E7_binding_geometry.json", "w"))

BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
ix = {d: i for i, d in enumerate(DIRN)}
PAIRS = [("ds@A_self", "du@A_self"), ("du@A_user", "ds@A_user"), ("ds@A_neut", "du@A_neut"),
         ("ds@A_self", "du@A_user"), ("ds@A_self", "du@U_self"), ("ds@A_self", "ds@U_addr"),
         ("ds@A_self", "ds@A_user"), ("du@A_self", "du@A_user"), ("ds@A_self", "ds@A_neut"), ("du@A_user", "du@A_neut"),
         ("ds@F_I", "du@F_Y"), ("ds@A_self", "ds@F_I")]
summary = {}
for space in ("raw", "diag"):
    pl = out["per_layer"][space]
    print(f"\n=== {space}: noise-corrected cosines (band means over layers)")
    for (x, y) in PAIRS:
        vals = {b: float(np.mean([pl[l]["C_corr"][ix[x]][ix[y]] for l in r])) for b, r in BANDS.items()}
        summary[f"{space}|cos|{x}~{y}"] = vals
        print(f"  {x:10s} ~ {y:10s} " + " ".join(f"{b}:{v:+.2f}" for b, v in vals.items()))
    print("  split-half reliability:")
    for d in ("ds@A_self", "du@A_self", "ds@A_user", "du@A_user", "ds@A_neut", "du@A_neut", "du@U_self", "ds@F_I", "du@F_Y"):
        vals = {b: float(np.mean([pl[l]["rel"][ix[d]] for l in r])) for b, r in BANDS.items()}
        summary[f"{space}|rel|{d}"] = vals
        print(f"    {d:10s} " + " ".join(f"{b}:{v:+.2f}" for b, v in vals.items()))
    print("  magnitude ratio ||du||/||ds|| at the same type (unbiased split-half):")
    for t in ("A_self", "A_user", "A_neut", "U_self", "U_addr", "F_I", "F_Y"):
        vals = {b: float(np.mean([pl[l]["mag"][ix['du@' + t]] / max(pl[l]["mag"][ix['ds@' + t]], 1e-9) for l in r])) for b, r in BANDS.items()}
        summary[f"{space}|magratio_u_over_s|{t}"] = vals
        print(f"    {t:7s} " + " ".join(f"{b}:{v:.2f}" for b, v in vals.items()))
    print("  raw-space BI along cross-validated optimal direction (self dir at self-ref types, user dir at user-ref types):")
    for t in ("A_self", "U_addr", "F_I"):
        ti = TYPES.index(t)
        vals = {b: float(np.mean([pl[l]["BI_raw_selfdir"][ti] for l in r])) for b, r in BANDS.items()}
        summary[f"{space}|BIraw|{t}"] = vals
        print(f"    {t:7s} " + " ".join(f"{b}:{v:+.2f}" for b, v in vals.items()))
    for t in ("A_user", "U_self", "F_Y"):
        ti = TYPES.index(t)
        vals = {b: float(np.mean([pl[l]["BI_raw_userdir"][ti] for l in r])) for b, r in BANDS.items()}
        summary[f"{space}|BIraw|{t}"] = vals
        print(f"    {t:7s} " + " ".join(f"{b}:{v:+.2f}" for b, v in vals.items()))
    print("  cos(direction, external readout) noise-corrected for the diff-of-means side only:")
    for d in ("ds@A_self", "du@A_self", "du@A_user", "ds@A_user", "ds@F_I", "du@F_Y"):
        for e in ("probe_ref", "probe_ctx", "pull_jpp_pron", "pull_jlens_pron", "pull_naive_pron"):
            vals = {b: float(np.mean([pl[l]["cos_ext_corr"][ix[d]][EXTN.index(e)] for l in r])) for b, r in BANDS.items()}
            summary[f"{space}|ext|{d}~{e}"] = vals
            print(f"    {d:10s} ~ {e:16s} " + " ".join(f"{b}:{v:+.3f}" for b, v in vals.items()))
json.dump(summary, open(f"{RES}/E7_binding_geometry_summary.json", "w"), indent=1)
