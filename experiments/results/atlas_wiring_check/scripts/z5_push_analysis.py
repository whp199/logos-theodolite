# Kept from the session of the night of 28-29 October 2026 (its scratchpad); only the paths changed, to this folder
# and atlas_data/offline/wiring_check_31b (see ../summary.json).
"""Adjudicator: z4's pushes on the 31B read against the index's g and u and the stored check."""
import sys, json, glob, math
import numpy as np

W = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b"
z = np.load("/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz", allow_pickle=True)
sg = lambda x: np.where(x > 0, 1, np.where(x < 0, -1, 0))
c = math.sqrt(2 / math.pi)
gelu = lambda x: 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
res = {}
for f in sorted(glob.glob(f"{W}/z4_out/L*.npz")):
    p = np.load(f)
    l = int(f.split("/L")[-1][:-4])
    js = list(p["js"]); doses = list(p["doses"]); NW = len(p["writers"])
    assert list(p["writers"]) == list(z[f"L{l}_writers"][:NW])
    rd = z[f"L{l}_readers"][:NW]; lk = z[f"L{l}_linked"][:NW]; gi = z[f"L{l}_g"][:NW]; ui = z[f"L{l}_u"][:NW]
    reads = z[f"L{l}_reads"][:NW, :, 0, :, 0]  # question 0 ("who"), last site: [W, R, 3]
    s = z[f"L{l}_sign"][:NW].astype(int); ga = z[f"L{l}_gate"][:NW].astype(int)
    out = {"ceil_rel_diff": float(np.abs(p["ceil_now"] / p["ceil_npz"] - 1).max())}
    rows = {}
    for grp, m0 in (("linked", lk), ("null", ~lk)):
        recs = []
        for a in range(NW):
            for r in range(rd.shape[1]):
                if rd[a, r, 0] < 0 or not m0[a, r]:
                    continue
                jj, b = int(rd[a, r, 0]), int(rd[a, r, 1])
                k = js.index(jj)
                g0, u0, a0 = p["g"][k, 0, b], p["u"][k, 0, b], p["a"][k, 0, b]
                dg = [p["g"][k, 1 + a * len(doses) + di, b] - g0 for di in range(len(doses))]
                du = [p["u"][k, 1 + a * len(doses) + di, b] - u0 for di in range(len(doses))]
                da = [p["a"][k, 1 + a * len(doses) + di, b] - a0 for di in range(len(doses))]
                recs.append((gi[a, r], ui[a, r], s[a, r], ga[a, r], g0, u0, a0, *dg, *du, *da, *reads[a, r]))
        R = np.array(recs, np.float64)
        gI, uI, sI, gaI, g0, u0, a0 = R[:, :7].T
        nd = len(doses)
        DG, DU, DA = R[:, 7:7 + nd], R[:, 7 + nd:7 + 2 * nd], R[:, 7 + 2 * nd:7 + 3 * nd]
        st = R[:, 7 + 3 * nd:]
        o = {"n": len(R)}
        o["repro_unpushed_sign"] = float((sg(a0) == sg(st[:, 0]))[st[:, 0] != 0].mean())
        o["repro_dplus_sign"] = float((sg(DA[:, 0]) == sg(st[:, 1] - st[:, 0]))[(st[:, 1] != st[:, 0])].mean())
        o["repro_dplus_corr"] = float(np.corrcoef(DA[:, 0], st[:, 1] - st[:, 0])[0, 1])
        for di, d in enumerate(doses):
            if d < 0:
                continue
            odd_g = (DG[:, di] - DG[:, di + 1]) / 2; odd_u = (DU[:, di] - DU[:, di + 1]) / 2
            o[f"dose{d}:sign(dg)==sign(g)"] = float((sg(odd_g) == sg(gI)).mean())
            o[f"dose{d}:sign(du)==sign(u)"] = float((sg(odd_u) == sg(uI)).mean())
            o[f"dose{d}:corr(dg,g)"] = float(np.corrcoef(odd_g, gI)[0, 1])
            o[f"dose{d}:median|dg|/|g0|"] = float(np.median(np.abs(DG[:, di]) / np.maximum(np.abs(g0), 1e-6)))
            o[f"dose{d}:share gate crosses 0 on gate face"] = float(((g0 <= 0) & (g0 + np.where(gaI > 0, DG[:, di], DG[:, di + 1]) > 0)).mean())
            dgf = np.where(gaI > 0, DA[:, di], DA[:, di + 1])
            o[f"dose{d}:sign_gate"] = float((sg(dgf) == sI)[dgf != 0].mean())
            o[f"dose{d}:sign_gate[g0>0]"] = float((sg(dgf) == sI)[(dgf != 0) & (g0 > 0)].mean()) if ((g0 > 0) & (dgf != 0)).any() else None
            al0 = 0.5 * (1 + np.tanh(c * (g0 + 0.044715 * g0 ** 3)))  # placeholder, exact gelu' below
            t = np.tanh(c * (g0 + 0.044715 * g0 ** 3))
            dgl = 0.5 * (1 + t) + 0.5 * g0 * (1 - t * t) * c * (1 + 3 * 0.044715 * g0 * g0)
            k_true = sg(dgl * u0 * gI + gelu(g0) * uI)
            o[f"dose{d}:use_gate_true_op"] = float((sg(dgf) == gaI * k_true)[dgf != 0].mean())
        rows[grp] = o
    out.update(rows)
    res[f"L{l}"] = out
print(json.dumps(res, indent=1))
