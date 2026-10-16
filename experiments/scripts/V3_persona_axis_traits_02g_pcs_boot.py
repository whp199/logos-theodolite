# V3_persona_axis_traits_02g (offline CPU): bootstrap CIs (cross-fitted, stratified within folds) for the persona-axis
# position of default, neutral_human(pooled) and their difference after removing the top-k token PCs (k = 0, 10, 30, 100).
import json, numpy as np, torch
torch.set_num_threads(16)
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
A = torch.load(f"{V3_CACHE}/self_A.pt"); Bd = torch.load(f"{V3_CACHE}/self_B.pt")
TA = torch.load(f"{V3_CACHE}/toks_A.pt"); TB = torch.load(f"{V3_CACHE}/toks_B.pt")
def gather(S, key):
    out = {}
    for c in dict.fromkeys(it["cond"] for it in S["items"]):
        d = {it["p"]: i for i, it in enumerate(S["items"]) if it["cond"] == c}
        out[c] = S[key][[d[p] for p in range(NP)]]
    return out
X = {**gather(A, "self"), **gather(Bd, "self")}
GPW = [w for w, m in PAIRS["persona"]]; GPM = [m for w, m in PAIRS["persona"]]
TW = [a for a, b in PAIRS["trait"]]; TBl = [b for a, b in PAIRS["trait"]]
NB = 300; rng = np.random.default_rng(9)
WA = np.zeros((NB, NP)); WB = np.zeros((NB, NP))
for r in range(NB):
    p = rng.permutation(NP); a, b = p[:12], p[12:]
    WA[r] = np.bincount(rng.choice(a, 12), minlength=NP) / 12; WB[r] = np.bincount(rng.choice(b, 12), minlength=NP) / 12
WA = torch.tensor(WA); WB = torch.tensor(WB)
out = {}
for l in COV_LAYERS:
    li = LAYERS.index(l)
    T = torch.cat([TA["toks"][l], TB["toks"][l]], 0).double(); T = T - T.mean(0)
    ev, V = torch.linalg.eigh((T.T @ T) / (T.shape[0] - 1)); V = V[:, torch.argsort(ev, descending=True)]
    Z = {c: v[:, li].double() @ V for c, v in X.items()}
    def means(W):
        M = {c: W @ z for c, z in Z.items()}
        M["W"] = sum(M[c] for c in GPW) / 4; M["M"] = sum(M[c] for c in GPM) / 4
        M["nh"] = (M["neutral_human"] + M["neutral_human2"]) / 2
        M["TPw"] = sum(M[c] for c in TW) / 4; M["TPb"] = sum(M[c] for c in TBl) / 4
        return M
    MA, MB = means(WA), means(WB)
    out[l] = {}
    for k in (0, 10, 30, 100):
        mask = torch.ones(V.shape[1], dtype=torch.float64); mask[:k] = 0
        def pos(Mf, Me, c):
            u = (Mf["W"] - Mf["M"]) * mask
            return ((Me[c] - Me["M"]) * u).sum(-1) / ((Me["W"] - Me["M"]) * u).sum(-1)
        r = {c: (0.5 * (pos(MA, MB, c) + pos(MB, MA, c))).numpy() for c in ("default", "nh", "AI_blunt", "AI_warm", "TPw", "TPb")}
        r["default-nh"] = r["default"] - r["nh"]; r["TPw-TPb"] = r["TPw"] - r["TPb"]; r["AIwarm-AIblunt"] = r["AI_warm"] - r["AI_blunt"]
        out[l][k] = {c: [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in r.items()}
    print(l, {k: {c: f"{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for c, v in d.items() if c in ("default", "nh", "default-nh", "TPw-TPb", "AIwarm-AIblunt")} for k, d in out[l].items()})
BAND = {"L20-25": [20, 25], "L30-35": [30, 35], "L40-50": [40, 45, 50]}
json.dump(out, open(f"{V3_RES}/V3_persona_axis_traits_02g_pcs_boot.json", "w"), indent=1)
