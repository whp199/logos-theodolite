# V4_latent_identity_06b_default_corr (CPU): within the default (no system prompt) condition, does the mid-layer lens ' Gemma'
# at identity slots predict whether the model actually emits ' Gemma'? NAME slots (model names itself, P~0.2) vs META3P slots
# ("...third person:\n\nMeet ___", model writes "the/an AI", P~0.007). Spearman over 48 prompts, bootstrap CI.
import json, numpy as np
R = "./experiments/results"
def rankdata(a):
    o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a)); 
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.bincount(inv, r) / cnt
    return s[inv]
def spear(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
z = np.load(f"{R}/V4_latent_identity_04_slots.npz"); m = json.load(open(f"{R}/V4_latent_identity_04_slots_meta.json"))
LP = z["LP"]; LAY = list(z["layers"]); K = list(z["kinds"]); T = list(z["toks"])
rows = m["rows"]; pers = np.array([r["persona"] for r in rows]); slot = np.array([r["slot"] for r in rows])
G = T.index(" Gemma"); rng = np.random.default_rng(5); out = {}
msk = (pers == "none") & np.isin(slot, ["name", "meta3p"]); sl = slot[msk]
for kind in K:
    k = K.index(kind); y = LP[msk][:, -1, k, G]
    for band in [(36, 39), (42, 48), (51, 55)]:
        li = [LAY.index(l) for l in range(band[0], band[1] + 1)]
        x = LP[msk][:, li, k, G].mean(1)
        r = spear(x, y); idx = np.arange(len(x))
        bs = [spear(x[s], y[s]) for s in (rng.choice(idx, len(idx)) for _ in range(1000))]
        dx = x[sl == "name"].mean() - x[sl == "meta3p"].mean()
        dy = y[sl == "name"].mean() - y[sl == "meta3p"].mean()
        key = f"{kind}|L{band[0]}-{band[1]}"
        out[key] = {"spearman": r, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                    "mid_name_minus_meta": float(dx), "out_name_minus_meta": float(dy)}
        print(f"{key:14s} spearman {r:+.2f} [{out[key]['ci'][0]:+.2f},{out[key]['ci'][1]:+.2f}]  mid logP(Gemma) name-meta {dx:+.2f}  output name-meta {dy:+.2f}")
mn = (pers == "none") & (slot == "name")
out["default_name_P"] = np.exp(LP[mn][:, -1, 0, G]).round(3).tolist()
out["default_name_top1"] = [m["top"][i][0] for i in np.where(mn)[0]]
print("default name-slot P(Gemma):", out["default_name_P"])
print("top1:", out["default_name_top1"])
json.dump(out, open(f"{R}/V4_latent_identity_06b_default_corr.json", "w"), indent=1)
