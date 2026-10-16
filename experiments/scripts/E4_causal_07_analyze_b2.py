# E4_causal_07 (CPU, offline): part (b) with multiple gender axes and neutral references (self.pt + self2.pt).
import json, math
import numpy as np, torch
import torch.nn.functional as F
C = "./experiments/results/E4_causal_cache"
OUT = "./experiments/results"
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
rng = np.random.default_rng(1)
S1 = torch.load(f"{C}/self.pt"); S2 = torch.load(f"{C}/self2.pt"); U = torch.load(f"{C}/directions.pt")
NP = 24
def gather(S, key):
    items = S["meta"]["items"]; out = {}
    for c in sorted(set(it["cond"] for it in items)):
        d = {it["p"]: i for i, it in enumerate(items) if it["cond"] == c}
        out[c] = S[key][[d[p] for p in range(NP)]]
    return out
res = {}
for loc in ("self", "bg"):
    X = {**gather(S1, f"{loc}_post"), **gather(S2, f"{loc}_post")}
    X = {c: v.double() for c, v in X.items()}
    R = {**gather(S1, f"{loc}_jpp_pron"), **gather(S2, f"{loc}_jpp_pron")}
    CONDS = list(X.keys())
    AXES = {"persona_named(woman-man)": ("woman", "man"), "persona_neutralname(woman_nn-man_nn)": ("woman_nn", "man_nn"),
            "AI_explicit(AI_female-AI_male)": ("AI_female", "AI_male"), "AI_named(Sarah_AI-James_AI)": ("Sarah_AI", "James_AI")}
    proj = {}
    for an, (cf, cm) in AXES.items():
        D = X[cf] - X[cm]; tot = D.sum(0)
        P = {c: np.zeros((NP, 60)) for c in CONDS}
        for p in range(NP):
            u = (tot - D[p]) / (NP - 1); u = u / u.norm(dim=-1, keepdim=True)
            for c in CONDS:
                P[c][p] = (X[c][p] * u).sum(-1).numpy()
        proj[an] = (P, cf, cm)
    uc = U["u_cont"].double(); uc = uc / uc.norm(dim=-1, keepdim=True)
    Pc = {c: (X[c] * uc.unsqueeze(0)).sum(-1).numpy() for c in CONDS}
    proj["third_person_prose(u_cont)|scale woman-man"] = (Pc, "woman", "man")
    proj["third_person_prose(u_cont)|scale AI_female-AI_male"] = (Pc, "AI_female", "AI_male")
    Rj = {c: v.numpy() for c, v in R.items()}
    proj["J++ readout pron|scale woman-man"] = (Rj, "woman", "man")
    proj["J++ readout pron|scale AI_female-AI_male"] = (Rj, "AI_female", "AI_male")
    out = {}
    for an, (P, cf, cm) in proj.items():
        o = {"per_layer": {}, "band": {}}
        for c in CONDS:
            o["per_layer"][c] = ((P[c] - P[cm]).mean(0) / (P[cf] - P[cm]).mean(0)).tolist()
        for b, r in BANDS.items():
            r = list(r); o["band"][b] = {}
            x = {c: P[c][:, r].mean(1) for c in CONDS}
            ii_all = rng.integers(0, NP, size=(2000, NP))
            for c in CONDS:
                f = lambda ii: (x[c][ii] - x[cm][ii]).mean() / (x[cf][ii] - x[cm][ii]).mean()
                bs = np.array([f(ii) for ii in ii_all])
                o["band"][b][c] = [float(f(np.arange(NP))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[an] = o
    res[loc] = out
    # cosines between axis directions (full-sample)
    if loc == "self":
        dirs = {an: (X[cf] - X[cm]).mean(0) for an, (cf, cm) in AXES.items()}
        dirs["u_cont"] = U["u_cont"].double()
        dirs["neutral_human-midpoint(W,M)"] = X["neutral_human"].mean(0) - 0.5 * (X["woman_nn"].mean(0) + X["man_nn"].mean(0))
        dirs["default-neutral_human"] = X["default"].mean(0) - X["neutral_human"].mean(0)
        names = list(dirs.keys()); cm_ = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cm_[f"{a} | {b}"] = F.cosine_similarity(dirs[a], dirs[b], dim=-1).tolist()
        res["axis_cos"] = cm_
        # magnitudes: |W-M| vs |neutral - mid| vs |default - neutral|
        res["norms"] = {k: v.norm(dim=-1).tolist() for k, v in dirs.items()}
json.dump(res, open(f"{OUT}/E4_causal_07_b2.json", "w"))
ORDER = ["default", "nameless_AI", "Gemma", "Atlas_AI", "Sarah_AI", "James_AI", "AI_female", "AI_male", "neutral_human", "nonbinary_human", "woman_nn", "man_nn", "woman", "man"]
for loc in ("self", "bg"):
    print(f"\n######## positions at {loc} tokens (0 = male pole of axis, 1 = female pole) ########")
    for an, o in res[loc].items():
        print(f"--- axis: {an}")
        for c in ORDER:
            print(f"   {c:16s} " + " ".join(f"{b}:{o['band'][b][c][0]:5.2f}[{o['band'][b][c][1]:5.2f},{o['band'][b][c][2]:5.2f}]" for b in BANDS))
print("\naxis cosines (band means):")
for k, v in res["axis_cos"].items():
    print(f"  {k:75s} " + " ".join(f"{b}:{np.mean([v[l] for l in r]):6.3f}" for b, r in BANDS.items()))
print("norms (band means):")
for k, v in res["norms"].items():
    print(f"  {k:45s} " + " ".join(f"{b}:{np.mean([v[l] for l in r]):7.2f}" for b, r in BANDS.items()))
