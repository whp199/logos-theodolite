# V3_persona_axis_traits_02f (offline CPU): WHERE in the token-covariance spectrum does the AI offset meet the persona axis?
# At each covariance layer, rotate into token-PC coordinates (descending variance). Report
#  (i) cumulative share of <AIoff, u_persona> and of ||persona||^2 carried by the top-k PCs,
#  (ii) cross-fitted default / neutral / trait-persona positions on the persona axis after REMOVING the top-k PCs
#      (k = 0, 1, 3, 10, 30, 100, 300, 1000) or KEEPING only the top-k.
# Prediction (written before running): if the female-side placement comes from a few dominant shared context directions,
#  removing the top <=30 PCs brings default to ~nh (|default-nh| < 0.15 of gap) while W/M stay separated.
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
rng = np.random.default_rng(3)
splits = []
for _ in range(10):
    p = rng.permutation(NP); splits.append((p[:12], p[12:]))
KS = [0, 1, 3, 10, 30, 100, 300, 1000]
out = {}
for l in COV_LAYERS:
    li = LAYERS.index(l)
    T = torch.cat([TA["toks"][l], TB["toks"][l]], 0).double(); T = T - T.mean(0)
    ev, V = torch.linalg.eigh((T.T @ T) / (T.shape[0] - 1))
    order = torch.argsort(ev, descending=True); ev = ev[order]; V = V[:, order]
    Z = {c: (v[:, li].double() @ V) for c, v in X.items()}       # [24, D] in PC coords
    def means(idx):
        M = {c: z[idx].mean(0) for c, z in Z.items()}
        M["W"] = sum(M[c] for c in GPW) / 4; M["M"] = sum(M[c] for c in GPM) / 4
        M["TPw"] = sum(M[c] for c in TW) / 4; M["TPb"] = sum(M[c] for c in TBl) / 4
        M["RLs"] = (M["RL1_serve"] + M["RL2_serve"]) / 2; M["RLo"] = (M["RL1_solo"] + M["RL2_solo"]) / 2
        M["nh"] = (M["neutral_human"] + M["neutral_human2"]) / 2
        return M
    Mall = means(np.arange(NP))
    up = Mall["W"] - Mall["M"]; off = Mall["default"] - Mall["nh"]; tr = Mall["TPw"] - Mall["TPb"]
    prod = (off * up); gapsq = up * up; tprod = tr * up
    cum = lambda v: (torch.cumsum(v, 0) / v.sum()).numpy()
    r = {"ev_frac_top": {k: float(ev[:k].sum() / ev.sum()) for k in KS[1:]},
         "cum_share_AIoff.persona": {k: float(cum(prod)[k - 1]) for k in KS[1:]},
         "cum_share_persona_norm2": {k: float(cum(gapsq)[k - 1]) for k in KS[1:]},
         "cum_share_trait.persona": {k: float(cum(tprod)[k - 1]) for k in KS[1:]},
         "AIoff.persona/gap^2_total": float(prod.sum() / gapsq.sum())}
    for mode in ("remove_top", "keep_top"):
        r[mode] = {}
        for k in KS:
            if mode == "keep_top" and k == 0:
                continue
            mask = torch.ones(V.shape[1], dtype=torch.float64)
            if mode == "remove_top":
                mask[:k] = 0
            else:
                mask[k:] = 0
            pos = {c: [] for c in ["default", "nameless_AI", "Gemma", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "TPw", "TPb", "RLs", "RLo"]}
            for a, b in splits:
                for fit, ev_ in ((a, b), (b, a)):
                    Mf, Me = means(fit), means(ev_)
                    u = (Mf["W"] - Mf["M"]) * mask
                    den = ((Me["W"] - Me["M"]) * u).sum()
                    for c in pos:
                        pos[c].append(float(((Me[c] - Me["M"]) * u).sum() / den))
            r[mode][k] = {c: float(np.mean(v)) for c, v in pos.items()}
    out[l] = r
    print(f"L{l}: top-k eig frac", {k: round(v, 3) for k, v in r["ev_frac_top"].items()})
    print("   cum share of <AIoff,persona> in top-k PCs", {k: round(v, 2) for k, v in r["cum_share_AIoff.persona"].items()})
    print("   cum share of |persona|^2 in top-k PCs   ", {k: round(v, 2) for k, v in r["cum_share_persona_norm2"].items()})
    print("   cum share of <trait,persona> in top-k   ", {k: round(v, 2) for k, v in r["cum_share_trait.persona"].items()})
    for mode in ("remove_top", "keep_top"):
        for k, d in r[mode].items():
            print(f"   {mode:10s} k={k:5d}: " + " ".join(f"{c}={v:+.2f}" for c, v in d.items()))
json.dump(out, open(f"{V3_RES}/V3_persona_axis_traits_02f_pcs.json", "w"), indent=1)
