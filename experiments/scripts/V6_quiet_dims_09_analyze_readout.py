# V6_quiet_dims_09_analyze_readout (local CPU): condition separation / background / IT-alignment under quiet-dim masking.
import torch, numpy as np, json
R = "./experiments/results/"
D = torch.load(R + "V6_quiet_dims_06_readout.pt")
meta = D["meta"]; LAYS = D["layers"]; VARS = D["vars"]; CONDS = D["conds"]
NP = 24
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
bidx = {b: [LAYS.index(l) for l in r] for b, r in BANDS.items()}
rng = np.random.default_rng(0)
labs = np.array([m["lab"] for m in meta]); conds = np.array([m["cond"] for m in meta]); qis = np.array([m["qi"] for m in meta])


def passage_means(X, lab):
    """X [N, nL, ...] -> [C, NP, nL, ...] mean over positions with label lab (np)."""
    X = X.numpy() if torch.is_tensor(X) else X
    out = np.full((len(CONDS), NP) + X.shape[1:], np.nan)
    for ci, c in enumerate(CONDS):
        for q in range(NP):
            m = (conds == c) & (qis == q) & (labs == lab)
            if m.any():
                out[ci, q] = X[m].mean(0)
    return out


ci = {c: i for i, c in enumerate(CONDS)}
res = {}
for lab in ("self", "nonself"):
    for kind in ("jpp", "jlens"):
        P = passage_means(D["SC"][kind], lab)  # [C, NP, nL, V, 2]
        W = (P[ci["woman_p1"]] + P[ci["woman_p2"]]) / 2; M = (P[ci["man_p1"]] + P[ci["man_p2"]]) / 2
        for mi, met in enumerate(("pron", "noun")):
            for b, li in bidx.items():
                rec = {}
                for vi, vn in enumerate(VARS):
                    w = W[:, li, vi, mi].mean(1); m = M[:, li, vi, mi].mean(1)
                    x = {c: P[ci[c]][:, li, vi, mi].mean(1) for c in ("default", "nameless", "gemma_short")}
                    gap = w - m
                    def stat(fn):
                        v0 = fn(np.arange(NP)); bs = [fn(rng.integers(0, NP, NP)) for _ in range(1000)]
                        return [float(v0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    rec[vn] = {"gap": stat(lambda i: gap[i].mean()),
                               "pos_default": stat(lambda i: (x["default"][i].mean() - m[i].mean()) / gap[i].mean()),
                               "pos_nameless": stat(lambda i: (x["nameless"][i].mean() - m[i].mean()) / gap[i].mean()),
                               "pos_gemma": stat(lambda i: (x["gemma_short"][i].mean() - m[i].mean()) / gap[i].mean()),
                               "def_minus_nameless_frac": stat(lambda i: (x["default"][i] - x["nameless"][i]).mean() / gap[i].mean()),
                               "gemma_minus_nameless_frac": stat(lambda i: (x["gemma_short"][i] - x["nameless"][i]).mean() / gap[i].mean()),
                               "raw_default": stat(lambda i: x["default"][i].mean()),
                               "raw_woman": stat(lambda i: w[i].mean()), "raw_man": stat(lambda i: m[i].mean())}
                # retention of the gap under each variant (paired)
                gf = (W[:, li, 0, mi] - M[:, li, 0, mi]).mean(1)
                for vi, vn in enumerate(VARS):
                    gv = (W[:, li, vi, mi] - M[:, li, vi, mi]).mean(1)
                    bs = []
                    for _ in range(1000):
                        i = rng.integers(0, NP, NP); bs.append(gv[i].mean() / gf[i].mean())
                    rec[vn]["gap_retention"] = [float(gv.mean() / gf.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                res[f"{lab}|{kind}|{met}|{b}"] = rec
# linear decomposition (pron, self tokens): components mu_loud, mu_quiet, dev_loud, dev_quiet
lin = {}
for kind in ("jpp", "jlens"):
    P = passage_means(D["LIN"][kind], "self")  # [C, NP, nL, 4]
    PN = passage_means(D["LIN"][kind], "nonself")
    W = (P[ci["woman_p1"]] + P[ci["woman_p2"]]) / 2; M = (P[ci["man_p1"]] + P[ci["man_p2"]]) / 2
    for b, li in bidx.items():
        g = (W[:, li] - M[:, li]).mean((0, 1))  # [4]
        dflt = P[ci["default"]][:, li].mean((0, 1)); dfn = PN[ci["default"]][:, li].mean((0, 1))
        dn = (P[ci["default"]][:, li] - P[ci["nameless"]][:, li]).mean((0, 1))
        gn = (P[ci["gemma_short"]][:, li] - P[ci["nameless"]][:, li]).mean((0, 1))
        lin[f"{kind}|{b}"] = {"gap_parts": g.tolist(), "gap_quiet_share": float(g[3] / (g[2] + g[3])),
                              "default_self_parts": dflt.tolist(), "default_nonself_parts": dfn.tolist(),
                              "def_minus_nameless_parts": dn.tolist(), "gemma_minus_nameless_parts": gn.tolist()}
# IT-gradient readout positions (self tokens), full vs mean10; and J++ positions for comparison
itr = {}
P = passage_means(D["ITR"], "self")  # [C, NP, nL, 3, 2]
W = (P[ci["woman_p1"]] + P[ci["woman_p2"]]) / 2; M = (P[ci["man_p1"]] + P[ci["man_p2"]]) / 2
for gi, gk in enumerate(("gbar", "gbar_chat", "gbar_prose")):
    for b, li in bidx.items():
        rec = {}
        for vi, vn in enumerate(("full", "mean10")):
            w = W[:, li, gi, vi].mean(1); m = M[:, li, gi, vi].mean(1)
            for c in ("default", "nameless", "gemma_short"):
                x = P[ci[c]][:, li, gi, vi].mean(1)
                bs = [((x[i].mean() - m[i].mean()) / (w[i] - m[i]).mean()) for i in (rng.integers(0, NP, NP) for _ in range(1000))]
                rec[f"{vn}|{c}"] = [float((x.mean() - m.mean()) / (w - m).mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            rec[f"{vn}|gap"] = float((w - m).mean())
        itr[f"{gk}|{b}"] = rec
# paired comparison J++ default position minus IT(gbar all) default position, full vs mean10, bootstrap over passages
cmp = {}
PJ = passage_means(D["SC"]["jpp"], "self")
WJ = (PJ[ci["woman_p1"]] + PJ[ci["woman_p2"]]) / 2; MJ = (PJ[ci["man_p1"]] + PJ[ci["man_p2"]]) / 2
for b, li in bidx.items():
    for vi_j, vn in ((0, "full"), (1, "mean10")):
        vi_i = 0 if vn == "full" else 1
        def posJ(i): return (PJ[ci["default"]][i][:, li, vi_j, 0].mean() - MJ[i][:, li, vi_j, 0].mean()) / (WJ[i][:, li, vi_j, 0] - MJ[i][:, li, vi_j, 0]).mean()
        def posI(i, gi): return (P[ci["default"]][i][:, li, gi, vi_i].mean() - M[i][:, li, gi, vi_i].mean()) / (W[i][:, li, gi, vi_i] - M[i][:, li, gi, vi_i]).mean()
        for gi, gk in enumerate(("gbar", "gbar_chat", "gbar_prose")):
            d0 = posJ(np.arange(NP)) - posI(np.arange(NP), gi)
            bs = []
            for _ in range(1000):
                i = rng.integers(0, NP, NP); bs.append(posJ(i) - posI(i, gi))
            cmp[f"{b}|{vn}|{gk}"] = [float(d0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# variance of per-token pron score explained by quiet coords (all labels): var(full - mean10)/var(full) per band
vshare = {}
for kind in ("jpp", "jlens"):
    X = D["SC"][kind].numpy()
    for b, li in bidx.items():
        f = X[:, li, 0, 0]; mm = X[:, li, 1, 0]
        vshare[f"{kind}|{b}"] = {"var_ratio_quietpart": float(np.var(f - mm) / np.var(f)), "corr_full_mean10": float(np.corrcoef(f.ravel(), mm.ravel())[0, 1])}
BG = {k: v for k, v in D["BG"].items()}
bg = {kind: {b: {"mu_full_pron": float(np.mean([BG[kind][LAYS.index(l)][0][0] for l in r])), "mu_zero10_pron": float(np.mean([BG[kind][LAYS.index(l)][1][0] for l in r])),
                 "mu_full_noun": float(np.mean([BG[kind][LAYS.index(l)][0][1] for l in r])), "mu_zero10_noun": float(np.mean([BG[kind][LAYS.index(l)][1][1] for l in r]))}
             for b, r in BANDS.items()} for kind in BG}
out = {"readout": res, "linear": lin, "it": itr, "jpp_minus_it_default": cmp, "token_var": vshare, "mu_background": bg}
json.dump(out, open(R + "V6_quiet_dims_09_readout_summary.json", "w"), indent=0)

# ------------------------------------------------------------------ print
def f3(v): return f"{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]"
for lab in ("self",):
    for kind in ("jpp", "jlens"):
        for met in ("pron", "noun"):
            print(f"\n=== {lab} tokens, {kind}, {met}")
            for b in BANDS:
                rec = res[f"{lab}|{kind}|{met}|{b}"]
                for vn in VARS:
                    r = rec[vn]
                    print(f" {b} {vn:8s} gap {r['gap'][0]:6.2f} ret {f3(r['gap_retention'])} | pos def {f3(r['pos_default'])} nameless {r['pos_nameless'][0]:+.2f} gemma {r['pos_gemma'][0]:+.2f} | def-nl {f3(r['def_minus_nameless_frac'])} gem-nl {f3(r['gemma_minus_nameless_frac'])} | raw def {r['raw_default'][0]:+.2f} W {r['raw_woman'][0]:+.2f} M {r['raw_man'][0]:+.2f}")
print("\n=== linear decomposition of pron (self tokens) [mu_loud, mu_quiet, dev_loud, dev_quiet]")
for k, v in lin.items():
    print(f" {k:14s} gap parts {np.round(v['gap_parts'], 2)} quiet share {v['gap_quiet_share']:+.2f} | default self {np.round(v['default_self_parts'], 2)} nonself {np.round(v['default_nonself_parts'], 2)} | def-nl {np.round(v['def_minus_nameless_parts'], 2)} gem-nl {np.round(v['gemma_minus_nameless_parts'], 2)}")
print("\n=== IT-gradient readout positions (self tokens)")
for k, v in itr.items():
    print(f" {k:18s} full: def {f3(v['full|default'])} nl {v['full|nameless'][0]:+.2f} gem {v['full|gemma_short'][0]:+.2f} gap {v['full|gap']:.2f} | mean10: def {f3(v['mean10|default'])} nl {v['mean10|nameless'][0]:+.2f} gem {v['mean10|gemma_short'][0]:+.2f} gap {v['mean10|gap']:.2f}")
print("\n=== J++ default position minus IT default position (paired bootstrap)")
for k, v in cmp.items():
    print(f" {k:28s} {f3(v)}")
print("\n=== per-token pron variance from quiet part", json.dumps(vshare))
print("\n=== mu background", json.dumps(bg))
