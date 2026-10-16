# E4_causal_04 (CPU, offline): parts (a) and (b) from cached activations. Run with the project venv python.
import json, math, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "./experiments/scripts")
CACHE = "./experiments/results/E4_causal_cache"
OUT = "./experiments/results"
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
rng = np.random.default_rng(0)
S = torch.load(f"{CACHE}/self.pt"); T3 = torch.load(f"{CACHE}/third.pt"); PB = torch.load(f"{CACHE}/pullbacks.pt")
items = S["meta"]["items"]; NP = 24
def idx(cond):
    d = {it["p"]: i for i, it in enumerate(items) if it["cond"] == cond}
    return [d[p] for p in range(NP)]
CONDS = ["woman", "man", "default", "nameless_AI", "Gemma", "Atlas_AI", "Sarah_AI", "James_AI"]
I = {c: idx(c) for c in CONDS}
X = {k: S[k].double() for k in ("self_post", "bg_post", "user_post")}  # [N,60,D]
t3 = T3["meta"]["items"]
iF = [i for i, it in enumerate(t3) if it["cond"] == "F"]; iM = [i for i, it in enumerate(t3) if it["cond"] == "M"]
Y = {k: T3[k].double() for k in ("ref_post", "cont_post", "last_post")}

def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)

def dirs_from(A, B):  # A,B [n,60,D] paired -> mean diff [60,D]
    return (A - B).mean(0)

res = {"layers": list(range(60))}
# ---------------- (a) directions
u_self = dirs_from(X["self_post"][I["woman"]], X["self_post"][I["man"]])
u_bg = dirs_from(X["bg_post"][I["woman"]], X["bg_post"][I["man"]])
u_user = dirs_from(X["user_post"][I["woman"]], X["user_post"][I["man"]])
u_ref = dirs_from(Y["ref_post"][iF], Y["ref_post"][iM])
u_cont = dirs_from(Y["cont_post"][iF], Y["cont_post"][iM])
u_last = dirs_from(Y["last_post"][iF], Y["last_post"][iM])
torch.save({"u_self": u_self.float(), "u_bg": u_bg.float(), "u_ref": u_ref.float(), "u_cont": u_cont.float(),
            "u_last": u_last.float()}, f"{CACHE}/directions.pt")

def split_half(A, B, nsplit=200, null=False):
    n = A.shape[0]; out = []
    D = A - B
    for s in range(nsplit):
        perm = rng.permutation(n); h1, h2 = perm[: n // 2], perm[n // 2:]
        if null:  # random sign flips per pair -> no consistent direction
            sg = torch.tensor(rng.choice([-1.0, 1.0], size=n), dtype=D.dtype).view(n, 1, 1)
            Dd = D * sg
        else:
            Dd = D
        out.append(cos(Dd[h1].mean(0), Dd[h2].mean(0)).numpy())
    out = np.array(out)
    return out.mean(0), np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)

sh = {}
for name, (A, B) in {"self": (X["self_post"][I["woman"]], X["self_post"][I["man"]]),
                     "bg": (X["bg_post"][I["woman"]], X["bg_post"][I["man"]]),
                     "third_cont": (Y["cont_post"][iF], Y["cont_post"][iM]),
                     "third_ref": (Y["ref_post"][iF], Y["ref_post"][iM]),
                     "third_last": (Y["last_post"][iF], Y["last_post"][iM])}.items():
    m, lo, hi = split_half(A, B)
    mn, _, hin = split_half(A, B, null=True)
    sh[name] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "null_mean": mn.tolist(), "null_hi": hin.tolist()}
# name split for self: names idx p%6 in {0,1,2} vs {3,4,5}
gA = [p for p in range(NP) if p % 6 < 3]; gB = [p for p in range(NP) if p % 6 >= 3]
DW = X["self_post"][I["woman"]] - X["self_post"][I["man"]]
sh["self_namesplit"] = cos(DW[gA].mean(0), DW[gB].mean(0)).tolist()
res["split_half"] = sh

# cross-source cosines
pb = {k: PB[k].double() for k in ("pron_jpp", "pron_jlens", "noun_jpp", "noun_jlens")}
pb["pron_naive"] = PB["pron_final"].double().unsqueeze(0).expand(60, -1)
vjp_mean = PB["vjp_pron"].double().mean(0)
cs = {}
for a_name, a in {"self": u_self, "bg": u_bg, "user": u_user, "third_cont": u_cont, "third_ref": u_ref, "third_last": u_last}.items():
    for b_name, b in {"self": u_self, "third_cont": u_cont, "third_ref": u_ref, "third_last": u_last, "bg": u_bg,
                      "pb_pron_jpp": pb["pron_jpp"], "pb_pron_jlens": pb["pron_jlens"], "pb_noun_jpp": pb["noun_jpp"],
                      "pb_pron_naive": pb["pron_naive"], "vjp_pron_mean": vjp_mean}.items():
        if a_name == b_name: continue
        cs[f"{a_name}__{b_name}"] = cos(a, b).tolist()
# null for cos with a fixed direction in D=5376: ~N(0, 1/sqrt(D)) => 95% |cos| < 0.027
res["cos"] = cs
res["cos_null_95"] = 1.96 / math.sqrt(5376)
# disattenuation using split-half reliabilities r_half -> full-sample reliability rho = 2r/(1+r); cos_true ~ cos/sqrt(rho_a*rho_b)
def rel(name):
    r = np.clip(np.array(sh[name]["mean"]), 1e-6, 1); return 2 * r / (1 + r)
res["cos_self_cont_disatt"] = (np.array(cs["self__third_cont"]) / np.sqrt(rel("self") * rel("third_cont"))).tolist()
res["cos_self_last_disatt"] = (np.array(cs["self__third_last"]) / np.sqrt(rel("self") * rel("third_last"))).tolist()

# separation (d') along u and along J++ pullback: how cleanly does each direction separate woman/man personas (LOO)?
def loo_proj(A_all, Bw, Bm, conds_idx, Xsrc):
    """leave-one-passage-out unit direction from woman-man self diffs; returns {cond: [NP,60]} projections."""
    D = Bw - Bm  # [NP,60,D]
    tot = D.sum(0)
    out = {c: np.zeros((NP, 60)) for c in conds_idx}
    for p in range(NP):
        u = (tot - D[p]) / (NP - 1)
        u = u / u.norm(dim=-1, keepdim=True)
        for c, ii in conds_idx.items():
            out[c][p] = (Xsrc[ii[p]] * u).sum(-1).numpy()
    return out

def proj_fixed(Xsrc, conds_idx, U):
    U = U / U.norm(dim=-1, keepdim=True)
    return {c: (Xsrc[ii] * U.unsqueeze(0)).sum(-1).numpy() for c, ii in conds_idx.items()}

def position(P, c, layers):
    """P {cond:[NP,60]} -> bootstrap ratio-of-means position of cond on man(0)..woman(1) scale for band."""
    x = np.stack([P[c][:, layers].mean(1), P["man"][:, layers].mean(1), P["woman"][:, layers].mean(1)], 1)  # [NP,3]
    def f(a):
        return (a[:, 0] - a[:, 1]).mean() / (a[:, 2] - a[:, 1]).mean()
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, NP, NP); bs.append(f(x[ii]))
    return [float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def position_layer(P, c):
    return ((P[c] - P["man"]).mean(0) / (P["woman"] - P["man"]).mean(0)).tolist()

def gap_dprime(P, layers):
    d = (P["woman"][:, layers].mean(1) - P["man"][:, layers].mean(1))
    return float(d.mean() / d.std(ddof=1))

readouts = {}
for loc in ("self", "bg"):
    Xs = X[f"{loc}_post"]
    projs = {
        "u_self_LOO": loo_proj(None, X["self_post"][I["woman"]], X["self_post"][I["man"]], I, Xs),
        "u_third_cont": proj_fixed(Xs, I, u_cont),
        "u_third_last": proj_fixed(Xs, I, u_last),
        "pb_pron_jpp_dir": proj_fixed(Xs, I, pb["pron_jpp"]),
    }
    # lens readouts (nonlinear, per-token averaged) from collection
    for kind in ("jpp", "naive"):
        for sc in ("pron", "noun"):
            arr = S[f"{loc}_{kind}_{sc}"].numpy()
            projs[f"readout_{kind}_{sc}"] = {c: arr[ii] for c, ii in I.items()}
    out = {}
    for pname, P in projs.items():
        out[pname] = {"per_layer_position": {c: position_layer(P, c) for c in CONDS if c not in ("woman", "man")},
                      "band_position": {b: {c: position(P, c, list(r)) for c in CONDS if c not in ("woman", "man")} for b, r in BANDS.items()},
                      "band_dprime_woman_vs_man": {b: gap_dprime(P, list(r)) for b, r in BANDS.items()},
                      "band_raw_means": {b: {c: float(P[c][:, list(r)].mean()) for c in CONDS} for b, r in BANDS.items()}}
    readouts[loc] = out
res["positions"] = readouts
# self vs bg separation along u_self (LOO): ratio of woman-man gap at self tokens vs background tokens
Ps = readouts["self"]["u_self_LOO"]["band_raw_means"]; Pb = readouts["bg"]["u_self_LOO"]["band_raw_means"]
res["self_vs_bg_gap_ratio"] = {b: (Ps[b]["woman"] - Ps[b]["man"]) / (Pb[b]["woman"] - Pb[b]["man"]) for b in BANDS}
# norms
res["u_self_norm_over_resid_norm"] = (u_self.norm(dim=-1) / S["norms"].double()[I["woman"] + I["man"]].mean(0)).tolist()
res["u_cont_norm_over_resid_norm"] = (u_cont.norm(dim=-1) / T3["norms"].double().mean(0)).tolist()
json.dump(res, open(f"{OUT}/E4_causal_04_ab.json", "w"))

# ---------------- print summary
def band(v): return {b: round(float(np.mean([v[l] for l in r])), 3) for b, r in BANDS.items()}
print("split-half cos (mean):")
for k in ("self", "bg", "third_cont", "third_ref", "third_last"):
    print(f"  {k:12s}", band(sh[k]["mean"]), " null:", band(sh[k]["null_mean"]))
print("  self_namesplit", band(sh["self_namesplit"]))
for k in ("self__third_cont", "self__third_last", "self__third_ref", "self__bg", "third_cont__third_ref", "self__pb_pron_jpp", "self__pb_pron_jlens", "self__pb_noun_jpp", "self__pb_pron_naive", "self__vjp_pron_mean", "third_cont__pb_pron_jpp", "third_last__pb_pron_jpp", "third_last__vjp_pron_mean"):
    print(f"cos {k:28s}", band(cs[k]))
print("disatt self-cont", band(res["cos_self_cont_disatt"]), "self-last", band(res["cos_self_last_disatt"]))
print("u_self norm / resid norm", band(res["u_self_norm_over_resid_norm"]))
print("u_cont norm / resid norm", band(res["u_cont_norm_over_resid_norm"]))
print("self/bg gap ratio along u_self", {b: round(v, 2) for b, v in res["self_vs_bg_gap_ratio"].items()})
for loc in ("self", "bg"):
    print(f"\n=== positions at {loc} tokens (0=man persona, 1=woman persona) ===")
    for pname, o in readouts[loc].items():
        print(f"-- {pname}   d'(W vs M): " + " ".join(f"{b}:{v:.1f}" for b, v in o["band_dprime_woman_vs_man"].items()))
        for c in ("default", "nameless_AI", "Gemma", "Atlas_AI", "Sarah_AI", "James_AI"):
            print(f"   {c:12s} " + " ".join(f"{b}:{o['band_position'][b][c][0]:.2f}[{o['band_position'][b][c][1]:.2f},{o['band_position'][b][c][2]:.2f}]" for b in BANDS))
