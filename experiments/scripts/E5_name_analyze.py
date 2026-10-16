#!/usr/bin/env python3
"""E5_name local analysis (CPU, numpy): bootstrap CIs over frames (third-person) / passages (self), tables for the report.
Run: python3 experiments/scripts/E5_name_analyze.py  -> experiments/results/E5_name_analysis.json + printed tables."""
import json, sys
import numpy as np

R = "./experiments/results"
NAMES = ["Gemma", "Emma", "Jemma", "Sarah", "Gem", "James", "Liam", "Llama", "Mistral", "Claude", "Qwen", "Gemini", "Phi", "Atlas"]
CONTEXTS = ["human", "ai_real", "ai_neutral", "bare"]
GENERIC = ["Llama", "Mistral", "Qwen"]
BANDS = {"L10-19": list(range(10, 20)), "L20-29": list(range(20, 30)), "L30-39": list(range(30, 40)),
         "L40-49": list(range(40, 50)), "L50-58": list(range(50, 59)), "L59": [59]}
NB = 2000
rng = np.random.default_rng(0)
OUT = {}


def boot(units, nb=NB, stat=np.mean):
    """units: array [n_units, ...]; resample units with replacement. Returns (est, lo, hi)."""
    u = np.asarray(units, float)
    n = u.shape[0]
    idx = rng.integers(0, n, (nb, n))
    bs = np.array([stat(u[i], axis=0) for i in idx])
    return stat(u, axis=0), np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def boot_fn(fn, n_units, nb=NB):
    """Generic paired bootstrap: fn(idx) -> scalar or array computed from resampled unit indices."""
    est = fn(np.arange(n_units))
    bs = np.array([fn(rng.integers(0, n_units, n_units)) for _ in range(nb)])
    return est, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def fmt(t, p=2):
    e, lo, hi = t
    return f"{float(e):+.{p}f} [{float(lo):+.{p}f},{float(hi):+.{p}f}]"


def trip(t):
    return [float(t[0]), float(t[1]), float(t[2])]


# =====================================================================================  job A: third person
A = np.load(f"{R}/E5_name_A_readouts.npz")
RA = A["R"]                        # [896, 60, 4, M]
mets = list(A["metrics"]); MI = {m: i for i, m in enumerate(mets)}
names, ctxs, frames = A["names"], A["ctx"], A["frame"]
nFr = int(frames.max()) + 1
assert RA.shape[0] == len(NAMES) * len(CONTEXTS) * nFr, RA.shape


def cell(arr, n, c):
    """Per-frame values (ordered by frame) for name n, context c."""
    m = (names == n) & (ctxs == c)
    o = np.argsort(frames[m])
    return arr[m][o]


beh = {k: A["beh_" + k] for k in ("p_f", "p_m", "p_it", "p_they", "lse_pron", "p_name")}
beh["mass4"] = beh["p_f"] + beh["p_m"] + beh["p_it"] + beh["p_they"]
beh["share_f_of_human"] = beh["p_f"] / (beh["p_f"] + beh["p_m"])
beh["share_it_of4"] = beh["p_it"] / beh["mass4"]
beh["share_human_of4"] = (beh["p_f"] + beh["p_m"]) / beh["mass4"]

print("\n=== A. BEHAVIOUR (real next-token), mean over 16 frames [95% bootstrap CI] ===")
print(f"{'name|ctx':20s} {'lse_pron F-M':>24s} {'P(she-set)':>8s} {'P(he-set)':>8s} {'P(it)':>7s} {'P(they)':>7s} {'P(name)':>7s} {'mass4':>6s}")
btab = {}
for n in NAMES:
    for c in CONTEXTS:
        row = {k: trip(boot(cell(beh[k], n, c))) for k in beh}
        btab[f"{n}|{c}"] = row
        print(f"{n+'|'+c:20s} {fmt(row['lse_pron']):>24s} {row['p_f'][0]:8.3f} {row['p_m'][0]:8.3f} {row['p_it'][0]:7.3f} "
              f"{row['p_they'][0]:7.3f} {row['p_name'][0]:7.3f} {row['mass4'][0]:6.3f}")
OUT["A_behaviour"] = btab

# woman-man scale (behaviour lse_pron): anchors Sarah|human (1) and James|human (0); alt anchors Emma/Liam
L = beh["lse_pron"]
def scale_pos(n, c, fa="Sarah", ma="James", ctx_anchor="human", arr=None):
    arr = L if arr is None else arr
    x, f, m = cell(arr, n, c), cell(arr, fa, ctx_anchor), cell(arr, ma, ctx_anchor)
    return boot_fn(lambda i: (x[i].mean() - m[i].mean()) / (f[i].mean() - m[i].mean()), nFr)
print("\n--- behaviour lse_pron on woman/man scale (0 = James|human, 1 = Sarah|human); alt = Emma/Liam anchors ---")
scal = {}
for n in NAMES:
    for c in CONTEXTS:
        s1 = scale_pos(n, c); s2 = scale_pos(n, c, "Emma", "Liam")
        scal[f"{n}|{c}"] = {"SarahJames": trip(s1), "EmmaLiam": trip(s2)}
    print(f"{n:8s} " + "  ".join(f"{c}:{scal[f'{n}|{c}']['SarahJames'][0]:+.2f}" for c in CONTEXTS))
OUT["A_behaviour_scale"] = scal

# de-gendering: paired over frames
print("\n--- de-gendering: lse_pron(AI ctx) - lse_pron(human ctx), and ratio, paired over frames ---")
deg = {}
for n in NAMES:
    h = cell(L, n, "human")
    for c in ("ai_real", "ai_neutral", "bare"):
        a = cell(L, n, c)
        d = boot(a - h)
        r = boot_fn(lambda i: a[i].mean() / h[i].mean(), nFr)
        dit = boot(cell(beh["share_human_of4"], n, c) - cell(beh["share_human_of4"], n, "human"))
        deg[f"{n}|{c}"] = {"delta_lse": trip(d), "ratio_lse": trip(r), "delta_human_share": trip(dit)}
    print(f"{n:8s} ai_real-human {fmt(deg[n+'|ai_real']['delta_lse'])} ratio {fmt(deg[n+'|ai_real']['ratio_lse'])} | "
          f"ai_neutral-human {fmt(deg[n+'|ai_neutral']['delta_lse'])} ratio {fmt(deg[n+'|ai_neutral']['ratio_lse'])} | "
          f"d human-pron share (ai_real) {fmt(deg[n+'|ai_real']['delta_human_share'])}")
OUT["A_degender"] = deg

# key contrasts
print("\n--- key paired contrasts (behaviour lse_pron) ---")
kc = {}
def contrast(label, xs, ys):
    """mean over frames of (mean of xs cells) - (mean of ys cells); each is list of (name, ctx)."""
    X = np.mean([cell(L, n, c) for n, c in xs], 0); Y = np.mean([cell(L, n, c) for n, c in ys], 0)
    kc[label] = trip(boot(X - Y)); print(f"{label:60s} {fmt(kc[label])}")
gen_real = [(g, "ai_real") for g in GENERIC]
contrast("Gemma|ai_real - genericAI|ai_real", [("Gemma", "ai_real")], gen_real)
contrast("Gemma|ai_real - Gemini|ai_real", [("Gemma", "ai_real")], [("Gemini", "ai_real")])
contrast("Gemma|ai_neutral - genericAI|ai_neutral", [("Gemma", "ai_neutral")], [(g, "ai_neutral") for g in GENERIC])
contrast("Gemma|ai_real - Gemma|ai_neutral (Google effect)", [("Gemma", "ai_real")], [("Gemma", "ai_neutral")])
contrast("Gemma|ai_real - Emma|ai_real", [("Gemma", "ai_real")], [("Emma", "ai_real")])
contrast("Gemma|ai_neutral - Emma|ai_neutral", [("Gemma", "ai_neutral")], [("Emma", "ai_neutral")])
contrast("Gemma|ai_neutral - Jemma|ai_neutral", [("Gemma", "ai_neutral")], [("Jemma", "ai_neutral")])
contrast("Gemma|ai_neutral - Sarah|ai_neutral", [("Gemma", "ai_neutral")], [("Sarah", "ai_neutral")])
contrast("Gemma|human - Emma|human", [("Gemma", "human")], [("Emma", "human")])
contrast("Gemma|human - Sarah|human", [("Gemma", "human")], [("Sarah", "human")])
contrast("Claude|ai_real - genericAI|ai_real", [("Claude", "ai_real")], gen_real)
contrast("Claude|ai_neutral - James|ai_neutral", [("Claude", "ai_neutral")], [("James", "ai_neutral")])
# mirror test: (Gemma_ai - generic) + (Claude_ai - generic) ~ 0 if symmetric
X = cell(L, "Gemma", "ai_real") + cell(L, "Claude", "ai_real") - 2 * np.mean([cell(L, g, "ai_real") for g in GENERIC], 0)
kc["mirror: (Gemma-gen)+(Claude-gen) ai_real"] = trip(boot(X)); print(f"{'mirror: (Gemma-gen)+(Claude-gen) ai_real':60s} {fmt(kc['mirror: (Gemma-gen)+(Claude-gen) ai_real'])}")
# mirror relative to the human context: retained fraction of human-context gender, Gemma vs Claude
for n in ("Gemma", "Claude", "Sarah", "Emma", "James", "Liam", "Jemma", "Gem", "Atlas"):
    a, h = cell(L, n, "ai_real"), cell(L, n, "human")
    g = np.mean([cell(L, x, "ai_real") for x in GENERIC], 0); gh = np.mean([cell(L, x, "human") for x in GENERIC], 0)
    t = boot_fn(lambda i: (a[i].mean() - g[i].mean()) / (h[i].mean() - gh[i].mean()), nFr)
    kc[f"retained_gender_vs_generic|{n}"] = trip(t)
    print(f"retained gender (ai_real, rel. generic AI) {n:8s} {fmt(t)}")
OUT["A_contrasts"] = kc

# closeness in 4-way pronoun distribution (JS divergence)
def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
def dist4(n, c, i):
    return np.array([cell(beh[k], n, c)[i].mean() for k in ("p_f", "p_m", "p_it", "p_they")])
print("\n--- closeness of 4-way pronoun distributions (JS divergence, nats) ---")
cl = {}
for n in ("Gemma", "Claude", "Gemini", "Sarah", "Emma", "Jemma", "James", "Liam", "Gem"):
    def f(i, n=n):
        pa = dist4(n, "ai_real", i); ph = dist4(n, "human", i)
        pg = np.mean([dist4(g, "ai_real", i) for g in GENERIC], 0)
        return np.array([js(pa, ph), js(pa, pg)])
    e, lo, hi = boot_fn(f, nFr, nb=1000)
    cl[n] = {"JS_ai_vs_own_human": [e[0], lo[0], hi[0]], "JS_ai_vs_genericAI": [e[1], lo[1], hi[1]]}
    print(f"{n:8s} JS(ai_real, own human)={e[0]:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  JS(ai_real, generic AI)={e[1]:.3f} [{lo[1]:.3f},{hi[1]:.3f}]")
OUT["A_closeness_JS"] = cl

# ---------------- J++ readouts (third person)
G = {"name1": 0, "name2": 1, "final": 2, "bg": 3}
def band_vals(metric, grp, n, c, band, minus_bg=False):
    x = RA[:, :, G[grp], MI[metric]]
    if minus_bg:
        x = x - RA[:, :, G["bg"], MI[metric]]
    return cell(x[:, BANDS[band]].mean(1), n, c)
print("\n=== A. J++ readout (pron F-M log-odds), band means over frames ===")
jt = {}
for grp in ("name2", "final", "bg"):
    for metric in ("pron", "noun", "neuter_vs_human"):
        for n in NAMES:
            for c in CONTEXTS:
                for b in BANDS:
                    jt[f"{grp}|{metric}|{n}|{c}|{b}"] = trip(boot(band_vals(metric, grp, n, c, b)))
                    if grp == "name2":
                        jt[f"name2-bg|{metric}|{n}|{c}|{b}"] = trip(boot(band_vals(metric, grp, n, c, b, minus_bg=True)))
OUT["A_jpp_bands"] = jt
for grp in ("name2", "name2-bg", "final", "bg"):
    print(f"\n[{grp}] pron   " + " ".join(f"{b:>8s}" for b in BANDS))
    for n in ["Gemma", "Emma", "Sarah", "James", "Liam", "Claude", "Gemini", "Llama", "Qwen"]:
        for c in ("human", "ai_real"):
            print(f"{n+'|'+c:16s}" + " ".join(f"{jt[f'{grp}|pron|{n}|{c}|{b}'][0]:+8.2f}" for b in BANDS))
# J++ scale positions (name2-bg and final), anchors Sarah|human / James|human
print("\n--- J++ woman/man scale position (0=James|human,1=Sarah|human) for name2-bg and final, pron ---")
jsc = {}
for grp, mb in (("name2", True), ("final", False)):
    for n in ["Gemma", "Emma", "Jemma", "Gem", "Claude", "Gemini", "Llama", "Mistral", "Qwen", "Phi", "Atlas", "Sarah", "James"]:
        for c in ("human", "ai_real", "ai_neutral"):
            for b in BANDS:
                x = band_vals("pron", grp, n, c, b, mb); f = band_vals("pron", grp, "Sarah", "human", b, mb); m = band_vals("pron", grp, "James", "human", b, mb)
                jsc[f"{grp}|{n}|{c}|{b}"] = trip(boot_fn(lambda i: (x[i].mean() - m[i].mean()) / (f[i].mean() - m[i].mean()), nFr, nb=1000))
    for n in ["Gemma", "Claude", "Gemini", "Llama"]:
        for c in ("human", "ai_real"):
            print(f"{grp:6s} {n+'|'+c:16s}" + " ".join(f"{jsc[f'{grp}|{n}|{c}|{b}'][0]:+7.2f}" for b in BANDS))
OUT["A_jpp_scale"] = jsc

# lens vs behaviour: correlation across the 56 condition means between band readout and final behaviour lse_pron
print("\n--- does the J++ readout predict behaviour? Pearson r across 56 name x ctx means (final token / name2) ---")
cm_beh = np.array([cell(L, n, c).mean() for n in NAMES for c in CONTEXTS])
lr = {}
for grp in ("final", "name2"):
    rs = []
    for l in range(60):
        x = np.array([cell(RA[:, l, G[grp], MI["lse_pron"]], n, c).mean() for n in NAMES for c in CONTEXTS])
        rs.append(float(np.corrcoef(x, cm_beh)[0, 1]))
    lr[grp] = rs
    print(grp, " ".join(f"L{l}:{rs[l]:+.2f}" for l in range(0, 60, 5)), f"L59:{rs[59]:+.3f}")
OUT["A_jpp_vs_behaviour_r"] = lr

# instrument test 2: within-name CONTEXT effects (ai_real - human), which the name-token identity cannot explain.
print("\n--- instrument: lens context effect vs behaviour context effect, Pearson r across 14 names (ai_real - human) ---")
beh_nvh = np.log(beh["p_it"]) - np.log(beh["p_f"] + beh["p_m"])
ce = {}
for bm, lm, lab_ in ((L, "lse_pron", "F-vs-M"), (beh_nvh, "neuter_vs_human", "it-vs-human")):
    db = np.array([cell(bm, n, "ai_real").mean() - cell(bm, n, "human").mean() for n in NAMES])
    for grp in ("final", "name2"):
        rs = []
        for l in range(60):
            x = RA[:, l, G[grp], MI[lm]]
            dl = np.array([cell(x, n, "ai_real").mean() - cell(x, n, "human").mean() for n in NAMES])
            rs.append(float(np.corrcoef(dl, db)[0, 1]))
        ce[f"{lab_}|{grp}"] = rs
        print(f"{lab_:12s} {grp:6s}", " ".join(f"L{l}:{rs[l]:+.2f}" for l in (10, 20, 30, 35, 40, 45, 50, 55, 59)))
OUT["A_jpp_context_effect_r"] = ce
# lens neuter_vs_human (final token) per band for key names: does J++ see the AI-context 'it' shift, and when?
print("\n--- J++ neuter_vs_human at final token (band means), human vs ai_real; behaviour log(P_it/P_human) ---")
for n in ["Gemma", "Emma", "Sarah", "Claude", "Gemini", "Llama"]:
    for c in ("human", "ai_real"):
        print(f"{n+'|'+c:16s}" + " ".join(f"{jt[f'final|neuter_vs_human|{n}|{c}|{b}'][0]:+7.2f}" for b in BANDS) + f"   beh {cell(beh_nvh, n, c).mean():+6.2f}")

# =====================================================================================  job D: robustness and self-voice
try:
    D1 = json.load(open(f"{R}/E5_name_D1_robustness.json"))
    print("\n=== D1 robustness: correlation of 56 condition means (lse_pron) with main design ===")
    keys = [(n, c) for n in NAMES for c in CONTEXTS]
    mv = np.array([cell(L, n, c).mean() for n, c in keys])
    rk = lambda a: np.argsort(np.argsort(a))
    d1o = {}
    for vn in ("nolead", "story", "raw"):
        r = D1["D1_" + vn]; acc = {}
        for row, v in zip(r["rows"], r["beh"]["lse_pron"]):
            acc.setdefault((row["name"], row["ctx"]), []).append(v)
        vv = np.array([np.mean(acc[k]) for k in keys])
        gem = {c: float(np.mean(acc[("Gemma", c)])) for c in CONTEXTS}
        gpf = {}
        for row, v in zip(r["rows"], r["beh"]["p_f"]):
            if row["name"] == "Gemma":
                gpf.setdefault(row["ctx"], []).append(v)
        d1o[vn] = {"pearson": float(np.corrcoef(mv, vv)[0, 1]), "spearman": float(np.corrcoef(rk(mv), rk(vv))[0, 1]),
                   "Gemma_lse": gem, "Gemma_pf": {c: float(np.mean(v)) for c, v in gpf.items()}}
        print(vn, json.dumps(d1o[vn]))
    OUT["D1"] = d1o
    D2 = json.load(open(f"{R}/E5_name_D2_selfvoice.json"))
    print("\n=== D2 own-voice third-person next token (16 frames, bootstrap CI) ===")
    d2o = {}
    conds2 = list(dict.fromkeys(r["cond"] for r in D2["rows"]))
    cond2 = np.array([r["cond"] for r in D2["rows"]]); fr2 = np.array([r["frame"] for r in D2["rows"]])
    def c2(k, c):
        a = np.array(D2["beh"][k]); m = cond2 == c; return a[m][np.argsort(fr2[m])]
    for c in conds2:
        d2o[c] = {k: trip(boot(c2(k, c))) for k in ("p_f", "p_m", "p_it", "p_they", "lse_pron")}
        print(f"{c:20s} lse {fmt(d2o[c]['lse_pron'])}  P(she) {d2o[c]['p_f'][0]:.3f}  P(he) {d2o[c]['p_m'][0]:.3f}  P(it) {d2o[c]['p_it'][0]:.3f}")
    d2o["self_default-vs-thirdperson_Gemma_ai_real_lse"] = trip(boot(c2("lse_pron", "self_gemma_default") - cell(L, "Gemma", "ai_real")))
    d2o["gemma_sys-minus-default_pf"] = trip(boot(c2("p_f", "self_gemma_sys") - c2("p_f", "self_gemma_default")))
    print("self_default - thirdperson Gemma|ai_real (lse, paired frames):", fmt(d2o["self_default-vs-thirdperson_Gemma_ai_real_lse"]))
    print("P(she): gemma_sys - default (paired frames):", fmt(d2o["gemma_sys-minus-default_pf"], 3))
    OUT["D2"] = d2o
except FileNotFoundError as e:
    print("D files missing", e)
try:
    D3 = json.load(open(f"{R}/E5_name_D3_generation.json"))
    print("\n=== D3 free generation (T=1, n=16 per prompt): fraction of samples containing each pronoun class ===")
    d3o = {}
    for k, v in D3.items():
        n = len(v["samples"])
        d3o[k] = {"frac_any": v["frac_any"], "n": n,
                  "mean_counts": {c: float(np.mean(x)) for c, x in v["counts"].items()}}
        print(f"{k:12s} n={n} " + " ".join(f"{c}:{v['frac_any'][c]:.2f}" for c in ("F", "M", "N", "T", "I")))
    OUT["D3"] = d3o
except FileNotFoundError:
    print("D3 missing")

# =====================================================================================  job B: self tokens
try:
    B = np.load(f"{R}/E5_name_B_readouts.npz")
except FileNotFoundError:
    B = None
if B is not None:
    RB = B["R"]; condB = B["cond"]; qB = B["q"]
    conds = list(dict.fromkeys(condB.tolist()))
    GB = {"self": 0, "asst_bg": 1, "user": 2}
    def bcell(arr, c):
        m = condB == c; o = np.argsort(qB[m]); return arr[m][o]
    def bband(metric, grp, c, band, minus_bg=False):
        x = RB[:, :, GB[grp], MI[metric]]
        if minus_bg:
            x = x - RB[:, :, GB["asst_bg"], MI[metric]]
        return bcell(x[:, BANDS[band]].mean(1), c)
    nP = 12
    print("\n=== B. SELF TOKENS: J++ pron/noun band means over 12 passages ===")
    bt = {}
    for grp, mb in (("self", False), ("self-bg", True), ("asst_bg", False), ("user", False)):
        g0 = "self" if grp.startswith("self") else grp
        for metric in ("pron", "noun", "neuter_vs_human", "lse_pron"):
            for c in conds:
                for b in BANDS:
                    bt[f"{grp}|{metric}|{c}|{b}"] = trip(boot(bband(metric, g0, c, b, mb)))
        print(f"\n[{grp}] pron  " + " ".join(f"{b:>8s}" for b in BANDS))
        for c in conds:
            print(f"{c:11s}" + " ".join(f"{bt[f'{grp}|pron|{c}|{b}'][0]:+8.2f}" for b in BANDS))
    OUT["B_jpp_bands"] = bt
    print("\n--- self-token woman/man persona scale (0=man persona, 1=woman persona), pron and noun, self and self-bg ---")
    bs_ = {}
    for metric in ("pron", "noun"):
        for grp, mb in (("self", False), ("self-bg", True)):
            for c in conds:
                for b in BANDS:
                    x = bband(metric, "self", c, b, mb); f = bband(metric, "self", "woman", b, mb); m = bband(metric, "self", "man", b, mb)
                    bs_[f"{metric}|{grp}|{c}|{b}"] = trip(boot_fn(lambda i: (x[i].mean() - m[i].mean()) / (f[i].mean() - m[i].mean()), nP, nb=1000))
            print(f"\n{metric} {grp}")
            for c in conds:
                print(f"  {c:11s}" + " ".join(f"{b}:{fmt(bs_[f'{metric}|{grp}|{c}|{b}'])}" for b in ("L20-29", "L30-39", "L40-49", "L50-58")))
    OUT["B_scale"] = bs_
    # paired differences vs default
    print("\n--- paired differences at self tokens (pron, self-bg), vs default ---")
    pdiff = {}
    for c in conds:
        for b in BANDS:
            x = bband("pron", "self", c, b, True); d = bband("pron", "self", "default", b, True)
            pdiff[f"{c}-default|{b}"] = trip(boot(x - d))
        print(f"  {c:11s}-default " + " ".join(f"{b}:{fmt(pdiff[f'{c}-default|{b}'])}" for b in ("L30-39", "L40-49", "L50-58")))
    OUT["B_pdiff_vs_default"] = pdiff
    # Gemma-ness readout: logit(" Gemma") - mean(other model names)
    print("\n--- J++ 'Gemma-ness' at self tokens: logit(' Gemma') - mean(' Claude',' GPT',' ChatGPT',' Gemini') ---")
    gm = RB[:, :, :, MI[" Gemma"]] - np.mean([RB[:, :, :, MI[t]] for t in (" Claude", " GPT", " ChatGPT", " Gemini")], 0)
    cl_ = RB[:, :, :, MI[" Claude"]] - np.mean([RB[:, :, :, MI[t]] for t in (" Gemma", " GPT", " ChatGPT", " Gemini")], 0)
    gness = {}
    for lab_, arr in (("gemma", gm), ("claude", cl_)):
        for c in conds:
            for b in BANDS:
                x = bcell(arr[:, BANDS[b], 0].mean(1), c)
                xb = bcell(arr[:, BANDS[b], 1].mean(1), c)
                gness[f"{lab_}|self|{c}|{b}"] = trip(boot(x))
                gness[f"{lab_}|self-bg|{c}|{b}"] = trip(boot(x - xb))
        print(f"[{lab_}-ness self-bg]")
        for c in conds:
            print(f"  {c:11s}" + " ".join(f"{b}:{gness[f'{lab_}|self-bg|{c}|{b}'][0]:+6.2f}" for b in BANDS))
    for b in BANDS:
        d1 = bcell(gm[:, BANDS[b], 0].mean(1) - gm[:, BANDS[b], 1].mean(1), "default")
        d0 = bcell(gm[:, BANDS[b], 0].mean(1) - gm[:, BANDS[b], 1].mean(1), "nameless")
        d2 = bcell(gm[:, BANDS[b], 0].mean(1) - gm[:, BANDS[b], 1].mean(1), "gemma_sys")
        gness[f"default-nameless|{b}"] = trip(boot(d1 - d0)); gness[f"gemma_sys-nameless|{b}"] = trip(boot(d2 - d0))
        gness[f"default_scale|{b}"] = trip(boot_fn(lambda i: (d1[i].mean() - d0[i].mean()) / (d2[i].mean() - d0[i].mean()), nP, nb=1000))
        print(f"  {b}: default-nameless {fmt(gness[f'default-nameless|{b}'])}  gemma_sys-nameless {fmt(gness[f'gemma_sys-nameless|{b}'])}  default on [nameless=0,gemma_sys=1] {fmt(gness[f'default_scale|{b}'])}")
    OUT["B_gemmaness"] = gness

# =====================================================================================  job C: representational similarity
try:
    C = json.load(open(f"{R}/E5_name_C_similarity.json"))
    lay = C["layers"]
    show = [i for i, l in enumerate(lay) if l in (20, 30, 35, 40, 45, 50, 55, 59)]
    print("\n=== C. centered-state geometry (full-sample estimate [bootstrap 95% CI]) ===")
    for space in ("raw", "jpp"):
        q = C["spaces"][space]["q"]
        print(f"\n[{space}] layer: " + " ".join(f"{lay[i]:>22d}" for i in show))
        for k in ("dcos_default_GemmaID", "dcos_default_GemmaID_neutral", "dcos_gemmasys_GemmaID", "dcos_llamasys_GemmaID",
                  "dcos_claudesys_GemmaID", "dcos_claudesys_ClaudeID", "dcos_llamasys_LlamaID", "dcos_default_gemmasys_selfonly",
                  "dcos_default_llamasys_selfonly", "dcos_woman-man_femH", "dcos_sarahai-jamesai_femA", "dcos_default_femH",
                  "dcos_gemmasys_femH", "femscale_H|default", "femscale_H|nameless", "femscale_H|gemma_sys", "femscale_H|sarah_ai",
                  "femscale_H|james_ai", "gemmascale|default", "gemmascale|llama_sys", "gemmascale|claude_sys",
                  "distratio_name2|Gemma", "distratio_final|Gemma", "distratio_final|Claude", "distratio_final|Sarah",
                  "distratio_final|Emma", "distratio_final|Gemini",
                  "map_final_ai|Gemma|ai_real", "map_final_fem|Gemma|ai_real", "map_final_fem|Gemma|human",
                  "map_final_ai|Emma|ai_real", "map_final_fem|Emma|ai_real", "map_final_ai|Claude|ai_real", "map_final_fem|Claude|ai_real",
                  "map_final_fem|Llama|ai_real", "map_final_fem|Gemini|ai_real",
                  "map_name2_ai|Gemma|ai_real", "map_name2_fem|Gemma|ai_real", "map_name2_ai|Emma|ai_real", "map_name2_fem|Emma|ai_real"):
            if k in q:
                print(f"{k:34s}" + " ".join(f"{q[k][i][0]:+6.2f}[{q[k][i][1]:+5.2f},{q[k][i][2]:+5.2f}]" for i in show))
    OUT["C_keys_note"] = "see E5_name_C_similarity.json for all quantities"
except FileNotFoundError:
    print("C missing")
try:
    LC = np.load(f"{R}/E5_name_C_lenscompare.npz")
    lay = list(LC["layers"]); gk = list(LC["metrics"])
    print("\n=== C2. lens comparison on identical stored states: r(lens lse_pron at final token, behaviour) across 56 conditions;"
          " and context-effect r across 14 names; self-token woman/man scale (pron) ===")
    lc = {}
    condB = np.load(f"{R}/E5_name_B_readouts.npz")["cond"]
    for kind in ("jpp", "jlens", "naive"):
        arr = LC[kind + "_A"]       # [896, L, 2(name2, final), K]
        r56, rce, sc_def = [], [], []
        for li in range(len(lay)):
            x = arr[:, li, 1, gk.index("lse_pron")]
            xm = np.array([cell(x, n, c).mean() for n in NAMES for c in CONTEXTS])
            r56.append(float(np.corrcoef(xm, cm_beh)[0, 1]))
            dl = np.array([cell(x, n, "ai_real").mean() - cell(x, n, "human").mean() for n in NAMES])
            db = np.array([cell(L, n, "ai_real").mean() - cell(L, n, "human").mean() for n in NAMES])
            rce.append(float(np.corrcoef(dl, db)[0, 1]))
            b = LC[kind + "_B"][:, li, 0, gk.index("pron")]
            w, m_, d_ = b[condB == "woman"].mean(), b[condB == "man"].mean(), b[condB == "default"].mean()
            g_ = b[condB == "gemma_sys"].mean()
            sc_def.append([float((d_ - m_) / (w - m_)), float((g_ - m_) / (w - m_)), float(w - m_)])
        lc[kind] = {"r56_final": r56, "r_context_effect_final": rce, "self_scale_default_gemmasys_gap": sc_def}
        print(kind.ljust(6), "r56:", " ".join(f"L{l}:{v:+.2f}" for l, v in zip(lay, r56)))
        print(" " * 6, "rCE:", " ".join(f"L{l}:{v:+.2f}" for l, v in zip(lay, rce)))
        print(" " * 6, "self scale default/gemma_sys (woman-man gap):", " ".join(f"L{l}:{a:+.2f}/{b_:+.2f}({g:+.1f})" for l, (a, b_, g) in zip(lay, sc_def)))
    OUT["C_lenscompare"] = lc
except FileNotFoundError:
    print("C lenscompare missing")

# =====================================================================================  job E: identity readout at self tokens
try:
    E = np.load(f"{R}/E5_name_E_identity_self.npz")
    toks = list(E["toks"]); TI = {t: i for i, t in enumerate(toks)}
    condE, qE = E["cond"], E["q"]
    condsE = list(dict.fromkeys(condE.tolist()))
    def ecell(arr, c):
        m = condE == c; return arr[m][np.argsort(qE[m])]
    J = E["jpp"]              # [132, 60, 3, K]
    RKJ = E["rank_jpp"]       # [132, 60, 3]
    eo = {}
    print("\n=== E. J++ log P(token) at self tokens (band means over 12 passages) ===")
    for tok in (" Gemma", " Google", " Claude", " OpenAI", " Sarah", " James", " Atlas"):
        print(f"[{tok}] self  " + " ".join(f"{b:>8s}" for b in BANDS) + "   | asst_bg L40-49 | user L40-49")
        for c in condsE:
            vals = [ecell(J[:, BANDS[b], 0, TI[tok]].mean(1), c).mean() for b in BANDS]
            print(f"  {c:11s}" + " ".join(f"{v:+8.2f}" for v in vals) +
                  f"   | {ecell(J[:, BANDS['L40-49'], 1, TI[tok]].mean(1), c).mean():+7.2f}   | {ecell(J[:, BANDS['L40-49'], 2, TI[tok]].mean(1), c).mean():+7.2f}")
            for b in BANDS:
                eo[f"logp|{tok}|self|{c}|{b}"] = trip(boot(ecell(J[:, BANDS[b], 0, TI[tok]].mean(1), c)))
                eo[f"logp|{tok}|self-abg|{c}|{b}"] = trip(boot(ecell(J[:, BANDS[b], 0, TI[tok]].mean(1) - J[:, BANDS[b], 1, TI[tok]].mean(1), c)))
    print("\n--- geometric-mean rank of ' Gemma' (J++), self tokens / asst_bg, at L35, L40, L45, L50 ---")
    for c in condsE:
        print(f"  {c:11s}" + " ".join(f"L{l}:{np.exp(np.log(ecell(RKJ[:, l, 0], c)).mean()):7.1f}/{np.exp(np.log(ecell(RKJ[:, l, 1], c)).mean()):7.1f}" for l in (35, 40, 45, 50)))
        for l in (35, 40, 42, 45, 50):
            eo[f"rank_gemma|self|{c}|L{l}"] = float(np.exp(np.log(ecell(RKJ[:, l, 0], c)).mean()))
            eo[f"rank_gemma|abg|{c}|L{l}"] = float(np.exp(np.log(ecell(RKJ[:, l, 1], c)).mean()))
    print("\n--- paired contrasts on J++ log P(' Gemma') at self tokens (passages) ---")
    for b in ("L30-39", "L40-49", "L50-58", "L59"):
        g = J[:, BANDS[b], 0, TI[" Gemma"]].mean(1)
        for a_, b_ in (("default", "nameless"), ("gemma_sys", "nameless"), ("llama_sys", "default"), ("claude_sys", "default"),
                       ("qwen_sys", "default"), ("atlas_sys", "default"), ("woman", "default"), ("man", "default")):
            eo[f"dGemma|{a_}-{b_}|{b}"] = trip(boot(ecell(g, a_) - ecell(g, b_)))
        gs = J[:, BANDS[b], 0, TI[" Gemma"]].mean(1) - J[:, BANDS[b], 1, TI[" Gemma"]].mean(1)
        eo[f"specificity_self-abg|default|{b}"] = trip(boot(ecell(gs, "default")))
        print(b, " ".join(f"{k.split('|')[1]}:{fmt(eo[k])}" for k in eo if k.startswith("dGemma|") and k.endswith(b)),
              "| self-abg(default):", fmt(eo[f"specificity_self-abg|default|{b}"]))
    print("\n--- persona-name token at self tokens: own-persona minus default (log P), by band ---")
    for c, tok in (("claude_sys", " Claude"), ("atlas_sys", " Atlas"), ("sarah_ai", " Sarah"), ("james_ai", " James"),
                   ("gemma_sys", " Gemma"), ("woman", " Sarah"), ("man", " James")):
        row = []
        for b in BANDS:
            x = J[:, BANDS[b], 0, TI[tok]].mean(1)
            t_ = boot(ecell(x, c) - ecell(x, "default")); eo[f"persona_tok|{c}|{tok}|{b}"] = trip(t_); row.append(fmt(t_))
        print(f"  {c:11s}{tok:8s}" + " ".join(row))
    # lens comparison on Gemma identity
    sub = list(E["sub"])
    print("\n--- lens comparison: log P(' Gemma') and rank at self tokens, default vs man persona ---")
    for kind in ("jpp", "jlens", "naive"):
        arr = E[kind] if kind != "jpp" else J[:, sub]
        rk = E["rank_" + kind] if kind != "jpp" else RKJ[:, sub]
        line = []
        for i, l in enumerate(sub):
            line.append(f"L{l}:{ecell(arr[:, i, 0, TI[' Gemma']], 'default').mean():+.1f}/{np.exp(np.log(ecell(rk[:, i, 0], 'default')).mean()):.0f}"
                        f"|man {np.exp(np.log(ecell(rk[:, i, 0], 'man')).mean()):.0f}")
            eo[f"lens|{kind}|L{l}|default|logp"] = float(ecell(arr[:, i, 0, TI[' Gemma']], 'default').mean())
            eo[f"lens|{kind}|L{l}|default|rank"] = float(np.exp(np.log(ecell(rk[:, i, 0], 'default')).mean()))
            eo[f"lens|{kind}|L{l}|man|rank"] = float(np.exp(np.log(ecell(rk[:, i, 0], 'man')).mean()))
        print(kind.ljust(6), " ".join(line))
    ET = np.load(f"{R}/E5_name_E_identity_thirdperson.npz")
    print("\n--- third-person control: geometric-mean rank of ' Gemma' (J++) at the 2nd name mention / final token ---")
    for n in ("Gemma", "Llama", "Claude", "Qwen", "Gemini", "Mistral", "Phi", "Sarah", "James", "Atlas"):
        for c in ("human", "ai_real"):
            m = (ET["names"] == n) & (ET["ctx"] == c)
            r2 = [float(np.exp(np.log(ET["rank"][m][:, l, 0]).mean())) for l in (35, 40, 45, 50)]
            rf = [float(np.exp(np.log(ET["rank"][m][:, l, 1]).mean())) for l in (35, 40, 45, 50)]
            eo[f"ctrl_rank|{n}|{c}|name2"] = r2; eo[f"ctrl_rank|{n}|{c}|final"] = rf
            print(f"  {n+'|'+c:16s} name2 L35/40/45/50: " + " ".join(f"{v:8.1f}" for v in r2) + "   final: " + " ".join(f"{v:8.1f}" for v in rf))
    OUT["E_identity"] = eo
    PB = json.load(open(f"{R}/E5_name_E_persona_behaviour.json"))
    print("\n--- persona behaviour (greedy) ---")
    for k, v in PB.items():
        print(f"  {k:40s} -> {v.strip()[:90]!r}")
    OUT["E_persona_behaviour"] = PB
except FileNotFoundError as e:
    print("E missing", e)

# =====================================================================================  job F: persona self vs other; identity leak
try:
    F1 = json.load(open(f"{R}/E5_name_F1_persona_pronoun.json"))
    rows_, b_ = F1["rows"], F1["beh"]
    fo = {}
    print("\n=== F1 self vs other (same name), paired over 16 frames ===")
    for k in dict.fromkeys(r["persona"] for r in rows_):
        sv = np.array([b_["lse_pron"][i] for i, r in enumerate(rows_) if r["persona"] == k and r["mode"] == "self"])
        ov = np.array([b_["lse_pron"][i] for i, r in enumerate(rows_) if r["persona"] == k and r["mode"] == "other"])
        pf = np.array([b_["p_f"][i] for i, r in enumerate(rows_) if r["persona"] == k and r["mode"] == "self"])
        fo[k] = {"self_minus_other_lse": trip(boot(sv - ov)), "self_lse": trip(boot(sv)), "other_lse": trip(boot(ov)), "self_pf": float(pf.mean())}
        print(f"  {k:8s} self-other {fmt(fo[k]['self_minus_other_lse'])}  self P(she) {pf.mean():.3f}")
    F2 = json.load(open(f"{R}/E5_name_F2_identity_leak.json"))["logp"]
    systems = sorted(set(k.split("|")[0] for k in F2))
    fo["F2_mean_logp_Gemma_prompts0to4"] = {sy: float(np.mean([F2[f"{sy}|{i}"]["Gemma"] for i in range(5)])) for sy in systems}
    fo["F2_logp_Gemma_after_My_name_is"] = {sy: F2[f"{sy}|4"]["Gemma"] for sy in systems}
    print("  F2 mean logP(' Gemma') over 5 leak prefixes:", {k: round(v, 1) for k, v in fo["F2_mean_logp_Gemma_prompts0to4"].items()})
    OUT["F"] = fo
except FileNotFoundError:
    print("F missing")

json.dump(OUT, open(f"{R}/E5_name_analysis.json", "w"), indent=0)
print("\nsaved", f"{R}/E5_name_analysis.json")
