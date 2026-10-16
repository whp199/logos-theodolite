# E1_self_readout analysis (CPU; run with .venv/bin/python). Unit of analysis = passage (n=40); tokens are averaged within
# passage first. Paired bootstrap (same resampled passage indices for every condition), 2000 resamples, percentile 95% CIs.
# p-values: two-sided sign-flip permutation test on per-passage paired differences (20000 flips).
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

R = "./experiments/results"
Z = np.load(f"{R}/E1_self_readout_passage_means.npz")
X = Z["X"].astype(np.float64)  # [lens, cond, q, class, layer, metric]
LENSES = list(Z["lenses"]); CONDS = list(Z["conds"]); CLASSES = list(Z["classes"]); METRICS = list(Z["metrics"])
NQ = X.shape[2]
BANDS = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40),
         "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
rng = np.random.default_rng(12345)
NB = 2000
BOOT = rng.integers(0, NQ, (NB, NQ))
FLIPS = rng.choice([-1.0, 1.0], (20000, NQ))

# derived (aggregate) conditions
GROUPS = {
    "woman": ["woman_p1", "woman_p2"], "man": ["man_p1", "man_p2"],
    "femAI": ["f_sarah", "f_emily", "f_olivia"], "maleAI": ["m_james", "m_michael", "m_david"],
    "neutAI": ["n_atlas", "n_quill", "n_unit7"],
}
ALLC = CONDS + list(GROUPS)


def aug(Y):
    """Y [cond, q] -> [cond+groups, q]"""
    extra = [Y[[CONDS.index(c) for c in g]].mean(0) for g in GROUPS.values()]
    return np.concatenate([Y, np.stack(extra)], 0)


def ci(samples):
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def perm_p(d):
    obs = abs(d.mean())
    null = np.abs((FLIPS * d[None, :]).mean(1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (len(null) + 1))


CONTRASTS = [
    ("woman-man", "woman", "man"),
    ("femAI-maleAI", "femAI", "maleAI"),
    ("femAI-neutAI", "femAI", "neutAI"),
    ("maleAI-neutAI", "maleAI", "neutAI"),
    ("default-nameless", "default", "nameless"),
    ("default-neutAI", "default", "neutAI"),
    ("default-deepmind_nameless", "default", "deepmind_nameless"),
    ("default-gemma_explicit", "default", "gemma_explicit"),
    ("nameless-neutAI", "nameless", "neutAI"),
    ("gemma_explicit-nameless", "gemma_explicit", "nameless"),
    ("gemma_explicit-deepmind_nameless", "gemma_explicit", "deepmind_nameless"),
    ("gemma_short-nameless", "gemma_short", "nameless"),
    ("gemma_short-neutAI", "gemma_short", "neutAI"),
    ("gemma_short-femAI", "gemma_short", "femAI"),
    ("default-woman", "default", "woman"),
    ("default-man", "default", "man"),
]


def analyze_Y(Y):
    """Y [cond, q] (one lens/metric/class/band). Returns per-cond means/CIs, scale positions, contrasts."""
    Ya = aug(Y)
    m = Ya.mean(1)
    bm = Ya[:, BOOT].mean(-1)  # [C, NB]
    iw, im = ALLC.index("woman"), ALLC.index("man")
    span = m[iw] - m[im]
    bspan = bm[iw] - bm[im]
    out = {"cond": {}, "scale": {}, "contrast": {}, "span": {"mean": float(span), "ci": ci(bspan)}}
    for k, c in enumerate(ALLC):
        out["cond"][c] = {"mean": float(m[k]), "ci": ci(bm[k])}
        pos = (bm[k] - bm[im]) / bspan
        out["scale"][c] = {"pos": float((m[k] - m[im]) / span), "ci": ci(pos)}
    for name, a, b in CONTRASTS:
        ia, ib = ALLC.index(a), ALLC.index(b)
        d = Ya[ia] - Ya[ib]
        bd = bm[ia] - bm[ib]
        out["contrast"][name] = {"diff": float(d.mean()), "ci": ci(bd), "p_perm": perm_p(d),
                                 "pct_span": float(100 * d.mean() / span), "pct_span_ci": ci(100 * bd / bspan),
                                 "dz": float(d.mean() / (d.std(ddof=1) + 1e-12))}
    return out


def band_mean(arr, b):
    return arr[..., list(BANDS[b]), :].mean(-2) if arr.ndim >= 2 else None


RES = {"bands": {k: [r.start, r.stop - 1] for k, r in BANDS.items()}, "conds": CONDS, "groups": GROUPS, "n_passages": NQ,
       "n_boot": NB, "analysis": {}}
# main analysis for classes self, nonself, user_all, header + specificity self-nonself and self-user_all
SPEC = {"spec_self_minus_nonself": ("self", "nonself"), "spec_self_minus_user": ("self", "user_all")}
for li, ln in enumerate(LENSES):
    RES["analysis"][ln] = {}
    for mi, mt in enumerate(METRICS):
        if mt not in ("pron", "noun", "neuter_vs_human", "lse_pron"):
            continue
        RES["analysis"][ln][mt] = {}
        for cl in CLASSES + list(SPEC):
            RES["analysis"][ln][mt][cl] = {}
            for b in BANDS:
                lay = list(BANDS[b])
                if cl in SPEC:
                    a, c2 = SPEC[cl]
                    Y = X[li, :, :, CLASSES.index(a), :, mi][..., lay].mean(-1) - X[li, :, :, CLASSES.index(c2), :, mi][..., lay].mean(-1)
                else:
                    Y = X[li, :, :, CLASSES.index(cl), :, mi][..., lay].mean(-1)
                RES["analysis"][ln][mt][cl][b] = analyze_Y(Y)

# per-layer arrays (means over passages) for key conditions, self/nonself/user/header, pron & noun, all lenses
RES["per_layer"] = {}
for li, ln in enumerate(LENSES):
    RES["per_layer"][ln] = {}
    for mt in ("pron", "noun", "neuter_vs_human"):
        mi = METRICS.index(mt)
        RES["per_layer"][ln][mt] = {}
        for cl in ("self", "nonself", "user_all", "header"):
            Yl = X[li, :, :, CLASSES.index(cl), :, mi]  # [cond, q, layer]
            ext = [Yl[[CONDS.index(c) for c in g]].mean(0) for g in GROUPS.values()]
            Ya = np.concatenate([Yl, np.stack(ext)], 0).mean(1)  # [C, layer]
            RES["per_layer"][ln][mt][cl] = {c: [round(float(v), 4) for v in Ya[k]] for k, c in enumerate(ALLC)}

# gender-specificity of persona effects: (A-B)_self - (A-B)_class2 per passage
RES["gender_specificity"] = {}
for li, ln in enumerate(LENSES):
    RES["gender_specificity"][ln] = {}
    for mt in ("pron", "noun"):
        mi = METRICS.index(mt)
        RES["gender_specificity"][ln][mt] = {}
        for b in BANDS:
            lay = list(BANDS[b])
            byc = {cl: aug(X[li, :, :, CLASSES.index(cl), :, mi][..., lay].mean(-1)) for cl in ("self", "nonself", "user_all", "header")}
            row = {}
            for name, a, c in [("woman-man", "woman", "man"), ("femAI-maleAI", "femAI", "maleAI"),
                               ("gemma_explicit-nameless", "gemma_explicit", "nameless"), ("default-nameless", "default", "nameless")]:
                ia, ic = ALLC.index(a), ALLC.index(c)
                for cl2 in ("nonself", "user_all"):
                    d = (byc["self"][ia] - byc["self"][ic]) - (byc[cl2][ia] - byc[cl2][ic])
                    bd = d[BOOT].mean(-1)
                    eff_self = (byc["self"][ia] - byc["self"][ic]).mean()
                    eff_2 = (byc[cl2][ia] - byc[cl2][ic]).mean()
                    row[f"{name}|self_vs_{cl2}"] = {"self_effect": float(eff_self), "other_effect": float(eff_2),
                                                    "diff": float(d.mean()), "ci": ci(bd), "p_perm": perm_p(d)}
            RES["gender_specificity"][ln][mt][b] = row

# lens agreement: correlation across the 18 conditions of self-token band means (and of condition effects relative to nameless)
RES["lens_agreement"] = {}
for mt in ("pron", "noun"):
    mi = METRICS.index(mt)
    RES["lens_agreement"][mt] = {}
    for b in BANDS:
        lay = list(BANDS[b])
        V = {ln: X[li, :, :, CLASSES.index("self"), :, mi][..., lay].mean(-1).mean(1) for li, ln in enumerate(LENSES)}  # [cond]
        r = {}
        for a, c in [("jpp", "jlens"), ("jpp", "naive"), ("jlens", "naive")]:
            r[f"{a}~{c}"] = float(np.corrcoef(V[a], V[c])[0, 1])
        r["values"] = {ln: {c: round(float(V[ln][k]), 3) for k, c in enumerate(CONDS)} for ln in LENSES}
        RES["lens_agreement"][mt][b] = r

json.dump(RES, open(f"{R}/E1_self_readout_analysis.json", "w"), indent=1)

# ------------------------------------------------------------------ console tables
def fmt(d):
    return f"{d['mean']:+6.2f} [{d['ci'][0]:+6.2f},{d['ci'][1]:+6.2f}]"

SHOW = ["default", "nameless", "gemma_explicit", "gemma_short", "deepmind_nameless", "femAI", "neutAI", "maleAI", "woman", "man"]
for ln in LENSES:
    for mt in ("pron", "noun"):
        print(f"\n=== {ln} {mt}  SELF tokens: band means [95% CI] ===")
        print(f"{'cond':18s}" + "".join(f"{b:>24s}" for b in BANDS))
        for c in SHOW:
            print(f"{c:18s}" + "".join(f"{fmt(RES['analysis'][ln][mt]['self'][b]['cond'][c]):>24s}" for b in BANDS))
        print("woman/man scale position (0=man, 1=woman):")
        for c in ["default", "nameless", "gemma_explicit", "gemma_short", "deepmind_nameless", "femAI", "neutAI", "maleAI"]:
            print(f"{c:18s}" + "".join(
                f"{RES['analysis'][ln][mt]['self'][b]['scale'][c]['pos']:>8.2f} [{RES['analysis'][ln][mt]['self'][b]['scale'][c]['ci'][0]:+.2f},{RES['analysis'][ln][mt]['self'][b]['scale'][c]['ci'][1]:+.2f}]".rjust(24)
                for b in BANDS))
for ln in ["jpp"]:
    for mt in ("pron", "noun"):
        print(f"\n=== {ln} {mt} SELF contrasts: diff [CI] p  (% of woman-man span) ===")
        for name, _, _ in CONTRASTS:
            s = f"{name:34s}"
            for b in ("L20-29", "L30-39", "L40-49", "L50-58"):
                d = RES["analysis"][ln][mt]["self"][b]["contrast"][name]
                s += f" | {d['diff']:+5.2f} [{d['ci'][0]:+5.2f},{d['ci'][1]:+5.2f}] p={d['p_perm']:.3f} ({d['pct_span']:+5.0f}%)"
            print(s)
