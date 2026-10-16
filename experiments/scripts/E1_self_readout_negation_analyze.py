# E1_self_readout: analysis of the exploratory negation / machine-vs-human readout and held-out gender replication (CPU).
import json, numpy as np
R = "./experiments/results"
Z = np.load(f"{R}/E1_self_readout_negation_means.npz")
OUT = Z["OUT"].astype(np.float64)  # [q, cond, cls, lens, layer, metric]
sets = list(Z["sets"]); conds = list(Z["conds"]); CLS = list(Z["classes"]); LNS = list(Z["lenses"]); MET = list(Z["metrics"])
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
GR = {"woman": ["woman_p1", "woman_p2"], "man": ["man_p1", "man_p2"], "human": ["woman_p1", "woman_p2", "man_p1", "man_p2"],
      "femAI": ["f_sarah", "f_emily", "f_olivia"], "maleAI": ["m_james", "m_michael", "m_david"], "neutAI": ["n_atlas", "n_quill", "n_unit7"],
      "AI_all": [c for c in conds if not c.startswith(("woman", "man"))]}
rng = np.random.default_rng(7)
def get(sub, cond, cls, ln, mt, band):
    qs = [i for i, s in enumerate(sets) if s == sub]
    cs = GR.get(cond, [cond])
    x = OUT[qs][:, [conds.index(c) for c in cs], CLS.index(cls), LNS.index(ln)][..., list(BANDS[band]), MET.index(mt)]
    return x.mean(-1).mean(1)  # [nq]
def summ(x):
    b = rng.integers(0, len(x), (2000, len(x)))
    return {"mean": float(x.mean()), "ci": [float(np.percentile(x[b].mean(1), 2.5)), float(np.percentile(x[b].mean(1), 97.5))], "n": len(x)}
res = {}
for sub in ("main", "held"):
    for ln in LNS:
        for mt in MET:
            for cls in ("self", "nonself", "header"):
                for band in BANDS:
                    key = f"{sub}|{ln}|{mt}|{cls}|{band}"
                    r = {}
                    for c in ["default", "nameless", "gemma_explicit", "neutAI", "femAI", "maleAI", "woman", "man", "human", "AI_all"]:
                        r[c] = summ(get(sub, c, cls, ln, mt, band))
                    for nm, a, b in [("AI_all-human", "AI_all", "human"), ("neutAI-human", "neutAI", "human"), ("femAI-woman", "femAI", "woman"),
                                     ("maleAI-man", "maleAI", "man"), ("default-nameless", "default", "nameless"), ("default-human", "default", "human"),
                                     ("woman-man", "woman", "man"), ("gemma_explicit-nameless", "gemma_explicit", "nameless"),
                                     ("default-neutAI", "default", "neutAI"), ("femAI-neutAI", "femAI", "neutAI"), ("maleAI-neutAI", "maleAI", "neutAI")]:
                        d = get(sub, a, cls, ln, mt, band) - get(sub, b, cls, ln, mt, band)
                        s = summ(d)
                        flips = rng.choice([-1, 1], (20000, len(d)))
                        s["p_perm"] = float((np.sum(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12) + 1) / 20001)
                        r["contrast:" + nm] = s
                    res[key] = r
json.dump(res, open(f"{R}/E1_self_readout_negation_analysis.json", "w"), indent=1)
def f(s): return f"{s['mean']:+6.2f}[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]"
for mt in ("neg_vs_aff", "machine_vs_human"):
    for sub in ("main", "held"):
        for cls in ("self", "nonself"):
            print(f"\n== jpp {mt} {cls} [{sub}]")
            for c in ["default", "nameless", "gemma_explicit", "neutAI", "femAI", "maleAI", "woman", "man"]:
                print(f"{c:15s}" + "".join(f(res[f'{sub}|jpp|{mt}|{cls}|{b}'][c]).rjust(22) for b in BANDS))
            for nm in ["AI_all-human", "femAI-woman", "maleAI-man", "default-nameless", "default-human"]:
                print(f"{nm:15s}" + "".join((f(res[f'{sub}|jpp|{mt}|{cls}|{b}']['contrast:' + nm]) + f"p{res[f'{sub}|jpp|{mt}|{cls}|{b}']['contrast:' + nm]['p_perm']:.3f}").rjust(28) for b in BANDS))
print("\n== HELD-OUT gender replication (jpp, self tokens)")
for mt in ("pron", "noun"):
    for nm in ["woman-man", "default-nameless", "default-neutAI", "gemma_explicit-nameless", "femAI-neutAI", "maleAI-neutAI"]:
        s = f"{mt} {nm:24s}"
        for b in ("L20-29", "L30-39", "L40-49", "L50-58"):
            x = res[f"held|jpp|{mt}|self|{b}"]["contrast:" + nm]; span = res[f"held|jpp|{mt}|self|{b}"]["contrast:woman-man"]["mean"]
            s += f" | {b} {x['mean']:+5.2f}[{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}] ({100*x['mean']/span:+.0f}%)"
        print(s)
print("\n== jlens neg_vs_aff self (held) AI_all-human:", {b: f(res[f'held|jlens|neg_vs_aff|self|{b}']['contrast:AI_all-human']) for b in BANDS})
print("== naive neg_vs_aff self (held) AI_all-human:", {b: f(res[f'held|naive|neg_vs_aff|self|{b}']['contrast:AI_all-human']) for b in BANDS})
