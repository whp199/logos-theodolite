# E1_self_readout: behavioral anchor analysis + lens-vs-behavior agreement across the 18 identity conditions (CPU).
import json, numpy as np
R = "./experiments/results"
B = json.load(open(f"{R}/E1_self_readout_behavior.json"))
A = json.load(open(f"{R}/E1_self_readout_analysis.json"))
conds = A["conds"]
rng = np.random.default_rng(0)
rows = B["forced"]
out = {"per_cond": {}, "contrasts": {}, "lens_behavior_corr": {}}
M = {}
for c in conds:
    rr = sorted([r for r in rows if r["cond"] == c], key=lambda r: r["t"])
    M[c] = {k: np.array([r[k] for r in rr]) for k in ("lse_fm", "p_f", "p_m", "p_neuter", "p_they")}
nt = len(M[conds[0]]["lse_fm"])
BOOT = rng.integers(0, nt, (4000, nt))
for c in conds:
    x = M[c]["lse_fm"]
    gc = B["gen"][c]["counts"]
    out["per_cond"][c] = {"lse_fm_mean": float(x.mean()), "lse_fm_ci": [float(np.percentile(x[BOOT].mean(1), q)) for q in (2.5, 97.5)],
                          "p_f": float(M[c]["p_f"].mean()), "p_m": float(M[c]["p_m"].mean()), "p_neuter": float(M[c]["p_neuter"].mean()),
                          "p_they": float(M[c]["p_they"].mean()),
                          "gen_pronoun_words": {k: int(sum(g[k] for g in gc)) for k in ("f", "m", "it", "they")},
                          "gen_samples_with_f": int(sum(g["f"] > 0 for g in gc)), "gen_samples_with_m": int(sum(g["m"] > 0 for g in gc)),
                          "gen_samples_with_it": int(sum(g["it"] > 0 for g in gc)), "n_gen": len(gc)}
grp = {"femAI": ["f_sarah", "f_emily", "f_olivia"], "maleAI": ["m_james", "m_michael", "m_david"], "neutAI": ["n_atlas", "n_quill", "n_unit7"],
       "woman": ["woman_p1", "woman_p2"], "man": ["man_p1", "man_p2"]}
G = {k: np.mean([M[c]["lse_fm"] for c in v], 0) for k, v in grp.items()}
for c in conds:
    G[c] = M[c]["lse_fm"]
for name, a, b in [("default-nameless", "default", "nameless"), ("default-neutAI", "default", "neutAI"),
                   ("gemma_explicit-nameless", "gemma_explicit", "nameless"), ("gemma_explicit-deepmind_nameless", "gemma_explicit", "deepmind_nameless"),
                   ("gemma_short-neutAI", "gemma_short", "neutAI"), ("femAI-neutAI", "femAI", "neutAI"), ("maleAI-neutAI", "maleAI", "neutAI"),
                   ("woman-man", "woman", "man"), ("woman-neutAI", "woman", "neutAI"), ("man-neutAI", "man", "neutAI")]:
    d = G[a] - G[b]
    span = G["woman"] - G["man"]
    out["contrasts"][name] = {"diff": float(d.mean()), "ci": [float(np.percentile(d[BOOT].mean(1), q)) for q in (2.5, 97.5)],
                              "pct_span": float(100 * d.mean() / span.mean()),
                              "pct_span_ci": [float(np.percentile(100 * d[BOOT].mean(1) / span[BOOT].mean(1), q)) for q in (2.5, 97.5)]}
beh = np.array([out["per_cond"][c]["lse_fm_mean"] for c in conds])
def rank(v):
    return np.argsort(np.argsort(v)).astype(float)
for ln in ("jpp", "jlens", "naive"):
    for mt in ("pron", "noun"):
        for b in A["bands"]:
            lens = np.array([A["analysis"][ln][mt]["self"][b]["cond"][c]["mean"] for c in conds])
            ai = [i for i, c in enumerate(conds) if not c.startswith(("woman", "man"))]
            out["lens_behavior_corr"][f"{ln}|{mt}|{b}"] = {
                "pearson_all18": float(np.corrcoef(lens, beh)[0, 1]), "spearman_all18": float(np.corrcoef(rank(lens), rank(beh))[0, 1]),
                "pearson_AI14": float(np.corrcoef(lens[ai], beh[ai])[0, 1]), "spearman_AI14": float(np.corrcoef(rank(lens[ai]), rank(beh[ai]))[0, 1])}
json.dump(out, open(f"{R}/E1_self_readout_behavior_analysis.json", "w"), indent=1)
for c in conds:
    p = out["per_cond"][c]
    print(f"{c:18s} lse_fm {p['lse_fm_mean']:+6.2f} [{p['lse_fm_ci'][0]:+.2f},{p['lse_fm_ci'][1]:+.2f}] pF {p['p_f']:.3f} pM {p['p_m']:.3f} pIt {p['p_neuter']:.3f} pThey {p['p_they']:.3f} gen(samples w/ she,he,it)={p['gen_samples_with_f']},{p['gen_samples_with_m']},{p['gen_samples_with_it']}/{p['n_gen']}")
for k, v in out["contrasts"].items():
    print(f"{k:34s} {v['diff']:+6.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]  {v['pct_span']:+5.1f}% of span [{v['pct_span_ci'][0]:+.1f},{v['pct_span_ci'][1]:+.1f}]")
for k, v in out["lens_behavior_corr"].items():
    if "L0-9" in k or "L59" in k:
        continue
    print(k, {kk: round(vv, 3) for kk, vv in v.items()})
