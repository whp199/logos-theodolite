# E4_causal_19 (CPU, offline): analyze calibrated shift (clamp clean + k*G_l along u_l) results (15a/15b).
import json, glob, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(6)
def bm(x, n=2000):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
out = {}
for f in sorted(glob.glob(f"{R}/E4_causal_15_shift_*.json")):
    D = json.load(open(f)); P = D["prompts"]; base = D["base"]
    sel = lambda st, c=None: [j for j, p in enumerate(P) if p["set"] == st and (c is None or p["cond"] == c)]
    print(f"\n=== {f.split('/')[-1]}  (k in units of the persona gap G_l; shift applied at every layer of the block, non-accumulating) ===")
    print(" baseline: P3 default lse %.2f (p_f %.4f p_m %.4f p_it %.3f) | P3 Gemma lse %.2f | P4 default wm %.2f | P4 Gemma wm %.2f | P4 AI_f %.2f AI_m %.2f" % (
        np.mean([base[j]["lse_pron"] for j in sel("P3", "default")]), np.mean([base[j]["p_f"] for j in sel("P3", "default")]), np.mean([base[j]["p_m"] for j in sel("P3", "default")]),
        np.mean([base[j]["p_it"] for j in sel("P3", "default")]), np.mean([base[j]["lse_pron"] for j in sel("P3", "Gemma")]),
        np.mean([base[j]["wm"] for j in sel("P4", "default")]), np.mean([base[j]["wm"] for j in sel("P4", "Gemma")]),
        np.mean([base[j]["wm"] for j in sel("P4", "AI_female")]), np.mean([base[j]["wm"] for j in sel("P4", "AI_male")])))
    for k, C in D["conds"].items():
        m = C["m"]; r = {}
        for st, c in (("P3", "default"), ("P3", "Gemma"), ("P1", "F"), ("P1", "M"), ("P2", "F"), ("P2", "M")):
            r[f"{st}_{c}_dlse"] = bm([m[j]["lse_pron"] - base[j]["lse_pron"] for j in sel(st, c)])
        r["P3_default_p_f"] = float(np.mean([m[j]["p_f"] for j in sel("P3", "default")])); r["P3_default_p_m"] = float(np.mean([m[j]["p_m"] for j in sel("P3", "default")]))
        r["P3_default_p_it"] = float(np.mean([m[j]["p_it"] for j in sel("P3", "default")]))
        for c in ("default", "Gemma", "AI_female", "AI_male"):
            r[f"P4_{c}_dwm"] = bm([m[j]["wm"] - base[j]["wm"] for j in sel("P4", c)])
        r["kl_neu"] = float(np.mean(C["kl_neu"]))
        out[f"{f.split('_')[-1][:-5]}|{k}"] = r
        print(f" {k:22s} P3def {r['P3_default_dlse'][0]:6.2f}[{r['P3_default_dlse'][1]:6.2f},{r['P3_default_dlse'][2]:6.2f}] P3gem {r['P3_Gemma_dlse'][0]:6.2f} | P4def {r['P4_default_dwm'][0]:6.2f}[{r['P4_default_dwm'][1]:6.2f},{r['P4_default_dwm'][2]:6.2f}] P4gem {r['P4_Gemma_dwm'][0]:6.2f} AIf {r['P4_AI_female_dwm'][0]:6.2f} AIm {r['P4_AI_male_dwm'][0]:6.2f} | P1 F/M {r['P1_F_dlse'][0]:5.2f}/{r['P1_M_dlse'][0]:5.2f} P2 F/M {r['P2_F_dlse'][0]:5.2f}/{r['P2_M_dlse'][0]:5.2f} | KL {r['kl_neu']:.4f}")
json.dump(out, open(f"{R}/E4_causal_19_shift_summary.json", "w"))
