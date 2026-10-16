# E4_causal_18 (CPU, offline): unified pair-gap analysis (log-odds AND bounded q = p_f/(p_f+p_m)) for pair-mode results
# (09b, 17) with bootstrap CIs over pairs. Cross-fit for u_cont on P1 handled for 09b.
import json, sys, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(5)
def frac(gb, ga, n=2000):
    gb, ga = np.asarray(gb), np.asarray(ga)
    f = lambda ii: 1 - ga[ii].mean() / gb[ii].mean()
    bs = [f(rng.integers(0, len(gb), len(gb))) for _ in range(n)]
    return [round(float(f(np.arange(len(gb)))), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
q = lambda m: m["p_f"] / (m["p_f"] + m["p_m"] + 1e-30)
def analyze(fname, xfit=False):
    D = json.load(open(f"{R}/{fname}")); P = D["prompts"]; base = D["base"]
    res = {}
    keys = list(D["conds"].keys())
    if xfit:
        keys += [f"u_cont_xfit|{b}" for b in sorted(set(k.split("|")[1] for k in keys))]
    for key in keys:
        dn, bn = key.split("|")
        row = {}
        for st in ("P1", "P2"):
            ks = sorted(set(p["key"] for p in P if p["set"] == st))
            gb, ga, qb, qa, mass_b, mass_a = [], [], [], [], [], []
            for kk in ks:
                use = dn
                if dn == "u_cont_xfit":
                    use = "u_cont_B" if (st == "P1" and kk < 18) else "u_cont_A"
                C = D["conds"][f"{use}|{bn}"]; pos = {j: i for i, j in enumerate(C["idx"])}
                jf = [j for j, p in enumerate(P) if p["set"] == st and p["key"] == kk and p["cond"] == "F"][0]
                jm = [j for j, p in enumerate(P) if p["set"] == st and p["key"] == kk and p["cond"] == "M"][0]
                if P[jf]["partner"] is None or jf not in pos: continue
                mf, mm = C["m"][pos[jf]], C["m"][pos[jm]]
                gb.append(base[jf]["lse_pron"] - base[jm]["lse_pron"]); ga.append(mf["lse_pron"] - mm["lse_pron"])
                qb.append(q(base[jf]) - q(base[jm])); qa.append(q(mf) - q(mm))
                mass_b.append(0.5 * (base[jf]["p_f"] + base[jf]["p_m"] + base[jm]["p_f"] + base[jm]["p_m"]))
                mass_a.append(0.5 * (mf["p_f"] + mf["p_m"] + mm["p_f"] + mm["p_m"]))
            row[st] = {"frac_logodds": frac(gb, ga), "frac_q": frac(qb, qa), "gap_logodds_base": float(np.mean(gb)), "gap_logodds_abl": float(np.mean(ga)),
                       "gap_q_base": float(np.mean(qb)), "gap_q_abl": float(np.mean(qa)), "pron_mass_ratio": float(np.mean(mass_a) / np.mean(mass_b)), "n": len(gb)}
        res[key] = row
    return res
out = {}
for fname, xf in (("E4_causal_09_ablate_pair.json", True), ("E4_causal_09_ablate_pair_ctrl.json", False), ("E4_causal_17_pb_specificity_data.json", False)):
    try:
        out[fname] = analyze(fname, xf)
    except FileNotFoundError:
        continue
    print(f"\n=== {fname}: fraction of F-M gap removed by pair-matched clamp  [log-odds | bounded q]  (pron-mass ratio abl/base) ===")
    for k, r in out[fname].items():
        print(f" {k:24s} " + " | ".join(f"{st}: lo {r[st]['frac_logodds'][0]:5.2f}[{r[st]['frac_logodds'][1]:5.2f},{r[st]['frac_logodds'][2]:5.2f}] q {r[st]['frac_q'][0]:5.2f}[{r[st]['frac_q'][1]:5.2f},{r[st]['frac_q'][2]:5.2f}] mass {r[st]['pron_mass_ratio']:.2f}" for st in ("P1", "P2")))
json.dump(out, open(f"{R}/E4_causal_18_pairs_summary.json", "w"))
