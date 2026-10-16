# E4_causal_13 (CPU, offline): analyze ablation results (const + pair modes).
import json, sys, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(3)
def boot_frac(gb, ga, n=2000):
    gb, ga = np.asarray(gb), np.asarray(ga)
    f = lambda ii: 1 - ga[ii].mean() / gb[ii].mean()
    bs = [f(rng.integers(0, len(gb), len(gb))) for _ in range(n)]
    return [float(f(np.arange(len(gb)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def boot_mean(x, n=2000):
    x = np.asarray(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
summary = {}
for mode in ("const", "pair"):
    try:
        D = json.load(open(f"{R}/E4_causal_09_ablate_{mode}.json"))
    except FileNotFoundError:
        continue
    P = D["prompts"]; base = D["base"]
    def sel(st, cond=None):
        return [j for j, p in enumerate(P) if p["set"] == st and (cond is None or p["cond"] == cond)]
    if mode == "const":
        print("=== BASELINE behavior ===")
        for st in ("P1", "P2"):
            for c in ("F", "M"):
                ii = sel(st, c)
                print(f" {st} {c}: lse_pron {np.mean([base[j]['lse_pron'] for j in ii]):6.2f}  p_f {np.mean([base[j]['p_f'] for j in ii]):.3f} p_m {np.mean([base[j]['p_m'] for j in ii]):.3f} p_it {np.mean([base[j]['p_it'] for j in ii]):.3f} p_they {np.mean([base[j]['p_they'] for j in ii]):.3f}")
        for c in ("default", "Gemma"):
            ii = sel("P3", c)
            print(f" P3 {c}: lse_pron {np.mean([base[j]['lse_pron'] for j in ii]):6.2f}  p_f {np.mean([base[j]['p_f'] for j in ii]):.4f} p_m {np.mean([base[j]['p_m'] for j in ii]):.4f} p_it {np.mean([base[j]['p_it'] for j in ii]):.3f} p_they {np.mean([base[j]['p_they'] for j in ii]):.3f}")
        for c in ("default", "Gemma", "AI_female", "AI_male"):
            ii = sel("P4", c)
            print(f" P4 {c}: woman-man logit {np.mean([base[j]['wm'] for j in ii]):6.2f}  p(woman)+p(man) {np.mean([base[j]['p_wm'] for j in ii]):.3f}  per-prefix {[round(base[j]['wm'],2) for j in ii]}")
        summary["baseline"] = {"P3": {c: boot_mean([base[j]["lse_pron"] for j in sel("P3", c)]) for c in ("default", "Gemma")},
                               "P4": {c: boot_mean([base[j]["wm"] for j in sel("P4", c)]) for c in ("default", "Gemma", "AI_female", "AI_male")}}
    print(f"\n=== ABLATION mode={mode}: fraction of F-M pronoun log-odds gap removed (bootstrap over pairs) ===")
    S = {}
    conds = D["conds"]
    dnames = sorted(set(k.split("|")[0] for k in conds)); blocks = sorted(set(k.split("|")[1] for k in conds))
    for dn in dnames + ["u_cont_xfit"]:
        for bn in blocks:
            row = {}
            for st, npairs in (("P1", 36), ("P2", 24)):
                gb, ga, shift = [], [], []
                for key in range(npairs):
                    if dn == "u_cont_xfit":
                        use = "u_cont_B" if (st == "P1" and key < 18) else "u_cont_A"
                    else:
                        use = dn
                    if use == "u_cont_xfit": continue
                    C = conds[f"{use}|{bn}"]; pos = {j: k for k, j in enumerate(C["idx"])}
                    jf = [j for j, p in enumerate(P) if p["set"] == st and p["key"] == key and p["cond"] == "F"][0]
                    jm = [j for j, p in enumerate(P) if p["set"] == st and p["key"] == key and p["cond"] == "M"][0]
                    if jf not in pos or jm not in pos: continue
                    gb.append(base[jf]["lse_pron"] - base[jm]["lse_pron"])
                    ga.append(C["m"][pos[jf]]["lse_pron"] - C["m"][pos[jm]]["lse_pron"])
                    shift.append(0.5 * (C["m"][pos[jf]]["lse_pron"] + C["m"][pos[jm]]["lse_pron"] - base[jf]["lse_pron"] - base[jm]["lse_pron"]))
                if gb:
                    row[st] = {"frac_removed": boot_frac(gb, ga), "gap_base": float(np.mean(gb)), "gap_abl": float(np.mean(ga)), "mean_shift": float(np.mean(shift)), "n": len(gb)}
            if mode == "const" and dn != "u_cont_xfit":
                C = conds[f"{dn}|{bn}"]; pos = {j: k for k, j in enumerate(C["idx"])}
                for c in ("default", "Gemma"):
                    ii = sel("P3", c)
                    row[f"P3_{c}_dlse"] = boot_mean([C["m"][pos[j]]["lse_pron"] - base[j]["lse_pron"] for j in ii])
                    row[f"P3_{c}_dp_it"] = float(np.mean([C["m"][pos[j]]["p_it"] - base[j]["p_it"] for j in ii]))
                for c in ("default", "Gemma"):
                    ii = sel("P4", c)
                    row[f"P4_{c}_dwm"] = boot_mean([C["m"][pos[j]]["wm"] - base[j]["wm"] for j in ii])
                jf = sel("P4", "AI_female"); jm = sel("P4", "AI_male")
                row["P4_AIfm"] = {"frac_removed": boot_frac([base[a]["wm"] - base[b]["wm"] for a, b in zip(jf, jm)],
                                                            [C["m"][pos[a]]["wm"] - C["m"][pos[b]]["wm"] for a, b in zip(jf, jm)])}
                row["kl_neu"] = float(np.mean(C["kl_neu"]))
            S[f"{dn}|{bn}"] = row
            line = f" {dn:12s} {bn:7s}"
            for st in ("P1", "P2"):
                if st in row:
                    fr = row[st]["frac_removed"]; line += f" | {st} frac {fr[0]:5.2f}[{fr[1]:5.2f},{fr[2]:5.2f}] shift {row[st]['mean_shift']:5.2f}"
            if "kl_neu" in row:
                line += f" | P3def dlse {row['P3_default_dlse'][0]:5.2f} P3gem {row['P3_Gemma_dlse'][0]:5.2f} | P4def dwm {row['P4_default_dwm'][0]:5.2f} P4gem {row['P4_Gemma_dwm'][0]:5.2f} | P4 AIf-m frac {row['P4_AIfm']['frac_removed'][0]:5.2f} | KL {row['kl_neu']:.4f}"
            print(line)
    summary[mode] = S
json.dump(summary, open(f"{R}/E4_causal_13_ablate_summary.json", "w"))
