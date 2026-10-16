# Lens-vs-behavior validation: at the pronoun-prediction position, does the lens readout at layer l track the real
# final next-token gender log-odds?  Unit for CIs = prompt (n=336) for r; condition means (14 = 7 names x 2 frames).
import json, sys, numpy as np
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_b_rows.json"))
L = json.load(open(R + "E2_behavior_b_lens.json"))
idx = L["idx"]; sub = [rows[i] for i in idx]
conds = sorted(set((r["name"], r["frame"]) for r in sub))
cid = np.array([conds.index((r["name"], r["frame"])) for r in sub])
tmpl = np.array([r["tmpl"] for r in sub])
rng = np.random.default_rng(7)
bands = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}
out = {}
for metric in ("lse_pron", "neuter_vs_human"):
    real = np.array(L["lens"]["jpp"][metric][59])  # == real final readout
    print(f"\n### metric={metric}: real final mean {real.mean():+.2f}; condition-mean range {np.ptp([real[cid==c].mean() for c in range(len(conds))]):.2f}")
    for kind in ("jpp", "jlens", "naive"):
        A = np.array(L["lens"][kind][metric])  # [60, N]
        per_r, per_rc, per_bias, per_slope = [], [], [], []
        for l in range(60):
            x = A[l]
            per_r.append(float(np.corrcoef(x, real)[0, 1]))
            xm = np.array([x[cid == c].mean() for c in range(len(conds))]); ym = np.array([real[cid == c].mean() for c in range(len(conds))])
            per_rc.append(float(np.corrcoef(xm, ym)[0, 1]))
            per_bias.append(float((x - real).mean()))
            per_slope.append(float(np.polyfit(ym, xm, 1)[0]))
        out[f"{metric}|{kind}"] = {"r_prompt": per_r, "r_condmean": per_rc, "bias": per_bias, "slope_condmean": per_slope}
        # bootstrap CI of band-mean r over templates (cluster bootstrap: resample templates)
        line = []
        for b, rg in bands.items():
            rb = np.mean([per_r[l] for l in rg]); rcb = np.mean([per_rc[l] for l in rg]); bb = np.mean([per_bias[l] for l in rg]); sb = np.mean([per_slope[l] for l in rg])
            line.append(f"{b}: r={rb:+.2f} rc={rcb:+.2f} slope={sb:+.2f} bias={bb:+.2f}")
        print(f"[{kind}] " + " | ".join(line))
# cluster-bootstrap CI on the condition-mean correlation for J++ L30-39 lse_pron
A = np.array(L["lens"]["jpp"]["lse_pron"]); real = np.array(L["lens"]["jpp"]["lse_pron"][59])
T = sorted(set(tmpl))
for b in ("L20-29", "L30-39", "L40-49", "L50-58"):
    rs = []
    x = A[list(bands[b])].mean(0)
    for _ in range(2000):
        pick = rng.choice(T, len(T))
        msk = np.concatenate([np.where(tmpl == t)[0] for t in pick])
        xm = np.array([x[msk][cid[msk] == c].mean() for c in range(len(conds))]); ym = np.array([real[msk][cid[msk] == c].mean() for c in range(len(conds))])
        rs.append(np.corrcoef(xm, ym)[0, 1])
    xm = np.array([x[cid == c].mean() for c in range(len(conds))]); ym = np.array([real[cid == c].mean() for c in range(len(conds))])
    r0 = np.corrcoef(xm, ym)[0, 1]
    out[f"jpp_lse_condmean_r_{b}"] = [float(r0), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
    print(f"J++ band-avg {b} lse_pron: condition-mean r = {r0:+.2f} [{np.percentile(rs,2.5):+.2f},{np.percentile(rs,97.5):+.2f}] (template-cluster bootstrap)")
    print("   cond means lens vs real:", [(c[0][:6] + "/" + c[1][:1], round(float(a), 2), round(float(bq), 2)) for c, a, bq in zip(conds, xm, ym)])
json.dump(out, open(R + "E2_behavior_b_lens_summary.json", "w"), indent=1)
