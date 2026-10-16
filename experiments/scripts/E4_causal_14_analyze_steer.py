# E4_causal_14 (CPU, offline): analyze deterministic steering sweeps (10a/10b) and generation (12*).
import json, glob, re, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(4)
def bm(x, n=2000):
    x = np.asarray(x, float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {"det": {}, "gen": {}}
for f in sorted(glob.glob(f"{R}/E4_causal_10_steer_det_*.json")):
    D = json.load(open(f)); P = D["prompts"]; base = D["base"]
    sel = lambda st, c=None: [j for j, p in enumerate(P) if p["set"] == st and (c is None or p["cond"] == c)]
    print(f"\n=== {f.split('/')[-1]} ===")
    print(" baseline: P3 default lse %.2f | P3 Gemma lse %.2f | P4 default wm %.2f | P4 Gemma wm %.2f | P2 gap %.2f" % (
        np.mean([base[j]["lse_pron"] for j in sel("P3", "default")]), np.mean([base[j]["lse_pron"] for j in sel("P3", "Gemma")]),
        np.mean([base[j]["wm"] for j in sel("P4", "default")]), np.mean([base[j]["wm"] for j in sel("P4", "Gemma")]),
        np.mean([base[j]["lse_pron"] for j in sel("P2", "F")]) - np.mean([base[j]["lse_pron"] for j in sel("P2", "M")])))
    for k, C in D["conds"].items():
        m = C["m"]
        r = {}
        for st, c in (("P3", "default"), ("P3", "Gemma")):
            r[f"{st}_{c}_dlse"] = bm([m[j]["lse_pron"] - base[j]["lse_pron"] for j in sel(st, c)])
            r[f"{st}_{c}_p_f"] = float(np.mean([m[j]["p_f"] for j in sel(st, c)])); r[f"{st}_{c}_p_m"] = float(np.mean([m[j]["p_m"] for j in sel(st, c)]))
            r[f"{st}_{c}_p_it"] = float(np.mean([m[j]["p_it"] for j in sel(st, c)]))
        for c in ("default", "Gemma", "AI_female", "AI_male"):
            r[f"P4_{c}_dwm"] = bm([m[j]["wm"] - base[j]["wm"] for j in sel("P4", c)])
            r[f"P4_{c}_wm"] = float(np.mean([m[j]["wm"] for j in sel("P4", c)]))
            r[f"P4_{c}_p_wm"] = float(np.mean([m[j]["p_wm"] for j in sel("P4", c)]))
        r["P2_F_dlse"] = bm([m[j]["lse_pron"] - base[j]["lse_pron"] for j in sel("P2", "F")])
        r["P2_M_dlse"] = bm([m[j]["lse_pron"] - base[j]["lse_pron"] for j in sel("P2", "M")])
        r["kl_neu"] = float(np.mean(C["kl_neu"])); r["top1_neu"] = float(np.mean(C["top1_neu"]))
        out["det"][k] = r
        print(f" {k:24s} P3def dlse {r['P3_default_dlse'][0]:6.2f}[{r['P3_default_dlse'][1]:5.2f},{r['P3_default_dlse'][2]:5.2f}] p_f {r['P3_default_p_f']:.3f} p_m {r['P3_default_p_m']:.3f} p_it {r['P3_default_p_it']:.2f} | P4def wm {r['P4_default_wm']:6.2f} (d {r['P4_default_dwm'][0]:5.2f}) pwm {r['P4_default_p_wm']:.2f} | P4gem wm {r['P4_Gemma_wm']:6.2f} | P2 F/M dlse {r['P2_F_dlse'][0]:5.2f}/{r['P2_M_dlse'][0]:5.2f} | KL {r['kl_neu']:.3f} top1 {r['top1_neu']:.2f}")
FEM = re.compile(r"\b(she|her|hers|herself)\b", re.I); MAS = re.compile(r"\b(he|him|his|himself)\b", re.I)
NEU = re.compile(r"\b(it|its|itself)\b", re.I); THEY = re.compile(r"\b(they|them|their|themselves)\b", re.I)
WOM = re.compile(r"\b(woman|female|girl|lady)\b", re.I); MAN = re.compile(r"\b(man|male|guy|boy)\b", re.I)
def distinct3(t):
    w = t.split(); g = list(zip(w, w[1:], w[2:]))
    return len(set(g)) / max(1, len(g))
for f in sorted(glob.glob(f"{R}/E4_causal_12_steer_gen_*.json")):
    D = json.load(open(f))
    print(f"\n=== {f.split('/')[-1]} ===")
    for k, S in D["settings"].items():
        d3 = S["texts"]["describe3p"]; hg = S["texts"]["human_gender"]
        lab = []
        for t in d3:
            nf, nm = len(FEM.findall(t)), len(MAS.findall(t))
            lab.append("F" if nf > nm else ("M" if nm > nf else ("0" if nf == 0 else "=")))
        fem_frac = np.mean([l == "F" for l in lab]); mas_frac = np.mean([l == "M" for l in lab])
        it_rate = np.mean([len(NEU.findall(t)) for t in d3]); they_rate = np.mean([len(THEY.findall(t)) for t in d3])
        first = []
        for t in hg:
            mw, mm = WOM.search(t), MAN.search(t)
            if mw and (not mm or mw.start() < mm.start()): first.append("W")
            elif mm and (not mw or mm.start() < mw.start()): first.append("M")
            else: first.append("none")
        r = {"describe3p_label_counts": {x: lab.count(x) for x in "FM0="}, "describe3p_fem_frac": float(fem_frac), "describe3p_mas_frac": float(mas_frac),
             "describe3p_it_per_sample": float(it_rate), "describe3p_they_per_sample": float(they_rate),
             "human_first_mention": {x: first.count(x) for x in ("W", "M", "none")},
             "clean_logprob_describe": float(np.nanmean(S["clean_logprob"]["describe3p"])), "clean_logprob_human": float(np.nanmean(S["clean_logprob"]["human_gender"])),
             "distinct3": float(np.mean([distinct3(t) for t in d3 + hg]))}
        out["gen"][f"{f.split('_')[-1][:-5]}|{k}"] = r
        print(f" {k:24s} describe3p F/M/none/= {r['describe3p_label_counts']} it/sample {it_rate:.1f} | human first-mention {r['human_first_mention']} | clean logprob {r['clean_logprob_describe']:.2f}/{r['clean_logprob_human']:.2f} distinct3 {r['distinct3']:.2f}")
json.dump(out, open(f"{R}/E4_causal_14_steer_summary.json", "w"))
