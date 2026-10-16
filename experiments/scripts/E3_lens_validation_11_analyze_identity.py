# E3_lens_validation (b2) identity analysis (local). Unit = passage (14), paired bootstrap over passages (2000).
import json, numpy as np
R = "./experiments/results/"
rng = np.random.default_rng(0); NB = 2000
recs = json.load(open(R + "E3_lens_validation_b2_identity.json"))
CONDS = ["default", "nameless_AI", "Gemma", "Sarah_AI", "James_AI", "woman", "man"]
BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
FEATS = ["jpp_pron", "jpp_noun", "jppc_pron", "jppc_noun", "jlens_pron", "jlensc_pron", "naive_pron", "naivec_pron", "itchat_raw_pron", "itchat_cent_pron", "itchat_raw_noun", "itprose_raw_pron",
         "it_raw_pron", "it_norm_pron", "it_cent_pron", "it_raw_noun", "it_cent_noun", "it_raw_rand1", "it_raw_rand2", "jpull_raw_pron", "jpull_cent_pron"]
NP = 14
def arr(cat, f):
    A = np.full((len(CONDS), NP, 60), np.nan)
    for r in recs:
        if cat in r: A[CONDS.index(r["cond"]), r["passage"]] = r[cat][f]
    return A
out = {"positions": {}, "sep": {}, "agree": {}, "per_layer_pos": {}, "raw_means": {}}
def pos_stat(B):  # B [C, P] band values -> positions [C]
    m = np.nanmean(B, 1); return (m - m[CONDS.index("man")]) / (m[CONDS.index("woman")] - m[CONDS.index("man")])
idxs = rng.integers(0, NP, (NB, NP))
for cat in ("self", "nonself_asst", "user_I"):
    for f in FEATS:
        A = arr(cat, f)
        if np.isnan(A).all(): continue
        # per-layer positions (point estimates)
        m = np.nanmean(A, 1)
        out["per_layer_pos"][f"{cat}|{f}"] = ((m - m[6]) / (m[5] - m[6])).tolist()
        out["raw_means"][f"{cat}|{f}"] = m.tolist()
        for b, rg in BANDS.items():
            B = A[:, :, list(rg)].mean(2)  # [C, P]
            p0 = pos_stat(B); bs = np.array([pos_stat(B[:, ii]) for ii in idxs])
            out["positions"][f"{cat}|{f}|{b}"] = {c: [round(float(p0[i]), 3), round(float(np.nanpercentile(bs[:, i], 2.5)), 3), round(float(np.nanpercentile(bs[:, i], 97.5)), 3)] for i, c in enumerate(CONDS)}
            gap = B[5] - B[6]  # paired woman - man per passage
            out["sep"][f"{cat}|{f}|{b}"] = {"gap_mean": float(gap.mean()), "gap_d": float(gap.mean() / (gap.std(ddof=1) + 1e-12)),
                                            "frac_passages_pos": float((gap > 0).mean())}
# agreement of default (and others) between J++ and IT readouts: paired bootstrap of position differences
for cat in ("self",):
    for fa, fb in (("jpp_pron", "itchat_raw_pron"), ("jpp_pron", "itprose_raw_pron"), ("jpp_pron", "it_raw_pron"), ("jpp_pron", "it_norm_pron"), ("jppc_pron", "it_cent_pron"), ("jpp_pron", "jpull_raw_pron"), ("jpp_noun", "it_raw_noun"), ("jlens_pron", "it_raw_pron")):
        A1, A2 = arr(cat, fa), arr(cat, fb)
        for b, rg in BANDS.items():
            B1 = A1[:, :, list(rg)].mean(2); B2 = A2[:, :, list(rg)].mean(2)
            d0 = pos_stat(B1) - pos_stat(B2)
            bs = np.array([pos_stat(B1[:, ii]) - pos_stat(B2[:, ii]) for ii in idxs])
            out["agree"][f"{cat}|{fa}-{fb}|{b}"] = {c: [round(float(d0[i]), 3), round(float(np.nanpercentile(bs[:, i], 2.5)), 3), round(float(np.nanpercentile(bs[:, i], 97.5)), 3)] for i, c in enumerate(CONDS) if c not in ("woman", "man")}
json.dump(out, open(R + "E3_lens_validation_b2_identity_analysis.json", "w"), indent=1)
def show(cat, f):
    print(f"\n[{cat}] {f}: position on woman(1)-man(0) scale, mean [95% CI]; woman-man gap d")
    for b in BANDS:
        P = out["positions"][f"{cat}|{f}|{b}"]; s = out["sep"][f"{cat}|{f}|{b}"]
        print(f"  {b}: " + " ".join(f"{c}:{P[c][0]:+.2f}[{P[c][1]:+.2f},{P[c][2]:+.2f}]" for c in CONDS[:5]) + f" | gap={s['gap_mean']:+.2f} d={s['gap_d']:+.2f} pos={s['frac_passages_pos']:.2f}")
for f in ["jpp_pron", "jppc_pron", "jpp_noun", "jlens_pron", "itchat_raw_pron", "itprose_raw_pron", "it_raw_pron", "it_norm_pron", "it_cent_pron", "it_raw_noun", "jpull_raw_pron", "it_raw_rand1", "it_raw_rand2"]:
    show("self", f)
for f in ["jpp_pron", "it_raw_pron"]:
    show("nonself_asst", f)
print("\n== agreement (position difference A-B, self tokens) ==")
for k, v in out["agree"].items():
    print(k, " ".join(f"{c}:{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]" for c, x in v.items()))

# ---- AI-anchored scale: James(AI)=0, Sarah(AI)=1 (removes the AI-vs-human component shared by all AI conditions)
def pos_ai(B):
    m = np.nanmean(B, 1); j, s_ = CONDS.index("James_AI"), CONDS.index("Sarah_AI")
    return (m - m[j]) / (m[s_] - m[j])
out["positions_ai_anchor"] = {}; out["sarah_minus_james_on_human_scale"] = {}
print("\n== AI-anchored position of default / nameless / Gemma (James_AI=0, Sarah_AI=1), self tokens; and (Sarah_AI-James_AI) as fraction of human woman-man gap ==")
for f in ["jpp_pron", "jpp_noun", "jlens_pron", "it_raw_pron", "itchat_raw_pron", "itprose_raw_pron", "it_raw_noun", "itchat_raw_noun"]:
    A = arr("self", f)
    for b, rg in BANDS.items():
        B = A[:, :, list(rg)].mean(2)
        p0 = pos_ai(B); bs = np.array([pos_ai(B[:, ii]) for ii in idxs])
        out["positions_ai_anchor"][f"self|{f}|{b}"] = {c: [round(float(p0[i]), 3), round(float(np.nanpercentile(bs[:, i], 2.5)), 3), round(float(np.nanpercentile(bs[:, i], 97.5)), 3)] for i, c in enumerate(CONDS)}
        ph = pos_stat(B); bh = np.array([pos_stat(B[:, ii]) for ii in idxs])
        sj = ph[3] - ph[4]; sjb = bh[:, 3] - bh[:, 4]
        out["sarah_minus_james_on_human_scale"][f"self|{f}|{b}"] = [round(float(sj), 3), round(float(np.nanpercentile(sjb, 2.5)), 3), round(float(np.nanpercentile(sjb, 97.5)), 3)]
    print(f"{f:17s} " + " ".join(f"{b}: def {out['positions_ai_anchor'][f'self|{f}|{b}']['default'][0]:+.2f}[{out['positions_ai_anchor'][f'self|{f}|{b}']['default'][1]:+.2f},{out['positions_ai_anchor'][f'self|{f}|{b}']['default'][2]:+.2f}] Gem {out['positions_ai_anchor'][f'self|{f}|{b}']['Gemma'][0]:+.2f} S-J {out['sarah_minus_james_on_human_scale'][f'self|{f}|{b}'][0]:+.2f}" for b in BANDS))
json.dump(out, open(R + "E3_lens_validation_b2_identity_analysis.json", "w"), indent=1)
