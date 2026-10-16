#!/usr/bin/env python3
"""V5_binding_dissociation_10: collect key numbers from the analysis JSONs into one summary JSON + printed tables."""
import json
RES = "./experiments/results"
F = json.load(open(f"{RES}/V5_binding_dissociation_analysis.json"))["band"]
N = json.load(open(f"{RES}/V5_binding_dissociation_analysis_noqt.json"))["band"]
T2 = json.load(open(f"{RES}/V5_binding_dissociation_twoparty_analysis.json"))
PP = json.load(open(f"{RES}/V5_binding_dissociation_predpos_analysis.json"))
G = json.load(open(f"{RES}/V5_binding_dissociation_geometry.json"))
S = {}
def ci(x): return f"[{x[0]:+.2f},{x[1]:+.2f}]"
print("TABLE 1: three-party factorial (24 items), band L30-39. a=G_A, b=G_U, c=G_T coefficients (J++ pron log-odds); DI = referent vs default")
crit = ["A_qT_I", "A_qS_you", "A_qU_I", "A_qS_I", "U_qA_I", "U_qT_I", "U_qA_you", "A_out_I", "A_init_I", "A_nearT_I", "A_out_you", "U_out_I", "U_out_you", "A_T"]
for t in crit:
    row = {}
    for r in ("jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "jppQ_pron", "probe_ref", "e7dir_self"):
        s = F.get(f"{r}|{t}|L30-39")
        row[r] = s
    s = row["jpp_pron"]
    key = "DI" if "DI" in s else "share_R"
    others = " ".join(f"{r.split('_')[0] if r!='jppQ_pron' else 'jppQ'}:{row[r][key]:+.2f}" for r in ("jpp_noun", "jlens_pron", "naive_pron", "jppQ_pron", "probe_ref", "e7dir_self"))
    print(f"{t:10s} a {s['A']:+.2f} b {s['U']:+.2f} c {s['T']:+.2f} {key} {s[key]:+.2f}{ci(s[key+'_ci'])} | {others}")
    S[f"three_party|{t}|L30-39"] = {r: {k: row[r][k] for k in row[r] if k in ("A", "U", "T", "DI", "DI_ci", "share_R", "share_R_ci")} for r in row}
print("\nTABLE 2: no-T-quote control (24 items), L30-39 J++ pron")
for t in ("A_qU_I", "A_qS_I", "A_qS_you", "U_qA_I", "U_qA_you", "U_out_you", "A_out_I"):
    s = N[f"jpp_pron|{t}|L30-39"]; key = "DI" if "DI" in s else "share_R"
    p = N[f"probe_ref|{t}|L30-39"]
    print(f"{t:10s} a {s['A']:+.2f} b {s['U']:+.2f} c {s['T']:+.2f} {key} {s[key]:+.2f}{ci(s[key+'_ci'])} | probe {key} {p[key]:+.2f}")
    S[f"noqt|{t}|L30-39"] = {k: s[k] for k in ("A", "U", "T", key, key + "_ci")}
print("\nTABLE 3: two-party control (24 items, E7 passages + quotes), L30-39")
for t in ("A_qU_I", "U_qA_I", "U_qA_you", "A_out_I", "A_out_you", "U_out_I", "U_out_you"):
    row = []
    for r in ("jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "jppQ_pron", "probe_ref", "e7dir_self"):
        s = T2["band"][f"{r}|{t}|L30-39"]; row.append((r, s))
    s = row[0][1]
    print(f"{t:10s} a {s['A']:+.2f} b {s['U']:+.2f} idx {s['idx']:+.2f}{ci(s['idx_ci'])} | " + " ".join(f"{r}:{x['idx']:+.2f}" for r, x in row[1:]))
    S[f"twoparty|{t}|L30-39"] = {r: {k: x[k] for k in ("A", "U", "idx", "idx_ci")} for r, x in row}
print("behaviour two-party:", {k: (round(v["A"], 2), round(v["U"], 2), round(v["idx"], 2), [round(x, 2) for x in v["idx_ci"]]) for k, v in T2["behaviour"].items()})
S["twoparty_behaviour"] = T2["behaviour"]
print("\nTABLE 4: prediction positions (24 items): behaviour (real next token, woman-set vs man-set LSE) and J++ L30-39")
for s_, v in PP["behaviour"].items():
    l = PP["band"][f"jpp_pron|{s_}|L30-39"]
    di = f"DI {v['DI']:+.2f}{ci(v['DI_ci'])}" if "DI" in v else f"share {v['share_R']:+.2f}"
    print(f"{s_:6s} beh a {v['A']:+.2f} b {v['U']:+.2f} c {v['T']:+.2f} {di} mass {v['mass_F4M4']:.3f} | J++L30-39 a {l['A']:+.2f} b {l['U']:+.2f} c {l['T']:+.2f}")
S["predpos"] = PP["behaviour"]
print("\nTABLE 5: late-layer re-test, A_out_I BI vs U (three-party | two-party), bands L20-29 L30-39 L40-49 L50-58")
for r in ("jpp_pron", "jpp_noun", "jpp_noun4", "jppQ_pron", "jppQ_noun", "jlensQ_pron", "probe_ref", "e7dir_self"):
    a = [F[f"{r}|A_out_I|{b}"] for b in ("L20-29", "L30-39", "L40-49", "L50-58")]
    t2 = [T2["band"][f"{r}|A_out_I|{b}"] for b in ("L20-29", "L30-39", "L40-49", "L50-58")]
    print(f"{r:12s} " + " ".join(f"{x['BI_vs_U']:+.2f}(a{x['A']:+.1f},c{x['T']:+.1f})" for x in a) + " | " + " ".join(f"{x['idx']:+.2f}(a{x['A']:+.1f})" for x in t2))
    S[f"late|{r}|A_out_I"] = {"three_party_BI_vs_U": [x["BI_vs_U"] for x in a], "three_party_a": [x["A"] for x in a], "three_party_c": [x["T"] for x in a],
                              "two_party_BI": [x["idx"] for x in t2], "two_party_a": [x["A"] for x in t2]}
print("\nTABLE 6: raw-residual corrected cosines with dA@A_out_I (layers", G["layers"], ")")
for k, v in G["pairs"].items():
    if k.startswith("raw|"):
        print(f"{k[4:70]:66s}", " ".join(f"{x:+.2f}" for x in v["ccos"]))
S["geometry_layers"] = G["layers"]; S["geometry_raw"] = {k[4:]: v["ccos"] for k, v in G["pairs"].items() if k.startswith("raw|")}
json.dump(S, open(f"{RES}/V5_binding_dissociation_summary.json", "w"), indent=1)
