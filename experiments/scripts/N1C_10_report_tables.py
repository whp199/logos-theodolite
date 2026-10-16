#!/usr/bin/env python3
# N1C_10_report_tables -- per-layer summary tables for N1_REPORT.md, generated from the JSON results (no hand copying).
# Run: .venv/bin/python experiments/scripts/N1C_10_report_tables.py > <file>
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import ROOT, LAYERS

M = json.load(open(f"{ROOT}/N1C_matrix.json"))
AL = json.load(open(f"{ROOT}/N1C_align.json"))
CO = json.load(open(f"{ROOT}/N1C_coherence.json"))
A1 = json.load(open(f"{ROOT}/N1A_metrics.json"))
F, MA, Q = M["full_p10"], M["matrix_A_p10"], M["q"]
H = "| | " + " | ".join(f"L{l}" for l in LAYERS) + " | all |\n|---|" + "---|" * (len(LAYERS) + 1)

def rn(res, d=3, star=True):
    def c(b):
        s = f"{b['real']:.{d}f}/{b['null']:.{d}f}"
        return s + ("*" if star and b["diff_ci"][0] > 0 else "")
    return [c(res["per_layer"][str(l)]) for l in LAYERS] + [c(res["overall"])]
def df(res, scale=1.0, d=3):
    c = lambda b: f"{b['diff']*scale:+.{d}f}" + ("*" if b["diff_ci"][0] > 0 else "")
    return [c(res["per_layer"][str(l)]) for l in LAYERS] + [c(res["overall"])]
def row(name, cells):
    return f"| {name} | " + " | ".join(cells) + " |"

print("### T1 output labels vs causal steering\n" + H)
for k, nm in (("JV->C", "P@10 J-VINDEX->causal (FULL)"), ("NV->C", "P@10 naive->causal (FULL)"), ("JL->C", "P@10 JLens->causal (FULL)"),
              ("JVz->C", "P@10 J-VINDEX prior-z->causal (FULL)")):
    print(row(nm, rn(F[k])))
for k, nm in (("kA.jpp.rho", "Spearman J-VINDEX vs causal, lift (N1A)"), ("kA.naive.rho", "Spearman naive vs causal, lift (N1A)")):
    s = A1["summary"][k]
    print(row(nm, [f"{s['per_layer'][str(l)]['lift']['mean']:+.3f}" for l in LAYERS] + [f"{s['overall']['lift']['mean']:+.3f}"]))
ce = A1["notes"]["ceiling_kA"]
print(row("reliability ceiling for rho (N1A split-half)", [f"{ce[str(l)]['ceiling_rho_median']:.2f}" for l in LAYERS] + ["-"]))
pj = Q["Q1_paired_full"]["JV-NV"]
print(row("paired P@10 J-VINDEX minus naive", [f"{pj[str(l)]['mean']:+.3f}" for l in LAYERS] + [f"{pj['overall']['mean']:+.3f} [{pj['overall']['ci'][0]:+.3f},{pj['overall']['ci'][1]:+.3f}]"]))
pz = Q["Q1_paired_full"]["JVz-JV"]
print(row("paired P@10 prior-z minus raw (J-VINDEX)", [f"{pz[str(l)]['mean']:+.3f}" for l in LAYERS] + [f"{pz['overall']['mean']:+.3f} [{pz['overall']['ci'][0]:+.3f},{pz['overall']['ci'][1]:+.3f}]"]))
print(row("P@10 J-VINDEX->causal on attested vocab A", rn(MA["JV"]["C"])))

print("\n### T2 say test: causal / static output labels vs the tokens that actually follow when the pole fires\n" + H)
print(row("P@10 causal->next (A, enrichment top-100)", rn(MA["C"]["Enx"])))
print(row("P@10 causal->next, top-0.1% events", rn(Q["Q2"]["C->Enx01"])))
print(row("P@10 J-VINDEX->next", rn(MA["JV"]["Enx"])))
print(row("P@10 naive->next", rn(MA["NV"]["Enx"])))
print(row("ceiling: next-token label half0->half1", rn(Q["Q2"]["ceiling_Enx_h0->Enx_h1"])))
print(row("ALIGN_next causal (pct points, lift)", df(AL["next"]["C"], 100, 1)))
print(row("ALIGN_next J-VINDEX (lift)", df(AL["next"]["JV"], 100, 1)))
print(row("ALIGN_next naive (lift)", df(AL["next"]["NV"], 100, 1)))
print(row("ALIGN_next ceiling (split-half, lift)", df(AL["ceiling"]["next"], 100, 1)))
sf = Q["Q2_say_fraction_ge2of10"]
print(row("'say' poles: >=2/10 causal tokens in next top-100 (null)", [f"{sf[str(l)]['frac']:.2f} ({sf[str(l)]['null_frac']:.2f})" for l in LAYERS] + ["-"]))
print(row("Spearman causal vs mean-act-by-next-token (W5)", rn(M["matrix_rho"]["C"]["Rnx"])))

print("\n### T3 input side: gate_trigger_tokens and alternatives vs the current tokens at firing positions\n" + H)
print(row("P@10 gate (tool, J++)->input", rn(MA["G"]["Ein"])))
print(row("P@10 gate prior-z->input", rn(Q["Q3"]["Gz->Ein"])))
print(row("P@10 gate direct embedding->input", rn(MA["Gemb"]["Ein"])))
print(row("P@10 W_up via J++ (pole-signed)->input", rn(MA["U"]["Ein"])))
print(row("ceiling: input label half0->half1", rn(Q["Q2"]["ceiling_Ein_h0->Ein_h1"])))
print(row("ALIGN_cur gate (tool) (pct points, lift)", df(AL["cur"]["G"], 100, 1)))
print(row("ALIGN_cur gate direct embedding (lift)", df(AL["cur"]["Gemb"], 100, 1)))
print(row("ALIGN_cur W_up via J++ (lift)", df(AL["cur"]["U"], 100, 1)))
print(row("ALIGN_cur ceiling (split-half, lift)", df(AL["ceiling"]["cur"], 100, 1)))
print(row("P@10 gate->J-VINDEX same neuron (shared-lens overlap)", rn(MA["G"]["JV"])))

print("\n### T4 bipolarity (partner = the other pole of the same neuron; null = opposite pole of another neuron)\n" + H)
q4 = Q["Q4"]
print(row("P@10 input(+/-) -> input(other pole)", rn(q4["Ein_s->Ein_-s"])))
print(row("P@10 next(+/-) -> next(other pole)", rn(q4["Enx_s->Enx_-s"])))
print(row("P@10 causal(s) -> NEGATED causal(-s) (antipodality)", rn(q4["C_s->(-C_-s)"])))
print(row("Spearman causal(s) vs causal(-s)", rn(q4["rho_C_s_C_-s"], star=False)))
print(row("Spearman input-token response(s) vs (-s)", rn(q4["rho_Rin_s_Rin_-s"], star=False)))
sc = q4["self_minus_cross_rho_C_Rnx"]
print(row("rho(causal_s, next-response_s) minus rho(causal_s, next-response_-s)", [f"{sc[str(l)]['mean']:+.3f}" for l in LAYERS] + [f"{sc['overall']['mean']:+.3f} [{sc['overall']['ci'][0]:+.3f},{sc['overall']['ci'][1]:+.3f}]"]))
print(row("ALIGN_next causal(s) at the OTHER pole's firing positions (lift)", df(AL["cross_pole_next"]["C"], 100, 1)))

print("\n### T5 label tightness (mean pairwise embedding cosine of the top-10 tokens)\n" + "| | " + " | ".join(f"L{l}" for l in LAYERS) + " |\n|---|" + "---|" * len(LAYERS))
for k, nm in (("real_JV", "J-VINDEX, sampled neurons"), ("single_JV", "J-VINDEX, 32 other single neurons"), ("mix512_JV", "J-VINDEX, 512-neuron random mixtures"),
              ("iso_JV", "J-VINDEX, isotropic random vectors"), ("real_NV", "naive, sampled neurons"), ("mix512_NV", "naive, 512-neuron mixtures"),
              ("iso_NV", "naive, isotropic"), ("real_C", "causal top-10"), ("real_Enx", "empirical next-token top-10")):
    print(row(nm, [f"{CO['per_layer'][str(l)][k]['mean']:.3f}" for l in LAYERS]))
for k, nm in (("real_JV-mix512_JV", "J-VINDEX: neurons minus mixtures"), ("real_JV-real_NV", "neurons: J-VINDEX minus naive")):
    print(row(nm, [f"{CO['per_layer'][str(l)][k]['diff']:+.3f}" + ("*" if CO['per_layer'][str(l)][k]['ci'][0] > 0 else "") for l in LAYERS]))
print("\nbands:", json.dumps({b: {k: v for k, v in d.items() if "-" in k} for b, d in CO["bands"].items()}))

print("\n### Matrix (P@10->100 on vocab A, real/null; * = lift CI > 0)")
MAIN = ["JV", "NV", "JVz", "C", "Enx", "Ein", "G", "Gemb", "U"]
NAMES = {"JV": "J-VINDEX", "NV": "naive", "JVz": "J-VINDEX z", "C": "causal", "Enx": "next (emp)", "Ein": "input (emp)", "G": "gate (tool)", "Gemb": "gate-embed", "U": "W_up J++"}
for band, title in (("late_L40_57", "L40-57"), ("early_L10_30", "L10-30")):
    print(f"\n{title}\n\n| X (top-10) \\ Y (top-100) | " + " | ".join(NAMES[y] for y in MAIN) + " |\n|---|" + "---|" * len(MAIN))
    for x in MAIN:
        cs = []
        for y in MAIN:
            if x == y:
                cs.append("-"); continue
            b = MA[x][y][band]
            cs.append(f"{b['real']:.3f}/{b['null']:.3f}" + ("*" if b["diff_ci"][0] > 0 else ""))
        print(f"| {NAMES[x]} | " + " | ".join(cs) + " |")
pr = Q["static_confidence_predictor"]
print("\nstatic predictor:", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'per_layer'}) for k, v in pr.items()}))
print("jv_nv_overlap per layer:", pr["jv_nv_top10_overlap"]["per_layer"])
