#!/usr/bin/env python3
# N1V_05_tables -- render markdown tables for N1V_robustness.md from N1V_analysis.json (CPU, no computation).
import json
R = "./experiments/results"
d = json.load(open(f"{R}/N1V_analysis.json"))
T, B, PA = d["tables"], d["bipolar"], d["paired"]
LAY = ["10", "20", "30", "40", "45", "50", "55", "57"]
COLS = LAY + ["early", "late", "all"]
HDR = "| | " + " | ".join(("L" + c if c.isdigit() else c) for c in COLS) + " |\n|" + "---|" * (len(COLS) + 1)


def cell(b, d=3, lift=False):
    sig = b["ci"][0] > 0 or b["ci"][1] < 0
    if lift:
        return f"{b['lift']:+.{d}f}" + ("*" if sig else "")
    return f"{b['real']:.{d}f}/{b['null']:.{d}f}" + ("*" if sig else "")


def row(name, tab, **kw):
    return f"| {name} | " + " | ".join(cell(tab[c], **kw) for c in COLS) + " |"


out = []
out.append("#### P@10 (label top-10 in causal top-100, FULL vocab), real/null\n" + HDR)
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    for X in ("JV", "NV", "JL"):
        out.append(row(f"{X} -> C[{c}]", T[f"p10|{X}->C:{c}"]))
out.append("\n#### Spearman lift over the full vocabulary (real - null)\n" + HDR)
for c in ["A8k60", "A8k10", "L8k60"]:
    for X in ("JV", "NV"):
        out.append(row(f"{X} ~ C[{c}]", T[f"rho|{X}->C:{c}"], lift=True))
    out.append(row(f"JV ~ Canti[{c}]", T[f"rho|JV->Canti:{c}"], lift=True))
out.append("\n#### Reliability of the causal measurement (median split-half rho over contexts) / attenuation ceiling sqrt(SB(r))")
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    r = d["reliability"][c]; ce = d["reliability"][c + "_attenuation_ceiling"]; kl = d["kl"][c]
    out.append(f"- {c}: " + ", ".join(f"L{l} {r[l]:.2f}/{ce[l]:.2f}" for l in LAY) + " | median KL " + ", ".join(f"{kl[l]:.3f}" for l in LAY))
out.append("\n#### Paired J-VINDEX minus naive, P@10 (neuron-cluster bootstrap CIs)")
out.append("| intervention | L40-45 | L50-57 | late | all | Spearman all |\n|---|---|---|---|---|---|")
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    p = PA[f"p10|JV-NV|{c}"]; q = PA[f"rho|JV-NV|{c}"]
    f = lambda x: f"{x['lift']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    out.append(f"| {c} | {f(p['L40_45_cluster'])} | {f(p['L50_57_cluster'])} | {f(p['late_cluster'])} | {f(p['all_cluster'])} | {f(q['all_cluster'])} |")
out.append("\n#### Paired JLens minus naive, P@10 (neuron-cluster CIs)")
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    p = PA[f"p10|JL-NV|{c}"]
    out.append(f"- {c}: L40-45 {p['L40_45_cluster']['lift']:+.3f} [{p['L40_45_cluster']['ci'][0]:+.3f}, {p['L40_45_cluster']['ci'][1]:+.3f}], "
               f"L50-57 {p['L50_57_cluster']['lift']:+.3f} [{p['L50_57_cluster']['ci'][0]:+.3f}, {p['L50_57_cluster']['ci'][1]:+.3f}], "
               f"all {p['all_cluster']['lift']:+.3f} [{p['all_cluster']['ci'][0]:+.3f}, {p['all_cluster']['ci'][1]:+.3f}]")
out.append("\n#### Prior-z (J-VINDEX z-scored vs isotropic random vectors) minus raw, P@10")
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    p = PA[f"p10|JVz-JV|{c}"]
    out.append(f"- {c}: late {p['late_cluster']['lift']:+.3f} [{p['late_cluster']['ci'][0]:+.3f}, {p['late_cluster']['ci'][1]:+.3f}], "
               f"all {p['all_cluster']['lift']:+.3f} [{p['all_cluster']['ci'][0]:+.3f}, {p['all_cluster']['ci'][1]:+.3f}]")
out.append("\n#### Bipolarity on the output side: Spearman(c+, c-), same neuron / null (other neurons' opposite pole)\n" + HDR)
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    out.append(row(f"rho(c+,c-) [{c}]", B[f"rho|{c}"]))
out.append("\n| between-neuron same-sign rho (generic response) | " + " | ".join("L" + l for l in LAY) + " |\n|" + "---|" * 9)
for c in ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]:
    g = B[f"generic_rho|{c}"]
    out.append(f"| {c} | " + " | ".join(f"{g[l]:+.2f}" for l in LAY) + " |")
out.append("\n#### Stability of the causal label across intervention styles: P@10 C[a] -> C[b], same pole, real/null\n" + HDR)
for k in ["A8k60->A1k60", "A8k60->A8k10", "A8k10->L8k10", "L8k10->L8k60"]:
    out.append(row(k, d["stability"][f"p10|{k}"]))
out.append("\n#### Activation-based labels (vocab A), P@10 real/null\n" + HDR)
for k, nm in [("A|C:A8k60->Enx (say)", "causal[A8k60] -> next-token label (say test)"),
              ("A|C:A1k60->Enx (say)", "causal[A1k60] -> next-token label"),
              ("A|JV->Enx", "J-VINDEX -> next-token label"), ("A|NV->Enx", "naive -> next-token label"),
              ("A|JV->Ein", "J-VINDEX -> input-token label"),
              ("A|G(tool gate)->Ein", "gate_trigger_tokens -> input-token label"),
              ("A|Gemb->Ein", "direct gate embedding readout -> input-token label"),
              ("A|ceiling input (half0->half1)", "ceiling: input label half0 -> half1"),
              ("A|ceiling next (half0->half1)", "ceiling: next label half0 -> half1"),
              ("A|Ein(s)->Ein(-s)", "input label pole s -> pole -s"),
              ("A|C:A8k60->JV (A)", "causal[A8k60] -> J-VINDEX (vocab A)")]:
    out.append(row(nm, T[k]))
out.append("\n#### ALIGN (percentile-point lift of the actual token at firing positions), real/null\n" + HDR)
for k, nm in [("ALIGN|G|cur", "gate_trigger_tokens, current token"), ("ALIGN|Gemb|cur", "direct gate embedding, current token"),
              ("ALIGN|Ein_h0|cur", "ceiling: input label half0, on half-1 positions"),
              ("ALIGN|C:A8k60|next", "causal[A8k60], next token"), ("ALIGN|Enx_h0|next", "ceiling: next label half0, on half-1 positions")]:
    tab = T[k]
    out.append(f"| {nm} | " + " | ".join(f"{tab[c]['real']:+.1f}/{tab[c]['null']:+.1f}" + ("*" if (tab[c]['ci'][0] > 0 or tab[c]['ci'][1] < 0) else "") for c in COLS) + " |")
m = d["matrix_A"]
names = ["JV", "NV", "C:A8k60", "Enx", "Ein"]
out.append("\n#### Agreement matrix on vocab A (row label top-10 inside column label top-100), real/null, early / late")
out.append("| row \\ col | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1))
for X in names + ["G"]:
    cells = []
    for Y in names:
        if X == Y:
            cells.append("-"); continue
        t = m[f"{X}->{Y}"]
        cells.append(" / ".join(cell(t[b]) for b in ("early", "late")))
    out.append(f"| {X} | " + " | ".join(cells) + " |")
tt = d["tightness"]
out.append("\n#### Label tightness (mean pairwise centred-embedding cosine of top-10)")
for b in ("early", "late"):
    x = tt[b]
    out.append(f"- {b}: J-VINDEX {x['JV']:.3f} vs 512-neuron mixtures {x['mixJV']:.3f}, diff {x['JV-mixJV']['diff']:+.3f} "
               f"[{x['JV-mixJV']['ci'][0]:+.3f}, {x['JV-mixJV']['ci'][1]:+.3f}]; naive minus mixtures {x['NV-mixNV']['diff']:+.3f} "
               f"[{x['NV-mixNV']['ci'][0]:+.3f}, {x['NV-mixNV']['ci'][1]:+.3f}]; J-VINDEX minus naive {x['JV-NV']['diff']:+.3f} "
               f"[{x['JV-NV']['ci'][0]:+.3f}, {x['JV-NV']['ci'][1]:+.3f}]")
open(f"{R}/N1V_tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
