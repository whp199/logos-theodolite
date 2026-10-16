#!/usr/bin/env python3
# N1C_03_tables -- render N1C_matrix.json into markdown tables (stdout / file). CPU only.
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import ROOT, LAYERS

R = json.load(open(f"{ROOT}/N1C_matrix.json"))
out = []
def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"
def cell(b, d=3, star=False):
    if not b or b.get("n", 0) == 0:
        return "-"
    s = f"{f(b['real'],d)}/{f(b['null'],d)}"
    sig = b["diff_ci"][0] > 0
    return s + ("*" if sig else "")
def diffcell(b, d=3):
    if not b or b.get("n", 0) == 0:
        return "-"
    return f"{b['diff']:+.{d}f} [{b['diff_ci'][0]:+.{d}f},{b['diff_ci'][1]:+.{d}f}]"

def layer_row(name, res, d=3, show="cell"):
    fn = cell if show == "cell" else diffcell
    cells = [fn(res["per_layer"][str(l)], d) for l in LAYERS]
    return f"| {name} | " + " | ".join(cells) + f" | {fn(res['overall'], d)} |"

hdr = "| pair | " + " | ".join(f"L{l}" for l in LAYERS) + " | all |\n|---|" + "---|" * (len(LAYERS) + 1)

M = R["matrix_A_p10"]
MAIN = ["JV", "NV", "JVz", "C", "Enx", "Ein", "G", "Gemb", "U"]
for band in ("overall", "early_L10_30", "late_L40_57"):
    out.append(f"\n### P@10->100 on vocab A, {band}: real/null (diff>0 CI*) rows X -> cols Y\n")
    out.append("| X \\ Y | " + " | ".join(MAIN) + " |\n|---|" + "---|" * len(MAIN))
    for x in MAIN:
        cs = []
        for y in MAIN:
            cs.append("—" if x == y else cell(M[x][y][band]))
        out.append(f"| {x} | " + " | ".join(cs) + " |")

out.append("\n### FULL vocab P@10->100 static/causal, per layer real/null\n" + hdr)
for k, v in R["full_p10"].items():
    out.append(layer_row(k, v))
out.append("\n### FULL vocab lift (diff) with CI\n" + hdr)
for k in ["JV->C", "JL->C", "NV->C", "JVz->C", "NVz->C", "JVzn->C", "C->JV", "C->NV", "JV->CB"]:
    out.append(layer_row(k, R["full_p10"][k], show="diff"))

out.append("\n### Key pairs on A, per layer real/null\n" + hdr)
for x, y in [("JV", "C"), ("NV", "C"), ("JVz", "C"), ("C", "Enx"), ("Enx", "C"), ("JV", "Enx"), ("NV", "Enx"), ("C", "EnxC"),
             ("G", "Ein"), ("Gemb", "Ein"), ("U", "Ein"), ("G", "JV"), ("JV", "G"), ("G", "C"), ("G", "Enx"), ("Ein", "Enx"), ("C", "Ein")]:
    out.append(layer_row(f"{x}->{y}", M[x][y]))
out.append("\n### Key pairs on A, lift diff [CI]\n" + hdr)
for x, y in [("JV", "C"), ("NV", "C"), ("C", "Enx"), ("JV", "Enx"), ("NV", "Enx"), ("G", "Ein"), ("Gemb", "Ein"), ("U", "Ein"), ("G", "JV")]:
    out.append(layer_row(f"{x}->{y}", M[x][y], show="diff"))

out.append("\n### Spearman on W5, per layer real/null\n" + hdr)
for x, d in R["matrix_rho"].items():
    for y, v in d.items():
        out.append(layer_row(f"rho({x},{y})", v))

Q = R["q"]
out.append("\n### Q1 paired differences (FULL p@10 vs causal)\n")
for k, v in Q["Q1_paired_full"].items():
    out.append(f"- {k}: overall {v['overall']['mean']:+.3f} [{v['overall']['ci'][0]:+.3f},{v['overall']['ci'][1]:+.3f}]; late {v['late']['mean']:+.3f} [{v['late']['ci'][0]:+.3f},{v['late']['ci'][1]:+.3f}]; "
               + ", ".join(f"L{l} {v[str(l)]['mean']:+.3f}" for l in LAYERS))
for k, v in Q["Q1_paired_A"].items():
    out.append(f"- A {k}: overall {v['overall']['mean']:+.3f} [{v['overall']['ci'][0]:+.3f},{v['overall']['ci'][1]:+.3f}]; late {v['late']['mean']:+.3f}")
out.append("\n### Q2\n" + hdr)
for k in ["C->Enx01", "JV->Enx01", "NV->Enx01", "CB->Enx", "ceiling_Enx_h0->Enx_h1", "ceiling_Ein_h0->Ein_h1"]:
    out.append(layer_row(k, Q["Q2"][k]))
v = Q["Q2"]["C-JV->Enx_paired"]
out.append(f"- C - JV (->Enx) paired: overall {v['overall']['mean']:+.3f} [{v['overall']['ci'][0]:+.3f},{v['overall']['ci'][1]:+.3f}], late {v['late']['mean']:+.3f} [{v['late']['ci'][0]:+.3f},{v['late']['ci'][1]:+.3f}]")
out.append("- say fraction (>=2/10 causal top tokens in Enx top-100): " + ", ".join(
    f"L{l} {d['frac']:.2f} [{d['ci'][0]:.2f},{d['ci'][1]:.2f}] (null {d['null_frac']:.2f})" for l, d in Q["Q2_say_fraction_ge2of10"].items()))
for k in ("C", "Enx"):
    pass
bs = M["C"]["Enx"]["by_sign"]
out.append("- C->Enx by sign: " + "; ".join(f"{k} {b['real']:.3f}/{b['null']:.3f} diff {b['diff']:+.3f} [{b['diff_ci'][0]:+.3f},{b['diff_ci'][1]:+.3f}]" for k, b in bs.items()))
out.append("\n### Q3\n" + hdr)
for k, vv in Q["Q3"].items():
    out.append(layer_row(k, vv))
out.append("\n### Q4 (cross-pole)\n" + hdr)
for k, vv in Q["Q4"].items():
    if "per_layer" in vv:
        out.append(layer_row(k, vv))
for k in ("self_minus_cross_C->Enx", "self_minus_cross_rho_C_Rnx"):
    v = Q["Q4"][k]
    out.append(f"- {k}: overall {v['overall']['mean']:+.3f} [{v['overall']['ci'][0]:+.3f},{v['overall']['ci'][1]:+.3f}]; late {v['late']['mean']:+.3f} [{v['late']['ci'][0]:+.3f},{v['late']['ci'][1]:+.3f}]; early {v['early']['mean']:+.3f}; "
               + ", ".join(f"L{l} {v[str(l)]['mean']:+.3f}" for l in LAYERS))
out.append("\n### static confidence predictor\n```\n" + json.dumps(Q["static_confidence_predictor"], indent=1) + "\n```")
txt = "\n".join(out)
p = sys.argv[1] if len(sys.argv) > 1 else None
if p:
    open(p, "w").write(txt)
print(txt)
