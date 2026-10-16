# Compact cross-metric table from the cross-fitted geometry JSON.
import json, sys
loc = sys.argv[1] if len(sys.argv) > 1 else "self"
R = json.load(open(f"./experiments/results/V3_persona_axis_traits_02c_crossfit_{loc}.json"))
B = R["bands"]
rows = [("rel|persona", "rel persona"), ("rel|trait", "rel trait"), ("rel|AIoff", "rel AIoff"), ("cosx|persona|trait", "cos P-T"), ("cosx|persona|prose", "cos P-prose"),
        ("cosx|persona|AIoff", "cos P-AIoff"), ("cosx|trait|AIoff", "cos T-AIoff"), ("cosx|role|AIoff", "cos R-AIoff"), ("cosx|trait|AI_warm-AI_blunt", "cos T-(AIw-AIb)"),
        ("dec||AIoff|/gap", "|AIoff|/gap"),
        ("pos|persona|default", "P: default"), ("pos|persona|neutral_human", "P: neutral_h"), ("pos|persona|neutral_human2", "P: neutral_h2"),
        ("pos|persona|AI_blunt", "P: AI_blunt"), ("pos|persona|AI_warm", "P: AI_warm"), ("pos|persona|TP_warm(pooled)", "P: TP_warm"), ("pos|persona|TP_blunt(pooled)", "P: TP_blunt"),
        ("pos|persona|RL_serve(pooled)", "P: RL_serve"), ("pos|persona|RL_solo(pooled)", "P: RL_solo"),
        ("pos|persona_perp_trait|default", "P⊥T: default"), ("pos|persona_perp_trait4|default", "P⊥T4: default"), ("pos|persona_perp_role2|default", "P⊥R2: default"),
        ("pos|persona_perp_trait4_role2|default", "P⊥T4R2: default"), ("pos|persona_perp_ctrl2|default", "P⊥C2: default"),
        ("pos|prose|default", "prose: default"), ("pos|prose|neutral_human", "prose: neutral_h"), ("pos|prose|TP_warm(pooled)", "prose: TP_warm"), ("pos|prose|TP_blunt(pooled)", "prose: TP_blunt"),
        ("pos|prose|AI_blunt", "prose: AI_blunt"), ("pos|prose|AI_warm", "prose: AI_warm"),
        ("tpos|trait|default", "T: default"), ("tpos|trait|neutral_human", "T: neutral_h"), ("tpos|trait|neutral_human2", "T: neutral_h2"), ("tpos|trait|W(pooled)", "T: W"), ("tpos|trait|M(pooled)", "T: M"),
        ("rpos|role|default", "R: default"), ("rpos|role|neutral_human", "R: neutral_h"), ("rpos|role|W(pooled)", "R: W"), ("rpos|role|M(pooled)", "R: M")]
mets = ["raw", "raw@cov", "white_l0.01", "white_l0.1", "white_l0.5", "diag_white", "drop_quiet10", "drop_top1var", "gain_next_ln"]
for bn in ("L20-29", "L30-39", "L40-50"):
    print(f"\n######## {bn}  (point estimate [95% CI]) ########")
    print(f"{'':16s}" + "".join(f"{m:>20s}" for m in mets))
    for k, lab_ in rows:
        line = f"{lab_:16s}"
        for m in mets:
            x = B[m][bn].get(k)
            line += f"{x[0]:+6.2f}[{x[1]:+.2f},{x[2]:+.2f}]".rjust(20) if x else " " * 20
        print(line)
    for kk in ("1", "2", "4"):
        line = f"{'NULL k='+kk+' mn/rng':16s}"
        for m in mets:
            x = B[m][bn][f"nullpos_k{kk}|default"]
            line += f"{x[0]:.2f} ({x[1]:.2f}-{x[2]:.2f})".rjust(20)
        print(line)
