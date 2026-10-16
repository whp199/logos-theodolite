# Print key tables from V3_persona_axis_traits_02_geometry_{loc}.json
import json, sys
loc = sys.argv[1] if len(sys.argv) > 1 else "self"
R = json.load(open(f"./experiments/results/V3_persona_axis_traits_02_geometry_{loc}.json"))
B = R["bands"]
def f(x): return f"{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]"
for mn in B:
    print(f"\n================ metric {mn} ================")
    bands = list(B[mn].keys())
    d = B[mn]
    keys = list(d[bands[0]].keys())
    sel = [k for k in keys if k.startswith(("cos|", "rel|", "norm|", "dec|", "reg|", "retained|"))]
    if mn != "raw":
        sel = [k for k in sel if any(s in k for s in ("persona|trait", "persona|role", "persona|prose_cont", "trait|AIoff(default-nh", "role|AIoff(default-nh", "prose_cont|AIoff(default-nh", "persona|AIoff(default-nh", "dec|", "rel|", "trait|AI_warm", "persona_i"))]
    for k in sel:
        print(f"  {k:62s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
    for vn in ["persona", "persona_GP1only", "persona_perp_trait", "persona_perp_trait4", "persona_perp_role", "persona_perp_role2", "persona_perp_trait_role", "persona_perp_trait4_role2", "persona_perp_ctrl2", "persona_perp_ctrl4(ctrl+role)", "persona_perp_passage4(control)", "persona_perp_prose", "persona_perp_AIoff", "trait_component_of_persona", "prose_cont", "prose_last", "trait_axis(on_W/M_scale)"]:
        if mn != "raw" and vn not in ("persona", "persona_perp_trait", "persona_perp_trait4", "persona_perp_trait4_role2", "persona_perp_ctrl4(ctrl+role)", "persona_perp_passage4(control)", "prose_cont", "trait_axis(on_W/M_scale)"):
            continue
        print(f"  -- positions on {vn} (M=0, W=1)")
        for c in ["default", "nameless_AI", "Gemma", "AI_plain", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "TP_warm(pooled)", "TP_blunt(pooled)", "RL_serve(pooled)", "RL_solo(pooled)", "CT1_org", "CT1_dis"] + (["GP1_W", "GP1_M"] if vn=="persona" else []):
            k = f"pos|{vn}|{c}"
            if k in d[bands[0]]:
                print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
    print("  -- positions on trait axis (blunt=0, warm=1)")
    for c in ["default", "nameless_AI", "Gemma", "AI_plain", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "W(pooled)", "M(pooled)", "RL_serve(pooled)", "RL_solo(pooled)"]:
        k = f"tpos|trait|{c}"
        print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
print("\n================ J++ / naive readout positions (pron; M(pooled)=0, W(pooled)=1) ================")
for kind, d in R["jreadout_bands"].items():
    print(" ", kind)
    for c in d["L20-29"]:
        print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][c])}" for b in d))
print("info", R["info"])
