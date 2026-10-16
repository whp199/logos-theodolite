# Print key tables from the cross-fitted geometry JSON.
import json, sys
loc = sys.argv[1] if len(sys.argv) > 1 else "self"
full = len(sys.argv) > 2
R = json.load(open(f"./experiments/results/V3_persona_axis_traits_02c_crossfit_{loc}.json"))
B = R["bands"]
f = lambda x: f"{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]"
COS = ["cosx|persona|trait", "cosin|persona|trait", "cosx|persona_1|trait_1", "cosx|persona|role", "cosx|persona|prose", "cosx|persona|ctrl1", "cosx|persona|ctrl2",
       "cosx|persona|AIoff", "cosx|persona|AI_warm-AI_blunt", "cosx|persona|AIblunt_off", "cosx|trait|AIoff", "cosx|trait|AI_warm-AI_blunt", "cosx|trait|role", "cosx|trait|prose",
       "cosx|role|AIoff", "cosx|prose|AIoff", "cosx|ctrl1|AIoff", "cosx|ctrl2|AIoff", "cosx|AIoff|AIblunt_off",
       "rel|persona", "rel|trait", "rel|role", "rel|prose", "rel|AIoff", "rel|ctrl1", "rel|AI_warm-AI_blunt", "norm|persona", "norm|trait", "norm|AIoff",
       "dec|AIoff_on_persona/gap", "dec|via_trait/gap", "dec|via_role/gap", "dec||AIoff|/gap"]
VARS = ["persona", "persona_GP1only", "persona_perp_trait", "persona_perp_trait4", "persona_perp_role", "persona_perp_role2", "persona_perp_trait_role",
        "persona_perp_trait4_role2", "persona_perp_ctrl2", "persona_perp_prose", "persona_perp_AIoff", "trait_component_of_persona", "prose", "trait_axis", "role_axis"]
CONDS = ["default", "nameless_AI", "Gemma", "AI_plain", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "TP_warm(pooled)", "TP_blunt(pooled)", "RL_serve(pooled)", "RL_solo(pooled)", "CT1_org", "CT1_dis", "GP1_W", "GP1_M"]
for mn in B:
    bands = list(B[mn].keys()); d = B[mn]
    print(f"\n================ metric {mn} ================")
    for k in COS:
        if k in d[bands[0]]:
            print(f"  {k:36s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
    for vn in VARS:
        if not full and mn not in ("raw", "raw@cov", "white_l0.1") and vn not in ("persona", "persona_perp_trait4", "persona_perp_role2", "prose"):
            continue
        cl = CONDS if (full or mn in ("raw", "white_l0.1")) else ["default", "AI_warm", "AI_blunt", "neutral_human", "TP_warm(pooled)", "TP_blunt(pooled)", "RL_serve(pooled)", "RL_solo(pooled)"]
        if vn not in ("persona",) and not full and mn != "raw":
            cl = ["default", "AI_blunt", "neutral_human", "TP_warm(pooled)", "TP_blunt(pooled)"]
        print(f"  -- positions on {vn} (M=0, W=1)" + (f"   retained cos={d[bands[0]].get('retained|'+vn,[0])[0]:.2f}/{d[bands[-1]].get('retained|'+vn,[0])[0]:.2f}" if 'retained|'+vn in d[bands[0]] else ""))
        for c in cl:
            k = f"pos|{vn}|{c}"
            if k in d[bands[0]]:
                print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
    for kk in ("1", "2", "4"):
        k = f"nullpos_k{kk}|default"
        print(f"  NULL proj-out k={kk}: default mean/min/max/sd " + "  ".join(f"{b}:{d[b][k][0]:.2f}/{d[b][k][1]:.2f}/{d[b][k][2]:.2f}/{d[b][k][3]:.2f}" for b in bands))
    print("  -- trait-axis positions (blunt=0, warm=1)")
    for c in ["default", "nameless_AI", "Gemma", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "W(pooled)", "M(pooled)", "RL_serve(pooled)", "RL_solo(pooled)"]:
        k = f"tpos|trait|{c}"
        print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
    print("  -- role-axis positions (solo=0, serve=1)")
    for c in ["default", "nameless_AI", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2", "W(pooled)", "M(pooled)", "TP_warm(pooled)", "TP_blunt(pooled)"]:
        k = f"rpos|role|{c}"
        print(f"     {c:18s} " + "  ".join(f"{b}:{f(d[b][k])}" for b in bands))
