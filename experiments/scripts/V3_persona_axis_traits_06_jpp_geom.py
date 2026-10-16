# V3_persona_axis_traits_06: axis geometry after mapping through the J++ lens (final-layer coordinates), L20-50.
# For condition means h (self tokens): hJ = h @ J[l].T. Spaces: 'J' (hJ), 'Jfnw' (hJ * final-norm weight, i.e. the
# coordinates the unembedding reads after RMSNorm, up to the per-token scale). Reports cosines among persona / trait /
# role / prose / AI-offset axes and positions of conditions on the persona axis in each space, plus positions along the
# linear J++ pronoun readout direction g = fnw*(mean W_U[she..] - mean W_U[he..]) (linearized J++ readout).
# Prediction: if J++ readout ignores the high-variance persona-style subspace, then in J space cos(AIoff, persona) ~ 0 and
#  default sits near neutral human on the J-space persona axis (as in the nonlinear J++ readout, 0.47).
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
CM = torch.load(f"{V3_CACHE}/cond_means.pt")
M = CM["means"]; LAY = CM["layers"]
GPW = [w for w, m in PAIRS["persona"]]; GPM = [m for w, m in PAIRS["persona"]]
TW = [a for a, b in PAIRS["trait"]]; TB = [b for a, b in PAIRS["trait"]]
fnw = lab.fnw.float()
gdir = lab.gender_direction("pron").float()
out = {}
for li, l in enumerate(LAY):
    H = {c: v[li].to(lab.dev).float() for c, v in M.items()}
    res_l = {}
    for space in ("raw", "J", "Jfnw", "naive_fnw"):
        if space == "raw":
            f = lambda h: h
        elif space == "J":
            f = lambda h: (h.to(torch.bfloat16) @ lab.J[l].T).float()
        elif space == "Jfnw":
            f = lambda h: (h.to(torch.bfloat16) @ lab.J[l].T).float() * fnw
        else:
            f = lambda h: h * fnw
        Z = {c: f(h) for c, h in H.items()}
        W_ = sum(Z[c] for c in GPW) / 4; M_ = sum(Z[c] for c in GPM) / 4
        Tw = sum(Z[c] for c in TW) / 4; Tb = sum(Z[c] for c in TB) / 4
        Rs = (Z["RL1_serve"] + Z["RL2_serve"]) / 2; Ro = (Z["RL1_solo"] + Z["RL2_solo"]) / 2
        nh = (Z["neutral_human"] + Z["neutral_human2"]) / 2
        ax = {"persona": W_ - M_, "trait": Tw - Tb, "role": Rs - Ro, "prose": Z["__prose_diff"], "AIoff": Z["default"] - nh,
              "AI_warm-AI_blunt": Z["AI_warm"] - Z["AI_blunt"]}
        cs = {}
        names = list(ax)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cs[f"{a}|{b}"] = float(F.cosine_similarity(ax[a], ax[b], dim=0))
        u = ax["persona"] / ax["persona"].norm()
        den = float(((W_ - M_) * u).sum())
        pos = {c: float(((Z[c] - M_) * u).sum()) / den for c in ["default", "nameless_AI", "Gemma", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2"]}
        pos["TP_warm"] = float(((Tw - M_) * u).sum()) / den; pos["TP_blunt"] = float(((Tb - M_) * u).sum()) / den
        pos["RL_serve"] = float(((Rs - M_) * u).sum()) / den; pos["RL_solo"] = float(((Ro - M_) * u).sum()) / den
        res_l[space] = {"cos": cs, "pos_persona": pos, "|AIoff|/gap": float(ax["AIoff"].norm() / ax["persona"].norm())}
    # linearized J++ pronoun readout: <hJ, g> (no RMS normalisation)
    Zg = {c: float(((H[c].to(torch.bfloat16) @ lab.J[l].T).float() * gdir).sum()) for c in H if not c.startswith("__")}
    Wg = sum(Zg[c] for c in GPW) / 4; Mg = sum(Zg[c] for c in GPM) / 4
    lin = {c: (Zg[c] - Mg) / (Wg - Mg) for c in ["default", "nameless_AI", "Gemma", "AI_warm", "AI_blunt", "neutral_human", "neutral_human2"]}
    lin["TP_warm"] = (sum(Zg[c] for c in TW) / 4 - Mg) / (Wg - Mg); lin["TP_blunt"] = (sum(Zg[c] for c in TB) / 4 - Mg) / (Wg - Mg)
    res_l["lin_jpp_pron"] = lin
    # raw-space J++ pullback covector p = g @ J[l]: cos with raw axes, and share of each axis' projection
    p = (gdir.to(torch.bfloat16) @ lab.J[l]).float()
    Hr = H
    W_ = sum(Hr[c] for c in GPW) / 4; M_ = sum(Hr[c] for c in GPM) / 4
    Tw = sum(Hr[c] for c in TW) / 4; Tb = sum(Hr[c] for c in TB) / 4
    nh = (Hr["neutral_human"] + Hr["neutral_human2"]) / 2
    res_l["pullback_cos_raw"] = {"persona": float(F.cosine_similarity(p, W_ - M_, dim=0)), "trait": float(F.cosine_similarity(p, Tw - Tb, dim=0)),
                                 "AIoff": float(F.cosine_similarity(p, Hr["default"] - nh, dim=0))}
    out[l] = res_l
BANDS3 = {"L20-29": range(20, 30), "L30-39": range(30, 40), "L40-50": range(40, 51)}
summ = {}
for bn, r in BANDS3.items():
    ls = list(r)
    s = {}
    for space in ("raw", "J", "Jfnw", "naive_fnw"):
        s[space] = {"cos": {k: round(sum(out[l][space]["cos"][k] for l in ls) / len(ls), 3) for k in out[ls[0]][space]["cos"]},
                    "pos_persona": {k: round(sum(out[l][space]["pos_persona"][k] for l in ls) / len(ls), 3) for k in out[ls[0]][space]["pos_persona"]},
                    "|AIoff|/gap": round(sum(out[l][space]["|AIoff|/gap"] for l in ls) / len(ls), 2)}
    s["lin_jpp_pron"] = {k: round(sum(out[l]["lin_jpp_pron"][k] for l in ls) / len(ls), 3) for k in out[ls[0]]["lin_jpp_pron"]}
    s["pullback_cos_raw"] = {k: round(sum(out[l]["pullback_cos_raw"][k] for l in ls) / len(ls), 3) for k in out[ls[0]]["pullback_cos_raw"]}
    summ[bn] = s
json.dump({"per_layer": out, "bands": summ}, open(f"{V3_RES}/V3_persona_axis_traits_06_jpp_geom.json", "w"))
for bn, s in summ.items():
    print("=====", bn)
    for space in ("raw", "J", "Jfnw", "naive_fnw"):
        print(" ", space, "|AIoff|/gap", s[space]["|AIoff|/gap"], "cos:", {k: v for k, v in s[space]["cos"].items() if k in ("persona|trait", "persona|prose", "persona|AIoff", "trait|AIoff", "role|AIoff", "persona|role")})
        print("     pos_persona:", s[space]["pos_persona"])
    print("  lin J++ pron:", s["lin_jpp_pron"])
    print("  pullback cos raw:", s["pullback_cos_raw"])
result = summ
