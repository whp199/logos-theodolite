# V4_latent_identity_02_analyze_readout (CPU): passage-level aggregation + paired bootstrap for jobs 01a (self) and 01b (tp).
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(0)
NB = 2000
PERS = ["none", "gemma", "llama", "claude", "qwen", "chatgpt", "atlas", "startup", "woman", "man"]
BAND = [36, 39, 42, 45, 48]


def load(fam):
    z = np.load(f"{R}/V4_latent_identity_01_{fam}.npz")
    m = json.load(open(f"{R}/V4_latent_identity_01_{fam}_meta.json"))
    return z, m


def boot_mean(x):  # x [n] (nan allowed) -> mean, lo, hi
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan] * 3
    bs = rng.choice(x, (NB, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"


out = {}
# ------------------------------------------------------------------ self family
z, m = load("self")
LP, RK, items, grp = z["LP"], z["RK"], z["items"], z["grp"]
LAY = list(z["layers"]); KINDS = list(z["kinds"]); TOK = list(z["toks"]); GN = m["groups"]
meta = m["meta"]
NQ = 12
bi = [LAY.index(l) for l in BAND]


def cell(persona, group, tok, kind, layers_idx, stat="lp"):
    """passage-level [12] values: mean over positions (and band layers) of log P (or log rank)."""
    ki = KINDS.index(kind); ti = TOK.index(tok); gi = GN.index(group)
    v = np.full(NQ, np.nan)
    for q in range(NQ):
        it = [i for i, mm in enumerate(meta) if mm["persona"] == persona and mm["q"] == q][0]
        msk = (items == it) & (grp == gi)
        if msk.sum() == 0:
            continue
        A = LP if stat == "lp" else np.log(RK)
        v[q] = A[msk][:, layers_idx, ki, ti].mean()
    return v


S = {}
for kind in KINDS:
    S[kind] = {}
    # table: Gemma log P and geo rank per persona x group, band L36-48
    tab = {}
    for pn in PERS:
        tab[pn] = {}
        for g in GN:
            lp = cell(pn, g, " Gemma", kind, bi); lr = cell(pn, g, " Gemma", kind, bi, "rk")
            tab[pn][g] = {"lp": boot_mean(lp), "georank": float(np.exp(np.nanmean(lr)))}
    S[kind]["gemma_table"] = tab
    # P1: persona - none at self tokens (paired over passages)
    base = cell("none", "self", " Gemma", kind, bi)
    S[kind]["P1_self_gemma_vs_none"] = {pn: boot_mean(cell(pn, "self", " Gemma", kind, bi) - base) for pn in PERS[1:]}
    # P3: group contrasts under none (and averaged over AI personas)
    c = {}
    for pn in ["none", "llama", "claude", "man"]:
        sf = cell(pn, "self", " Gemma", kind, bi)
        for g in ["neutral", "asst_you", "user_you", "user_I"]:
            c[f"{pn}:self-{g}"] = boot_mean(sf - cell(pn, g, " Gemma", kind, bi))
    S[kind]["P3_group_contrasts"] = c
    # P2: own-persona tokens at self tokens: persona minus none, band and per layer
    own = {"gemma": [" Gemma", " Google"], "llama": [" L", " Meta"], "claude": [" Claude", " Anth"],
           "qwen": [" Q", " Alibaba"], "chatgpt": [" ChatGPT", " OpenAI", " GPT"], "atlas": [" Atlas"],
           "woman": [" Sarah"], "man": [" James"]}
    p2 = {}
    for pn, tl in own.items():
        for t in tl:
            row = {"band": boot_mean(cell(pn, "self", t, kind, bi) - cell("none", "self", t, kind, bi))}
            for l in LAY:
                row[f"L{l}"] = boot_mean(cell(pn, "self", t, kind, [LAY.index(l)]) - cell("none", "self", t, kind, [LAY.index(l)]))[0]
            # same at user_you (refers to the assistant)
            row["user_you_band"] = boot_mean(cell(pn, "user_you", t, kind, bi) - cell("none", "user_you", t, kind, bi))
            p2[f"{pn}:{t}"] = row
    S[kind]["P2_own_token_rise"] = p2
    # per-layer Gemma at self vs user_I vs neutral under none
    S[kind]["per_layer_none"] = {g: {f"L{l}": float(np.nanmean(cell("none", g, " Gemma", kind, [LAY.index(l)]))) for l in LAY}
                                 for g in GN}
    S[kind]["per_layer_none_georank"] = {g: {f"L{l}": float(np.exp(np.nanmean(cell("none", g, " Gemma", kind, [LAY.index(l)], "rk")))) for l in LAY}
                                         for g in GN}
    # specificity of Gemma among AI names at self tokens (none): Gemma minus each other name token
    S[kind]["self_none_name_lp"] = {t: boot_mean(cell("none", "self", t, kind, bi)) for t in TOK}
out["self"] = S

# ------------------------------------------------------------------ third-party family
z, m = load("tp")
LP, RK, items, grp = z["LP"], z["RK"], z["items"], z["grp"]
GN = m["groups"]; meta = m["meta"]
REFS = ["ChatGPT", "Claude", "Llama", "Gemini", "Siri", "Gemma", "Mark"]
REFTOK = {"ChatGPT": " ChatGPT", "Claude": " Claude", "Llama": " L", "Gemini": " Gemini", "Siri": " Siri", "Gemma": " Gemma", "Mark": " Mark"}


def tcell(persona, ref, group, tok, kind, layers_idx, stat="lp"):
    ki = KINDS.index(kind); ti = TOK.index(tok); gi = GN.index(group)
    v = np.full(8, np.nan)
    for t in range(8):
        it = [i for i, mm in enumerate(meta) if mm["persona"] == persona and mm["ref"] == ref and mm["t"] == t][0]
        msk = (items == it) & (grp == gi)
        if msk.sum() == 0:
            continue
        A = LP if stat == "lp" else np.log(RK)
        v[t] = A[msk][:, layers_idx, ki, ti].mean()
    return v


T = {}
for kind in KINDS:
    T[kind] = {}
    # Gemma at tp_pron per referent (persona none), band
    T[kind]["gemma_at_tp_pron"] = {pn: {r: {"lp": boot_mean(tcell(pn, r, "tp_pron", " Gemma", kind, bi)),
                                            "georank": float(np.exp(np.nanmean(tcell(pn, r, "tp_pron", " Gemma", kind, bi, "rk"))))}
                                        for r in REFS} for pn in ["none", "llama", "man"]}
    T[kind]["gemma_at_tp_name"] = {r: {"lp": boot_mean(tcell("none", r, "tp_name", " Gemma", kind, bi)),
                                       "georank": float(np.exp(np.nanmean(tcell("none", r, "tp_name", " Gemma", kind, bi, "rk"))))} for r in REFS}
    T[kind]["gemma_at_neutral"] = {r: boot_mean(tcell("none", r, "neutral", " Gemma", kind, bi)) for r in REFS}
    # P4 diagonal: referent token at tp_pron for own referent minus mean over other AI referents (paired by template)
    diag = {}
    for r in REFS:
        tk = REFTOK[r]
        own_v = tcell("none", r, "tp_pron", tk, kind, bi)
        oth = np.nanmean([tcell("none", r2, "tp_pron", tk, kind, bi) for r2 in REFS if r2 != r and r2 != "Mark"], 0)
        row = {"band": boot_mean(own_v - oth), "own_lp": boot_mean(own_v),
               "own_georank": float(np.exp(np.nanmean(tcell("none", r, "tp_pron", tk, kind, bi, "rk"))))}
        for l in LAY:
            li = [LAY.index(l)]
            o2 = np.nanmean([tcell("none", r2, "tp_pron", tk, kind, li) for r2 in REFS if r2 != r and r2 != "Mark"], 0)
            row[f"L{l}"] = float(np.nanmean(tcell("none", r, "tp_pron", tk, kind, li) - o2))
        diag[r] = row
    T[kind]["P4_diag_tp_pron"] = diag
    # per-layer Gemma at tp_pron for AI refs (excl Gemma) and Mark, persona none
    T[kind]["per_layer_none"] = {r: {f"L{l}": float(np.nanmean(tcell("none", r, "tp_pron", " Gemma", kind, [LAY.index(l)]))) for l in LAY} for r in REFS}
    T[kind]["per_layer_none_georank"] = {r: {f"L{l}": float(np.exp(np.nanmean(tcell("none", r, "tp_pron", " Gemma", kind, [LAY.index(l)], "rk")))) for l in LAY} for r in REFS}
out["tp"] = T

# cross-family: self (none) vs it->other AI (none), unpaired bootstrap of difference in means
X = {}
for kind in KINDS:
    s_self = out_self = None
    zs = np.load(f"{R}/V4_latent_identity_01_self.npz"); ms = json.load(open(f"{R}/V4_latent_identity_01_self_meta.json"))
    sv = S[kind]["gemma_table"]["none"]["self"]["lp"][0]
    vals_self = []
    # reconstruct passage-level self values
    LPs, its, grs = zs["LP"], zs["items"], zs["grp"]
    gi = ms["groups"].index("self"); ki = KINDS.index(kind); ti = TOK.index(" Gemma")
    for q in range(12):
        it = [i for i, mm in enumerate(ms["meta"]) if mm["persona"] == "none" and mm["q"] == q][0]
        vals_self.append(LPs[(its == it) & (grs == gi)][:, bi, ki, ti].mean())
    vals_self = np.array(vals_self)
    tpv = np.concatenate([tcell("none", r, "tp_pron", " Gemma", kind, bi) for r in ["ChatGPT", "Claude", "Llama", "Siri"]])
    bs = [rng.choice(vals_self, 12).mean() - rng.choice(tpv, len(tpv)).mean() for _ in range(NB)]
    X[kind] = {"self_minus_itOtherAI": [float(vals_self.mean() - tpv.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    tpg = tcell("none", "Gemini", "tp_pron", " Gemma", kind, bi)
    bs = [rng.choice(vals_self, 12).mean() - rng.choice(tpg, 8).mean() for _ in range(NB)]
    X[kind]["self_minus_itGemini"] = [float(vals_self.mean() - tpg.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    tgm = tcell("none", "Gemma", "tp_pron", " Gemma", kind, bi)
    bs = [rng.choice(vals_self, 12).mean() - rng.choice(tgm, 8).mean() for _ in range(NB)]
    X[kind]["self_minus_itGemma"] = [float(vals_self.mean() - tgm.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out["cross"] = X
json.dump(out, open(f"{R}/V4_latent_identity_02_readout_analysis.json", "w"), indent=1)

# ---------------- printing
for kind in KINDS:
    print(f"\n===== {kind}  (band L36-48) =====")
    tab = S[kind]["gemma_table"]
    print("Gemma logP [CI] / georank at self-family groups:")
    for pn in PERS:
        print(f"  {pn:8s}", " | ".join(f"{g}:{tab[pn][g]['lp'][0]:+.1f}/{tab[pn][g]['georank']:.0f}" for g in out and m and GN_ if False) if False else "", end="")
        print(f"  {pn:8s}", " | ".join(f"{g}:{tab[pn][g]['lp'][0]:+.1f}/r{tab[pn][g]['georank']:.0f}" for g in tab[pn]))
    print("P1 self Gemma, persona - none:", {k: fmt(v) for k, v in S[kind]["P1_self_gemma_vs_none"].items()})
    print("P3 group contrasts:", {k: fmt(v) for k, v in S[kind]["P3_group_contrasts"].items()})
    print("P2 own-token rise at self (band | L55 | L59) and at user_you band:")
    for k, v in S[kind]["P2_own_token_rise"].items():
        print(f"   {k:18s} {fmt(v['band'])}  L55 {v['L55']:+.2f}  L59 {v['L59']:+.2f} | user_you {fmt(v['user_you_band'])}")
    print("per-layer Gemma georank (none):")
    for g, d in S[kind]["per_layer_none_georank"].items():
        print(f"   {g:9s}", " ".join(f"{k}:{v:.0f}" for k, v in d.items()))
    print("TP: Gemma at tp_pron (none):", {r: f"{v['lp'][0]:+.1f}/r{v['georank']:.0f}" for r, v in T[kind]["gemma_at_tp_pron"]["none"].items()})
    print("TP: Gemma at tp_pron (llama):", {r: f"{v['lp'][0]:+.1f}/r{v['georank']:.0f}" for r, v in T[kind]["gemma_at_tp_pron"]["llama"].items()})
    print("TP: Gemma at tp_name (none):", {r: f"{v['lp'][0]:+.1f}/r{v['georank']:.0f}" for r, v in T[kind]["gemma_at_tp_name"].items()})
    print("P4 referent diag at tp_pron (own - other AI refs):")
    for r, v in T[kind]["P4_diag_tp_pron"].items():
        print(f"   {r:8s} band {fmt(v['band'])} own lp {v['own_lp'][0]:+.1f} r{v['own_georank']:.0f} | ", " ".join(f"L{l}:{v[f'L{l}']:+.1f}" for l in LAY))
    print("cross:", {k: fmt(v) for k, v in X[kind].items()})
    print("TP per-layer Gemma georank at tp_pron (none):")
    for r, d in T[kind]["per_layer_none_georank"].items():
        print(f"   {r:8s}", " ".join(f"{k}:{v:.0f}" for k, v in d.items()))
