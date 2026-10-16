# V4_latent_identity_10_analyze_lead (CPU): "Gemma-specific lead" = logP(' Gemma') - logsumexp over rival AI product names
# (' ChatGPT',' GPT',' Claude',' Siri',' OpenAI') per position, averaged per passage/template; band L36-48. Separates
# Gemma-specific identity from generic "AI products" association. Bootstrap over passages/templates.
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(6)
RIV = [" ChatGPT", " GPT", " Claude", " Siri", " OpenAI"]
BAND = [36, 39, 42, 45, 48]
def lse(a, ax):
    m = a.max(ax, keepdims=True); return (m + np.log(np.exp(a - m).sum(ax, keepdims=True))).squeeze(ax)
def bm(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = rng.choice(x, (2000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {}
for fam in ["self", "tp"]:
    z = np.load(f"{R}/V4_latent_identity_01_{fam}.npz"); m = json.load(open(f"{R}/V4_latent_identity_01_{fam}_meta.json"))
    LP, items, grp = z["LP"], z["items"], z["grp"]; LAY = list(z["layers"]); K = list(z["kinds"]); T = list(z["toks"]); GN = m["groups"]
    bi = [LAY.index(l) for l in BAND]
    lead = LP[:, :, :, T.index(" Gemma")] - lse(LP[:, :, :, [T.index(t) for t in RIV]], -1)   # [N, L, K]
    googl = LP[:, :, :, T.index(" Google")] - lse(LP[:, :, :, [T.index(t) for t in RIV]], -1)
    meta = m["meta"]
    unit = (lambda mm: mm["q"]) if fam == "self" else (lambda mm: mm["t"])
    keyf = (lambda mm: (mm["persona"],)) if fam == "self" else (lambda mm: (mm["persona"], mm["ref"]))
    cells = {}
    for i, mm in enumerate(meta):
        cells.setdefault(keyf(mm), {})[unit(mm)] = i
    for kind in K:
        ki = K.index(kind)
        for key, units in cells.items():
            for gi, g in enumerate(GN):
                vals, vg = [], []
                for u, it in sorted(units.items()):
                    msk = (items == it) & (grp == gi)
                    vals.append(lead[msk][:, bi, ki].mean() if msk.sum() else np.nan)
                    vg.append(googl[msk][:, bi, ki].mean() if msk.sum() else np.nan)
                out[f"{fam}|{kind}|{'/'.join(key)}|{g}"] = {"gemma_lead": bm(vals), "google_lead": bm(vg)}
json.dump(out, open(f"{R}/V4_latent_identity_10_lead.json", "w"), indent=1)
for kind in ["jpp", "jlens", "naive"]:
    print(f"== {kind}: Gemma lead over rival AI names (nats, band L36-48)")
    for pn in ["none", "gemma", "llama", "claude", "chatgpt", "startup", "woman", "man"]:
        print(f"  self-fam {pn:8s}", " ".join(f"{g}:{out[f'self|{kind}|{pn}|{g}']['gemma_lead'][0]:+.1f}" for g in ["self", "neutral", "asst_you", "user_you", "user_I"]))
    for r in ["ChatGPT", "Claude", "Llama", "Gemini", "Siri", "Gemma", "Mark"]:
        o = out[f"tp|{kind}|none/{r}|tp_pron"]["gemma_lead"]; o2 = out[f"tp|{kind}|none/{r}|tp_name"]["gemma_lead"]
        print(f"  tp none it/he->{r:8s} pron {o[0]:+.1f} [{o[1]:+.1f},{o[2]:+.1f}]  name-token {o2[0]:+.1f}")
    s = out[f"self|{kind}|none|self"]["gemma_lead"]; u = out[f"self|{kind}|none|user_I"]["gemma_lead"]; n = out[f"self|{kind}|none|neutral"]["gemma_lead"]
    print(f"  none: self {s[0]:+.1f} [{s[1]:+.1f},{s[2]:+.1f}] neutral {n[0]:+.1f} [{n[1]:+.1f},{n[2]:+.1f}] user_I {u[0]:+.1f} [{u[1]:+.1f},{u[2]:+.1f}]")
