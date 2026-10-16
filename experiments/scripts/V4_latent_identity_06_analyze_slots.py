# V4_latent_identity_06_analyze_slots (CPU): per-layer competition at identity answer slots.
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(2)
z = np.load(f"{R}/V4_latent_identity_04_slots.npz"); m = json.load(open(f"{R}/V4_latent_identity_04_slots_meta.json"))
LP, RK = z["LP"], z["RK"]; LAY = list(z["layers"]); KINDS = list(z["kinds"]); TOK = list(z["toks"])
rows = m["rows"]
pers = np.array([r["persona"] for r in rows]); slot = np.array([r["slot"] for r in rows])
OWN = {"none": " Gemma", "gemma": " Gemma", "llama": " L", "claude": " Claude", "qwen": " Q", "chatgpt": " ChatGPT", "atlas": " Atlas",
       "woman": " Sarah", "man": " James", "startup": None}
MAKER = {"none": " Google", "gemma": " Google", "llama": " Meta", "claude": " Anth", "qwen": " Alibaba", "chatgpt": " OpenAI"}
RIV = [" ChatGPT", " Claude", " Gemini", " Siri", " GPT"]   # generic rival AI names


def bm(x):
    b = rng.choice(x, (2000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


out = {}
for kind in KINDS:
    ki = KINDS.index(kind)
    out[kind] = {}
    for s in ["name", "meta3p", "maker"]:
        out[kind][s] = {}
        for pn in ["none", "gemma", "llama", "claude", "qwen", "chatgpt", "atlas", "startup", "woman", "man"]:
            msk = (pers == pn) & (slot == s)
            G = LP[msk][:, :, ki, TOK.index(" Gemma")]
            res = {"P_gemma_out": float(np.exp(G[:, -1]).mean())}
            own_t = (MAKER.get(pn) if s == "maker" else OWN.get(pn))
            for l in [30, 36, 39, 42, 45, 48, 51, 55, 57, 59]:
                li = LAY.index(l)
                riv = [t for t in RIV if t != own_t]
                rivmax = LP[msk][:, li, ki, [TOK.index(t) for t in riv]].max(-1)
                d = {"gemma_lp": bm(G[:, li]), "gemma_minus_rivmax": bm(G[:, li] - rivmax),
                     "gemma_georank": float(np.exp(np.log(RK[msk][:, li, ki, TOK.index(" Gemma")]).mean()))}
                if s == "maker":
                    d["google_lp"] = bm(LP[msk][:, li, ki, TOK.index(" Google")])
                if own_t and own_t != " Gemma":
                    d["gemma_minus_own"] = bm(G[:, li] - LP[msk][:, li, ki, TOK.index(own_t)])
                    d["own_lp"] = bm(LP[msk][:, li, ki, TOK.index(own_t)])
                res[f"L{l}"] = d
            out[kind][s][pn] = res
json.dump(out, open(f"{R}/V4_latent_identity_06_slots_analysis.json", "w"), indent=1)
for kind in KINDS:
    print(f"\n===== {kind} =====")
    for s in ["name", "meta3p"]:
        print(f"-- slot {s}: Gemma minus persona-own-name (mean [CI]) per layer; and Gemma minus max rival AI name")
        for pn, res in out[kind][s].items():
            line = f"  {pn:8s} Pout(G)={res['P_gemma_out']:.3f} "
            for l in [36, 42, 45, 48, 51, 55, 57, 59]:
                d = res[f"L{l}"]
                if "gemma_minus_own" in d:
                    line += f"L{l}:{d['gemma_minus_own'][0]:+.1f}/{d['gemma_minus_rivmax'][0]:+.1f} "
                else:
                    line += f"L{l}:  -- /{d['gemma_minus_rivmax'][0]:+.1f} "
            print(line)
    s = "maker"
    print("-- slot maker: Google logP / Gemma minus own maker / Google - own maker")
    for pn, res in out[kind][s].items():
        line = f"  {pn:8s} "
        for l in [36, 42, 48, 55, 59]:
            d = res[f"L{l}"]
            line += f"L{l}: G{d['google_lp'][0]:+.1f}" + (f" own{d['own_lp'][0]:+.1f}" if "own_lp" in d else "") + " | "
        print(line)
