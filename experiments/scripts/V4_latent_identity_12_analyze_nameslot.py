# V4_latent_identity_12_analyze_nameslot (CPU): job 07 contrasts. Gemma minus max(common given names) and Gemma minus
# max(rival AI names: ChatGPT, Claude, Gemini, Siri, Alexa) per slot type, band L36-48 and per layer; paired over 6 templates
# is impossible across slot types (different templates) -> unpaired bootstrap of slot means.
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(7)
z = np.load(f"{R}/V4_latent_identity_07_nameslot.npz"); m = json.load(open(f"{R}/V4_latent_identity_07_nameslot_meta.json"))
LP = z["LP"]; LAY = list(z["layers"]); K = list(z["kinds"]); T = list(z["toks"])
rows = m["rows"]; pers = np.array([r["persona"] for r in rows]); slot = np.array([r["slot"] for r in rows])
COMMON = [" Emma", " Anna", " Sarah", " Maria", " Luna", " Max", " John", " James", " Bella", " Charlie", " Lily", " David"]
RIV = [" ChatGPT", " Claude", " Gemini", " Siri", " Alexa"]
ci = [T.index(t) for t in COMMON]; ri = [T.index(t) for t in RIV]; G = T.index(" Gemma")
bi = [LAY.index(l) for l in [36, 39, 42, 45, 48]]
def bm(x):
    b = rng.choice(x, (2000, len(x))).mean(1); return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {}
for kind in K:
    k = K.index(kind)
    for pn in ["none", "llama", "man"]:
        for s in ["self", "human", "pet", "ai3p", "fiction"]:
            msk = (pers == pn) & (slot == s)
            A = LP[msk][:, :, k]
            gc = (A[:, :, G] - A[:, :, ci].max(-1))[:, bi].mean(1)
            gr = (A[:, :, G] - A[:, :, ri].max(-1))[:, bi].mean(1)
            gl = A[:, bi, G].mean(1)
            out[f"{kind}|{pn}|{s}"] = {"gemma_lp": bm(gl), "gemma_minus_common": bm(gc), "gemma_minus_rivalAI": bm(gr),
                                       "out_P_gemma": float(np.exp(A[:, -1, G]).mean()),
                                       "per_layer_gemma_minus_rivalAI": {f"L{l}": float((A[:, LAY.index(l), G] - A[:, LAY.index(l), ri].max(-1)).mean()) for l in LAY}}
json.dump(out, open(f"{R}/V4_latent_identity_12_nameslot_analysis.json", "w"), indent=1)
for kind in K:
    print(f"== {kind}  band L36-48:  Gemma logP | Gemma-max(common names) | Gemma-max(rival AI names) | output P(Gemma)")
    for pn in ["none", "llama", "man"]:
        for s in ["self", "ai3p", "human", "pet", "fiction"]:
            o = out[f"{kind}|{pn}|{s}"]
            print(f"  {pn:6s} {s:8s} {o['gemma_lp'][0]:+6.1f} | {o['gemma_minus_common'][0]:+5.1f} [{o['gemma_minus_common'][1]:+.1f},{o['gemma_minus_common'][2]:+.1f}] | {o['gemma_minus_rivalAI'][0]:+5.1f} [{o['gemma_minus_rivalAI'][1]:+.1f},{o['gemma_minus_rivalAI'][2]:+.1f}] | {o['out_P_gemma']:.3f}")
    for pn in ["none", "llama", "man"]:
        for s in ["self", "ai3p"]:
            print(f"   per-layer Gemma-rivalAI {pn} {s}:", " ".join(f"{k_}:{v:+.1f}" for k_, v in out[f"{kind}|{pn}|{s}"]["per_layer_gemma_minus_rivalAI"].items()))
