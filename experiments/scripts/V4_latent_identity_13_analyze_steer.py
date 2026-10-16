# V4_latent_identity_13_analyze_steer (CPU): job 08 effects with bootstrap CIs over the 8 prompts per slot.
# Identity-specific index at NAME: d(Gemma - Claude) (both non-target names in the Llama run), d(Gemma - L), d(Claude - L);
# MAKER: d(Google - Meta), d(Google - Anth). Fluency: mean KL(steered||clean) on 8 neutral prompts.
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(8)
d = json.load(open(f"{R}/V4_latent_identity_08_steer.json"))
TR = d["track"]; keys = d["keys"]; C = d["conds"]
tgt = np.array([k[0] for k in keys]); kk = np.array([k[1] for k in keys])
ix = {t: TR.index(t) for t in TR}
b = np.array(C["clean"]["lp"])
def bm(x):
    bb = rng.choice(x, (2000, len(x))).mean(1); return [float(x.mean()), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
out = {"info": {k: {kk_: [round(v, 3) for v in vv[::7]] for kk_, vv in d["info"][k].items()} if isinstance(d["info"][k], dict) else d["info"][k][::7] for k in d["info"]}}
def eff(name, t):
    lp = np.array(C[name]["lp"]); D = lp - b; kl = np.array(C[name]["kl"])
    mN = (tgt == t) & (kk == "name"); mM = (tgt == t) & (kk == "maker"); mU = (tgt == t) & (kk == "neut")
    return {"d_gemma_minus_L": bm((D[:, ix[" Gemma"]] - D[:, ix[" L"]])[mN]),
            "d_claude_minus_L": bm((D[:, ix[" Claude"]] - D[:, ix[" L"]])[mN]),
            "d_gemma_minus_claude": bm((D[:, ix[" Gemma"]] - D[:, ix[" Claude"]])[mN]),
            "d_google_minus_meta": bm((D[:, ix[" Google"]] - D[:, ix[" Meta"]])[mM]),
            "d_google_minus_anth": bm((D[:, ix[" Google"]] - D[:, ix[" Anth"]])[mM]),
            "d_anth_minus_meta": bm((D[:, ix[" Anth"]] - D[:, ix[" Meta"]])[mM]),
            "kl_neutral": bm(kl[mU]), "top1_name": [C[name]["top1"][i] for i in np.where(mN)[0]],
            "top1_maker": [C[name]["top1"][i] for i in np.where(mM)[0]]}
for dn in ["GL", "CL", "DL", "DS", "GLn", "RND"]:
    for a in [0.5, 1, 2, 4, 8]:
        out[f"{dn}|{a:+g}|llama"] = eff(f"{dn}|{a:+g}|llama", "llama")
        out[f"{dn}|{-a:+g}|none"] = eff(f"{dn}|{-a:+g}|none", "none")
json.dump(out, open(f"{R}/V4_latent_identity_13_steer_analysis.json", "w"), indent=1)
f = lambda v: f"{v[0]:+5.1f} [{v[1]:+.1f},{v[2]:+.1f}]"
print("Llama-persona target, +alpha:  d(Gemma-Claude) | d(Gemma-L) | d(Claude-L) | d(Google-Meta) | d(Anth-Meta) | KL neutral | top1 name / maker (first 3)")
for dn in ["GL", "CL", "DL", "DS", "GLn", "RND"]:
    for a in [0.5, 1, 2, 4]:
        o = out[f"{dn}|{a:+g}|llama"]
        print(f"  {dn:4s} {a:+4g}  {f(o['d_gemma_minus_claude'])} | {f(o['d_gemma_minus_L'])} | {f(o['d_claude_minus_L'])} | {f(o['d_google_minus_meta'])} | {f(o['d_anth_minus_meta'])} | {o['kl_neutral'][0]:.3f} | {o['top1_name'][:3]} {o['top1_maker'][:3]}")
print("Default target, -alpha:  d(Gemma-L) | d(Google-Meta) | KL")
for dn in ["GL", "DL", "DS"]:
    for a in [0.5, 1, 2]:
        o = out[f"{dn}|{-a:+g}|none"]
        print(f"  {dn:4s} {-a:+4g}  {f(o['d_gemma_minus_L'])} | {f(o['d_google_minus_meta'])} | {o['kl_neutral'][0]:.3f} | {o['top1_name'][:3]} {o['top1_maker'][:3]}")
