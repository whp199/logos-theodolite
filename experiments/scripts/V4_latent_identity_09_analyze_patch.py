# V4_latent_identity_09_analyze_patch (CPU): per-token effects of activation patching (jobs 05a/05b), bootstrap over 12 items.
import json, sys, collections, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(3)
q = sys.argv[1] if len(sys.argv) > 1 else "name"
d = json.load(open(f"{R}/V4_latent_identity_05_patch_{q}.json"))
TR = d["track"]; rows = d["rows"]
base = {r["item"]: np.array(r["lp"]) for r in rows if r["spec"] is None}
own = collections.defaultdict(list)
for r in rows:
    if r["spec"] is not None and r["spec"][0] == "SOURCE_OWN":
        own[r["spec"][1]].append(np.array(r["lp"]))
by = collections.defaultdict(list); top = collections.defaultdict(list)
for r in rows:
    if r["spec"] is None or r["spec"][0] == "SOURCE_OWN":
        continue
    by[tuple(r["spec"])].append(np.array(r["lp"]) - base[r["item"]])
    top[tuple(r["spec"])].append(r["top1"])
def bm(x):
    x = np.array(x); b = rng.choice(x, (2000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
tok_of = {"name": {"default": " Gemma", "gemma": " Gemma", "claude": " Claude", "startup": " Gemma", "man": " James", "llama": " L"},
          "maker": {"default": " Google", "gemma": " Google", "claude": " Anth", "startup": " Google", "man": " Google", "llama": " Meta"}}[q]
tgt_tok = " L" if q == "name" else " Meta"
out = {}
print(f"Q={q}: unpatched Llama-run logP:", {t: round(float(np.mean([base[i][TR.index(t)] for i in base])), 2) for t in TR})
print("source-own logP of own token:", {s: round(float(np.mean([v[TR.index(tok_of[s])] for v in vs])), 2) for s, vs in own.items()})
for spec, v in sorted(by.items()):
    s, ps, w = spec
    V = np.array(v)
    o = {"d_src_tok": bm(V[:, TR.index(tok_of[s])]), "d_tgt_tok": bm(V[:, TR.index(tgt_tok)]),
         "d_gemma": bm(V[:, TR.index(" Gemma")]), "d_google": bm(V[:, TR.index(" Google")]),
         "top1": collections.Counter(top[spec]).most_common(3)}
    out["|".join(spec)] = o
json.dump(out, open(f"{R}/V4_latent_identity_09_patch_{q}_analysis.json", "w"), indent=1)
for ps in ["self", "asst1", "turn1", "final", "allns"]:
    print(f"--- position set {ps}: Δ logP(source's own answer token) [CI] ; Δ logP(target {tgt_tok.strip()}) ; top1")
    for s in ["default", "gemma", "claude", "startup", "man", "llama"]:
        line = f"  {s:8s}"
        for w in ["L20-34", "L35-49", "L50-58"]:
            o = out[f"{s}|{ps}|{w}"]
            line += f" {w}: src{tok_of[s].strip()[:6]} {o['d_src_tok'][0]:+.2f} [{o['d_src_tok'][1]:+.2f},{o['d_src_tok'][2]:+.2f}] tgt {o['d_tgt_tok'][0]:+.2f} top {o['top1'][0][0]!r}{o['top1'][0][1]} |"
        print(line)
