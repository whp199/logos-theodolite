import json, sys, numpy as np, collections, math
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_c3_rows.json"))
# heroine/hero excluded: " Hero"/"Hero" is a token prefix of "Heroine" -> double counting (mass ~1.9)
rows = [r for r in rows if r["pair"] != "heroine/hero"]
pairs = []; refs = []
for r in rows:
    if r["pair"] not in pairs: pairs.append(r["pair"])
    if r["ref"] not in refs: refs.append(r["ref"])
def lo(r): return math.log(max(r["pF"], 1e-30)) - math.log(max(r["pM"], 1e-30))
def cf(r): return r["pF"] / (r["pF"] + r["pM"]) if r["pF"] + r["pM"] > 0 else float("nan")
out = {}
M = {}  # ref -> pair -> per-paraphrase order-averaged condF
print("P(F | F or M), order-averaged, mean over 6 paraphrases [95% bootstrap CI]; (mass = P(F)+P(M)); order effect = condF(MF)-condF(FM)")
print(f"{'pair':24s}" + "".join(f"{x:>30s}" for x in refs))
for p in pairs:
    line = f"{p:24s}"
    for x in refs:
        rr = [r for r in rows if r["pair"] == p and r["ref"] == x]
        per = [np.mean([cf(r) for r in rr if r["q"] == q]) for q in range(6)]
        oe = np.mean([cf(r) for r in rr if r["order"] == "MF"]) - np.mean([cf(r) for r in rr if r["order"] == "FM"])
        mass = np.mean([r["pF"] + r["pM"] for r in rr])
        b = boot_mean(per)
        M.setdefault(x, {})[p] = b[0]
        out[f"{x}|{p}"] = {"condF": b, "order_effect": float(oe), "mass": float(mass), "logodds": boot_mean([np.mean([lo(r) for r in rr if r["q"] == q]) for q in range(6)])}
        line += f"   {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}] oe{oe:+.2f} m{mass:.2f}"
    print(line)
print()
for x in refs:
    v = np.array([M[x][p] for p in pairs])
    nF = int((v > 0.5).sum())
    out[f"{x}|n_pairs_F"] = nF
    print(f"{x:16s}: F wins on {nF}/{len(pairs)} pairs; mean P(F|FM) over pairs = {v.mean():.2f}")
for a in refs:
    for b in refs:
        if a < b:
            r = np.corrcoef([M[a][p] for p in pairs], [M[b][p] for p in pairs])[0, 1]
            out[f"corr|{a}|{b}"] = float(r)
            print(f"corr across pairs  {a:16s} vs {b:16s}: r = {r:+.2f}")
# self minus AI-assistant per pair (paired by paraphrase)
print("\nself - 'an AI assistant' (P(F|FM), paired by paraphrase&order):")
for p in pairs:
    a = [cf(r) for r in sorted([r for r in rows if r["pair"] == p and r["ref"] == "you"], key=lambda r: (r["q"], r["order"]))]
    b = [cf(r) for r in sorted([r for r in rows if r["pair"] == p and r["ref"] == "an AI assistant"], key=lambda r: (r["q"], r["order"]))]
    d = boot_mean(np.array(a) - np.array(b)); out[f"self-aiassistant|{p}"] = d
    print(f"  {p:24s} {fmt(d)}")
json.dump(out, open(R + "E2_behavior_c3_summary.json", "w"), indent=1)
