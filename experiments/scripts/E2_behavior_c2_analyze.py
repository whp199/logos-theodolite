import json, sys, numpy as np, collections, math
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
out = {}
def per_q(rows, key_filter):
    byq = collections.defaultdict(list)
    for r in rows:
        if key_filter(r): byq[r["q"]].append(r)
    pF = np.array([np.mean([r["pF"] for r in byq[q]]) for q in sorted(byq)])
    pM = np.array([np.mean([r["pM"] for r in byq[q]]) for q in sorted(byq)])
    cF = np.array([np.mean([r["pF"] / (r["pF"] + r["pM"]) for r in byq[q]]) for q in sorted(byq)])
    lo = np.array([np.mean([math.log(max(r["pF"], 1e-30)) - math.log(max(r["pM"], 1e-30)) for r in byq[q]]) for q in sorted(byq)])
    oe = np.array([np.mean([r["pF"] / (r["pF"] + r["pM"]) for r in byq[q] if r["order"] in ("MW", "MF")]) -
                   np.mean([r["pF"] / (r["pF"] + r["pM"]) for r in byq[q] if r["order"] in ("WM", "FM")]) for q in sorted(byq)])
    return pF, pM, cF, lo, oe
rows1 = json.load(open(R + "E2_behavior_c2_thirdparty_rows.json"))
print("=== third-party referents, hard variant, unit=paraphrase (n=10, both orders averaged) ===")
print(f"{'entity':16s} {'P(woman)':>20s} {'P(man)':>20s} {'P(F|F or M)':>20s} {'log-odds F-M':>22s} {'order eff':>8s}")
ents = []
for r in rows1:
    if r["entity"] not in ents: ents.append(r["entity"])
LO = {}
for e in ents:
    pF, pM, cF, lo, oe = per_q(rows1, lambda r: r["entity"] == e)
    LO[e] = lo
    res = {"pF": boot_mean(pF), "pM": boot_mean(pM), "condF": boot_mean(cF), "logodds": boot_mean(lo), "order_eff": boot_mean(oe)}
    out[f"thirdparty|{e}"] = res
    print(f"{e:16s} {fmtp(res['pF']):>20s} {fmtp(res['pM']):>20s} {fmtp(res['condF']):>20s} {fmt(res['logodds']):>22s} {res['order_eff'][0]:+.3f}")
print("paired log-odds(you) - log-odds(X):")
for e in ents[1:]:
    d = boot_mean(LO["you"] - LO[e]); out[f"thirdparty|diff_you-{e}"] = d
    print(f"   vs {e:16s} {fmt(d)}")
rows2 = json.load(open(R + "E2_behavior_c2_synonym_rows.json"))
print("\n=== synonym axes, self, hard variant (n=6 paraphrases; voice n=4) ===")
axes = []
for r in rows2:
    if r["axis"] not in axes: axes.append(r["axis"])
for a in axes:
    pF, pM, cF, lo, oe = per_q(rows2, lambda r: r["axis"] == a)
    res = {"pF": boot_mean(pF), "pM": boot_mean(pM), "condF": boot_mean(cF), "logodds": boot_mean(lo), "order_eff": boot_mean(oe)}
    out[f"synonym|{a}"] = res
    tops = [r["top"][:2] for r in rows2 if r["axis"] == a][:2]
    print(f"{a:22s} P(F)={fmtp(res['pF'])} P(M)={fmtp(res['pM'])} P(F|FM)={fmtp(res['condF'])} lo={fmt(res['logodds'])} oe={res['order_eff'][0]:+.3f}  e.g. {tops}")
rows3 = json.load(open(R + "E2_behavior_c2_raw_rows.json"))
print("\n=== raw-text (non-chat) interview frame (n=4 paraphrases) ===")
pF, pM, cF, lo, oe = per_q(rows3, lambda r: True)
res = {"pF": boot_mean(pF), "pM": boot_mean(pM), "condF": boot_mean(cF), "logodds": boot_mean(lo), "order_eff": boot_mean(oe)}
out["raw"] = res
print(f"P(F)={fmtp(res['pF'])} P(M)={fmtp(res['pM'])} P(F|FM)={fmtp(res['condF'])} lo={fmt(res['logodds'])} oe={res['order_eff'][0]:+.3f}")
for r in rows3: print("  ", r["q"], r["order"], r["top"], round(r["pF"], 3), round(r["pM"], 3))
json.dump(out, open(R + "E2_behavior_c2_summary.json", "w"), indent=1)
