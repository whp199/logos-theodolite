import json, sys, numpy as np
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_e_identity_rows.json"))
ids = []
for r in rows:
    if r["identity"] not in ids: ids.append(r["identity"])
out = {}
for nm in ("self_name", "The assistant"):
    print(f"\n=== self frame, NAME={nm} (unit=template n=24) ===")
    print(f"{'identity':16s} {'P(she)':>20s} {'P(he)':>20s} {'P(it)':>20s} {'P(they)':>20s} {'P(name)':>7s} {'lse F-M':>22s}")
    for k in ids:
        rr = [r for r in rows if r["identity"] == k and r["name"] == nm]
        res = {x: boot_mean([r[x] for r in rr]) for x in ("p_F", "p_M", "p_N", "p_T", "p_name", "lse_FM", "share_F", "share_N")}
        out[f"{k}|{nm}"] = res
        print(f"{k:16s} {fmtp(res['p_F']):>20s} {fmtp(res['p_M']):>20s} {fmtp(res['p_N']):>20s} {fmtp(res['p_T']):>20s} {res['p_name'][0]:7.3f} {fmt(res['lse_FM']):>22s}")
json.dump(out, open(R + "E2_behavior_e_summary.json", "w"), indent=1)
g = json.load(open(R + "E2_behavior_e_identity_gens.json"))
import re, collections
for k, v in g.items():
    c = collections.Counter()
    for t in v:
        for n in ("Gemma", "Google", "DeepMind", "ChatGPT", "OpenAI", "large language model"):
            if n in t: c[n] += 1
    print(k, len(v), dict(c))
    for t in v[:5]: print("   ", repr(t[:160]))
