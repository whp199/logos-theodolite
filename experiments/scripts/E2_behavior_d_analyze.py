import json, sys, numpy as np
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_d_rows.json"))
ENTS = []
for r in rows:
    if r["entity"] not in ENTS: ENTS.append(r["entity"])
out = {}
for fr in ("chat", "raw"):
    print(f"\n=== frame={fr} (unit=template n=12; bootstrap 95% CI) ===")
    print(f"{'entity':13s} {'P(she)':>6s} {'P(he)':>6s} {'P(it)':>6s} {'P(they)':>7s} {'P(name)':>7s} | {'share she':>9s} {'share he':>8s} {'share it':>8s} {'share they':>10s} | {'lse F-M':>22s} pos")
    lse = {e: np.array([r["lse_FM"] for r in sorted([r for r in rows if r["frame"] == fr and r["entity"] == e], key=lambda r: r["tmpl"])]) for e in ENTS}
    W, M_ = lse["Sarah|human"].mean(), lse["James|human"].mean()
    for e in ENTS:
        rr = [r for r in rows if r["frame"] == fr and r["entity"] == e]
        res = {k: boot_mean([r[k] for r in rr]) for k in ("p_F", "p_M", "p_N", "p_T", "p_name", "share_F", "share_M", "share_N", "share_T", "lse_FM")}
        res["pos"] = float((lse[e].mean() - M_) / (W - M_))
        out[f"{fr}|{e}"] = res
        print(f"{e:13s} {res['p_F'][0]:6.3f} {res['p_M'][0]:6.3f} {res['p_N'][0]:6.3f} {res['p_T'][0]:7.3f} {res['p_name'][0]:7.3f} | {res['share_F'][0]:9.3f} {res['share_M'][0]:8.3f} {res['share_N'][0]:8.3f} {res['share_T'][0]:10.3f} | {fmt(res['lse_FM']):>22s} {res['pos']:.2f}")
    print("  paired lse(Gemma AI) - lse(X):")
    for e in ENTS:
        if e == "Gemma": continue
        d = boot_mean(lse["Gemma"] - lse[e]); out[f"{fr}|diff_Gemma-{e}"] = d
        print(f"    vs {e:13s} {fmt(d)}")
    for k in ("share_F",):
        g = np.array([r[k] for r in sorted([r for r in rows if r["frame"] == fr and r["entity"] == "Gemma"], key=lambda r: r["tmpl"])])
        for e in ("Siri", "Alexa", "ChatGPT", "Gemini", "Gemma|human"):
            x = np.array([r[k] for r in sorted([r for r in rows if r["frame"] == fr and r["entity"] == e], key=lambda r: r["tmpl"])])
            d = boot_mean(g - x); out[f"{fr}|diff_shareF_Gemma-{e}"] = d
            print(f"  paired share_she(Gemma AI) - share_she({e}) = {fmt(d,3)}")
json.dump(out, open(R + "E2_behavior_d_summary.json", "w"), indent=1)
