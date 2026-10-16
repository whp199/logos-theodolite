import json, sys, numpy as np
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_b_rows.json"))
NAMES = ["Gemma", "The assistant", "This AI", "Atlas", "Sarah", "James", "Alex"]
out = {}
for sysn in ("default", "gemma_explicit"):
    for fr in ("self", "other"):
        print(f"\n=== system={sysn} frame={fr} (unit=template, n=24; bootstrap 95% CI) ===")
        print(f"{'name':14s} {'P(she-set)':>22s} {'P(he-set)':>22s} {'P(it-set)':>22s} {'P(they-set)':>22s} {'P(name)':>8s} {'lse F-M':>22s} {'pos':>6s}")
        sub = {n: [r for r in rows if r["system"] == sysn and r["frame"] == fr and r["name"] == n] for n in NAMES}
        lse = {n: np.array([r["lse_FM"] for r in sub[n]]) for n in NAMES}
        for n in NAMES:
            rr = sub[n]
            res = {k: boot_mean([r[k] for r in rr]) for k in ("p_F", "p_M", "p_N", "p_T", "p_name", "lse_FM", "share_F", "share_M", "share_N", "share_T")}
            # woman/man position: per-template paired (x - James)/(Sarah - James) on lse, computed on means
            pos = (lse[n].mean() - lse["James"].mean()) / (lse["Sarah"].mean() - lse["James"].mean())
            res["pos_lse"] = float(pos)
            out[f"{sysn}|{fr}|{n}"] = res
            print(f"{n:14s} {fmtp(res['p_F']):>22s} {fmtp(res['p_M']):>22s} {fmtp(res['p_N']):>22s} {fmtp(res['p_T']):>22s} {res['p_name'][0]:8.3f} {fmt(res['lse_FM']):>22s} {pos:6.2f}")
        # paired diffs vs Gemma
        for n in ["The assistant", "This AI", "Atlas", "Alex"]:
            d = lse["Gemma"] - lse[n]
            out[f"{sysn}|{fr}|diff_lse_Gemma-{n}"] = boot_mean(d)
            print(f"   paired lse(Gemma)-lse({n}) = {fmt(boot_mean(d))}")
# self minus other, paired per template, per name
print("\n=== self - other frame (paired by template), lse F-M and P(it) ===")
for sysn in ("default", "gemma_explicit"):
    for n in NAMES:
        a = sorted([r for r in rows if r["system"] == sysn and r["frame"] == "self" and r["name"] == n], key=lambda r: r["tmpl"])
        b = sorted([r for r in rows if r["system"] == sysn and r["frame"] == "other" and r["name"] == n], key=lambda r: r["tmpl"])
        dl = [x["lse_FM"] - y["lse_FM"] for x, y in zip(a, b)]
        dn = [x["p_N"] - y["p_N"] for x, y in zip(a, b)]
        dF = [x["p_F"] - y["p_F"] for x, y in zip(a, b)]
        out[f"{sysn}|self-other|{n}"] = {"lse": boot_mean(dl), "p_N": boot_mean(dn), "p_F": boot_mean(dF)}
        print(f"{sysn:15s} {n:14s} dlse={fmt(boot_mean(dl))}  dP(it)={fmt(boot_mean(dn),3)}  dP(she)={fmt(boot_mean(dF),3)}")
# default vs gemma_explicit (self frame)
print("\n=== gemma_explicit - default (self frame, paired) ===")
for n in NAMES:
    a = sorted([r for r in rows if r["system"] == "gemma_explicit" and r["frame"] == "self" and r["name"] == n], key=lambda r: r["tmpl"])
    b = sorted([r for r in rows if r["system"] == "default" and r["frame"] == "self" and r["name"] == n], key=lambda r: r["tmpl"])
    dl = [x["lse_FM"] - y["lse_FM"] for x, y in zip(a, b)]
    out[f"explicit-default|self|{n}"] = boot_mean(dl)
    print(f"{n:14s} dlse={fmt(boot_mean(dl))}")
# by slot type for Gemma default self
print("\n=== Gemma default self by slot ===")
for slot in ("subj", "poss", "obj"):
    rr = [r for r in rows if r["system"] == "default" and r["frame"] == "self" and r["name"] == "Gemma" and r["slot"] == slot]
    print(slot, len(rr), "P(F)", round(np.mean([r["p_F"] for r in rr]), 4), "P(M)", round(np.mean([r["p_M"] for r in rr]), 4), "P(N)", round(np.mean([r["p_N"] for r in rr]), 3), "P(T)", round(np.mean([r["p_T"] for r in rr]), 3), "lse", round(np.mean([r["lse_FM"] for r in rr]), 2))
# examples top tokens
print("\nexample top tokens (default/self):")
for n in NAMES:
    r = [r for r in rows if r["system"] == "default" and r["frame"] == "self" and r["name"] == n][0]
    print(n, r["top"])
json.dump(out, open(R + "E2_behavior_b_summary.json", "w"), indent=1)
