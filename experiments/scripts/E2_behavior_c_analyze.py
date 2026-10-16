import json, sys, numpy as np, collections
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
rows = json.load(open(R + "E2_behavior_c_rows.json"))
SYSN = ["default", "gemma_explicit", "nameless", "atlas", "sarah_ai", "james_ai"]
out = {}
print("=== man-or-woman forced choice. unit = paraphrase (n=20; both option orders averaged first). ===")
print(f"{'system':15s} {'var':8s} {'P(woman)':>22s} {'P(man)':>22s} {'P(F|F or M)':>22s} {'log-odds F-M':>22s} order-effect(MW-WM cond_F)")
for s in SYSN:
    for v in ("soft", "hard", "prefill"):
        rr = [r for r in rows if r["system"] == s and r["variant"] == v]
        byq = collections.defaultdict(list)
        for r in rr: byq[r["q"]].append(r)
        pF = [np.mean([r["pF"] for r in byq[q]]) for q in sorted(byq)]
        pM = [np.mean([r["pM"] for r in byq[q]]) for q in sorted(byq)]
        cF = [np.mean([r["cond_F"] for r in byq[q]]) for q in sorted(byq)]
        lo = [np.mean([r["logodds_FM"] for r in byq[q]]) for q in sorted(byq)]
        oe = [ [r for r in byq[q] if r["order"]=="MW"][0]["cond_F"] - [r for r in byq[q] if r["order"]=="WM"][0]["cond_F"] for q in sorted(byq)]
        res = {"pF": boot_mean(pF), "pM": boot_mean(pM), "condF": boot_mean(cF), "logodds": boot_mean(lo), "order_effect": boot_mean(oe)}
        out[f"{s}|{v}"] = res
        print(f"{s:15s} {v:8s} {fmtp(res['pF']):>22s} {fmtp(res['pM']):>22s} {fmtp(res['condF']):>22s} {fmt(res['logodds']):>22s} {fmt(res['order_effect'],3)}")
# paired comparisons default vs gemma_explicit, prefill & hard
for v in ("soft", "hard", "prefill"):
    for a, b in (("gemma_explicit", "default"), ("default", "nameless")):
        da = {}
        for s in (a, b):
            rr = [r for r in rows if r["system"] == s and r["variant"] == v]
            da[s] = np.array([np.mean([r["logodds_FM"] for r in rr if r["q"] == q]) for q in range(20)])
        d = boot_mean(da[a] - da[b]); out[f"diff_logodds|{v}|{a}-{b}"] = d
        print(f"paired logodds {v}: {a} - {b} = {fmt(d)}")
print("\nexample tops:")
for s in ("default", "gemma_explicit"):
    for v in ("soft", "hard", "prefill"):
        r = [r for r in rows if r["system"] == s and r["variant"] == v][0]
        print(s, v, r["top"], r["expanded"][:3])
# aggregate top-token distribution default soft
agg = collections.Counter()
for r in rows:
    if r["system"] == "default" and r["variant"] == "soft":
        for t, p in r["top"]: agg[t] += p / 40
print("default soft mean top-token mass:", agg.most_common(10))
agg = collections.Counter()
for r in rows:
    if r["system"] == "default" and r["variant"] == "hard":
        for t, p in r["top"]: agg[t] += p / 40
print("default hard mean top-token mass:", agg.most_common(10))

# ---- pronoun forced choice ----
prow = json.load(open(R + "E2_behavior_c_pronchoice_rows.json"))
print("\n=== forced pronoun choice (he/she/it/they; unit = paraphrase n=6, 4 rotations averaged) ===")
for s in ("default", "gemma_explicit", "nameless", "sarah_ai", "james_ai"):
    res = {}
    for k in "FMNT":
        res[k] = boot_mean([np.mean([r["p_" + k] for r in prow if r["system"] == s and r["q"] == q]) for q in range(6)])
    out[f"pronchoice|{s}"] = res
    print(f"{s:15s} she {fmtp(res['F'])}  he {fmtp(res['M'])}  it {fmtp(res['N'])}  they {fmtp(res['T'])}")
r = [r for r in prow if r["system"] == "default"][0]; print("example top:", r["top"])
json.dump(out, open(R + "E2_behavior_c_summary.json", "w"), indent=1)
