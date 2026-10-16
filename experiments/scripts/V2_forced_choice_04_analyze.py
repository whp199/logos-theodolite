# V2_forced_choice 04 (local analysis, no GPU): per-entity summaries with paraphrase-bootstrap CIs, paired contrasts,
# order effects, correlations with independent ratings, regression LO ~ rating/G + W.
import json, math, collections, os, sys
import numpy as np
exec(open("./experiments/scripts/V2_forced_choice_common.py").read())
rng = np.random.default_rng(0)
NB = 4000
CLIP = 15.0
def lo(r): return float(np.clip(math.log(max(r["pF"], 1e-30)) - math.log(max(r["pM"], 1e-30)), -CLIP, CLIP))
def q(r): return r["pF"] / max(r["pF"] + r["pM"], 1e-30)
def load_fc(path, key="rows"):
    d = json.load(open(path)); return d[key] if isinstance(d, dict) else d
def table(rows, var=None):
    """{ent: {'lo': [8] order-averaged per paraphrase, 'q': [8], 'ord': [8] (LO woman-first minus man-first), 'pO': [8]}}"""
    T = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if var is not None and r.get("var", "hard") != var: continue
        T[r["ent"]][r["q"]][r["order"]] = r
    out = {}
    for e, d in T.items():
        qs = sorted(d)
        out[e] = {"lo": np.array([(lo(d[i]["MW"]) + lo(d[i]["WM"])) / 2 for i in qs]),
                  "q": np.array([(q(d[i]["MW"]) + q(d[i]["WM"])) / 2 for i in qs]),
                  "ord": np.array([lo(d[i]["WM"]) - lo(d[i]["MW"]) for i in qs]),
                  "pO": np.array([(d[i]["MW"]["pO"] + d[i]["WM"]["pO"]) / 2 for i in qs]),
                  "pBin": np.array([(d[i]["MW"]["pF"] + d[i]["MW"]["pM"] + d[i]["WM"]["pF"] + d[i]["WM"]["pM"]) / 2 for i in qs])}
    return out
def boot(x, f=np.mean):
    x = np.asarray(x); n = len(x)
    bs = [f(x[rng.integers(0, n, n)]) for _ in range(NB)]
    return float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def fmt(t, d=2): return f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"
def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def rankdata(a):
    a = np.asarray(a); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r
def spearman(a, b): return pearson(rankdata(a), rankdata(b))
def boot_corr(a, b, f=pearson):
    a, b = np.asarray(a), np.asarray(b); n = len(a); bs = []
    for _ in range(NB):
        i = rng.integers(0, n, n)
        if np.std(a[i]) > 0 and np.std(b[i]) > 0: bs.append(f(a[i], b[i]))
    return f(a, b), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y))] + X); beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta; r2 = 1 - res.var() / y.var()
    # bootstrap betas
    n = len(y); bb = []
    for _ in range(NB):
        i = rng.integers(0, n, n); b_, *_ = np.linalg.lstsq(X1[i], y[i], rcond=None); bb.append(b_)
    bb = np.array(bb)
    return {"r2": float(r2), **{nm: (float(beta[k + 1]), float(np.percentile(bb[:, k + 1], 2.5)), float(np.percentile(bb[:, k + 1], 97.5))) for k, nm in enumerate(names)}}
OUT = {}
R = os.path.join(V2_DIR, "")
fc = load_fc(R + "V2_forced_choice_01_fc_rows.json")
rat = json.load(open(R + "V2_forced_choice_02_rating_rows.json"))
TAB = {v: table(fc, v) for v in ("hard", "allow", "plain")}
ENT_META = {e[0]: e for e in V2_ENT}
order_ents = [e[0] for e in V2_ENT] + [s[0] for s in V2_SELF]
# ratings
RT = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rat:
    RT[r["ent"]]["all" if r["frame"] != "self3p" else "self3p"].append(r["lse_FM"])
    RT[r["ent"]][r["frame"]].append(r["lse_FM"])
    RT[r["ent"]]["pN_" + r["frame"]].append(r["p_N"])
lines = []
P = lines.append
P("## Per-entity forced choice (8 paraphrases, order-averaged; LO = log pF/pM clipped at +-15; CI = paraphrase bootstrap)")
P("| entity | cat | hard q | hard LO [95% CI] | order eff (W-first minus M-first) | hard P(other) | allow P(neither/other) | allow q | plain q | plain P(other) | rating lse F-M (8 tpl) | W | G |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
summ = {}
for e in order_ents:
    h, a, p = TAB["hard"][e], TAB["allow"][e], TAB["plain"][e]
    m = ENT_META.get(e)
    cat = m[1] if m else "SELF"
    rl = np.mean(RT[e]["all"]) if RT[e]["all"] else np.mean(RT[e]["self3p"])
    summ[e] = {"cat": cat, "hard_q": float(h["q"].mean()), "hard_lo": boot(h["lo"]), "ord": float(h["ord"].mean()), "hard_pO": float(h["pO"].mean()),
               "allow_pO": float(a["pO"].mean()), "allow_q": float(a["q"].mean()), "plain_q": float(p["q"].mean()), "plain_pO": float(p["pO"].mean()),
               "allow_lo": float(a["lo"].mean()), "plain_lo": float(p["lo"].mean()),
               "rating": float(rl), "rating_raw": float(np.mean(RT[e]["raw"])) if RT[e]["raw"] else None,
               "rating_chat": float(np.mean(RT[e]["chat"])) if RT[e]["chat"] else None,
               "W": m[4] if m else None, "G": m[5] if m else None, "hard_lo_q_per_para": h["lo"].round(2).tolist()}
    s = summ[e]
    P(f"| {e} | {cat} | {s['hard_q']:.3f} | {fmt(s['hard_lo'],1)} | {s['ord']:+.1f} | {s['hard_pO']:.2f} | {s['allow_pO']:.2f} | {s['allow_q']:.2f} | {s['plain_q']:.2f} | {s['plain_pO']:.2f} | {s['rating']:+.2f}{'*' if cat=='SELF' else ''} | {s['W'] if s['W'] is not None else '-'} | {s['G'] if s['G'] is not None else '-'} |")
P("(* self rating = E2-style third-person self-description frame, 'The assistant'/'This AI', 4 templates)")
OUT["summary"] = summ
# paired contrasts (paraphrase-matched, order-averaged)
def pc(a, b, var="hard"):
    x = TAB[var][a]["lo"] - TAB[var][b]["lo"]; return boot(x)
P("\n## Paired contrasts (hard; paraphrase-matched LO differences, bootstrap over 8 paraphrases)")
CON = [("ai_warm", "ai_assistant"), ("ai_cs_chatbot", "ai_assistant"), ("ai_sassy", "ai_assistant"), ("ai_helpful", "ai_assistant"),
       ("ai_bare", "ai_assistant"), ("ai_robot", "ai_assistant"), ("ai_blunt", "ai_assistant"), ("ai_code", "ai_assistant"),
       ("ai_drone", "ai_assistant"), ("ai_cold", "ai_assistant"),
       ("self_default", "ai_assistant"), ("self_helpful", "ai_helpful"), ("self_warm", "ai_warm"), ("self_blunt", "ai_blunt"),
       ("self_cold", "ai_cold"), ("self_drone", "ai_drone"),
       ("self_helpful", "self_default"), ("self_warm", "self_default"), ("self_blunt", "self_default"), ("self_cold", "self_default"), ("self_drone", "self_default"),
       ("occ_personal_asst", "occ_butler"), ("occ_secretary", "occ_butler")]
OUT["contrasts"] = {}
for a, b in CON:
    t = pc(a, b); OUT["contrasts"][f"{a}-{b}"] = t
    P(f"- {a} - {b}: {fmt(t,1)}")
# self compression: ratio of self trait shift to third-party trait shift
P("\n## Self trait-shift vs third-party trait-shift (hard, LO units)")
for tr in ("warm", "blunt", "cold", "drone"):
    ss = summ[f"self_{tr}"]["hard_lo"][0] - summ["self_default"]["hard_lo"][0]
    tp = summ[f"ai_{tr}"]["hard_lo"][0] - summ["ai_assistant"]["hard_lo"][0]
    P(f"- {tr}: self shift {ss:+.1f}, third-party shift {tp:+.1f}, ratio {ss/tp:.2f}")
# category level
P("\n## Category means (hard)")
for cat in ("AI", "OCC", "OBJ", "SELF"):
    es = [e for e in order_ents if summ[e]["cat"] == cat]
    qq = np.array([summ[e]["hard_q"] for e in es]); ll = np.array([summ[e]["hard_lo"][0] for e in es])
    P(f"- {cat}: n={len(es)}, mean q {qq.mean():.2f}, frac q>0.8 {np.mean(qq>0.8):.2f}, frac q<0.3 {np.mean(qq<0.3):.2f}, mean LO {ll.mean():+.1f}")
# correlations across 35 third-party entities
tp = [e[0] for e in V2_ENT]
y = np.array([summ[e]["hard_lo"][0] for e in tp]); yq = np.array([summ[e]["hard_q"] for e in tp])
rate = np.array([summ[e]["rating"] for e in tp]); rraw = np.array([summ[e]["rating_raw"] for e in tp]); rchat = np.array([summ[e]["rating_chat"] for e in tp])
G = np.array([summ[e]["G"] for e in tp]); W = np.array([summ[e]["W"] for e in tp])
yA = np.array([summ[e]["allow_lo"] for e in tp]); yP = np.array([summ[e]["plain_lo"] for e in tp])
P("\n## Correlations across the 35 third-party entities (hard LO unless noted; bootstrap over entities)")
CORR = {}
for nm, x in (("rating(8 tpl)", rate), ("rating raw", rraw), ("rating chat", rchat), ("apriori G", G), ("apriori W", W)):
    CORR[nm] = {"pearson": boot_corr(y, x), "spearman": boot_corr(y, x, spearman)}
    P(f"- LO vs {nm}: Pearson {fmt(CORR[nm]['pearson'])}, Spearman {fmt(CORR[nm]['spearman'])}")
CORR["hard_vs_allow"] = boot_corr(y, yA, spearman); CORR["hard_vs_plain"] = boot_corr(y, yP, spearman)
P(f"- hard LO vs allow LO (conditional within binary tail): Spearman {fmt(CORR['hard_vs_allow'])}; hard vs plain: {fmt(CORR['hard_vs_plain'])}")
CORR["G_vs_rating"] = boot_corr(G, rate)
P(f"- a-priori G vs model rating: Pearson {fmt(CORR['G_vs_rating'])}")
for cat in ("AI", "OCC", "OBJ"):
    idx = [i for i, e in enumerate(tp) if summ[e]["cat"] == cat]
    CORR[f"{cat}_rating"] = spearman(y[idx], rate[idx]); CORR[f"{cat}_G"] = spearman(y[idx], G[idx]); CORR[f"{cat}_W"] = spearman(y[idx], W[idx])
    P(f"- within {cat} (n={len(idx)}): Spearman LO~rating {CORR[f'{cat}_rating']:+.2f}, LO~G {CORR[f'{cat}_G']:+.2f}, LO~W {CORR[f'{cat}_W']:+.2f}")
OUT["corr"] = CORR
P("\n## Regressions (35 entities, hard LO; bootstrap CIs over entities)")
REG = {}
REG["G+W"] = ols(y, [G, W], ["G", "W"]); REG["rating+W"] = ols(y, [rate, W], ["rating", "W"])
REG["rating+G+W"] = ols(y, [rate, G, W], ["rating", "G", "W"])
isAI = np.array([1.0 if summ[e]["cat"] == "AI" else 0.0 for e in tp])
REG["rating+G+W+isAI"] = ols(y, [rate, G, W, isAI], ["rating", "G", "W", "isAI"])
for k, v in REG.items():
    P(f"- LO ~ {k}: R2 {v['r2']:.2f}; " + "; ".join(f"{nm} {fmt(v[nm],2)}" for nm in v if nm != "r2"))
OUT["reg"] = REG
# predictions evaluation
P("\n## Pre-registered predictions")
d = lambda a, b: summ[a]["hard_lo"][0] - summ[b]["hard_lo"][0]
p1 = {k: d(k, "ai_assistant") for k in ("ai_cold", "ai_drone", "ai_code", "ai_warm", "ai_cs_chatbot")}
P(f"- P1: LO shifts vs 'an AI assistant': " + ", ".join(f"{k} {v:+.1f}" for k, v in p1.items()))
objs = [e for e in tp if summ[e]["cat"] == "OBJ"]
p2 = {e: summ[e]["hard_q"] for e in ("obj_hammer", "obj_sun", "obj_mountain", "obj_rock", "obj_dog")}
P(f"- P2: q for male-coded objects: " + ", ".join(f"{k} {v:.2f}" for k, v in p2.items()) + f"; objects with q>0.8: {sum(summ[e]['hard_q']>0.8 for e in objs)}/{len(objs)}")
P(f"- P3: r(LO, rating) = {CORR['rating(8 tpl)']['pearson'][0]:+.2f}; r(LO, G) = {CORR['apriori G']['pearson'][0]:+.2f}; W beyond G: {fmt(REG['G+W']['W'])}")
p4 = {k: d(k, "self_default") for k in ("self_blunt", "self_cold", "self_drone")}
P(f"- P4: self shifts vs self_default: " + ", ".join(f"{k} {v:+.1f}" for k, v in p4.items()) + "; self q: " + ", ".join(f"{k} {summ[k]['hard_q']:.2f}" for k in ("self_blunt", "self_cold", "self_drone")) + "; 3p q: " + ", ".join(f"{k} {summ[k]['hard_q']:.2f}" for k in ("ai_blunt", "ai_cold", "ai_drone")))
p5a = np.mean([summ[e]["allow_pO"] for e in order_ents if summ[e]["cat"] in ("AI", "SELF")]); p5o = np.mean([summ[e]["allow_pO"] for e in tp if summ[e]["cat"] == "OCC"])
P(f"- P5: allow P(neither/other): AI+SELF mean {p5a:.2f}, occupations mean {p5o:.2f}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(V2_DIR, "V2_forced_choice_04_analysis.md"), "w").write(txt)
json.dump(OUT, open(os.path.join(V2_DIR, "V2_forced_choice_04_analysis.json"), "w"), indent=1)
