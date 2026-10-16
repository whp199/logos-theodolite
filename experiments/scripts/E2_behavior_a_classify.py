# E2_behavior (a) classifier: regex pronoun counts + self-name detection; flags ambiguous samples for manual review.
import json, re, sys, os, collections, numpy as np
sys.path.insert(0, "./experiments/scripts")
from E2_behavior_stats import *
R = "./experiments/results/"
CONDS = ["default", "gemma_explicit", "nameless", "atlas", "sarah_ai", "james_ai", "woman_human", "man_human"]
GENERIC = re.compile(r"\b(he or she|she or he|him or her|her or him|his or her|her or his|himself or herself|herself or himself|he/she|she/he|s/he|his/her|her/his|him/her|her/him|\(s\)he)\b", re.I)
PAT = {"F": re.compile(r"\b(she|her|hers|herself|she's|she'd|she'll)\b", re.I),
       "M": re.compile(r"\b(he|him|his|himself|he's|he'd|he'll)\b", re.I),
       "N": re.compile(r"\b(it|its|itself|it's)\b", re.I),
       "T": re.compile(r"\b(they|them|their|theirs|themselves|themself|they're|they've|they'll)\b", re.I)}
NAMES = ["Gemma", "ChatGPT", "Gemini", "Bard", "Claude", "Sarah", "James", "Atlas", "Google", "OpenAI", "DeepMind", "GPT"]
# manual overrides after review: key "cond|para|k" -> label (filled in after reading flagged samples)
OVR_PATH = R + "E2_behavior_a_manual_overrides.json"
OVR = json.load(open(OVR_PATH)) if os.path.exists(OVR_PATH) else {}
def norm(t):
    return t.replace("’", "'").replace("‘", "'")
def classify(text):
    t = GENERIC.sub(" ", norm(text))
    c = {k: len(p.findall(t)) for k, p in PAT.items()}
    # 'it' heuristics: discount expletive/idiomatic 'it' (e.g. "make it", "it's like" is self-ref so keep) -- keep raw, flag if only 1
    lab_ = "none"
    gendered = {k: c[k] for k in ("F", "M")}
    if c["F"] or c["M"]:
        lab_ = "F" if c["F"] >= c["M"] else "M"
        if c["F"] and c["M"]: lab_ = "mixed_FM"
    elif c["N"] >= 1:
        lab_ = "N"
    elif c["T"] >= 1:
        lab_ = "T?"
    flag = (lab_ in ("T?", "mixed_FM")) or (lab_ == "N" and (c["N"] == 1 or c["T"] > 0)) or (lab_ in ("F", "M") and (c["N"] >= 2 or c["T"] > 0))
    return c, lab_, flag
rows = []
for cond in CONDS:
    p = R + f"E2_behavior_a_gen_{cond}.json"
    if not os.path.exists(p): continue
    for r in json.load(open(p)):
        c, l, fl = classify(r["text"])
        key = f"{cond}|{r['para']}|{r['k']}"
        final = OVR.get(key, l)
        names = [n for n in NAMES if re.search(r"\b" + n + r"\b", r["text"])]
        rows.append({**r, "counts": c, "auto": l, "flag": fl, "label": final, "names": names,
                     "truncated": not norm(r["text"]).rstrip().endswith((".", "!", '"', "'", ")", "*"))})
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if mode == "flags":
        cond = sys.argv[2] if len(sys.argv) > 2 else None
        for r in rows:
            if r["flag"] and (cond is None or r["cond"] == cond) and f"{r['cond']}|{r['para']}|{r['k']}" not in OVR:
                print(f"### {r['cond']}|{r['para']}|{r['k']} auto={r['auto']} counts={r['counts']}\n{r['text']}\n")
    else:
        out = {}
        cats = ["F", "M", "N", "T", "none", "mixed_FM"]
        print(f"{'cond':15s} n   " + "  ".join(f"{c:>18s}" for c in ["she", "he", "it", "they(self)", "name/none"]) + "   names")
        for cond in CONDS:
            rr = [r for r in rows if r["cond"] == cond]
            if not rr: continue
            n = len(rr)
            lab2 = [("T" if r["label"] in ("T", "T?") else r["label"]) for r in rr]
            res = {}
            for c, nm in (("F", "she"), ("M", "he"), ("N", "it"), ("T", "they"), ("none", "none")):
                k = sum(1 for x in lab2 if x == c)
                res[nm] = wilson(k, n) + (k,)
            # paraphrase-stratified bootstrap (resample samples within each paraphrase), 4000 resamples
            rng = np.random.default_rng(99)
            byp = {pp: [l for r_, l in zip(rr, lab2) if r_["para"] == pp] for pp in sorted(set(r_["para"] for r_ in rr))}
            for c, nm in (("F", "she"), ("M", "he"), ("N", "it"), ("T", "they"), ("none", "none")):
                bs = []
                for _ in range(4000):
                    tot = 0; cnt = 0
                    for pp, ls in byp.items():
                        idx = rng.integers(0, len(ls), len(ls)); tot += len(ls); cnt += sum(1 for i in idx if ls[i] == c)
                    bs.append(cnt / tot)
                res[nm + "_strat_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            res["mixed"] = sum(1 for x in lab2 if x == "mixed_FM")
            res["any_F"] = wilson(sum(1 for r in rr if r["counts"]["F"] > 0), n)
            res["any_M"] = wilson(sum(1 for r in rr if r["counts"]["M"] > 0), n)
            nc = collections.Counter(n_ for r in rr for n_ in r["names"])
            res["names"] = dict(nc); res["n"] = n
            res["n_unresolved_T?"] = sum(1 for r in rr if r["label"] == "T?")
            out[cond] = res
            print(f"{cond:15s} {n:2d}  " + "  ".join(f"{res[k][0]:.2f} [{res[k][1]:.2f},{res[k][2]:.2f}]" for k in ("she", "he", "it", "they", "none")) + f"  mixed={res['mixed']} {dict(nc)}")
        json.dump(out, open(R + "E2_behavior_a_summary.json", "w"), indent=1)
        json.dump(rows, open(R + "E2_behavior_a_classified.json", "w"), indent=1)
