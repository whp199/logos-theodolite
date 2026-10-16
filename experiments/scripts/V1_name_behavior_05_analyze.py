# V1_name_behavior 05 (local CPU analysis): classification, Wilson CIs, provenance of 'she', TF slots, swap test, audit.
# Usage: python3 V1_name_behavior_05_analyze.py [audit_dump|summary]
import json, os, re, sys, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "V1_name_behavior_common.py")).read())
R = V1_DIR + "/"
CN = [c for c, _ in V1_CONDS + V1_EXTRA]
UL = ["U1 para", "U2 intro", "U3 no-name", "U4 use-pronouns"]


def load_gen():
    rows = []
    for c in CN:
        for k in range(4):
            p = R + f"V1_name_behavior_gen_{c}_c{k}.json"
            if os.path.exists(p):
                rows += json.load(open(p))
    for un in ["A_c0", "A_c1", "B_c0", "B_c1", "C_c0", "D_c0"]:
        p = R + f"V1_name_behavior_mt_{un}.json"
        if os.path.exists(p):
            for i, r in enumerate(json.load(open(p))):
                rows.append({**r, "cond": "mt_" + un[0], "u": un, "k": i})
    for r in rows:
        r["key"] = f"{r['cond']}|{r['u']}|{r['k']}"
        r["label"] = v1_label(r["text"])
        fp = v1_first_pron(r["text"])
        r["first"] = fp[0] if fp else None
        r["counts"] = v1_counts(r["text"])
        r["names"] = [n for n in NAMES if re.search(r"\b" + n + r"\b", r["text"])]
        nm = V1_NAME.get(r["cond"]) if r["cond"] in V1_NAME else "Gemma"
        r["persona_name_in_text"] = bool(nm and re.search(r"\b" + nm + r"\b", r["text"]))
        r["gemma_in_text"] = bool(re.search(r"\bGemma\b", r["text"]))
        r["assistant_in_text"] = bool(re.search(r"\bassistant\b", r["text"], re.I))
        r["truncated"] = not v1_norm(r["text"]).rstrip().endswith((".", "!", '"', "'", ")", "*", "?"))
    return rows


AUD = R + "V1_name_behavior_audit_labels.json"


def apply_audit(rows):
    if not os.path.exists(AUD):
        return rows, None
    A = json.load(open(AUD))
    for r in rows:
        if r["key"] in A["labels"]:
            r["hand"] = A["labels"][r["key"]]
    return rows, A


def rate_table(rr, lab_key="label"):
    n = len(rr)
    c = collections.Counter(("T" if r[lab_key] == "T" else r[lab_key]) for r in rr)
    return {k: (c.get(k, 0),) + v1_wilson(c.get(k, 0), n) for k in ["F", "M", "N", "T", "none", "mixed_FM"]} | {"n": n}


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fw(t):
    return f"{t[0]:2d} ({t[1]*100:4.0f}% [{t[2]*100:3.0f},{t[3]*100:3.0f}])"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    rows = load_gen()
    rows, A = apply_audit(rows)
    if mode == "audit_dump":
        # stratified random sample: 5 per single-turn condition (60) + 12 multi-turn = 72, seeded
        # per-condition seeded RNG (so the audit can be done in waves as conditions finish); only complete conditions
        want = sys.argv[2].split(",") if len(sys.argv) > 2 else CN + ["mt_A", "mt_B", "mt_C", "mt_D"]
        pick = []
        for c in want:
            rr = sorted([r for r in rows if r["cond"] == c], key=lambda r: r["key"])
            full = len(rr) >= (32 * V1_NCALL.get(c, 1) if not c.startswith("mt") else 32)
            if not full:
                print("skip incomplete", c, len(rr)); continue
            rnd = random.Random("V1audit-" + c)
            pick += rnd.sample(rr, 5 if not c.startswith("mt") else 4)
        random.Random(7).shuffle(pick)
        with open(R + f"V1_name_behavior_audit_dump_{'_'.join(sorted(set(r['cond'] for r in pick)))[:60]}.txt", "w") as f:
            for r in pick:
                f.write(f"### {r['key']}\n{r['text']}\n\n")
        ka = R + "V1_name_behavior_audit_keys.json"
        prev = json.load(open(ka)) if os.path.exists(ka) else []
        json.dump(sorted(set(prev) | {r["key"] for r in pick}), open(ka, "w"))
        print("dumped", len(pick))
        sys.exit()
    out = {}
    # ---- per condition x prompt
    print("=== free generation: label = pronoun used for the self (regex) ===")
    hdr = f"{'cond':14s} {'cell':16s} {'n':>3s}  {'she':>18s} {'he':>18s} {'it':>18s} {'they':>18s} {'name/none':>18s} mixed  names"
    print(hdr)
    out["cells"] = {}
    for c in CN:
        for u in range(4):
            rr = [r for r in rows if r["cond"] == c and r["u"] == u]
            if not rr:
                continue
            t = rate_table(rr)
            out["cells"][f"{c}|{u}"] = t
            nc = collections.Counter(n for r in rr for n in r["names"])
            print(f"{c:14s} {UL[u]:16s} {t['n']:3d}  " + " ".join(fw(t[k]) for k in ["F", "M", "N", "T", "none"]) + f"  {t['mixed_FM'][0]}  {dict(nc.most_common(4))}")
        rr = [r for r in rows if r["cond"] == c]
        if rr:
            t = rate_table(rr)
            out["cells"][f"{c}|all"] = t
            t2 = rate_table([r for r in rr if r["u"] != 2])
            out["cells"][f"{c}|nameok"] = t2
            print(f"{c:14s} {'ALL':16s} {t['n']:3d}  " + " ".join(fw(t[k]) for k in ["F", "M", "N", "T", "none"]))
            print(f"{c:14s} {'U1+U2+U4':16s} {t2['n']:3d}  " + " ".join(fw(t2[k]) for k in ["F", "M", "N", "T", "none"]))
        print()
    for c in ["mt_A", "mt_B", "mt_C", "mt_D"]:
        rr = [r for r in rows if r["cond"] == c]
        if not rr:
            continue
        t = rate_table(rr)
        out["cells"][c] = t
        nc = collections.Counter(n for r in rr for n in r["names"])
        print(f"{c:14s} {'multi-turn':16s} {t['n']:3d}  " + " ".join(fw(t[k]) for k in ["F", "M", "N", "T", "none"]) + f"  {t['mixed_FM'][0]}  {dict(nc.most_common(5))}")
    # ---- provenance: does 'she' require the name in the generated text?
    print("\n=== provenance of 'she' ===")
    out["prov"] = {}
    for grp, conds in [("gemma5", ["gemma_bare", "gemma_name", "gemma_asst", "gemma_gdm", "gemma4_ow"]),
                       ("gemma_noversion", ["gemma_bare", "gemma_name", "gemma_asst", "gemma_gdm", "gemma_ow"]),
                       ("gemma_version4", ["gemma4_ow", "gemma4_bare", "gemma4_asst"]), ("emma", ["emma_asst"]),
                       ("mt_all", ["mt_A", "mt_B", "mt_C", "mt_D"]), ("none", ["none"])]:
        rr = [r for r in rows if r["cond"] in conds]
        if not rr:
            continue
        for key in ["persona_name_in_text", "gemma_in_text", "assistant_in_text"]:
            for val in [True, False]:
                sub = [r for r in rr if r[key] == val]
                k = sum(r["label"] == "F" for r in sub)
                w = v1_wilson(k, len(sub))
                out["prov"][f"{grp}|{key}={val}"] = (k, len(sub)) + w
                print(f"{grp:8s} {key}={str(val):5s}: she {k:3d}/{len(sub):3d} = {w[0]*100:5.1f}% [{w[1]*100:4.1f},{w[2]*100:4.1f}]")
    # 'she' by framing within gemma: assistant-framed system vs bare
    json.dump(out, open(R + "V1_name_behavior_summary.json", "w"), indent=1, default=str)
    json.dump([{k: r[k] for k in ["key", "cond", "u", "k", "label", "first", "names", "persona_name_in_text", "gemma_in_text",
                                   "assistant_in_text", "truncated", "text"] + (["hand"] if "hand" in r else []) + (["t1"] if "t1" in r else [])}
               for r in rows], open(R + "V1_name_behavior_classified.json", "w"), indent=1)


def tf_analysis(rows_gen):
    """Teacher-forced slot tables (fp and s2) per condition x prompt, plus fp-slot P(F) split by name-in-prefix."""
    out = {}
    gmap = {r["key"]: r for r in rows_gen}
    for r in rows_gen:  # TF keys for multi-turn are "mt|<unit>|<i>"
        if r["cond"].startswith("mt_"):
            gmap[f"mt|{r['u']}|{r['k']}"] = r
    print("\n=== teacher-forced pronoun probabilities (real logits) ===")
    print(f"{'cond':14s} {'cell':16s} | fp-slot: n   P(she)            P(he)   P(it)   P(they) | s2-slot: n   P(she)  P(he)   P(it)   P(they) P(name)")
    for c in CN + ["mt"]:
        p = R + f"V1_name_behavior_tf_{c}.json"
        if not os.path.exists(p):
            continue
        T = json.load(open(p))
        cells = list(range(4)) + ["all"] if c != "mt" else ["A", "B", "C", "D", "all"]
        for u in cells:
            def sel(slot):
                rr = [t for t in T if t["slot"] == slot]
                if u == "all":
                    return rr
                if c == "mt":
                    return [t for t in rr if t["key"].split("|")[1].startswith(u)]
                return [t for t in rr if int(t["key"].split("|")[1]) == u]
            fp, s2 = sel("fp"), sel("s2")
            nm = V1_NAME.get(c) or "Gemma"
            m = lambda rr, k: boot([t[k] for t in rr])
            fF, s2F = m(fp, "F"), m(s2, "F")
            res = {"fp_n": len(fp), "s2_n": len(s2)}
            for k in "FMNT":
                res["fp_" + k] = m(fp, k)
                res["s2_" + k] = m(s2, k)
            res["s2_name"] = m(s2, "x" + nm) if s2 and ("x" + nm) in s2[0] else None
            # provenance at fp slot: name in prefix?
            for val in (True, False):
                sub = [t for t in fp if (re.search(r"\b" + nm + r"\b", gmap.get(t["key"], {}).get("text", "")[:t["off"]]) is not None) == val]
                res[f"fp_F_name_in_prefix={val}"] = (len(sub),) + m(sub, "F") if sub else (0,)
            out[f"{c}|{u}"] = res
            lbl = UL[u] if isinstance(u, int) else ("multi-turn " + u if c == "mt" else "ALL")
            fmtc = lambda t: f"{t[0]:.3f}"
            print(f"{c:14s} {lbl:16s} | {len(fp):3d} {fF[0]:.3f} [{fF[1]:.3f},{fF[2]:.3f}] {fmtc(res['fp_M'])} {fmtc(res['fp_N'])} {fmtc(res['fp_T'])} |"
                  f" {len(s2):3d} {fmtc(s2F)} {fmtc(res['s2_M'])} {fmtc(res['s2_N'])} {fmtc(res['s2_T'])} "
                  + (fmtc(res['s2_name']) if res['s2_name'] else "  -  "))
    return out


def swap_analysis():
    p = R + "V1_name_behavior_swap.json"
    if not os.path.exists(p):
        return None
    S = json.load(open(p))
    by = collections.defaultdict(dict)
    for r in S:
        by[(r["group"], r["key"])][r["variant"]] = r
    out = {}
    print("\n=== counterfactual swaps at the model's own first-pronoun slot (paired over samples) ===")
    for grp, base in [("GEM", "orig"), ("OTH", "orig")]:
        keys = [k for k in by if k[0] == grp]
        variants = sorted({v for k in keys for v in by[k]})
        for v in ["orig"] + [x for x in variants if x != "orig"]:
            vals = {kk: [by[k][v][kk] for k in keys if v in by[k]] for kk in "FMNT"}
            d = [by[k][v]["F"] - by[k][base]["F"] for k in keys if v in by[k] and base in by[k]]
            res = {kk: boot(vals[kk]) for kk in "FMNT"} | {"dF_vs_orig": boot(d), "n": len(vals["F"])}
            out[f"{grp}|{v}"] = res
            print(f"{grp} {v:13s} n={res['n']:3d} P(she) {res['F'][0]:.3f} [{res['F'][1]:.3f},{res['F'][2]:.3f}]  P(he) {res['M'][0]:.3f}"
                  f"  P(it) {res['N'][0]:.3f}  P(they) {res['T'][0]:.3f}  | dP(she) vs orig {res['dF_vs_orig'][0]:+.3f} "
                  f"[{res['dF_vs_orig'][1]:+.3f},{res['dF_vs_orig'][2]:+.3f}]")
        if grp == "OTH":
            for c in ["gemini_asst", "llama_asst", "claude_asst", "atlas_asst", "emma_asst"]:
                ks = [k for k in keys if k[1].startswith(c)]
                if not ks:
                    continue
                line = f"   {c:12s} n={len(ks):2d} " + " ".join(
                    f"{v}:{np.mean([by[k][v]['F'] for k in ks]):.3f}/{np.mean([by[k][v]['M'] for k in ks]):.3f}" for v in
                    ["orig", "txt_gemma", "both_gemma", "nosys_gemma", "nosys"] if all(v in by[k] for k in ks))
                print(line + "   (P(she)/P(he))")
                out[f"OTH|{c}"] = {v: {kk: float(np.mean([by[k][v][kk] for k in ks])) for kk in "FMNT"} for v in
                                   ["orig", "txt_gemma", "both_gemma", "nosys_gemma", "nosys"] if all(v in by[k] for k in ks)}
        else:
            for c in ["gemma_bare", "gemma_name", "gemma_asst", "gemma_gdm", "gemma4_ow", "gemma4_bare", "gemma4_asst", "gemma_ow"]:
                ks = [k for k in keys if k[1].startswith(c + "|")]
                if not ks:
                    continue
                vs = ["orig", "nosys", "nameless_sys", "txt_theai", "txt_gemini", "txt_emma", "both_gemini", "nosys_gemini",
                      "txt_drop4", "txt_add4", "sys_bare", "sys_gemma4"]
                print(f"   {c:12s} n={len(ks):2d} " + " ".join(f"{v}:{np.mean([by[k][v]['F'] for k in ks]):.3f}" for v in vs if all(v in by[k] for k in ks)) + "  (P(she))")
                out[f"GEM|{c}"] = {v: {kk: float(np.mean([by[k][v][kk] for k in ks])) for kk in "FMNT"} for v in vs if all(v in by[k] for k in ks)}
    return out


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] == "summary"):
    S0 = json.load(open(R + "V1_name_behavior_summary.json"))
    S0["tf"] = tf_analysis(rows)
    S0["swap"] = swap_analysis()
    json.dump(S0, open(R + "V1_name_behavior_summary.json", "w"), indent=1, default=str)


def audit_and_extras(rows):
    out = {}
    # ---- hand audit agreement (random, stratified sample; labels in V1_name_behavior_audit_labels.json)
    A = json.load(open(AUD))
    keys = json.load(open(R + "V1_name_behavior_audit_keys.json"))
    rm = {r["key"]: r for r in rows}
    pairs = [(rm[k]["label"], A["labels"][k]) for k in keys if k in A["labels"] and k in rm]
    agree = sum(a == b for a, b in pairs)
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = agree / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    kappa = (po - pe) / (1 - pe)
    dis = [(k, rm[k]["label"], A["labels"][k]) for k in keys if k in A["labels"] and rm[k]["label"] != A["labels"][k]]
    out["audit"] = {"n": n, "agree": agree, "po": po, "kappa": kappa, "disagreements": dis,
                    "hand_label_counts": dict(collections.Counter(b for _, b in pairs))}
    print(f"\n=== hand audit: {agree}/{n} agree ({po*100:.1f}%), Cohen kappa {kappa:.3f}; disagreements: {dis}")
    # ---- first-pronoun and any-she tabulation per condition (pooled over prompts, and name-allowed prompts)
    print("\n=== label variants per condition (name-allowed prompts U1+U2+U4) ===")
    out["variants"] = {}
    for c in CN + ["mt_A", "mt_B", "mt_C", "mt_D"]:
        rr = [r for r in rows if r["cond"] == c and (c.startswith("mt") or r["u"] != 2)]
        if not rr:
            continue
        n = len(rr)
        anyF = sum(r["counts"]["F"] > 0 for r in rr)
        firstF = sum(r["first"] == "F" for r in rr)
        anyM = sum(r["counts"]["M"] > 0 for r in rr)
        mixNF = sum(r["counts"]["F"] > 0 and r["first"] == "N" for r in rr)
        d = {"n": n, "any_she": (anyF,) + v1_wilson(anyF, n), "first_she": (firstF,) + v1_wilson(firstF, n),
             "any_he": (anyM,) + v1_wilson(anyM, n), "it_then_she": mixNF,
             "first_dist": dict(collections.Counter(r["first"] for r in rr))}
        out["variants"][c] = d
        print(f"{c:14s} n={n:3d} any-she {anyF:3d} ({d['any_she'][1]*100:5.1f}% [{d['any_she'][2]*100:4.1f},{d['any_she'][3]*100:5.1f}])"
              f"  first-pronoun-she {firstF:3d} ({d['first_she'][1]*100:5.1f}%)  it->she mixtures {mixNF:2d}  any-he {anyM:3d}  first: {d['first_dist']}")
    # ---- key contrasts (Fisher exact)
    from math import comb
    def fisher(a, b, c, d):  # two-sided Fisher exact for [[a,b],[c,d]]
        n1, n2, m = a + b, c + d, a + c
        N = n1 + n2
        p0 = comb(n1, a) * comb(n2, c) / comb(N, m)
        tot = 0.0
        for x in range(max(0, m - n2), min(m, n1) + 1):
            p = comb(n1, x) * comb(n2, m - x) / comb(N, m)
            if p <= p0 * (1 + 1e-9):
                tot += p
        return tot
    def cnt(conds, us, lab="F"):
        rr = [r for r in rows if r["cond"] in conds and r["u"] in us]
        return sum(r["label"] == lab for r in rr), len(rr)
    G = ["gemma_bare", "gemma_name", "gemma_asst", "gemma_gdm", "gemma_ow"]
    G4 = ["gemma4_ow", "gemma4_bare", "gemma4_asst"]
    tests = {
        "Gemma(no version) U3 vs Emma U3 (she)": (cnt(G, [2]), cnt(["emma_asst"], [2])),
        "Gemma(no version) U1+U2+U4 vs Gemma U3 (she)": (cnt(G, [0, 1, 3]), cnt(G, [2])),
        "Gemma(no version) vs Gemma 4 versions, U1+U2+U4 (she)": (cnt(G, [0, 1, 3]), cnt(G4, [0, 1, 3])),
        "Gemma(no version) vs Gemma 4 versions, U1+U2 only (she)": (cnt(G, [0, 1]), cnt(G4, [0, 1])),
        "gemma_gdm vs gemma_asst U1+U2+U4 (she)": (cnt(["gemma_gdm"], [0, 1, 3]), cnt(["gemma_asst"], [0, 1, 3])),
        "none vs gemma_asst all prompts (she)": (cnt(["none"], [0, 1, 2, 3]), cnt(["gemma_asst"], [0, 1, 2, 3])),
    }
    out["tests"] = {}
    print("\n=== key contrasts (two-sided Fisher exact) ===")
    for k, ((a, n1), (c, n2)) in tests.items():
        p = fisher(a, n1 - a, c, n2 - c)
        out["tests"][k] = {"a": a, "n1": n1, "c": c, "n2": n2, "p": p}
        print(f"{k:55s} {a}/{n1} vs {c}/{n2}  p={p:.2e}")
    # E2 replication: E2 used U1-like and U2 (+ a 3rd paraphrase); compare gemma_asst U1+U2
    a, n1 = cnt(["gemma_asst"], [0, 1])
    out["E2_replication_gemma_asst_U1U2"] = (a, n1) + v1_wilson(a, n1)
    a, n1 = cnt(["none"], [0, 1, 2, 3], "F")
    # identity confusion in default
    rr = [r for r in rows if r["cond"] == "none"]
    out["none_identity"] = {"ChatGPT": sum("ChatGPT" in r["names"] for r in rr), "Claude": sum("Claude" in r["names"] for r in rr),
                            "Gemma": sum("Gemma" in r["names"] for r in rr), "n": len(rr),
                            "he_all_Claude": all("Claude" in r["names"] for r in rr if r["label"] == "M")}
    print("default identity names:", out["none_identity"])
    # turn-1 answers
    t1 = json.load(open(R + "V1_name_behavior_mt_B_t1.json"))
    out["mt_t1_sampled_named_gemma"] = (sum("Gemma" in r["t1"] for r in t1), len(t1)) + v1_wilson(sum("Gemma" in r["t1"] for r in t1), len(t1))
    print("turn-1 sampled answers naming Gemma:", out["mt_t1_sampled_named_gemma"])
    return out


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] == "summary"):
    S0 = json.load(open(R + "V1_name_behavior_summary.json"))
    S0["audit_extras"] = audit_and_extras(rows)
    json.dump(S0, open(R + "V1_name_behavior_summary.json", "w"), indent=1, default=str)
