# E3_lens_validation (a) analysis (local CPU). Unit = passage; bootstrap 2000 resamples over passages.
import json, numpy as np
R = "./experiments/results/"
rng = np.random.default_rng(0)
NB = 2000
KINDS = ["jpp", "jlens", "naive"]
BANDS = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40),
         "L40-49": range(40, 50), "L50-58": range(50, 59)}
summ = json.load(open(R + "E3_lens_validation_a_summary_prose.json")) + json.load(open(R + "E3_lens_validation_a_summary_chat.json"))
npz = {**dict(np.load(R + "E3_lens_validation_a_perpos_prose.npz")), **dict(np.load(R + "E3_lens_validation_a_perpos_chat.npz"))}

def boot_mean(M):  # M [n, ...] -> mean, lo, hi
    M = np.asarray(M, float); n = len(M)
    idx = rng.integers(0, n, (NB, n))
    bs = M[idx].mean(1)
    return M.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)

def boot_diff(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    ia = rng.integers(0, len(A), (NB, len(A))); ib = rng.integers(0, len(B), (NB, len(B)))
    bs = A[ia].mean(1) - B[ib].mean(1)
    return A.mean(0) - B.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)

def band(v):
    v = np.asarray(v)
    return {b: float(v[list(r)].mean()) for b, r in BANDS.items()}

cats = {"prose": [r for r in summ if r["src"] == "prose"],
        "asst": [r for r in summ if r["src"] == "chat"],
        "user": [r for r in summ if r["src"] == "chat"],
        "self": [r for r in summ if r["src"] == "chat" and r["n"].get("self", 0) > 0],
        "nonself_asst": [r for r in summ if r["src"] == "chat"]}
out = {"per_layer": {}, "bands": {}, "n_passages": {}, "entropy": {}}
for cat, recs in cats.items():
    out["n_passages"][cat] = len(recs)
    out["entropy"][cat] = float(np.mean([r[cat]["ent"] for r in recs]))
    for kind in KINDS:
        for m in ("kl", "top1", "top10"):
            M = np.array([r[cat][f"{m}_{kind}"] for r in recs])  # [n, 60]
            mu, lo, hi = boot_mean(M)
            out["per_layer"][f"{cat}|{m}|{kind}"] = {"mean": mu.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
            # band CIs: band-average per passage first
            bm = {}
            for b, rg in BANDS.items():
                x = M[:, list(rg)].mean(1); m_, l_, h_ = boot_mean(x)
                bm[b] = [round(float(m_), 4), round(float(l_), 4), round(float(h_), 4)]
            out["bands"][f"{cat}|{m}|{kind}"] = bm
# chat-vs-prose (unpaired) and self-vs-nonself (paired) band differences
diffs = {}
for kind in KINDS:
    for m in ("kl", "top1"):
        P = np.array([r["prose"][f"{m}_{kind}"] for r in cats["prose"]])
        A = np.array([r["asst"][f"{m}_{kind}"] for r in cats["asst"]])
        U = np.array([r["user"][f"{m}_{kind}"] for r in cats["user"]])
        S = np.array([r["self"][f"{m}_{kind}"] for r in cats["self"]])
        Ns = np.array([r["nonself_asst"][f"{m}_{kind}"] for r in cats["self"]])
        for nm, (X, Y, paired) in {"asst-prose": (A, P, False), "user-prose": (U, P, False), "self-nonself": (S, Ns, True)}.items():
            bd = {}
            for b, rg in BANDS.items():
                x = X[:, list(rg)].mean(1); y = Y[:, list(rg)].mean(1)
                if paired:
                    mu, lo, hi = boot_mean(x - y)
                else:
                    mu, lo, hi = boot_diff(x, y)
                bd[b] = [round(float(mu), 4), round(float(lo), 4), round(float(hi), 4)]
            diffs[f"{nm}|{m}|{kind}"] = bd
out["diffs"] = diffs
# entropy-matched asst-vs-prose KL: per-position pooled within passage, bins by final entropy
edges = np.array([0, 0.05, 0.25, 0.75, 1.5, 3.0, 99])
def perpos(src, cat):
    rows = []
    for r in summ:
        if r["src"] != src: continue
        key = f"{src}{r['i']:02d}"
        jj = npz[f"{key}__pos_{cat}"]
        if len(jj) == 0: continue
        rows.append({k: npz[f"{key}__{k}"][..., jj] for k in ["ent"] + [f"kl_{k}" for k in KINDS] + [f"top1_{k}" for k in KINDS]})
    return rows
PP = perpos("prose", "prose"); AA = perpos("chat", "asst"); SS = perpos("chat", "self")
def binned(rows, kind, m="kl"):
    # returns per-passage [n_bins, 60] sums and counts
    S = np.zeros((len(rows), len(edges) - 1, 60)); C = np.zeros((len(rows), len(edges) - 1))
    for i, r in enumerate(rows):
        b = np.clip(np.searchsorted(edges, r["ent"], side="right") - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            sel = b == k
            C[i, k] = sel.sum(); S[i, k] = r[f"{m}_{kind}"][:, sel].sum(1)
    return S, C
em = {}
for kind in KINDS:
    for m in ("kl", "top1"):
        Sp, Cp = binned(PP, kind, m); Sa, Ca = binned(AA, kind, m); Ss, Cs = binned(SS, kind, m)
        def stat(Sp, Cp, Sa, Ca):
            mp = Sp.sum(0) / np.maximum(Cp.sum(0), 1)[:, None]; ma = Sa.sum(0) / np.maximum(Ca.sum(0), 1)[:, None]
            w = Ca.sum(0) * (Cp.sum(0) > 0); w = w / w.sum()
            return (w[:, None] * (ma - mp)).sum(0), (w[:, None] * ma).sum(0), (w[:, None] * mp).sum(0)
        res = {}
        for nm, (Sa_, Ca_) in {"asst": (Sa, Ca), "self": (Ss, Cs)}.items():
            d0, a0, p0 = stat(Sp, Cp, Sa_, Ca_)
            bs = []
            for _ in range(500):
                ip = rng.integers(0, len(Sp), len(Sp)); ia = rng.integers(0, len(Sa_), len(Sa_))
                bs.append(stat(Sp[ip], Cp[ip], Sa_[ia], Ca_[ia])[0])
            bs = np.array(bs)
            res[nm] = {"diff_bands": {b: [round(float(d0[list(rg)].mean()), 4), round(float(np.percentile(bs[:, list(rg)].mean(1), 2.5)), 4),
                                          round(float(np.percentile(bs[:, list(rg)].mean(1), 97.5)), 4)] for b, rg in BANDS.items()},
                       "matched_target_bands": band(a0), "matched_prose_bands": band(p0)}
        em[f"{m}|{kind}"] = res
    out["entropy_bin_counts"] = {"prose": binned(PP, "jpp")[1].sum(0).tolist(), "asst": binned(AA, "jpp")[1].sum(0).tolist(), "self": binned(SS, "jpp")[1].sum(0).tolist(), "edges": edges.tolist()}
out["entropy_matched"] = em
json.dump(out, open(R + "E3_lens_validation_a_analysis.json", "w"), indent=1)
# print compact tables
print("n_passages", out["n_passages"], "\nmean final entropy (nats)", {k: round(v, 3) for k, v in out["entropy"].items()})
for m in ("kl", "top1", "top10"):
    print(f"\n== {m} (band mean [95% CI over passages]) ==")
    for kind in KINDS:
        for cat in ("prose", "asst", "self", "user"):
            b = out["bands"][f"{cat}|{m}|{kind}"]
            print(f"{kind:6s} {cat:6s} " + "  ".join(f"{bn}:{v[0]:7.3f}[{v[1]:.2f},{v[2]:.2f}]" for bn, v in b.items()))
print("\n== differences ==")
for k, bd in diffs.items():
    print(f"{k:22s} " + "  ".join(f"{bn}:{v[0]:+7.3f}[{v[1]:+.2f},{v[2]:+.2f}]" for bn, v in bd.items()))
print("\n== entropy-matched (asst/self minus prose, weighted by target entropy-bin counts) ==")
print("bin counts", out["entropy_bin_counts"])
for k, res in em.items():
    for nm, r in res.items():
        print(f"{k:12s} {nm:5s} " + "  ".join(f"{bn}:{v[0]:+7.3f}[{v[1]:+.2f},{v[2]:+.2f}]" for bn, v in r["diff_bands"].items()))

# ---- gender-readout fidelity: corr across positions between lens pron score at layer l and the model's FINAL
#      pron score at the same position (pooled positions within category; CI by passage bootstrap)
def corr_layers(rows, kind, k="pron"):
    X = np.concatenate([r[f"{k}_{kind}"] for r in rows], 1)  # [60, Npos]
    y = np.concatenate([r[f"{k}_f"] for r in rows])
    Xc = X - X.mean(1, keepdims=True); yc = y - y.mean()
    return (Xc @ yc) / (np.sqrt((Xc ** 2).sum(1)) * np.sqrt((yc ** 2).sum()) + 1e-12)
def perpos2(src, cat):
    rows = []
    for r in summ:
        if r["src"] != src: continue
        key = f"{src}{r['i']:02d}"; jj = npz[f"{key}__pos_{cat}"]
        if len(jj) == 0: continue
        rows.append({kk: npz[f"{key}__{kk}"][..., jj] for kk in ["pron_f", "noun_f"] + [f"{m}_{k}" for k in KINDS for m in ("pron", "noun")]})
    return rows
gc = {}
for cat, (src, c) in {"prose": ("prose", "prose"), "asst": ("chat", "asst")}.items():
    rows = perpos2(src, c)
    for kind in KINDS:
        for k in ("pron", "noun"):
            r0 = corr_layers(rows, kind, k)
            bs = np.array([corr_layers([rows[i] for i in rng.integers(0, len(rows), len(rows))], kind, k) for _ in range(300)])
            gc[f"{cat}|{kind}|{k}"] = {"r": r0.tolist(), "bands": {b: [round(float(r0[list(rg)].mean()), 3), round(float(np.percentile(bs[:, list(rg)].mean(1), 2.5)), 3), round(float(np.percentile(bs[:, list(rg)].mean(1), 97.5)), 3)] for b, rg in BANDS.items()}}
out["gender_corr_with_final"] = gc
json.dump(out, open(R + "E3_lens_validation_a_analysis.json", "w"), indent=1)
print("\n== corr(lens gender score at layer l, FINAL gender score at same position) ==")
for key, v in gc.items():
    print(f"{key:20s} " + "  ".join(f"{bn}:{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]" for bn, x in v["bands"].items()))

# ---- lens GAIN over the naive logit lens (KL_naive - KL_lens, per passage), prose vs assistant vs self vs user
gain = {}
print("\n== lens gain over naive (KL_naive - KL_lens, nats; band mean [95% CI]) and asst-minus-prose difference ==")
for kind in ("jpp", "jlens"):
    G = {}
    for cat in ("prose", "asst", "self", "user"):
        recs = cats[cat]
        G[cat] = np.array([np.array(r[cat]["kl_naive"]) - np.array(r[cat][f"kl_{kind}"]) for r in recs])
    for cat in G:
        gain[f"{kind}|{cat}"] = {b: [round(float(x), 3) for x in boot_mean(G[cat][:, list(rg)].mean(1))] for b, rg in BANDS.items()}
        print(f"{kind:5s} {cat:6s} " + " ".join(f"{b}:{v[0]:+6.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for b, v in gain[f'{kind}|{cat}'].items()))
    for cat in ("asst", "user"):
        dd = {b: [round(float(x), 3) for x in boot_diff(G[cat][:, list(rg)].mean(1), G["prose"][:, list(rg)].mean(1))] for b, rg in BANDS.items()}
        gain[f"{kind}|{cat}-prose"] = dd
        print(f"{kind:5s} {cat}-prose " + " ".join(f"{b}:{v[0]:+6.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for b, v in dd.items()))
out["lens_gain_over_naive"] = gain
json.dump(out, open(R + "E3_lens_validation_a_analysis.json", "w"), indent=1)
