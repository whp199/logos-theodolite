#!/usr/bin/env python3
# N1A_03_metrics -- OFFLINE (CPU) agreement metrics between static output-side pole labels and causal steering.
# Run: .venv/bin/python experiments/scripts/N1A_03_metrics.py  (reads experiments/results/N1A_cache/*.pt)
# Methods m: jpp (tool J-VINDEX), jlens, naive (tool naive lens), jpp_z/jlens_z/naive_z (isotropic-prior z),
#            jpp_noscale (no post_ffn_norm weight); extras: jpp_zn/jlens_zn/naive_zn (z vs 512 random neurons' labels).
# Metrics per pole p (layer l, neuron i, sign s) and causal tag (kA = chosen k, kB = 3k):
#   rho     = Spearman(static_m(p), c_p) over the full 262144-token vocab (average ranks for ties)
#   p@10    = |top10(static_m) & top100(c_p)| / 10 ;  p@50 = |top50(static_m) & top500(c_p)| / 50
#   r@10    = |top10(c_p) & top100(static_m)| / 10   (reverse direction)
#   null    = the same metric with c_q of every OTHER sampled neuron q in the same layer, same sign (23 pairs =
#             exhaustive permutation null), averaged; lift = real - null. Layer-level permutation p-value from
#             2000 random within-layer neuron relabelings.
#   sign    = rho(static_+, c_-) (expected < 0) and rho(c_+, c_-).
# CIs: bootstrap over poles (2000 resamples), per layer and overall.
# PRE-STATED PREDICTIONS:
#  M1 naive and jpp labels both carry real signal: mean rho lift > 0 with CI excluding 0 at every layer >= 40;
#     at early layers (10, 20) all static labels are weak (rho lift < 0.1) because downstream layers transform the write.
#  M2 raw jpp top-10 precision is worse than naive's at most layers because jpp top tokens are prior-dominated
#     (HTML tags); z-scoring vs the isotropic prior (jpp_z) recovers precision, approaching or exceeding naive.
#  M3 the post_ffn_norm scaling matters little for rank metrics (jpp vs jpp_noscale rho differ by < 0.05).
#  M4 sign test holds: rho(static_+, c_-) < 0 for > 90% of poles wherever rho(static_+, c_+) lift is clearly > 0.
import json, math, os, random, sys, glob
import torch

ROOT = "./experiments/results"
CACHE = f"{ROOT}/N1A_cache"
NEU = json.load(open(f"{ROOT}/N1_neurons.json"))
LAYERS = [int(l) for l in NEU["layers"]]
NEURONS = {int(l): [int(i) for i in v] for l, v in NEU["neurons"].items()}
LAYERS = [l for l in LAYERS if os.path.exists(f"{CACHE}/causal_kA_L{l}.pt")]  # allow partial runs
TAGS = [t for t in ("kA", "kB") if os.path.exists(f"{CACHE}/causal_{t}_L{LAYERS[0]}.pt")]
METHODS = ["jpp", "jlens", "naive", "jpp_z", "jlens_z", "naive_z", "jpp_noscale", "jpp_zn", "jlens_zn", "naive_zn"]
torch.set_num_threads(max(1, os.cpu_count() - 4))

_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*"))[0]
        _tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    return _tok
def dec(i):
    return tok().decode([int(i)])


def avg_rank(X):
    """Average ranks (1..V) along dim -1 for a 2D tensor, ties averaged."""
    X = X.float()
    out = torch.empty_like(X, dtype=torch.float64)
    for r in range(X.shape[0]):
        v, idx = X[r].sort()
        _, inv, cnt = torch.unique_consecutive(v, return_inverse=True, return_counts=True)
        ends = cnt.cumsum(0).double()
        avg = ends - (cnt.double() - 1) / 2
        out[r, idx] = avg[inv]
    return out


def zrows(R):
    R = R - R.mean(-1, keepdim=True)
    return (R / R.norm(dim=-1, keepdim=True)).float()


def topset(X, k):
    return X.topk(k, -1).indices


def overlap_matrix(A_idx, B_idx, V):
    """|A_p & B_q| for all pairs: A_idx [P, ka], B_idx [Q, kb] -> [P, Q]."""
    A = torch.zeros(A_idx.shape[0], V, dtype=torch.bfloat16)
    A.scatter_(1, A_idx, 1.0)
    B = torch.zeros(B_idx.shape[0], V, dtype=torch.bfloat16)
    B.scatter_(1, B_idx, 1.0)
    return (A.float() @ B.float().T)


def boot(xs, n=2000, seed=0):
    xs = [float(x) for x in xs]
    k = len(xs)
    if k == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(xs)
    ms = t[torch.randint(0, k, (n, k), generator=g)].mean(1).sort().values
    return {"mean": float(t.mean()), "lo": float(ms[int(0.025 * n)]), "hi": float(ms[int(0.975 * n) - 1]), "n": k}


def static_methods(l, st, pri, signs_rows):
    """Return {method: [P, V] fp32} for poles given as (row in st, sign)."""
    out = {}
    for base in ("jpp", "jlens", "naive", "jpp_noscale"):
        X = st[base]
        out[base] = torch.stack([s * X[r] for r, s in signs_rows])
    for k in ("jpp", "jlens", "naive"):
        mu, sd = pri["iso"][k]
        out[f"{k}_z"] = (out[k] - mu) / sd
        out[f"{k}_zn"] = out[k] / pri["neu"][k]
    return out


def pair_metrics(S, C, V, RS, RC):
    """S [P,V] static, C [Q,V] causal (+ their standardized avg ranks) -> dict of [P,Q] matrices: rho, p10, p50, r10."""
    rho = RS @ RC.T
    p10 = overlap_matrix(topset(S, 10), topset(C, 100), V) / 10
    p50 = overlap_matrix(topset(S, 50), topset(C, 500), V) / 50
    r10 = overlap_matrix(topset(S, 100), topset(C, 10), V) / 10
    return {"rho": rho, "p10": p10, "p50": p50, "r10": r10}


def main():
    out = {"tags": TAGS, "methods": METHODS, "per_pole": [], "summary": {}, "sign_test": {}, "notes": {}}
    labels = []
    top2000 = {"poles": [], **{t: {"top_idx": [], "top_val": [], "bot_idx": [], "bot_val": []} for t in TAGS}}
    acc = {}  # (tag, method, metric, layer) -> lists of real, null
    sign_acc = {}
    causal_meta = {}
    static_agree, kxk = {}, {}
    for l in LAYERS:
        st = torch.load(f"{CACHE}/static_L{l}.pt")
        pri = torch.load(f"{CACHE}/prior_L{l}.pt")
        neurons = NEURONS[l]
        assert st["neurons"] == neurons
        poles = [(n, s) for n in neurons for s in (1, -1)]
        rows = [(neurons.index(n), s) for n, s in poles]
        SM = static_methods(l, st, pri, rows)
        V = SM["jpp"].shape[1]
        RSM = {m: zrows(avg_rank(SM[m])) for m in METHODS}
        for tag in TAGS:
            cz = torch.load(f"{CACHE}/causal_{tag}_L{l}.pt")
            meta = [m for m in cz["poles"] if m["set"] == "sample"]
            C = cz["c"][[i for i, m in enumerate(cz["poles"]) if m["set"] == "sample"]].float()
            assert [(m["n"], m["s"]) for m in meta] == poles
            causal_meta[(tag, l)] = (meta, C)
            # top2000 for integrator
            tv, ti = C.topk(2000, -1); bv, bi = (-C).topk(2000, -1)
            top2000[tag]["top_idx"].append(ti.int()); top2000[tag]["top_val"].append(tv.half())
            top2000[tag]["bot_idx"].append(bi.int()); top2000[tag]["bot_val"].append((-bv).half())
            same_sign = torch.tensor([[poles[p][1] == poles[q][1] and poles[p][0] != poles[q][0]
                                       for q in range(len(poles))] for p in range(len(poles))])
            RC = zrows(avg_rank(C))
            for m in METHODS:
                M = pair_metrics(SM[m], C, V, RSM[m], RC)
                for met, mat in M.items():
                    real = mat.diag()
                    null = (mat * same_sign).sum(1) / same_sign.sum(1)
                    acc.setdefault((tag, m, met, l), {"real": [], "null": [], "mat": mat, "mask": same_sign})
                    acc[(tag, m, met, l)]["real"] += real.tolist()
                    acc[(tag, m, met, l)]["null"] += null.tolist()
                # sign test: rho(static_+, c_-) for each neuron, and rho(c_+, c_-)
                rho = M["rho"]
                plus = [i for i, (n, s) in enumerate(poles) if s == 1]
                sign_acc.setdefault((tag, m, l), []).extend(float(rho[i, i + 1]) for i in plus)
            # causal internals
            cc = RC @ RC.T
            sign_acc.setdefault((tag, "c+_vs_c-", l), []).extend(float(cc[i, i + 1]) for i in range(0, len(poles), 2))
            # causal null: c_p vs c_q (other neurons same sign) -- how similar are causal effects across neurons?
            sign_acc.setdefault((tag, "causal_cross_neuron_rho", l), []).extend(
                float(((cc * same_sign).sum(1) / same_sign.sum(1))[i]) for i in range(len(poles)))
        # static-static agreement (per pole): jpp vs naive, jlens vs naive, jpp vs jlens
        for a, b in (("jpp", "naive"), ("jlens", "naive"), ("jpp", "jlens"), ("jpp", "jpp_noscale")):
            rr = (RSM[a] * RSM[b]).sum(-1)
            ov = [len(set(SM[a][i].topk(10).indices.tolist()) & set(SM[b][i].topk(10).indices.tolist())) / 10 for i in range(len(poles))]
            static_agree.setdefault(f"{a}_vs_{b}", {})[l] = {"rho": boot(rr.tolist()), "top10_overlap": boot(ov)}
        # k-robustness: Spearman(c_kA, c_kB) per pole
        if len(TAGS) == 2:
            RA = zrows(avg_rank(causal_meta[("kA", l)][1])); RB = zrows(avg_rank(causal_meta[("kB", l)][1]))
            kxk[l] = boot((RA * RB).sum(-1).tolist())
        # labels
        for pi, (n, s) in enumerate(poles):
            rec = {"layer": l, "neuron": n, "sign": s}
            for m in METHODS:
                v, i = SM[m][pi].topk(30)
                rec[m] = [[dec(t), round(float(x), 4)] for t, x in zip(i.tolist(), v.tolist())]
            for tag in TAGS:
                meta, C = causal_meta[(tag, l)]
                v, i = C[pi].topk(30)
                rec[f"causal_{tag}"] = [[dec(t), round(float(x), 5)] for t, x in zip(i.tolist(), v.tolist())]
                v, i = (-C[pi]).topk(15)
                rec[f"causal_{tag}_bottom"] = [[dec(t), round(-float(x), 5)] for t, x in zip(i.tolist(), v.tolist())]
                rec[f"kl_{tag}"] = meta[pi]["kl_mean"]
                rec[f"delta_{tag}"] = meta[pi]["delta"]
                rec[f"split_rho_{tag}"] = meta[pi]["split_rho"]
            labels.append(rec)
            top2000["poles"].append((l, n, s))
        print(f"L{l} done", flush=True)

    # per-pole metrics table
    pp = []
    for l in LAYERS:
        poles = [(n, s) for n in NEURONS[l] for s in (1, -1)]
        for pi, (n, s) in enumerate(poles):
            rec = {"layer": l, "neuron": n, "sign": s}
            for tag in TAGS:
                meta, _ = causal_meta[(tag, l)]
                rec[f"kl_{tag}"] = meta[pi]["kl_mean"]; rec[f"split_rho_{tag}"] = meta[pi]["split_rho"]
                for m in METHODS:
                    for met in ("rho", "p10", "p50", "r10"):
                        a = acc[(tag, m, met, l)]
                        rec[f"{tag}.{m}.{met}"] = round(a["real"][pi], 5)
                        rec[f"{tag}.{m}.{met}.null"] = round(a["null"][pi], 5)
            pp.append(rec)
    out["per_pole"] = pp

    # summaries
    g = torch.Generator().manual_seed(1)
    for tag in TAGS:
        for m in METHODS:
            for met in ("rho", "p10", "p50", "r10"):
                key = f"{tag}.{m}.{met}"
                lay = {}
                allr, alln = [], []
                for l in LAYERS:
                    a = acc[(tag, m, met, l)]
                    r, nl = a["real"], a["null"]
                    allr += r; alln += nl
                    # layer permutation test: relabel neurons within layer (keep sign), mean over poles
                    mat, mask = a["mat"], a["mask"]
                    P = mat.shape[0]; nn = P // 2
                    obs = float(mat.diag().mean())
                    perms = torch.argsort(torch.rand(2000, nn, generator=g), 1)
                    idx = torch.stack([perms * 2, perms * 2 + 1], 2).reshape(2000, -1)  # [2000, P]
                    pm = mat[torch.arange(P).unsqueeze(0).expand(2000, -1), idx].mean(1)
                    cnt = int((pm >= obs).sum())
                    lay[l] = {"real": boot(r), "null": boot(nl), "lift": boot([x - y for x, y in zip(r, nl)]),
                              "perm_p": (cnt + 1) / 2001}
                out["summary"][key] = {"per_layer": lay, "overall": {"real": boot(allr), "null": boot(alln),
                                                                     "lift": boot([x - y for x, y in zip(allr, alln)])}}
        for k, v in sign_acc.items():
            if k[0] != tag:
                continue
            _, m, l = k
            out["sign_test"].setdefault(f"{tag}.{m}", {})[l] = {"mean": boot(v), "frac_neg": sum(x < 0 for x in v) / len(v)}
    for tag in TAGS:
        kls = [r[f"kl_{tag}"] for r in pp]
        out["notes"][f"kl_{tag}"] = {"median": float(torch.tensor(kls).median()),
                                     "q10": float(torch.tensor(kls).quantile(0.1)), "q90": float(torch.tensor(kls).quantile(0.9)),
                                     "per_layer_median": {l: float(torch.tensor([r[f'kl_{tag}'] for r in pp if r['layer'] == l]).median()) for l in LAYERS}}
        sr = [r[f"split_rho_{tag}"] for r in pp]
        out["notes"][f"split_rho_{tag}"] = {"median": float(torch.tensor(sr).median()),
                                            "per_layer_median": {l: float(torch.tensor([r[f'split_rho_{tag}'] for r in pp if r['layer'] == l]).median()) for l in LAYERS}}
    # paired method differences (same poles): bootstrap CI of mean(real_a - real_b)
    pairs = [("jpp", "naive"), ("jlens", "naive"), ("jpp", "jlens"), ("jpp_z", "jpp"), ("naive_z", "naive"),
             ("jpp_zn", "jpp"), ("jpp", "jpp_noscale")]
    out["paired_diff"] = {}
    for tag in TAGS:
        for met in ("rho", "p10", "p50"):
            for a, b in pairs:
                d = {}
                alld = []
                for l in LAYERS:
                    ra = acc[(tag, a, met, l)]["real"]; rb = acc[(tag, b, met, l)]["real"]
                    dd = [x - y for x, y in zip(ra, rb)]
                    alld += dd
                    d[l] = boot(dd)
                d["overall"] = boot(alld)
                out["paired_diff"][f"{tag}.{met}.{a}-{b}"] = d
    out["static_agreement"] = static_agree
    out["causal_kA_vs_kB_rho"] = kxk
    # noise ceiling for rho: split-half reliability r (18 vs 18 contexts) -> Spearman-Brown full-c reliability 2r/(1+r);
    # a static label can at most reach ~sqrt(full reliability) correlation with the measured c.
    for tag in TAGS:
        ce = {}
        for l in LAYERS:
            r = torch.tensor([x[f"split_rho_{tag}"] for x in pp if x["layer"] == l]).clamp(min=0)
            full = 2 * r / (1 + r)
            ce[l] = {"split_rho_median": float(r.median()), "ceiling_rho_median": float(full.sqrt().median())}
        out["notes"][f"ceiling_{tag}"] = ce
    json.dump(out, open(f"{ROOT}/N1A_metrics.json", "w"), indent=1)
    json.dump(labels, open(f"{ROOT}/N1A_labels.json", "w"))
    t2k = {"poles": top2000["poles"], "vocab": 262144}
    for tag in TAGS:
        t2k[tag] = {k: torch.cat(v, 0) for k, v in top2000[tag].items()}
    torch.save(t2k, f"{ROOT}/N1A_causal_top2000.pt")
    print("saved")


if __name__ == "__main__":
    main()
