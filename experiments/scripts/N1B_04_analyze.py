#!/usr/bin/env python3
"""N1B step 4 (CPU, local): activation-based (Goodfire-style dataset-example) labels for the 192 fixed neurons / 384
poles, and validation of the tool's static INPUT-side label gate_trigger_tokens = unembed(W_gate[i] @ J++[l].T).

PRE-STATED PREDICTIONS (copied from N1B_02_acts.py, written before any activation was recorded):
 P1 Strong events: g>0 and sign(act)=sign(u); >50% of neurons bipolar at the 30% criterion, ~30% at 50%.
 P2 Empirical input-token labels concentrated at L10-20, diffuse at L45-57.
 P3 G_jpp precision@10 vs empirical top-50 input tokens ~ permutation null (lift CI includes 1) at most layers.
 P4 G_naive (embedding match) beats G_jpp at L10-20, ~null late.
 P5 (static job) G_jpp top-10 overlaps strongly with the random-isotropic J++ prior and is shared across neurons.

Definitions
 - pole s in {+1,-1}; s*act > 0 means the pole is active (pole + = mlp_act > 0).
 - top positions of a pole: top 1% of all non-BOS positions ranked by s*act, restricted to s*act > 0.
 - input label: counts of CURRENT tokens at top positions; output-empirical label: counts of NEXT tokens (within chunk).
 - E_count50 = top-50 tokens by count (task definition); E_enrich50 = top-50 by log-enrichment vs corpus frequency
   (count >= 3), a frequency-controlled set closer to how Goodfire-style labels are read.
 - Null: every metric is computed for all (pole of neuron i, label of neuron j) pairs in the same layer; the null is the
   mean over j != i (= exact expectation over random within-layer re-pairings, i.e. infinitely many permutations);
   a permutation p-value uses 2000 random derangements. lift = mean(metric) / mean(null). 95% bootstrap CIs over poles.
Usage: .venv/bin/python N1B_04_analyze.py
"""
import glob, json, math, os, sys
from collections import Counter

import numpy as np

R = os.environ.get("N1B_R", "./experiments/results/")
sys.path.insert(0, ".")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
rng = np.random.default_rng(1234)
NB = int(os.environ.get("N1B_NB", 2000))

neu = json.load(open(R + "N1_neurons.json"))
LAYERS = neu["layers"]
COLS = [(l, int(i)) for l in LAYERS for i in neu["neurons"][str(l)]]
NPL = 24

# ------------------------------------------------------------------ corpus + activations
corpus = [json.loads(x) for x in open(R + "N1B_corpus.jsonl")]
parts = sorted(glob.glob(R + "N1B_acts/part_*.npz"))
D = [np.load(p) for p in parts]
act = np.concatenate([d["act"] for d in D])
g = np.concatenate([d["g"] for d in D]).astype(np.float32)
u = np.concatenate([d["u"] for d in D]).astype(np.float32)
cid = np.concatenate([d["cid"] for d in D])
pos = np.concatenate([d["pos"] for d in D]).astype(np.int64)
assert len(set(cid.tolist())) == len(corpus), (len(set(cid.tolist())), len(corpus))
cur_tok = np.array([corpus[c]["ids"][p] for c, p in zip(cid, pos)], dtype=np.int64)
nxt_tok = np.array([corpus[c]["ids"][p + 1] if p + 1 < len(corpus[c]["ids"]) else -1 for c, p in zip(cid, pos)], dtype=np.int64)
src = np.array([corpus[c]["src"] for c in cid])
import zlib
half = np.array([zlib.crc32(corpus[c]["doc"].encode()) % 2 for c in cid], dtype=np.int8)   # split-half by DOCUMENT
keep = pos > 0                                      # drop BOS (attention-sink position, constant token)
act, g, u, cid, pos, cur_tok, nxt_tok, src, half = (x[keep] for x in (act, g, u, cid, pos, cur_tok, nxt_tok, src, half))
N = len(act)
print("positions (non-BOS):", N, "chunks:", len(corpus))
src_counts = Counter(src.tolist())
tot_cnt = Counter(cur_tok.tolist())
tot_next = Counter(nxt_tok[nxt_tok >= 0].tolist())

_sc = {}
def ts(t):
    if t not in _sc:
        _sc[t] = tok.decode([int(t)]) if t >= 0 else "<END>"
    return _sc[t]

def norm(s):
    k = s.strip().lower()
    return k if k else repr(s)

def context(c, p, w=12):
    ids = corpus[c]["ids"]
    left = "".join(ts(t) for t in ids[max(1, p - w + 1):p])
    return left + "[[" + ts(ids[p]) + "]]"

def enrich_rank(cnt_top, n_top, base_cnt, base_n, min_c=3, k=50):
    sc = []
    for t, c in cnt_top.items():
        if c < min_c:
            continue
        e = math.log((c + 0.5) / n_top) - math.log((base_cnt.get(t, 0) + 0.5) / base_n)
        sc.append((e, t, c))
    sc.sort(reverse=True)
    return sc[:k]

def rank01(x):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    r -= r.mean()
    return r / (np.linalg.norm(r) + 1e-12)

# ------------------------------------------------------------------ per-neuron / per-pole empirical labels
ntop = int(math.ceil(0.01 * N))
ntop01 = int(math.ceil(0.001 * N))
HM = {h: np.where(half == h)[0] for h in (0, 1)}
cnt_half = {h: Counter(cur_tok[HM[h]].tolist()) for h in (0, 1)}
# tokens with >= 5 occurrences in BOTH halves (for split-half token-response reliability)
_t5 = sorted(t for t in tot_cnt if cnt_half[0].get(t, 0) >= 5 and cnt_half[1].get(t, 0) >= 5)
t5_idx = {t: k for k, t in enumerate(_t5)}
def half_tokmean(vals, h):
    idx_ = HM[h]
    tt = cur_tok[idx_]
    m_ = np.array([t in t5_idx for t in tt.tolist()])
    k_ = np.array([t5_idx[t] for t in tt[m_].tolist()])
    ssum = np.bincount(k_, weights=vals[idx_][m_], minlength=len(_t5))
    n_ = np.bincount(k_, minlength=len(_t5))
    return ssum / np.maximum(n_, 1)
split_rel = []
labels = {"neurons": [], "poles": []}
pole_sets = {}       # (j, s) -> dict of sets
tokmean = {}         # (j, s) -> (token ids array, mean s*act)
# token -> position index lists for conditional means (tokens with >= 5 occurrences)
order = np.argsort(cur_tok, kind="stable")
st = cur_tok[order]
uniq, starts, counts = np.unique(st, return_index=True, return_counts=True)
m5 = counts >= 5
tok5, start5, count5 = uniq[m5], starts[m5], counts[m5]
seg_id = np.repeat(np.arange(len(uniq)), counts)  # segment id in sorted order

mech = []
neuron_sets = {}
for j, (l, i) in enumerate(COLS):
    a = act[:, j]
    absmax = float(np.abs(a).max())
    hist_c, hist_e = np.histogram(a, bins=40)
    nrec = {"layer": l, "neuron": i, "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "q": {str(q): float(np.quantile(a, q)) for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)},
            "frac_pos": float((a > 0).mean()), "hist_counts": hist_c.tolist(), "hist_edges": [float(x) for x in hist_e]}
    pmax = {1: float(a.max()), -1: float(-a.min())}
    pq = {1: float(np.quantile(a, 0.999)), -1: float(-np.quantile(a, 0.001))}
    strong = 1 if pmax[1] >= pmax[-1] else -1
    nrec["strong_pole"] = "+" if strong == 1 else "-"
    nrec["bipolarity_max_ratio"] = min(pmax.values()) / max(max(pmax.values()), 1e-12)
    nrec["bipolarity_q999_ratio"] = max(min(pq.values()), 0.0) / max(max(pq.values()), 1e-12)
    labels["neurons"].append(nrec)
    # conditional means by current token
    seg_sum = np.bincount(seg_id, weights=a[order], minlength=len(uniq))
    seg_mean = seg_sum / counts
    # neuron-level (pole-agnostic) empirical sets: top-1% |act| and top-1% gate pre-activation g
    nsets = {}
    for nm, val in (("abs", np.abs(a)), ("gate", g[:, j])):
        tp = np.argpartition(-val, ntop)[:ntop]
        c_ = Counter(cur_tok[tp].tolist())
        nsets[nm + "_count"] = [t for t, _ in c_.most_common(50)]
        nsets[nm + "_enrich"] = [t for _, t, _ in enrich_rank(c_, ntop, tot_cnt, N)]
    gseg = np.bincount(seg_id, weights=g[order, j], minlength=len(uniq)) / counts
    aseg = np.bincount(seg_id, weights=np.abs(a)[order], minlength=len(uniq)) / counts
    nsets["tokmean_g"] = gseg[m5]
    nsets["tokmean_abs"] = aseg[m5]
    neuron_sets[j] = nsets
    for s in (1, -1):
        sa = s * a
        top = np.argpartition(-sa, ntop)[:ntop]
        top = top[np.argsort(-sa[top])]
        top = top[sa[top] > 0]
        nt = len(top)
        ci = Counter(cur_tok[top].tolist())
        nn_ = nxt_tok[top]
        cn = Counter(nn_[nn_ >= 0].tolist())
        nn_n = int((nn_ >= 0).sum())
        ein = enrich_rank(ci, max(nt, 1), tot_cnt, N)
        enx = enrich_rank(cn, max(nn_n, 1), tot_next, len(nxt_tok[nxt_tok >= 0]))
        gtop, utop = g[top, j], u[top, j]
        mech.append({"layer": l, "s": s, "n": nt, "frac_g_pos": float((gtop > 0).mean()) if nt else float("nan"),
                     "frac_sign_u_eq_s": float((np.sign(utop) == s).mean()) if nt else float("nan"),
                     "frac_double_neg": float(((gtop < 0) & (utop < 0)).mean()) if nt else float("nan"),
                     "mass_double_neg": float(sa[top][(gtop < 0) & (utop < 0)].sum() / max(sa[top].sum(), 1e-12)) if nt else float("nan")})
        top30 = top[:30]
        strong_sel = np.where(sa > 0.3 * pmax[s])[0] if pmax[s] > 0 else np.array([], int)
        mech[-1].update({"frac_g_pos_top30": float((g[top30, j] > 0).mean()) if len(top30) else float("nan"),
                         "frac_g_pos_gt30pct_own": float((g[strong_sel, j] > 0).mean()) if len(strong_sel) else float("nan")})
        top01 = top[:ntop01]
        ci01 = Counter(cur_tok[top01].tolist())
        nn01 = nxt_tok[top01]
        cn01 = Counter(nn01[nn01 >= 0].tolist())
        # split-half (by document) empirical labels
        sh = {}
        for h in (0, 1):
            idx_ = HM[h]
            sah = sa[idx_]
            nth = int(math.ceil(0.01 * len(idx_)))
            th_ = idx_[np.argpartition(-sah, nth)[:nth]]
            th_ = th_[sa[th_] > 0]
            ch_ = Counter(cur_tok[th_].tolist())
            sh[h] = {"count": [t for t, _ in ch_.most_common(50)],
                     "enrich": [t for _, t, _ in enrich_rank(ch_, max(len(th_), 1), cnt_half[h], len(idx_))]}
        # open-gate (g>0) pole max, for bipolarity restricted to the open-gate regime
        og = g[:, j] > 0
        pmax_open = float(sa[og].max()) if og.any() else 0.0
        prec = {
            "layer": l, "neuron": i, "pole": "+" if s == 1 else "-", "is_strong_pole": s == strong,
            "max": pmax[s], "q999": pq[s], "n_top": nt, "top1pct_threshold": float(sa[top[-1]]) if nt else None,
            "fire_freq_gt10pct_strong": float((sa > 0.1 * pmax[strong]).mean()),
            "fire_freq_gt30pct_strong": float((sa > 0.3 * pmax[strong]).mean()),
            "fire_freq_gt10pct_own": float((sa > 0.1 * pmax[s]).mean()) if pmax[s] > 0 else 0.0,
            "frac_active": float((sa > 0).mean()),
            "mech_frac_g_pos": mech[-1]["frac_g_pos"], "mech_frac_sign_u_eq_s": mech[-1]["frac_sign_u_eq_s"],
            "input_entropy_bits": float(-sum((c / nt) * math.log2(c / nt) for c in ci.values())) if nt else None,
            "input_top1_share": (ci.most_common(1)[0][1] / nt) if nt else None,
            "input_top10_share": (sum(c for _, c in ci.most_common(10)) / nt) if nt else None,
            "top_input_tokens": [[ts(t), int(t), c] for t, c in ci.most_common(30)],
            "top_input_enriched": [[ts(t), int(t), c, round(e, 3)] for e, t, c in ein[:30]],
            "top_next_tokens": [[ts(t), int(t), c] for t, c in cn.most_common(30)],
            "top_next_enriched": [[ts(t), int(t), c, round(e, 3)] for e, t, c in enx[:30]],
            "top_contexts": [{"cid": int(cid[k]), "pos": int(pos[k]), "act": float(a[k]), "src": str(src[k]),
                              "token": ts(cur_tok[k]), "context": context(int(cid[k]), int(pos[k])),
                              "next": ts(nxt_tok[k])} for k in top30],
            "src_mix_top": dict(Counter(src[top].tolist())),
        }
        labels["poles"].append(prec)
        prec["max_open_gate"] = pmax_open
        prec["mech_frac_g_pos_top30"] = mech[-1]["frac_g_pos_top30"]
        prec["top_input_tokens_top0.1pct"] = [[ts(t), int(t), c] for t, c in ci01.most_common(30)]
        prec["top_next_tokens_top0.1pct"] = [[ts(t), int(t), c] for t, c in cn01.most_common(30)]
        pole_sets[(j, s)] = {
            "sh0_count": sh[0]["count"], "sh1_count": sh[1]["count"], "sh0_enrich": sh[0]["enrich"], "sh1_enrich": sh[1]["enrich"],
            "in01_count": [t for t, _ in ci01.most_common(50)],
            "in01_enrich": [t for _, t, _ in enrich_rank(ci01, max(len(top01), 1), tot_cnt, N)],
            "nx01_count": [t for t, _ in cn01.most_common(50)],
            "in_count": [t for t, _ in ci.most_common(50)], "in_enrich": [t for _, t, _ in ein],
            "nx_count": [t for t, _ in cn.most_common(50)], "nx_enrich": [t for _, t, _ in enx],
            "nt": nt, "strong": s == strong, "q999": pq[s]}
        tokmean[(j, s)] = s * seg_mean[m5]
    po = {s_: next(p_ for p_ in labels["poles"][-2:] if p_["pole"] == ("+" if s_ == 1 else "-"))["max_open_gate"] for s_ in (1, -1)}
    nrec["bipolarity_open_gate_ratio"] = min(po.values()) / max(max(po.values()), 1e-12)
    rr = {}
    for nm, vals in (("g", g[:, j]), ("act", a), ("abs", np.abs(a))):
        m0, m1 = half_tokmean(vals, 0), half_tokmean(vals, 1)
        rr[nm] = float(np.corrcoef(rank01(m0), rank01(m1))[0, 1])
    split_rel.append({"layer": l, **rr})
    if j % 24 == 23:
        print("layer done", l, flush=True)

# ------------------------------------------------------------------ static labels
S = np.load(R + "N1B_static.npz")
sub_ids = S["sub_ids"]
sub_pos = {int(t): k for k, t in enumerate(sub_ids)}
tok5_in_sub = np.array([sub_pos.get(int(t), -1) for t in tok5])
assert (tok5_in_sub >= 0).all()
VARS_AGN = ["G_jpp", "G_jpp_pre", "G_naive", "G_naive_pre", "G_embed"]
VARS_POL = ["P_geglu0", "U_jpp_pre", "O_jvindex", "O_naive"]
PRIMARY = ["G_jpp", "G_jpp_pre", "G_naive"]

def static_top(var, j, s, k=10, attested=False):
    if attested:
        z = S[var + "__sub"][j].astype(np.float32)
        if var in VARS_POL and s == -1:
            z = -z
        return [int(sub_ids[t]) for t in np.argsort(-z)[:k]]
    if var in VARS_POL and s == -1:
        return [int(t) for t in S[var + "__bot"][j][:k]]
    return [int(t) for t in S[var + "__top"][j][:k]]

def static_scores5(var, j, s):
    z = S[var + "__sub"][j].astype(np.float32)[tok5_in_sub]
    return -z if (var in VARS_POL and s == -1) else z


POLES = [(j, s) for j in range(len(COLS)) for s in (1, -1)]

def pair_matrix(metric_fn, layer_idx, r=2):
    """M[p, j'] = metric(unit p of layer, label of neuron j' of same layer). r=2: units are poles (row 2a+b, b=0 '+',
    b=1 '-'); r=1: units are neurons."""
    js = list(range(layer_idx * NPL, (layer_idx + 1) * NPL))
    M = np.zeros((r * NPL, NPL))
    for a_, j in enumerate(js):
        for b_ in range(r):
            s = 1 if b_ == 0 else -1
            for c_, j2 in enumerate(js):
                M[r * a_ + b_, c_] = metric_fn(j, s, j2)
    return M

def summarize(Ms, mask_fn=None, r=2):
    """Ms: list over layers of [r*24, 24] matrices (diagonal block = true pairing). Per-layer + overall:
    metric, exact within-layer null (mean over other neurons), lift, diff, bootstrap CIs, permutation p."""
    out = {"per_layer": {}}
    rows = np.arange(r * NPL)
    own = rows // r
    sels, X_, N_ = [], [], []
    for li, M in enumerate(Ms):
        x = M[rows, own]
        nullp = (M.sum(1) - x) / (NPL - 1)
        sel = np.ones(r * NPL, bool) if mask_fn is None else np.array(
            [bool(mask_fn(li * NPL + p // r, 1 if p % r == 0 else -1)) for p in rows])
        out["per_layer"][str(LAYERS[li])] = boot(x[sel], nullp[sel])
        sels.append(sel); X_.append(x[sel]); N_.append(nullp[sel])
    X, Nn = np.concatenate(X_), np.concatenate(N_)
    out["overall"] = boot(X, Nn)
    obs = X.mean()
    cnt = 0
    for _ in range(NB):
        tot = []
        for li, M in enumerate(Ms):
            perm = rng.permutation(NPL)
            while (perm == np.arange(NPL)).any():
                perm = rng.permutation(NPL)
            tot.append(M[rows, perm[own]][sels[li]])
        if np.concatenate(tot).mean() >= obs:
            cnt += 1
    out["overall"]["perm_p_one_sided"] = (cnt + 1) / (NB + 1)
    return out

def boot(x, n):
    x, n = np.asarray(x, float), np.asarray(n, float)
    k = len(x)
    if k == 0:
        return {"n": 0}
    mx, mn = x.mean(), n.mean()
    idx = rng.integers(0, k, size=(NB, k))
    bx, bn = x[idx].mean(1), n[idx].mean(1)
    bd = bx - bn
    with np.errstate(divide="ignore", invalid="ignore"):
        bl = bx / bn
    lift = (mx / mn) if mn > 0 else (float("inf") if mx > 0 else float("nan"))
    fin = np.isfinite(bl)
    return {"n": k, "metric": mx, "metric_ci": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))],
            "null": mn, "lift": lift,
            "lift_ci": [float(np.quantile(bl[fin], .025)), float(np.quantile(bl[fin], .975))] if fin.sum() > NB * 0.9 else None,
            "diff": mx - mn, "diff_ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}

_empc = {}
def emp_set(j, s, which, normalized):
    k = (j, s, which, normalized)
    if k not in _empc:
        src_ = neuron_sets[j] if which.startswith(("abs_", "gate_")) else pole_sets[(j, s)]
        emp = src_[which]
        _empc[k] = {norm(ts(t)) for t in emp} if normalized else set(emp)
    return _empc[k]

def prec_fn(var, which, attested=False, normalized=False):
    cache = {}
    def f(j, s, j2):
        key = (j2, s)
        if key not in cache:
            # pole-agnostic labels do not depend on s; pole-specific labels use the SAME pole sign of neuron j2
            lab_ = static_top(var, j2, s, 10, attested)
            cache[key] = [norm(ts(t)) for t in lab_] if normalized else lab_
        es = emp_set(j, s, which, normalized)
        return sum(t in es for t in cache[key]) / 10.0
    return f

metrics = {"definitions": __doc__, "n_positions": N, "n_chunks": len(corpus), "src_positions": dict(src_counts),
           "top1pct_n": ntop, "n_tokens_ge5": int(len(tok5))}

# ---- input-side precision@10
res = {}
for var in VARS_AGN + ["P_geglu0", "U_jpp_pre"]:
    for which in ("in_count", "in_enrich"):
        for att in (False, True):
            for nrm in (False, True):
                key = f"{var}|{which}|{'attested' if att else 'fullvocab'}|{'norm' if nrm else 'strict'}"
                Ms = [pair_matrix(prec_fn(var, which, att, nrm), li) for li in range(len(LAYERS))]
                res[key] = summarize(Ms)
                o = res[key]["overall"]
                print(f"{key:55s} P@10={o['metric']:.4f} null={o['null']:.4f} lift={o['lift']:.2f} p={o['perm_p_one_sided']:.4f}", flush=True)
metrics["input_precision_at10"] = res

# ---- strong-pole-only subset (primary variants, strict + norm, full vocab)
res = {}
for var in PRIMARY + ["G_embed", "P_geglu0"]:
    for which in ("in_count", "in_enrich"):
        key = f"{var}|{which}|fullvocab|norm|strong_pole_only"
        Ms = [pair_matrix(prec_fn(var, which, False, True), li) for li in range(len(LAYERS))]
        res[key] = summarize(Ms, mask_fn=lambda j, s: pole_sets[(j, s)]["strong"])
metrics["input_precision_at10_strong_poles"] = res

# ---- token-response Spearman (static score vs token-conditional mean s*act, tokens with >= 5 occurrences)
res = {}
# (pole-agnostic labels are excluded here: their score is identical for both poles while s*act flips sign, so the
#  per-pole correlations cancel exactly; they are evaluated at neuron level against mean g and mean |act| below)
for var in ["P_geglu0", "U_jpp_pre", "O_jvindex", "O_naive"]:
    Ms = []
    for li in range(len(LAYERS)):
        js = list(range(li * NPL, (li + 1) * NPL))
        Rs = {s: np.stack([rank01(static_scores5(var, j2, s)) for j2 in js]) for s in (1, -1)}
        M = np.zeros((2 * NPL, NPL))
        for a_, j in enumerate(js):
            for b_, s in enumerate((1, -1)):
                re_ = rank01(tokmean[(j, s)])
                M[2 * a_ + b_] = Rs[s] @ re_
        Ms.append(M)
    res[var] = summarize(Ms)
    o = res[var]["overall"]
    print(f"spearman {var:12s} rho={o['metric']:.4f} null={o['null']:.4f} diff={o['diff']:.4f} p={o['perm_p_one_sided']:.4f}", flush=True)
metrics["token_response_spearman"] = res

# ---- neuron-level (pole-agnostic) evaluation of the gate labels: vs top-1% |act| and vs top-1% gate pre-activation
res = {}
for var in VARS_AGN:
    for which in ("abs_count", "abs_enrich", "gate_count", "gate_enrich"):
        for att in (False, True):
            key = f"{var}|{which}|{'attested' if att else 'fullvocab'}|norm"
            Ms = [pair_matrix(prec_fn(var, which, att, True), li, r=1) for li in range(len(LAYERS))]
            res[key] = summarize(Ms, r=1)
            o = res[key]["overall"]
            print(f"{key:55s} P@10={o['metric']:.4f} null={o['null']:.4f} lift={o['lift']:.2f} p={o['perm_p_one_sided']:.4f}", flush=True)
    for tm in ("tokmean_g", "tokmean_abs"):
        Ms = []
        for li in range(len(LAYERS)):
            js = list(range(li * NPL, (li + 1) * NPL))
            Rs = np.stack([rank01(static_scores5(var, j2, 1)) for j2 in js])
            M = np.stack([Rs @ rank01(neuron_sets[j][tm]) for j in js])
            Ms.append(M)
        key = f"{var}|spearman_{tm}"
        res[key] = summarize(Ms, r=1)
        o = res[key]["overall"]
        print(f"{key:55s} rho={o['metric']:.4f} null={o['null']:.4f} diff={o['diff']:.4f} p={o['perm_p_one_sided']:.4f}", flush=True)
metrics["gate_label_neuron_level"] = res

# ---- what does G_jpp actually track? compare static input labels to NEXT-token empirical sets; output labels too
res = {}
for var in PRIMARY + ["O_jvindex", "O_naive", "U_jpp_pre"]:
    for which in ("nx_count", "nx_enrich"):
        for att in (False, True):
            key = f"{var}|{which}|{'attested' if att else 'fullvocab'}|norm"
            Ms = [pair_matrix(prec_fn(var, which, att, True), li) for li in range(len(LAYERS))]
            res[key] = summarize(Ms)
            o = res[key]["overall"]
            print(f"{key:55s} P@10={o['metric']:.4f} null={o['null']:.4f} lift={o['lift']:.2f} p={o['perm_p_one_sided']:.4f}", flush=True)
metrics["vs_next_token_precision_at10"] = res

# ---- ceilings: split-half (by document) reliability of the empirical labels themselves, same metric + null
def sh_fn(kind, normalized):
    def f(j, s, j2):
        tot = 0.0
        for a_, b_ in ((0, 1), (1, 0)):
            lab_ = pole_sets[(j2, s)][f"sh{a_}_{kind}"][:10]
            emp = pole_sets[(j, s)][f"sh{b_}_{kind}"]
            if normalized:
                es = {norm(ts(t)) for t in emp}
                tot += sum(norm(ts(t)) in es for t in lab_) / 10.0
            else:
                es = set(emp)
                tot += sum(t in es for t in lab_) / 10.0
        return tot / 2
    return f
res = {}
for kind in ("count", "enrich"):
    for nrm in (False, True):
        key = f"splithalf_empirical_top10_vs_top50|{kind}|{'norm' if nrm else 'strict'}"
        Ms = [pair_matrix(sh_fn(kind, nrm), li) for li in range(len(LAYERS))]
        res[key] = summarize(Ms)
        o = res[key]["overall"]
        print(f"{key:55s} P@10={o['metric']:.4f} null={o['null']:.4f} lift={o['lift']:.2f}", flush=True)
sr = {}
for li, l in enumerate(LAYERS):
    S_ = [r for r in split_rel if r["layer"] == l]
    sr[str(l)] = {k: float(np.mean([r[k] for r in S_])) for k in ("g", "act", "abs")}
sr["overall"] = {k: float(np.mean([r[k] for r in split_rel])) for k in ("g", "act", "abs")}
res["splithalf_token_response_spearman"] = sr
res["n_tokens_ge5_both_halves"] = len(_t5)
metrics["ceilings"] = res

# ---- robustness: strong events only (top-0.1% per pole)
res = {}
for var in PRIMARY + ["G_embed", "O_jvindex", "O_naive"]:
    whichs = ("nx01_count",) if var.startswith("O_") else ("in01_count", "in01_enrich")
    for which in whichs:
        key = f"{var}|{which}|fullvocab|norm"
        Ms = [pair_matrix(prec_fn(var, which, False, True), li) for li in range(len(LAYERS))]
        res[key] = summarize(Ms)
        o = res[key]["overall"]
        print(f"{key:55s} P@10={o['metric']:.4f} null={o['null']:.4f} lift={o['lift']:.2f} p={o['perm_p_one_sided']:.4f}", flush=True)
metrics["top0.1pct_precision_at10"] = res

# ---- hit distribution: fraction of poles whose static top-10 has >= k hits in the empirical top-50 (norm, full vocab)
res = {}
for var in PRIMARY + ["G_embed"]:
    for which in ("in_count", "in_enrich"):
        f = prec_fn(var, which, False, True)
        diag_, off_ = [], []
        for li in range(len(LAYERS)):
            M = pair_matrix(f, li) * 10
            rows = np.arange(2 * NPL)
            diag_.append(M[rows, rows // 2])
            mask = np.ones_like(M, bool); mask[rows, rows // 2] = False
            off_.append(M[mask])
        d_, o_ = np.concatenate(diag_), np.concatenate(off_)
        res[f"{var}|{which}"] = {f"frac_ge{k}": float((d_ >= k).mean()) for k in (1, 2, 3, 5)}
        res[f"{var}|{which}"].update({f"null_frac_ge{k}": float((o_ >= k).mean()) for k in (1, 2, 3, 5)})
metrics["hit_distribution"] = res

# ------------------------------------------------------------------ diagnostics
prior = json.load(open(R + "N1B_static_prior.json"))
diag = {"label_diversity": {}, "prior_overlap": {}}
for var in VARS_AGN + VARS_POL:
    diag["label_diversity"][var] = {}
    diag["prior_overlap"][var] = {}
    for li, l in enumerate(LAYERS):
        js = range(li * NPL, (li + 1) * NPL)
        tops = [static_top(var, j, 1, 10) for j in js]
        allt = [t for tt in tops for t in tt]
        diag["label_diversity"][var][str(l)] = {"unique_frac": len(set(allt)) / len(allt),
                                                "most_common": [[ts(t), c] for t, c in Counter(allt).most_common(5)]}
        pk = "jpp" if "jpp" in var or "jvindex" in var else "naive"
        pset = {t for t, c in prior["prior_top10_counts"][f"{pk}_L{l}"] if c >= 2}
        diag["prior_overlap"][var][str(l)] = float(np.mean([sum(t in pset for t in tt) / 10 for tt in tops]))
diag["prior_top_tokens"] = {k: [[ts(t), c] for t, c in v[:12]] for k, v in prior["prior_top10_counts"].items()}
diag["static_fidelity_vs_tool"] = prior["fidelity"]
if "top10_softmax_mass" in prior:
    diag["top10_softmax_mass_median"] = {l: {k: float(np.median(v)) for k, v in d.items()}
                                         for l, d in prior["top10_softmax_mass"].items()}

# bipolarity / firing / mechanism
nb = labels["neurons"]
bip = {}
for li, l in enumerate(LAYERS):
    r1 = np.array([n["bipolarity_max_ratio"] for n in nb[li * NPL:(li + 1) * NPL]])
    r2 = np.array([n["bipolarity_q999_ratio"] for n in nb[li * NPL:(li + 1) * NPL]])
    bip[str(l)] = {"frac_ge30_max": float((r1 >= .3).mean()), "frac_ge50_max": float((r1 >= .5).mean()),
                   "frac_ge30_q999": float((r2 >= .3).mean()), "frac_ge50_q999": float((r2 >= .5).mean()),
                   "median_ratio_max": float(np.median(r1)), "frac_strong_plus": float(np.mean([n["strong_pole"] == "+" for n in nb[li * NPL:(li + 1) * NPL]]))}
r1 = np.array([n["bipolarity_max_ratio"] for n in nb]); r2 = np.array([n["bipolarity_q999_ratio"] for n in nb])
def bci(v):
    b = rng.integers(0, len(v), size=(NB, len(v)))
    return [float(np.quantile(v[b].mean(1), .025)), float(np.quantile(v[b].mean(1), .975))]
r3 = np.array([n["bipolarity_open_gate_ratio"] for n in nb])
for li, l in enumerate(LAYERS):
    v = r3[li * NPL:(li + 1) * NPL]
    bip[str(l)].update({"frac_ge30_open_gate": float((v >= .3).mean()), "frac_ge50_open_gate": float((v >= .5).mean())})
bip["overall"] = {"frac_ge30_open_gate": float((r3 >= .3).mean()), "frac_ge30_open_gate_ci": bci((r3 >= .3).astype(float)),
                  "frac_ge50_open_gate": float((r3 >= .5).mean()), "frac_ge50_open_gate_ci": bci((r3 >= .5).astype(float)),
                  "frac_ge30_max": float((r1 >= .3).mean()), "frac_ge30_max_ci": bci((r1 >= .3).astype(float)),
                  "frac_ge50_max": float((r1 >= .5).mean()), "frac_ge50_max_ci": bci((r1 >= .5).astype(float)),
                  "frac_ge30_q999": float((r2 >= .3).mean()), "frac_ge50_q999": float((r2 >= .5).mean()),
                  "median_ratio_max": float(np.median(r1))}
diag["bipolarity"] = bip
pp = labels["poles"]
fire = {}
for li, l in enumerate(LAYERS):
    P = pp[li * 2 * NPL:(li + 1) * 2 * NPL]
    fire[str(l)] = {
        "strong_pole_fire_gt10pct": float(np.mean([p["fire_freq_gt10pct_strong"] for p in P if p["is_strong_pole"]])),
        "weak_pole_fire_gt10pct_strong": float(np.mean([p["fire_freq_gt10pct_strong"] for p in P if not p["is_strong_pole"]])),
        "strong_pole_fire_gt30pct": float(np.mean([p["fire_freq_gt30pct_strong"] for p in P if p["is_strong_pole"]])),
        "weak_pole_fire_gt30pct_strong": float(np.mean([p["fire_freq_gt30pct_strong"] for p in P if not p["is_strong_pole"]])),
        "frac_active_plus": float(np.mean([p["frac_active"] for p in P if p["pole"] == "+"])),
        "input_top1_share_mean": float(np.mean([p["input_top1_share"] for p in P])),
        "input_top10_share_mean": float(np.mean([p["input_top10_share"] for p in P])),
        "input_entropy_bits_mean": float(np.mean([p["input_entropy_bits"] for p in P])),
        "input_top10_share_strong": float(np.mean([p["input_top10_share"] for p in P if p["is_strong_pole"]])),
    }
diag["firing_and_concentration"] = fire
mm = {}
for li, l in enumerate(LAYERS):
    M_ = [m for m in mech if m["layer"] == l]
    mm[str(l)] = {k: float(np.nanmean([m[k] for m in M_])) for k in ("frac_g_pos", "frac_double_neg", "mass_double_neg",
                                                                      "frac_g_pos_top30", "frac_g_pos_gt30pct_own")}
mm["note"] = ("frac_g_pos over a pole's top-1% positions equals frac(sign(u)==s) by construction (s*act>0). top-1% of "
              "positions includes many weak, closed-gate (g<0) positions for sparse neurons; *_top30 and *_gt30pct_own "
              "restrict to strong events.")
diag["mechanism_top1pct"] = mm
# pole distinctness: Jaccard of the two poles' input sets within a neuron vs across neurons in the layer
dist = {}
for which in ("in_count", "in_enrich", "nx_count"):
    within, across = [], []
    for li in range(len(LAYERS)):
        js = list(range(li * NPL, (li + 1) * NPL))
        for j in js:
            A = set(pole_sets[(j, 1)][which]); B = set(pole_sets[(j, -1)][which])
            within.append(len(A & B) / max(len(A | B), 1))
            for j2 in js:
                if j2 != j:
                    B2 = set(pole_sets[(j2, -1)][which])
                    across.append(len(A & B2) / max(len(A | B2), 1))
    dist[which] = {"jaccard_within_neuron_pm": float(np.mean(within)), "jaccard_across_neuron_pm": float(np.mean(across))}
diag["pole_distinctness"] = dist
metrics["diagnostics"] = diag

# attach static labels to pole records for convenience
for p_, (j, s) in zip(labels["poles"], POLES):
    p_["static_labels_top10"] = {var: [ts(t) for t in static_top(var, j, s, 10)] for var in VARS_AGN + VARS_POL}

json.dump(labels, open(R + "N1B_labels.json", "w"))
json.dump({f"{j}|{s}": {k: (v if isinstance(v, list) else v) for k, v in d.items() if isinstance(v, list)}
           for (j, s), d in pole_sets.items()}, open(R + "N1B_pole_sets.json", "w"))
json.dump(metrics, open(R + "N1B_metrics.json", "w"), indent=1)
print("wrote N1B_labels.json, N1B_metrics.json")
