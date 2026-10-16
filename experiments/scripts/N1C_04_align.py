#!/usr/bin/env python3
# N1C_04_align -- position-level alignment of labels with what actually happens when a pole fires (CPU, local).
# Run: .venv/bin/python experiments/scripts/N1C_04_align.py -> experiments/results/N1C_align.json
#
# Set-overlap metrics (N1C_02) only look at top-10 vs top-100 lists. This metric uses the whole label ranking:
#   pr_p(t) = percentile of token t in label X_p over vocab A (0 = bottom, 1 = top; ties averaged)
#   ALIGN_next(X, p) = mean_{top-1% positions of p} pr_p(next token) - mean_{all positions} pr_p(next token)
#   ALIGN_cur (X, p) = same with the CURRENT token (input-side labels)
# i.e. how many percentile points higher the label ranks the token that actually follows (or is present) when the
# pole fires, relative to baseline text. NULL: label X_q of every other sampled neuron (same layer, same sign) evaluated
# at p's firing positions (23 pairs). lift = real - null; bootstrap CIs over poles; within-layer permutation p.
# CEILING: smoothed next/current-token enrichment label estimated on document half 0, evaluated on half-1 positions.
# PRE-STATED PREDICTIONS:
#  A1 ALIGN_next(C) lift > 0 at L40-57 (a few percentile points) and ~0 at L10-30; ALIGN_next(JV) ~ ALIGN_next(NV)
#     ~ ALIGN_next(C) at L40-57; all far below the empirical split-half ceiling.
#  A2 ALIGN_cur(G) lift small (< 2 pct points) everywhere; ALIGN_cur(Gemb) larger than G at L10.
#  A3 Polarity: ALIGN_next(C_s) at the opposite pole's firing positions is NEGATIVE at L40-57 (the - pole's contexts are
#     followed by tokens the + pole suppresses) -- tests whether the two poles are one signed axis in natural text.
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

t0 = time.time()
Z = np.load(f"{ROOT}/N1C_labels.npz")
A = Z["A"]; nA = len(A)
LAB = {k[4:]: Z[k] for k in Z.files if k.startswith("lab_")}
V = 262144
a_of = -np.ones(V, dtype=np.int64); a_of[A] = np.arange(nA)
D = load_corpus_acts()
act, cur, nxt, half = D["act"], D["cur"], D["nxt"], D["half"]
N = len(act)
ntop = int(math.ceil(0.01 * N))
cur_a = a_of[cur]
nxt_a = np.where(nxt >= 0, a_of[np.maximum(nxt, 0)], -1)
mc, mn = cur_a >= 0, nxt_a >= 0
base_cur = np.bincount(cur_a[mc], minlength=nA).astype(float)
base_nxt = np.bincount(nxt_a[mn], minlength=nA).astype(float)
rng = np.random.default_rng(11)


def pct(X):
    """row-wise percentile ranks in [0,1] with ties averaged; -inf entries tie at the bottom."""
    X = np.where(np.isfinite(X), X, -1e30)
    out = np.empty(X.shape, dtype=np.float32)
    for r in range(X.shape[0]):
        x = X[r]
        o = np.argsort(x, kind="mergesort")
        xs = x[o]
        _, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
        avg = np.cumsum(cnt) - (cnt - 1) / 2.0
        rk = np.empty(len(x)); rk[o] = avg[inv]
        out[r] = (rk - 1) / (len(x) - 1)
    return out


TOP = {}
for row, (l, n, s) in enumerate(POLES):
    sa = s * act[:, row // 2]
    t = np.argpartition(-sa, ntop)[:ntop]
    TOP[row] = t[sa[t] > 0]

# baseline mean percentile of next/current token over all positions, per label row: = (base . pr) / total
def align_matrix(PR, rows, which, partner_rows):
    """M[p, q] = ALIGN of label row q evaluated at firing positions of pole p (local indices)."""
    ta = nxt_a if which == "next" else cur_a
    base = base_nxt if which == "next" else base_cur
    basemean = (PR[[r for r in partner_rows]] @ base) / base.sum()      # [Q]
    M = np.zeros((len(rows), len(partner_rows)))
    for i, p in enumerate(rows):
        tk = ta[TOP[p]]; tk = tk[tk >= 0]
        if len(tk) == 0:
            M[i] = np.nan; continue
        h = np.bincount(tk, minlength=nA).astype(float)
        M[i] = (PR[partner_rows] @ h) / h.sum() - basemean
    return M


def stats(Mfull, rows, scheme):
    R_ = len(rows)
    sg = np.array([POLES[r][2] for r in rows]); nn = np.array([POLES[r][1] for r in rows])
    real_idx = np.arange(R_) if scheme == "same" else np.array([p + 1 if sg[p] == 1 else p - 1 for p in range(R_)])
    tgt = sg if scheme == "same" else -sg
    mask = (sg[None, :] == tgt[:, None]) & (nn[None, :] != nn[:, None])
    real = Mfull[np.arange(R_), real_idx]
    null = np.nanmean(np.where(mask, Mfull, np.nan), 1)
    obs = np.nanmean(real); cnt = 0
    for _ in range(2000):
        perm = rng.permutation(R_ // 2)
        q = 2 * perm[np.arange(R_) // 2] + (real_idx % 2)
        cnt += np.nanmean(Mfull[np.arange(R_), q]) >= obs
    return real, null, (cnt + 1) / 2001


def run(label, which, scheme="same", PRd=None):
    PR = PRd if PRd is not None else pct(LAB[label])
    out = {"per_layer": {}}
    acc = {"all": ([], []), "early": ([], []), "late": ([], [])}
    for l in LAYERS:
        rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        M = align_matrix(PR, rows, which, rows)
        real, null, pp = stats(M, rows, scheme)
        b = boot_pair(real, null, seed=l); b["perm_p"] = pp
        out["per_layer"][str(l)] = b
        for k in ("all", "early" if l in EARLY else "late"):
            acc[k][0].append(real); acc[k][1].append(null)
    for k, (r, n) in acc.items():
        out[{"all": "overall", "early": "early_L10_30", "late": "late_L40_57"}[k]] = boot_pair(np.concatenate(r), np.concatenate(n), seed=5)
    return out


RES = {"definition": "ALIGN = mean percentile (in the pole's label ranking over vocab A) of the actual next/current token at the pole's top-1% positions minus the same mean over all positions; null = other neurons' labels at the same positions",
       "next": {}, "cur": {}, "cross_pole_next": {}, "ceiling": {}}
for lab in ["C", "CB", "JV", "JL", "NV", "JVz", "NVz", "U"]:
    RES["next"][lab] = run(lab, "next"); print("next", lab, f"{time.time()-t0:.0f}s", flush=True)
for lab in ["G", "Gz", "Gpre", "Gemb", "Gnaive", "U", "JV", "C"]:
    RES["cur"][lab] = run(lab, "cur"); print("cur", lab, f"{time.time()-t0:.0f}s", flush=True)
for lab in ["C", "JV"]:
    RES["cross_pole_next"][lab] = run(lab, "next", "cross")
# ceiling: smoothed enrichment label from half 0, evaluated at half-1 firing positions (replace TOP temporarily)
H0, H1 = np.where(half == 0)[0], np.where(half == 1)[0]
TOP_full = dict(TOP)
for which, ta in (("next", nxt_a), ("cur", cur_a)):
    Lh = np.zeros((P, nA), dtype=np.float32)
    b0 = np.bincount(ta[H0][ta[H0] >= 0], minlength=nA).astype(float)
    for row in range(P):
        sa = POLES[row][2] * act[:, row // 2]
        n0 = int(math.ceil(0.01 * len(H0)))
        t0_ = H0[np.argpartition(-sa[H0], n0)[:n0]]; t0_ = t0_[sa[t0_] > 0]
        tk = ta[t0_]; tk = tk[tk >= 0]
        c = np.bincount(tk, minlength=nA).astype(float)
        Lh[row] = np.log((c + 0.5) / max(len(tk), 1)) - np.log((b0 + 0.5) / b0.sum())
        n1 = int(math.ceil(0.01 * len(H1)))
        t1_ = H1[np.argpartition(-sa[H1], n1)[:n1]]; TOP[row] = t1_[sa[t1_] > 0]
    RES["ceiling"][which] = run(None, which, PRd=pct(Lh))
    TOP = dict(TOP_full)
print("ceiling", f"{time.time()-t0:.0f}s")
json.dump(RES, open(f"{ROOT}/N1C_align.json", "w"), indent=1)

def row(name, r):
    c = lambda b: f"{b['diff']*100:+.1f} [{b['diff_ci'][0]*100:+.1f},{b['diff_ci'][1]*100:+.1f}]"
    return f"| {name} | " + " | ".join(c(r["per_layer"][str(l)]) for l in LAYERS) + f" | {c(r['overall'])} |"
print("| label | " + " | ".join(f"L{l}" for l in LAYERS) + " | all |")
for k in ("next", "cur", "cross_pole_next", "ceiling"):
    for lab, r in RES[k].items():
        print(row(f"{k}:{lab}", r))
print("real/null overall:", {f"{k}:{lab}": (round(r['overall']['real'] * 100, 2), round(r['overall']['null'] * 100, 2)) for k in RES if isinstance(RES[k], dict) for lab, r in RES[k].items() if isinstance(r, dict) and 'overall' in r})
