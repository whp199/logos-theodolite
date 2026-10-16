#!/usr/bin/env python3
# N1C_01_labels -- build every label type for the 384 fixed poles on ONE common footing (CPU, local).
# Run: .venv/bin/python experiments/scripts/N1C_01_labels.py   -> experiments/results/N1C_labels.npz
#
# Vocabularies
#   FULL = all 262144 tokens (only weight-derived and causal labels exist here).
#   A    = "attested" sub-vocab: tokens with corpus count >= 3 (N1B sub_ids, 5544 tokens). Every label is restricted
#          to A before ranking, so static/causal labels are compared with empirical labels on tokens that can occur.
#   W5   = tokens of A seen >= 5 times as current AND as next token (for continuous token-response correlations).
# Label score vectors per pole p = (l, i, s)   (higher = more associated; -inf = not rankable)
#   JV  J-VINDEX (tool) = s*unembed(W_down[:,i]*w_post @ J++[l].T)       JL  same through JLens      NV naive lens
#   JVz/NVz/JLz  z vs isotropic-random prior (N1A)                         JVzn/NVzn  / neuron-prior sd (N1A)
#   C   causal: mean over 36 contexts of logprob(steered) - logprob(base), delta = s*60*max|act| on mlp_act[l,last,i]
#   CB  same at k = 180
#   G   gate_trigger_tokens (tool) = unembed(W_gate[i] @ J++[l].T)  -- neuron-level (identical for both poles)
#   Gz  = G z-scored vs the isotropic J++ prior (null-calibrated gate label)
#   Gpre (W_gate*w_pre through J++), Gemb (W_gate*w_pre . W_E, direct), U = s*unembed(W_up*w_pre @ J++) (pole-signed)
#   Enx empirical NEXT token at the pole's top-1% positions (ranked by s*act, s*act>0): log-enrichment vs corpus
#       next-token frequency, count >= 3 required (Goodfire-style frequency-controlled label)
#   EnxC same, ranked by raw count (count >= 1);  Enx01 top-0.1% positions, ranked by count
#   Ein/EinC/Ein01  same for the CURRENT (input) token
#   Enx_h0/_h1, Ein_h0/_h1  split-half by document (reliability ceiling)
#   Rnx/Rin  continuous: mean relu(s*act) over all positions whose next/current token is t (t in W5)
import json, math, os, sys, time
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

t0 = time.time()
CACHE = f"{ROOT}/N1A_cache"
S = np.load(f"{ROOT}/N1B_static.npz")
A = S["sub_ids"].astype(np.int64)
nA = len(A)
V = 262144
a_of = -np.ones(V, dtype=np.int64); a_of[A] = np.arange(nA)

lab = {}            # name -> [P, nA] float32
top_full = {}       # name -> [P, 100] int32 (full-vocab top-100 token ids)
def put(name, row, vec):
    lab.setdefault(name, np.full((P, nA), -np.inf, dtype=np.float32))[row] = vec

# ---------------------------------------------------------------- static output + causal (N1A cache)
meta_rows = []
for l in LAYERS:
    st = torch.load(f"{CACHE}/static_L{l}.pt")
    pri = torch.load(f"{CACHE}/prior_L{l}.pt")
    assert st["neurons"] == NEURONS[l]
    cz = {t: torch.load(f"{CACHE}/causal_{t}_L{l}.pt") for t in ("kA", "kB")}
    for t in cz:
        sel = [k for k, m in enumerate(cz[t]["poles"]) if m["set"] == "sample"]
        cz[t] = (cz[t]["c"][sel].float(), [cz[t]["poles"][k] for k in sel])
    for ni, n in enumerate(NEURONS[l]):
        for s in (1, -1):
            row = POLES.index((l, n, s))
            pi = 2 * ni + (0 if s == 1 else 1)
            assert (cz["kA"][1][pi]["n"], cz["kA"][1][pi]["s"]) == (n, s)
            vecs = {}
            for base, nm in (("jpp", "JV"), ("jlens", "JL"), ("naive", "NV")):
                v = s * st[base][ni].float()
                vecs[nm] = v
                mu, sd = pri["iso"][base]
                vecs[nm + "z"] = (v - mu) / sd
                vecs[nm + "zn"] = v / pri["neu"][base]
            vecs["C"] = cz["kA"][0][pi]
            vecs["CB"] = cz["kB"][0][pi]
            for nm, v in vecs.items():
                put(nm, row, v.numpy()[A])
                top_full.setdefault(nm, np.zeros((P, 100), dtype=np.int32))[row] = v.topk(100).indices.numpy()
            meta_rows.append({"row": row, "layer": l, "neuron": n, "sign": s, "kl_kA": cz["kA"][1][pi]["kl_mean"],
                              "split_rho_kA": cz["kA"][1][pi]["split_rho"]})
    print(f"static/causal L{l} {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- static input labels (N1B)
for nm, var, pol in (("G", "G_jpp", False), ("Gpre", "G_jpp_pre", False), ("Gemb", "G_embed", False),
                     ("Gnaive", "G_naive", False), ("U", "U_jpp_pre", True), ("Oj", "O_jvindex", True)):
    Z = S[var + "__sub"].astype(np.float32)  # [192, nA] neuron order == pole order // 2
    Z = np.where(np.isfinite(Z), Z, np.nan)
    for row, (l, n, s) in enumerate(POLES):
        z = Z[row // 2]
        put(nm, row, (s * z) if pol else z)
    lab[nm] = np.nan_to_num(lab[nm], nan=-np.inf)
# Gz: tool gate label z-scored against the isotropic-random-vector J++ prior (N1A prior_L*.pt, post-softcap logits),
# i.e. the "null-calibrated" version of gate_trigger_tokens (same J++ -> unembed map as J-VINDEX).
Gsub = S["G_jpp__sub"].astype(np.float32)
for l in LAYERS:
    mu, sd = torch.load(f"{CACHE}/prior_L{l}.pt")["iso"]["jpp"]
    mu, sd = mu.numpy()[A], sd.numpy()[A]
    for row, (ll, n, s) in enumerate(POLES):
        if ll == l:
            put("Gz", row, (30 * np.tanh(Gsub[row // 2] / 30) - mu) / sd)

# ---------------------------------------------------------------- empirical labels (N1B activations)
D = load_corpus_acts()
act, cur, nxt, half = D["act"], D["cur"], D["nxt"], D["half"]
N = len(act)
ntop, ntop01 = int(math.ceil(0.01 * N)), int(math.ceil(0.001 * N))
cur_a = a_of[cur]
nxt_a = np.where(nxt >= 0, a_of[np.maximum(nxt, 0)], -1)
base_cur = np.bincount(cur_a[cur_a >= 0], minlength=nA).astype(np.float64)
base_nxt = np.bincount(nxt_a[nxt_a >= 0], minlength=nA).astype(np.float64)
Ncur, Nnxt = float(len(cur)), float((nxt >= 0).sum())
W5 = np.where((base_cur >= 5) & (base_nxt >= 5))[0]
HM = {h: np.where(half == h)[0] for h in (0, 1)}
hb = {}
for h in (0, 1):
    ia = cur_a[HM[h]]; na = nxt_a[HM[h]]
    hb[h] = (np.bincount(ia[ia >= 0], minlength=nA).astype(np.float64), float(len(HM[h])),
             np.bincount(na[na >= 0], minlength=nA).astype(np.float64), float((nxt[HM[h]] >= 0).sum()))

def enrich(cnt, n, base, nb, minc=3):
    e = np.log((cnt + 0.5) / max(n, 1)) - np.log((base + 0.5) / nb)
    return np.where(cnt >= minc, e, -np.inf).astype(np.float32)

def countscore(cnt, n, base, nb):
    e = np.log((cnt + 0.5) / max(n, 1)) - np.log((base + 0.5) / nb)
    return np.where(cnt >= 1, cnt + 1e-3 * np.tanh(e), -np.inf).astype(np.float32)

def toks_at(idx, arr_a):
    x = arr_a[idx]
    x = x[x >= 0]
    return np.bincount(x, minlength=nA).astype(np.float64), len(x)

emp_meta = []
for row, (l, n, s) in enumerate(POLES):
    j = row // 2
    sa = s * act[:, j]
    top = np.argpartition(-sa, ntop)[:ntop]
    top = top[np.argsort(-sa[top])]
    top = top[sa[top] > 0]
    cn, nn = toks_at(top, nxt_a); ci, ni = toks_at(top, cur_a)
    put("Enx", row, enrich(cn, nn, base_nxt, Nnxt)); put("EnxC", row, countscore(cn, nn, base_nxt, Nnxt))
    put("Ein", row, enrich(ci, ni, base_cur, Ncur)); put("EinC", row, countscore(ci, ni, base_cur, Ncur))
    t01 = top[:ntop01]
    cn1, nn1 = toks_at(t01, nxt_a); ci1, ni1 = toks_at(t01, cur_a)
    put("Enx01", row, countscore(cn1, nn1, base_nxt, Nnxt)); put("Ein01", row, countscore(ci1, ni1, base_cur, Ncur))
    for h in (0, 1):
        idx = HM[h]
        sh = sa[idx]
        nth = int(math.ceil(0.01 * len(idx)))
        th = idx[np.argpartition(-sh, nth)[:nth]]
        th = th[sa[th] > 0]
        bc, nbc, bn, nbn = hb[h]
        c_, n_ = toks_at(th, nxt_a); put(f"Enx_h{h}", row, enrich(c_, n_, bn, nbn))
        c_, n_ = toks_at(th, cur_a); put(f"Ein_h{h}", row, enrich(c_, n_, bc, nbc))
    r = np.maximum(sa, 0)
    m = nxt_a >= 0
    sn = np.bincount(nxt_a[m], weights=r[m], minlength=nA) / np.maximum(base_nxt, 1)
    m = cur_a >= 0
    si = np.bincount(cur_a[m], weights=r[m], minlength=nA) / np.maximum(base_cur, 1)
    v = np.full(nA, -np.inf, np.float32); v[W5] = sn[W5]; put("Rnx", row, v)
    v = np.full(nA, -np.inf, np.float32); v[W5] = si[W5]; put("Rin", row, v)
    emp_meta.append({"row": row, "n_top": int(len(top)), "n_valid_Enx": int(np.isfinite(lab["Enx"][row]).sum()),
                     "n_valid_Ein": int(np.isfinite(lab["Ein"][row]).sum())})
    if row % 48 == 47:
        print(f"empirical L{l} {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- sanity checks vs N1B pole sets / N1B static
ps = json.load(open(f"{ROOT}/N1B_pole_sets.json"))
agree = []
for row, (l, n, s) in enumerate(POLES):
    key = f"{row//2}|{s}"
    mine = A[np.argsort(-lab["Ein"][row])[:20]]
    theirs = ps[key]["in_enrich"][:20]
    k = min(len(theirs), int(np.isfinite(lab["Ein"][row]).sum()), 20)
    agree.append(len(set(mine[:k].tolist()) & set(theirs[:k])) / max(k, 1))
ojv = [np.corrcoef(np.nan_to_num(lab["Oj"][r], neginf=0), lab["JV"][r])[0, 1] for r in range(0, P, 37)]
checks = {"ein_top20_overlap_with_N1B_mean": float(np.mean(agree)), "ein_top20_overlap_min": float(np.min(agree)),
          "pearson_N1B_Ojvindex_vs_N1A_JV_on_A": [float(x) for x in ojv], "nA": int(nA), "nW5": int(len(W5)),
          "N_positions": int(N), "ntop": ntop, "ntop01": ntop01}
print(json.dumps(checks, indent=1))
np.savez_compressed(f"{ROOT}/N1C_labels.npz", A=A, W5=W5, **{f"lab_{k}": v for k, v in lab.items()},
                    **{f"top_{k}": v for k, v in top_full.items()})
json.dump({"checks": checks, "meta": meta_rows, "emp_meta": emp_meta, "labels": sorted(lab), "full_labels": sorted(top_full)},
          open(f"{ROOT}/N1C_labels_meta.json", "w"), indent=1)
print(f"saved {time.time()-t0:.0f}s")
