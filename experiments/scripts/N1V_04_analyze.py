#!/usr/bin/env python3
# N1V_04_analyze -- (CPU, local .venv python) robustness re-measurement of the N1 headline claims on the FRESH sample.
# Run: .venv/bin/python experiments/scripts/N1V_04_analyze.py  -> experiments/results/N1V_analysis.json (+ printed tables)
#
# Poles: 64 fresh neurons (seed 99) x {+,-} = 128, order layer-major, neuron, sign (+ then -).
# Labels (score vectors; higher = more associated):
#   JV / JL / NV   static output labels (s * tool logits), FULL vocab;  JVz / NVz  z vs N1A isotropic prior (lens property)
#   C:<cond>       causal c_last (mean over 32 fresh contexts of logprob change at the last position)
#   C8:<cond>      causal averaged over the 8 window positions
#   Canti:<cond>   antisymmetric part (c(+) - c(-))/2 * s   (removes anything both poles share)
#   Ccent:<cond>   c minus the mean c over all 16 poles of the layer (removes the layer-generic response)
#   Enx / Ein      activation-based: next / current token at the pole's top-1% positions (s*act > 0) on the N1B corpus,
#                  log-enrichment vs corpus frequency, count >= 3 (vocab A = N1B attested tokens);  _h0/_h1 split halves
#   G              tool gate_trigger_tokens (neuron-level);  Gemb  direct W_gate*w_pre . W_E;  Uemb s*(W_up*w_pre . W_E)
# Metrics: P@10->100 (|top10(X) & top100(Y)| / k), Spearman (FULL vocab), each with the exhaustive within-layer
#   permutation null (X of pole p vs Y of the 7 other neurons' same-sign poles), lift = real - null, 95% bootstrap CIs
#   over poles (2000 resamples).
import json, math, os, glob, sys, zlib, time
import numpy as np
import torch

ROOT = "./experiments/results"
CACHE = f"{ROOT}/N1V_cache"
NEUJ = json.load(open(f"{ROOT}/N1V_neurons.json"))
LAYERS = [int(l) for l in NEUJ["layers"]]
NEU = {int(l): [int(i) for i in v] for l, v in NEUJ["neurons"].items()}
POLES = [(l, n, s) for l in LAYERS for n in NEU[l] for s in (1, -1)]
P = len(POLES)
EARLY, LATE = [10, 20, 30], [40, 45, 50, 55, 57]
V = 262144
CONDS = ["A8k60", "A1k60", "A8k10", "L8k10", "L8k60"]
torch.set_num_threads(16)
t0 = time.time()
rng0 = np.random.default_rng(0)


def boot(real, null=None, nb=2000, seed=0):
    real = np.asarray(real, float)
    null = np.zeros_like(real) if null is None else np.asarray(null, float)
    ok = np.isfinite(real) & np.isfinite(null)
    real, null = real[ok], null[ok]
    k = len(real)
    if k == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(nb, k))
    bd = real[idx].mean(1) - null[idx].mean(1)
    return {"n": k, "real": float(real.mean()), "null": float(null.mean()), "lift": float(real.mean() - null.mean()),
            "ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}


def boot_cluster(d, layers, nb=2000, seed=1):
    """bootstrap of the mean of per-pole values d, resampling NEURONS (both poles together) within the given layers."""
    units = [(l, n) for l in layers for n in NEU[l]]
    vals = np.array([np.mean([d[POLES.index((l, n, s))] for s in (1, -1)]) for l, n in units])
    rng = np.random.default_rng(seed)
    b = vals[rng.integers(0, len(vals), (nb, len(vals)))].mean(1)
    return {"n_neurons": len(vals), "lift": float(vals.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}


def fmt(b, d=3):
    if not b or b.get("n", 0) == 0:
        return "-"
    star = "*" if (b["ci"][0] > 0 or b["ci"][1] < 0) else ""
    return f"{b['real']:.{d}f}/{b['null']:.{d}f}{star}"


def fmtl(b, d=3):
    if not b or b.get("n", 0) == 0:
        return "-"
    return f"{b['lift']:+.{d}f} [{b['ci'][0]:+.{d}f}, {b['ci'][1]:+.{d}f}]"


# ------------------------------------------------------------------------------------------------ load labels (FULL)
L = {}  # name -> torch [P, V] float32
def put(name, row, vec):
    if name not in L:
        L[name] = torch.full((P, V), float("nan"), dtype=torch.float32)
    L[name][row] = vec

prior = {l: torch.load(f"{ROOT}/N1A_cache/prior_L{l}.pt") for l in LAYERS}
static = {l: torch.load(f"{CACHE}/static_L{l}.pt") for l in LAYERS}
for l in LAYERS:
    assert static[l]["neurons"] == NEU[l]
causal = {}
for c in CONDS:
    for l in LAYERS:
        fn = f"{CACHE}/causal_{c}_L{l}.pt"
        if os.path.exists(fn):
            causal[(c, l)] = torch.load(fn)
have_conds = [c for c in CONDS if all((c, l) in causal for l in LAYERS)]
print("conditions complete:", have_conds, " partial:", sorted({c for c, l in causal} - set(have_conds)))

meta = {"causal": {}}
for row, (l, n, s) in enumerate(POLES):
    j = NEU[l].index(n)
    st = static[l]
    for nm in ("JV", "JL", "NV"):
        put(nm, row, s * st[nm][j].float())
    for nm, base in (("JVz", "jpp"), ("NVz", "naive"), ("JLz", "jlens")):
        mu, sd = prior[l]["iso"][base]
        put(nm, row, (s * st[{"jpp": "JV", "naive": "NV", "jlens": "JL"}[base]][j].float() - mu) / sd)
    put("G", row, st["G"][j].float())
    put("Gemb", row, st["Gemb"][j].float())
    put("Uemb", row, s * st["Uemb"][j].float())
for c in have_conds:
    for l in LAYERS:
        d = causal[(c, l)]
        pm = d["poles"]
        for k, m in enumerate(pm):
            if m["set"] != "sample":
                continue
            row = POLES.index((l, m["n"], m["s"]))
            put(f"C:{c}", row, d["c_last"][k].float())
            put(f"C8:{c}", row, d["c8"][k].float())
            meta["causal"].setdefault(c, {})[row] = {"kl": m["kl_mean"], "split_rho": m["split_rho"]}
    # antisymmetric / centred
    for row, (l, n, s) in enumerate(POLES):
        rp, rm = POLES.index((l, n, 1)), POLES.index((l, n, -1))
        put(f"Canti:{c}", row, s * 0.5 * (L[f"C:{c}"][rp] - L[f"C:{c}"][rm]))
        put(f"Csym:{c}", row, 0.5 * (L[f"C:{c}"][rp] + L[f"C:{c}"][rm]))
    for l in LAYERS:
        rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        mu = L[f"C:{c}"][rows].mean(0)
        for r in rows:
            put(f"Ccent:{c}", r, L[f"C:{c}"][r] - mu)
print(f"loaded labels {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------------------------------------ helpers
TOPK = {}
def topk_full(name, k=100):
    key = (name, k)
    if key not in TOPK:
        X = torch.nan_to_num(L[name], nan=-float("inf"))
        TOPK[key] = X.topk(k, dim=-1).indices.numpy()
    return TOPK[key]

RANKS = {}
def ranks_full(name):
    if name not in RANKS:
        X = torch.nan_to_num(L[name], nan=0.0)
        R = torch.empty_like(X)
        for r in range(P):
            o = X[r].argsort()
            R[r, o] = torch.arange(V, dtype=torch.float32)
        R -= R.mean(-1, keepdim=True)
        R /= R.norm(dim=-1, keepdim=True)
        RANKS[name] = R
    return RANKS[name]

def partners(row, opposite=False):
    l, n, s = POLES[row]
    ss = -s if opposite else s
    return [POLES.index((l, m, ss)) for m in NEU[l] if m != n]

def p_at(xtop10, ytop100):
    return len(set(xtop10.tolist()) & set(ytop100.tolist())) / max(len(xtop10), 1)

def pairwise_full(X, Y, metric="p10", opposite=False, negY=False, same_neuron_other_pole=False):
    """per-pole real and null arrays. same_neuron_other_pole: real partner = other pole of the same neuron,
    null partners = opposite-sign poles of the other neurons."""
    real, null = np.zeros(P), np.zeros(P)
    if metric == "p10":
        tx = topk_full(X, 10)
        if negY:
            Yn = "__neg__" + Y
            if Yn not in L:
                L[Yn] = -L[Y]
            ty = topk_full(Yn, 100)
        else:
            ty = topk_full(Y, 100)
        f = lambda a, b: p_at(tx[a], ty[b])
    else:
        RX = ranks_full(X); RY = ranks_full(Y)
        sgn = -1.0 if negY else 1.0
        f = lambda a, b: sgn * float(RX[a] @ RY[b])
    for r in range(P):
        l, n, s = POLES[r]
        if same_neuron_other_pole:
            real[r] = f(r, POLES.index((l, n, -s)))
            null[r] = np.mean([f(r, q) for q in partners(r, opposite=True)])
        else:
            real[r] = f(r, r)
            null[r] = np.mean([f(r, q) for q in partners(r, opposite=opposite)])
    return real, null

def by_layer(real, null, extra=None):
    out = {}
    for l in LAYERS:
        idx = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        out[str(l)] = boot(real[idx], null[idx])
    for nm, ls in (("early", EARLY), ("late", LATE), ("all", LAYERS)):
        idx = [r for r, (ll, _, _) in enumerate(POLES) if ll in ls]
        out[nm] = boot(real[idx], null[idx])
    return out

OUT = {"tables": {}, "paired": {}, "reliability": {}, "kl": {}, "bipolar": {}, "stability": {}, "notes": {}}
COLS = [str(l) for l in LAYERS] + ["early", "late", "all"]
def row_str(name, tab):
    return f"| {name} | " + " | ".join(fmt(tab[c]) for c in COLS) + " |"
def lift_str(name, tab):
    return f"| {name} | " + " | ".join(f"{tab[c]['lift']:+.3f}" + ("*" if (tab[c]['ci'][0] > 0 or tab[c]['ci'][1] < 0) else "") for c in COLS) + " |"
HDR = "| | " + " | ".join(("L" + c if c.isdigit() else c) for c in COLS) + " |\n|" + "---|" * (len(COLS) + 1)

# ------------------------------------------------------------------------------------------------ T1 static vs causal
lines = []
PER = {}
for c in have_conds:
    lines.append(f"\n### T1[{c}] P@10 label->causal top-100 (FULL vocab), real/null\n" + HDR)
    for X in ("JV", "NV", "JL", "JVz", "NVz"):
        for Cn in (f"C:{c}", f"C8:{c}", f"Canti:{c}", f"Ccent:{c}"):
            if Cn.startswith(("C8", "Canti", "Ccent")) and X not in ("JV", "NV"):
                continue
            real, null = pairwise_full(X, Cn, "p10")
            tab = by_layer(real, null)
            OUT["tables"][f"p10|{X}->{Cn}"] = tab
            PER[f"p10|{X}->{Cn}"] = real
            lines.append(row_str(f"{X}->{Cn}", tab))
    lines.append(f"\n### T1s[{c}] Spearman (FULL vocab) lift (real - null)\n" + HDR)
    for X in ("JV", "NV", "JVz"):
        for Cn in (f"C:{c}", f"Canti:{c}", f"Ccent:{c}"):
            if X == "JVz" and Cn != f"C:{c}":
                continue
            real, null = pairwise_full(X, Cn, "rho")
            tab = by_layer(real, null)
            OUT["tables"][f"rho|{X}->{Cn}"] = tab
            PER[f"rho|{X}->{Cn}"] = real
            lines.append(lift_str(f"rho {X}~{Cn}", tab))
    # paired differences
    for a, b in (("JV", "NV"), ("JVz", "JV"), ("JL", "NV")):
        for met in ("p10", "rho"):
            ka, kb = f"{met}|{a}->C:{c}", f"{met}|{b}->C:{c}"
            if ka not in PER or kb not in PER:
                continue
            d = PER[ka] - PER[kb]
            res = {}
            for l in LAYERS:
                idx = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
                res[str(l)] = boot(d[idx])
            for nm, ls in (("early", EARLY), ("late", LATE), ("all", LAYERS)):
                idx = [r for r, (ll, _, _) in enumerate(POLES) if ll in ls]
                res[nm] = boot(d[idx])
            for nm, ls in (("late_cluster", LATE), ("all_cluster", LAYERS), ("L40_45_cluster", [40, 45]), ("L50_57_cluster", [50, 55, 57])):
                res[nm] = boot_cluster(d, ls)
            OUT["paired"][f"{met}|{a}-{b}|{c}"] = res
            lines.append(f"paired {met} {a}-{b} [{c}]: " + "; ".join(f"{k}:{res[k]['lift']:+.3f}" for k in COLS[:-3]) +
                         f" | late {fmtl(res['late'])} (neuron-cluster CI {res['late_cluster']['ci'][0]:+.3f},{res['late_cluster']['ci'][1]:+.3f})"
                         f" | all {fmtl(res['all'])} (cluster {res['all_cluster']['ci'][0]:+.3f},{res['all_cluster']['ci'][1]:+.3f})"
                         f" | L40-45 {res['L40_45_cluster']['lift']:+.3f} (cluster {res['L40_45_cluster']['ci'][0]:+.3f},{res['L40_45_cluster']['ci'][1]:+.3f})"
                         f" | L50-57 {res['L50_57_cluster']['lift']:+.3f} (cluster {res['L50_57_cluster']['ci'][0]:+.3f},{res['L50_57_cluster']['ci'][1]:+.3f})")
    for sg in (1, -1):
        idx = [r for r, (ll, _, ss) in enumerate(POLES) if ss == sg and ll in LATE]
        rr, nn_ = pairwise_full("JV", f"C:{c}", "p10")
        OUT["tables"].setdefault("sign_split", {})[f"{c}|{sg}"] = boot(rr[idx], nn_[idx])
        lines.append(f"sign {sg:+d} late P@10 JV->C:{c}: {fmt(boot(rr[idx], nn_[idx]))}")
    # reliability + KL
    rel, kl = {}, {}
    for l in LAYERS:
        rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        rel[str(l)] = float(np.median([meta["causal"][c][r]["split_rho"] for r in rows]))
        kl[str(l)] = float(np.median([meta["causal"][c][r]["kl"] for r in rows]))
    fl = [m for l in LAYERS for m in causal[(c, l)]["poles"] if m["set"] == "floor"]
    OUT["reliability"][c] = rel
    OUT["kl"][c] = kl
    lines.append(f"split-half rho median [{c}]: " + ", ".join(f"L{l} {rel[str(l)]:.2f}" for l in LAYERS))
    ceil = {k: math.sqrt(max(2 * v / (1 + v), 0)) for k, v in rel.items()}
    OUT["reliability"][c + "_attenuation_ceiling"] = ceil
    tj = OUT["tables"][f"rho|JV->C:{c}"]
    lines.append(f"attenuation ceiling sqrt(SB(r)) [{c}]: " + ", ".join(f"L{l} {ceil[str(l)]:.2f}" for l in LAYERS) +
                 " | JV rho lift / ceiling: " + ", ".join(f"L{l} {tj[str(l)]['lift'] / ceil[str(l)]:.2f}" if ceil[str(l)] > 0 else "-" for l in LAYERS))
    lines.append(f"KL median [{c}]: " + ", ".join(f"L{l} {kl[str(l)]:.4f}" for l in LAYERS))
    if fl:
        OUT["notes"][f"floor_{c}"] = [{"kl": m["kl_mean"], "rho": m["split_rho"]} for m in fl]
        lines.append("floor (k=0.01) KL / split rho: " + ", ".join(f"{m['kl_mean']:.1e}/{m['split_rho']:.2f}" for m in fl))
print("\n".join(lines), flush=True)
TXT = lines[:]

# ------------------------------------------------------------------------------------------------ T1b bipolarity + generic
lines = ["\n### T4 bipolarity (output side): Spearman(c+, c-) real (same neuron) / null (other neurons' opposite pole); "
         "P@10 c(s)->top100(-c(-s))\n" + HDR]
for c in have_conds:
    real, null = pairwise_full(f"C:{c}", f"C:{c}", "rho", same_neuron_other_pole=True)
    tab = by_layer(real, null); OUT["bipolar"][f"rho|{c}"] = tab
    lines.append(row_str(f"rho(c+,c-) {c}", tab))
    real, null = pairwise_full(f"C:{c}", f"C:{c}", "p10", negY=True, same_neuron_other_pole=True)
    tab = by_layer(real, null); OUT["bipolar"][f"p10neg|{c}"] = tab
    lines.append(row_str(f"P@10 c->-c(-s) {c}", tab))
    # genericity: mean pairwise Spearman between DIFFERENT neurons' same-sign c (shared response)
    R = ranks_full(f"C:{c}")
    gen = {}
    for l in LAYERS:
        rows = [r for r, (ll, _, s) in enumerate(POLES) if ll == l and s == 1]
        rows_m = [r for r, (ll, _, s) in enumerate(POLES) if ll == l and s == -1]
        vals = [float(R[a] @ R[b]) for grp in (rows, rows_m) for i, a in enumerate(grp) for b in grp[i + 1:]]
        gen[str(l)] = float(np.mean(vals))
    OUT["bipolar"][f"generic_rho|{c}"] = gen
    lines.append(f"generic (between-neuron same-sign rho) {c}: " + ", ".join(f"L{l} {gen[str(l)]:+.2f}" for l in LAYERS))
print("\n".join(lines), flush=True)
TXT += lines

# ------------------------------------------------------------------------------------------------ T1c stability of causal labels across intervention styles
lines = ["\n### T1c causal-label stability across intervention styles: P@10 C:a -> C:b (same pole) real/null\n" + HDR]
pairs = [("A8k60", "A1k60"), ("A8k60", "A8k10"), ("A8k60", "L8k60"), ("A8k10", "L8k10"), ("L8k10", "L8k60"), ("A1k60", "L8k10")]
for a, b in pairs:
    if a in have_conds and b in have_conds:
        real, null = pairwise_full(f"C:{a}", f"C:{b}", "p10")
        tab = by_layer(real, null); OUT["stability"][f"p10|{a}->{b}"] = tab
        lines.append(row_str(f"{a}->{b}", tab))
        real, null = pairwise_full(f"C:{a}", f"C:{b}", "rho")
        tab = by_layer(real, null); OUT["stability"][f"rho|{a}->{b}"] = tab
        lines.append(row_str(f"rho {a}~{b}", tab))
print("\n".join(lines), flush=True)
TXT += lines

# ------------------------------------------------------------------------------------------------ T1d JV-NV agreement as confidence signal
lines = []
if "A8k60" in have_conds:
    tj, tn = topk_full("JV", 10), topk_full("NV", 10)
    agree = np.array([len(set(tj[r]) & set(tn[r])) / 10 for r in range(P)])
    pc = PER["p10|JV->C:A8k60"]
    def _rk(x):
        x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
        for v in np.unique(x):  # average ties
            m = x == v; r[m] = r[m].mean()
        return r
    class _R:
        def __init__(self, c): self.correlation = c
    def spearmanr(a, b):
        ra, rb = _rk(a), _rk(b)
        return _R(float(np.corrcoef(ra, rb)[0, 1]))
    def roc_auc_score(y, sc):
        y = np.asarray(y); r = _rk(sc); n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() + n1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    late = [r for r, (l, _, _) in enumerate(POLES) if l in LATE]
    wl = {}
    for l in LATE:
        idx = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        wl[str(l)] = float(spearmanr(agree[idx], pc[idx]).correlation) if np.std(agree[idx]) > 0 and np.std(pc[idx]) > 0 else None
    y = (pc[late] >= 0.3).astype(int)
    auc = float(roc_auc_score(y, agree[late])) if 0 < y.sum() < len(y) else None
    layer_auc = float(roc_auc_score((pc >= 0.3).astype(int), np.array([l >= 40 for l, _, _ in POLES]).astype(float))) \
        if 0 < (pc >= 0.3).sum() < P else None
    OUT["notes"]["jv_nv_agreement"] = {"within_layer_spearman_late": wl, "auc_late_p10ge0.3": auc,
                                       "auc_layer_ge40_all": layer_auc, "frac_late_p10ge0.3": float(y.mean())}
    lines.append(f"\nJV-NV top10 agreement vs P@10(JV->C:A8k60): within-layer rho {wl}; AUC(late, P@10>=0.3) {auc}; "
                 f"AUC(layer>=40, all) {layer_auc}")
print("\n".join(lines), flush=True)
TXT += lines

# ------------------------------------------------------------------------------------------------ T2 activation-based labels
S = np.load(f"{ROOT}/N1B_static.npz")
A = S["sub_ids"].astype(np.int64)
nA = len(A)
a_of = -np.ones(V, dtype=np.int64); a_of[A] = np.arange(nA)
act_files = sorted(glob.glob(f"{ROOT}/N1V_acts/part_*.npz"))
have_acts = len(act_files) == 10
lines = []
if have_acts:
    corpus = [json.loads(x) for x in open(f"{ROOT}/N1B_corpus.jsonl")]
    D = [np.load(p) for p in act_files]
    act = np.concatenate([d["act"] for d in D]).astype(np.float32)
    gg = np.concatenate([d["g"] for d in D]).astype(np.float32)
    uu = np.concatenate([d["u"] for d in D]).astype(np.float32)
    cid = np.concatenate([d["cid"] for d in D]); pos = np.concatenate([d["pos"] for d in D]).astype(np.int64)
    cur = np.array([corpus[c]["ids"][p] for c, p in zip(cid, pos)], dtype=np.int64)
    nxt = np.array([corpus[c]["ids"][p + 1] if p + 1 < len(corpus[c]["ids"]) else -1 for c, p in zip(cid, pos)], dtype=np.int64)
    half = np.array([zlib.crc32(corpus[c]["doc"].encode()) % 2 for c in cid], dtype=np.int8)
    keep = pos > 0
    act, gg, uu, cur, nxt, half = act[keep], gg[keep], uu[keep], cur[keep], nxt[keep], half[keep]
    N = len(act)
    ntop = int(math.ceil(0.01 * N))
    cur_a = a_of[cur]; nxt_a = np.where(nxt >= 0, a_of[np.maximum(nxt, 0)], -1)
    base_cur = np.bincount(cur_a[cur_a >= 0], minlength=nA).astype(np.float64)
    base_nxt = np.bincount(nxt_a[nxt_a >= 0], minlength=nA).astype(np.float64)
    Ncur, Nnxt = float(len(cur)), float((nxt >= 0).sum())
    HM = {h: np.where(half == h)[0] for h in (0, 1)}
    hb = {}
    for h in (0, 1):
        ia = cur_a[HM[h]]; na = nxt_a[HM[h]]
        hb[h] = (np.bincount(ia[ia >= 0], minlength=nA).astype(np.float64), float(len(HM[h])),
                 np.bincount(na[na >= 0], minlength=nA).astype(np.float64), float((nxt[HM[h]] >= 0).sum()))
    def enrich(cnt, n, base, nb, minc=3):
        e = np.log((cnt + 0.5) / max(n, 1)) - np.log((base + 0.5) / nb)
        return np.where(cnt >= minc, e, -np.inf).astype(np.float32)
    def toks_at(idx, arr_a):
        x = arr_a[idx]; x = x[x >= 0]
        return np.bincount(x, minlength=nA).astype(np.float64), len(x)
    LA = {}  # name -> [P, nA]
    def putA(name, row, vec):
        LA.setdefault(name, np.full((P, nA), -np.inf, dtype=np.float32))[row] = vec
    TOPPOS = {}
    mech = {"g_pos_frac_top30": [], "sign_act_eq_sign_u_top30": [], "weak_over_strong": []}
    col = {}
    ci = 0
    for l in LAYERS:
        for n in NEU[l]:
            col[(l, n)] = ci; ci += 1
    for row, (l, n, s) in enumerate(POLES):
        j = col[(l, n)]
        sa = s * act[:, j]
        top = np.argpartition(-sa, ntop)[:ntop]
        top = top[np.argsort(-sa[top])]
        top = top[sa[top] > 0]
        TOPPOS[row] = top
        cn, nn = toks_at(top, nxt_a); cc, nc = toks_at(top, cur_a)
        putA("Enx", row, enrich(cn, nn, base_nxt, Nnxt)); putA("Ein", row, enrich(cc, nc, base_cur, Ncur))
        for h in (0, 1):
            idx = HM[h]
            sh = sa[idx]
            nth = int(math.ceil(0.01 * len(idx)))
            th = idx[np.argpartition(-sh, nth)[:nth]]
            th = th[sa[th] > 0]
            bc, nbc, bn, nbn = hb[h]
            c_, n_ = toks_at(th, nxt_a); putA(f"Enx_h{h}", row, enrich(c_, n_, bn, nbn))
            c_, n_ = toks_at(th, cur_a); putA(f"Ein_h{h}", row, enrich(c_, n_, bc, nbc))
        t30 = top[:30]
        mech["g_pos_frac_top30"].append(float((gg[t30, j] > 0).mean()) if len(t30) else np.nan)
        mech["sign_act_eq_sign_u_top30"].append(float((np.sign(act[t30, j]) == np.sign(uu[t30, j])).mean()) if len(t30) else np.nan)
        if s == 1:
            mx, mn = act[:, j].max(), -act[:, j].min()
            mech["weak_over_strong"].append(float(min(mx, mn) / max(mx, mn)))
    for nm in [k for k in L if k in ("JV", "NV", "JL", "G", "Gemb", "Uemb") or k.startswith(("C:", "Canti:"))]:
        X = torch.nan_to_num(L[nm], nan=-float("inf")).numpy()[:, A]
        for row in range(P):
            putA(nm, row, X[row])
    OUT["notes"]["mechanism"] = {k: float(np.nanmean(v)) for k, v in mech.items()}
    OUT["notes"]["bipolar_frac_weak_ge0.3"] = float(np.mean(np.array(mech["weak_over_strong"]) >= 0.3))
    OUT["notes"]["n_positions"] = int(N)

    def topA(name, k):
        X = LA[name]
        o = np.argsort(-X, axis=1)[:, :k]
        return [o[r][np.isfinite(X[r, o[r]])] for r in range(P)]
    def pairA(X, Y, same_neuron_other_pole=False):
        tx, ty = topA(X, 10), topA(Y, 100)
        real, null = np.full(P, np.nan), np.full(P, np.nan)
        for r in range(P):
            if len(tx[r]) == 0:
                continue
            l, n, s = POLES[r]
            if same_neuron_other_pole:
                real[r] = p_at(tx[r], ty[POLES.index((l, n, -s))])
                null[r] = np.mean([p_at(tx[r], ty[q]) for q in partners(r, opposite=True)])
            else:
                real[r] = p_at(tx[r], ty[r])
                null[r] = np.mean([p_at(tx[r], ty[q]) for q in partners(r)])
        return real, null
    lines.append("\n### T2 activation-based labels (vocab A = 5544 attested tokens), P@10->100 real/null\n" + HDR)
    comps = [("Enx_h0", "Enx_h1", "ceiling next (half0->half1)"), ("Ein_h0", "Ein_h1", "ceiling input (half0->half1)"),
             ("JV", "Enx", "JV->Enx"), ("NV", "Enx", "NV->Enx"),
             ("JV", "Ein", "JV->Ein"), ("G", "Ein", "G(tool gate)->Ein"), ("Gemb", "Ein", "Gemb->Ein"), ("Uemb", "Ein", "Uemb->Ein"),
             ("G", "JV", "G->JV same neuron")]
    for c in have_conds:
        comps += [(f"C:{c}", "Enx", f"C:{c}->Enx (say)"), (f"C:{c}", "JV", f"C:{c}->JV (A)")]
    if "A8k60" in have_conds:
        comps += [("Canti:A8k60", "Enx", "Canti:A8k60->Enx"), ("Enx", "C:A8k60", "Enx->C:A8k60"), ("Enx", "JV", "Enx->JV")]
    for X, Y, nm in comps:
        real, null = pairA(X, Y)
        tab = by_layer(real, null); OUT["tables"][f"A|{nm}"] = tab
        lines.append(row_str(nm, tab))
    # compact agreement matrix on vocab A (primary causal condition), early / late / all
    MAT = {}
    cprim = "A8k60" if "A8k60" in have_conds else (have_conds[0] if have_conds else None)
    names = ["JV", "NV"] + ([f"C:{cprim}"] if cprim else []) + ["Enx", "Ein", "G"]
    for X in names:
        for Y in names:
            if X == Y or Y == "G":
                continue
            real, null = pairA(X, Y)
            MAT[f"{X}->{Y}"] = by_layer(real, null)
    OUT["matrix_A"] = MAT
    lines.append(f"\n### T2m agreement matrix on vocab A (P@10 X top-10 in Y top-100), real/null [early | late | all]; causal = {cprim}")
    lines.append("| X \\ Y | " + " | ".join(n for n in names if n != "G") + " |\n|" + "---|" * len([n for n in names if n != "G"]) + "---|")
    for X in names:
        cells = []
        for Y in names:
            if Y == "G":
                continue
            if X == Y:
                cells.append("-"); continue
            t = MAT[f"{X}->{Y}"]
            cells.append(" / ".join(fmt(t[b]) for b in ("early", "late", "all")))
        lines.append(f"| {X} | " + " | ".join(cells) + " |")
    for X, nm in (("Ein", "Ein(s)->Ein(-s)"), ("Enx", "Enx(s)->Enx(-s)")):
        real, null = pairA(X, X, same_neuron_other_pole=True)
        tab = by_layer(real, null); OUT["tables"][f"A|{nm}"] = tab
        lines.append(row_str(nm, tab))

    # ALIGN_cur for gate labels: percentile of current token within label ranking over A at top positions minus all positions
    def align(name, use_next=False):
        X = LA[name]
        arr = nxt_a if use_next else cur_a
        okpos = arr >= 0
        real, null = np.full(P, np.nan), np.full(P, np.nan)
        pct = {}
        for r in range(P):
            x = np.where(np.isfinite(X[r]), X[r], -1e30)
            rk = np.empty(nA); rk[np.argsort(x, kind="mergesort")] = np.arange(nA) / (nA - 1) * 100
            pct[r] = rk
        heldout = name.endswith("_h0")  # split-half ceiling: label from half 0, evaluated on half-1 positions only
        if heldout:
            okpos = okpos & (half == 1)
        for r in range(P):
            top = TOPPOS[r]
            if heldout:
                top = top[half[top] == 1]
            t = arr[top]; t = t[t >= 0]
            if len(t) == 0:
                continue
            f = lambda q: pct[q][t].mean() - pct[q][arr[okpos]].mean()
            real[r] = f(r)
            null[r] = np.mean([f(q) for q in partners(r)])
        return real, null
    lines.append("\n### T2b ALIGN (percentile-point lift of the actual token at firing positions), real/null\n" + HDR)
    for nm, nxtflag in (("G", False), ("Gemb", False), ("Ein_h0", False)) + tuple((f"C:{c}", True) for c in have_conds[:2]) + (("Enx_h0", True),):
        real, null = align(nm, nxtflag)
        tab = by_layer(real, null); OUT["tables"][f"ALIGN|{nm}|{'next' if nxtflag else 'cur'}"] = tab
        lines.append(f"| ALIGN_{'next' if nxtflag else 'cur'} {nm} | " + " | ".join(
            f"{tab[c]['real']:+.1f}/{tab[c]['null']:+.1f}" + ("*" if (tab[c]['ci'][0] > 0 or tab[c]['ci'][1] < 0) else "") for c in COLS) + " |")
    lines.append(f"mechanism: {OUT['notes']['mechanism']}, frac neurons weak/strong>=0.3: {OUT['notes']['bipolar_frac_weak_ge0.3']:.2f}")
else:
    lines.append("\n(T2 skipped: N1V_acts incomplete)")
print("\n".join(lines), flush=True)
TXT += lines

# ------------------------------------------------------------------------------------------------ T5 tightness
lines = []
try:
    from safetensors import safe_open
    SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/")
    SNAP += sorted(os.listdir(SNAP))[0] + "/model.safetensors"
    with safe_open(SNAP, "pt") as f:
        WE = f.get_tensor("model.language_model.embed_tokens.weight")
    mu = torch.zeros(WE.shape[1])
    for s0 in range(0, WE.shape[0], 32768):
        mu += WE[s0:s0 + 32768].float().sum(0)
    mu /= WE.shape[0]
    def tight(ids_list):
        out = []
        for ids in ids_list:
            ids = [int(t) for t in ids]
            E = WE[ids].float() - mu
            E = E / E.norm(dim=-1, keepdim=True)
            G_ = E @ E.T
            k = len(ids)
            out.append(float((G_.sum() - k) / (k * (k - 1))))
        return np.array(out)
    ND = json.load(open(f"{CACHE}/null_dirs_top10.json"))
    T5 = {}
    tj, tn = topk_full("JV", 10), topk_full("NV", 10)
    rng = np.random.default_rng(5)
    def bd(x, y):
        b = x[rng.integers(0, len(x), (2000, len(x)))].mean(1) - y[rng.integers(0, len(y), (2000, len(y)))].mean(1)
        return {"diff": float(x.mean() - y.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
    band = {"early": {k: [] for k in ("JV", "NV", "mixJV", "mixNV", "singJV", "isoJV")}, "late": {k: [] for k in ("JV", "NV", "mixJV", "mixNV", "singJV", "isoJV")}}
    for l in LAYERS:
        rows = [r for r, (ll, _, s) in enumerate(POLES) if ll == l and s == 1]  # + poles (- pole label = bottom tokens; use +)
        rows_all = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
        d = {"JV": tight([tj[r] for r in rows_all]), "NV": tight([tn[r] for r in rows_all]),
             "mixJV": tight(ND[str(l)]["mix512_jpp"]), "mixNV": tight(ND[str(l)]["mix512_naive"]),
             "singJV": tight(ND[str(l)]["single_jpp"]), "isoJV": tight(ND[str(l)]["iso_jpp"])}
        T5[str(l)] = {k: float(v.mean()) for k, v in d.items()}
        b = "early" if l in EARLY else "late"
        for k in d:
            band[b][k].append(d[k])
    for b in band:
        dd = {k: np.concatenate(v) for k, v in band[b].items()}
        T5[b] = {"JV": float(dd["JV"].mean()), "mixJV": float(dd["mixJV"].mean()), "NV": float(dd["NV"].mean()),
                 "JV-mixJV": bd(dd["JV"], dd["mixJV"]), "NV-mixNV": bd(dd["NV"], dd["mixNV"]), "JV-NV": bd(dd["JV"], dd["NV"]),
                 "JV-singJV": bd(dd["JV"], dd["singJV"])}
    OUT["tightness"] = T5
    lines.append("\n### T5 tightness (mean pairwise centred-embedding cosine of top-10)")
    for l in LAYERS:
        lines.append(f"L{l}: " + ", ".join(f"{k} {v:.3f}" for k, v in T5[str(l)].items()))
    for b in ("early", "late"):
        lines.append(f"{b}: JV {T5[b]['JV']:.3f} vs mix {T5[b]['mixJV']:.3f}; JV-mix {T5[b]['JV-mixJV']}; NV-mixNV {T5[b]['NV-mixNV']}; JV-NV {T5[b]['JV-NV']}")
except Exception as e:
    lines.append(f"T5 failed: {e!r}")
print("\n".join(lines), flush=True)
TXT += lines

# ------------------------------------------------------------------------------------------------ examples
ex = []
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
dec = lambda t: tok.decode([int(t)])
for row, (l, n, s) in enumerate(POLES):
    if n not in NEU[l][:2]:
        continue
    e = {"l": l, "n": n, "s": s, "JV": [dec(t) for t in topk_full("JV", 10)[row][:8]], "NV": [dec(t) for t in topk_full("NV", 10)[row][:8]]}
    for c in have_conds[:3]:
        e[f"C:{c}"] = [dec(t) for t in topk_full(f"C:{c}", 10)[row][:8]]
    if have_acts:
        o = np.argsort(-LA["Ein"][row])[:8]; e["Ein"] = [dec(A[t]) for t in o if np.isfinite(LA["Ein"][row][t])]
        o = np.argsort(-LA["Enx"][row])[:8]; e["Enx"] = [dec(A[t]) for t in o if np.isfinite(LA["Enx"][row][t])]
    ex.append(e)
OUT["examples"] = ex
OUT["have_conds"] = have_conds
OUT["have_acts"] = have_acts
json.dump(OUT, open(f"{ROOT}/N1V_analysis.json", "w"), indent=1)
open(f"{ROOT}/N1V_analysis_tables.txt", "w").write("\n".join(TXT))
print(f"saved {time.time()-t0:.0f}s")
