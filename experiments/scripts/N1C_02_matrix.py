#!/usr/bin/env python3
# N1C_02_matrix -- agreement matrix between all label types, with permutation nulls, lifts and bootstrap CIs.
# Run: .venv/bin/python experiments/scripts/N1C_02_matrix.py  -> experiments/results/N1C_matrix.json
#
# METRIC DEFINITIONS
#  P@10->100 (X -> Y) for pole p: |top10(X_p) & top100(Y_p)| / k, k = number of rankable tokens in top10(X_p) (<=10).
#     On vocab A (attested, 5544 tokens) unless marked FULL (262144 tokens; static/causal only). Tokens with -inf
#     score (e.g. empirical tokens seen < 3 times in the pole's top set) are never in a top set, so an empirical
#     "top-100" can be shorter than 100.
#  rho (X, Y): Spearman over W5 tokens (3471 tokens of A seen >= 5x as current and next token); continuous labels only.
#  NULL: the same metric for pole p paired with the label of every OTHER sampled neuron of the same layer and the
#     same sign (23 pairs; exhaustive within-layer permutation null), averaged. For cross-pole metrics (Q4) the null
#     partner is the opposite-sign pole of another neuron. lift = real - null (diff) and real/null (ratio).
#  CIs: 2000 bootstrap resamples over poles. perm_p: 2000 within-layer neuron relabelings (sign kept), one-sided.
#
# PRE-STATED PREDICTIONS (written before running this script; N1A/N1B summaries were known):
#  H1 (Q1) JV->C and NV->C lifts are ~0 at L10-30 and large (>0.3 on FULL) at L40-57; JV - NV ~ 0; prior-z (JVz) lower
#     than JV for top-k precision. On vocab A the same ordering holds.
#  H2 (Q2) C->Enx lift is clearly > 0 at L45-57 (the late neurons are partly 'say' neurons) and ~0 at L10-30; C->Enx
#     exceeds JV->Enx (causal is the better proxy for what follows) but stays far below the split-half ceiling
#     Enx_h0->Enx_h1, because next tokens are mostly determined by context, not by one neuron.
#  H3 (Q3) G->Ein lift is tiny at every layer (< 0.02 diff on A) and below Gemb->Ein at L10; G overlaps JV (same
#     neuron) above null at late layers because both inherit the J++ output prior (shared-prior artifact).
#  H4 (Q4) Input side: Ein_+ vs Ein_- overlap near null (distinct triggers). Output side: C_+ top-10 lies in the top-100
#     of -C_- far above null at L40-57 (antipodal effect) but not at L10-30. Self (C_s -> Enx_s) > cross (C_s -> Enx_-s)
#     at late layers, i.e. each pole's 'say' content is polarity-specific.
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

t0 = time.time()
Z = np.load(f"{ROOT}/N1C_labels.npz")
A, W5 = Z["A"], Z["W5"]
nA = len(A)
LAB = {k[4:]: Z[k] for k in Z.files if k.startswith("lab_")}
TOPF = {k[4:]: Z[k] for k in Z.files if k.startswith("top_")}
LAB["negC"] = -LAB["C"]
TOPF_NEG = None
NB = 2000
rng = np.random.default_rng(7)
LROWS = {l: np.array([r for r, (ll, n, s) in enumerate(POLES) if ll == l]) for l in LAYERS}
CONT = ["JV", "JL", "NV", "JVz", "NVz", "C", "CB", "G", "Gpre", "Gemb", "U", "Rnx", "Rin"]


def topk_valid(X, k):
    """X [R, nA] -> list of index arrays (only finite entries)."""
    out = []
    for r in range(X.shape[0]):
        x = X[r]
        nv = int(np.isfinite(x).sum())
        kk = min(k, nv)
        if kk == 0:
            out.append(np.array([], dtype=np.int64)); continue
        idx = np.argpartition(-np.where(np.isfinite(x), x, -1e30), kk - 1)[:kk]
        out.append(idx[np.argsort(-x[idx])])
    return out


def pmat_A(X, Y, kx=10, ky=100):
    """[R, R] matrix M[p, q] = |top10(X_p) & top100(Y_q)| / k_p on vocab A."""
    tx, ty = topk_valid(X, kx), topk_valid(Y, ky)
    Yb = np.zeros((Y.shape[0], nA), dtype=np.float32)
    for q, t in enumerate(ty):
        Yb[q, t] = 1
    M = np.full((X.shape[0], Y.shape[0]), np.nan)
    for p, t in enumerate(tx):
        if len(t):
            M[p] = Yb[:, t].sum(1) / len(t)
    return M


def pmat_full(nx, ny, rows, kx=10, ky=100, negy=False):
    tx = TOPF[nx][rows][:, :kx]
    ty = TOPF[ny][rows][:, :ky]
    M = np.zeros((len(rows), len(rows)))
    sets = [set(t.tolist()) for t in ty]
    for p in range(len(rows)):
        for q in range(len(rows)):
            M[p, q] = len(sets[q].intersection(tx[p].tolist())) / kx
    return M


def rmat(X, Y):
    RX = rank_std(X[:, W5]); RY = rank_std(Y[:, W5])
    return RX @ RY.T


def pairing(rows, scheme):
    """real partner index and null partner mask within a layer block (local indices)."""
    R = len(rows)
    nn = np.array([POLES[r][1] for r in rows]); sg = np.array([POLES[r][2] for r in rows])
    real = np.arange(R) if scheme == "same" else np.array([p + 1 if sg[p] == 1 else p - 1 for p in range(R)])
    tgt_sign = sg if scheme == "same" else -sg
    mask = (sg[None, :] == tgt_sign[:, None]) & (nn[None, :] != nn[:, None])
    return real, mask


def layer_stats(M, scheme, rows):
    real_idx, mask = pairing(rows, scheme)
    R = len(rows)
    real = M[np.arange(R), real_idx]
    Mm = np.where(mask, M, np.nan)
    null = np.nanmean(Mm, 1)
    ok = np.isfinite(real)
    # permutation: relabel neurons (keep sign)
    obs = np.nanmean(real)
    nneu = R // 2
    cnt = 0
    for _ in range(NB):
        perm = rng.permutation(nneu)
        q = 2 * perm[np.arange(R) // 2] + (real_idx % 2)
        if np.nanmean(M[np.arange(R), q]) >= obs:
            cnt += 1
    return real, null, (cnt + 1) / (NB + 1)


def summarize(Ms, scheme, rowsets):
    out = {"per_layer": {}}
    allr, alln, er, en, lr, ln = [], [], [], [], [], []
    for l, M in Ms.items():
        real, null, pp = layer_stats(M, scheme, rowsets[l])
        b = boot_pair(real, null, NB, seed=l)
        b["perm_p"] = pp
        out["per_layer"][str(l)] = b
        allr.append(real); alln.append(null)
        (er if l in EARLY else lr).append(real); (en if l in EARLY else ln).append(null)
    out["overall"] = boot_pair(np.concatenate(allr), np.concatenate(alln), NB, 1)
    out["early_L10_30"] = boot_pair(np.concatenate(er), np.concatenate(en), NB, 2)
    out["late_L40_57"] = boot_pair(np.concatenate(lr), np.concatenate(ln), NB, 3)
    out["_per_pole_real"] = np.concatenate(allr).tolist()
    out["_per_pole_null"] = np.concatenate(alln).tolist()
    return out


def run_A(nx, ny, scheme="same", metric="p10", sign=None):
    Ms, rowsets = {}, {}
    for l in LAYERS:
        rows = LROWS[l]
        X, Y = LAB[nx][rows], LAB[ny][rows]
        Ms[l] = pmat_A(X, Y) if metric == "p10" else rmat(X, Y)
        rowsets[l] = rows
    res = summarize(Ms, scheme, rowsets)
    if sign is None:
        # per-sign breakdown (overall)
        r = np.array(res["_per_pole_real"]); n = np.array(res["_per_pole_null"])
        sg = np.array([POLES[x][2] for l in LAYERS for x in LROWS[l]])
        lay = np.array([POLES[x][0] for l in LAYERS for x in LROWS[l]])
        late = np.isin(lay, LATE)
        res["by_sign"] = {"+": boot_pair(r[sg == 1], n[sg == 1]), "-": boot_pair(r[sg == -1], n[sg == -1]),
                          "late_+": boot_pair(r[(sg == 1) & late], n[(sg == 1) & late]),
                          "late_-": boot_pair(r[(sg == -1) & late], n[(sg == -1) & late])}
    return res


def run_full(nx, ny):
    Ms, rowsets = {}, {}
    for l in LAYERS:
        rows = LROWS[l]
        Ms[l] = pmat_full(nx, ny, rows)
        rowsets[l] = rows
    return summarize(Ms, "same", rowsets)


R = {"definitions": {
    "P@10->100": "|top10(X) & top100(Y)|/k (k = rankable tokens in top10(X)), per pole; vocab A = 5544 attested tokens (corpus count>=3) unless FULL",
    "rho": "Spearman over W5 (3471 tokens seen >=5x as current and next token)",
    "null": "same metric with Y from each other sampled neuron of the same layer, same sign (cross-pole: opposite sign), 23 pairs averaged",
    "diff": "real - null (lift), bootstrap CI over poles", "ratio": "mean real / mean null",
    "perm_p": "2000 within-layer neuron relabelings, one-sided"},
    "matrix_A_p10": {}, "matrix_rho": {}, "full_p10": {}, "q": {}}

# ------------------------------------------------ 1. full agreement matrix on A (P@10->100), rows X -> cols Y
MAT = ["JV", "JL", "NV", "JVz", "NVz", "C", "Enx", "EnxC", "Ein", "EinC", "G", "Gemb", "U", "Rnx", "Rin"]
for x in MAT:
    for y in MAT:
        if x == y:
            continue
        R["matrix_A_p10"].setdefault(x, {})[y] = run_A(x, y)
    print("row", x, f"{time.time()-t0:.0f}s", flush=True)
# ------------------------------------------------ 2. Spearman matrix (continuous labels) on W5
for i, x in enumerate(CONT):
    for y in CONT[i + 1:]:
        R["matrix_rho"].setdefault(x, {})[y] = run_A(x, y, metric="rho")
print("rho matrix", f"{time.time()-t0:.0f}s", flush=True)
# ------------------------------------------------ 3. FULL vocab static <-> causal
for x in ["JV", "JL", "NV", "JVz", "NVz", "JVzn", "NVzn", "CB"]:
    R["full_p10"][f"{x}->C"] = run_full(x, "C")
    R["full_p10"][f"C->{x}"] = run_full("C", x)
R["full_p10"]["JV->NV"] = run_full("JV", "NV")
R["full_p10"]["JV->CB"] = run_full("JV", "CB")
print("full", f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------ 4. question-specific extras
Q = R["q"]
# Q1 paired differences (FULL and A) between label methods vs causal
def paired(resA, resB):
    a = np.array(resA["_per_pole_real"]); b = np.array(resB["_per_pole_real"])
    lay = np.array([POLES[x][0] for l in LAYERS for x in LROWS[l]])
    out = {"overall": boot_mean(a - b)}
    for l in LAYERS:
        out[str(l)] = boot_mean((a - b)[lay == l])
    out["late"] = boot_mean((a - b)[np.isin(lay, LATE)])
    out["early"] = boot_mean((a - b)[np.isin(lay, EARLY)])
    return out
F_ = R["full_p10"]
Q["Q1_paired_full"] = {"JV-NV": paired(F_["JV->C"], F_["NV->C"]), "JV-JL": paired(F_["JV->C"], F_["JL->C"]),
                       "JVz-JV": paired(F_["JVz->C"], F_["JV->C"]), "NVz-NV": paired(F_["NVz->C"], F_["NV->C"]),
                       "JVzn-JV": paired(F_["JVzn->C"], F_["JV->C"])}
M_ = R["matrix_A_p10"]
Q["Q1_paired_A"] = {"JV-NV": paired(M_["JV"]["C"], M_["NV"]["C"]), "JVz-JV": paired(M_["JVz"]["C"], M_["JV"]["C"])}
# Q2: say-neuron tests
Q["Q2"] = {"C->Enx01": run_A("C", "Enx01"), "JV->Enx01": run_A("JV", "Enx01"), "NV->Enx01": run_A("NV", "Enx01"),
           "CB->Enx": run_A("CB", "Enx"),
           "ceiling_Enx_h0->Enx_h1": run_A("Enx_h0", "Enx_h1"), "ceiling_Ein_h0->Ein_h1": run_A("Ein_h0", "Ein_h1"),
           "C-JV->Enx_paired": paired(M_["C"]["Enx"], M_["JV"]["Enx"])}
# fraction of 'say' poles: >= 2 of causal top-10 (A) inside the empirical next-token top-100
cr = np.array(M_["C"]["Enx"]["_per_pole_real"]); cn = np.array(M_["C"]["Enx"]["_per_pole_null"])
lay = np.array([POLES[x][0] for l in LAYERS for x in LROWS[l]])
say = {}
for l in LAYERS:
    rows = LROWS[l]
    M = pmat_A(LAB["C"][rows], LAB["Enx"][rows])
    real_idx, mask = pairing(rows, "same")
    real = M[np.arange(len(rows)), real_idx]
    nullfrac = np.nanmean(np.where(mask, (M >= 0.2).astype(float), np.nan))
    k = int((real >= 0.2).sum())
    say[str(l)] = {"frac": k / len(rows), "ci": wilson(k, len(rows)), "null_frac": float(nullfrac)}
Q["Q2_say_fraction_ge2of10"] = say
# Q3: gate label
Q["Q3"] = {"G->EinC": M_["G"]["EinC"], "G->Ein01": run_A("G", "Ein01"), "Gpre->Ein": run_A("Gpre", "Ein"),
           "Gnaive->Ein": run_A("Gnaive", "Ein"), "Gz->Ein": run_A("Gz", "Ein"), "Gz->EinC": run_A("Gz", "EinC"),
           "Gemb->EinC": run_A("Gemb", "EinC"), "G->JV": M_["G"]["JV"], "JV->G": M_["JV"]["G"],
           "G->C": M_["G"]["C"], "G->Enx": M_["G"]["Enx"], "rho_G_Rin": R["matrix_rho"]["G"]["Rin"],
           "rho_Gemb_Rin": R["matrix_rho"]["Gemb"]["Rin"], "rho_U_Rin": R["matrix_rho"]["U"]["Rin"]}
# Q4: bipolarity (cross-pole pairing: partner = the other pole of the same neuron)
Q["Q4"] = {"Ein_s->Ein_-s": run_A("Ein", "Ein", "cross"), "Enx_s->Enx_-s": run_A("Enx", "Enx", "cross"),
           "C_s->C_-s": run_A("C", "C", "cross"), "C_s->(-C_-s)": run_A("C", "negC", "cross"),
           "C_s->Enx_-s(cross)": run_A("C", "Enx", "cross"),
           "rho_C_s_C_-s": run_A("C", "C", "cross", "rho"), "rho_Rin_s_Rin_-s": run_A("Rin", "Rin", "cross", "rho"),
           "rho_Rnx_s_Rnx_-s": run_A("Rnx", "Rnx", "cross", "rho"),
           "rho_C_s_Rnx_s": R["matrix_rho"]["C"]["Rnx"], "rho_C_s_Rnx_-s": run_A("C", "Rnx", "cross", "rho")}
Q["Q4"]["self_minus_cross_C->Enx"] = paired(M_["C"]["Enx"], Q["Q4"]["C_s->Enx_-s(cross)"])
Q["Q4"]["self_minus_cross_rho_C_Rnx"] = paired(R["matrix_rho"]["C"]["Rnx"], Q["Q4"]["rho_C_s_Rnx_-s"])

# ------------------------------------------------ 5. static-only confidence predictor for label validity (Q1 follow-up)
feat = {"jv_nv_top10_overlap": [], "nv_top10_zn_mean": [], "jv_top10_z_mean": [], "jv_top10_zn_mean": []}
for l in LAYERS:
    st = torch.load(f"{ROOT}/N1A_cache/static_L{l}.pt"); pri = torch.load(f"{ROOT}/N1A_cache/prior_L{l}.pt")
    for r in LROWS[l]:
        _, n, s = POLES[r]
        ni = NEURONS[l].index(n)
        jv = s * st["jpp"][ni].float(); nv = s * st["naive"][ni].float()
        feat["jv_nv_top10_overlap"].append(len(set(jv.topk(10).indices.tolist()) & set(nv.topk(10).indices.tolist())) / 10)
        feat["nv_top10_zn_mean"].append(float((nv / pri["neu"]["naive"]).topk(10).values.mean()))
        mu, sd = pri["iso"]["jpp"]
        feat["jv_top10_z_mean"].append(float(((jv - mu) / sd).topk(10).values.mean()))
        feat["jv_top10_zn_mean"].append(float((jv / pri["neu"]["jpp"]).topk(10).values.mean()))
target = np.array(F_["JV->C"]["_per_pole_real"])
def auc(score, y):
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
pred = {}
for k, v in feat.items():
    v = np.array(v)
    late = np.isin(lay, LATE)
    within = []
    for l in LAYERS:
        m = lay == l
        rv = rank_std(v[m][None])[0]; rt = rank_std(target[m][None])[0]
        within.append(float(rv @ rt))
    pred[k] = {"spearman_within_layer_mean": float(np.mean(within)), "per_layer": dict(zip(map(str, LAYERS), within)),
               "spearman_pooled": float(rank_std(v[None])[0] @ rank_std(target[None])[0]),
               "auc_late_target_p10_ge_0.3": auc(v[late], target[late] >= 0.3)}
pred["layer_only_auc_pooled_ge_0.3"] = auc(lay.astype(float), target >= 0.3)
Q["static_confidence_predictor"] = pred

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o
json.dump(strip(R), open(f"{ROOT}/N1C_matrix.json", "w"), indent=1)
# keep per-pole vectors for the audit / report
pp = {"poles": POLES}
for x in ["JV", "NV", "JVz", "G", "Gemb", "U"]:
    for y in ["C", "Enx", "Ein"]:
        if x != y and y in M_.get(x, {}):
            pp[f"{x}->{y}"] = M_[x][y]["_per_pole_real"]
pp["C->Enx"] = M_["C"]["Enx"]["_per_pole_real"]; pp["full_JV->C"] = F_["JV->C"]["_per_pole_real"]
pp["full_NV->C"] = F_["NV->C"]["_per_pole_real"]
json.dump(pp, open(f"{ROOT}/N1C_per_pole.json", "w"))
print("saved", f"{time.time()-t0:.0f}s")
