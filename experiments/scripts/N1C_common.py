# N1C_common -- shared loaders/metrics for the N1C integration (CPU only, run with .venv/bin/python).
# Inputs: N1A_cache/*.pt (static full-vocab logits, iso prior, fp32 causal vectors), N1B_acts/*.npz + N1B_corpus.jsonl
# (activations of the same 192 neurons over a 147.5k-position corpus), N1B_static.npz (gate / up / output static labels
# on the attested sub-vocabulary = tokens with corpus count >= 3).
import json, math, os, glob, zlib
from collections import Counter
import numpy as np

ROOT = "./experiments/results"
NEU = json.load(open(f"{ROOT}/N1_neurons.json"))
LAYERS = [int(l) for l in NEU["layers"]]
NEURONS = {int(l): [int(i) for i in v] for l, v in NEU["neurons"].items()}
NPL = 24
# pole order (shared by N1A and N1B): layer-major, neuron, then sign +1, -1
POLES = [(l, n, s) for l in LAYERS for n in NEURONS[l] for s in (1, -1)]
P = len(POLES)  # 384
EARLY = [10, 20, 30]
LATE = [40, 45, 50, 55, 57]

_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
    return _tok

_dc = {}
def dec(t):
    t = int(t)
    if t not in _dc:
        _dc[t] = tok().decode([t]) if t >= 0 else "<END>"
    return _dc[t]


def rank_std(X):
    """Row-wise average ranks (ties averaged), centred and L2-normalised -> Spearman = dot product. X [R, V] float."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    for r in range(X.shape[0]):
        x = X[r]
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        _, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
        ends = np.cumsum(cnt).astype(np.float64)
        avg = ends - (cnt - 1) / 2.0
        rk = np.empty_like(x)
        rk[order] = avg[inv]
        rk -= rk.mean()
        out[r] = rk / (np.linalg.norm(rk) + 1e-12)
    return out


def boot_pair(real, null, nb=2000, seed=0):
    """Bootstrap over units (poles): mean real, mean null, diff (= lift as real-null) and ratio (real/null) CIs."""
    real = np.asarray(real, float); null = np.asarray(null, float)
    ok = np.isfinite(real) & np.isfinite(null)
    real, null = real[ok], null[ok]
    k = len(real)
    if k == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(nb, k))
    br, bn = real[idx].mean(1), null[idx].mean(1)
    bd = br - bn
    with np.errstate(divide="ignore", invalid="ignore"):
        bq = br / bn
    q = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
    mr, mn = float(real.mean()), float(null.mean())
    fin = np.isfinite(bq)
    return {"n": k, "real": mr, "real_ci": q(br), "null": mn, "null_ci": q(bn), "diff": mr - mn, "diff_ci": q(bd),
            "ratio": (mr / mn) if mn > 0 else None, "ratio_ci": q(bq[fin]) if fin.mean() > 0.95 else None}


def boot_mean(x, nb=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(nb, len(x)))].mean(1)
    return {"n": len(x), "mean": float(x.mean()), "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def load_corpus_acts():
    """Returns dict with act [N,192] (non-BOS), cur, nxt (-1 = unknown), half (doc split), src, cid, pos, corpus."""
    corpus = [json.loads(x) for x in open(f"{ROOT}/N1B_corpus.jsonl")]
    D = [np.load(p) for p in sorted(glob.glob(f"{ROOT}/N1B_acts/part_*.npz"))]
    act = np.concatenate([d["act"] for d in D]).astype(np.float32)
    cid = np.concatenate([d["cid"] for d in D])
    pos = np.concatenate([d["pos"] for d in D]).astype(np.int64)
    cur = np.array([corpus[c]["ids"][p] for c, p in zip(cid, pos)], dtype=np.int64)
    nxt = np.array([corpus[c]["ids"][p + 1] if p + 1 < len(corpus[c]["ids"]) else -1 for c, p in zip(cid, pos)], dtype=np.int64)
    half = np.array([zlib.crc32(corpus[c]["doc"].encode()) % 2 for c in cid], dtype=np.int8)
    src = np.array([corpus[c]["src"] for c in cid])
    keep = pos > 0
    return {"act": act[keep], "cid": cid[keep], "pos": pos[keep], "cur": cur[keep], "nxt": nxt[keep],
            "half": half[keep], "src": src[keep], "corpus": corpus}


def context_str(corpus, c, p, w=14, wr=3):
    ids = corpus[c]["ids"]
    left = "".join(dec(t) for t in ids[max(1, p - w + 1):p])
    right = "".join(dec(t) for t in ids[p + 1:p + 1 + wr])
    return (left + "[[" + dec(ids[p]) + "]]" + right).replace("\n", "\\n")
