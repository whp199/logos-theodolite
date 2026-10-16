#!/usr/bin/env python3
"""V5_binding_dissociation_05: lens-free raw-residual geometry from variant-averaged residual means (layers LSAVE).
For each position type t and entity e in {A,U,T}: d_e(t) = item-mean of (mean over cells e=W - mean over cells e=M)/2.
Noise-corrected cosine between two directions via random item split-halves (300 splits):
  ccos(x, y) = mean cos(x^h1, y^h2) (both orders) / sqrt(mean cos(x^h1, x^h2) * mean cos(y^h1, y^h2)).
Unbiased squared norm |d|^2 ~ mean(d^h1 . d^h2) (so norm ratios are not inflated by noise).
Coordinates: raw, and diag-standardised by the per-dim token std from _01 (floored at 0.1 x median).
Question: is the gender of the QUOTED referent written along the same direction as the assistant's gender at its own
"I" (shared 'referent-gender' direction), and is the default (turn-speaker) entity's gender present and where?
"""
import json
import numpy as np

ROOT = "."
RES = ROOT + "/experiments/results"
exec(open(ROOT + "/experiments/scripts/V5_binding_dissociation_common.py").read())
Z = np.load(f"{RES}/V5_binding_dissociation_factorial_chunk0.npz")
TY = list(Z["types"]); CELLN = list(Z["cells"]); LS = [int(x) for x in Z["lsave"]]
R = np.concatenate([np.load(f"{CACHE}/resid_means_chunk{c}.npy", mmap_mode="r") for c in range(3)], 0)  # [24, 8, nT, nL, D]
NP = R.shape[0]
sd = np.load(f"{CACHE}/quiet_sd.npy")[LS]  # [nL, D]
sdf = np.maximum(sd, 0.1 * np.median(sd, axis=1, keepdims=True))
IDX = {e: ([i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "W"], [i for i, c in enumerate(CELLN) if c["AUT".index(e)] == "M"]) for e in "AUT"}
rng = np.random.default_rng(5)
NS = 300
splits = []
for _ in range(NS):
    p = rng.permutation(NP); splits.append((p[:NP // 2], p[NP // 2:]))


def per_item_dir(t, e):
    ti = TY.index(t)
    X = np.asarray(R[:, :, ti], dtype=np.float32)  # [NP, 8, nL, D]
    w, m = IDX[e]
    return (X[:, w].mean(1) - X[:, m].mean(1)) / 2  # [NP, nL, D]


cache = {}
def get(t, e, std):
    k = (t, e, std)
    if k not in cache:
        d = per_item_dir(t, e)
        cache[k] = d / sdf[None] if std else d
    return cache[k]


def cosv(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def ccos(x, y):
    """x, y: [NP, nL, D] per-item directions -> corrected cos per layer, plus raw split-half reliabilities."""
    num, rx, ry = [], [], []
    for h1, h2 in splits:
        x1, x2, y1, y2 = x[h1].mean(0), x[h2].mean(0), y[h1].mean(0), y[h2].mean(0)
        num.append((cosv(x1, y2) + cosv(x2, y1)) / 2); rx.append(cosv(x1, x2)); ry.append(cosv(y1, y2))
    num, rx, ry = np.mean(num, 0), np.mean(rx, 0), np.mean(ry, 0)
    den = np.sqrt(np.clip(rx, 1e-6, None) * np.clip(ry, 1e-6, None))
    return num / den, rx, ry


def unorm2(x):
    v = []
    for h1, h2 in splits[:100]:
        v.append((x[h1].mean(0) * x[h2].mean(0)).sum(-1))
    return np.mean(v, 0)


REF = ("A_out_I", "A")
PAIRS = [
    ("dA@A_out_I ~ dU@A_out_you (E7 referent sharing, I vs you)", ("A_out_I", "A"), ("A_out_you", "U")),
    ("dA@A_out_I ~ dA@A_init_I (same token id as quoted I)", ("A_out_I", "A"), ("A_init_I", "A")),
    ("dA@A_out_I ~ dU@A_qU_I (quoted user I: referent)", ("A_out_I", "A"), ("A_qU_I", "U")),
    ("dA@A_out_I ~ dA@A_qU_I (quoted user I: default/speaker)", ("A_out_I", "A"), ("A_qU_I", "A")),
    ("dA@A_out_I ~ dT@A_qT_I (quoted T I: referent)", ("A_out_I", "A"), ("A_qT_I", "T")),
    ("dA@A_out_I ~ dA@A_qT_I (quoted T I: default/speaker)", ("A_out_I", "A"), ("A_qT_I", "A")),
    ("dA@A_out_I ~ dT@A_qS_you (suggested-line you -> T)", ("A_out_I", "A"), ("A_qS_you", "T")),
    ("dA@A_out_I ~ dA@U_qA_I (user turn quoting A)", ("A_out_I", "A"), ("U_qA_I", "A")),
    ("dA@A_out_I ~ dT@U_qT_I (user turn quoting T)", ("A_out_I", "A"), ("U_qT_I", "T")),
    ("dA@A_out_I ~ dT@A_T (T name tokens)", ("A_out_I", "A"), ("A_T", "T")),
    ("dA@A_out_I ~ dT@A_nearT_I (proximity ctrl, non-referent T)", ("A_out_I", "A"), ("A_nearT_I", "T")),
    ("dA@A_out_I ~ dU@A_out_I (E7: same token, other entity)", ("A_out_I", "A"), ("A_out_I", "U")),
    ("dA@A_out_I ~ dT@A_qU_I (quoted user I: third party T)", ("A_out_I", "A"), ("A_qU_I", "T")),
    ("dA@A_out_I ~ dT@U_qA_I (quoted asst I in user turn: third party T)", ("A_out_I", "A"), ("U_qA_I", "T")),
    ("dA@A_out_I ~ dU@U_out_I (user's own I, user turn)", ("A_out_I", "A"), ("U_out_I", "U")),
    ("dA@A_out_I ~ dA@U_out_you (user-turn you -> A)", ("A_out_I", "A"), ("U_out_you", "A")),
    ("dU@A_qU_I ~ dA@A_qU_I (quoted I: referent vs speaker)", ("A_qU_I", "U"), ("A_qU_I", "A")),
    ("dT@A_qT_I ~ dA@A_qT_I (quoted I: referent vs speaker)", ("A_qT_I", "T"), ("A_qT_I", "A")),
    ("dT@A_T ~ dT@A_qT_I (T at name vs T at quoted I)", ("A_T", "T"), ("A_qT_I", "T")),
]
out = {"layers": LS, "pairs": {}, "norm_ratios": {}}
for std in (False, True):
    tag = "std" if std else "raw"
    for name, (t1, e1), (t2, e2) in PAIRS:
        c, r1, r2 = ccos(get(t1, e1, std), get(t2, e2, std))
        out["pairs"][f"{tag}|{name}"] = {"ccos": c.round(3).tolist(), "rel1": r1.round(3).tolist(), "rel2": r2.round(3).tolist()}
    for t in ("A_out_I", "A_init_I", "A_out_you", "A_qU_I", "A_qT_I", "A_qS_you", "U_qA_I", "U_qT_I", "A_nearT_I", "A_T", "U_out_I", "U_out_you"):
        n2 = {e: unorm2(get(t, e, std)) for e in "AUT"}
        out["norm_ratios"][f"{tag}|{t}"] = {e: np.sqrt(np.clip(n2[e], 0, None)).round(3).tolist() for e in "AUT"}
json.dump(out, open(f"{RES}/V5_binding_dissociation_geometry.json", "w"), indent=1)
print("layers", LS)
for k, v in out["pairs"].items():
    print(f"{k:75s}", " ".join(f"{x:+.2f}" for x in v["ccos"]), " rel2min", round(min(v["rel2"]), 2))
for k, v in out["norm_ratios"].items():
    print(f"{k:22s}", " | ".join(f"{e}:" + ",".join(f"{x:.1f}" for x in v[e][2:9]) for e in "AUT"))
