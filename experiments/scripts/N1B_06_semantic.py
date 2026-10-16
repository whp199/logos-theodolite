#!/usr/bin/env python3
"""N1B step 6 (CPU, local): concept-level (semantic) agreement between static labels and activation-based labels.

Motivation: exact-token precision@10 undervalues labels that name the right concept with different tokens
(e.g. empirical ' SSH',' encryption',' SSL' vs static ' VPN',' password',' Privacy'). Here each token set is embedded
with the model's own (tied) input embedding W_E, mean-centred over the vocabulary and L2-normalised per token; the
agreement score is the cosine between the two set centroids. Null: within-layer re-pairing with every other neuron's
label (exact expectation), permutation p from 2000 derangements; bootstrap CIs over poles (same machinery as N1B_04).
PRE-STATED PREDICTION (written before running): G_jpp shows a clearer semantic lift than exact-token lift at L20-45
(J++ maps the gate covector onto output-concept directions that are related to, but not identical with, the input
tokens), but the absolute centroid cosine stays small (< 0.15); G_naive/G_embed lift concentrates at L10.
Empirical sets: top-20 enriched current tokens (input) / next tokens (output) at the pole's top-1% positions;
static sets: top-10 tokens of each variant (full vocabulary, exactly what the tool shows).
"""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open

R = "./experiments/results/"
sys.path.insert(0, "./experiments/scripts")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/")
SNAP += sorted(os.listdir(SNAP))[0] + "/model.safetensors"
rng = np.random.default_rng(7)
NB = 2000
NPL = 24

with safe_open(SNAP, "pt") as f:
    WE = f.get_tensor("model.language_model.embed_tokens.weight")      # [V, D] bf16
mu = WE.float().mean(0)
labels = json.load(open(R + "N1B_labels.json"))
S = np.load(R + "N1B_static.npz")
LAYERS = [int(x) for x in S["layers"]]
poles = labels["poles"]
assert len(poles) == 2 * NPL * len(LAYERS)

def centroid(ids):
    ids = [int(t) for t in ids if t >= 0]
    if not ids:
        return None
    E = WE[torch.tensor(ids)].float() - mu
    E = E / E.norm(dim=-1, keepdim=True)
    c = E.mean(0)
    return c / c.norm()

VARS_AGN = ["G_jpp", "G_jpp_pre", "G_naive", "G_embed"]
VARS_POL = ["P_geglu0", "O_jvindex", "O_naive"]

def static_ids(var, j, s):
    if var in VARS_POL and s == -1:
        return S[var + "__bot"][j][:10]
    return S[var + "__top"][j][:10]

emp_c = {}
for k, p in enumerate(poles):
    j, s = k // 2, (1 if p["pole"] == "+" else -1)
    emp_c[(j, s, "in")] = centroid([t[1] for t in p["top_input_enriched"][:20]])
    emp_c[(j, s, "nx")] = centroid([t[1] for t in p["top_next_enriched"][:20]])

def boot(x, n):
    k = len(x)
    idx = rng.integers(0, k, size=(NB, k))
    bx, bn = x[idx].mean(1), n[idx].mean(1)
    bd = bx - bn
    return {"n": int(k), "metric": float(x.mean()), "metric_ci": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))],
            "null": float(n.mean()), "diff": float(x.mean() - n.mean()),
            "diff_ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}

out = {"definitions": __doc__}
for var in VARS_AGN + VARS_POL:
    st_c = {(j, s): centroid(static_ids(var, j, s)) for j in range(NPL * len(LAYERS)) for s in (1, -1)}
    for side in ("in", "nx"):
        Ms = []
        for li in range(len(LAYERS)):
            js = list(range(li * NPL, (li + 1) * NPL))
            M = np.zeros((2 * NPL, NPL))
            for a_, j in enumerate(js):
                for b_, s in enumerate((1, -1)):
                    e = emp_c[(j, s, side)]
                    for c_, j2 in enumerate(js):
                        sc = st_c[(j2, s)]
                        M[2 * a_ + b_, c_] = float(e @ sc) if (e is not None and sc is not None) else 0.0
            Ms.append(M)
        rows = np.arange(2 * NPL)
        own = rows // 2
        res = {"per_layer": {}}
        X, Nn = [], []
        for li, M in enumerate(Ms):
            x = M[rows, own]
            nul = (M.sum(1) - x) / (NPL - 1)
            res["per_layer"][str(LAYERS[li])] = boot(x, nul)
            X.append(x); Nn.append(nul)
        X, Nn = np.concatenate(X), np.concatenate(Nn)
        res["overall"] = boot(X, Nn)
        obs, cnt = X.mean(), 0
        for _ in range(NB):
            v = []
            for M in Ms:
                perm = rng.permutation(NPL)
                while (perm == np.arange(NPL)).any():
                    perm = rng.permutation(NPL)
                v.append(M[rows, perm[own]])
            cnt += np.concatenate(v).mean() >= obs
        res["overall"]["perm_p_one_sided"] = (cnt + 1) / (NB + 1)
        out[f"{var}|{side}"] = res
        o = res["overall"]
        print(f"{var:10s} {side}: cos={o['metric']:.4f} null={o['null']:.4f} diff={o['diff']:.4f} "
              f"CI={o['diff_ci'][0]:.4f},{o['diff_ci'][1]:.4f} p={o['perm_p_one_sided']:.4f} | " +
              " ".join(f"{l}:{v['metric']:.3f}/{v['null']:.3f}" for l, v in res["per_layer"].items()), flush=True)

# ceiling: split-half (by document) empirical input labels of the same pole (top-20 enriched each half), same null
PS = json.load(open(R + "N1B_pole_sets.json"))
shc = {(j, s, h): centroid(PS[f"{j}|{s}"][f"sh{h}_enrich"][:20]) for j in range(NPL * len(LAYERS)) for s in (1, -1) for h in (0, 1)}
X, Nn, per = [], [], {}
for li in range(len(LAYERS)):
    js = list(range(li * NPL, (li + 1) * NPL))
    xl, nl = [], []
    for j in js:
        for s in (1, -1):
            a = shc[(j, s, 0)]
            if a is None or shc[(j, s, 1)] is None:
                continue
            xl.append(float(a @ shc[(j, s, 1)]))
            nl.append(float(np.mean([float(a @ shc[(j2, s, 1)]) for j2 in js if j2 != j and shc[(j2, s, 1)] is not None])))
    per[str(LAYERS[li])] = boot(np.array(xl), np.array(nl))
    X += xl; Nn += nl
out["ceiling_splithalf_input_enrich20"] = {"overall": boot(np.array(X), np.array(Nn)), "per_layer": per}
o = out["ceiling_splithalf_input_enrich20"]["overall"]
print(f"CEILING split-half in: cos={o['metric']:.4f} null={o['null']:.4f} diff={o['diff']:.4f} CI={o['diff_ci']}")
ref = []
for j in range(NPL * len(LAYERS)):
    a, b = emp_c[(j, 1, "in")], emp_c[(j, -1, "in")]
    if a is not None and b is not None:
        ref.append(float(a @ b))
out["ref_plus_vs_minus_input_centroid_cos_mean"] = float(np.mean(ref))
json.dump(out, open(R + "N1B_semantic.json", "w"), indent=1)
print("plus-vs-minus pole input centroid cos:", np.mean(ref))
