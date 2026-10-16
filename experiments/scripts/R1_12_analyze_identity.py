#!/usr/bin/env python3
# R1_IT_LENS 12 (local): identity-condition gender comparison under J++ / JLens / naive vs the IT-refitted directional lens.
# Unit = passage (40 E1 passages). Per passage x condition: mean over the assistant's self tokens (I/me/my...) per layer.
# Position = (x - man)/(woman - man), woman/man = mean of the two woman/man personas, computed on passage means;
# 95% CIs by paired bootstrap over passages (2000 resamples). AI-anchored scale: (x - James_AI)/(Sarah_AI - James_AI).
import glob, json
import numpy as np
import torch

R = "./experiments/results/"
BANDS = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]
BN = [f"L{a}-{b}" for a, b in BANDS]
rng = np.random.default_rng(2)
files = sorted(glob.glob(R + "R1_07_identity_*.pt"))
meta, scores = [], []
off = 0
for f in files:
    d = torch.load(f, weights_only=False)
    for m in d["meta"]:
        m = dict(m); m["off"] += off; meta.append(m)
    scores.append(d["scores"].float()); off += d["scores"].shape[0]
    cols, conds = d["cols"], d["conds"]
S = torch.cat(scores, 0).numpy()   # [rows,60,C]
qis = sorted({m["qi"] for m in meta})
nQ, nC = len(qis), len(conds)


def agg(label_set):
    A = np.full((nQ, nC, 60, len(cols)), np.nan)
    for m in meta:
        idx = [m["off"] + i for i, lb in enumerate(m["labels"]) if lb in label_set]
        if idx:
            A[qis.index(m["qi"]), conds.index(m["cond"])] = S[idx].mean(0)
    return A


SELF = agg({"self"})
NONSELF = agg({"nonself"})
ci = {c: i for i, c in enumerate(conds)}
AI = ["default", "nameless", "gemma_short", "gemma_explicit", "f_sarah", "m_james", "n_atlas", "n_quill", "n_unit7"]


def bandavg(A):  # [...,60,C] -> [...,5,C]
    return np.stack([A[..., a:b + 1, :].mean(-2) for a, b in BANDS], -2)


def positions(A, sel=None):
    """A [Q,Cond,5,C] -> dict cond -> [5,C] positions on woman-man scale (+ AI-anchored), from passage means."""
    if sel is not None:
        A = A[sel]
    M = np.nanmean(A, 0)
    w = (M[ci["woman_p1"]] + M[ci["woman_p2"]]) / 2; mn = (M[ci["man_p1"]] + M[ci["man_p2"]]) / 2
    out = {c: (M[ci[c]] - mn) / (w - mn) for c in conds}
    out["neutral3"] = np.mean([out[c] for c in ("n_atlas", "n_quill", "n_unit7")], 0)
    sa, ja = M[ci["f_sarah"]], M[ci["m_james"]]
    out["_ai_default"] = (M[ci["default"]] - ja) / (sa - ja)
    out["_ai_nameless"] = (M[ci["nameless"]] - ja) / (sa - ja)
    out["_ai_gemma"] = (M[ci["gemma_short"]] - ja) / (sa - ja)
    out["_gap"] = w - mn
    out["_ai_gap_ratio"] = (sa - ja) / (w - mn)
    return out


READ = ["jpp_pron", "jpp_noun", "jlens_pron", "naive_pron", "it_mix_pron", "it_chat_pron", "it_prose_pron", "it_asstmode_pron",
        "it_mix_noun", "it_mix_she_he", "it_mix_woman_man", "itloud_mix_pron", "jlin_pron", "jlinloud_pron"]
RI = [cols.index(r) for r in READ]
res = {"bands": BN, "readouts": READ, "n_passages": nQ, "conds": conds}
for tag, A0 in (("self", SELF), ("nonself", NONSELF)):
    A = bandavg(A0)[..., RI]                    # [Q,Cond,5,R]
    pt = positions(A)
    B = 2000
    bs = [positions(A, rng.integers(0, nQ, nQ)) for _ in range(B)]
    R_ = {}
    keys = conds + ["neutral3", "_ai_default", "_ai_nameless", "_ai_gemma", "_gap", "_ai_gap_ratio"]
    for k in keys:
        arr = np.stack([b[k] for b in bs])
        R_[k] = {"mean": pt[k].round(3).tolist(), "ci": np.percentile(arr, [2.5, 97.5], 0).round(3).tolist()}
    for a, b in (("default", "nameless"), ("default", "neutral3"), ("gemma_short", "nameless"), ("gemma_explicit", "nameless"),
                 ("f_sarah", "nameless"), ("m_james", "nameless")):
        arr = np.stack([x[a] - x[b] for x in bs])
        R_[f"{a}-{b}"] = {"mean": (pt[a] - pt[b]).round(3).tolist(), "ci": np.percentile(arr, [2.5, 97.5], 0).round(3).tolist()}
    # late artifact: how many AI conditions read at/above the woman persona (position >= 1), and the min AI position
    arr = np.stack([[x[c] for c in AI] for x in bs])          # [B,9,5,R]
    R_["n_ai_ge_woman"] = {"mean": np.sum(np.stack([pt[c] for c in AI]) >= 1.0, 0).tolist()}
    R_["min_ai_pos"] = {"mean": np.min(np.stack([pt[c] for c in AI]), 0).round(3).tolist(),
                        "ci": np.percentile(arr.min(1), [2.5, 97.5], 0).round(3).tolist()}
    R_["mean_ai_pos"] = {"mean": np.mean(np.stack([pt[c] for c in AI]), 0).round(3).tolist(),
                         "ci": np.percentile(arr.mean(1), [2.5, 97.5], 0).round(3).tolist()}
    res[tag] = R_

json.dump(res, open(R + "R1_12_identity_analysis.json", "w"), indent=1)


def show(tag, key, r):
    j = READ.index(r)
    m = np.array(res[tag][key]["mean"])[:, j]
    c = np.array(res[tag][key]["ci"])[:, :, j] if "ci" in res[tag][key] else None
    if c is None:
        return " ".join(f"{x:+.2f}" for x in m)
    return " ".join(f"{x:+.2f}[{a:+.2f},{b:+.2f}]" for x, a, b in zip(m, c[0], c[1]))


print("bands", BN)
for r in READ:
    print(f"\n### {r}  (self tokens; woman-man gap {show('self', '_gap', r)})")
    for k in ("default", "nameless", "neutral3", "gemma_short", "gemma_explicit", "f_sarah", "m_james", "default-nameless",
              "default-neutral3", "gemma_short-nameless", "_ai_default", "min_ai_pos", "n_ai_ge_woman"):
        print(f"  {k:22s}", show("self", k, r))

# ---------------- residual-mean background under each linear readout, in units of the woman-man gap at self tokens
v6 = torch.load(R + "V6_quiet_dims_stats.pt", weights_only=False)
mu = v6["mean_all"].float()                                    # [60,D]
GB = torch.load(R + "R1_04_gbar.pt", weights_only=False)
DD = torch.load(R + "R1_02_dirs.pt", weights_only=False)
std = v6["std_all"].float(); loud = (std > torch.quantile(std, 0.1, dim=1, keepdim=True)).float()
bg = {}
for nm_, vec in (("it_mix_pron", GB["mix"][0]), ("jlin_pron", DD["pb_jpp"][0]), ("itloud_mix_pron", GB["mix"][0] * loud),
                 ("jlinloud_pron", DD["pb_jpp"][0] * loud)):
    m = (mu * vec).sum(-1).numpy()                              # [60]
    mb = np.array([m[a:b + 1].mean() for a, b in BANDS])
    gap = np.array(res["self"]["_gap"]["mean"])[:, READ.index(nm_)]
    bg[nm_] = {"mu_readout": mb.round(3).tolist(), "in_gap_units": (mb / gap).round(3).tolist()}
res["mu_background"] = bg
json.dump(res, open(R + "R1_12_identity_analysis.json", "w"), indent=1)
print("\nresidual-mean background (units of woman-man gap):")
for k, v in bg.items():
    print(f"  {k:16s}", " ".join(f"{x:+.2f}" for x in v["in_gap_units"]))
