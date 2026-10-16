# E1_self_readout: reduce per-position raw scores to passage-level class means (CPU only; run with .venv/bin/python).
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
RAW = "./experiments/results/E1_self_readout_raw"
OUT = "./experiments/results/E1_self_readout_passage_means.npz"
LENSES = ["jpp", "jlens", "naive"]
CLASSES = ["self", "nonself", "user_all", "user_I", "user_other", "header", "asst_all"]
CLASSMAP = {"self": {"self"}, "nonself": {"nonself"}, "user_all": {"user_I", "user_other"}, "user_I": {"user_I"},
            "user_other": {"user_other"}, "header": {"header"}, "asst_all": {"self", "nonself", "asst_contraction"}}
files = sorted(f for f in os.listdir(RAW) if f.startswith("main_"))
data = [torch.load(f"{RAW}/{f}", weights_only=False) for f in files]
conds = data[0]["conds"]; metrics = data[0]["metrics"]
nq = 40
X = np.full((len(LENSES), len(conds), nq, len(CLASSES), 60, len(metrics)), np.nan, dtype=np.float32)
NPOS = np.zeros((len(conds), nq, len(CLASSES)), dtype=np.int32)
for d in data:
    S = {k: d["scores"][k].float().numpy() for k in LENSES}
    for m in d["meta"]:
        ci = conds.index(m["cond"]); qi = m["qi"]
        labs = np.array(m["labels"])
        for k, cl in enumerate(CLASSES):
            idx = np.where(np.isin(labs, list(CLASSMAP[cl])))[0]
            NPOS[ci, qi, k] = len(idx)
            if len(idx) == 0:
                continue
            for li, ln in enumerate(LENSES):
                X[li, ci, qi, k] = S[ln][m["off"] + idx].mean(0)
np.savez_compressed(OUT, X=X, NPOS=NPOS, lenses=np.array(LENSES), conds=np.array(conds), classes=np.array(CLASSES),
                    metrics=np.array(metrics))
# token-type table: mean jpp pron per self-token type per condition (layers), passage-pooled
tt = {}
for d in data:
    S = d["scores"]["jpp"].float().numpy()
    for m in d["meta"]:
        for j, (lb, t) in enumerate(zip(m["labels"], m["toks"])):
            if lb == "self":
                tt.setdefault((m["cond"], t.strip().lower()), []).append(S[m["off"] + j, :, :2])
TT = {f"{c}|{t}": np.stack(v).mean(0).tolist() for (c, t), v in tt.items()}
TTn = {f"{c}|{t}": len(v) for (c, t), v in tt.items()}
import json
json.dump({"mean_by_type": TT, "n": TTn}, open("./experiments/results/E1_self_readout_tokentype.json", "w"))
print("saved", OUT, X.shape, "nan frac", float(np.isnan(X).mean()))
print("NPOS self mean", NPOS[..., 0].mean(), "nonself", NPOS[..., 1].mean(), "user_I", NPOS[..., 3].mean())
