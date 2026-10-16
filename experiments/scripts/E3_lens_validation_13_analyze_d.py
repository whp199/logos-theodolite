# E3_lens_validation (d) summary (local): per-layer means with bootstrap CIs over 12 prompts.
import json, numpy as np
R = "./experiments/results/"
d = json.load(open(R + "E3_lens_validation_d_toolbug.json")); rng = np.random.default_rng(0)
s = np.array(d["layer_scalar"]); M = {k: np.array(v) for k, v in d["per_prompt"].items()}  # [12, 60]
def ci(X):
    b = np.array([X[rng.integers(0, len(X), len(X))].mean(0) for _ in range(2000)])
    return X.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
out = {"layer_scalar": s.tolist()}
for k in ("cos_tool_true_ffn", "share_rescale", "ratio_resc_norm", "dom_tool_J", "dom_fix_J", "dom_tool_raw", "dom_fix_raw"):
    m, lo, hi = ci(M[k]); out[k] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
dd = M["dom_fix_J"] - M["dom_tool_J"]; m, lo, hi = ci(dd); out["dom_change_J"] = {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
c = np.array(out["cos_tool_true_ffn"]["mean"]); sh = np.array(out["share_rescale"]["mean"])
summ = {"n_layers_cos_lt_0.5": int((c < 0.5).sum()), "n_layers_cos_lt_0": int((c < 0).sum()), "n_layers_share_gt_0.5": int((sh > 0.5).sum()),
        "median_cos": float(np.median(c)), "corr_cos_vs_s": float(np.corrcoef(c, s)[0, 1]),
        "mean_abs_dom_change_J": float(np.abs(np.array(out["dom_change_J"]["mean"])).mean()),
        "max_abs_dom_change_J": float(np.abs(np.array(out["dom_change_J"]["mean"])).max()),
        "layers_dom_change_gt_0.1": [int(i) for i in np.where(np.abs(np.array(out["dom_change_J"]["mean"])) > 0.1)[0]],
        "n_layers_dom_CI_excludes0": int(((np.array(out["dom_change_J"]["lo"]) > 0) | (np.array(out["dom_change_J"]["hi"]) < 0)).sum()),
        "pole": {k: {l: float(np.mean(v)) for l, v in vv.items()} for k, vv in d["pole"].items() if k != "inv_rms_y"}}
inv = d["pole"]["inv_rms_y"]
summ["pole"]["inv_rms_y_cv_within_prompt"] = {l: float(np.mean([np.std(x) / np.mean(x) for x in v])) for l, v in inv.items()}
summ["pole"]["inv_rms_y_median"] = {l: float(np.median(np.concatenate([np.array(x) for x in v]))) for l, v in inv.items()}
out["summary"] = summ
json.dump(out, open(R + "E3_lens_validation_d_analysis.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print("layer s cos(tool,true) share_rescale dom_tool dom_fix")
for l in range(60):
    print(f"L{l:02d} s={s[l]:.3f} cos={c[l]:+.2f} [{out['cos_tool_true_ffn']['lo'][l]:+.2f},{out['cos_tool_true_ffn']['hi'][l]:+.2f}] share={sh[l]:+.2f} dom {out['dom_tool_J']['mean'][l]:.2f}->{out['dom_fix_J']['mean'][l]:.2f}")
