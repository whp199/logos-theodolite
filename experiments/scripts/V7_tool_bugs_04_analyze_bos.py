# V7_tool_bugs_04_analyze_bos (OFFLINE, CPU): summarize (a) double-BOS and (b) monument-order results from job 01.
import json, glob, math, random, sys
import numpy as np
RES = "./experiments/results/"
BANDS = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40),
         "L40-49": range(40, 50), "L50-58": range(50, 59), "L59": range(59, 60)}


def boot(xs, n=2000, seed=0, fn=np.mean):
    xs = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    bs = [fn(xs[rng.integers(0, len(xs), len(xs))]) for _ in range(n)]
    return {"est": float(fn(xs)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(xs)}


def circ(a, b):
    d = (np.asarray(a) - np.asarray(b) + 180) % 360 - 180
    return np.abs(d)


def bandmean(arr):
    arr = np.asarray(arr, float)
    return {b: float(np.nanmean(arr[list(r)])) for b, r in BANDS.items()}


chat = []
for f in sorted(glob.glob(RES + "V7_tool_bugs_01_bos_chat*.partial.json")):
    chat += json.load(open(f))
raw = json.load(open(RES + "V7_tool_bugs_01_bos_raw.partial.json")) if glob.glob(RES + "V7_tool_bugs_01_bos_raw.partial.json") else []
out = {"n_chat": len(chat), "n_raw": len(raw)}


def pair_metrics(rows, A, Bv):
    m = {"ids_A": [r[A]["ids_first3"] for r in rows], "ids_B": [r[Bv]["ids_first3"] for r in rows]}
    m["gen_identical"] = sum(r[A]["gen"] == r[Bv]["gen"] for r in rows)
    m["gens"] = [[r[A]["gen"], r[Bv]["gen"]] for r in rows]
    m["kl_next"] = boot([r["kl_ref_vs_alt"] for r in rows])
    m["kl_next_max"] = max(r["kl_ref_vs_alt"] for r in rows)
    m["top1_same"] = sum(r["top1_same"] for r in rows)
    # per layer
    top1_agree = np.array([[r[A]["jpp_top1"][l] == r[Bv]["jpp_top1"][l] for l in range(60)] for r in rows], float)
    m["jpp_top1_agree_band"] = bandmean(top1_agree.mean(0))
    for lens in ("jpp", "naive"):
        sh = lambda r, v: np.array(r[v][f"{lens}_logit"]["she"]) - np.array(r[v][f"{lens}_logit"]["he"])
        d = np.array([sh(r, A) - sh(r, Bv) for r in rows])        # [P,60]
        m[f"{lens}_she_he_absdiff_band"] = bandmean(np.abs(d).mean(0))
        m[f"{lens}_she_he_signflip_frac_band"] = bandmean(np.mean([np.sign(sh(r, A)) != np.sign(sh(r, Bv)) for r in rows], 0))
        m[f"{lens}_she_he_band_A"] = bandmean(np.mean([sh(r, A) for r in rows], 0))
        m[f"{lens}_she_he_band_B"] = bandmean(np.mean([sh(r, Bv) for r in rows], 0))
    lr = np.array([[abs(math.log10(r[A]["jpp_rank"]["she"][l]) - math.log10(r[Bv]["jpp_rank"]["she"][l])) for l in range(60)] for r in rows])
    m["jpp_log10rank_she_absdiff_band"] = bandmean(lr.mean(0))
    daz = np.array([circ(r[A]["az"], r[Bv]["az"]) for r in rows])
    m["az_absdiff_median_band"] = bandmean(np.median(daz, 0))
    m["az_absdiff_max"] = float(daz.max())
    m["az_absdiff_L59"] = [float(x) for x in daz[:, 59]]
    del_ = np.array([np.abs(np.array(r[A]["el"]) - np.array(r[Bv]["el"])) for r in rows])
    m["el_absdiff_median_band"] = bandmean(np.median(del_, 0))
    m["cka_A"] = boot([r[A]["cka_offdiag"] for r in rows]); m["cka_B"] = boot([r[Bv]["cka_offdiag"] for r in rows])
    m["cka_diff"] = boot([r[A]["cka_offdiag"] - r[Bv]["cka_offdiag"] for r in rows])
    jac = np.array([[len(set(r[A]["poles"][l]) & set(r[Bv]["poles"][l])) / len(set(r[A]["poles"][l]) | set(r[Bv]["poles"][l]))
                     for l in range(60)] for r in rows])
    m["poles_jaccard_band"] = bandmean(jac.mean(0))
    fj = [len(set(map(tuple, r[A]["fil"])) & set(map(tuple, r[Bv]["fil"]))) / max(1, len(set(map(tuple, r[A]["fil"])) | set(map(tuple, r[Bv]["fil"])))) for r in rows]
    m["filament_jaccard"] = boot(fj)
    dd = np.array([np.abs(np.array(r[A]["ffn_dom"]) - np.array(r[Bv]["ffn_dom"])) for r in rows])
    m["ffn_dom_absdiff_band"] = bandmean(dd.mean(0))
    return m


if chat:
    out["chat_double_vs_single"] = pair_metrics(chat, "double", "single")
if raw:
    out["raw_withBOS_vs_noBOS"] = pair_metrics(raw, "withBOS", "noBOS")

# ---------------- (b) monument order
allrows = [("chat", r, "single") for r in chat] + [("raw", r, "withBOS") for r in raw]
perm_keys = ["shheit", "shithe", "heshit", "heitsh", "itshhe", "ithesh"]
az_range, el_range = [], []
hud_sh_first, hud_he_first, hud_lw, jpp_sign = [], [], [], []
lw_inv_err = []
for kind, r, ref in allrows:
    AZ = np.array([r["perm"][k]["az"] for k in perm_keys])     # [6,60]
    EL = np.array([r["perm"][k]["el"] for k in perm_keys])
    # max pairwise circular difference across permutations
    rng_ = np.zeros(60)
    for i in range(6):
        for j in range(6):
            rng_ = np.maximum(rng_, circ(AZ[i], AZ[j]))
    az_range.append(rng_); el_range.append(EL.max(0) - EL.min(0))
    c1 = r["perm"]["shheit"]["coord"]; c2 = r["perm"]["heshit"]["coord"]
    hud_sh_first.append(np.array(c1["she"]) - np.array(c1["he"]))   # x[she] - y[he] with she first
    hud_he_first.append(np.array(c2["she"]) - np.array(c2["he"]))   # she now 2nd axis
    lw = r["loewdin"]
    lw_inv_err.append(max(np.max(np.abs(np.array(lw[k]["she"]) - np.array(lw["shheit"]["she"]))) for k in lw))
    hud_lw.append(np.array(lw["shheit"]["she"]) - np.array(lw["shheit"]["he"]))
    jpp_sign.append(np.array(r[ref]["jpp_logit"]["she"]) - np.array(r[ref]["jpp_logit"]["he"]))
az_range = np.array(az_range); el_range = np.array(el_range)
H1, H2, HL, JS = map(np.array, (hud_sh_first, hud_he_first, hud_lw, jpp_sign))
mo = {"n_prompts": len(allrows),
      "az_range_across_perms_median_band": bandmean(np.median(az_range, 0)),
      "az_range_frac_gt30deg_band": bandmean((az_range > 30).mean(0)),
      "el_range_median_band": bandmean(np.median(el_range, 0)),
      "hud_she_minus_he_positive_frac_she_first_band": bandmean((H1 > 0).mean(0)),
      "hud_she_minus_he_positive_frac_he_first_band": bandmean((H2 > 0).mean(0)),
      "hud_sign_flips_with_order_frac_band": bandmean((np.sign(H1) != np.sign(H2)).mean(0)),
      "jpp_she_minus_he_logit_positive_frac_band": bandmean((JS > 0).mean(0)),
      "hud_she_first_agrees_with_jpp_logit_sign_band": bandmean((np.sign(H1) == np.sign(JS)).mean(0)),
      "hud_he_first_agrees_with_jpp_logit_sign_band": bandmean((np.sign(H2) == np.sign(JS)).mean(0)),
      "loewdin_agrees_with_jpp_logit_sign_band": bandmean((np.sign(HL) == np.sign(JS)).mean(0)),
      "loewdin_perm_invariance_max_abs_err": float(max(lw_inv_err)),
      "frac_hJ_captured_band": bandmean(np.mean([r["frac_hJ_captured"] for _, r, _ in allrows], 0)),
      "perm_check_max_abs_az_diff": max(r["perm_check_max_abs_az_diff"] for _, r, _ in allrows)}
# per-prompt example at L45
mo["example_L45"] = [{"prompt": r["prompt"][-60:], "az_by_perm": {k: round(r["perm"][k]["az"][45], 1) for k in perm_keys},
                      "el_by_perm": {k: round(r["perm"][k]["el"][45], 1) for k in perm_keys}} for _, r, _ in allrows[:6]]
out["monument_order"] = mo
json.dump(out, open(RES + "V7_tool_bugs_04_bos_monument_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:12000])
