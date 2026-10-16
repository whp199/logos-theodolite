#!/usr/bin/env python3
"""E6_tool_audit_07_analyze -- offline aggregation of E6_tool_audit_03 sweep chunks (run locally, no GPU).
Unit of analysis = prompt; per-prompt band means first; bootstrap 95% CI over prompts (2000 resamples);
paired differences where conditions share the 12 prompt frames (same index)."""
import glob, json, math
import numpy as np

R = "./experiments/results/"
BANDS = {"L10-19": (10, 20), "L20-29": (20, 30), "L30-39": (30, 40), "L40-49": (40, 50), "L50-58": (50, 59), "L59": (59, 60)}
rng = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def bandvals(arr):  # arr [60] -> {band: mean}
    a = np.asarray(arr, float)
    return {b: float(a[s:e].mean()) for b, (s, e) in BANDS.items()}


def gs3(rows):
    """Gram-Schmidt of 3 row vectors (3x3) -> orthonormal rows."""
    out = []
    for v in rows:
        u = np.array(v, float)
        for e in out:
            u = u - (u @ e) * e
        out.append(u / np.linalg.norm(u))
    return np.stack(out)


data, Rraw = {}, None
for f in sorted(glob.glob(R + "E6_tool_audit_03_sweeps_c*.partial.json")):
    d = json.load(open(f))
    Rraw = np.array(d["R_raw"])
    data.update(d["conds"])
conds = list(data.keys())
print("conditions:", {c: len(v) for c, v in data.items()})
she, he, it = Rraw  # monument coordinates in displayed (she,he,it) frame
E_hs = gs3([he, she, it])  # frame if the user enters monuments in order (he, she, it), in base coords
E_ith = gs3([it, she, he])

per = {}  # cond -> metric -> band -> list over prompts
for c, rows in data.items():
    M = {}
    def add(name, arr60):
        for b, v in bandvals(arr60).items():
            M.setdefault(name, {}).setdefault(b, []).append(v)
    for r in rows:
        add("lens_pron_single", r["lens1"]["pron"])
        add("lens_pron_double", r["lens2"]["pron"])
        add("lens_noun_single", r["lens1"]["noun"])
        add("lens_she_he_single", r["lens1"]["she_he"])
        add("lens_neuter_vs_human_single", r["lens1"]["neuter_vs_human"])
        X = np.array(r["xyz_raw"])  # [60,3]
        add("hud_x_she_minus_y_he", X[:, 0] - X[:, 1])
        Xh = X @ E_hs.T  # coords in (he,she,it) frame: x[he], y[she]
        add("hud_permuted_y_she_minus_x_he", Xh[:, 1] - Xh[:, 0])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        cs = Xn @ (she / np.linalg.norm(she)); ch = Xn @ (he / np.linalg.norm(he))
        add("angle_cos_she_minus_cos_he", cs - ch)
        add("az_order_she_he_it", np.array(r["az"]))
        az_hs = np.degrees(np.arctan2(Xh[:, 1], Xh[:, 0]))
        add("az_order_he_she_it", az_hs)
        Xi = X @ E_ith.T
        add("az_order_it_she_he", np.degrees(np.arctan2(Xi[:, 1], Xi[:, 0])))
        add("el_order_she_he_it", np.array(r["el"]))
        add("el_order_he_she_it", np.degrees(np.arcsin(np.clip(Xh[:, 2] / (np.linalg.norm(Xh, axis=1) + 1e-12), -1, 1))))
        add("proj_frac", np.array(r["proj_frac"]))
        Xe = np.array(r["xyz_eff"])
        add("eff_frame_x_she_minus_y_he", Xe[:, 0] - Xe[:, 1])
        for k in ("jpp", "naive"):
            rs = np.log10(np.array(r[f"{k}_rank"]["she"], float)); rh = np.log10(np.array(r[f"{k}_rank"]["he"], float))
            add(f"{k}_log10rank_he_minus_she", rh - rs)
            add(f"{k}_she_ranked_above_he", (rs < rh).astype(float))
            ls_ = np.array(r[f"{k}_logit"]["she"]); lh = np.array(r[f"{k}_logit"]["he"])
            add(f"{k}_logit_she_minus_he", ls_ - lh)
        add("m1_tool_dom", r["m1_tool_dom"]); add("m1_corr_dom", r["m1_corr_dom"]); add("m1_cos_ffn", r["m1_cos_ffn"])
        add("m1_dom_class_disagree", (np.array(r["m1_tool_dom"]) > 0.5) != (np.array(r["m1_corr_dom"]) > 0.5))
    per[c] = M

summary = {"conditions": {c: len(v) for c, v in data.items()}, "metrics": {}}
METRICS = ["lens_pron_single", "lens_pron_double", "lens_noun_single", "lens_neuter_vs_human_single",
           "hud_x_she_minus_y_he", "hud_permuted_y_she_minus_x_he", "angle_cos_she_minus_cos_he", "eff_frame_x_she_minus_y_he",
           "jpp_log10rank_he_minus_she", "jpp_she_ranked_above_he", "naive_log10rank_he_minus_she", "jpp_logit_she_minus_he", "naive_logit_she_minus_he",
           "az_order_she_he_it", "az_order_he_she_it", "az_order_it_she_he", "el_order_she_he_it", "el_order_he_she_it", "proj_frac",
           "m1_tool_dom", "m1_corr_dom", "m1_cos_ffn", "m1_dom_class_disagree"]
for m in METRICS:
    summary["metrics"][m] = {c: {b: boot(per[c][m][b]) for b in BANDS} for c in conds}


def scale_pos(metric, x, w, mname, band):
    xm = np.array(per[x][metric][band]); wm = np.array(per[w][metric][band]); mm = np.array(per[mname][metric][band])
    # paired by prompt index (same frames/questions)
    pos = []
    idx = rng.integers(0, len(xm), (2000, len(xm)))
    for ii in idx:
        dw, dm, dx = wm[ii].mean(), mm[ii].mean(), xm[ii].mean()
        pos.append((dx - dm) / (dw - dm) if abs(dw - dm) > 1e-9 else np.nan)
    p0 = (xm.mean() - mm.mean()) / (wm.mean() - mm.mean())
    return [float(p0), float(np.nanpercentile(pos, 2.5)), float(np.nanpercentile(pos, 97.5)), float(wm.mean() - mm.mean())]


scale = {}
for metric in ["lens_pron_single", "lens_noun_single", "hud_x_she_minus_y_he", "hud_permuted_y_she_minus_x_he",
               "angle_cos_she_minus_cos_he", "jpp_log10rank_he_minus_she", "eff_frame_x_she_minus_y_he"]:
    for fam, (w, mname, xs) in {"A": ("A_woman", "A_man", ["A_default", "A_bg", "A_raw"]),
                                "B": ("B_woman", "B_man", ["B_gemma", "B_atlas", "B_nameless", "B_gemma_human"])}.items():
        if w not in per or mname not in per:
            continue
        for x in xs:
            if x not in per:
                continue
            for b in BANDS:
                scale.setdefault(metric, {}).setdefault(x, {})[b] = scale_pos(metric, x, w, mname, b)
summary["woman_man_scale"] = scale

# paired difference: self vs background token (A_default - A_bg), and lens double-single
paired = {}
if "A_default" in per and "A_bg" in per:
    for m in ["lens_pron_single", "hud_x_she_minus_y_he", "jpp_log10rank_he_minus_she"]:
        paired[f"A_default_minus_A_bg:{m}"] = {b: boot(np.array(per["A_default"][m][b]) - np.array(per["A_bg"][m][b])) for b in BANDS}
for c in conds:
    paired[f"{c}:double_minus_single_lens_pron"] = {b: boot(np.array(per[c]["lens_pron_double"][b]) - np.array(per[c]["lens_pron_single"][b])) for b in BANDS}
summary["paired"] = paired

# behavior (real model, single BOS) at the last position
beh = {}
for c, rows in data.items():
    lo = [math.log(r["beh1"]["pF"] + 1e-12) - math.log(r["beh1"]["pM"] + 1e-12) for r in rows]
    beh[c] = {"logodds_F_M": boot(lo), "pF": boot([r["beh1"]["pF"] for r in rows]), "pM": boot([r["beh1"]["pM"] for r in rows]),
              "pIt": boot([r["beh1"]["pIt"] for r in rows]), "pThey": boot([r["beh1"]["pThey"] for r in rows]),
              "top1": [r["top1_single"] for r in rows],
              "kl_single_vs_double": boot([r["kl_single_vs_double"] for r in rows]),
              "kl_max": max(r["kl_single_vs_double"] for r in rows),
              "top1_same_frac": sum(r["top1_same"] for r in rows) / len(rows),
              "double_bos": all(r["first_ids"][:2] == [2, 2] for r in rows),
              "space_hack_triggered": sum(r["space_hack_triggered"] for r in rows),
              "n_filaments": boot([r["n_fil"] for r in rows]),
              "frac_filaments_src_lt10": boot([sum(1 for s in r["fil_src_layers"] if s < 10) / max(1, len(r["fil_src_layers"])) for r in rows]),
              "cka_offdiag": boot([r["cka_offdiag"] for r in rows]),
              "jpp_top5_gendered_F_M": [sum(r["jpp_top5_gcount"][0] for r in rows), sum(r["jpp_top5_gcount"][1] for r in rows)],
              "naive_top5_gendered_F_M": [sum(r["naive_top5_gcount"][0] for r in rows), sum(r["naive_top5_gcount"][1] for r in rows)],
              "pole_jpp_gendered_F_M": [sum(r["pole_jpp_gcount"][0] for r in rows), sum(r["pole_jpp_gcount"][1] for r in rows)],
              "pole_naive_gendered_F_M": [sum(r["pole_naive_gcount"][0] for r in rows), sum(r["pole_naive_gcount"][1] for r in rows)],
              "generated_examples": [r["generated"] for r in rows[:4]]}
summary["behavior_and_tool_misc"] = beh
# all gendered pole labels for the self conditions (what a naive user might screenshot)
summary["gendered_pole_labels_examples"] = {c: [lab for r in data[c] for lab in list(r["pole_gendered_labels"].items())[:2]][:8]
                                            for c in conds}
json.dump(summary, open(R + "E6_tool_audit_07_summary.json", "w"), indent=1)


def fmt(v):
    return "NA" if v is None else f"{v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]"


for m in ["lens_pron_single", "hud_x_she_minus_y_he", "hud_permuted_y_she_minus_x_he", "angle_cos_she_minus_cos_he",
          "jpp_log10rank_he_minus_she", "jpp_she_ranked_above_he", "az_order_she_he_it", "az_order_he_she_it", "proj_frac"]:
    print(f"\n## {m}")
    print("cond".ljust(16) + "".join(b.rjust(24) for b in BANDS))
    for c in conds:
        print(c.ljust(16) + "".join(fmt(summary["metrics"][m][c][b]).rjust(24) for b in BANDS))
print("\n## woman/man scale positions (0=man,1=woman)")
for metric, d in scale.items():
    for x, bd in d.items():
        print(metric[:28].ljust(30), x.ljust(14), "  ".join(f"{b}:{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}](gap{v[3]:+.2f})" for b, v in bd.items() if b in ("L30-39", "L40-49", "L50-58")))
print("\n## behavior")
for c, v in beh.items():
    print(c.ljust(16), "logodds", fmt(v["logodds_F_M"]), "pF", fmt(v["pF"]), "pM", fmt(v["pM"]), "pIt", fmt(v["pIt"]),
          "KL", fmt(v["kl_single_vs_double"]), "top1same", v["top1_same_frac"], "hack", v["space_hack_triggered"],
          "polesF/M jpp", v["pole_jpp_gendered_F_M"], "naive", v["pole_naive_gendered_F_M"], "top5F/M", v["jpp_top5_gendered_F_M"])
print("\n## paired")
for k, v in paired.items():
    print(k.ljust(60), "  ".join(f"{b}:{fmt(x)}" for b, x in v.items() if b in ("L30-39", "L40-49", "L50-58")))
print("\n## module 1 (all conditions pooled shown per cond)")
for c in conds:
    print(c.ljust(16), "tool_dom", fmt(summary["metrics"]["m1_tool_dom"][c]["L50-58"]), "corr_dom", fmt(summary["metrics"]["m1_corr_dom"][c]["L50-58"]),
          "cos_ffn L10-19", fmt(summary["metrics"]["m1_cos_ffn"][c]["L10-19"]), "L50-58", fmt(summary["metrics"]["m1_cos_ffn"][c]["L50-58"]),
          "L59", fmt(summary["metrics"]["m1_cos_ffn"][c]["L59"]))

# ---------------- plume (J++ top-5) audit: what a naive user sees in the ghost-token plumes
print("\n## plumes: prompts (of 12) with ' she' / ' he' / ' it' / ' they' in J++ top-5 at any layer 30-58; top-1 counts; lens-vs-output top-1 agreement")
plume = {}
for c, rows in data.items():
    def anyin(tok, lo=30, hi=59):
        return sum(1 for r in rows if any(tok in [t for t in r["jpp_top5_by_layer"][l]] for l in range(lo, hi)))
    def top1count(tok, lo=30, hi=59):
        return sum(1 for r in rows for l in range(lo, hi) if r["jpp_top5_by_layer"][l][0] == tok)
    agree = {b: float(np.mean([r["jpp_top5_by_layer"][l][0] == r["top1_single"] for r in rows for l in range(s, e)])) for b, (s, e) in BANDS.items()}
    plume[c] = {"any_she": anyin(" she"), "any_he": anyin(" he"), "any_it": anyin(" it"), "any_they": anyin(" they"),
                "top1_she_layers": top1count(" she"), "top1_he_layers": top1count(" he"), "top1_it_layers": top1count(" it"),
                "top1_they_layers": top1count(" they"), "lens_top1_agrees_output": agree}
    print(c.ljust(16), {k: v for k, v in plume[c].items() if k != "lens_top1_agrees_output"},
          {b: round(v, 2) for b, v in agree.items()})
summary["plumes"] = plume
# naive logit lens she-he (tool's naive readout) per band
print("\n## tool tracked logit she-he: J++ vs naive logit lens")
for c in conds:
    print(c.ljust(16), "J++ ", "  ".join(f"{b}:{fmt(summary['metrics']['jpp_logit_she_minus_he'][c][b])}" for b in ("L30-39", "L40-49", "L50-58")))
    print(c.ljust(16), "naive", "  ".join(f"{b}:{fmt(summary['metrics']['naive_logit_she_minus_he'][c][b])}" for b in ("L30-39", "L40-49", "L50-58")))
json.dump(summary, open(R + "E6_tool_audit_07_summary.json", "w"), indent=1)
