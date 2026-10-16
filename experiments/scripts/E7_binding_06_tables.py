#!/usr/bin/env python3
"""E7_binding_06: build markdown tables (stdout) + compact summary JSON from the analysis outputs."""
import json, glob
import numpy as np

RES = "./experiments/results"
A = json.load(open(f"{RES}/E7_binding_analysis.json"))
G = json.load(open(f"{RES}/E7_binding_geometry_summary.json"))
BANDS = ["L10-19", "L20-29", "L30-39", "L40-49", "L50-58", "L59"]
bt = A["band_table"]
summary = {}


def cell(s):
    return f"{s['a']:+.2f} / {s['b']:+.2f} / **{s['BI']:+.2f}** [{s['BI_ci'][0]:+.2f},{s['BI_ci'][1]:+.2f}]"


for r in ("jpp_pron", "jlens_pron", "naive_pron", "probe_ref"):
    print(f"\n#### {r}: a (G_self coef) / b (G_user coef) / BI_ref [95% CI]\n")
    print("| type | " + " | ".join(BANDS) + " |")
    print("|---|" + "---|" * len(BANDS))
    for t in ("A_self", "U_addr", "F_I", "A_user", "U_self", "F_Y", "A_neut"):
        print(f"| {t} | " + " | ".join(cell(bt[f'{r}|{t}|{b}']) for b in BANDS) + " |")
# behaviour
b = A["behaviour"]
print("\n#### Behaviour (real next-token log-odds, noun4)\n")
print("| probe | a | b | BI_ref [CI] | mass F4+M4 (WW/MM/UU/D) |")
print("|---|---|---|---|---|")
for t, lab in (("F_I", '"... I am a"'), ("F_Y", '"... you are a"')):
    s = b[t]; m = s["mass_F4_plus_M4"]
    print(f"| {lab} | {s['a']:+.2f} [{s['a_ci'][0]:+.2f},{s['a_ci'][1]:+.2f}] | {s['b']:+.2f} [{s['b_ci'][0]:+.2f},{s['b_ci'][1]:+.2f}] | "
          f"{s['BI']:+.2f} [{s['BI_ci'][0]:+.2f},{s['BI_ci'][1]:+.2f}] | {m['WW']:.3f}/{m['MM']:.3f}/{m['UU']:.3f}/{m['D']:.3f} |")
print("\ncell means (lse4):")
for t in ("F_I", "F_Y"):
    print(t, {k: round(v, 2) for k, v in b[t]["cell_means_lse4"].items()})
# emergence: per-layer BI for A_self/A_user under jpp_pron
pl = A["per_layer"]
print("\n#### Per-layer BI (J++ pron), every 3rd layer\n")
Ls = list(range(0, 60, 3)) + [59]
print("| type | " + " | ".join(f"L{l}" for l in Ls) + " |")
print("|---|" + "---|" * len(Ls))
for t in ("A_self", "A_user", "U_self", "U_addr", "F_I", "F_Y"):
    print(f"| {t} | " + " | ".join(f"{pl[f'jpp_pron|{t}'][l]['BI']:+.2f}" for l in Ls) + " |")
print("\n| type | " + " | ".join(f"L{l}" for l in Ls) + " |")
print("|---|" + "---|" * len(Ls))
for t in ("A_self", "A_user", "F_I", "F_Y"):
    print(f"| {t} a/b | " + " | ".join(f"{pl[f'jpp_pron|{t}'][l]['a']:.1f}/{pl[f'jpp_pron|{t}'][l]['b']:.1f}" for l in Ls) + " |")
for t in ("A_self", "A_user", "F_I", "F_Y"):
    for r in ("jpp_pron", "jlens_pron", "naive_pron", "probe_ref"):
        bis = [pl[f"{r}|{t}"][l] for l in range(60)]
        good = [l for l in range(60) if bis[l]["BI_ci"][0] > 0.5]
        summary[f"bound_layers_ciLow>0.5|{r}|{t}"] = good
        print(r, t, "layers with BI CI-low > 0.5:", good)
# woman-man scale
print("\n#### Woman-man scale at A_self (0 = man persona MU, 1 = woman persona WU)\n")
print("| readout | band | WU | MU | UU | D (no system) | pos_UU [CI] | pos_D [CI] |")
print("|---|---|---|---|---|---|---|---|")
for r in ("jpp_pron", "jpp_noun", "jlens_pron", "probe_ref"):
    for bn in ("L20-29", "L30-39", "L40-49"):
        s = A["woman_man_scale"][f"{r}|A_self|{bn}"]
        print(f"| {r} | {bn} | {s['WU']:+.2f} | {s['MU']:+.2f} | {s['UU']:+.2f} | {s['D']:+.2f} | {s['pos_UU']:.2f} [{s['pos_UU_ci'][0]:.2f},{s['pos_UU_ci'][1]:.2f}] | {s['pos_D']:.2f} [{s['pos_D_ci'][0]:.2f},{s['pos_D_ci'][1]:.2f}] |")
# cell means 3x3 at A_self / A_user J++ pron L30-39 and L59
for t in ("A_self", "A_user", "A_neut"):
    for bn in ("L30-39", "L50-58"):
        cm = A["cell_means"][f"jpp_pron|{t}|{bn}"]
        print(f"\n3x3 J++ pron cell means at {t} {bn} (rows G_self W/M/U, cols G_user W/M/U); D={cm['D']:+.2f}")
        for gs in "WMU":
            print("  " + "  ".join(f"{cm[gs + gu]:+6.2f}" for gu in "WMU"))
# geometry table
print("\n#### Raw-residual geometry (noise-corrected cosines; diag-standardised coordinates)\n")
pairs = [("ds@A_self", "du@A_self", "same token, other entity (mood test)"),
         ("ds@A_self", "du@A_user", "referent gender shared across I / you"),
         ("ds@A_self", "ds@U_addr", "assistant gender at assistant-I vs user-turn 'you'"),
         ("ds@A_self", "du@U_self", "assistant-I vs user-turn 'I' (user referent)"),
         ("ds@A_self", "ds@A_user", "same entity (assistant) at I vs you tokens"),
         ("ds@A_neut", "du@A_neut", "neutral tokens: self vs user gender dirs"),
         ("ds@F_I", "du@F_Y", "prediction positions I-am-a vs you-are-a")]
print("| pair | meaning | " + " | ".join(BANDS[:5]) + " |")
print("|---|---|" + "---|" * 5)
for x, y, m in pairs:
    for sp in ("diag", "raw"):
        v = G[f"{sp}|cos|{x}~{y}"]
        print(f"| {x} ~ {y} ({sp}) | {m} | " + " | ".join(f"{v[b]:+.2f}" for b in BANDS[:5]) + " |")
print("\n| quantity | " + " | ".join(BANDS[:5]) + " |")
print("|---|" + "---|" * 5)
for t in ("A_self", "A_user", "A_neut", "U_self", "F_I", "F_Y"):
    v = G[f"diag|magratio_u_over_s|{t}"]
    print(f"| ‖d_user‖/‖d_self‖ at {t} | " + " | ".join(f"{v[b]:.2f}" for b in BANDS[:5]) + " |")
for t in ("A_self", "A_user", "U_self", "U_addr", "F_I", "F_Y"):
    v = G[f"diag|BIraw|{t}"]
    print(f"| raw-space BI (CV optimal dir) {t} | " + " | ".join(f"{v[b]:+.2f}" for b in BANDS[:5]) + " |")
for d in ("ds@A_self", "du@A_self", "du@A_user", "ds@A_user"):
    for e in ("pull_jpp_pron", "probe_ref"):
        v = G[f"diag|ext|{d}~{e}"]
        print(f"| cos({d}, {e}) | " + " | ".join(f"{v[b]:+.2f}" for b in BANDS[:5]) + " |")
# patching
P = []
for f in sorted(glob.glob(f"{RES}/E7_binding_patch_chunk*.json")):
    P += json.load(open(f))["rows"]
if P:
    rng = np.random.default_rng(3)
    print(f"\n#### Activation patching (flip fraction; mean over passages, 95% bootstrap CI over passages); n rows={len(P)}\n")
    pids = sorted(set(r["pid"] for r in P))
    print("| set | layer | flip 'I am a' (self) | flip 'you are a' (user) |")
    print("|---|---|---|---|")
    pt = {}
    for S in ("sys", "passage", "A_self", "A_user"):
        for l in (10, 20, 30, 40, 50):
            cells = []
            for suf in ("I", "Y"):
                per = np.array([np.mean([r["flip"] for r in P if r["set"] == S and r["layer"] == l and r["suf"] == suf and r["pid"] == p]) for p in pids])
                bs = per[rng.integers(0, len(per), (2000, len(per)))].mean(1)
                pt[f"{S}|L{l}|{suf}"] = [float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                cells.append(f"{per.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
            print(f"| {S} | L{l} | {cells[0]} | {cells[1]} |")
    summary["patch"] = pt
json.dump(summary, open(f"{RES}/E7_binding_tables_summary.json", "w"), indent=1)
# follow-up single-layer sets and band patches
for tag, pat, keyf, levels in (("patchb", "E7_binding_patchb_chunk*.json", "layer", (10, 20, 30, 40, 50)),
                               ("bandpatch", "E7_binding_bandpatch_chunk*.json", "band", ("all", "mid")),
                               ("flowmap", "E7_binding_patchd.json", "layer", (20, 24, 27, 30, 35, 40, 50)),
                               ("dirpatch", "E7_binding_dirpatch_chunk*.json", "sub", ("ref", "non", "both", "rand", "full"))):
    R = []
    for f in sorted(glob.glob(f"{RES}/{pat}")):
        R += json.load(open(f))["rows"]
    if not R:
        continue
    rng = np.random.default_rng(4)
    sets = sorted(set(r["set"] for r in R), key=lambda s: ["sys", "tmpl_sysend", "tmpl_modelhdr", "rest", "user", "suffix", "passage", "A_self", "A_user"].index(s))
    pids = sorted(set(r["pid"] for r in R))
    print(f"\n#### {tag} (flip fraction, 95% bootstrap CI over passages; n passages={len(pids)})\n")
    print(f"| set | {keyf} | flip 'I am a' | flip 'you are a' |")
    print("|---|---|---|---|")
    d = {}
    for S in sets:
        for lv in levels:
            cells = []
            for suf in ("I", "Y"):
                per = np.array([np.mean([r["flip"] for r in R if r["set"] == S and r[keyf] == lv and r["suf"] == suf and r["pid"] == p]) for p in pids])
                bs = per[rng.integers(0, len(per), (2000, len(per)))].mean(1)
                d[f"{S}|{lv}|{suf}"] = [float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                cells.append(f"{per.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
            print(f"| {S} | {lv} | {cells[0]} | {cells[1]} |")
    summary[tag] = d
json.dump(summary, open(f"{RES}/E7_binding_tables_summary.json", "w"), indent=1)
