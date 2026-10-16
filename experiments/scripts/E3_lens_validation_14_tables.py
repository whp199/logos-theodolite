# E3_lens_validation: generate markdown tables for the report from result JSONs (local).
import json, numpy as np, sys
R = "./experiments/results/"
B5 = ["L10-19", "L20-29", "L30-39", "L40-49", "L50-58"]
part = sys.argv[1] if len(sys.argv) > 1 else "a"
f3 = lambda v: f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]"
if part == "a":
    a = json.load(open(R + "E3_lens_validation_a_analysis.json"))
    for m in ("kl", "top1"):
        print(f"\n**{m.upper()}** (band mean over passages, 95% CI)\n")
        print("| lens | positions | " + " | ".join(B5) + " |"); print("|---|---|" + "---|" * len(B5))
        for kind in ("jpp", "jlens", "naive"):
            for cat in ("prose", "asst", "self", "user"):
                b = a["bands"][f"{cat}|{m}|{kind}"]
                print(f"| {kind} | {cat} | " + " | ".join(f3(b[x]) for x in B5) + " |")
    print("\n**Entropy-matched difference (assistant minus prose; self minus prose)**\n")
    print("| metric | lens | target | " + " | ".join(B5) + " |"); print("|---|---|---|" + "---|" * len(B5))
    for m in ("kl", "top1"):
        for kind in ("jpp", "jlens", "naive"):
            for t in ("asst", "self"):
                b = a["entropy_matched"][f"{m}|{kind}"][t]["diff_bands"]
                print(f"| {m} | {kind} | {t} | " + " | ".join(f"{b[x][0]:+.2f} [{b[x][1]:+.2f},{b[x][2]:+.2f}]" for x in B5) + " |")
    print("\n**Per-layer at L58 / L55 / L45 / L35 (prose | asst) top-1**")
    for kind in ("jpp", "jlens", "naive"):
        pl = a["per_layer"]
        print(kind, [round(pl[f"prose|top1|{kind}"]["mean"][l], 3) for l in (35, 45, 55, 58)], [round(pl[f"asst|top1|{kind}"]["mean"][l], 3) for l in (35, 45, 55, 58)])
    print("\n**corr(lens gender score, final gender score)**\n")
    print("| positions | lens | axis | " + " | ".join(B5) + " |"); print("|---|---|---|" + "---|" * len(B5))
    for k, v in a["gender_corr_with_final"].items():
        cat, kind, ax = k.split("|")
        print(f"| {cat} | {kind} | {ax} | " + " | ".join(f"{v['bands'][x][0]:+.2f} [{v['bands'][x][1]:+.2f},{v['bands'][x][2]:+.2f}]" for x in B5) + " |")
if part == "b":
    v = json.load(open(R + "E3_lens_validation_b_vjp_analysis.json"))
    BB = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
    bm = lambda arr: [np.nanmean([arr[l] for l in rg]) if not np.all(np.isnan([arr[l] for l in rg])) else np.nan for rg in BB.values()]
    print(f"n positions={v['n']}, groups={v['n_groups']}, subsets={v['subset_n']}")
    print("\n| direction | quantity | " + " | ".join(BB) + " |"); print("|---|---|" + "---|" * len(BB))
    for di, dn in enumerate(v["dirs"]):
        rows = [("cos(gbar, J++ pullback)", np.array(v["cos_gbar_jpp"]["mean"])[di]), ("  95% lo", np.array(v["cos_gbar_jpp"]["lo"])[di]), ("  95% hi", np.array(v["cos_gbar_jpp"]["hi"])[di]),
                ("cos(gbar, JLens pullback)", np.array(v["cos_gbar_jlens"]["mean"])[di]), ("  95% lo", np.array(v["cos_gbar_jlens"]["lo"])[di]), ("  95% hi", np.array(v["cos_gbar_jlens"]["hi"])[di]),
                ("cos(gbar, d) naive", np.array(v["cos_gbar_naive"]["mean"])[di]), ("cos(gbar, J++^T pullback)", np.array(v["cos_gbar_jppT"]["mean"])[di]),
                ("split-half cos(gbar_A,gbar_B)", np.array(v["splithalf"])[di]), ("SB ceiling", np.array(v["ceiling_SB"])[di]),
                ("disattenuated cos J++", np.array(v["disatt_cos_jpp"], dtype=float)[di]), ("disattenuated cos JLens", np.array(v["disatt_cos_jlens"], dtype=float)[di]),
                ("single-position cos(g, J++)", np.array(v["single_cos_jpp"])[di]), ("single-position cos(g, JLens)", np.array(v["single_cos_jlens"])[di]),
                ("coherence abs(gbar)/E abs(g)", np.array(v["coherence"])[di]),
                ("cos(gbar_chat, gbar_prose)", np.array(v["cos_chat_vs_prose"])[di]), ("split-half chat", np.array(v["splithalf_chat"])[di]), ("split-half prose", np.array(v["splithalf_prose"])[di]),
                ("cos(gbar_chat, JLens)", np.array(v["cos_gbar_chat_jlens"])[di]), ("cos(gbar_prose, JLens)", np.array(v["cos_gbar_prose_jlens"])[di]),
                ("cos(gbar_chat, J++)", np.array(v["cos_gbar_chat_jpp"])[di]), ("cos(gbar_prose, J++)", np.array(v["cos_gbar_prose_jpp"])[di]),
                ("cos(gbar_self, gbar_nonself)", np.array(v["cos_self_vs_asst_nonself"])[di])]
        for nm, arr in rows:
            print(f"| {dn} | {nm} | " + " | ".join(f"{x:+.3f}" if not np.isnan(x) else "n/a" for x in bm(arr)) + " |")
if part == "b2":
    v = json.load(open(R + "E3_lens_validation_b2_identity_analysis.json"))
    C = ["default", "nameless_AI", "Gemma", "Sarah_AI", "James_AI"]
    for cat in ("self", "nonself_asst"):
        for f in ("jpp_pron", "jppc_pron", "jpp_noun", "jlens_pron", "it_raw_pron", "it_norm_pron", "it_cent_pron", "it_raw_noun", "jpull_raw_pron", "it_raw_rand1", "it_raw_rand2"):
            if f"{cat}|{f}|L30-39" not in v["positions"]: continue
            print(f"\n**{cat} / {f}** position on woman(1)-man(0) scale [95% CI]; woman-man gap effect size d\n")
            print("| band | " + " | ".join(C) + " | gap d |"); print("|---|" + "---|" * (len(C) + 1))
            for b in B5:
                P = v["positions"][f"{cat}|{f}|{b}"]; s = v["sep"][f"{cat}|{f}|{b}"]
                print(f"| {b} | " + " | ".join(f"{P[c][0]:+.2f} [{P[c][1]:+.2f},{P[c][2]:+.2f}]" for c in C) + f" | {s['gap_d']:+.2f} |")
    print("\n**Agreement: position difference (A minus B), self tokens**\n")
    for k, vv in v["agree"].items():
        print(f"- {k}: " + "; ".join(f"{c} {x[0]:+.2f} [{x[1]:+.2f},{x[2]:+.2f}]" for c, x in vv.items()))
if part == "c":
    d = json.load(open(R + "E3_lens_validation_c_background.json"))
    BB = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
    bm = lambda v: [float(np.mean([v[l] for l in rg])) for rg in BB.values()]
    print("| lens | quantity | " + " | ".join(BB) + " |"); print("|---|---|" + "---|" * len(BB))
    for kind in ("jpp", "jlens", "naive"):
        g = d["gauss"][kind]
        print(f"| {kind} | Gaussian pron mean (sd) | " + " | ".join(f"{m:+.2f} ({s:.1f})" for m, s in zip(bm(g['pron_mean']), bm(g['pron_sd']))) + " |")
        print(f"| {kind} | Gaussian noun mean | " + " | ".join(f"{m:+.2f}" for m in bm(g['noun_mean'])) + " |")
        for nm in ("mu", "mu_prose", "mu_chat"):
            for k in ("pron", "noun"):
                print(f"| {kind} | {nm} {k} | " + " | ".join(f"{m:+.2f}" for m in bm(d[nm][kind][k])) + " |")
        for sname in ("prose_neutral", "chat_asst", "prose_bio_f", "prose_bio_m"):
            for v in ("raw", "meanpart", "cent"):
                for k in ("pron", "noun"):
                    arr = np.array(d["real"][sname][f"{kind}|{v}|{k}"]).mean(0)
                    print(f"| {kind} | {sname} {v} {k} | " + " | ".join(f"{m:+.2f}" for m in bm(arr)) + " |")
    c2 = np.array(d["real"]["prose_neutral"]["jpp|cos2mu|pron"]).mean(0)
    print("| - | cos^2(h, mu) neutral prose | " + " | ".join(f"{m:.2f}" for m in bm(c2)) + " |")
    print("| - | norm(mu)/mean norm(h) | " + " | ".join(f"{m:.2f}" for m in bm(np.array(d['mu_norm']) / np.array(d['mean_norm']))) + " |")
