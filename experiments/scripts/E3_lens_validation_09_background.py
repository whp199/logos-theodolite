# E3_lens_validation (c) lens background prior.
# PRE-STATED PREDICTIONS:
#  P10 Isotropic Gaussian inputs give mean gender score ~0 under every lens (a linear map of a zero-mean symmetric
#      input cannot create a sign bias), so the layer-dependent background is NOT an intrinsic property of J alone.
#  P11 The per-layer MEAN residual mu_l (dominated by a few massive/DC directions) maps through each lens to a vector
#      with nonzero pron/noun projections that reproduce the background pattern (L10-19 male, L40-49 pron+ / noun-).
#  P12 Mean-centering states (h - mu_l) removes most of the background (neutral prose ~0 on both axes) while
#      preserving between-condition differences.
import json, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
mp = torch.load(E3_RES + "E3_lens_validation_musum_prose.pt"); mc = torch.load(E3_RES + "E3_lens_validation_musum_chat.pt")
mu_p = (mp["mu_sum"] / mp["n"]).float(); mu_c = (mc["mu_sum"] / mc["n"]).float()
mu = ((mp["mu_sum"] + mc["mu_sum"]) / (mp["n"] + mc["n"])).float()     # [60, D]
mnorm = ((mp["norm_sum"] + mc["norm_sum"]) / (mp["n"] + mc["n"])).float()  # mean ||h|| per layer
torch.save({"mu": mu, "mu_prose": mu_p, "mu_chat": mu_c, "mean_norm": mnorm}, E3_RES + "E3_lens_validation_mu.pt")
KS = ["jpp", "jlens", "naive"]
g = torch.Generator().manual_seed(11)
N = 512
res = {"gauss": {}, "mu": {}, "mu_prose": {}, "mu_chat": {}, "mu_top": {}, "mean_norm": mnorm.tolist(), "mu_norm": mu.norm(dim=-1).tolist(),
       "cos_mu_prose_chat": F.cosine_similarity(mu_p, mu_c, dim=-1).tolist()}
for kind in KS:
    gp, gn, gps, gns = [], [], [], []
    for nm, M_ in (("mu", mu), ("mu_prose", mu_p), ("mu_chat", mu_c)):
        acc = {"pron": [], "noun": [], "neuter_vs_human": []}
        for l in range(60):
            sc = lab.gender_scores(lab.lens(M_[l:l + 1], l, kind))
            for k in acc: acc[k].append(float(sc[k][0]))
        res[nm][kind] = acc
    for l in range(60):
        X = torch.randn(N, 5376, generator=g)
        X = X / X.norm(dim=-1, keepdim=True) * mnorm[l]
        sc = lab.gender_scores(lab.lens(X, l, kind))
        gp.append(float(sc["pron"].mean())); gps.append(float(sc["pron"].std()))
        gn.append(float(sc["noun"].mean())); gns.append(float(sc["noun"].std()))
    res["gauss"][kind] = {"pron_mean": gp, "pron_sd": gps, "noun_mean": gn, "noun_sd": gns}
    res["mu_top"][kind] = {str(l): lab.topk(lab.lens(mu[l:l + 1], l, kind), 12) for l in (5, 12, 15, 18, 25, 33, 42, 45, 48, 55, 58)}
# (iii) real states: raw vs centered, plus decomposition of the raw score into mean and deviation parts
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
sets = {"prose_neutral": [lab.ids(t) for tag, t in PROSE if tag == "neutral"][:12],
        "prose_bio_f": [lab.ids(t) for tag, t in PROSE if tag == "bio_f"],
        "prose_bio_m": [lab.ids(t) for tag, t in PROSE if tag == "bio_m"],
        "chat_asst": [lab.ids(lab.chat(c["q"], assistant=c["a"])) for c in chats[12:24]]}
real = {}
mu_d = mu.to(lab.dev)
DG = {'pron': lab.gender_direction('pron').float(), 'noun': lab.gender_direction('noun').float()}
for sname, idl in sets.items():
    per = {f"{kind}|{v}|{k}": [] for kind in KS for v in ("raw", "cent", "meanpart", "lin", "cos2mu") for k in ("pron", "noun")}
    for ids in idl:
        if sname == "chat_asst":
            pos = e3_chat_positions(lab, ids, SELF_TOKS)["asst"]
        else:
            pos = list(range(4, len(ids)))
        H = lab.forward_resid([ids])[0][:, pos].to(lab.dev)  # [60,P,D]
        rows = {k: [] for k in per}
        for l in range(60):
            h = H[l]; hc = h - mu_d[l]
            c2 = float(F.cosine_similarity(h, mu_d[l].unsqueeze(0), dim=-1).pow(2).mean())
            for kind in KS:
                hJ = lab.lens(h, l, kind).float(); muJ = lab.lens(mu_d[l:l + 1], l, kind).float()
                sr = lab.gender_scores(hJ); scn = lab.gender_scores(lab.lens(hc, l, kind))
                rms = hJ.pow(2).mean(-1).sqrt()
                for k in ("pron", "noun"):
                    dk = DG[k]
                    rows[f"{kind}|raw|{k}"].append(float(sr[k].mean())); rows[f"{kind}|cent|{k}"].append(float(scn[k].mean()))
                    # linear (softcap-free) decomposition of the raw score: <muJ,d>/rms(hJ) + <(h-mu)J,d>/rms(hJ)
                    rows[f"{kind}|meanpart|{k}"].append(float(((muJ @ dk) / rms).mean()))
                    rows[f"{kind}|lin|{k}"].append(float(((hJ @ dk) / rms).mean()))
                    rows[f"{kind}|cos2mu|{k}"].append(c2)
        for k in per: per[k].append(rows[k])
    real[sname] = per
    print(sname, "done", round(time.time() - t0, 1), flush=True)
res["real"] = real
json.dump(res, open(E3_RES + "E3_lens_validation_c_background.json", "w"))
B = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
bm = lambda v: " ".join(f"{b}:{np.mean([v[l] for l in r]):+6.2f}" for b, r in B.items())
for kind in KS:
    print(f"--- {kind}")
    print(" gauss pron mean", bm(res["gauss"][kind]["pron_mean"]), "| sd", bm(res["gauss"][kind]["pron_sd"]))
    print(" gauss noun mean", bm(res["gauss"][kind]["noun_mean"]))
    print(" mu    pron     ", bm(res["mu"][kind]["pron"]), "\n mu    noun     ", bm(res["mu"][kind]["noun"]))
    for sname in real:
        for v in ("raw", "cent", "meanpart", "lin"):
            for k in ("pron", "noun"):
                arr = np.array(real[sname][f"{kind}|{v}|{k}"]).mean(0)
                print(f" {sname:13s} {v:4s} {k:4s}", bm(arr))
print("cos2(h,mu) chat/prose", bm(np.array(real["prose_neutral"]["jpp|cos2mu|pron"]).mean(0)))
for kind in KS:
    print(kind, {l: [t for t, p in v[:6]] for l, v in res["mu_top"][kind].items()})
result = {"ok": 1}
