#!/usr/bin/env python3
# N1A_04_gender -- OFFLINE bonus: the tool's 'gender neurons' (LarqlVaultEngine.query_nearest_edges('she'/'he',
# layer in [30,40,50], top_k=2) -> 12 poles). Same static-vs-causal comparison as N1A_03 + a direct test:
# does steering the pole (sign chosen by the tool) causally raise log p(she) vs log p(he)?
# Null: the 48 random sampled poles of the same layer (same k-rule for delta).
# PRE-STATED PREDICTIONS:
#  G1 all 12 poles have ' she'/' he' (or close variants) in their static jpp top tokens (by construction).
#  G2 causally, most of them do NOT selectively push she-vs-he: |c[she]-c[he]| is within the random-pole range
#     (percentile < 95) for the majority; they push pronoun-generic tokens (she AND he together), since W_U[she] and
#     W_U[he] are highly collinear and cosine-to-W_U[she] selects shared-pronoun directions.
#  G3 c[she] + c[he] (generic pronoun push) is elevated vs random poles for >= half of them.
import json, os, glob
import torch

ROOT = "./experiments/results"
CACHE = f"{ROOT}/N1A_cache"
exec(open("./experiments/scripts/N1A_03_metrics.py").read().split("def main():")[0])

PF = [" she", " her", " hers", " herself", " She", " Her"]
PM = [" he", " him", " his", " himself", " He", " His"]


def tid(s):
    e = tok().encode(s, add_special_tokens=False)
    assert len(e) == 1, (s, e)
    return e[0]


def main():
    gz = torch.load(f"{CACHE}/gender_static.pt")
    gp = gz["poles"]
    she, he = tid(" she"), tid(" he")
    pf = [tid(s) for s in PF]; pm = [tid(s) for s in PM]
    out = {"tags": TAGS, "poles": [], "summary": {}}
    for tag in TAGS:
        rows = []
        for l in (30, 40, 50):
            cz = torch.load(f"{CACHE}/causal_{tag}_L{l}.pt")
            meta = cz["poles"]; C = cz["c"].float()
            samp = [i for i, m in enumerate(meta) if m["set"] == "sample"]
            gidx = {(m["n"], m["s"]): i for i, m in enumerate(meta) if m["set"] == "gender"}
            Cs = C[samp]
            ss_null = (Cs[:, she] - Cs[:, he])
            gen_null = (Cs[:, pf].mean(1) - Cs[:, pm].mean(1))
            sum_null = (Cs[:, she] + Cs[:, he]) / 2 - Cs.mean(1)
            st = gz["static"][l]; pri = torch.load(f"{CACHE}/prior_L{l}.pt")
            stS = torch.load(f"{CACHE}/static_L{l}.pt")
            # static methods for sampled poles (for null of agreement metrics)
            sp = [(n, s) for n in NEURONS[l] for s in (1, -1)]
            SMs = static_methods(l, stS, pri, [(NEURONS[l].index(n), s) for n, s in sp])
            RCs = zrows(avg_rank(Cs))
            lp = [g for g in gp if g["layer"] == l]
            for gi, g in enumerate(lp):
                r = st["neurons"].index(g["neuron"])
                c = C[gidx[(g["neuron"], g["sign"])]]
                c_opp = C[gidx[(g["neuron"], -g["sign"])]]
                SM = static_methods(l, st, pri, [(r, g["sign"])])
                rc = zrows(avg_rank(c[None]))
                rec = {"concept": g["concept"], "layer": l, "neuron": g["neuron"], "sign": g["sign"], "cos": g["cos"],
                       "kl": meta[gidx[(g["neuron"], g["sign"])]]["kl_mean"],
                       "delta": meta[gidx[(g["neuron"], g["sign"])]]["delta"],
                       "c_she_minus_he": float(c[she] - c[he]),
                       "c_she_minus_he_pct_vs_random_abs": float((ss_null.abs() < abs(float(c[she] - c[he]))).float().mean() * 100),
                       "c_pronF_minus_M": float(c[pf].mean() - c[pm].mean()),
                       "c_pronF_minus_M_pct_vs_random_abs": float((gen_null.abs() < abs(float(c[pf].mean() - c[pm].mean()))).float().mean() * 100),
                       "c_sheHe_generic": float((c[she] + c[he]) / 2 - c.mean()),
                       "c_sheHe_generic_pct_vs_random": float((sum_null < float((c[she] + c[he]) / 2 - c.mean())).float().mean() * 100),
                       "opp_sign_c_she_minus_he": float(c_opp[she] - c_opp[he]),
                       "rank_she_in_c": int((c > c[she]).sum()) + 1, "rank_he_in_c": int((c > c[he]).sum()) + 1,
                       "causal_top10": [dec(t) for t in c.topk(10).indices.tolist()],
                       "causal_bottom10": [dec(t) for t in (-c).topk(10).indices.tolist()],
                       "tool_jv": g["tool_jv"], "tool_nv": g["tool_nv"]}
                for m in ("jpp", "naive", "jpp_z", "naive_z", "jpp_zn"):
                    S = SM[m][0]
                    rec[f"{m}_top10"] = [dec(t) for t in S.topk(10).indices.tolist()]
                    rec[f"{m}_rank_she"] = int((S > S[she]).sum()) + 1
                    rec[f"{m}_rank_he"] = int((S > S[he]).sum()) + 1
                    rs = zrows(avg_rank(S[None]))
                    rho = float((rs @ rc.T)[0, 0])
                    same = [i for i, (n, s) in enumerate(sp) if s == g["sign"]]
                    rho_null = float((rs @ RCs[same].T).mean())
                    p10 = len(set(S.topk(10).indices.tolist()) & set(c.topk(100).indices.tolist())) / 10
                    p10_null = sum(len(set(S.topk(10).indices.tolist()) & set(Cs[q].topk(100).indices.tolist())) / 10 for q in same) / len(same)
                    rec[f"{m}_rho"] = rho; rec[f"{m}_rho_null"] = rho_null
                    rec[f"{m}_p10"] = p10; rec[f"{m}_p10_null"] = p10_null
                rows.append(rec)
                print(tag, g["concept"], l, g["neuron"], g["sign"], f"she-he {rec['c_she_minus_he']:+.4f} (pct {rec['c_she_minus_he_pct_vs_random_abs']:.0f})",
                      f"generic pct {rec['c_sheHe_generic_pct_vs_random']:.0f}", "jpp rho", round(rec["jpp_rho"], 3), "null", round(rec["jpp_rho_null"], 3), flush=True)
        out["poles"] += [dict(r, tag=tag) for r in rows]
        sel = [r for r in rows]
        correct_dir = [((r["c_she_minus_he"] > 0) == (r["concept"] == "she")) for r in sel]
        out["summary"][tag] = {
            "n": len(sel),
            "frac_correct_direction_she_vs_he": sum(correct_dir) / len(sel),
            "frac_selective_pct95": sum(r["c_she_minus_he_pct_vs_random_abs"] >= 95 for r in sel) / len(sel),
            "frac_generic_pronoun_pct95": sum(r["c_sheHe_generic_pct_vs_random"] >= 95 for r in sel) / len(sel),
            "median_she_minus_he": float(torch.tensor([r["c_she_minus_he"] for r in sel]).median()),
            "jpp_rho_lift": boot([r["jpp_rho"] - r["jpp_rho_null"] for r in sel]),
            "naive_rho_lift": boot([r["naive_rho"] - r["naive_rho_null"] for r in sel]),
            "jpp_p10_lift": boot([r["jpp_p10"] - r["jpp_p10_null"] for r in sel]),
            "naive_p10_lift": boot([r["naive_p10"] - r["naive_p10_null"] for r in sel]),
            "frac_jpp_top10_has_she_or_he": sum((" she" in r["jpp_top10"]) or (" he" in r["jpp_top10"]) or ("she" in r["jpp_top10"]) or ("he" in r["jpp_top10"]) for r in sel) / len(sel),
        }
    json.dump(out, open(f"{ROOT}/N1A_gender.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
