CHUNK = 3
# E6_tool_audit_03_sweeps -- run the ACTUAL tool sweep (rt.run_theodolite_sweep, naive defaults: prepend_bos=True,
# monuments she/he/it) on self-reference prompts + controls, and record (i) what the tool displays and (ii) principled
# readouts (lens pron log-odds, real next-token gender probs) with correct single-BOS tokenization.
# Set CHUNK before exec'ing this file.
# PRE-STATED PREDICTIONS:
#  H1 Tool display metrics (x[she]-y[he] HUD coordinates; J++ she-vs-he rank) will NOT track persona gender reliably
#     and will be dominated by monument order / layer background; principled pron log-odds will (woman > man).
#  H2 Default self (A_default, B_gemma) sits near the woman/man midpoint on principled lens readouts at the self token;
#     at the third-person slot the real model prefers "it"/name over she/he for Gemma-self.
#  H3 B_gemma_human (Gemma = human friend) is strongly female (name prior), B_gemma (self) much less so.
#  H4 Double BOS changes real next-token distribution only slightly (KL < 0.05 nats) but shifts some displays.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
CONDS = build_conditions()
TRACK = ["she", "he", "it", "her", "him", "woman", "man", "they"]
FSET = {"she", "her", "hers", "herself", "woman", "women", "girl", "female", "lady", "mother", "sister", "daughter",
        "wife", "queen", "feminine", "She", "Her", "Woman", "Girl", "Female"}
MSET = {"he", "him", "his", "himself", "man", "men", "boy", "male", "guy", "father", "brother", "son", "husband",
        "king", "masculine", "He", "Him", "His", "Man", "Boy", "Male"}

WU = rt.lm_head_weight
fnw = rt.final_norm_weight.float()
mon_ids = [tid1(" she"), tid1(" he"), tid1(" it")]
Mraw = torch.stack([WU[i].float() for i in mon_ids])            # what Module 4 uses
Meff = Mraw * fnw                                               # effective logit directions
E_raw = gram_schmidt_orthonormalize_3d(Mraw[0], Mraw[1], Mraw[2])  # [3,D]
E_eff = gram_schmidt_orthonormalize_3d(Meff[0], Meff[1], Meff[2])
R_raw = (Mraw @ E_raw.T).tolist()  # monument coordinates in the displayed frame (rows she,he,it)
R_eff = (Meff @ E_eff.T).tolist()


def gcount(strs):
    f = sum(1 for s in strs if s.strip() in FSET)
    m = sum(1 for s in strs if s.strip() in MSET)
    return f, m


def kl(p_logits, q_logits):
    lp = F.log_softmax(p_logits.float(), -1); lq = F.log_softmax(q_logits.float(), -1)
    return float((lp.exp() * (lp - lq)).sum())


out = {"R_raw": R_raw, "R_eff": R_eff, "conds": {}}
t_start = time.time()
for cname in CHUNKS[CHUNK]:
    rows = []
    for pi, prompt in enumerate(CONDS[cname]):
        b = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=4,
                                          tracked_concepts=TRACK, prepend_bos=True)
        r = {"pi": pi, "first_ids": b.prompt_token_ids[:3], "target_tok": b.prompt_token_strs[b.target_pos],
             "generated": b.generated_text,
             "space_hack_triggered": bool(len(b.generated_token_ids) >= 2 and rt.tokenizer.decode([b.generated_token_ids[0]]).strip() == "")}
        r["xyz_raw"] = b.trajectory_3d.xyz.float().cpu().tolist()
        r["az"] = b.trajectory_3d.azimuth_deg.float().cpu().tolist()
        r["el"] = b.trajectory_3d.elevation_deg.float().cpu().tolist()
        HJ = torch.stack([b.causal_splits[l].resid_post_J.float() for l in range(60)])  # [60,D] double-BOS states
        r["proj_frac"] = (b.trajectory_3d.projection_norm.float().cpu() / HJ.norm(dim=-1).cpu()).tolist()
        r["xyz_eff"] = (HJ @ E_eff.T.to(HJ.device)).cpu().tolist()
        for k in ("jpp", "naive"):
            ro = b.jpp_layer_readouts if k == "jpp" else b.naive_layer_readouts
            r[f"{k}_rank"] = {t: [ro[l].tracked_token_ranks[t] for l in range(60)] for t in TRACK}
            r[f"{k}_logit"] = {t: [ro[l].tracked_token_logits[t] for l in range(60)] for t in TRACK}
            tops = [s for l in range(60) for s in ro[l].top_token_strs]
            r[f"{k}_top5_gcount"] = gcount(tops)
        r["jpp_top5_by_layer"] = [b.jpp_layer_readouts[l].top_token_strs for l in range(60)]
        pj = [s for v in b.pole_top_tokens_jpp.values() for s, _ in v]
        pn = [s for v in b.pole_top_tokens_naive.values() for s, _ in v]
        r["pole_jpp_gcount"] = gcount(pj); r["pole_naive_gcount"] = gcount(pn)
        r["pole_gendered_labels"] = {f"{l}:{rk}": v for (l, rk), v in b.pole_top_tokens_jpp.items()
                                     if any(s.strip() in FSET or s.strip() in MSET for s, _ in v)}
        # principled lens readouts on the tool's (double-BOS) states
        sc2 = lab.gender_scores(HJ.to(lab.dev))
        r["lens2"] = {k: v.float().cpu().tolist() for k, v in sc2.items() if k in ("pron", "noun", "she_he", "neuter_vs_human")}
        # module 1 audit
        r["m1_tool_dom"] = [float(b.causal_splits[l].ffn_dominance_ratio) for l in range(60)]
        cd, cf = [], []
        for l in range(60):
            a_c, f_c, s = corrected_split_J(b, l)
            cd.append(float(f_c.norm() / (a_c.norm() + f_c.norm() + 1e-8)))
            cf.append(float(F.cosine_similarity(b.causal_splits[l].delta_ffn_J.float(), f_c, dim=0)))
        r["m1_corr_dom"] = cd; r["m1_cos_ffn"] = cf
        r["n_fil"] = len(b.filaments)
        r["fil_src_layers"] = [f.src_layer for f in b.filaments]
        r["fil_abs_c"] = [abs(f.coupling_score) for f in b.filaments]
        r["cka_offdiag"] = float(b.cka_matrix[~torch.eye(60, dtype=torch.bool, device=b.cka_matrix.device)].mean())
        # correct single-BOS tokenization: lens readouts + real next-token distribution, vs double BOS
        ids1 = lab.ids(prompt)
        ids2 = list(b.prompt_token_ids)
        H1 = lab.forward_resid([ids1], positions=[[len(ids1) - 1]])[0, :, 0]  # [60,D]
        HJ1 = torch.stack([lab.lens(H1[l], l, "jpp") for l in range(60)])
        sc1 = lab.gender_scores(HJ1)
        r["lens1"] = {k: v.float().cpu().tolist() for k, v in sc1.items() if k in ("pron", "noun", "she_he", "neuter_vs_human")}
        lg1 = lab.next_logits(ids1); lg2 = lab.next_logits(ids2)
        r["kl_single_vs_double"] = kl(lg1, lg2)
        r["top1_same"] = int(lg1.argmax()) == int(lg2.argmax())
        r["top1_single"] = rt.tokenizer.decode([int(lg1.argmax())])
        g = lab.gender; lp1 = F.log_softmax(lg1, -1); lp2 = F.log_softmax(lg2, -1)
        P = lambda lp, ids: float(torch.logsumexp(lp[torch.tensor(ids, device=lp.device)], 0).exp())
        r["beh1"] = {"pF": P(lp1, g["pron_f"]), "pM": P(lp1, g["pron_m"]), "pIt": P(lp1, g["neuter"]), "pThey": P(lp1, g["they"])}
        r["beh2"] = {"pF": P(lp2, g["pron_f"]), "pM": P(lp2, g["pron_m"]), "pIt": P(lp2, g["neuter"]), "pThey": P(lp2, g["they"])}
        rows.append(r)
    out["conds"][cname] = rows
    print(cname, "done", f"{time.time() - t_start:.0f}s")
    import json as _j
    _j.dump(out, open(f"./experiments/results/E6_tool_audit_03_sweeps_c{CHUNK}.partial.json", "w"))
result = {"chunk": CHUNK, "conds": list(out["conds"].keys()), "elapsed": time.time() - t_start,
          "file": f"experiments/results/E6_tool_audit_03_sweeps_c{CHUNK}.partial.json"}
