# V7_tool_bugs_01_bos: (a) double BOS through the tool's own path + (b) monument-frame order dependence.
# Set CHUNK = (start, stop) over CHAT prompts before exec'ing, or RAW=True for the raw-prompt chunk.
# PRE-REGISTERED PREDICTIONS
#  A1 rt.run_theodolite_sweep(chat_string) (default prepend_bos=True) yields ids [2,2,105,...] for 100% of chat prompts.
#  A2 Real next-token distribution barely changes (mean KL < 0.05 nats; top-1 agreement >= 80%); greedy generations
#     mostly identical. Refutation of "it matters": KL ~0 and readouts within noise.
#  A3 Displayed quantities that average over all positions (CKA) or that depend on small lens logit gaps (tracked
#     she-he logit, azimuth) shift measurably (|d azimuth| median > 5 deg at L40-58 would count as display-relevant).
#  A4 For RAW strings, prepend_bos=False gives ZERO BOS and wrecks the output (KL >> 1 nat) -> the naive fix is unsafe.
#  B1 Permuting monuments (she,he,it) changes azimuth/elevation by tens of degrees (GS order dependence); the
#     coordinate attributed to a monument depends on its slot: coord(she) - coord(he) is positive under (she,he,it)
#     for most prompts incl. male contexts, and flips/shrinks under (he,she,it). An order-free (Loewdin) frame
#     gives identical per-monument coordinates under all 6 permutations (by construction; checked numerically).
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
from backend.theodolite_math import (build_monument_frame, project_trajectory_to_3d, gram_schmidt_orthonormalize_3d,
                                     compute_linear_cka_matrix)
TRACK = ["she", "he", "it", "they", "her", "him", "woman", "man"]
MON = ("she", "he", "it")
mon_ids = [rt.resolve_concept_token_id(m)[0] for m in MON]
RAWMODE = globals().get("RAW", False)
if RAWMODE:
    prompts = RAW_PROMPTS
    variants = (("withBOS", True), ("noBOS", False))      # raw: prepend True = single BOS (correct); False = no BOS
else:
    a, b_ = CHUNK
    prompts = chat_prompts()[a:b_]
    variants = (("double", True), ("single", False))      # chat: prepend True = double BOS (tool default)


def stack_traj_inputs(bd):
    cs = bd.causal_splits
    hJ = torch.stack([cs[l].resid_post_J for l in range(60)]).float()
    da = torch.stack([cs[l].delta_attn_J for l in range(60)]).float()
    df = torch.stack([cs[l].delta_ffn_J for l in range(60)]).float()
    return hJ, da, df


def loewdin(W):
    W = W.float()
    G = W @ W.T
    ev, U = torch.linalg.eigh(G)
    return (U @ torch.diag(ev.rsqrt()) @ U.T) @ W


def summarize(bd):
    o = {"ids_first3": bd.prompt_token_ids[:3], "n": len(bd.prompt_token_ids), "gen": bd.generated_text,
         "gen_ids": bd.generated_token_ids}
    o["jpp_top1"] = [bd.jpp_layer_readouts[l].top_token_strs[0] for l in range(60)]
    o["naive_top1"] = [bd.naive_layer_readouts[l].top_token_strs[0] for l in range(60)]
    o["jpp_logit"] = {k: [bd.jpp_layer_readouts[l].tracked_token_logits.get(k) for l in range(60)] for k in TRACK}
    o["jpp_rank"] = {k: [bd.jpp_layer_readouts[l].tracked_token_ranks.get(k) for l in range(60)] for k in TRACK}
    o["naive_logit"] = {k: [bd.naive_layer_readouts[l].tracked_token_logits.get(k) for l in range(60)] for k in TRACK}
    o["az"] = bd.trajectory_3d.azimuth_deg.float().cpu().tolist()
    o["el"] = bd.trajectory_3d.elevation_deg.float().cpu().tolist()
    o["xyz"] = bd.trajectory_3d.xyz.float().cpu().tolist()
    C = bd.cka_matrix.float().cpu()
    o["cka_offdiag"] = float(C[~torch.eye(60, dtype=torch.bool)].mean())
    o["poles"] = [bd.bipolar_poles[l].neuron_indices.cpu().tolist() for l in range(60)]
    o["fil"] = sorted({(f.src_layer, f.src_neuron, f.dst_layer, f.dst_neuron) for f in bd.filaments})
    o["ffn_dom"] = [float(bd.causal_splits[l].ffn_dominance_ratio) for l in range(60)]
    o["elapsed_ms"] = bd.elapsed_ms
    return o


rows = []
t0 = time.time()
for pi, p in enumerate(prompts):
    row = {"prompt": p[-120:]}
    bundles = {}
    for tag, pb in variants:
        bd = await rt.run_theodolite_sweep(p, monument_concepts=MON, max_new_tokens=6, tracked_concepts=TRACK,
                                           prepend_bos=pb)
        row[tag] = summarize(bd)
        # real next-token logits on exactly the ids the tool used
        row[tag]["_ids"] = bd.prompt_token_ids
        bundles[tag] = bd
    # real next-token KL between variants (on the tool's own ids)
    t1, t2 = variants[0][0], variants[1][0]
    lg1 = lab.next_logits(bundles[t1].prompt_token_ids); lg2 = lab.next_logits(bundles[t2].prompt_token_ids)
    lp1, lp2 = F.log_softmax(lg1, -1), F.log_softmax(lg2, -1)
    row["kl_ref_vs_alt"] = float((lp2.exp() * (lp2 - lp1)).sum())   # KL(second || first)
    row["top1_same"] = bool(int(lg1.argmax()) == int(lg2.argmax()))
    row["top1"] = [rt.tokenizer.decode([int(lg1.argmax())]), rt.tokenizer.decode([int(lg2.argmax())])]
    # (b) monument-order analysis on the reference-tokenized bundle (single BOS for chat; withBOS for raw)
    ref = bundles["single" if not RAWMODE else "withBOS"]
    hJ, da, df = stack_traj_inputs(ref)
    rn = torch.stack([ref.causal_splits[l].resid_post_J for l in range(60)]).float()  # placeholder for resid_post (norm only)
    perm_res = {}
    for perm in itertools.permutations(range(3)):
        tids = tuple(mon_ids[i] for i in perm)
        basis, _ = build_monument_frame(tids, rt.lm_head_weight, rt.jpp_lenses, pullback_per_layer=False)
        tr = project_trajectory_to_3d(rn, hJ, da, df, basis, use_jpp_states=True)
        xyz = tr.xyz.float().cpu()
        # coordinate attributed to each monument (by name) in this permutation
        coord = {MON[perm[k]]: xyz[:, k].tolist() for k in range(3)}
        perm_res["".join(MON[i][0:2] for i in perm)] = {"az": tr.azimuth_deg.float().cpu().tolist(),
                                                        "el": tr.elevation_deg.float().cpu().tolist(), "coord": coord}
    # check canonical perm reproduces the tool's own trajectory
    row["perm_check_max_abs_az_diff"] = float(max(abs(a_ - b2) for a_, b2 in zip(perm_res["shheit"]["az"], row["single" if not RAWMODE else "withBOS"]["az"])))
    # Loewdin (order-free) coordinates: same for all permutations by construction; verify numerically
    W = rt.lm_head_weight[torch.tensor(mon_ids, device=lab.dev)].float()
    Wn = W / W.norm(dim=-1, keepdim=True)
    lw = {}
    for perm in [(0, 1, 2), (1, 0, 2), (2, 1, 0)]:
        Bl = loewdin(Wn[list(perm)])
        c = (hJ @ Bl.T).cpu()
        lw["".join(MON[i][0:2] for i in perm)] = {MON[perm[k]]: c[:, k].tolist() for k in range(3)}
    row["loewdin"] = lw
    # contrastive fnw-weighted she-he readout (what the HUD should show): <normed hJ, fnw*(W_she - W_he)> = logit diff
    row["perm"] = perm_res
    row["frac_hJ_captured"] = (torch.stack([torch.tensor(v) for v in perm_res["shheit"]["coord"].values()], 1).norm(dim=1) /
                               hJ.norm(dim=1).cpu()).tolist()
    for tag, _ in variants:
        row[tag].pop("_ids", None)
    rows.append(row)
    del bundles, ref, hJ, da, df, rn
    torch.cuda.empty_cache()
    print(f"[{pi}] {time.time()-t0:.0f}s KL={row['kl_ref_vs_alt']:.4f} top1={row['top1']} gens={[row[t]['gen'][:30] for t,_ in variants]}")

tag = "raw" if RAWMODE else f"chat{CHUNK[0]}_{CHUNK[1]}"
json.dump(rows, open(RES + f"V7_tool_bugs_01_bos_{tag}.partial.json", "w"))
result = {"n": len(rows), "kl": [r["kl_ref_vs_alt"] for r in rows], "top1_same": [r["top1_same"] for r in rows],
          "ids": [[r[t]["ids_first3"] for t, _ in variants] for r in rows]}
