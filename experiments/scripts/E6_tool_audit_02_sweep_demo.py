# E6_tool_audit_02_sweep_demo -- what a naive user sees from rt.run_theodolite_sweep on a self-reference prompt.
# Predictions: (a) prompt_token_ids start [2,2,...] with prepend_bos=True (double BOS); (b) J++ top-5 at the self
# token are dominated by continuation tokens ("'m", "am", ...), not gender tokens; (c) tool ffn_dominance differs
# strongly from corrected decomposition at late layers (layer_scalar bug); (d) azimuth/elevation wrt (she,he,it)
# move a lot across layers in ways unrelated to she-vs-he log-odds.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
TRACK = ["she", "he", "it", "her", "him", "woman", "man", "they"]
prompt = lab.chat("Tell me about yourself.", assistant="Hi! I")
out = {}
for pb in (True, False):
    t0 = time.time()
    b = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=6,
                                      tracked_concepts=TRACK, prepend_bos=pb)
    s = summarize_bundle(b, keep_cka=True)
    s["wall_s"] = time.time() - t0
    out[f"prepend_bos={pb}"] = s
    print(f"prepend_bos={pb}: first ids {s['first_ids']} target={s['target_tok']!r} gen={s['generated']!r} wall={s['wall_s']:.1f}s")
    print("  az(deg) L10..59 step5:", [round(s['azimuth_deg'][l]) for l in range(10, 60, 5)])
    print("  el(deg) L10..59 step5:", [round(s['elevation_deg'][l]) for l in range(10, 60, 5)])
    print("  proj frac of ||hJ||:", [round(s['proj_frac_of_hJ'][l], 4) for l in range(10, 60, 5)])
    print("  jpp rank she:", [s['jpp_tracked_rank']['she'][l] for l in range(10, 60, 5)])
    print("  jpp rank he :", [s['jpp_tracked_rank']['he'][l] for l in range(10, 60, 5)])
    print("  jpp rank it :", [s['jpp_tracked_rank']['it'][l] for l in range(10, 60, 5)])
    print("  naive rank she:", [s['naive_tracked_rank']['she'][l] for l in range(10, 60, 5)])
    print("  naive rank he :", [s['naive_tracked_rank']['he'][l] for l in range(10, 60, 5)])
    print("  jpp top5:", {l: v[:3] for l, v in s['jpp_top5'].items()})
    print("  tool ffn_dom:", [round(s['m1_tool_ffn_dominance'][l], 2) for l in range(0, 60, 6)] + [round(s['m1_tool_ffn_dominance'][59], 2)])
    print("  corr ffn_dom:", [round(s['m1_corrected_ffn_dominance'][l], 2) for l in range(0, 60, 6)] + [round(s['m1_corrected_ffn_dominance'][59], 2)])
    print("  cos(tool ffn, true ffn):", [round(s['m1_cos_tool_vs_true_ffn'][l], 2) for l in range(0, 60, 6)] + [round(s['m1_cos_tool_vs_true_ffn'][59], 2)])
    print("  n_filaments:", s['n_filaments'], "top:", s['filaments_top'][:3])
    print("  cka mean offdiag", round(s['cka_mean_offdiag'], 3), "adjacent", round(s['cka_adjacent_mean'], 3))
    print("  pole labels L35:", {k: v for k, v in s['pole_labels_jpp'].items() if k.startswith('35:')})
    print("  pole labels naive L35:", {k: v for k, v in s['pole_labels_naive'].items() if k.startswith('35:')})
result = out
