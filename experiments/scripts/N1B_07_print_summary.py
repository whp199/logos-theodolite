#!/usr/bin/env python3
"""Print the key N1B numbers (metric / null / lift / diff CI) from N1B_metrics.json and N1B_semantic.json."""
import json
R = "./experiments/results/"
M = json.load(open(R + "N1B_metrics.json"))
try:
    SEM = json.load(open(R + "N1B_semantic.json"))
except FileNotFoundError:
    SEM = {}

def line(name, d, fmt="{:.4f}"):
    o = d["overall"]
    lift = o.get("lift")
    ls = f"{lift:.2f}" if isinstance(lift, (int, float)) and lift == lift and lift != float("inf") else str(lift)
    lci = o.get("lift_ci")
    lcs = f"[{lci[0]:.2f},{lci[1]:.2f}]" if lci else "[n/a]"
    print(f"{name:52s} {fmt.format(o['metric'])} null {fmt.format(o['null'])} lift {ls} {lcs} "
          f"diff {o['diff']:+.4f} [{o['diff_ci'][0]:+.4f},{o['diff_ci'][1]:+.4f}] p_perm={o.get('perm_p_one_sided', float('nan')):.4f}")
    print("    per-layer metric/null: " + " ".join(f"L{l}:{fmt.format(v['metric'])}/{fmt.format(v['null'])}" for l, v in d["per_layer"].items()))

print("== corpus:", M["n_positions"], "non-BOS positions;", M["src_positions"], "; top-1% =", M["top1pct_n"], "positions/pole")
print("== CEILING split-half empirical labels")
for k in ("splithalf_empirical_top10_vs_top50|count|norm", "splithalf_empirical_top10_vs_top50|enrich|norm"):
    line(k, M["ceilings"][k])
print("   split-half token-response Spearman:", json.dumps(M["ceilings"]["splithalf_token_response_spearman"]["overall"]))
print("== TOOL input label vs empirical input tokens (per pole, full vocab, normalized-string match)")
for v in ("G_jpp", "G_jpp_pre", "G_naive", "G_embed"):
    for w in ("in_count", "in_enrich"):
        line(f"{v}|{w}", M["input_precision_at10"][f"{v}|{w}|fullvocab|norm"])
print("== strict token-id match (task definition)")
for v in ("G_jpp", "G_jpp_pre", "G_naive"):
    line(f"{v}|in_count|strict", M["input_precision_at10"][f"{v}|in_count|fullvocab|strict"])
print("== attested-vocab re-ranking (top-10 among tokens seen >=3x)")
for v in ("G_jpp", "G_naive", "G_embed"):
    line(f"{v}|in_enrich|attested", M["input_precision_at10"][f"{v}|in_enrich|attested|norm"])
print("== neuron level: gate label vs top-1% gate pre-activation tokens; Spearman with token-mean g")
for v in ("G_jpp", "G_jpp_pre", "G_naive", "G_naive_pre", "G_embed"):
    line(f"{v}|gate_enrich", M["gate_label_neuron_level"][f"{v}|gate_enrich|fullvocab|norm"])
    line(f"{v}|spearman_tokmean_g", M["gate_label_neuron_level"][f"{v}|spearman_tokmean_g"])
print("== strong events (top-0.1%)")
for k, d in M["top0.1pct_precision_at10"].items():
    line(k, d)
print("== static vs NEXT-token empirical sets")
for k in ("G_jpp|nx_enrich|fullvocab|norm", "G_naive|nx_enrich|fullvocab|norm", "O_jvindex|nx_count|fullvocab|norm",
          "O_jvindex|nx_enrich|fullvocab|norm", "O_naive|nx_count|fullvocab|norm", "O_naive|nx_enrich|fullvocab|norm"):
    line(k, M["vs_next_token_precision_at10"][k])
print("== hit distribution", json.dumps(M["hit_distribution"], indent=0))
D = M["diagnostics"]
print("== bipolarity overall", json.dumps(D["bipolarity"]["overall"]))
print("== mechanism", json.dumps({l: {k: round(v, 3) for k, v in d.items()} for l, d in D["mechanism_top1pct"].items() if l != "note"}))
print("== pole distinctness", json.dumps(D["pole_distinctness"]))
print("== label diversity (unique frac of 240 top-10 tokens/layer)",
      json.dumps({v: {l: round(x["unique_frac"], 2) for l, x in d.items()} for v, d in D["label_diversity"].items() if v in ("G_jpp", "G_naive", "O_jvindex")}))
print("== prior overlap", json.dumps({v: d for v, d in D["prior_overlap"].items() if v in ("G_jpp", "G_jpp_pre", "G_naive", "O_jvindex")}))
print("== top10 softmax mass (median)", json.dumps(D.get("top10_softmax_mass_median")))
fc = D["firing_and_concentration"]
print("== concentration", json.dumps({l: {k: round(v, 4) for k, v in d.items()} for l, d in fc.items()}))
if SEM:
    print("== SEMANTIC centroid cosine (W_E, centred)")
    for k, d in SEM.items():
        if isinstance(d, dict) and "overall" in d:
            o = d["overall"]
            print(f"   {k:28s} {o['metric']:.4f} null {o['null']:.4f} diff {o['diff']:+.4f} [{o['diff_ci'][0]:+.4f},{o['diff_ci'][1]:+.4f}] p={o.get('perm_p_one_sided', float('nan'))}")
