#!/usr/bin/env python3
# N1C_05_audit_dump -- pick the 48 audit poles (6 per layer, seeded) and dump human-readable evidence in two parts:
#   part A (activation side: contexts, input/next tokens)  -> judged FIRST, without seeing part B
#   part B (output side: causal top/bottom tokens, J-VINDEX, naive, gate label, metrics)
# Run: .venv/bin/python experiments/scripts/N1C_05_audit_dump.py <outdir>
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from N1C_common import *

outdir = sys.argv[1]
rng = np.random.default_rng(20261016)
picks = []
for l in LAYERS:
    rows = [r for r, (ll, _, _) in enumerate(POLES) if ll == l]
    picks += sorted(rng.choice(rows, 6, replace=False).tolist())
json.dump({"seed": 20261016, "rows": picks, "poles": [POLES[r] for r in picks]}, open(f"{ROOT}/N1C_audit_picks.json", "w"), indent=1)

L1B = json.load(open(f"{ROOT}/N1B_labels.json"))
L1A = json.load(open(f"{ROOT}/N1A_labels.json"))
PP = json.load(open(f"{ROOT}/N1C_per_pole.json"))
D = load_corpus_acts()
act, corpus = D["act"], D["corpus"]
N = len(act); ntop = int(math.ceil(0.01 * N))
S = np.load(f"{ROOT}/N1B_static.npz")

def toks(lst, k, withcount=True):
    return " | ".join((f"{repr(x[0])}({x[2]})" if withcount else repr(x[0])) for x in lst[:k])

A_lines, B_lines = [], []
for k, r in enumerate(picks):
    l, n, s = POLES[r]
    pb = L1B["poles"][r]; nb = L1B["neurons"][r // 2]; pa = L1A[r]
    assert (pb["layer"], pb["neuron"], pb["pole"]) == (l, n, "+" if s == 1 else "-")
    assert (pa["layer"], pa["neuron"], pa["sign"]) == (l, n, s)
    sa = s * act[:, r // 2]
    top = np.argpartition(-sa, ntop)[:ntop]; top = top[np.argsort(-sa[top])]; top = top[sa[top] > 0]
    sel = list(top[:10]) + sorted(rng.choice(top[10:], min(6, len(top) - 10), replace=False).tolist(), key=lambda i: -sa[i])
    hdr = f"=== #{k:02d} row{r} L{l} n{n} pole{'+' if s == 1 else '-'}"
    A_lines.append(hdr + f"  strong_pole={pb['is_strong_pole']} max={pb['max']:.2f} q999={pb['q999']:.2f} "
                   f"fire>10%max={pb['fire_freq_gt10pct_own']*100:.3f}% src={pb['src_mix_top']} bipolar_ratio={nb['bipolarity_max_ratio']:.2f}")
    A_lines.append("  IN enriched : " + " | ".join(f"{repr(x[0])}({x[2]})" for x in pb["top_input_enriched"][:12]))
    A_lines.append("  IN count    : " + toks(pb["top_input_tokens"], 10))
    A_lines.append("  NEXT enrich : " + " | ".join(f"{repr(x[0])}({x[2]})" for x in pb["top_next_enriched"][:12]))
    for j, i in enumerate(sel):
        tag = "top" if j < 10 else "smp"
        A_lines.append(f"  [{tag} {sa[i]:6.2f} {D['src'][i][:6]}] " + context_str(corpus, int(D['cid'][i]), int(D['pos'][i]), 16, 4))
    B_lines.append(hdr + f"  kl_kA={pa['kl_kA']:.4f} split_rho={pa['split_rho_kA']:.2f} full_P@10(JV->C)={PP['full_JV->C'][r]:.2f} "
                   f"P@10_A(C->Enx)={PP['C->Enx'][r]:.2f}")
    B_lines.append("  CAUSAL top  : " + " | ".join(repr(t) for t, v in pa["causal_kA"][:15]))
    B_lines.append("  CAUSAL bot  : " + " | ".join(repr(t) for t, v in pa["causal_kA_bottom"][:8]))
    B_lines.append("  J-VINDEX    : " + " | ".join(repr(t) for t, v in pa["jpp"][:12]))
    B_lines.append("  naive       : " + " | ".join(repr(t) for t, v in pa["naive"][:10]))
    B_lines.append("  gate(tool)  : " + " | ".join(repr(t) for t in pb["static_labels_top10"]["G_jpp"]))
    B_lines.append("  gate_embed  : " + " | ".join(repr(t) for t in pb["static_labels_top10"]["G_embed"]))
open(f"{outdir}/audit_A.txt", "w").write("\n".join(A_lines))
open(f"{outdir}/audit_B.txt", "w").write("\n".join(B_lines))
print(len(picks), "poles;", len(A_lines), len(B_lines))
