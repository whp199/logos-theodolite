#!/usr/bin/env python3
# N1A_05_report -- print compact tables from N1A_metrics.json (offline).
import json, sys
M = json.load(open("./experiments/results/N1A_metrics.json"))
tags = M["tags"]
methods = M["methods"]
layers = None
for tag in tags:
    for met in ("rho", "p10", "p50", "r10"):
        print(f"\n=== {tag} {met}: real / null / lift [95% CI] (perm p) ===")
        for m in methods:
            s = M["summary"][f"{tag}.{m}.{met}"]
            layers = list(s["per_layer"].keys())
            o = s["overall"]
            row = f"{m:12s} ALL {o['real']['mean']:.3f}/{o['null']['mean']:.3f}/{o['lift']['mean']:+.3f}[{o['lift']['lo']:+.3f},{o['lift']['hi']:+.3f}] |"
            for l in layers:
                pl = s["per_layer"][l]
                row += f" L{l} {pl['lift']['mean']:+.3f}" + ("*" if pl["perm_p"] < 0.01 else " ")
            print(row)
    print(f"\n=== {tag} sign test: mean rho(static_+, c_-)  (frac<0) ===")
    for k, v in M["sign_test"].items():
        if not k.startswith(tag + "."):
            continue
        print(f"{k:34s}", " ".join(f"L{l} {d['mean']['mean']:+.3f}({d['frac_neg']:.2f})" for l, d in v.items()))
for k, v in M["notes"].items():
    print(k, json.dumps(v))
