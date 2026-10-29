#!/usr/bin/env python3
"""The night's wiring check as its own file sums it up: the per-layer medians of the stored summary (a change of
exactly 0 counted as a miss, the sign in use at the passages' operating point), the cost line of every layer, and the
count of links whose two signs differ. Reads the pinned copy of wiring_check.npz in atlas_data/offline/wiring_check_31b
(sha256 1c34a34a…, the file the GPU run wrote at 2026-10-29T05:23:53Z) and writes nothing but its report.

    CUDA_VISIBLE_DEVICES= /home/eric/bin/ai/theodolite2/.venv/bin/python -I stored_summary.py > ../outputs/stored_summary.json
"""

from __future__ import annotations

import json

import numpy as np

Z = "/home/eric/bin/ai/theodolite2/atlas_data/offline/wiring_check_31b/wiring_check.npz"
z = np.load(Z, allow_pickle=False)
meta = json.loads(str(z["meta"]))
summ, cost = meta["summary"], meta["cost"]
layers = sorted(summ, key=int)
out = {"layers": len(layers), "built_at": meta["built_at"], "wiring_built_at": meta["wiring_built_at"],
       "passages_built_at": meta["passages_built_at"], "dose": meta["dose"], "writers": meta["writers"]}
for site in ("last", "mean"):
    for grp in ("linked", "null"):
        for k in ("sign_plus", "sign_gate", "use_plus", "use_gate", "same_way", "rho"):
            v = [summ[l][site][grp].get(k) for l in layers]
            v = [x for x in v if x is not None]
            if v:
                out[f"{site}.{k}.{grp}"] = round(float(np.median(v)), 3)
    mv = [summ[l][site].get("moved_over_null") for l in layers]
    out[f"{site}.moved_over_null"] = round(float(np.median([x for x in mv if x is not None])), 2)
    out[f"{site}.moved_over_null_layers"] = sum(x is not None for x in mv)
sec = [cost[l]["seconds"] for l in sorted(cost, key=int)]
out["cost"] = {"layers": len(sec), "seconds_min": min(sec), "seconds_max": max(sec),
               "seconds_median": round(float(np.median(sec)), 1), "seconds_sum": round(sum(sec), 1),
               "minutes_sum": round(sum(sec) / 60, 1)}
sd = meta["sign_disagreement"]
out["sign_disagreement"] = {
    "links": sd["links"],
    "+": {**sd["faces"]["+"], "share": round(sd["faces"]["+"]["differ"] / sd["faces"]["+"]["read"], 4)},
    "-": {**sd["faces"]["-"], "share": round(sd["faces"]["-"]["differ"] / sd["faces"]["-"]["read"], 4)},
    "differ_on_both": sd["differ_on_both"], "differ_on_both_share": round(sd["differ_on_both"] / sd["links"], 4),
}
print(json.dumps(out, indent=1))
