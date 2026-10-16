#!/usr/bin/env python3
# R1_IT_LENS 03 helper (local): split the fit passages into daemon jobs of <= MAXPOS positions, one half per job.
import json, sys
R = "./experiments/results/"
MAXPOS = int(sys.argv[1]) if len(sys.argv) > 1 else 72
P = json.load(open(R + "R1_02_positions.json"))
jobs = []
for half in (0, 1):
    idx = [i for i, p in enumerate(P) if p["half"] == half]
    # interleave sources so every job mixes chat and prose
    bysrc = {}
    for i in idx:
        bysrc.setdefault(P[i]["src"], []).append(i)
    order = []
    while any(bysrc.values()):
        for s in sorted(bysrc):
            if bysrc[s]:
                order.append(bysrc[s].pop(0))
    cur, n = [], 0
    for i in order:
        k = len(P[i]["pos"])
        if n + k > MAXPOS and cur:
            jobs.append({"half": half, "passages": cur}); cur, n = [], 0
        cur.append(i); n += k
    if cur:
        jobs.append({"half": half, "passages": cur})
# alternate halves so partial results stay balanced
h0 = [j for j in jobs if j["half"] == 0]; h1 = [j for j in jobs if j["half"] == 1]
alt = [x for pair in zip(h0, h1) for x in pair] + h0[len(h1):] + h1[len(h0):]
json.dump(alt, open(R + "R1_03_jobs.json", "w"))
print(len(alt), "jobs", [sum(len(P[i]["pos"]) for i in j["passages"]) for j in alt])
