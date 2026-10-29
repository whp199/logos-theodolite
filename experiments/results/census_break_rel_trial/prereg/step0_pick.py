#!/usr/bin/env python3
"""Step 0's rule, fixed before the trial: from the floor file (step0_floor.py), the six questions and the batch.
- half A is who, orange, free; each must keep >= 32 word-pieces in common with itself written alone in a census batch
  of 64 (else stop: the design goes back to the owner);
- half B takes, per genre slot, the first of (bicycle | kettle), (walk | sleep), (chair | bridge) that keeps >= 32 at
  batch 64 (none in a slot: stop);
- the batch is 96 when all six keep >= 32 at batch 96, else 64.
Prints a JSON line: {"questions": [...], "batch": B, "floors": {...}}; exit 1 when it stops."""
import json, sys
d = json.load(open(sys.argv[1]))
MIN = 32
def floor(q, B):
    rs = d['results'][q]
    ks = [r['shared'] for n, r in rs.items() if n.startswith(f'census {B},') and not r['same']]
    return min(ks) if ks else None  # None: never parted within max_new
def holds(q, B):
    k = floor(q, B)
    return k is None or k >= MIN
floors = {q: {B: floor(q, B) for B in (64, 96)} for q in d['results']}
A = ['who', 'orange', 'free']
bad = [q for q in A if not holds(q, 64)]
if bad:
    print(json.dumps({'stop': f'half A question(s) {bad} part from themselves before {MIN} pieces at batch 64', 'floors': floors})); sys.exit(1)
B = []
for slot in (('bicycle', 'kettle'), ('walk', 'sleep'), ('chair', 'bridge')):
    ok = [q for q in slot if holds(q, 64)]
    if not ok:
        print(json.dumps({'stop': f'no question of slot {slot} keeps {MIN} pieces at batch 64', 'floors': floors})); sys.exit(1)
    B.append(ok[0])
qs = A + B
batch = 96 if all(holds(q, 96) for q in qs) else 64
print(json.dumps({'questions': qs, 'batch': batch, 'floors': floors, 'alone_twice_same': d.get('alone_twice_same')}))
sys.exit(0 if d.get('alone_twice_same') else 1)
