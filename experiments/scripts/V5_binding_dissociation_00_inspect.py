# V5_binding_dissociation_00: tokenization / position-typing sanity check (no model forward except one behaviour check).
exec(open("./experiments/scripts/V5_binding_dissociation_common.py").read())
out = {}
for nm in A_NAMES + U_NAMES + T_NAMES + ["woman", "man"]:
    out[nm] = lab.tok.encode(" " + nm, add_special_tokens=False)
print("name toks", out)
counts = {t: [] for t in TYPES}
lens_ = []
aligned = True
ex = None
for it in range(len(ITEMS)):
    for k in range(4):
        seqs = []
        for cell in CELLS:
            c = dict(item=it, cell="".join(cell), k=k, system=system_prompt(cell, k))
            ids, pos = build_seq(lab, c)
            seqs.append((ids, pos))
        L0 = [len(s[0]) for s in seqs]
        if len(set(L0)) != 1 or any(s[1] != seqs[0][1] for s in seqs):
            aligned = False; print("NOT ALIGNED", it, k, L0)
        ids, pos = seqs[0]
        lens_.append(len(ids))
        for t in TYPES:
            counts[t].append(len(pos[t]))
        if it == 0 and k == 0:
            strs = lab.strs(ids)
            ex = {t: [(p, strs[p]) for p in pos[t]] for t in TYPES}
            print(repr("".join(strs[:]))[:3000])
for t in TYPES:
    import numpy as _np
    a = _np.array(counts[t]); print(f"{t:12s} mean {a.mean():.2f} min {a.min()} items_with>0 {(a.reshape(24,4)[:,0]>0).sum()}")
print("aligned", aligned, "len range", min(lens_), max(lens_))
for t in TYPES:
    print(t, ex[t])
# quote tokenization example
s = 'When Avery said, "I am tired," and you told me, "I never listen."'
print([lab.tok.decode([i]) for i in lab.tok.encode(s, add_special_tokens=False)])
result = {"counts": {t: counts[t] for t in TYPES}, "aligned": aligned, "example": ex, "name_toks": out}
