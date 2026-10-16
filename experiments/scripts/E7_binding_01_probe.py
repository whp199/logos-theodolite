# E7_binding_01: (a) third-person referent probe (diff of means) per layer + held-out validation;
# (b) lens positive control: does J++/JLens/naive gender readout at a neutral-name mention reflect the referent's gender?
# (c) pullback directions of the lens gender readouts (for raw-space geometry analysis).
import os, json, time
exec(open("./experiments/scripts/E7_binding_common.py").read())
os.makedirs(CACHE, exist_ok=True)
t0 = time.time()
items = probe_items()
seqs, refpos, ctxpos = [], [], []
for it in items:
    ids = lab.ids(it["text"])
    strs = lab.strs(ids)
    nm = [p for p in range(len(ids)) if strs[p] == " " + it["name"]]
    kin = [p for p in range(len(ids)) if strs[p].strip() in sum(map(list, KIN), [])][0]
    assert len(nm) >= 3, (it["text"], nm)
    seqs.append(ids)
    refpos.append(nm[1:])
    ctxpos.append([p for p in range(kin + 1, len(ids)) if p not in nm and strs[p].strip().isalpha()])
print("n items", len(items), "mean ref/seq", np.mean([len(r) for r in refpos]), "ctx/seq", np.mean([len(c) for c in ctxpos]))
D = 5376
R = torch.zeros(len(items), 60, D)
C = torch.zeros(len(items), 60, D)
LENS = {k: torch.zeros(len(items), 60, 3) for k in ("jpp", "jlens", "naive")}  # lens scores at ref tokens
sid = e7_score_ids(lab)
BS = 32
for s in range(0, len(items), BS):
    bids = seqs[s:s + BS]
    bpos = [refpos[i] + ctxpos[i] for i in range(s, s + len(bids))]
    H = e7_capture(lab, bids, bpos)  # [60, N, D]
    off = 0
    segs = []
    for j, i in enumerate(range(s, s + len(bids))):
        nr, nc = len(refpos[i]), len(ctxpos[i])
        segs.append((i, off, off + nr, off + nr + nc))
        off += nr + nc
    for (i, a0, a1, a2) in segs:
        R[i] = H[:, a0:a1].float().mean(1).cpu()
        C[i] = H[:, a1:a2].float().mean(1).cpu()
    for l in range(60):
        for kind in ("jpp", "jlens", "naive"):
            sc = e7_scores(lab, lab.lens(H[l], l, kind), sid)  # [N,3]
            for (i, a0, a1, a2) in segs:
                LENS[kind][i, l] = sc[a0:a1].mean(0).cpu()
    del H
print("captured %.1fs" % (time.time() - t0))
g = np.array([1 if it["g"] == "F" else 0 for it in items])
t = np.array([it["t"] for it in items])


def dom(X, mask):
    return X[mask].mean(0)


out = {"layers": list(range(60))}
for name, X in (("ref", R), ("ctx", C)):
    X = X.numpy()
    muf, mum = X[g == 1].mean(0), X[g == 0].mean(0)
    np.save(f"{CACHE}/probe_{name}_muf.npy", muf.astype(np.float32))
    np.save(f"{CACHE}/probe_{name}_mum.npy", mum.astype(np.float32))
    STORE[f"{EXP}_probe_{name}"] = (torch.tensor(muf), torch.tensor(mum))
    # held-out: train on even templates, test on odd (and vice versa)
    acc, dprime, sep_scaled = [], [], []
    for l in range(60):
        accs, dps, scs = [], [], []
        for tr_par in (0, 1):
            tr = (t % 2) == tr_par
            te = ~tr
            w = X[tr & (g == 1), l].mean(0) - X[tr & (g == 0), l].mean(0)
            mid = (X[tr & (g == 1), l].mean(0) + X[tr & (g == 0), l].mean(0)) / 2
            sc = (X[te, l] - mid) @ w / (w @ w)
            accs.append(float(((sc > 0) == (g[te] == 1)).mean()))
            f, m = sc[g[te] == 1], sc[g[te] == 0]
            dps.append(float((f.mean() - m.mean()) / np.sqrt(0.5 * (f.var(ddof=1) + m.var(ddof=1)) + 1e-12)))
            scs.append(float(f.mean() - m.mean()))  # 1.0 = same separation as training
        acc.append(np.mean(accs)); dprime.append(np.mean(dps)); sep_scaled.append(np.mean(scs))
    out[f"probe_{name}_heldout_acc"] = acc
    out[f"probe_{name}_heldout_dprime"] = dprime
    out[f"probe_{name}_heldout_sep"] = sep_scaled
# lens positive control at neutral-name referent tokens (F minus M, per-template paired)
for kind in LENS:
    Lk = LENS[kind].numpy()  # [n, 60, 3]
    diff = Lk[g == 1] - Lk[g == 0]  # items are ordered (F, M) pairs
    out[f"lens_{kind}_ref_FminusM_mean"] = diff.mean(0).tolist()  # [60,3]
    out[f"lens_{kind}_ref_FminusM_se"] = (diff.std(0, ddof=1) / np.sqrt(len(diff))).tolist()
    out[f"lens_{kind}_ref_F_mean"] = Lk[g == 1].mean(0).tolist()
    out[f"lens_{kind}_ref_M_mean"] = Lk[g == 0].mean(0).tolist()
# pullback directions of lens gender readouts (raw residual space at layer l)
for which in ("pron", "noun"):
    d = lab.gender_direction(which)
    for kind in ("jpp", "jlens", "naive"):
        P = torch.stack([lab.pullback(d, l, kind).cpu() for l in range(60)], 0).numpy().astype(np.float32)
        np.save(f"{CACHE}/pullback_{kind}_{which}.npy", P)
print("total %.1fs" % (time.time() - t0))
bands = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
for name in ("ref", "ctx"):
    print(name, "heldout acc by band", {b: round(float(np.mean([out[f"probe_{name}_heldout_acc"][l] for l in r])), 3) for b, r in bands.items()},
          "dprime", {b: round(float(np.mean([out[f"probe_{name}_heldout_dprime"][l] for l in r])), 2) for b, r in bands.items()})
for kind in LENS:
    m = np.array(out[f"lens_{kind}_ref_FminusM_mean"])
    print(kind, "lens F-M at name tokens (pron,noun,noun4) by band", {b: [round(float(x), 2) for x in m[list(r)].mean(0)] for b, r in bands.items()})
result = out
