CHUNK = 0
# E7_binding_05e: DIRECTIONAL interchange patching at pronoun tokens, all layers (0-59).
# Added after 05c showed that swapping the whole residual stream of the passage "I/me/my" tokens moves the USER-gender
# behaviour ("you are a", flip ~0.09) more than the SELF-gender behaviour ("I am a", ~0.02).
# Hypothesis: the I-token stream carries the user's gender in an off-referent subspace (orthogonal to the lens-visible
# referent-gender direction, ~1/3 norm, see E7_binding_04), and THAT component is what "you are a" reads.
# Interchange only the projection on a subspace U (per layer): h <- h + ((h_src - h) U^T) U at the set's positions.
# Subspaces (cross-fitted from the other half of the passages, E7_binding_08):
#   ref  = referent-gender direction (A_self: G_self diff-of-means; A_user: G_user diff-of-means)
#   non  = other-entity gender direction, orthogonalised against ref
#   both = span(ref, non);  rand = random unit direction (control);  full = whole-stream swap (reference, as in 05c)
# PREDICTIONS (before running): A_self/non: Y-flip >= 0.05 and I-flip ~ 0; A_self/ref: I-flip <= 0.02, Y ~ 0;
#   A_user/ref: Y-flip >= 0.1; rand ~ 0 everywhere; both captures >= 70% of full.
import time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 12, CHUNK * 12 + 12))
D_ = np.load(f"{CACHE}/dirs_xfit{CHUNK}.npz")   # estimated from the OTHER half of passages
SETS = ["A_self", "A_user"]
SUBS = ["ref", "non", "both", "rand", "full"]
KS = [0, 2]
t0 = time.time()
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]
U = {}
for S in SETS:
    for sub in ("ref", "non", "rand"):
        U[(S, sub)] = torch.tensor(D_[f"{S}_{sub}"], device=lab.dev).unsqueeze(1)  # [60, 1, D]
    U[(S, "both")] = torch.cat([U[(S, "ref")], U[(S, "non")]], 1)            # [60, 2, D]


def lse4(h59):
    L = lab.tok_logits(h59, f4 + m4)
    return (torch.logsumexp(L[:, :4], -1) - torch.logsumexp(L[:, 4:], -1)).float().cpu().numpy()


rows = []
for pid in PIDS:
    q, a = PASSAGES[pid]
    ctx = []
    for k in KS:
        for cell in ("WM", "MW"):
            sysp = system_prompt(cell[0], cell[1], k)
            prefix = lab.chat(q, system=sysp, assistant=a)
            pre = lab.ids(prefix)
            ids = {"I": lab.ids(prefix + SUF_I), "Y": lab.ids(prefix + SUF_Y)}
            pos = e7_positions(lab, ids["I"], len(pre))
            ctx.append(dict(k=k, cell=cell, ids=ids, npre=len(pre), sets={"A_self": pos["A_self"], "A_user": pos["A_user"]}))
    key = {(c["k"], c["cell"]): i for i, c in enumerate(ctx)}
    for c in ctx:
        o = ctx[key[(c["k"], "MW" if c["cell"] == "WM" else "WM")]]
        assert len(o["ids"]["I"]) == len(c["ids"]["I"]) and o["sets"] == c["sets"]
        c["partner"] = key[(o["k"], o["cell"])]
    H = e7_capture(lab, [c["ids"]["I"] for c in ctx], [list(range(c["npre"])) + [len(c["ids"]["I"]) - 1] for c in ctx])
    SRC, fin, off = [], [], 0
    for c in ctx:
        SRC.append(H[:, off:off + c["npre"]])
        fin.append(off + c["npre"]); off += c["npre"] + 1
    cleanI = lse4(H[59, fin])
    HY = e7_capture(lab, [c["ids"]["Y"] for c in ctx], [[len(c["ids"]["Y"]) - 1] for c in ctx])
    cleanY = lse4(HY[59, :])
    del HY
    items = [(i, S, sub, suf) for i in range(len(ctx)) for S in SETS for sub in SUBS for suf in ("I", "Y")]
    for s in range(0, len(items), 40):
        batch = items[s:s + 40]
        bids = [ctx[i]["ids"][suf] for (i, S, sub, suf) in batch]
        for l in range(60):
            groups = []
            for S in SETS:
                for sub in SUBS:
                    bi, pi, vals = [], [], []
                    for b, (i, S2, sub2, suf) in enumerate(batch):
                        if S2 == S and sub2 == sub:
                            P = ctx[i]["sets"][S]
                            bi += [b] * len(P); pi += P
                            vals.append(SRC[ctx[i]["partner"]][l, P])
                    if bi:
                        groups.append((torch.tensor(bi, device=lab.dev), torch.tensor(pi, device=lab.dev),
                                       torch.cat(vals, 0).float(), None if sub == "full" else U[(S, sub)][l]))

            def fn(h, groups=groups):
                h = h.clone()
                for (bi_t, pi_t, V, Ub) in groups:
                    if Ub is None:
                        h[bi_t, pi_t] = V.to(h.dtype)
                    else:
                        cur = h[bi_t, pi_t].float()
                        h[bi_t, pi_t] = (cur + ((V - cur) @ Ub.T) @ Ub).to(h.dtype)
                return h
            lab.add_resid_hook(l, fn)
        try:
            Hp = e7_capture(lab, bids, [[len(x) - 1] for x in bids])
        finally:
            lab.clear_hooks()
        out = lse4(Hp[59, :])
        del Hp
        for b, (i, S, sub, suf) in enumerate(batch):
            c = ctx[i]; p = c["partner"]
            ct = cleanI[i] if suf == "I" else cleanY[i]
            cs_ = cleanI[p] if suf == "I" else cleanY[p]
            rows.append(dict(pid=pid, k=c["k"], cell=c["cell"], set=S, sub=sub, suf=suf, patched=float(out[b]),
                             clean_t=float(ct), clean_s=float(cs_), flip=float((out[b] - ct) / (cs_ - ct))))
    del H, SRC
    print(f"pid {pid} done {time.time() - t0:.1f}s", flush=True)
import json
json.dump({"rows": rows}, open(f"{RES}/E7_binding_dirpatch_chunk{CHUNK}.json", "w"))
summ = {}
for S in SETS:
    for sub in SUBS:
        for suf in ("I", "Y"):
            summ[f"{S}|{sub}|{suf}"] = float(np.mean([r["flip"] for r in rows if r["set"] == S and r["sub"] == sub and r["suf"] == suf]))
for S in SETS:
    print(S, " ".join(f"{sub}: I={summ[f'{S}|{sub}|I']:+.3f} Y={summ[f'{S}|{sub}|Y']:+.3f}" for sub in SUBS))
print("elapsed %.1fs" % (time.time() - t0))
result = {"summary": summ, "chunk": CHUNK}
