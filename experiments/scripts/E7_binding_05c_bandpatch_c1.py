CHUNK = 1
# E7_binding_05c: MULTI-LAYER (band) activation patching. Added after 05/05b: single-layer patches underestimate a
# position set's contribution when the information is re-read from elsewhere at later layers. Here resid_post is
# replaced at ALL layers of a band at the chosen positions (band "all" = L0-59 => that position set's entire residual
# stream comes from the source context; band "mid" = L20-39, the window where lens readouts are entity-bound).
# Pairs WM <-> MW as in 05. Read: real next-token log-odds after "... I am a" / "... you are a"; flip fraction.
# PREDICTIONS (written before running): sys/all ~ 1.0 (sanity). passage/all: I-flip < 0.3, Y-flip 0.2-0.5.
#   A_self/all: I-flip < 0.1 (self-gender at "I" tokens mostly not used for "I am a"); A_user/all: Y-flip 0.1-0.3 > I-flip.
import time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 12, CHUNK * 12 + 12))
SETS = ["sys", "user", "passage", "A_self", "A_user"]
BANDS_P = {"all": list(range(60)), "mid": list(range(20, 40))}
KS = [0, 2]
t0 = time.time()
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]


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
            strs = lab.strs(ids["I"])
            pos = e7_positions(lab, ids["I"], len(pre))
            mt = lab.model_turn_pos(ids["I"])
            chan = [p for p in range(mt, len(strs)) if strs[p] == "<channel|>"][0]
            s0 = [p for p in range(len(strs)) if strs[p] == "system"][0] + 2
            s1 = [p for p in range(s0, len(strs)) if strs[p] == "<turn|>"][0]
            ut = [p for p in range(1, mt) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")][-1]
            u0 = ut + 2
            u1 = [p for p in range(u0, mt) if strs[p] == "<turn|>"][0]
            sets = {"sys": list(range(s0, s1)), "user": list(range(u0, u1)), "passage": list(range(chan + 1, len(pre))),
                    "A_self": pos["A_self"], "A_user": pos["A_user"]}
            ctx.append(dict(k=k, cell=cell, ids=ids, npre=len(pre), sets=sets))
    key = {(c["k"], c["cell"]): i for i, c in enumerate(ctx)}
    for c in ctx:
        o = ctx[key[(c["k"], "MW" if c["cell"] == "WM" else "WM")]]
        assert len(o["ids"]["I"]) == len(c["ids"]["I"]) and o["sets"] == c["sets"]
        c["partner"] = key[(o["k"], o["cell"])]
    # source capture: all layers over the prefix (identical for I/Y), + clean final logits for both suffixes
    H = e7_capture(lab, [c["ids"]["I"] for c in ctx], [list(range(c["npre"])) + [len(c["ids"]["I"]) - 1] for c in ctx])
    SRC, fin, off = [], [], 0
    for c in ctx:
        SRC.append(H[:, off:off + c["npre"]])  # [60, npre, D] stays on GPU (4 contexts)
        fin.append(off + c["npre"]); off += c["npre"] + 1
    cleanI = lse4(H[59, fin])
    HY = e7_capture(lab, [c["ids"]["Y"] for c in ctx], [[len(c["ids"]["Y"]) - 1] for c in ctx])
    cleanY = lse4(HY[59, :])
    del HY
    items = [(i, S, bn, suf) for i in range(len(ctx)) for S in SETS for bn in BANDS_P for suf in ("I", "Y")]
    for s in range(0, len(items), 40):
        batch = items[s:s + 40]
        bids = [ctx[i]["ids"][suf] for (i, S, bn, suf) in batch]
        for l in range(60):
            bi, pi, vals = [], [], []
            for b, (i, S, bn, suf) in enumerate(batch):
                if l in BANDS_P[bn]:
                    P = ctx[i]["sets"][S]
                    bi += [b] * len(P); pi += P
                    vals.append(SRC[ctx[i]["partner"]][l, P])
            if not bi:
                continue
            bi_t = torch.tensor(bi, device=lab.dev); pi_t = torch.tensor(pi, device=lab.dev)
            V = torch.cat(vals, 0)

            def fn(h, bi_t=bi_t, pi_t=pi_t, V=V):
                h = h.clone()
                h[bi_t, pi_t] = V.to(h.dtype)
                return h
            lab.add_resid_hook(l, fn)
        try:
            Hp = e7_capture(lab, bids, [[len(x) - 1] for x in bids])
        finally:
            lab.clear_hooks()
        out = lse4(Hp[59, :])
        del Hp
        for b, (i, S, bn, suf) in enumerate(batch):
            c = ctx[i]; p = c["partner"]
            ct = cleanI[i] if suf == "I" else cleanY[i]
            cs_ = cleanI[p] if suf == "I" else cleanY[p]
            rows.append(dict(pid=pid, k=c["k"], cell=c["cell"], set=S, band=bn, suf=suf, patched=float(out[b]),
                             clean_t=float(ct), clean_s=float(cs_), flip=float((out[b] - ct) / (cs_ - ct))))
    del H, SRC
    print(f"pid {pid} done {time.time() - t0:.1f}s", flush=True)
import json
json.dump({"rows": rows}, open(f"{RES}/E7_binding_bandpatch_chunk{CHUNK}.json", "w"))
summ = {}
for S in SETS:
    for bn in BANDS_P:
        for suf in ("I", "Y"):
            summ[f"{S}|{bn}|{suf}"] = float(np.mean([r["flip"] for r in rows if r["set"] == S and r["band"] == bn and r["suf"] == suf]))
for S in SETS:
    print(S, " ".join(f"{bn}: I={summ[f'{S}|{bn}|I']:+.3f} Y={summ[f'{S}|{bn}|Y']:+.3f}" for bn in BANDS_P))
print("elapsed %.1fs" % (time.time() - t0))
result = {"summary": summ, "chunk": CHUNK}
