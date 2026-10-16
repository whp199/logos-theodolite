CHUNK = 0
# E7_binding_05: CAUSAL binding / information-flow test by activation patching (no lenses involved).
# Target context WM (self=woman, user=man) <- source MW (self=man, user=woman), and the reverse; sequences are
# token-aligned (all names and gender nouns are single tokens). At ONE layer l, replace resid_post[l] at a position
# set S with the source run's values, then read the real next-token log-odds LSE(woman,girl,lady,female)-LSE(man,boy,guy,male)
# after "... By the way, I am a" (self) and "... By the way, you are a" (user).
# flip = (patched - clean_target) / (clean_source - clean_target): 0 = no effect, 1 = full swap.
# Sets: A_self (assistant I/me/my in passage), A_user (assistant you/your), passage (all assistant passage tokens),
#       sys (all system-prompt content tokens).  Layers: 10, 20, 30, 40, 50.  Variants k in {0 (self-first), 2 (user-first)}.
# PRE-REGISTERED PREDICTIONS:
#  C1 sys patch at L10 flips both behaviours ~fully (>0.8); the flip decays with layer as gender info is copied
#     out of the system prompt into later positions.
#  C2 pronoun patches: if mid-layer bound gender at pronoun tokens is causally used, A_self patch flips "I am a" more than
#     "you are a", and A_user the reverse (causal binding). Expected magnitude small (<0.2) because the final position
#     can attend to the system prompt directly; this is uncertain.
#  C3 passage patch effects grow with layer (info accumulates in the conversation stream), moderate (0.1-0.5) by L30-50.
import time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 12, CHUNK * 12 + 12))
LP = [10, 20, 30, 40, 50]
SETS = ["A_self", "A_user", "passage", "sys"]
KS = [0, 2]
t0 = time.time()
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]


def lse4(h59):  # h59 [B, D] resid_post[59] -> real log-odds
    L = lab.tok_logits(h59, f4 + m4)
    return (torch.logsumexp(L[:, :4], -1) - torch.logsumexp(L[:, 4:], -1)).float().cpu().numpy()


ctx = []
for pid in PIDS:
    q, a = PASSAGES[pid]
    for k in KS:
        for cell in ("WM", "MW"):
            sysp = system_prompt(cell[0], cell[1], k)
            prefix = lab.chat(q, system=sysp, assistant=a)
            pre = lab.ids(prefix)
            idsI = lab.ids(prefix + SUF_I); idsY = lab.ids(prefix + SUF_Y)
            assert idsI[:len(pre)] == pre and idsY[:len(pre)] == pre
            strs = lab.strs(idsI)
            pos = e7_positions(lab, idsI, len(pre))
            mt = lab.model_turn_pos(idsI)
            chan = [p for p in range(mt, len(idsI)) if strs[p] == "<channel|>"][0]
            s0 = [p for p in range(len(strs)) if strs[p] == "system"][0] + 2
            s1 = [p for p in range(s0, len(strs)) if strs[p] == "<turn|>"][0]
            sets = {"A_self": pos["A_self"], "A_user": pos["A_user"], "passage": list(range(chan + 1, len(pre))), "sys": list(range(s0, s1))}
            ctx.append(dict(pid=pid, k=k, cell=cell, idsI=idsI, idsY=idsY, npre=len(pre), sets=sets))
# partner index + alignment checks
key = {(c["pid"], c["k"], c["cell"]): i for i, c in enumerate(ctx)}
for c in ctx:
    o = ctx[key[(c["pid"], c["k"], "MW" if c["cell"] == "WM" else "WM")]]
    assert len(o["idsI"]) == len(c["idsI"]) and o["sets"] == c["sets"], "misaligned pair"
    c["partner"] = key[(o["pid"], o["k"], o["cell"])]
# 1) clean runs + source capture at LP layers over the prefix
SRC = {}
cleanI = np.zeros(len(ctx)); cleanY = np.zeros(len(ctx))
BS = 16
for s in range(0, len(ctx), BS):
    cs = list(range(s, min(s + BS, len(ctx))))
    H = e7_capture(lab, [ctx[i]["idsI"] for i in cs], [list(range(ctx[i]["npre"])) + [len(ctx[i]["idsI"]) - 1] for i in cs])
    off = 0
    fin = []
    for i in cs:
        n = ctx[i]["npre"]
        SRC[i] = H[LP, off:off + n].cpu()
        fin.append(off + n)
        off += n + 1
    cleanI[cs] = lse4(H[59, fin])
    del H
    H = e7_capture(lab, [ctx[i]["idsY"] for i in cs], [[len(ctx[i]["idsY"]) - 1] for i in cs])
    cleanY[cs] = lse4(H[59, :])
    del H
print("clean done %.1fs" % (time.time() - t0), flush=True)
# 2) patched runs grouped by layer
items = [(i, S, l, suf) for l in LP for i in range(len(ctx)) for S in SETS for suf in ("I", "Y")]
patched = {}
BS = 48
for l in LP:
    li = LP.index(l)
    its = [x for x in items if x[2] == l]
    for s in range(0, len(its), BS):
        batch = its[s:s + BS]
        bids, bi, pi, vals = [], [], [], []
        for b, (i, S, _, suf) in enumerate(batch):
            c = ctx[i]
            bids.append(c["idsI"] if suf == "I" else c["idsY"])
            P = c["sets"][S]
            src = SRC[c["partner"]][li]  # [npre, D]
            bi += [b] * len(P); pi += P
            vals.append(src[P])
        bi_t = torch.tensor(bi, device=lab.dev); pi_t = torch.tensor(pi, device=lab.dev)
        V = torch.cat(vals, 0).to(lab.dev)

        def fn(h, bi_t=bi_t, pi_t=pi_t, V=V):
            h = h.clone()
            h[bi_t, pi_t] = V.to(h.dtype)
            return h
        lab.add_resid_hook(l, fn)
        try:
            H = e7_capture(lab, bids, [[len(x) - 1] for x in bids])
        finally:
            lab.clear_hooks()
        out = lse4(H[59, :])
        for b, it in enumerate(batch):
            patched[it] = float(out[b])
        del H
    print(f"layer {l} done {time.time() - t0:.1f}s", flush=True)
# 3) flips
rows = []
for (i, S, l, suf), v in patched.items():
    c = ctx[i]; p = c["partner"]
    ct = cleanI[i] if suf == "I" else cleanY[i]
    cs_ = cleanI[p] if suf == "I" else cleanY[p]
    rows.append(dict(pid=c["pid"], k=c["k"], cell=c["cell"], set=S, layer=l, suf=suf, patched=v, clean_t=float(ct), clean_s=float(cs_),
                     flip=float((v - ct) / (cs_ - ct))))
import json
json.dump({"rows": rows, "cleanI": cleanI.tolist(), "cleanY": cleanY.tolist(),
           "ctx": [{k_: c[k_] for k_ in ("pid", "k", "cell", "npre")} | {"nsets": {S: len(v) for S, v in c["sets"].items()}} for c in ctx]},
          open(f"{RES}/E7_binding_patch_chunk{CHUNK}.json", "w"))
summ = {}
for S in SETS:
    for l in LP:
        for suf in ("I", "Y"):
            f = [r["flip"] for r in rows if r["set"] == S and r["layer"] == l and r["suf"] == suf]
            summ[f"{S}|L{l}|{suf}"] = float(np.mean(f))
for S in SETS:
    print(S, " ".join(f"L{l}: I={summ[f'{S}|L{l}|I']:+.2f} Y={summ[f'{S}|L{l}|Y']:+.2f}" for l in LP))
print("elapsed %.1fs" % (time.time() - t0))
result = {"summary": summ, "chunk": CHUNK, "clean_I_mean_WM": float(np.mean([cleanI[i] for i, c in enumerate(ctx) if c["cell"] == "WM"])),
          "clean_I_mean_MW": float(np.mean([cleanI[i] for i, c in enumerate(ctx) if c["cell"] == "MW"])),
          "clean_Y_mean_WM": float(np.mean([cleanY[i] for i, c in enumerate(ctx) if c["cell"] == "WM"])),
          "clean_Y_mean_MW": float(np.mean([cleanY[i] for i, c in enumerate(ctx) if c["cell"] == "MW"]))}
