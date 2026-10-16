CHUNK = 0
# E7_binding_05b: follow-up patching sets (same design as E7_binding_05, added after seeing 05 chunk 0):
#   user   = user-turn message tokens,
#   suffix = the " By the way, I am a" / " ... you are a" tokens incl. the final prediction position,
#   rest   = every position except the system prompt (sanity: sys + rest should approximately account for the full swap).
# Motivation (post hoc, stated honestly): in chunk 0, by L40 neither sys nor passage patches moved the "you are a"
# prediction, so the user-gender information must already sit elsewhere (user turn / suffix / final position).
# Prediction: suffix patch dominates at L40-50 for both behaviours; user-turn patch small (<0.15).
import time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 12, CHUNK * 12 + 12))
LP = [10, 20, 30, 40, 50]
SETS = ["user", "suffix", "rest"]
KS = [0, 2]
t0 = time.time()
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]


def lse4(h59):
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
            ids = {"I": lab.ids(prefix + SUF_I), "Y": lab.ids(prefix + SUF_Y)}
            strs = lab.strs(ids["I"])
            mt = lab.model_turn_pos(ids["I"])
            s0 = [p for p in range(len(strs)) if strs[p] == "system"][0] + 2
            s1 = [p for p in range(s0, len(strs)) if strs[p] == "<turn|>"][0]
            ut = [p for p in range(1, mt) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")][-1]
            u0 = ut + 2
            u1 = [p for p in range(u0, mt) if strs[p] == "<turn|>"][0]
            sets = {}
            for suf in ("I", "Y"):
                n = len(ids[suf])
                sets[suf] = {"user": list(range(u0, u1)), "suffix": list(range(len(pre), n)),
                             "rest": [p for p in range(n) if not (s0 <= p < s1)]}
            ctx.append(dict(pid=pid, k=k, cell=cell, ids=ids, npre=len(pre), sets=sets))
key = {(c["pid"], c["k"], c["cell"]): i for i, c in enumerate(ctx)}
for c in ctx:
    o = ctx[key[(c["pid"], c["k"], "MW" if c["cell"] == "WM" else "WM")]]
    assert all(len(o["ids"][s]) == len(c["ids"][s]) for s in "IY") and o["sets"] == c["sets"]
    c["partner"] = key[(o["pid"], o["k"], o["cell"])]
SRC = {}
clean = {"I": np.zeros(len(ctx)), "Y": np.zeros(len(ctx))}
BS = 12
for suf in ("I", "Y"):
    for s in range(0, len(ctx), BS):
        cs = list(range(s, min(s + BS, len(ctx))))
        H = e7_capture(lab, [ctx[i]["ids"][suf] for i in cs], [list(range(len(ctx[i]["ids"][suf]))) for i in cs])
        off = 0; fin = []
        for i in cs:
            n = len(ctx[i]["ids"][suf])
            SRC[(i, suf)] = H[LP, off:off + n].cpu()
            fin.append(off + n - 1)
            off += n
        clean[suf][cs] = lse4(H[59, fin])
        del H
print("clean done %.1fs" % (time.time() - t0), flush=True)
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
            bids.append(c["ids"][suf])
            P = c["sets"][suf][S]
            src = SRC[(c["partner"], suf)][li]
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
rows = []
for (i, S, l, suf), v in patched.items():
    c = ctx[i]; p = c["partner"]
    ct, cs_ = clean[suf][i], clean[suf][p]
    rows.append(dict(pid=c["pid"], k=c["k"], cell=c["cell"], set=S, layer=l, suf=suf, patched=v, clean_t=float(ct), clean_s=float(cs_),
                     flip=float((v - ct) / (cs_ - ct))))
import json
json.dump({"rows": rows, "clean": {k_: v.tolist() for k_, v in clean.items()}}, open(f"{RES}/E7_binding_patchb_chunk{CHUNK}.json", "w"))
summ = {}
for S in SETS:
    for l in LP:
        for suf in ("I", "Y"):
            summ[f"{S}|L{l}|{suf}"] = float(np.mean([r["flip"] for r in rows if r["set"] == S and r["layer"] == l and r["suf"] == suf]))
for S in SETS:
    print(S, " ".join(f"L{l}: I={summ[f'{S}|L{l}|I']:+.2f} Y={summ[f'{S}|L{l}|Y']:+.2f}" for l in LP))
print("elapsed %.1fs" % (time.time() - t0))
result = {"summary": summ, "chunk": CHUNK}
