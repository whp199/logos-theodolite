CHUNK = 1
# E7_binding_05f: does the bound referent-gender at pronoun tokens act as an "explaining-away" signal that REDUCES
# leakage? (Added after 05e: interchanging only the referent-gender coordinate at "I" tokens moved the USER prediction
# "you are a" in the OPPOSITE direction of the new self gender (+0.82 / -1.09 nats) and left "I am a" unchanged.)
# Test: NEUTRALISE the referent coordinate at the pronoun tokens by interchanging it (all layers) with a context in which
# that entity is unspecified ("a person"), then measure the cross-entity leak.
#   A_self: targets WU, MU, WM, MW <- sources UU, UU, UM, UW (self made neutral, user unchanged)
#           leak_Y = (Y[W.] - Y[M.]) / 2  (self gender leaking into "you are a"); self_I = (I[W.] - I[M.]) / 2
#   A_user: targets UW, UM, WM, MW <- sources UU, UU, WU, MU (user made neutral)
#           leak_I = (I[.W] - I[.M]) / 2; user_Y = (Y[.W] - Y[.M]) / 2
# Interventions: ref (referent direction, cross-fitted), rand (random direction control), full (whole-stream interchange).
# PREDICTION (explaining-away hypothesis): neutralising ref at A_self INCREASES leak_Y (by >= 0.2 nats) and leaves self_I
# unchanged; neutralising ref at A_user INCREASES leak_I. rand: no change.
import time
exec(open("./experiments/scripts/E7_binding_common.py").read())
CHUNK = globals().get("CHUNK", 0)
PIDS = list(range(CHUNK * 12, CHUNK * 12 + 12))
D_ = np.load(f"{CACHE}/dirs_xfit{CHUNK}.npz")
KS = [0, 2]
CELLS_NEED = ["WU", "MU", "WM", "MW", "UW", "UM", "UU"]
PLAN = {"A_self": {"WU": "UU", "MU": "UU", "WM": "UM", "MW": "UW"},
        "A_user": {"UW": "UU", "UM": "UU", "WM": "WU", "MW": "MU"}}
INTS = ["none", "ref", "rand", "full"]
t0 = time.time()
f4 = [lab.tid(x) for x in F4]; m4 = [lab.tid(x) for x in M4]
U = {(S, sub): torch.tensor(D_[f"{S}_{sub}"], device=lab.dev).unsqueeze(1) for S in PLAN for sub in ("ref", "rand")}


def lse4(h59):
    L = lab.tok_logits(h59, f4 + m4)
    return (torch.logsumexp(L[:, :4], -1) - torch.logsumexp(L[:, 4:], -1)).float().cpu().numpy()


rows = []
for pid in PIDS:
    q, a = PASSAGES[pid]
    ctx = {}
    for k in KS:
        for cell in CELLS_NEED:
            sysp = system_prompt(cell[0], cell[1], k)
            prefix = lab.chat(q, system=sysp, assistant=a)
            pre = lab.ids(prefix)
            ids = {"I": lab.ids(prefix + SUF_I), "Y": lab.ids(prefix + SUF_Y)}
            pos = e7_positions(lab, ids["I"], len(pre))
            ctx[(k, cell)] = dict(ids=ids, npre=len(pre), sets={"A_self": pos["A_self"], "A_user": pos["A_user"]})
    keys = list(ctx)
    H = e7_capture(lab, [ctx[x]["ids"]["I"] for x in keys], [list(range(ctx[x]["npre"])) for x in keys])
    SRC, off = {}, 0
    for x in keys:
        SRC[x] = H[:, off:off + ctx[x]["npre"]]; off += ctx[x]["npre"]
    items = [(None, k, cell, None, "none", suf) for k in KS for cell in CELLS_NEED for suf in ("I", "Y")]
    for S, mp in PLAN.items():
        for k in KS:
            for tgt, src in mp.items():
                assert len(ctx[(k, tgt)]["ids"]["I"]) == len(ctx[(k, src)]["ids"]["I"]) and ctx[(k, tgt)]["sets"] == ctx[(k, src)]["sets"]
                for it in INTS[1:]:
                    for suf in ("I", "Y"):
                        items.append((S, k, tgt, src, it, suf))
    for s in range(0, len(items), 40):
        batch = items[s:s + 40]
        bids = [ctx[(k, tgt)]["ids"][suf] for (S, k, tgt, src, it, suf) in batch]
        for l in range(60):
            groups = []
            for S in PLAN:
                for it in ("ref", "rand", "full"):
                    bi, pi, vals = [], [], []
                    for b, (S2, k, tgt, src, it2, suf) in enumerate(batch):
                        if S2 == S and it2 == it:
                            P = ctx[(k, tgt)]["sets"][S]
                            bi += [b] * len(P); pi += P
                            vals.append(SRC[(k, src)][l, P])
                    if bi:
                        groups.append((torch.tensor(bi, device=lab.dev), torch.tensor(pi, device=lab.dev),
                                       torch.cat(vals, 0).float(), None if it == "full" else U[(S, it)][l]))
            if groups:
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
        for b, (S, k, tgt, src, it, suf) in enumerate(batch):
            rows.append(dict(pid=pid, set=S, k=k, tgt=tgt, src=src, int=it, suf=suf, lse4=float(out[b])))
    del H, SRC
    print(f"pid {pid} done {time.time() - t0:.1f}s", flush=True)
import json
json.dump({"rows": rows}, open(f"{RES}/E7_binding_explainaway_chunk{CHUNK}.json", "w"))


def val(S, tgt, it, suf, pid=None):
    rr = [r for r in rows if r["tgt"] == tgt and r["suf"] == suf and (pid is None or r["pid"] == pid) and
          ((r["int"] == "none") if it == "none" else (r["int"] == it and r["set"] == S))]
    return float(np.mean([r["lse4"] for r in rr]))


summ = {}
for it in INTS:
    summ[f"A_self|{it}|leak_Y_userU"] = (val("A_self", "WU", it, "Y") - val("A_self", "MU", it, "Y")) / 2
    summ[f"A_self|{it}|self_I_userU"] = (val("A_self", "WU", it, "I") - val("A_self", "MU", it, "I")) / 2
    summ[f"A_self|{it}|leak_Y_opp"] = (val("A_self", "WM", it, "Y") - val("A_self", "MW", it, "Y"))  # = (selfW-selfM leak) - user effect
    summ[f"A_user|{it}|leak_I_selfU"] = (val("A_user", "UW", it, "I") - val("A_user", "UM", it, "I")) / 2
    summ[f"A_user|{it}|user_Y_selfU"] = (val("A_user", "UW", it, "Y") - val("A_user", "UM", it, "Y")) / 2
for k_, v in summ.items():
    print(f"{k_:32s} {v:+.3f}")
print("elapsed %.1fs" % (time.time() - t0))
result = {"summary": summ, "chunk": CHUNK}
