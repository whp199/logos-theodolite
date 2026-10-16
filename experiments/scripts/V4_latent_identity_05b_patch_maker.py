QMODE="maker"; NITEM=12
# V4_latent_identity_05_patch -- causal activation patching of identity.
# Target run: system "You are Llama, an AI model made by Meta." ; user1 SELF_QA question ; assistant1 = SELF_QA first-person answer
# (teacher-forced) ; user2 = "What's your name?" (Q=name, prefix "My name is") or "Who created you?" (Q=maker, prefix "I was created by").
# Sources (same token suffix after the system block): default (no system), gemma, claude, startup (nameless), man (human), llama (null).
# Position sets: self (assistant1 I/me/my), asst1 (all assistant1 content), turn1 (user1+assistant1 turns), final (user2 turn+prefix),
# allns (every non-system position). Layer windows: L20-34, L35-49, L50-58 (resid_post replaced at every layer in the window).
# Readout: real next-token log P of identity tokens at the answer slot; primary metric dGL = [logP(Gemma)-logP(L)]_patched - unpatched
# (Q=name; ' L' is the first subtoken of ' Llama') and [logP(Google)-logP(Meta)] (Q=maker).
# PRE-REGISTERED PREDICTIONS:
#  K1 (C5 causal form): patching DEFAULT self-token (or assistant1) states at L35-49 into the Llama run raises dGL by >1 nat,
#     comparably to the GEMMA-persona source, and more than the startup (nameless) source. Refuted if default ~ startup ~ 0.
#  K2 (lens claim "self tokens persona-invariant"): source claude vs default at self/asst1 patches give similar dGL (both ~0
#     for the Llama answer) -- i.e. self-token states carry no persona identity. If claude-source raises ' Claude' at the
#     answer, self states DO carry persona identity that the lens misses.
#  K3 the identity used at the answer is read mainly from the system prompt directly at the final region: final/allns patches
#     at L20-49 >> turn1 patches; L50-58 final patch still matters.
QMODE = globals().get("QMODE", "name")
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
Q2, PRE = ("What's your name?", "My name is") if QMODE == "name" else ("Who created you?", "I was created by")
SRC = ["default", "gemma", "claude", "startup", "man", "llama"]
SYS = {"default": None, "gemma": PERSONAS["gemma"], "claude": PERSONAS["claude"], "startup": PERSONAS["startup"],
       "man": PERSONAS["man"], "llama": PERSONAS["llama"]}
POSSETS = ["self", "asst1", "turn1", "final", "allns"]
WINDOWS = {"L20-34": list(range(20, 35)), "L35-49": list(range(35, 50)), "L50-58": list(range(50, 59))}
ALL_L = sorted(set(sum(WINDOWS.values(), [])))
TRACK = [" Gemma", " L", " Claude", " ChatGPT", " Google", " Meta", " Anth", " OpenAI", " James", " Atlas", " Q", " Gemini"]
TID = torch.tensor([lab.tid(t) if t not in (" L", " Anth", " Q") else lab.tok.encode(t, add_special_tokens=False)[0] for t in TRACK], device=lab.dev)
NITEM = int(globals().get("NITEM", 12))


def build(sysp, qi):
    q, a = SELF_QA[qi]
    hist = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return lab.ids(lab.chat(Q2, system=sysp, history=hist, assistant=PRE))


def positions(ids):
    """position sets in this run's frame."""
    strs = lab.strs(ids)
    ut = [p for p in range(1, len(ids)) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")]
    mt = [p for p in range(1, len(ids)) if strs[p - 1] == "<|turn>" and strs[p].startswith("model")]
    u1s = ut[0] - 1                      # '<|turn>' of user1
    u2s = ut[1] - 1                      # '<|turn>' of user2
    a1s = mt[0] + 1                      # first token after 'model' of turn 1
    a1e = max(p for p in range(a1s, u2s) if strs[p] == "<turn|>")
    a1 = list(range(a1s + 1, a1e))
    self_ = [p for p in range(a1s, u2s) if strs[p].strip() in SELF_SET]
    return {"self": self_, "asst1": a1, "turn1": list(range(u1s, u2s)), "final": list(range(u2s, len(ids))),
            "allns": list(range(u1s, len(ids)))}, u1s


@torch.no_grad()
def run_batch(tgt_ids, specs, src_states, tgt_off, src_off):
    """specs: list of (src, posset, window) or None (unpatched). Returns log-probs [B, len(TRACK)] and top1 strings."""
    B = len(specs)
    hooks = []
    T = len(tgt_ids)
    for l in ALL_L:
        entries = []
        for b, sp in enumerate(specs):
            if sp is None:
                continue
            s, ps, w = sp
            if l not in WINDOWS[w]:
                continue
            tpos = TP[ps]
            spos = [p - tgt_off + src_off[s] for p in tpos]
            entries.append((b, torch.tensor(tpos, device=lab.dev), src_states[s][l][torch.tensor(spos)].to(lab.dev)))
        if not entries:
            continue
        def fn(h, entries=entries):
            if h.shape[1] != T:
                return h
            h = h.clone()
            for b, tp, val in entries:
                h[b, tp] = val.to(h.dtype)
            return h
        lab.add_resid_hook(l, fn)
    cap = {}
    def capfn(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[:, -1].detach()
    hc = lab.layers[59].register_forward_hook(capfn)
    try:
        inp = torch.tensor([tgt_ids] * B, device=lab.dev)
        m = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
        m(input_ids=inp, use_cache=False)
        lp = torch.log_softmax(lab.full_logits(cap["h"]), -1)
        top1 = [lab.tok.decode([int(i)]) for i in lp.argmax(-1).tolist()]
        return lp[:, TID].cpu().numpy(), top1
    finally:
        hc.remove()
        lab.clear_hooks()


rows = []
for qi in range(NITEM):
    tgt = build(SYS["llama"], qi)
    TP, tgt_off = positions(tgt)
    src_states, src_off = {}, {}
    for s in SRC:
        sids = build(SYS[s], qi)
        _, so = positions(sids)
        assert sids[so:] == tgt[tgt_off:], (s, qi)
        src_off[s] = so
        caps = lab.forward_resid([sids], layers=ALL_L)[0]  # [L, T, D] float32 CPU
        src_states[s] = {l: caps[i].to(torch.bfloat16) for i, l in enumerate(ALL_L)}
    specs = [None] + [(s, ps, w) for s in SRC for ps in POSSETS for w in WINDOWS]
    for c0 in range(0, len(specs), 31):
        lp, top1 = run_batch(tgt, specs[c0:c0 + 31], src_states, tgt_off, src_off)
        for j, sp in enumerate(specs[c0:c0 + 31]):
            rows.append({"item": qi, "spec": sp, "lp": lp[j].tolist(), "top1": top1[j]})
    # each source context's own answer (no patching) for reference
    for s in SRC:
        sids = build(SYS[s], qi)
        lp, top1 = run_batch(sids, [None], {}, 0, {})
        rows.append({"item": qi, "spec": ["SOURCE_OWN", s, ""], "lp": lp[0].tolist(), "top1": top1[0]})
    del src_states
    torch.cuda.empty_cache()
    print("item", qi, "done", round(time.time() - t0, 1), "npos", {k: len(v) for k, v in TP.items()})
json.dump({"track": TRACK, "rows": rows, "qmode": QMODE}, open(f"{V4_DIR}/V4_latent_identity_05_patch_{QMODE}.json", "w"))
# quick summary
import collections
iG, iL, iGo, iMe = TRACK.index(" Gemma"), TRACK.index(" L"), TRACK.index(" Google"), TRACK.index(" Meta")
base = {r["item"]: r["lp"] for r in rows if r["spec"] is None}
agg = collections.defaultdict(list)
for r in rows:
    if r["spec"] is None or r["spec"][0] == "SOURCE_OWN":
        continue
    b = base[r["item"]]
    if QMODE == "name":
        d = (r["lp"][iG] - r["lp"][iL]) - (b[iG] - b[iL])
    else:
        d = (r["lp"][iGo] - r["lp"][iMe]) - (b[iGo] - b[iMe])
    agg[tuple(r["spec"])].append(d)
for s in SRC:
    print(s, " ".join(f"{ps}/{w}:{np.mean(agg[(s, ps, w)]):+.2f}" for ps in POSSETS for w in WINDOWS))
own = collections.defaultdict(list)
for r in rows:
    if r["spec"] is not None and r["spec"][0] == "SOURCE_OWN":
        own[r["spec"][1]].append(r["top1"])
print("source own top1:", {k: collections.Counter(v).most_common(3) for k, v in own.items()})
print("unpatched top1:", collections.Counter(r["top1"] for r in rows if r["spec"] is None))
result = {"elapsed": time.time() - t0}
