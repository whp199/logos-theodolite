# E1_self_readout job 0: build checks (self-token counts), batched-forward equivalence vs lab.capture, timing.
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
import collections
counts = []
bad = []
for qi in range(len(E1_QA)):
    for cname, sysp in E1_CONDS:
        ids, pos, labs, toks = e1_build(qi, cname, sysp)
        c = collections.Counter(labs)
        if cname == "default":
            counts.append(dict(c))
        if c["self"] < 3 or c["user_I"] < 1:
            bad.append((qi, cname, dict(c)))
print("n passages", len(E1_QA), "bad:", bad[:5])
print("self counts:", [c["self"] for c in counts])
print("nonself counts:", [c["nonself"] for c in counts])
# example labels
ids, pos, labs, toks = e1_build(16, "woman_p1", E1_CONDS[14][1])
print(list(zip(toks, labs)))
# equivalence: batched gather vs lab.capture for 2 prompts
t0 = time.time()
bl = [e1_build(0, c, s) for c, s in E1_CONDS[:18]]
store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
torch.cuda.synchronize(); t1 = time.time()
sc = e1_readout(store)
torch.cuda.synchronize(); t2 = time.time()
print(f"forward 18 prompts {t1-t0:.2f}s, readout 3 lenses {t2-t1:.2f}s")
eq = {}
for k in (0, 14):
    ids, pos = bl[k][0], bl[k][1]
    caps = await lab.capture(ids)
    H = caps["resid_post"]
    off = sum(len(b[1]) for b in bl[:k])
    for l in (5, 30, 45, 58, 59):
        a = store[l][off:off + len(pos)].float().cpu()
        b = H[l, pos].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        eq[f"{k}_{l}"] = [float(cos.min()), float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())]
    # readout equivalence via lab.gender_scores on capture
    for l in (30, 50):
        g = lab.gender_scores(lab.lens(H[l, pos], l, "jpp"))["pron"].float().cpu()
        mine = sc["jpp"][off:off + len(pos), l, 0].float()
        eq[f"pron_{k}_{l}"] = float((g - mine).abs().max())
print(json.dumps(eq, indent=0))
result = {"counts": counts, "bad": bad, "eq": eq, "t_forward": t1 - t0, "t_readout": t2 - t1}
