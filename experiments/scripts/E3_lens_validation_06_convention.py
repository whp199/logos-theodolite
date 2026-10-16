# E3_lens_validation (a2) Lens-convention + affine-offset diagnostics.
# Q1: is h @ J[l].T (tool convention) better than h @ J[l] (transposed) for J++ and JLens?
# Q2: does adding a per-layer offset b_l = mean(h_59) - mean(h_l @ J^T) (cross-fitted on the other half of passages)
#     rescue fidelity (i.e. is the lens missing an affine term)?
# PREDICTION: the tool convention is correct (if transposed was better, the whole instrument would be broken); the
#   offset will reduce KL substantially at mid layers because Jacobian lenses have no bias term.
import json, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
items = []
for i in range(0, 24, 3):
    ids = lab.ids(PROSE[i][1]); items.append(("prose", i, ids, list(range(4, len(ids)))))
for i in list(range(0, 12, 2)) + list(range(12, 24, 3)):
    ids = lab.ids(lab.chat(chats[i]["q"], assistant=chats[i]["a"]))
    items.append(("chat", i, ids, e3_chat_positions(lab, ids, SELF_TOKS)["asst"]))
LAY = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58]
# pass 1: states + mapped means per half
Hs = []
for (src, i, ids, pos) in items:
    H = lab.forward_resid([ids])[0][:, pos]  # [60,P,D]
    Hs.append(H)
half = [k % 2 for k in range(len(items))]
def mapped(H, l, kind):
    if kind == "jpp": return lab.lens(H[l], l, "jpp").float()
    if kind == "jppT": return (H[l].to(lab.dev, torch.bfloat16) @ lab.J[l]).float()
    if kind == "jlens": return lab.lens(H[l], l, "jlens").float()
    if kind == "jlensT": return (H[l].to(lab.dev, torch.bfloat16) @ lab.JL[l].to(lab.dev)).float()
    if kind == "naive": return H[l].to(lab.dev).float()
KINDS2 = ["jpp", "jppT", "jlens", "jlensT", "naive"]
# offsets b[kind][l][h] from half h
off = {k: {} for k in KINDS2}
for kind in KINDS2:
    for l in LAY:
        sums = [torch.zeros(5376, device=lab.dev), torch.zeros(5376, device=lab.dev)]; n = [0, 0]
        for k, H in enumerate(Hs):
            m = mapped(H, l, kind); f = H[59].to(lab.dev).float()
            sums[half[k]] += (f - m).sum(0); n[half[k]] += m.shape[0]
        off[kind][l] = [sums[0] / n[0], sums[1] / n[1]]
res = {}
for kind in KINDS2:
    for corr in (False, True):
        name = kind + ("+b" if corr else "")
        rec = {"prose": {"kl": [], "top1": []}, "chat": {"kl": [], "top1": []}}
        for l in LAY:
            agg = {"prose": [[], []], "chat": [[], []]}
            for k, ((src, i, ids, pos), H) in enumerate(zip(items, Hs)):
                lf = lab.full_logits(H[59].to(lab.dev)); logpf = F.log_softmax(lf, -1); pf = logpf.exp()
                m = mapped(H, l, kind)
                if corr: m = m + off[kind][l][1 - half[k]]  # cross-fitted offset
                lg = lab.full_logits(m.to(torch.bfloat16)); logp = F.log_softmax(lg, -1)
                agg[src][0].append(float((pf * (logpf - logp)).sum(-1).mean()))
                agg[src][1].append(float((lg.argmax(-1) == lf.argmax(-1)).float().mean()))
            for s in ("prose", "chat"):
                rec[s]["kl"].append(float(np.mean(agg[s][0]))); rec[s]["top1"].append(float(np.mean(agg[s][1])))
        res[name] = rec
        print(f"{name:9s} prose KL " + " ".join(f"{x:5.1f}" for x in rec["prose"]["kl"]) + " | top1 " + " ".join(f"{x:.2f}" for x in rec["prose"]["top1"]), flush=True)
        print(f"{name:9s} chat  KL " + " ".join(f"{x:5.1f}" for x in rec["chat"]["kl"]) + " | top1 " + " ".join(f"{x:.2f}" for x in rec["chat"]["top1"]), flush=True)
# top-5 decode examples (one prose position, one chat self position)
ex = {}
for (src, i, ids, pos), H in [(items[0], Hs[0]), (items[8], Hs[8])]:
    j = len(pos) // 2
    p = pos[j]
    ctx = lab.tok.decode(ids[max(0, p - 12):p + 1])
    ex_k = {"ctx": ctx, "final": lab.topk(H[59, j].to(lab.dev), 5)}
    for kind in ("jpp", "jppT", "jlens", "naive"):
        for l in (20, 30, 40, 50, 58):
            ex_k[f"{kind}_L{l}"] = lab.topk(mapped(H[:, j:j + 1], l, kind).to(torch.bfloat16), 5)
            ex_k[f"{kind}+b_L{l}"] = lab.topk((mapped(H[:, j:j + 1], l, kind) + off[kind][l][0]).to(torch.bfloat16), 5) if l in off[kind] else None
    ex[f"{src}{i}"] = ex_k
    print(json.dumps(ex_k)[:3000])
res["layers"] = LAY; res["examples"] = ex
json.dump(res, open(E3_RES + "E3_lens_validation_a2_convention.json", "w"), indent=1)
print("time", time.time() - t0)
result = {"ok": 1}
