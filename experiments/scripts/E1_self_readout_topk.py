# E1_self_readout: qualitative top-10 lens tokens at assistant self positions (and non-self background) for
# default / nameless / gemma_explicit / woman (p1+p2) / man (p1+p2) at layers 30, 40, 50 (0-based), lenses jpp and jlens.
# Aggregation: per passage, mean softmax prob over self positions; then mean over the 40 passages (and over phrasings p1/p2).
# Also mean logits for contrast rankings (woman-man, default-nameless, self-nonself).
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
import torch.nn.functional as F
TK_CONDS = ["default", "nameless", "gemma_explicit", "woman_p1", "woman_p2", "man_p1", "man_p2"]
TK_LAYERS = [30, 40, 50]
TK_LENSES = ["jpp", "jlens"]
cd = dict(E1_CONDS)
acc = {}  # (cond, cls, lens, layer) -> [sum_prob [V], sum_logit [V], n]
t0 = time.time()
JLg = {l: lab.JL[l].to(lab.dev) for l in TK_LAYERS}
for qi in range(len(E1_QA)):
    bl = [e1_build(qi, c, cd[c]) for c in TK_CONDS]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    off = 0
    for c, (ids, pos, labs, toks) in zip(TK_CONDS, bl):
        for cls in ("self", "nonself"):
            if cls == "nonself" and c not in ("default", "woman_p1", "man_p1"):
                continue
            idx = torch.tensor([off + j for j, lb in enumerate(labs) if lb == cls], device=lab.dev)
            for ln in TK_LENSES:
                for l in TK_LAYERS:
                    h = store[l].index_select(0, idx)
                    hJ = (h @ JLg[l].T) if ln == "jlens" else lab.lens(h, l, "jpp")
                    lg = lab.full_logits(hJ)  # [n, V] float32
                    pr = F.softmax(lg, -1).mean(0)
                    k = (c, cls, ln, l)
                    if k not in acc:
                        acc[k] = [torch.zeros_like(pr), torch.zeros_like(pr), 0]
                    acc[k][0] += pr; acc[k][1] += lg.mean(0); acc[k][2] += 1
        off += len(pos)
    del store
del JLg
print(f"accumulated in {time.time()-t0:.1f}s")
dec = lambda i: lab.tok.decode([int(i)])
WATCH = [" she", " he", " her", " his", " woman", " man", " it", " they", " herself", " himself", " female", " male", " AI", " human"]
WID = [lab.tid(w) for w in WATCH]
def top(v, k=10):
    p, i = torch.topk(v, k)
    return [(dec(t), round(float(x), 4)) for x, t in zip(p.tolist(), i.tolist())]
def merged(c, cls, ln, l):
    if c in ("woman", "man"):
        a, b = acc[(c + "_p1", cls, ln, l)], acc[(c + "_p2", cls, ln, l)]
        return (a[0] + b[0]) / (a[2] + b[2]), (a[1] + b[1]) / (a[2] + b[2])
    a = acc[(c, cls, ln, l)]
    return a[0] / a[2], a[1] / a[2]
out = {}
for ln in TK_LENSES:
    for l in TK_LAYERS:
        P, Lg = {}, {}
        for c in ["default", "nameless", "gemma_explicit", "woman", "man"]:
            P[c], Lg[c] = merged(c, "self", ln, l)
            ranks = {w: int((Lg[c] > Lg[c][t]).sum()) + 1 for w, t in zip(WATCH, WID)}
            out[f"{ln}|L{l}|self|{c}"] = {"top10_prob": top(P[c]), "watch_rank_by_mean_logit": ranks,
                                          "watch_prob": {w: round(float(P[c][t]), 6) for w, t in zip(WATCH, WID)}}
        a = acc[("default", "nonself", ln, l)]
        out[f"{ln}|L{l}|nonself|default"] = {"top10_prob": top(a[0] / a[2])}
        bg = a[1] / a[2]
        out[f"{ln}|L{l}|contrast"] = {
            "woman_minus_man_top10": top(Lg["woman"] - Lg["man"]),
            "man_minus_woman_top10": top(Lg["man"] - Lg["woman"]),
            "default_minus_nameless_top10": top(Lg["default"] - Lg["nameless"]),
            "nameless_minus_default_top10": top(Lg["nameless"] - Lg["default"]),
            "gemma_minus_nameless_top10": top(Lg["gemma_explicit"] - Lg["nameless"]),
            "default_self_minus_nonself_top10": top(Lg["default"] - bg),
        }
json.dump(out, open(f"{E1_ROOT}/E1_self_readout_topk.json", "w"), indent=1)
for k, v in out.items():
    if k.startswith("jpp"):
        print(k, json.dumps(v)[:900])
result = {"file": f"{E1_ROOT}/E1_self_readout_topk.json", "elapsed": time.time() - t0}
