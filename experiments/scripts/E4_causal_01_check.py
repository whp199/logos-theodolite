# E4_causal_01: sanity checks - tokenization of self tokens, 3rd-person continuation identity, decomposition identity.
exec(open("./experiments/scripts/E4_causal_common.py").read())
L0 = lab.layers[0]
print("enable_moe_block", getattr(L0, "enable_moe_block", None), "per_layer", getattr(L0, "hidden_size_per_layer_input", None))
items = self_items()
it = items[0]
p = lab.chat(QA[it["p"]][0], system=it["sys"], assistant=QA[it["p"]][1])
ids = lab.ids(p); strs = lab.strs(ids); mt = lab.model_turn_pos(ids)
print(repr(p[:400]))
print([ (i, strs[i]) for i in range(mt-2, len(ids))][:60])
selfpos = [i for i in range(mt+1, len(ids)) if strs[i].strip() in SELF_TOKS]
print("self", [(i, strs[i]) for i in selfpos])
# third person identity
th = third_items()
bad = 0
for k in range(0, len(th), 2):
    a, b = th[k], th[k+1]
    ia, ib = lab.ids(a["text"]), lab.ids(b["text"])
    na, nb = len(lab.ids(a["prefix"])), len(lab.ids(b["prefix"]))
    if ia[na:] != ib[nb:]:
        bad += 1; print("MISMATCH", a["text"], b["text"])
print("third pairs", len(th)//2, "mismatches", bad)
print(lab.strs(lab.ids(th[16]["text"])))
# decomposition identity
rec = {}
def red(kind, l, h):
    if l in (0, 10, 30, 59) or kind == "pre0":
        rec[(kind, l)] = h[0].float().clone()
    if kind == "post" and l in (9, 29, 58):
        rec[("post", l)] = h[0].float().clone()
lg = batched_hook_forward(lab, [ids], red, want_parts=True)
for l in (0, 10, 30, 59):
    s = lab.layers[l].layer_scalar.float()
    pre = rec[("pre0", 0)] if l == 0 else rec[("post", l-1)]
    recon = s * (pre + rec[("A", l)] + rec[("M", l)])
    post = rec[("post", l)]
    err = (recon - post).norm(dim=-1) / post.norm(dim=-1)
    print(l, "s", float(s), "rel err max", float(err[1:].max()), "mean", float(err[1:].mean()))
# compare to lab.decompose on capture
caps = await lab.capture(ids, points=("resid_pre","resid_mid","resid_post"))
d = lab.decompose(caps, 30)
A30 = float(lab.layers[30].layer_scalar.float()) * rec[("A", 30)]
print("cos(decompose.attn, s*A) L30", float(F.cosine_similarity(d["attn"][5:], A30[5:].to(d["attn"].device), dim=-1).mean()))
print("rel diff", float(((d["attn"]-A30.to(d["attn"].device)).norm()/A30.norm())))
pg = pron_logodds_from_logits(lab, lg)
print("next-token", {k: float(v[0]) for k,v in pg.items()})
result = {"ok": True}
