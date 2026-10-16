# S1: Does the "already clean" neuron basis line up with J-space?  + GeGLU sign-regime audit.
# Predictions: (1) if neurons are a privileged basis AND J++ is a good output map, neuron write vectors (w_down) decoded
# through J++ should be much more token-sparse than random mixtures of neurons from the same layer (rotation control).
# (2) Tool 'polarity' = sign(gelu(g)*u) is mostly sign(u); a non-trivial share of strong events may be double-negative (g<0,u<0).
import math
torch.manual_seed(0)
def sparsity(vecs, l, kind):
    hJ = lab.lens(vecs, l, kind)
    lg = lab.full_logits(hJ)                     # [N, V]
    p = F.softmax(lg, -1)
    top10 = p.topk(10, -1).values.sum(-1)        # prob mass on top-10 tokens
    z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
    kurt = (z**4).mean(-1)                       # excess concentration of logit distribution
    return top10.cpu(), kurt.cpu()
rows = []
for l in [10, 20, 30, 40, 50, 57]:
    idx = torch.randperm(21504)[:256].to(lab.dev)
    W = rt.slice_w_down(l, idx).float() * rt._post_ffn_norm_weights[l].float()   # [256, D] neuron write vectors
    # rotation control: random Gaussian mixtures of 16 neurons each, same layer
    mix = torch.randn(256, 256, device=lab.dev) * (torch.rand(256, 256, device=lab.dev) < 16/256)
    R = mix @ W
    G = torch.randn_like(W)                      # isotropic control
    out = {"layer": l}
    for name, V in (("neuron", W), ("neuron_mix16", R), ("gaussian", G)):
        for kind in ("jpp", "naive"):
            t10, k = sparsity(V, l, kind)
            out[f"{name}/{kind}/top10mass"] = round(float(t10.median()), 4)
            out[f"{name}/{kind}/kurtosis"] = round(float(k.median()), 2)
    rows.append(out)
    print(out)
# GeGLU regime audit
prompts = [lab.chat("Tell me a little about yourself.", assistant="I'm curious by nature, and I love learning about almost anything."),
           lab.ids("The number of legs on the animal that spins webs is") ,
           lab.chat("What is the capital of France?", assistant="The capital of France is Paris.")]
reg = {l: {"pp":0,"pn":0,"nn":0,"np":0, "mass_pp":0.,"mass_pn":0.,"mass_nn":0.,"mass_np":0.} for l in [10,20,30,40,50,57]}
for pr in prompts:
    ids = pr if isinstance(pr, list) else lab.ids(pr)
    store = {}
    hs = []
    for l in reg:
        m = lab.layers[l].mlp
        hs.append(m.gate_proj.register_forward_hook(lambda mod,a,o,l=l: store.__setitem__((l,"g"), o.detach())))
        hs.append(m.up_proj.register_forward_hook(lambda mod,a,o,l=l: store.__setitem__((l,"u"), o.detach())))
    try:
        lab.hf(input_ids=torch.tensor([ids], device=lab.dev), use_cache=False)
    finally:
        for h in hs: h.remove()
    for l in reg:
        g = store[(l,"g")][0].float(); u = store[(l,"u")][0].float()
        act = F.gelu(g, approximate="tanh") * u            # [T, 21504]
        top = act.abs().topk(15, -1).indices               # tool's top-15 poles per position
        gs = torch.gather(g, 1, top); us = torch.gather(u, 1, top); a = torch.gather(act, 1, top).abs()
        for key, m in (("pp",(gs>0)&(us>0)),("pn",(gs>0)&(us<0)),("nn",(gs<0)&(us<0)),("np",(gs<0)&(us>0))):
            reg[l][key] += int(m.sum()); reg[l]["mass_"+key] += float(a[m].sum())
for l, r in reg.items():
    n = sum(r[k] for k in ("pp","pn","nn","np")); M = sum(r["mass_"+k] for k in ("pp","pn","nn","np"))
    print(f"L{l}: top-15 events n={n}  g+u+ {r['pp']/n:.2%}  g+u- {r['pn']/n:.2%}  g-u- (double-neg) {r['nn']/n:.2%}  g-u+ {r['np']/n:.2%} | mass share dn {r['mass_nn']/M:.2%}")
result = {"sparsity": rows, "regimes": reg}
