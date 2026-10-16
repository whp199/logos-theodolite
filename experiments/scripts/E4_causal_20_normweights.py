# E4_causal_20: dump RMSNorm weights of all decoder layers (to explain hypersensitivity to random/off-manifold directions).
exec(open("./experiments/scripts/E4_causal_common.py").read())
W = {}
for nm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
    W[nm] = torch.stack([getattr(lab.layers[l], nm).weight.detach().float().cpu() for l in range(60)])
L0 = lab.layers[30].input_layernorm
import inspect
print(inspect.getsource(type(L0).forward)[:1500])
print({k: (float(v.min()), float(v.max()), float(v.abs().mean())) for k, v in W.items()})
torch.save(W, f"{E4_CACHE}/normweights.pt")
result = {"ok": True}
