# E3_lens_validation smoke: timing + basic facts (layer scalars, VJP cost, multi-direction backward)
import time
t0=time.time()
s = [float(lab.layers[l].layer_scalar.float().mean()) for l in range(60)]
print("layer_scalar:", [round(x,3) for x in s])
txt = "The river Thames flows through southern England, passing through Oxford and London before reaching the North Sea. It has been an important trade route for centuries."
ids = lab.ids(txt)
print("T=", len(ids))
torch.cuda.reset_peak_memory_stats()
t=time.time(); G = lab.same_pos_vjp(ids, len(ids)-1, lab.gender_direction("pron")); torch.cuda.synchronize()
print("vjp time", time.time()-t, "peak GB", torch.cuda.max_memory_allocated()/2**30, G.shape, G.norm(dim=-1)[[0,10,30,50,58,59]].tolist())
t=time.time(); X = lab.forward_resid([ids]); print("forward_resid", time.time()-t, X.shape)
t=time.time()
for l in range(60):
    lg = lab.full_logits(lab.lens(X[0,l].to(lab.dev), l, "jlens"))
torch.cuda.synchronize(); print("jlens 60-layer full logits", time.time()-t)
t=time.time()
for l in range(60):
    lg = lab.full_logits(lab.lens(X[0,l].to(lab.dev), l, "jpp"))
torch.cuda.synchronize(); print("jpp 60-layer full logits", time.time()-t)
print("resid norms", X[0,:,-1].norm(dim=-1)[[0,10,20,30,40,50,58,59]].tolist())
print("VRAM alloc", torch.cuda.memory_allocated()/2**30, "STORE keys", list(STORE.keys())[:20])
result = {"layer_scalar": s}
