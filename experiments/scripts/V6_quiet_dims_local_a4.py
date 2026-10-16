import torch, numpy as np
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt"); T=torch.load(R+"V6_quiet_dims_samples.pt")
sd=S["std_all"].numpy(); mu=S["mean_all"].numpy(); D=5376
amax=S["amax"].numpy(); bos=S["bos"].numpy(); atok=S["amax_tok"].numpy()
g={k[2:]:S[k].numpy() for k in S if k.startswith("g_")}; fnw=S["fnw"].numpy()
pers=np.load("./experiments/results/V6_quiet_dims_pers181.npy")
import json
from transformers import AutoTokenizer
print("massive activations: per layer, top-5 dims by max|h| over non-BOS tokens (dim, max, std, in persistent?)  and top BOS dims")
for l in [3,10,20,30,40,50,58]:
    o=np.argsort(-amax[l])[:5]; ob=np.argsort(-np.abs(bos[l]))[:5]
    med=np.median(amax[l])
    print(f"L{l}: med max {med:.2f}; top:", [(int(i), round(float(amax[l][i]),1), round(float(sd[l][i]),3), bool(pers[i]), int(atok[l][i])) for i in o])
    print(f"      BOS top:", [(int(i), round(float(bos[l][i]),1), bool(pers[i])) for i in ob], " |BOS| median", round(float(np.median(np.abs(bos[l]))),3), " ||BOS||/med||h||", )
# overlap of persistent quiet set with top-1% amax dims and top-1% |BOS| dims at each layer
print("overlap of persistent-quiet set with top-54 (1%) max|h| dims / top-54 |BOS| dims / top-54 |mean| dims")
for l in [3,10,15,20,25,30,35,40,45,50,55,58]:
    top=lambda v: set(np.argsort(-v)[:54].tolist())
    P=set(np.where(pers)[0].tolist())
    print(f" L{l:02d}: amax {len(top(amax[l])&P)/54:.2f}  bos {len(top(np.abs(bos[l]))&P)/54:.2f}  mean {len(top(np.abs(mu[l]))&P)/54:.2f}   persistent dims: med max|h| {np.median(amax[l][pers]):.3g} vs others {np.median(amax[l][~pers]):.3g}; med|BOS| {np.median(np.abs(bos[l][pers])):.3g} vs {np.median(np.abs(bos[l][~pers])):.3g}")
# energy shares in samples
H=T["H"].float()  # [N,60,D]
print("\nraw energy share of persistent dims in ||h||^2 (tokens), of BOS, and of mean")
for l in [5,10,15,20,25,30,35,40,45,50,55,58]:
    h=H[:,l].numpy(); e=(h[:,pers]**2).sum(1)/(h**2).sum(1)
    print(f" L{l:02d}: tokens {e.mean():.4f}  BOS {(bos[l][pers]**2).sum()/(bos[l]**2).sum():.4f}  mean-vec {(mu[l][pers]**2).sum()/(mu[l]**2).sum():.4f}  ||h|| med {np.median(np.linalg.norm(h,axis=1)):.1f}  ||BOS|| {np.linalg.norm(bos[l]):.1f}")
# effective (normalized-input) coordinates: z = w_in[l+1]*h/rms(h)
print("\nEFFECTIVE coordinates of next layer's input: std over tokens of z_i = w_in[l+1]_i h_i / rms(h); persistent vs others; energy share of persistent dims in ||z||^2 and in token-varying part")
for l in [20,25,30,35,40,45,50,55,57,58]:
    h=H[:,l].numpy(); rms=np.sqrt((h**2).mean(1,keepdims=True)); 
    for nm,w in (("in",g["w_in"][l+1]),("pre",g["w_preffn"][l+1])):
        z=w*h/rms; zs=z.std(0); zc=z-z.mean(0)
        print(f" L{l:02d} {nm}: med std(z) pers {np.median(zs[pers]):.3g} others {np.median(zs[~pers]):.3g} ratio {np.median(zs[pers])/np.median(zs[~pers]):.2f} | share ||z||^2 {((z[:,pers]**2).sum(1)/(z**2).sum(1)).mean():.3f} | share var {(zc[:,pers]**2).sum()/(zc**2).sum():.3f}")
h=H[:,59].numpy(); rms=np.sqrt((h**2).mean(1,keepdims=True)); z=fnw*h/rms; zc=z-z.mean(0)
print(f" L59 final-norm: share ||z||^2 {((z[:,pers]**2).sum(1)/(z**2).sum(1)).mean():.3f} share var {(zc[:,pers]**2).sum()/(zc**2).sum():.3f}  ; std ratio raw at L59 {np.median(sd[59][pers])/np.median(sd[59][~pers]):.3f}")
