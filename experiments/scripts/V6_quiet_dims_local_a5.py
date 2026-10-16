import torch, numpy as np, glob
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt"); T=torch.load(R+"V6_quiet_dims_samples.pt")
sd=S["std_all"].numpy(); mu=S["mean_all"].numpy(); D=5376
sc=S["g_scalar"].numpy().ravel()
pers=np.load("./experiments/results/V6_quiet_dims_pers181.npy")
H=T["H"].float().numpy()
# memory channel test: for persistent dims, per-dim corr across tokens between layer a and layer b, and regression slope vs prod of scalars
print("persistent dims: median per-dim corr(h_a, h_b) across 540 tokens; median slope; prod s_(a+1..b)")
for a,b in [(20,25),(20,30),(25,35),(25,45),(30,45),(35,45),(40,55),(45,58),(20,45)]:
    xa=H[:,a][:,pers]; xb=H[:,b][:,pers]
    xa=xa-xa.mean(0); xb=xb-xb.mean(0)
    c=(xa*xb).sum(0)/np.sqrt((xa**2).sum(0)*(xb**2).sum(0)+1e-30)
    sl=(xa*xb).sum(0)/((xa**2).sum(0)+1e-30)
    print(f" L{a}->L{b}: corr {np.median(c):+.3f} (IQR {np.percentile(c,25):+.2f},{np.percentile(c,75):+.2f})  slope {np.median(sl):.3g}  prod s {np.prod(sc[a+1:b+1]):.3g}")
# same for loud dims as reference
loud=~pers
xa=H[:,25][:,loud]; xb=H[:,45][:,loud]; xa=xa-xa.mean(0); xb=xb-xb.mean(0)
c=(xa*xb).sum(0)/np.sqrt((xa**2).sum(0)*(xb**2).sum(0)+1e-30); print(" loud dims L25->L45 corr median", np.median(c))
