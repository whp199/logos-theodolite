import torch, numpy as np
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt")
sd=S["std_all"].numpy(); mu=S["mean_all"].numpy()
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("layer  q.005 q.01 q.02 q.05 q.10 q.20 q.50 q.90 max | n(std<0.05*med) n(<0.1*med) n(<0.01*med) | log10 hist")
for l in list(range(0,60,3))+[59]:
    s=sd[l]; med=np.median(s)
    qs=np.quantile(s,[.005,.01,.02,.05,.1,.2,.5,.9])
    h,_=np.histogram(np.log10(s+1e-9),bins=np.arange(-4,2.01,0.5))
    print(f"L{l:02d}", " ".join(f"{x:.3g}" for x in qs), f"{s.max():.3g}", "|", (s<0.05*med).sum(), (s<0.1*med).sum(), (s<0.01*med).sum(), "|", h)
