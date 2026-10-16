import torch, numpy as np, json
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt"); T=torch.load(R+"V6_quiet_dims_samples.pt")
sd=S["std_all"]; D=5376
pers=torch.from_numpy(np.load("./experiments/results/V6_quiet_dims_pers181.npy"))
q10=torch.stack([sd[l]<torch.quantile(sd[l],.1) for l in range(60)])
H=T["H"].float(); kinds=np.array([m["kind"] for m in T["meta"]])
PB=torch.load(R+"E3_lens_validation_pullbacks.pt"); gb=torch.load(R+"E3_lens_validation_gbar.pt")
def corr(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/torch.sqrt((a*a).sum()*(b*b).sum()))
rng=np.random.default_rng(0)
LS=[20,25,30,35,40,45,50,55,58]
res={}
print("Readout agreement on natural tokens: corr over tokens of <h,gbar_l> vs <h,PB_l> (pron). Variants: full; loud-only (quiet coords -> mean); quiet-only. Set = q10 (bottom 10% std)")
print("Also: share of readout variance from quiet coords for each covector.  [chat tokens n=%d, prose n=%d]"%((kinds=='chat').sum(),(kinds=='prose').sum()))
for gname in ("gbar","gbar_chat","gbar_prose"):
  for pk in ("jpp","jlens"):
    for tk in ("all","chat","prose"):
      idx=np.arange(len(kinds)) if tk=="all" else np.where(kinds==tk)[0]
      line=[]
      for l in LS:
        X=H[idx,l]; Xc=X-X.mean(0)
        M=q10[l].float()
        g=gb[gname][0,l]; p=PB[pk][0,l]
        full=corr(Xc@g, Xc@p); loud=corr(Xc@(g*(1-M)), Xc@(p*(1-M))); quiet=corr(Xc@(g*M), Xc@(p*M))
        vq_g=float((Xc@(g*M)).var()/(Xc@g).var()); vq_p=float((Xc@(p*M)).var()/(Xc@p).var())
        # bootstrap CI over tokens for full and loud
        bs=[]
        for _ in range(200):
            ii=rng.integers(0,len(idx),len(idx)); Y=Xc[ii]
            bs.append((corr(Y@g,Y@p), corr(Y@(g*(1-M)),Y@(p*(1-M)))))
        bs=np.array(bs)
        line.append((l,full,loud,quiet,vq_g,vq_p,np.percentile(bs[:,0],2.5),np.percentile(bs[:,0],97.5),np.percentile(bs[:,1],2.5),np.percentile(bs[:,1],97.5)))
      res[f"{gname}|{pk}|{tk}"]=line
      if pk=="jpp" or gname=="gbar":
        print(f"{gname:10s} {pk:5s} {tk:5s} "+" ".join(f"L{l}: {f:+.2f}/{lo:+.2f}/{q:+.2f} (vq {vg:.2f},{vp:.2f})" for (l,f,lo,q,vg,vp,*_) in line))
json.dump(res,open(R+"V6_quiet_dims_04_dataagree.json","w"))
# raw cos restricted
print("\nRaw cosines cos(gbar, PB) full / loud-only / quiet-only (q10 set):")
for gname in ("gbar","gbar_chat","gbar_prose"):
  for pk in ("jpp","jlens"):
    s=[]
    for l in LS:
        M=q10[l].float(); g=gb[gname][0,l]; p=PB[pk][0,l]
        c=lambda a,b: float(a@b/(a.norm()*b.norm()+1e-30))
        s.append(f"L{l}: {c(g,p):+.2f}/{c(g*(1-M),p*(1-M)):+.2f}/{c(g*M,p*M):+.2f}")
    print(f"{gname:10s} {pk:5s} "+" ".join(s))
