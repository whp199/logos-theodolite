import torch, numpy as np, glob, json
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt"); sd=S["std_all"]
pers=torch.from_numpy(np.load("./experiments/results/V6_quiet_dims_pers181.npy"))
q10=torch.stack([sd[l]<torch.quantile(sd[l],.1) for l in range(60)])
deep=torch.stack([sd[l]<0.05*sd[l].median() for l in range(60)])
gin=S["g_w_in"].abs()
top5g=torch.stack([gin[min(l+1,59)]>=torch.quantile(gin[min(l+1,59)],.95) for l in range(60)])  # top-5% read-gain dims of next layer
meta,G=[],[]
for f in sorted(glob.glob(R+"E3_lens_validation_vjp_chunk*.pt")):
    d=torch.load(f); meta+=d["meta"]; G.append(d["G"][:,0])  # pron only
G=torch.cat(G,0).float()  # [n,60,D]
PB=torch.load(R+"E3_lens_validation_pullbacks.pt")
groups=sorted(set(m["group"] for m in meta)); gidx={g:[i for i,m in enumerate(meta) if m["group"]==g] for g in groups}
src=np.array([m["src"] for m in meta])
rng=np.random.default_rng(0)
BANDS={"L20-29":range(20,30),"L30-39":range(30,40),"L40-49":range(40,50),"L50-58":range(50,59)}
def cosb(a,b,M):  # a [60,D], b [60,D], M [60,D] keep-mask -> [60]
    a=a*M; b=b*M; return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-30)
masks={"full":torch.ones(60,5376,dtype=torch.bool),"no_q10":~q10,"no_deep":~deep,"no_pers":~pers.unsqueeze(0).expand(60,-1),"no_top5gain":~top5g}
out={}
for subset in ("all","chat","prose"):
    gs=[g for g in groups if subset=="all" or g.startswith(subset)]
    for pk in ("jpp","jlens"):
        for mn,M in masks.items():
            idx=[i for g in gs for i in gidx[g]]
            c0=cosb(G[idx].mean(0),PB[pk][0],M).numpy()
            bs=[]
            for _ in range(300):
                pick=rng.choice(len(gs),len(gs)); ii=[i for k in pick for i in gidx[gs[k]]]
                bs.append(cosb(G[ii].mean(0),PB[pk][0],M).numpy())
            bs=np.array(bs)
            # split-half reliability in this subspace
            sh=[]
            for _ in range(100):
                perm=rng.permutation(len(gs)); A=[i for k in perm[:len(gs)//2] for i in gidx[gs[k]]]; B=[i for k in perm[len(gs)//2:] for i in gidx[gs[k]]]
                sh.append(cosb(G[A].mean(0),G[B].mean(0),M).numpy())
            sh=np.array(sh).mean(0)
            out[f"{subset}|{pk}|{mn}"]={b:[float(c0[list(r)].mean()),float(np.percentile(bs[:,list(r)].mean(1),2.5)),float(np.percentile(bs[:,list(r)].mean(1),97.5)),float(sh[list(r)].mean())] for b,r in BANDS.items()}
for k,v in out.items():
    if k.split("|")[1]=="jpp" or k.split("|")[2] in ("full","no_q10"):
        print(f"{k:28s} "+"  ".join(f"{b}: {x[0]:+.2f} [{x[1]:+.2f},{x[2]:+.2f}] sh{x[3]:.2f}" for b,x in v.items()))
json.dump(out,open(R+"V6_quiet_dims_07_gradalign_raw.json","w"))
