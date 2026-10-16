import torch, numpy as np, glob, json
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt")
sd=S["std_all"]; D=5376
pers=torch.from_numpy(np.load("./experiments/results/V6_quiet_dims_pers181.npy"))
q10=torch.stack([sd[l]<torch.quantile(sd[l],.1) for l in range(60)])
deep=torch.stack([sd[l]<0.05*sd[l].median() for l in range(60)])
meta,G=[],[]
for f in sorted(glob.glob(R+"E3_lens_validation_vjp_chunk*.pt")):
    d=torch.load(f); meta+=d["meta"]; G.append(d["G"])
G=torch.cat(G,0).float()  # [n,4,60,D]
PB=torch.load(R+"E3_lens_validation_pullbacks.pt")
gb=torch.load(R+"E3_lens_validation_gbar.pt")
print("n", G.shape)
def share(v, M):  # v [...,60,D], M [60,D] bool
    e=v**2; return ((e*M).sum(-1)/e.sum(-1))
LS=[10,20,25,30,35,40,45,50,55,58]
out={}
for mname,M in (("q10",q10),("deep",deep),("pers",pers.unsqueeze(0).expand(60,-1))):
    print(f"=== energy share in {mname} set (sizes at L30/L45: {int(M[30].sum())}/{int(M[45].sum())})")
    rows={}
    sg=share(G,M)  # [n,4,60]
    rows["VJP single (pron)"]=sg[:,0].mean(0); rows["VJP single (rand1)"]=sg[:,2].mean(0)
    rows["VJP gbar (pron)"]=share(gb["gbar"][0],M); rows["VJP gbar_chat"]=share(gb["gbar_chat"][0],M); rows["VJP gbar_prose"]=share(gb["gbar_prose"][0],M)
    rows["J++ pb pron"]=share(PB["jpp"][0],M); rows["J++ pb rand1"]=share(PB["jpp"][2],M)
    rows["JLens pb pron"]=share(PB["jlens"][0],M); rows["JLens pb rand1"]=share(PB["jlens"][2],M)
    rows["isotropic (expected)"]=M.float().mean(-1)
    for k,v in rows.items():
        print(f" {k:22s}"+" ".join(f"L{l}:{float(v[l]):.3f}" for l in LS))
    out[mname]={k:[float(x) for x in v] for k,v in rows.items()}
# gain-weighted check: does VJP energy follow w_in[l+1]^2?
g_in=S["g_w_in"]; 
print("Spearman over dims of single-position VJP energy vs |w_in[l+1]|  (mean over positions), pron")
def spear(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float(); return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])
for l in [20,30,40,50,58]:
    e=(G[:,0,l]**2).mean(0)
    print(f" L{l}: spearman(E g_i^2, |w_in[l+1]_i|) {spear(e,g_in[l+1].abs()):+.2f}  spearman(E g^2, 1/std) {spear(e,1/sd[l]):+.2f}  J++pb^2 vs 1/std {spear(PB['jpp'][0,l]**2,1/sd[l]):+.2f}")
json.dump(out,open(R+"V6_quiet_dims_03_energy_shares.json","w"))
