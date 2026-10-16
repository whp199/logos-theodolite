import torch, numpy as np
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt")
sd=S["std_all"].numpy(); mu=S["mean_all"].numpy()
D=5376
def q10(l, key="std_all"):
    s=S[key][l].numpy(); return s < np.quantile(s, .1)
def deep(l, key="std_all"):
    s=S[key][l].numpy(); return s < 0.05*np.median(s)
def jac(a,b): return (a&b).sum()/max((a|b).sum(),1)
L=[20,25,30,35,39,42,45,48,51,54,57]
print("Jaccard bottom-10% sets across layers")
print("     "+" ".join(f"L{l:02d} " for l in L))
for a in L: print(f"L{a:02d} "+" ".join(f"{jac(q10(a),q10(b)):.2f} " for b in L))
print("Jaccard deep-quiet sets (std<0.05*median); sizes:", [int(deep(l).sum()) for l in L])
for a in L: print(f"L{a:02d} "+" ".join(f"{jac(deep(a),deep(b)):.2f} " for b in L))
# chance Jaccard for 10% sets ~ 0.053
# stability prose vs chat, even vs odd
print("prose-vs-chat / even-vs-odd Jaccard (bottom10, deep) and spearman of log std")
from numpy import argsort
def spear(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
for l in [10,20,30,35,41,47,53,58]:
    print(l, round(jac(q10(l,"std_prose"),q10(l,"std_chat")),2), round(jac(deep(l,"std_prose"),deep(l,"std_chat")),2),
          round(jac(q10(l,"std_even"),q10(l,"std_odd")),2), round(spear(S["std_prose"][l].numpy(),S["std_chat"][l].numpy()),3))
# persistent dims: deep quiet at all of L42-57
pers = np.all([deep(l) for l in range(42,58)],0)
print("dims deep-quiet at every layer L42-57:", pers.sum(), " at >=12 of 16:", (np.sum([deep(l) for l in range(42,58)],0)>=12).sum())
union=np.any([deep(l) for l in range(36,58)],0); print("union deep L36-57", union.sum())
# where are persistent dims in early layers: their std rank at L10,20,30
rk = lambda l: np.argsort(np.argsort(sd[l]))/D
for l in [5,10,15,20,25,30,35,38]:
    print(f"  L{l}: mean std-percentile of persistent-quiet dims {rk(l)[pers].mean():.3f}, frac in deep(l) {deep(l)[pers].mean():.2f}, frac in bottom10 {q10(l)[pers].mean():.2f}")
np.save("./experiments/results/V6_quiet_dims_pers181.npy", pers)
