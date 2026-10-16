import torch, numpy as np
R="./experiments/results/"
S=torch.load(R+"V6_quiet_dims_stats.pt")
sd=S["std_all"].numpy(); mu=S["mean_all"].numpy(); D=5376
pers=np.load("./experiments/results/V6_quiet_dims_pers181.npy")
g={k[2:]:S[k].numpy() for k in S if k.startswith("g_")}
fnw=S["fnw"].numpy(); wu=S["wu_col"].numpy()
def deep(l): s=sd[l]; return s<0.05*np.median(s)
def q10(l): s=sd[l]; return s<np.quantile(s,.1)
def spear(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
print("scalars", np.round(g["scalar"].ravel(),3))
print("Spearman(std of resid_post[l], gain of the NEXT reader)  [read gains: |w_in[l+1]|, |w_in|*qkv_col, |w_preffn[l+1]|, |w_pre|*gu_col ; fnw ; write gains of layer l: |w_postattn|, |w_postffn|*down_row]")
print("  l   in   in*col  pre  pre*col  fnw  fnw*wu | wpostattn  wpostffn  postffn*down  o_row | med|w_in| quiet  loud | med|w_pre| quiet loud | med fnw quiet loud")
for l in [5,10,15,20,25,30,35,39,42,45,48,51,54,57,58]:
    s=np.log(sd[l]); n=l+1
    win=np.abs(g["w_in"][n]); wpre=np.abs(g["w_preffn"][n])
    r=[spear(s,win), spear(s,win*g["qkv_col"][n]), spear(s,wpre), spear(s,wpre*g["gu_col"][n]), spear(s,np.abs(fnw)), spear(s,np.abs(fnw)*wu),
       spear(s,np.abs(g["w_postattn"][l])), spear(s,np.abs(g["w_postffn"][l])), spear(s,np.abs(g["w_postffn"][l])*g["down_row"][l]), spear(s,g["o_row"][l])]
    Q=deep(l) if deep(l).sum()>50 else q10(l)
    print(f" L{l:02d} "+" ".join(f"{x:+.2f}" for x in r), f"| {np.median(win):.1f} {np.median(win[Q]):.1f} {np.median(win[~Q]):.1f} | {np.median(wpre):.1f} {np.median(wpre[Q]):.1f} {np.median(wpre[~Q]):.1f} | {np.median(np.abs(fnw)):.1f} {np.median(np.abs(fnw[Q])):.1f} {np.median(np.abs(fnw[~Q])):.1f}", "nQ",Q.sum())
# write gains for persistent dims across layers
print("\npersistent-quiet dims (n=%d): median write gain ratio quiet/other (|w_postattn|, |w_postffn|) and read gain ratio (|w_in|, |w_pre|) per layer"%pers.sum())
for l in [2,5,10,15,20,25,28,30,32,35,40,45,50,55,58]:
    f=lambda v: np.median(np.abs(v[pers]))/np.median(np.abs(v[~pers]))
    print(f" L{l:02d} write attn {f(g['w_postattn'][l]):.3f} ffn {f(g['w_postffn'][l]):.3f} | read in {f(g['w_in'][l]):.2f} pre {f(g['w_preffn'][l]):.2f} | std ratio {np.median(sd[l][pers])/np.median(sd[l][~pers]):.3g} | |mean| ratio {np.median(np.abs(mu[l][pers]))/np.median(np.abs(mu[l][~pers])):.3g}")
print(" final norm |fnw| ratio", np.median(np.abs(fnw[pers]))/np.median(np.abs(fnw[~pers])), " fnw*wu ratio", np.median(np.abs(fnw[pers])*wu[pers])/np.median(np.abs(fnw[~pers])*wu[~pers]))
print(" top-k |fnw| dims in persistent set: top50", np.isin(np.argsort(-np.abs(fnw))[:50], np.where(pers)[0]).mean(), "top200", np.isin(np.argsort(-np.abs(fnw))[:200], np.where(pers)[0]).mean())
for l in [30,40,50]:
    n=l+1
    print(f" L{l}: top-100 |w_in[{n}]| dims in persistent set: {np.isin(np.argsort(-np.abs(g['w_in'][n]))[:100], np.where(pers)[0]).mean():.2f};  top-100 |w_pre| {np.isin(np.argsort(-np.abs(g['w_preffn'][n]))[:100], np.where(pers)[0]).mean():.2f}")
