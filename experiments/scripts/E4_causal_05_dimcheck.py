# E4_causal_05 (CPU, offline): are the mean-difference directions dominated by a few massive residual dimensions?
import torch, numpy as np, torch.nn.functional as F
C = "./experiments/results/E4_causal_cache"
S = torch.load(f"{C}/self.pt"); U = torch.load(f"{C}/directions.pt"); PB = torch.load(f"{C}/pullbacks.pt")
items = S["meta"]["items"]
Iw = [i for i, it in enumerate(items) if it["cond"] == "woman"]; Im = [i for i, it in enumerate(items) if it["cond"] == "man"]
Iai = [i for i, it in enumerate(items) if it["cond"] not in ("woman", "man")]
X = S["self_post"].double()
vjp = PB["vjp_pron"].double().mean(0)
for l in (5, 15, 25, 35, 45, 55):
    u = U["u_self"][l].double(); uc = U["u_cont"][l].double()
    mu = X[:, l].mean(0); sd = X[:, l].std(0)
    top = torch.argsort(mu.abs(), descending=True)[:8]
    def conc(v, k):
        s = (v ** 2); return float(s[torch.argsort(s, descending=True)[:k]].sum() / s.sum())
    print(f"L{l}: |mean resid| top dims {top.tolist()} vals {[round(float(mu[t]),1) for t in top[:4]]}  mean-resid top10 energy {conc(mu,10):.3f}")
    print(f"   u_self top1/10/100 energy {conc(u,1):.3f}/{conc(u,10):.3f}/{conc(u,100):.3f}; u_cont {conc(uc,1):.3f}/{conc(uc,10):.3f}/{conc(uc,100):.3f}; vjp {conc(vjp[l],1):.3f}/{conc(vjp[l],10):.3f}; pbjpp {conc(PB['pron_jpp'][l].double(),10):.3f}")
    ut = torch.argsort(u ** 2, descending=True)[:5]
    print(f"   u_self top dims {ut.tolist()} (in mean-resid top8? {[int(t) in top.tolist() for t in ut]})")
    # cos after removing top-k energy dims of the mean residual
    for k in (0, 10, 50):
        mask = torch.ones_like(u); mask[torch.argsort(mu.abs(), descending=True)[:k]] = 0
        print(f"   drop top{k:3d} resid dims: cos(u_self,u_cont)={float(F.cosine_similarity(u*mask, uc*mask, dim=0)):.3f}  cos(u_self,vjp)={float(F.cosine_similarity(u*mask, vjp[l]*mask, dim=0)):.3f}  cos(u_cont,vjp)={float(F.cosine_similarity(uc*mask, vjp[l]*mask, dim=0)):.3f}")
    # standardized (per-dim z) directions
    us = u / sd; ucs = uc / sd
    print(f"   z-scored: cos(u_self,u_cont)={float(F.cosine_similarity(us, ucs, dim=0)):.3f}; cos(u_self*sd... vjp*sd)={float(F.cosine_similarity(u/sd, vjp[l]*sd, dim=0)):.3f}")
    # AI offset diagnostics
    mid = 0.5 * (X[Iw, l].mean(0) + X[Im, l].mean(0)); off = X[Iai, l].mean(0) - mid
    print(f"   ||AI-mid||/||W-M|| = {float(off.norm()/u.norm()):.2f}; cos(AI-mid, u_self)={float(F.cosine_similarity(off,u,dim=0)):.3f}; dist AI->W {float((X[Iai,l].mean(0)-X[Iw,l].mean(0)).norm()):.1f} AI->M {float((X[Iai,l].mean(0)-X[Im,l].mean(0)).norm()):.1f}")
