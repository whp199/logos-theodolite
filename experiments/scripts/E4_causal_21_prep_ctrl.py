# E4_causal_21 (CPU, offline): ON-MANIFOLD (orthogonalized to the gender span) control directions (random isotropic directions proved destructive in Gemma-4:
# additive 2%-of-norm random steering at L25-39 -> KL 11-15 nats on neutral prompts, vs 0.1-0.3 for u_self).
#  ctrl_nat{0,1,2}: covariance-shaped random directions sum_j z_j (x_j - mean) over ~1000 cached per-passage mean states
#                   (self/bg/user tokens, all 14 conditions), z ~ N(0,1)  -> random direction within natural variation.
#  ctrl_topic: content direction = mean(passages 0-11) - mean(passages 12-23) within the woman+man persona self states
#              (gender-balanced, so not a gender direction).
#  ctrl_name:  Atlas_AI - nameless_AI self states (a named-vs-unnamed AI contrast without gender).
import torch, torch.nn.functional as F
C = "./experiments/results/E4_causal_cache"
S1 = torch.load(f"{C}/self.pt"); S2 = torch.load(f"{C}/self2.pt"); DC = torch.load(f"{C}/dirs_causal.pt")
X = torch.cat([S1["self_post"], S1["bg_post"], S1["user_post"], S2["self_post"], S2["bg_post"], S2["user_post"]]).double()  # [N,60,D]
Xc = X - X.mean(0, keepdim=True)
g = torch.Generator().manual_seed(77)
dirs = {}
for k in range(3):
    z = torch.randn(X.shape[0], generator=g, dtype=torch.float64)
    dirs[f"ctrl_nat{k}"] = torch.einsum("n,nld->ld", z, Xc)
def idx(S, c):
    it = S["meta"]["items"]; d = {x["p"]: i for i, x in enumerate(it) if x["cond"] == c}; return [d[p] for p in range(24)]
W = S1["self_post"][idx(S1, "woman")].double(); M = S1["self_post"][idx(S1, "man")].double()
WM = 0.5 * (W + M)
dirs["ctrl_topic"] = WM[:12].mean(0) - WM[12:].mean(0)
dirs["ctrl_name"] = (S1["self_post"][idx(S1, "Atlas_AI")].double() - S1["self_post"][idx(S1, "nameless_AI")].double()).mean(0)
us = DC["dirs"]["u_self"].double()
# orthogonalize every control against the gender directions (u_self, u_nn, u_aiexp, u_cont) per layer
G = torch.stack([DC["dirs"][k].double() for k in ("u_self", "u_nn", "u_aiexp", "u_cont")], 1)  # [60,4,D]
Q, _ = torch.linalg.qr(G.transpose(1, 2))  # [60,D,4] orthonormal basis of the gender span
for k, v in dirs.items():
    v = v - torch.einsum("ldk,lk->ld", Q, torch.einsum("ldk,ld->lk", Q, v))
    v = F.normalize(v, dim=-1)
    print(k, "cos with u_self L15/L30/L45:", [round(float((v[l] * us[l]).sum()), 3) for l in (15, 30, 45)])
    dirs[k] = v.float()
torch.save({"dirs": dirs}, f"{C}/dirs_ctrl.pt")
