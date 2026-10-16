# R1_IT_LENS 09 check: multi-position copies pass == single-position copies pass (PREDICTION: cos >= 0.99 at all layers).
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/R1_vjplib.py").read())
P = json.load(open(R1_RES + "R1_02_positions.json"))
DD = torch.load(R1_RES + "R1_02_dirs.pt")
d = DD["dirs"][:8].to(lab.dev)
ids = P[3]["ids"]; ps = [p for p, _ in P[3]["pos"]][:4]
R = d.repeat(4, 1)                      # copy k -> position ps[k // 8], dir k % 8
poss = [p for p in ps for _ in range(8)]
t0 = time.time(); Gm = r1_vjp_multi(lab, ids, poss, R); tm = time.time() - t0
out = []
for j, p in enumerate(ps):
    Gs, _ = r1_vjp_copies(lab, ids, p, d)
    c = F.cosine_similarity(Gm[8 * j:8 * j + 8], Gs, dim=-1)[:, :59]
    out.append({"pos": p, "cos_min": float(c.min()), "cos_med": float(c.median())})
print(out, "t_multi", round(tm, 2))
result = out
