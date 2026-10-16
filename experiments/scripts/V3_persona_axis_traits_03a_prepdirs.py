# V3_persona_axis_traits_03a (offline CPU): steering directions (raw full-sample diff-of-means at self tokens, L20-50).
import torch
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
A = torch.load(f"{V3_CACHE}/self_A.pt"); B = torch.load(f"{V3_CACHE}/self_B.pt")
def cm(S, c):
    idx = [i for i, it in enumerate(S["items"]) if it["cond"] == c]
    return S["self"][idx].double().mean(0)  # [31, D]
M = {}
for S in (A, B):
    for c in dict.fromkeys(it["cond"] for it in S["items"]):
        M[c] = cm(S, c)
pool = lambda cs: sum(M[c] for c in cs) / len(cs)
d = {}
d["persona"] = pool([w for w, m in PAIRS["persona"]]) - pool([m for w, m in PAIRS["persona"]])
d["trait"] = pool([a for a, b in PAIRS["trait"]]) - pool([b for a, b in PAIRS["trait"]])
d["role"] = pool([a for a, b in PAIRS["role"]]) - pool([b for a, b in PAIRS["role"]])
d["ctrl1"] = M["CT1_org"] - M["CT1_dis"]
d["ctrl2"] = M["CT2_city"] - M["CT2_rural"]
d["AIoff"] = M["default"] - 0.5 * (M["neutral_human"] + M["neutral_human2"])
u = lambda v: v / v.norm(dim=-1, keepdim=True)
d["trait_perp_persona"] = d["trait"] - (d["trait"] * u(d["persona"])).sum(-1, keepdim=True) * u(d["persona"])
d["persona_perp_trait"] = d["persona"] - (d["persona"] * u(d["trait"])).sum(-1, keepdim=True) * u(d["trait"])
for k, v in d.items():
    print(k, "norm L25/L30/L35", [round(float(v[l - 20].norm()), 1) for l in (25, 30, 35)],
          "cos w persona L30", round(float(torch.nn.functional.cosine_similarity(v[10], d["persona"][10], dim=0)), 3))
torch.save({"layers": LAYERS, "dirs": {k: v.float() for k, v in d.items()}}, f"{V3_CACHE}/dirs.pt")
