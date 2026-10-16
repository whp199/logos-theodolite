# S2: What does J++ output for random inputs? (lens prior) and how much of real-state readouts it explains.
from collections import Counter
torch.manual_seed(1)
ids = lab.ids(lab.chat("Tell me a little about yourself.", assistant="I'm curious by nature, and I love learning about almost anything. I like long conversations."))
caps = await lab.capture(ids)
H = caps["resid_post"]
out = {}
for l in [10, 20, 30, 40, 50, 57]:
    G = torch.randn(200, 5376, device=lab.dev)
    hJ = lab.lens(G, l, "jpp").float()
    lg = lab.full_logits(hJ)
    top1 = [lab.tok.decode([int(t)]) for t in lg.argmax(-1).tolist()]
    c = Counter(top1).most_common(5)
    # singular structure of J: share of output energy for isotropic input in top-k output directions
    S = torch.linalg.svdvals(lab.J[l].float())
    e = (S**2).cumsum(0) / (S**2).sum()
    # real states: cosine between real hJ and the mean random-input hJ direction (prior direction)
    prior = hJ.mean(0)
    real = lab.lens(H[l].to(lab.dev), l, "jpp").float()
    cos_real_prior = F.cosine_similarity(real, prior.unsqueeze(0), dim=-1)
    rr = F.cosine_similarity(hJ, prior.unsqueeze(0), dim=-1)
    out[l] = {"random_top1_most_common": c, "J_energy_top1/top10/top100": [round(float(e[0]),3), round(float(e[9]),3), round(float(e[99]),3)],
              "cos(random hJ, prior) median": round(float(rr.median()),3),
              "cos(real hJ, prior) median": round(float(cos_real_prior.median()),3)}
    print(l, out[l])
result = out
