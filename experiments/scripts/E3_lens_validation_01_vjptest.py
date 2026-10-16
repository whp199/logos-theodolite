# E3: verify batched multi-direction VJP against labkit.same_pos_vjp, and time it.
import time
exec(open("./experiments/scripts/E3_lens_validation_vjplib.py").read())
d = lab.gender_direction("pron")
g = torch.Generator().manual_seed(0)
r1 = torch.randn(5376, generator=g); r2 = torch.randn(5376, generator=g)
dirs = torch.stack([d.cpu(), r1 * d.norm().cpu()/r1.norm(), r2 * d.norm().cpu()/r2.norm()])
txts = ["The river Thames flows through southern England, passing through Oxford and London before reaching the North Sea. It has been an important trade route for centuries, and many towns grew up along its banks.",
        lab.chat("Tell me about yourself.", assistant="I'm curious by nature, and I love learning about almost anything. I like long conversations, and I always try to be honest."),
        "In 1905 Albert Einstein published four papers that changed physics. The first explained the photoelectric effect, which later earned him the Nobel Prize.",
        lab.chat("What causes the seasons?", assistant="The seasons are caused by the tilt of the Earth's axis relative to its orbit around the Sun. When my part of the world tilts toward the Sun, days are longer.")]
seqs = [lab.ids(t) for t in txts]
poss = [len(s) - 3 for s in seqs]
torch.cuda.reset_peak_memory_stats()
t = time.time(); G = torch.cat([e3_vjp_batch(lab, seqs[:2], poss[:2], dirs), e3_vjp_batch(lab, seqs[2:], poss[2:], dirs)], 1); torch.cuda.synchronize()
tb = time.time() - t; peak = torch.cuda.max_memory_allocated()/2**30
print("batched 2x(B=2) K=3 time", tb, "peak", peak, "T", [len(s) for s in seqs])
torch.cuda.reset_peak_memory_stats()
t = time.time(); Gs = e3_vjp_batch(lab, seqs[1:2], poss[1:2], dirs); torch.cuda.synchronize()
print("B=1 K=3 time", time.time()-t, "peak", torch.cuda.max_memory_allocated()/2**30)
t = time.time(); G1 = lab.same_pos_vjp(seqs[1], poss[1], d); t1 = time.time() - t
G2 = lab.same_pos_vjp(seqs[3], poss[3], dirs[2])
cos1 = F.cosine_similarity(G[0, 1].float(), G1.cpu().float(), dim=-1)
cos2 = F.cosine_similarity(G[2, 3].float(), G2.cpu().float(), dim=-1)
rel1 = ((G[0,1]-G1.cpu()).norm(dim=-1)/G1.cpu().norm(dim=-1))
print("single vjp time", t1)
print("cos batched vs single (gender, item1) min/mean", cos1.min().item(), cos1.mean().item(), "rel err max", rel1.max().item())
print("cos batched vs single (rand2, item3) min/mean", cos2.min().item(), cos2.mean().item())
result = {"t_batch": tb, "peak": peak, "t_single": t1, "cos1": cos1.tolist(), "cos2": cos2.tolist()}
