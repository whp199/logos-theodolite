# R1_IT_LENS 05a: method test. Same positions, same 128 directions: exact same-position VJP vs the LRP-linearised VJP
# (frozen norms / attention patterns / GeGLU gate; see r1_lrp_patches). Does linearisation move the IT gradient toward J++
# inside the quiet subspace? PREDICTION: no (quiet-dim disagreement is not a linearisation-convention effect): cos(LRP, J++)
# in q10 stays < 0.5 at L40-58 while cos(LRP, true) in loud dims stays > 0.8.
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/R1_vjplib.py").read())
t0 = time.time()
PART = globals().get("PART", 0)
P = json.load(open(R1_RES + "R1_02_positions.json"))
DD = torch.load(R1_RES + "R1_02_dirs.pt")
DIRS = DD["dirs"].to(lab.dev)
import random
rng = random.Random(55)
sel = []
for src, n in (("prose_enc", 6), ("prose_doc", 6), ("chat_self", 6), ("chat_task", 6)):
    cand = [i for i, p in enumerate(P) if p["src"] == src and p["half"] == 0][:n]
    for i in cand:
        for pos, cls in sorted(rng.sample(P[i]["pos"], min(3, len(P[i]["pos"])))):
            sel.append((i, pos, cls))
sel = sel[PART::2]
CL = ["prose_enc", "prose_doc", "user", "asst", "self"]
acc = {m: torch.zeros(len(CL), 128, 60, 5376) for m in ("true", "lrp")}
cnt = torch.zeros(len(CL))
pcos = []
for i, pos, cls in sel:
    ids = P[i]["ids"]
    Gt, _ = r1_vjp_robust(lab, ids, pos, DIRS)
    Gt = Gt.cpu(); torch.cuda.empty_cache()
    with r1_lrp_patches(lab):
        Gl, _ = r1_vjp_robust(lab, ids, pos, DIRS)
    Gl = Gl.cpu(); torch.cuda.empty_cache()
    c = CL.index(cls)
    acc["true"][c] += Gt; acc["lrp"][c] += Gl; cnt[c] += 1
    pcos.append(F.cosine_similarity(Gt, Gl, dim=-1).half())
    print(i, pos, cls, round(time.time() - t0, 1), flush=True)
torch.save({"acc": acc, "cnt": cnt, "pcos": torch.stack(pcos), "sel": sel}, R1_RES + f"R1_05_lrpsame_part{PART}.pt")
result = {"n": len(sel), "t": time.time() - t0}
