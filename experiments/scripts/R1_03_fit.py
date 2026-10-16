# R1_IT_LENS 03: directional IT-lens fit + stochastic full-matrix accumulation (resumable; one half per job).
# For every fit position: one K-copy pass with 128 fixed final-space directions + 32 random probes r = fnw*z (z~N(0,I)),
# i.e. isotropic probes in the final *normalised* coordinates y = fnw*h59 that the unembedding actually reads.
#  - directional: sums of exact same-position VJPs g_l(d) per (class, half) -> gbar_l(d) per class; split-half by passage.
#  - full matrix:  A[l] += sum_k g_l(r_k) z_k^T  ->  E[A]/n = L_y[l]^T, where L_y = diag(fnw) E[J_l] (IT lens in y-coords).
# PRE-STATED PREDICTIONS (before any fit data):
#  F1 split-half reliability of gbar_l(d) (pooled) >= 0.9 at L30-58 for token-pair directions, lower (<0.6) at L0-15.
#  F2 IT gbar vs J++ pullback: low raw cosine at L40-58 (0.2-0.5, as E3/V6), rising to >=0.8 outside the q10 quiet set.
#  F3 if base->IT transfer is the cause, prose-fitted gbar disagrees with J++ inside the quiet set as much as chat gbar does.
import json, time, os
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/R1_vjplib.py").read())
t0 = time.time()
JOBS = json.load(open(R1_RES + "R1_03_jobs.json"))
JOB = JOBS[JOBI]
HALF = JOB["half"]
T_BUDGET = globals().get("T_BUDGET", 200)
P = json.load(open(R1_RES + "R1_02_positions.json"))
DD = torch.load(R1_RES + "R1_02_dirs.pt")
dev = lab.dev
DIRS = DD["dirs"].to(dev)
PBJ = DD["pb_jpp"]; PBL = DD["pb_jlens"]  # CPU (GPU memory is tight at 192 copies)
v6 = torch.load(R1_RES + "V6_quiet_dims_stats.pt", weights_only=False)
std = v6["std_all"].float()
q10 = (std <= torch.quantile(std, 0.1, dim=1, keepdim=True)).float()  # [60, D] CPU
fnw = lab.fnw.float()
CLASSES = ["prose_enc", "prose_doc", "user", "asst", "self"]
KEY = [0, 1, 2, 3, 10, 11]
FM = R1_FM_LAYERS
NR = R1_NRAND
CK = R1_RES + "R1_03_fit_ckpt.pt"
st = STORE.get("R1_fit")
if st is None:
    if os.path.exists(CK):
        st = torch.load(CK)
        print("loaded ckpt from disk, done", len(st["done"]), flush=True)
    else:
        st = {"done": [], "S": torch.zeros(10, R1_NDIR, 60, 5376), "N": torch.zeros(10),
              "A": torch.zeros(2, len(FM), 5376, 5376), "NA": torch.zeros(2)}
    STORE["R1_fit"] = st
done = set(st["done"])
todo = [i for i in JOB["passages"] if P[i]["pid"] not in done]
S_loc = torch.zeros(10, R1_NDIR, 60, 5376)
N_loc = torch.zeros(10)
A_loc = torch.zeros(len(FM), 5376, 5376)  # CPU fp32
NA_loc = 0
recs, Hs, PP, fin = [], [], {}, []
for i in todo:
    if time.time() - t0 > T_BUDGET:
        break
    p = P[i]; ids = p["ids"]
    assert p["half"] == HALF
    pp = torch.zeros(len(KEY), 60, 5376)
    for pos, cls in p["pos"]:
        gen = torch.Generator().manual_seed(1000003 * i + pos)
        Z = torch.randn(NR, 5376, generator=gen).to(dev)
        R = torch.cat([DIRS, Z * fnw], 0)
        G, H = r1_vjp_robust(lab, ids, pos, R)
        Gd = G[:R1_NDIR]; Gr = G[R1_NDIR:]
        Gc = Gd.cpu(); Grc = Gr.cpu(); Zc = Z.cpu()
        del G, Gd, Gr
        gn = Gc.norm(dim=-1)
        rec = {"i": i, "pid": p["pid"], "src": p["src"], "cls": cls, "pos": pos, "half": HALF,
               "tok": lab.strs([ids[pos]])[0],
               "gn": gn.cpu(),
               "cj": F.cosine_similarity(Gc, PBJ, dim=-1).half(),
               "cl": F.cosine_similarity(Gc, PBL, dim=-1).half(),
               "qs": ((Gc.pow(2) * q10).sum(-1) / Gc.pow(2).sum(-1).clamp_min(1e-30)).half()}
        recs.append(rec); Hs.append(H.half().cpu())
        si = CLASSES.index(cls) * 2 + HALF
        S_loc[si] += Gc; N_loc[si] += 1
        pp += Gc[KEY]
        for j, l in enumerate(FM):
            A_loc[j].addmm_(Grc[:, l, :].T, Zc)
        NA_loc += NR
        del Gc, Grc
    PP[p["pid"]] = {"sum": pp, "n": len(p["pos"]), "half": HALF, "src": p["src"]}
    fin.append(p["pid"])
    print("passage", p["pid"], len(recs), round(time.time() - t0, 1), flush=True)
# merge + checkpoint
st["S"] += S_loc; st["N"] += N_loc
st["A"][HALF] += A_loc; st["NA"][HALF] += NA_loc
st["done"] = sorted(set(st["done"]) | set(fin))
del A_loc
torch.cuda.empty_cache()
tag = f"job{JOBI:02d}_{int(time.time())}"
torch.save({"recs": recs, "H": torch.stack(Hs, 0) if Hs else None, "pp": PP}, R1_RES + f"R1_03_fit_{tag}.pt")
torch.save(st, CK + ".tmp"); os.replace(CK + ".tmp", CK)
left = [P[i]["pid"] for i in JOB["passages"] if P[i]["pid"] not in set(st["done"])]
print("finished", len(fin), "positions", len(recs), "left in job", len(left), "total done", len(st["done"]), round(time.time() - t0, 1))
result = {"job": JOBI, "fin": fin, "left": left, "n_pos": len(recs), "t": time.time() - t0}
