# R1_IT_LENS 09: Phase B, stochastic full-matrix IT lens at R1_FM_LAYERS.
# Probes r = fnw*z, z ~ N(0, I_D) (isotropic in the normalised final coordinates y = fnw*h59 that W_U reads), 4 positions x 40
# probes per pass (r1_vjp_multi). A[l] += sum g_l(r) z^T  ->  E[A]/n = L_y[l]^T = (diag(fnw) E[J_l])^T, D_in x D_out.
# Positions: a NEW stratified sample (seed 909) inside the fit passages; halves by passage parity (as R1_03).
# Unbiased; noise variance ~ D/n per unit of signal, so this estimate is usable mainly at L40-58 (high Jacobian coherence).
import json, time, os, random
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/R1_vjplib.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
T_BUDGET = globals().get("T_BUDGET", 205)
NPOS, NPR = 4, 40
P = json.load(open(R1_RES + "R1_02_positions.json"))
fnw = lab.fnw.float()
FM = R1_FM_LAYERS
CK = R1_RES + "R1_09_fm_ckpt.pt"
st = STORE.get("R1_fm")
if st is None:
    if os.path.exists(CK):
        st = torch.load(CK)
    else:
        st = {"done": [], "A": torch.zeros(2, len(FM), 5376, 5376), "NA": torch.zeros(2), "npos": 0}
    STORE["R1_fm"] = st
done = set(st["done"])
# unit of work = (passage index, chunk of 4 new positions)
rng = random.Random(909)
units = []
for i, p in enumerate(P):
    ids = p["ids"]; T = min(len(ids), 990)
    if p["src"].startswith("chat"):
        cats = e3_chat_positions(lab, ids, SELF_TOKS)
        cand = [q for q in cats["asst"][1:] + cats["user"] if q < T]
    else:
        cand = list(range(8, T))
    ps = sorted(rng.sample(cand, min(24, len(cand))))
    for c in range(0, len(ps) - NPOS + 1, NPOS):
        units.append((i, tuple(ps[c:c + NPOS])))
order = list(range(len(units)))
random.Random(1).shuffle(order)
A_loc = {0: None, 1: None}
NA_loc = [0, 0]
fin, npos = [], 0
for u in order:
    key = f"{units[u][0]}:{units[u][1]}"
    if key in done:
        continue
    if time.time() - t0 > T_BUDGET:
        break
    i, ps = units[u]
    half = P[i]["half"]
    gen = torch.Generator().manual_seed(7919 * u + 13)
    Z = torch.randn(NPOS * NPR, 5376, generator=gen)
    R = Z.to(lab.dev) * fnw
    poss = [q for q in ps for _ in range(NPR)]
    try:
        G = r1_vjp_multi(lab, P[i]["ids"], poss, R)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); time.sleep(5)
        continue
    Gc = G.cpu(); del G
    if A_loc[half] is None:
        A_loc[half] = torch.zeros(len(FM), 5376, 5376)
    for j, l in enumerate(FM):
        A_loc[half][j].addmm_(Gc[:, l, :].T, Z)
    NA_loc[half] += NPOS * NPR
    npos += NPOS
    fin.append(key)
    del Gc
for h in (0, 1):
    if A_loc[h] is not None:
        st["A"][h] += A_loc[h]
    st["NA"][h] += NA_loc[h]
st["npos"] += npos
st["done"] = sorted(set(st["done"]) | set(fin))
torch.save(st, CK + ".tmp"); os.replace(CK + ".tmp", CK)
print("units done this job", len(fin), "total", len(st["done"]), "of", len(units), "NA", st["NA"].tolist(), round(time.time() - t0, 1))
result = {"fin": len(fin), "total_done": len(st["done"]), "n_units": len(units), "NA": st["NA"].tolist(), "t": time.time() - t0}
