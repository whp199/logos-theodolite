# R1_IT_LENS 05b: definition test. Total-derivative covectors sum_{p'>=p} J_{p',p}^T d (what a lens fitted by back-propagating
# a position-summed objective would average), true and LRP-linearised, vs J++ in quiet / loud subspaces.
# PREDICTION: cross-position terms do not reconcile IT gradients with J++ in the quiet subspace (cos in q10 stays < 0.5).
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/R1_vjplib.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
PART = globals().get("PART", 0)
P = json.load(open(R1_RES + "R1_02_positions.json"))
DD = torch.load(R1_RES + "R1_02_dirs.pt")
KD = [0, 1, 10, 11, 30, 31]
DIRS = DD["dirs"][KD]
sel = []
for src, n in (("prose_enc", 6), ("prose_doc", 6), ("chat_self", 6)):
    sel += [i for i, p in enumerate(P) if p["src"] == src and p["half"] == 1][:n]
sel = sel[PART::3]
acc, cnt = {}, {}
for i in sel:
    ids = P[i]["ids"][:200]
    T = len(ids)
    if P[i]["src"].startswith("chat"):
        cats = e3_chat_positions(lab, ids, SELF_TOKS)
        regions = {"asst": [p for p in cats["asst"][1:] if p < T], "user": [p for p in cats["user"] if p < T]}
    else:
        regions = {P[i]["src"]: list(range(16, T))}
    for lrp in (False, True):
        G = r1_total_vjp(lab, ids, DIRS, lrp=lrp)    # [K,60,T,D]
        torch.cuda.empty_cache()
        for rn, ps in regions.items():
            key = ("lrp" if lrp else "true", rn)
            acc.setdefault(key, torch.zeros(len(KD), 60, 5376))
            acc[key] += G[:, :, ps, :].sum(2)
            cnt[key] = cnt.get(key, 0) + len(ps)
        del G
    print(i, T, round(time.time() - t0, 1), flush=True)
torch.save({"acc": {f"{a}|{b}": v for (a, b), v in acc.items()}, "cnt": {f"{a}|{b}": v for (a, b), v in cnt.items()},
            "dirs": KD, "sel": sel}, R1_RES + f"R1_05_total_part{PART}.pt")
result = {"n": len(sel), "t": time.time() - t0}
