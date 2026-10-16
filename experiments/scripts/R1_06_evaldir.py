# R1_IT_LENS 06: held-out directional fidelity on the E3 stimuli (24 encyclopedic paragraphs + 24 default chats; disjoint
# from the R1 fit corpus). For each of the 128 directions d and every layer, how well does each lens readout of h_l predict
# the model's real final readout t_d = <h_59, d>/rms(h_59) (= the real pre-softcap logit difference) across positions?
# Readouts: IT directional <h_l, gbar_l(d)> (mix / chat / prose fits), J++ and JLens (linear <h_l, d J_l> and the tool's
# normalised <h_l J_l^T, d>/rms(h_l J_l^T)), naive (<h_l,d> and <h_l,d>/rms(h_l)).
# Saves per (passage, class) sufficient statistics for Pearson r (bootstrap over passages later) and per-position values for
# 8 key directions. PREDICTION: IT readouts beat J++ in r at assistant/self positions at L20-49 (where E3 found J++'s chat gain
# collapses), and roughly tie with J++ on prose.
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
dev = lab.dev
GB = torch.load(R1_RES + "R1_04_gbar.pt")
DD = torch.load(R1_RES + "R1_02_dirs.pt")
DIRS = DD["dirs"].to(dev)
IT = {v: GB[v].to(dev) for v in ("mix", "chat", "prose")}       # [128,60,D]
PBJ = DD["pb_jpp"].to(dev); PBL = DD["pb_jlens"].to(dev)
RD = ["it_mix", "it_chat", "it_prose", "jpp_lin", "jpp_norm", "jl_lin", "jl_norm", "naive_lin", "naive_norm", "it_mix_loud", "jpp_lin_loud"]
PAIRS = [("it_mix", "jpp_lin"), ("it_mix_loud", "jpp_lin_loud"), ("it_mix", "it_mix_loud"), ("jpp_lin", "jpp_lin_loud"), ("it_chat", "it_prose")]
v6 = torch.load(R1_RES + "V6_quiet_dims_stats.pt", weights_only=False)
_std = v6["std_all"].float()
LOUD = (_std > torch.quantile(_std, 0.1, dim=1, keepdim=True)).float().to(dev)     # [60,D]
KEYD = [0, 1, 2, 3, 4, 5, 10, 11]
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
seqs = []
for i, (tag, txt) in enumerate(PROSE):
    ids = lab.ids(txt)
    seqs.append((f"prose{i:02d}", ids, {"prose": list(range(4, len(ids)))}))
for i, c in enumerate(chats):
    ids = lab.ids(lab.chat(c["q"], assistant=c["a"]))
    cats = e3_chat_positions(lab, ids, SELF_TOKS)
    seqs.append((f"chat{i:02d}", ids, {"asst": cats["nonself_asst"], "self": cats["self"], "user": cats["user"]}))
stats, perpos, meta = {}, [], []
for name, ids, regions in seqs:
    H = lab.forward_resid([ids])[0]          # [60,T,D] float32 CPU
    allp = [p for cls, ps in regions.items() for p in ps]
    Hs = H[:, torch.tensor(allp)].to(dev)    # [60,P,D]
    del H
    h59 = Hs[59]
    Y = (h59 @ DIRS.T) / h59.pow(2).mean(-1, keepdim=True).sqrt()     # [P,128]
    X = {r: torch.empty(60, len(allp), 128, device=dev) for r in RD}
    for l in range(60):
        h = Hs[l]
        for v in ("mix", "chat", "prose"):
            X[f"it_{v}"][l] = h @ IT[v][:, l].T
        X["jpp_lin"][l] = h @ PBJ[:, l].T
        X["jl_lin"][l] = h @ PBL[:, l].T
        X["it_mix_loud"][l] = (h * LOUD[l]) @ IT["mix"][:, l].T
        X["jpp_lin_loud"][l] = (h * LOUD[l]) @ PBJ[:, l].T
        X["naive_lin"][l] = h @ DIRS.T
        X["naive_norm"][l] = X["naive_lin"][l] / h.pow(2).mean(-1, keepdim=True).sqrt()
        if l < 59:
            hJ = (h.to(torch.bfloat16) @ lab.J[l].T).float()
            X["jpp_norm"][l] = (hJ @ DIRS.T) / hJ.pow(2).mean(-1, keepdim=True).sqrt()
            hL = (h.to(torch.bfloat16) @ lab.JL[l].to(dev, non_blocking=True).T).float()
            X["jl_norm"][l] = (hL @ DIRS.T) / hL.pow(2).mean(-1, keepdim=True).sqrt()
        else:
            X["jpp_norm"][l] = X["naive_norm"][l]; X["jl_norm"][l] = X["naive_norm"][l]
    off = 0
    for cls, ps in regions.items():
        n = len(ps)
        if n == 0:
            continue
        sl = slice(off, off + n); off += n
        Yd = Y[sl].double()
        st = {"n": n, "sy": Yd.sum(0).cpu(), "syy": (Yd * Yd).sum(0).cpu()}
        for r in RD:
            Xd = X[r][:, sl].double()
            st[r] = torch.stack([Xd.sum(1), (Xd * Xd).sum(1), (Xd * Yd[None]).sum(1)], 0).cpu()   # [3,60,128]
        for a, b in PAIRS:
            st[f"pair:{a}:{b}"] = (X[a][:, sl].double() * X[b][:, sl].double()).sum(1).cpu()      # [60,128]
        stats[(name, cls)] = st
        perpos.append({"name": name, "cls": cls, "pos": ps, "Y": Y[sl][:, KEYD].half().cpu(),
                       "X": torch.stack([X[r][:, sl][:, :, KEYD] for r in RD], 0).half().cpu()})   # [11,60,P,8]
    del X, Hs
    print(name, round(time.time() - t0, 1), flush=True)
torch.save({"stats": {f"{a}|{b}": v for (a, b), v in stats.items()}, "perpos": perpos, "readouts": RD, "keyd": KEYD, "pairs": PAIRS},
           R1_RES + "R1_06_evaldir.pt")
result = {"n_seqs": len(seqs), "t": time.time() - t0}
