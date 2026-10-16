# R1_IT_LENS 08: second-moment matrix of the final *normalised* state y = fnw*h59/rms(h59) (what W_U reads), over the fit
# corpus (all positions >= 1) and, separately, over the E3 evaluation stimuli. Used to choose the probe subspace for the
# full-matrix IT lens (Phase B) and to check that subspace on held-out text.
import json, time
exec(open("./experiments/scripts/R1_common.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
P = json.load(open(R1_RES + "R1_02_positions.json"))
fnw = lab.fnw.float()
def acc_y(seqs):
    M = torch.zeros(5376, 5376, device=lab.dev, dtype=torch.float64); s = torch.zeros(5376, device=lab.dev, dtype=torch.float64); n = 0
    for b in range(0, len(seqs), 8):
        bt = seqs[b:b + 8]
        H = lab.forward_resid(bt, layers=[59])          # [B,1,T,D] (equal lengths not required? -> use positions)
        for k, ids in enumerate(bt):
            h = H[k, 0, 1:len(ids)].to(lab.dev)
            y = (h * fnw / h.pow(2).mean(-1, keepdim=True).sqrt()).double()
            M += y.T @ y; s += y.sum(0); n += y.shape[0]
    return M.cpu(), s.cpu(), n
# forward_resid with positions=None needs equal lengths -> pad by giving explicit positions instead
def acc_y2(seqs):
    M = torch.zeros(5376, 5376, device=lab.dev, dtype=torch.float64); s = torch.zeros(5376, device=lab.dev, dtype=torch.float64); n = 0
    for ids in seqs:
        h = lab.forward_resid([ids], layers=[59])[0, 0, 1:].to(lab.dev)
        y = (h * fnw / h.pow(2).mean(-1, keepdim=True).sqrt()).double()
        M += y.T @ y; s += y.sum(0); n += y.shape[0]
    return M.cpu(), s.cpu(), n
Mf, sf, nf = acc_y2([p["ids"] for p in P])
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
ev = [lab.ids(t) for _, t in PROSE] + [lab.ids(lab.chat(c["q"], assistant=c["a"])) for c in chats]
Me, se, ne = acc_y2(ev)
torch.save({"M_fit": Mf, "s_fit": sf, "n_fit": nf, "M_eval": Me, "s_eval": se, "n_eval": ne}, R1_RES + "R1_08_ystats.pt")
result = {"n_fit": nf, "n_eval": ne, "t": time.time() - t0}
