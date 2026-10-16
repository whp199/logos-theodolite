# E6_tool_audit_13_quiet_readout -- do J++ / JLens READOUTS depend on the quiet residual dimensions found in E6_12?
# For all 132 E6 prompts (single BOS, last position), compare gender pron readouts from full h vs h with quiet-dim coordinates
# replaced by their dataset mean ("quiet-clamped"). Quiet dims = per-layer dims with token-std below the 10th percentile.
# PRE-STATED PREDICTIONS: (a) quiet dims contribute a large share of ||h J^T|| at L35+ for J++ (>30%) despite carrying <0.1% of
# ||h||^2; (b) persona effects (B woman vs man) survive clamping, but absolute values/background shift substantially.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
C = build_conditions()
LAYS = list(range(20, 59, 2))
names, allids = [], []
for cname, ps in C.items():
    for p in ps:
        names.append(cname); allids.append(lab.ids(p))
# per-dim std/mean over all non-BOS tokens of a 24-prompt reference set
ref = C["A_default"] + C["B_gemma"]
acc = {l: [] for l in LAYS}
for p in ref:
    H = lab.forward_resid([lab.ids(p)], layers=LAYS)[0]
    for i, l in enumerate(LAYS):
        acc[l].append(H[i, 1:])
stat = {}
for l in LAYS:
    X = torch.cat(acc[l], 0)
    sd = X.std(0); mu = X.mean(0)
    stat[l] = (sd.to(lab.dev), mu.to(lab.dev), (sd < torch.quantile(sd, 0.10)).to(lab.dev))
del acc
Hs = []
for i in range(0, len(allids), 24):
    ch = allids[i:i + 24]
    Hs.append(lab.forward_resid(ch, layers=LAYS, positions=[[len(x) - 1] for x in ch])[:, :, 0])
H = torch.cat(Hs, 0)  # [N, nL, D]
out = {"layers": LAYS, "names": names, "jpp": {}, "jlens": {}}
share = {"jpp": [], "jlens": []}
for kind in ("jpp", "jlens"):
    full, clamp = [], []
    for i, l in enumerate(LAYS):
        sd, mu, q = stat[l]
        h = H[:, i].to(lab.dev)
        hc = torch.where(q.unsqueeze(0), mu.unsqueeze(0), h)
        hq = (h - hc)  # quiet-dim deviation only
        hJ = lab.lens(h, l, kind); hJc = lab.lens(hc, l, kind); hJq = lab.lens(hq, l, kind)
        share[kind].append({"layer": l, "quiet_norm2_share_of_h": float((hq.float().pow(2).sum(-1) / h.float().pow(2).sum(-1)).mean()),
                            "quiet_dev_share_of_hJ_norm": float((hJq.float().norm(dim=-1) / hJ.float().norm(dim=-1)).mean()),
                            "cos_hJ_full_vs_clamped": float(F.cosine_similarity(hJ.float(), hJc.float(), dim=-1).mean())})
        full.append(lab.gender_scores(hJ)["pron"].float().cpu()); clamp.append(lab.gender_scores(hJc)["pron"].float().cpu())
    out[kind] = {"full": torch.stack(full, 1).tolist(), "clamped": torch.stack(clamp, 1).tolist()}
out["share"] = share
for kind in share:
    print(kind, [(s["layer"], round(s["quiet_norm2_share_of_h"], 5), round(s["quiet_dev_share_of_hJ_norm"], 3), round(s["cos_hJ_full_vs_clamped"], 3)) for s in share[kind]])
import json as _j
_j.dump(out, open("./experiments/results/E6_tool_audit_13_quiet_readout.partial.json", "w"))
result = {"ok": True}
