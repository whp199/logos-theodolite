# V5_binding_dissociation_01: per-layer residual statistics on this study's own chat sequences, to define "quiet" dims
# (10% of dims with the smallest token-to-token std, per layer, BOS excluded) and their clamp values (dataset mean).
# Same definition as E6_tool_audit_12/13, but computed on V5 contexts (all 24 items, 2 cells x 1 variant each = 48 seqs).
exec(open("./experiments/scripts/V5_binding_dissociation_common.py").read())
import time
t0 = time.time()
ctxs = []
for it in range(24):
    for cell, k in ((("W", "M", "W"), it % 4), (("M", "W", "M"), (it + 2) % 4)):
        ctxs.append(dict(item=it, cell="".join(cell), k=k, system=system_prompt(cell, k)))
seqs = [build_seq(lab, c)[0] for c in ctxs]
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
D = 5376
S1 = torch.zeros(60, D, device=lab.dev, dtype=torch.float64)
S2 = torch.zeros(60, D, device=lab.dev, dtype=torch.float64)
NT = 0
BS = 16
for s in range(0, len(seqs), BS):
    batch = seqs[s:s + BS]
    B = len(batch); T = max(len(x) for x in batch)
    inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch):
        inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
    valid = att.clone(); valid[:, 0] = 0
    vm = valid.bool().to(lab.dev)
    hs = []
    for l in range(60):
        def mk(l):
            def hook(mod, args, out):
                h = (out[0] if isinstance(out, tuple) else out)[vm].double()
                S1[l] += h.sum(0); S2[l] += (h * h).sum(0)
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.no_grad():
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    NT += int(valid.sum())
mu = (S1 / NT); sd = (S2 / NT - mu * mu).clamp_min(0).sqrt()
q = torch.zeros(60, D, dtype=torch.bool, device=lab.dev)
for l in range(60):
    q[l] = sd[l] < torch.quantile(sd[l].float(), 0.10)
np.save(f"{CACHE}/quiet_mu.npy", mu.float().cpu().numpy())
np.save(f"{CACHE}/quiet_sd.npy", sd.float().cpu().numpy())
np.save(f"{CACHE}/quiet_mask.npy", q.cpu().numpy())
# share of J++ pullback energy of the pron direction on quiet dims (sanity vs E6: 99-100% at L35-53)
gd = lab.gender_direction("pron")
share = []
for l in range(0, 59, 3):
    pb = lab.pullback(gd, l, "jpp")
    share.append((l, round(float((pb[q[l]] ** 2).sum() / (pb ** 2).sum()), 3), round(float(sd[l][q[l]].max()), 4), round(float(sd[l].median()), 3)))
print("NT", NT, "elapsed", round(time.time() - t0, 1))
print("layer, pullback-energy share on quiet dims, max quiet sd, median sd:", share)
result = {"NT": NT, "share": share}
