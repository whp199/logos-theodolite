# E4_causal_16: diagnostic for the pair-matched clamp at L40-54 with RANDOM directions (control removed 58-75% of P1 gap).
# Tests: (1) self-clamp (target = own clean coefficient, should be a no-op) at L40-54; (2) pair-midpoint clamp per layer;
# (3) magnitude of |c_F - c_M| per position along rand0 vs u_self; (4) whether batch composition changes clean coefficients.
exec(open("./experiments/scripts/E4_causal_common.py").read())
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); DIRS = DC["dirs"]
P = [p for p in causal_prompts(lab) if p["set"] == "P1"][:12]  # 6 pairs
ids_l = [p["ids"] for p in P]
def coefs(dname, ids_list):
    rec = {}
    U = DIRS[dname].to(lab.dev)
    def red(kind, l, h):
        rec[l] = (h.float() @ U[l]).cpu()
    lg = batched_hook_forward(lab, ids_list, red)
    return rec, lg
def pf(lg):
    m = pron_logodds_from_logits(lab, lg)
    return [round(float(x), 4) for x in m["p_f"]], [round(float(x), 4) for x in m["p_m"]]
rec, lg0 = coefs("rand0", ids_l)
rec_solo, _ = coefs("rand0", [ids_l[0]])
print("batch-composition check L45 pos 1..5: batched", rec[45][0, 1:6].tolist(), "solo", rec_solo[45][0, 1:6].tolist())
print("base p_f/p_m", pf(lg0))
for l in (40, 47, 54):
    d = (rec[l][0::2] - rec[l][1::2]).abs()  # F - M per pair, [6, T]
    print(f"L{l} rand0 |cF-cM| max per pair", [round(float(x), 2) for x in d.max(1).values], "argmax pos", d.argmax(1).tolist(), " |c| at pos1", [round(float(x), 1) for x in rec[l][::2, 1]])
recu, _ = coefs("u_self", ids_l)
for l in (40, 47):
    d = (recu[l][0::2] - recu[l][1::2]).abs()
    print(f"L{l} u_self |cF-cM| max per pair", [round(float(x), 2) for x in d.max(1).values], "mean", round(float(d[:, 1:].mean()), 3))
cur = {}
def install(dname, layers, mode):
    for l in layers:
        u = DIRS[dname][l].to(lab.dev, torch.float32)
        def fn(h, u=u, l=l):
            hf = h.float(); c = hf @ u
            tgt = cur[l].to(hf.device)
            delta = torch.where(torch.isnan(tgt), torch.zeros_like(c), tgt - c)
            return (hf + delta.unsqueeze(-1) * u).to(h.dtype)
        lab.add_resid_hook(l, fn)
T = max(len(x) for x in ids_l)
def targets(rec, layers, mode):
    tg = {}
    for l in layers:
        t = torch.full((len(ids_l), T), float("nan"))
        for k in range(len(ids_l)):
            q = k + 1 if k % 2 == 0 else k - 1
            Tk = len(ids_l[k])
            if mode == "self":
                t[k, 1:Tk] = rec[l][k, 1:Tk]
            else:
                t[k, 1:Tk] = 0.5 * (rec[l][k, 1:Tk] + rec[l][q, 1:Tk])
        tg[l] = t
    return tg
for mode in ("self", "pair"):
    for layers in (list(range(40, 55)), [40], [47], list(range(25, 40))):
        cur.clear(); cur.update(targets(rec, layers, mode))
        lab.clear_hooks(); install("rand0", layers, mode)
        try:
            lg = batched_hook_forward(lab, ids_l)
        finally:
            lab.clear_hooks()
        a, b = pf(lg)
        print(mode, f"L{layers[0]}-{layers[-1]}", "p_f", a[:6], "p_m", b[:6])
result = {"ok": True}
