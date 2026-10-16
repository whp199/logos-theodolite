E4_MODE = "const"
# E4_causal_09: causal ablation of gender directions (mean-replacement to the woman/man midpoint) over layer blocks.
# PRE-STATED PREDICTIONS:
#  C1 Ablating u_self over L10-24 or L25-39 removes a substantial fraction (30-70%) of the persona third-person pronoun gap
#     (P2: neutral-name personas 'Alex ... Every day,'), less (10-40%) for kin prompts (P1 'My sister ... Every day,') where
#     gender is lexically explicit and can be re-read after the block.
#  C2 Ablating in L40-54 (where u_self is orthogonal to prose/readout directions) removes little of either gap.
#  C3 Ablating the J++ pullback direction (cos ~0.0-0.24 with u) removes little; random directions remove ~0.
#  C4 AI-self prompts (P3/P4) move by less than 1 nat; no evidence that a female-coded default is held up by u.
# Modes: 'const' = set coefficient along unit u_l to a constant midpoint m_l (mean coefficient over non-BOS positions of the
#        balanced F/M P1+P2 prompts, clean run); 'pair' = set to the position-wise average of the paired F and M clean runs.
import sys
exec(open("./experiments/scripts/E4_causal_common.py").read())
MODE = globals().get("E4_MODE", "const")
t0 = time.time()
DC = torch.load(f"{E4_CACHE}/dirs_causal.pt"); DIRS = DC["dirs"]
DN = ["u_self", "u_cont_A", "u_cont_B", "u_aiexp", "pb_jpp", "rand0", "rand1"]
Ust = torch.stack([DIRS[d] for d in DN], 1).to(lab.dev)  # [60, nd, D]
BLOCKS = {"L10-24": list(range(10, 25)), "L25-39": list(range(25, 40)), "L40-54": list(range(40, 55))}
P = causal_prompts(lab)
N = len(P)
BS = 24
batches = [list(range(b, min(b + BS, N))) for b in range(0, N, BS)]
if MODE == "pair":
    keep = [j for j, p in enumerate(P) if p["set"] in ("P1", "P2")]
    batches = [keep[b:b + BS] for b in range(0, len(keep), BS)]

def run(batch_idx, extra_hooks=None):
    ids_l = [P[j]["ids"] for j in batch_idx]
    return batched_hook_forward(lab, ids_l)

# ---- clean pass with coefficient recording
coef = [None] * N
base_lg = torch.zeros(N, lab.WU.shape[0])
for bi in [list(range(b, min(b + BS, N))) for b in range(0, N, BS)]:
    ids_l = [P[j]["ids"] for j in bi]
    rec = {}
    def red(kind, l, h):
        rec[l] = (h.float() @ Ust[l].T).cpu()  # [B,T,nd]
    lg = batched_hook_forward(lab, ids_l, red)
    base_lg[bi] = lg.cpu()
    for k, j in enumerate(bi):
        T = len(P[j]["ids"])
        coef[j] = torch.stack([rec[l][k, :T] for l in range(60)], 0)  # [60,T,nd]
base_m = metrics_from_logits(lab, base_lg.to(lab.dev), P)
# constant midpoints from balanced P1+P2 non-BOS positions
mid = torch.zeros(60, len(DN))
cnt = 0
for j, p in enumerate(P):
    if p["set"] in ("P1", "P2"):
        mid += coef[j][:, 1:].mean(1); cnt += 1
mid /= cnt
print("clean done", round(time.time() - t0, 1))

cur = {}
def install(dname, layers, mode):
    di = DN.index(dname)
    for l in layers:
        u = DIRS[dname][l].to(lab.dev, torch.float32)
        if mode == "const":
            mv = float(mid[l, di])
            def fn(h, u=u, mv=mv):
                hf = h.float(); c = hf @ u
                delta = mv - c; delta[:, 0] = 0.0
                return (hf + delta.unsqueeze(-1) * u).to(h.dtype)
        else:
            def fn(h, u=u, l=l, di=di):
                hf = h.float(); c = hf @ u
                tgt = cur["tgt"][l].to(hf.device)
                delta = torch.where(torch.isnan(tgt), torch.zeros_like(c), tgt - c)
                return (hf + delta.unsqueeze(-1) * u).to(h.dtype)
        lab.add_resid_hook(l, fn)

def pair_targets(batch_idx, dname, layers):
    di = DN.index(dname)
    T = max(len(P[j]["ids"]) for j in batch_idx)
    tg = {}
    for l in layers:
        t = torch.full((len(batch_idx), T), float("nan"))
        for k, j in enumerate(batch_idx):
            q = P[j]["partner"]
            if q is None: continue
            Tj = len(P[j]["ids"])
            t[k, 1:Tj] = 0.5 * (coef[j][l, 1:Tj, di] + coef[q][l, 1:Tj, di])
        tg[l] = t
    return tg

results = {"mode": MODE, "base": base_m, "prompts": [{k: v for k, v in p.items() if k != "ids"} for p in P],
           "mid": mid.tolist(), "dirs": DN, "conds": {}}
base_lp = F.log_softmax(base_lg.float(), -1)
for dname in DN:
    for bname, layers in BLOCKS.items():
        lgs = torch.zeros(N, base_lg.shape[1]) if MODE == "const" else None
        idx_all = []
        out_lg = {}
        for bi in batches:
            lab.clear_hooks()
            if MODE == "pair":
                cur["tgt"] = pair_targets(bi, dname, layers)
            install(dname, layers, MODE)
            try:
                lg = batched_hook_forward(lab, [P[j]["ids"] for j in bi])
            finally:
                lab.clear_hooks()
            for k, j in enumerate(bi):
                out_lg[j] = lg[k].cpu()
        js = sorted(out_lg)
        LG = torch.stack([out_lg[j] for j in js])
        mets = metrics_from_logits(lab, LG.to(lab.dev), [P[j] for j in js])
        kl = []
        for k, j in enumerate(js):
            if P[j]["set"] == "NEU":
                lp = F.log_softmax(LG[k].float(), -1)
                kl.append(float((base_lp[j].exp() * (base_lp[j] - lp)).sum()))
        results["conds"][f"{dname}|{bname}"] = {"idx": js, "m": mets, "kl_neu": kl}
        print(dname, bname, "done", round(time.time() - t0, 1))
json.dump(results, open(f"{E4_DIR}/E4_causal_09_ablate_{MODE}.json", "w"))
result = {"ok": True, "elapsed": time.time() - t0}
