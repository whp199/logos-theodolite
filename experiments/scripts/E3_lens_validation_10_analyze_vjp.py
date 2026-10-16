# E3_lens_validation (b) VJP analysis (local CPU). Loads per-position true VJPs, compares average Jacobian
# pullbacks gbar_l with J++/JLens pullbacks and the naive direction; split-half reliability by passage group.
import json, numpy as np, torch, glob
R = "./experiments/results/"
LP = __import__("os").path.expanduser("~/.cache/huggingface/") + "hub/models--neuronpedia--jacobian-lens/snapshots/e166deb5ad540932100ef803c8c9bcc2a5453694/gemma-4-31b/"
rng = np.random.default_rng(0)
meta, G = [], []
for f in sorted(glob.glob(R + "E3_lens_validation_vjp_chunk*.pt")):
    d = torch.load(f); meta += d["meta"]; G.append(d["G"])
G = torch.cat(G, 0).float()  # [n, 4, 60, D]
n = G.shape[0]
DIRS = torch.load(R + "E3_lens_validation_vjp_dirs.pt").float()  # [4, D]
DN = ["pron", "noun", "rand1", "rand2"]
print("n positions", n, "groups", len(set(m["group"] for m in meta)))
# pullbacks d @ J[l]
PB = {}
for kind, fn in (("jpp", "jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt"), ("jlens", "jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt")):
    Jd = torch.load(LP + fn, map_location="cpu", mmap=True, weights_only=True)["J"]
    PB[kind] = torch.stack([DIRS @ Jd[l].float() if l < 59 else DIRS.clone() for l in range(60)], 1)  # [4, 60, D]
    PB[kind + "T"] = torch.stack([DIRS @ Jd[l].float().T if l < 59 else DIRS.clone() for l in range(60)], 1)
    del Jd
PB["naive"] = DIRS.unsqueeze(1).expand(4, 60, -1)
torch.save(PB, R + "E3_lens_validation_pullbacks.pt")
groups = sorted(set(m["group"] for m in meta))
gidx = {g: [i for i, m in enumerate(meta) if m["group"] == g] for g in groups}
def cosv(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)
def gbar(idx, unit=False):
    X = G[idx]
    if unit: X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return X.mean(0)  # [4, 60, D]
out = {"n": n, "n_groups": len(groups), "dirs": DN}
allidx = list(range(n))
gb = gbar(allidx); gbu = gbar(allidx, unit=True)
# bootstrap over groups for cos(gbar, PB)
NB = 300
bs = {k: [] for k in PB}
for _ in range(NB):
    gs = rng.choice(len(groups), len(groups), replace=True)
    idx = [i for gi in gs for i in gidx[groups[gi]]]
    g_ = gbar(idx)
    for k in PB: bs[k].append(cosv(g_, PB[k]).numpy())
for k in PB:
    c = cosv(gb, PB[k]).numpy(); b = np.array(bs[k])
    out[f"cos_gbar_{k}"] = {"mean": c.tolist(), "lo": np.percentile(b, 2.5, 0).tolist(), "hi": np.percentile(b, 97.5, 0).tolist()}
    out[f"cos_gbarunit_{k}"] = cosv(gbu, PB[k]).numpy().tolist()
# split-half reliability by group (200 random splits)
rel = []; rel_u = []
for _ in range(200):
    perm = rng.permutation(len(groups)); A = [i for gi in perm[: len(groups) // 2] for i in gidx[groups[gi]]]
    B = [i for gi in perm[len(groups) // 2:] for i in gidx[groups[gi]]]
    rel.append(cosv(gbar(A), gbar(B)).numpy()); rel_u.append(cosv(gbar(A, True), gbar(B, True)).numpy())
r = np.array(rel).mean(0); out["splithalf"] = r.tolist(); out["splithalf_unit"] = np.array(rel_u).mean(0).tolist()
ceil = np.sqrt(np.clip(2 * r / (1 + r), 1e-9, 1)); out["ceiling_SB"] = ceil.tolist()
for k in ("jpp", "jlens", "naive"):
    out[f"disatt_cos_{k}"] = np.where(r > 0.2, np.array(out[f"cos_gbar_{k}"]["mean"]) / ceil, np.nan).tolist()  # only where split-half r>0.2
# single-instance cos (from per-position meta) and coherence |mean g| / mean |g|
for k in ("jpp", "jlens", "naive"):
    out[f"single_cos_{k}"] = np.array([m[f"cos_{k}"] for m in meta]).mean(0).tolist()  # [4, 60]
out["coherence"] = (gb.norm(dim=-1) / G.norm(dim=-1).mean(0)).numpy().tolist()
out["gbar_norm"] = gb.norm(dim=-1).numpy().tolist()
# subsets: chat vs prose, self vs nonself
def sub(pred): return [i for i, m in enumerate(meta) if pred(m)]
S = {"chat": sub(lambda m: m["src"] == "chat"), "prose": sub(lambda m: m["src"] == "prose"),
     "self": sub(lambda m: m["cat"] == "self"), "asst_nonself": sub(lambda m: m["cat"] == "asst_nonself")}
out["subset_n"] = {k: len(v) for k, v in S.items()}
for a, b in (("chat", "prose"), ("self", "asst_nonself")):
    out[f"cos_{a}_vs_{b}"] = cosv(gbar(S[a]), gbar(S[b])).numpy().tolist()
    for k in ("jpp", "jlens"):
        out[f"cos_gbar_{a}_{k}"] = cosv(gbar(S[a]), PB[k]).numpy().tolist()
        out[f"cos_gbar_{b}_{k}"] = cosv(gbar(S[b]), PB[k]).numpy().tolist()
    # within-subset split-half (by group) as a ceiling for the cross-subset cos
    for s_ in (a, b):
        gs_ = sorted(set(meta[i]["group"] for i in S[s_]))
        rr = []
        for _ in range(100):
            perm = rng.permutation(len(gs_)); hA = set(gs_[j] for j in perm[: len(gs_) // 2])
            A = [i for i in S[s_] if meta[i]["group"] in hA]; B = [i for i in S[s_] if meta[i]["group"] not in hA]
            rr.append(cosv(gbar(A), gbar(B)).numpy())
        out[f"splithalf_{s_}"] = np.array(rr).mean(0).tolist()
json.dump(out, open(R + "E3_lens_validation_b_vjp_analysis.json", "w"))
torch.save({"gbar": gb, "gbar_unit": gbu, "gbar_chat": gbar(S["chat"]), "gbar_prose": gbar(S["prose"]), "dirs": DIRS, "names": DN},
           R + "E3_lens_validation_gbar.pt")
B = {"L0-9": range(0, 10), "L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40), "L40-49": range(40, 50), "L50-58": range(50, 59)}
bm = lambda v: " ".join(f"{b}:{np.nanmean([v[l] for l in rg]) if not np.all(np.isnan([v[l] for l in rg])) else float('nan'):+.3f}" for b, rg in B.items())
for di, dn in enumerate(DN):
    print(f"===== direction {dn}")
    for k in ("jpp", "jlens", "naive", "jppT", "jlensT"):
        print(f" cos(gbar, {k:6s})      ", bm(np.array(out[f'cos_gbar_{k}']['mean'])[di]))
    for k in ("jpp", "jlens", "naive"):
        print(f" single cos(g, {k:6s})  ", bm(np.array(out[f'single_cos_{k}'])[di]))
        print(f" disatt cos({k:6s})     ", bm(np.array(out[f'disatt_cos_{k}'])[di]))
    print(" split-half            ", bm(np.array(out["splithalf"])[di]))
    print(" coherence |gbar|/E|g| ", bm(np.array(out["coherence"])[di]))
    print(" cos(gbar_chat,gbar_prose)", bm(np.array(out["cos_chat_vs_prose"])[di]), "| SH chat", bm(np.array(out["splithalf_chat"])[di]), "| SH prose", bm(np.array(out["splithalf_prose"])[di]))
    print(" cos(gbar_self,gbar_nonself)", bm(np.array(out["cos_self_vs_asst_nonself"])[di]))

# ---- subset alignment with bootstrap CIs over passage groups (chat vs prose; self vs non-self assistant)
sub_ci = {}
for sname, idx in S.items():
    gs_ = sorted(set(meta[i]["group"] for i in idx))
    g0 = gbar(idx)
    for k in ("jpp", "jlens"):
        c0 = cosv(g0, PB[k]).numpy()
        bsl = []
        for _ in range(500):
            pick = rng.choice(len(gs_), len(gs_), replace=True)
            ii = [i for gi in pick for i in idx if meta[i]["group"] == gs_[gi]]
            bsl.append(cosv(gbar(ii), PB[k]).numpy())
        bsl = np.array(bsl)
        sub_ci[f"{sname}|{k}"] = {"mean": c0.tolist(), "bands": {b: [float(c0[:, list(rg)].mean(1)[0]), float(np.percentile(bsl[:, 0, list(rg)].mean(1), 2.5)), float(np.percentile(bsl[:, 0, list(rg)].mean(1), 97.5))] for b, rg in B.items()},
                                  "bands_all_dirs": {b: c0[:, list(rg)].mean(1).tolist() for b, rg in B.items()},
                                  "bands_ci_all_dirs": {b: [np.percentile(bsl[:, :, list(rg)].mean(2), 2.5, 0).tolist(), np.percentile(bsl[:, :, list(rg)].mean(2), 97.5, 0).tolist()] for b, rg in B.items()}}
out["subset_cos_ci"] = sub_ci
json.dump(out, open(R + "E3_lens_validation_b_vjp_analysis.json", "w"))
print("\n== subset cos(gbar_subset, lens pullback) with 95% CI (direction-wise) ==")
for key, v in sub_ci.items():
    for di, dn in enumerate(DN):
        print(f"{key:18s} {dn:5s} " + " ".join(f"{b}:{v['bands_all_dirs'][b][di]:+.2f}[{v['bands_ci_all_dirs'][b][0][di]:+.2f},{v['bands_ci_all_dirs'][b][1][di]:+.2f}]" for b in B))
