# N1B step 3: static (weight-only) labels for the 192 neurons, computed exactly as the tool does plus variants.
# Input-side (pole-agnostic, PRIMARY):
#   G_jpp      = unembed(W_gate[i] @ J++[l].T)                      (larql_vault.inspect_bipolar_slot gate_trigger_tokens)
#   G_jpp_pre  = unembed((W_gate[i] * w_pre_ffn) @ J++[l].T)
#   G_naive    = unembed(W_gate[i])
# Input-side EXPLORATORY:
#   G_naive_pre = unembed(W_gate[i]*w_pre);  G_embed = (W_gate[i]*w_pre) . W_E[t]  (tied W_E = W_U; no final norm weight)
#   P_geglu0 (pole-specific) = s * gelu_tanh(g0_t) * u0_t with x_t = RMSnorm(W_E[t]) * w_pre (token-embedding-only residual)
#   U_jpp_pre (pole-specific) = s * unembed((W_up[i]*w_pre) @ J++[l].T)
# Output-side (pole-specific, for comparison with empirical next tokens):
#   O_jvindex = unembed((s*W_down[:,i]*w_post) @ J++[l].T),  O_naive = unembed(s*W_down[:,i]*w_post)
# Ranking uses z = (RMSnorm(v)*fnw) @ W_U.T (softcap is monotone, RMS a positive scalar -> identical top-k/ranks).
# PREDICTIONS: see N1B_02_acts.py header (P3/P4). Additionally: G_jpp top-10 tokens overlap strongly with the
# random-isotropic-vector J++ prior set, and are shared across many neurons (low label diversity).
import json, time
from collections import Counter
from backend.larql_vault import LarqlVaultEngine
R = "./experiments/results/"
t0 = time.time()
neu = json.load(open(R + "N1_neurons.json"))
LAYERS = neu["layers"]
cnt = Counter()
for line in open(R + "N1B_corpus.jsonl"):
    cnt.update(json.loads(line)["ids"])
sub = np.array(sorted(t for t, c in cnt.items() if c >= 3), dtype=np.int64)
sub_t = torch.tensor(sub, device=lab.dev)
WU = lab.WU                        # [V, D] bf16 (tied with W_E)
fnw = lab.fnw.float()
V = WU.shape[0]
TOPK = 100

def zfinal(v):
    """v [K, D] final-space vectors -> raw pre-softcap logits [K, V] float32 (ranking == tool's unembed)."""
    v = v.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-12) * fnw
    return (v.to(WU.dtype) @ WU.T).float()

def zembed(v):
    return (v.float().to(WU.dtype) @ WU.T).float()

# token-embedding-only residual, RMS-normalized (embed_scale cancels under RMSNorm)
def geglu0_scores(Wg_pre, Wu_pre, chunk=8192):
    out = torch.empty(Wg_pre.shape[0], V, device=lab.dev)
    for s0 in range(0, V, chunk):
        e = WU[s0:s0 + chunk].float()
        x = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        g = x @ Wg_pre.float().T
        u = x @ Wu_pre.float().T
        out[:, s0:s0 + chunk] = (F.gelu(g, approximate="tanh") * u).T
    return out

save = {"sub_ids": sub, "layers": np.array(LAYERS)}
store = {}
def put(name, z, both_signs):
    """z [K, V]: save top-100 ids (and bottom-100 for '-' pole if both_signs) + subset scores (fp16)."""
    vt, it = torch.topk(z, TOPK, dim=-1)
    store.setdefault(name + "__top", []).append(it.cpu().numpy().astype(np.int32))
    store.setdefault(name + "__topv", []).append(vt.cpu().numpy().astype(np.float32))
    if both_signs:
        vb, ib = torch.topk(-z, TOPK, dim=-1)
        store.setdefault(name + "__bot", []).append(ib.cpu().numpy().astype(np.int32))
    zs = z.index_select(1, sub_t)
    store.setdefault(name + "__sub", []).append(zs.cpu().numpy().astype(np.float16))

from backend.theodolite_math import unembed_gemma4
def top10_mass(vfinal):
    """Exact tool unembed (RMSNorm w/ eps floor, softcap 30) -> softmax top-10 mass per row."""
    lg = unembed_gemma4(vfinal.to(J.dtype), lab.fnw, WU).float()
    return torch.softmax(lg, -1).topk(10, dim=-1).values.sum(-1).cpu().numpy().tolist()
mass = {}
prior = {}
fid = []
eng = LarqlVaultEngine(rt)
with torch.no_grad():
    for l in LAYERS:
        idx = torch.tensor(neu["neurons"][str(l)], device=lab.dev)
        Wg = rt.slice_w_gate(l, idx)          # [24, D]
        Wu = rt.slice_w_up(l, idx)
        Wd = rt.slice_w_down(l, idx)          # [24, D] write vectors
        wpre = rt._pre_ffn_norm_weights[l].to(lab.dev).float()
        wpost = rt._post_ffn_norm_weights[l].to(lab.dev).float()
        J = lab.J[l]
        Jf = J.float()
        put("G_jpp", zfinal(Wg.float() @ Jf.T), False)
        put("G_jpp_pre", zfinal((Wg.float() * wpre) @ Jf.T), False)
        put("G_naive", zfinal(Wg.float()), False)
        put("G_naive_pre", zfinal(Wg.float() * wpre), False)
        put("G_embed", zembed(Wg.float() * wpre), False)
        put("P_geglu0", geglu0_scores(Wg.float() * wpre, Wu.float() * wpre), True)
        put("U_jpp_pre", zfinal((Wu.float() * wpre) @ Jf.T), True)
        wd = Wd.float() * wpost
        put("O_jvindex", zfinal(wd @ Jf.T), True)
        put("O_naive", zfinal(wd), True)
        # random isotropic prior through the same maps (64 vectors)
        gen = torch.Generator(device=lab.dev).manual_seed(l)
        rv = torch.randn(64, Wg.shape[1], device=lab.dev, generator=gen)
        mass[str(l)] = {"G_jpp": top10_mass(Wg.float() @ Jf.T), "G_naive": top10_mass(Wg.float()),
                        "O_jvindex_plus": top10_mass(wd @ Jf.T), "O_naive_plus": top10_mass(wd),
                        "random_jpp": top10_mass(rv @ Jf.T), "random_naive": top10_mass(rv),
                        "random_jpp_matched_norm": top10_mass((rv / rv.norm(dim=-1, keepdim=True) * Wg.float().norm(dim=-1).mean()) @ Jf.T)}
        for nm, z in (("jpp", zfinal(rv @ Jf.T)), ("naive", zfinal(rv))):
            top10 = torch.topk(z, 10, dim=-1).indices.cpu().numpy()
            c = Counter(top10.ravel().tolist())
            prior[f"{nm}_L{l}"] = [[int(t), n] for t, n in c.most_common(60)]
        # fidelity check against the tool itself for the first neuron in the layer
        i0 = int(neu["neurons"][str(l)][0])
        tool = eng.inspect_bipolar_slot(l, i0, 1.0, top_k_tokens=8)
        mine = [lab.tok.decode([int(t)]) for t in store["G_jpp__top"][-1][0][:8]]
        mineO = [lab.tok.decode([int(t)]) for t in store["O_jvindex__top"][-1][0][:8]]
        fid.append({"layer": l, "neuron": i0, "tool_gate": [d["token"] for d in tool["gate_trigger_tokens"]],
                    "mine_gate": mine, "tool_jv_plus": [d["token"] for d in tool["north_pole"]["j_vindex_tokens"]],
                    "mine_jv_plus": mineO})
        print(l, round(time.time() - t0, 1), flush=True)
for k, v in store.items():
    save[k] = np.concatenate(v, 0)
np.savez_compressed(R + "N1B_static.npz", **save)
json.dump({"prior_top10_counts": prior, "fidelity": fid, "top10_softmax_mass": mass}, open(R + "N1B_static_prior.json", "w"))
result = {"fidelity": fid, "n_sub": int(len(sub)), "elapsed": time.time() - t0}
