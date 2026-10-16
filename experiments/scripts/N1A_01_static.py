# N1A_01_static -- static output-side pole labels for the 192 sampled neurons (+ the tool's 'gender' poles),
# exactly as the tool computes them, plus lens priors for z-scoring.
#   jpp        = unembed((W_down[:,i]*w_post) @ J++[l].T)      (tool's J-VINDEX label; '-' pole = negation)
#   jlens      = unembed((W_down[:,i]*w_post) @ JLens[l].T)
#   naive      = unembed(W_down[:,i]*w_post)                     (tool's naive logit-lens label)
#   jpp_noscale= unembed(W_down[:,i] @ J++[l].T)                 (no post_ffn_norm weight)
# Priors per layer & lens: (a) 'iso': 1024 isotropic Gaussian vectors in layer-l residual space -> per-token mu, sd
#                          (b) 'neu': 512 random OTHER neurons of the layer (both signs => mu=0) -> per-token sd.
# PRE-STATED PREDICTIONS:
#  S1 my implementation reproduces vault.inspect_bipolar_slot top-8 J-VINDEX / naive tokens exactly.
#  S2 raw jpp top tokens of many poles are prior-dominated: the most frequent jpp top-1 tokens across the 192 neurons
#     overlap the isotropic-prior top-1 tokens (HTML closing tags etc.), while naive top-1 tokens are diverse.
#  S3 single-neuron jpp label distributions are near-flat (top-10 softmax mass << prior's 0.4-0.96).
#  S4 the 12 tool 'gender' poles (query_nearest_edges she/he, top_k=2, L30/40/50) have she/he in their jpp top-10
#     by construction (they were selected by cosine to the pulled-back W_U[she/he]).
exec(open("./experiments/scripts/N1A_common.py").read())
from collections import Counter
from backend.larql_vault import LarqlVaultEngine
torch.manual_seed(7)
res = {"prior_top1": {}, "static_top1": {}, "top10_mass": {}}
KINDS = ["jpp", "jlens", "naive"]
D = 5376
t_start = time.time()
dec = lambda t: lab.tok.decode([int(t)])

def static_block(l, neurons):
    v = n1a_static_vecs(l, neurons, scale=True)
    v0 = n1a_static_vecs(l, neurons, scale=False)
    out = {"neurons": list(neurons)}
    for k in KINDS:
        out[k] = n1a_unembed_fp32(n1a_lens_vec(v, l, k)).cpu()
    out["jpp_noscale"] = n1a_unembed_fp32(n1a_lens_vec(v0, l, "jpp")).cpu()
    return out

def top10_mass(lg):
    p = F.softmax(lg.float(), -1)
    return p.topk(10, -1).values.sum(-1)

for l in N1A_LAYERS:
    st = static_block(l, N1A_NEURONS[l])
    torch.save(st, f"{N1A_CACHE}/static_L{l}.pt")
    pri = {"iso": {}, "neu": {}}
    # isotropic prior
    for k in KINDS:
        s1 = torch.zeros(lab.WU.shape[0], dtype=torch.float64, device=lab.dev)
        s2 = torch.zeros_like(s1)
        n = 0
        top1 = Counter()
        masses = []
        for _ in range(8):
            G = torch.randn(128, D, device=lab.dev).to(torch.bfloat16)
            lg = n1a_unembed_fp32(n1a_lens_vec(G, l, k))
            s1 += lg.sum(0).double(); s2 += (lg.double() ** 2).sum(0); n += lg.shape[0]
            top1.update(dec(t) for t in lg.argmax(-1).tolist())
            masses.append(top10_mass(lg).cpu())
            del lg
        mu = s1 / n
        sd = (s2 / n - mu ** 2).clamp(min=1e-12).sqrt()
        pri["iso"][k] = (mu.float().cpu(), sd.float().cpu())
        res["prior_top1"][f"{k}_L{l}"] = top1.most_common(6)
        res["top10_mass"][f"prior_{k}_L{l}"] = float(torch.cat(masses).median())
    # neuron prior: 512 random other neurons
    others = [i for i in torch.randperm(21504).tolist() if i not in set(N1A_NEURONS[l])][:512]
    for k in KINDS:
        s2 = torch.zeros(lab.WU.shape[0], dtype=torch.float64, device=lab.dev)
        for c in range(0, 512, 128):
            v = n1a_static_vecs(l, others[c:c + 128], scale=True)
            lg = n1a_unembed_fp32(n1a_lens_vec(v, l, k))
            s2 += (lg.double() ** 2).sum(0)
            del lg, v
        pri["neu"][k] = (s2 / 512).sqrt().float().cpu()
    torch.save(pri, f"{N1A_CACHE}/prior_L{l}.pt")
    for k in KINDS + ["jpp_noscale"]:
        res["static_top1"][f"{k}_L{l}"] = Counter(dec(t) for t in st[k].argmax(-1).tolist()).most_common(6)
        res["top10_mass"][f"pole_{k}_L{l}"] = float(top10_mass(st[k]).median())
    print(f"L{l} done {time.time() - t_start:.1f}s | jpp top1 {res['static_top1'][f'jpp_L{l}'][:3]} | "
          f"prior {res['prior_top1'][f'jpp_L{l}'][:3]} | mass pole {res['top10_mass'][f'pole_jpp_L{l}']:.2e} "
          f"prior {res['top10_mass'][f'prior_jpp_L{l}']:.2f}", flush=True)
    torch.cuda.empty_cache()

# S1: tool reproduction check
vault = LarqlVaultEngine(rt)
chk = []
for l, n in [(30, N1A_NEURONS[30][0]), (50, N1A_NEURONS[50][3]), (10, N1A_NEURONS[10][5])]:
    tool = vault.inspect_bipolar_slot(l, n, 1.0, top_k_tokens=8)
    st = torch.load(f"{N1A_CACHE}/static_L{l}.pt")
    r = st["neurons"].index(n)
    mine_j = [dec(t) for t in st["jpp"][r].topk(8).indices.tolist()]
    mine_n = [dec(t) for t in st["naive"][r].topk(8).indices.tolist()]
    mine_jm = [dec(t) for t in (-st["jpp"][r]).topk(8).indices.tolist()]
    tj = [d["token"] for d in tool["north_pole"]["j_vindex_tokens"]]
    tn = [d["token"] for d in tool["north_pole"]["naive_tokens"]]
    tjm = [d["token"] for d in tool["south_pole"]["j_vindex_tokens"]]
    chk.append({"l": l, "n": n, "tool_jpp": tj, "mine_jpp": mine_j, "tool_naive": tn, "mine_naive": mine_n,
                "tool_jpp_minus": tjm, "mine_jpp_minus": mine_jm,
                "overlap_jpp": len(set(tj) & set(mine_j)), "overlap_naive": len(set(tn) & set(mine_n)),
                "overlap_jpp_minus": len(set(tjm) & set(mine_jm))})
    print("check", chk[-1], flush=True)
res["tool_check"] = chk

# Bonus: tool 'gender' poles
gender = []
for concept in ("she", "he"):
    for l in (30, 40, 50):
        q = vault.query_nearest_edges(concept, layer_idx=l, top_k=2)
        for row in q["results"]:
            gender.append({"concept": concept, "layer": l, "neuron": int(row["neuron_idx"]),
                           "sign": int(row["polarity_num"]), "cos": row["cosine_similarity"],
                           "tool_jv": [t["token"] for t in row["j_vindex_top_tokens"]],
                           "tool_nv": [t["token"] for t in row["naive_lens_top_tokens"]]})
        torch.cuda.empty_cache()
gst = {}
for l in (30, 40, 50):
    ns = [g["neuron"] for g in gender if g["layer"] == l]
    gst[l] = static_block(l, ns)
torch.save({"poles": gender, "static": gst}, f"{N1A_CACHE}/gender_static.pt")
res["gender_poles"] = gender
print("gender", [(g["concept"], g["layer"], g["neuron"], g["sign"], g["cos"]) for g in gender])
res["elapsed"] = time.time() - t_start
result = res
