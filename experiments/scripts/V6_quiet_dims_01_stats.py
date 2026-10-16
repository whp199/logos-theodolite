# V6_quiet_dims_01_stats -- per-coordinate statistics of resid_post at all 60 layers over a broad chat+prose corpus
# (~9k non-BOS tokens), plus read/write gains of every residual coordinate, plus a random token sample for offline analysis.
# PRE-REGISTERED PREDICTIONS (what would refute / sharpen C7 structural claim):
#  P1 (claim): ~10% of coordinates have std << median (ratio < 0.05) at L35-53.  Refuted if the std distribution has no
#      distinct low tail (10th pct / median > 0.3) when measured on >=2000 chat+prose tokens.
#  P2: the quiet set is largely the SAME indices across layers (Jaccard of bottom-10% sets between L30 and L50 > 0.5);
#      refuted if Jaccard is near chance (~0.05).
#  P3: quietness is explained by READ gain: corr(log std, log |w_input_layernorm[l+1]|) strongly negative (< -0.5);
#      a weaker relation for write gain (post-attn / post-ffn norm weights).
#  P4 (alternative hypothesis to rule out): quiet dims are NOT the massive-activation / attention-sink coordinates
#      (those have huge |h| at BOS/delimiters, i.e. high max and high std); overlap of quiet set with top-1% absmax dims ~0.
#  P5: the quiet set measured on prose and on chat separately agrees (Jaccard > 0.6) -> a property of the network, not data.
exec(open("./experiments/scripts/V6_quiet_dims_common.py").read())
t0 = time.time()
D = 5376
seqs = v6_stats_corpus()
KEYS = [("prose", 0), ("prose", 1), ("chat", 0), ("chat", 1)]
acc = {k: {"s1": torch.zeros(60, D, dtype=torch.float64, device=lab.dev),
           "s2": torch.zeros(60, D, dtype=torch.float64, device=lab.dev),
           "n": 0} for k in KEYS}
amax = torch.zeros(60, D, device=lab.dev); amax_tok = torch.zeros(60, D, dtype=torch.long, device=lab.dev)
amax_pos = torch.zeros(60, D, dtype=torch.long, device=lab.dev)
bos = torch.zeros(60, D, device=lab.dev)
rng = random.Random(0)
samp_meta, samp = [], {l: [] for l in range(60)}
# group sequences by kind & parity, batch 6
groups = {k: [] for k in KEYS}
for j, s in enumerate(seqs):
    groups[(s["kind"], j % 2)].append(j)
ntok = 0
for key in KEYS:
    idxs = groups[key]
    for c0 in range(0, len(idxs), 6):
        ch = idxs[c0:c0 + 6]
        bids = [seqs[j]["ids"] for j in ch]
        # sample ~6 random non-BOS positions per sequence
        sp = [sorted(rng.sample(range(1, len(x)), min(6, len(x) - 1))) for x in bids]
        bi = torch.tensor([b for b, ps in enumerate(sp) for _ in ps], device=lab.dev)
        ti = torch.tensor([p for ps in sp for p in ps], device=lab.dev)
        for b, ps in enumerate(sp):
            for p in ps:
                samp_meta.append({"group": seqs[ch[b]]["group"], "kind": key[0], "pos": p, "tok": lab.strs([bids[b][p]])[0]})
        tokid = torch.zeros(len(bids), max(len(x) for x in bids), dtype=torch.long, device=lab.dev)
        for b, x in enumerate(bids):
            tokid[b, :len(x)] = torch.tensor(x, device=lab.dev)

        def fn(l, h, att, key=key, bi=bi, ti=ti, tokid=tokid):
            m = att.clone(); m[:, 0] = False
            hf = h.float()
            hm = hf * m.unsqueeze(-1)
            acc[key]["s1"][l] += hm.sum((0, 1)).double()
            acc[key]["s2"][l] += (hm.double() ** 2).sum((0, 1))
            a = hf.abs().masked_fill(~m.unsqueeze(-1), 0).reshape(-1, D)
            v, ix = a.max(0)
            upd = v > amax[l]
            amax[l] = torch.where(upd, v, amax[l])
            amax_tok[l] = torch.where(upd, tokid.reshape(-1)[ix], amax_tok[l])
            amax_pos[l] = torch.where(upd, ix % h.shape[1], amax_pos[l])
            bos[l] = hf[0, 0]
            samp[l].append(hf[bi, ti].half().cpu())
        v6_forward_hooked(bids, fn)
        n_here = sum(len(x) - 1 for x in bids)
        acc[key]["n"] += n_here; ntok += n_here
print("tokens", ntok, {str(k): acc[k]["n"] for k in KEYS}, "t", round(time.time() - t0, 1), flush=True)


def mstd(keys):
    n = sum(acc[k]["n"] for k in keys)
    s1 = sum(acc[k]["s1"] for k in keys); s2 = sum(acc[k]["s2"] for k in keys)
    mu = s1 / n
    var = (s2 / n - mu ** 2).clamp_min(0) * n / (n - 1)
    return mu.float().cpu(), var.sqrt().float().cpu(), n


out = {}
for name, keys in (("all", KEYS), ("prose", KEYS[:2]), ("chat", KEYS[2:]), ("even", [KEYS[0], KEYS[2]]), ("odd", [KEYS[1], KEYS[3]])):
    mu, sd, n = mstd(keys)
    out[f"mean_{name}"] = mu; out[f"std_{name}"] = sd; out[f"n_{name}"] = n
out["amax"] = amax.cpu(); out["amax_tok"] = amax_tok.cpu(); out["amax_pos"] = amax_pos.cpu(); out["bos"] = bos.cpu()

# ---- read/write gains of each residual coordinate (input dim of q/k/v/gate/up; output dim of o/down)
from backend.engine_runtime import dequantize_w4a16_full
def colsq(mod):
    if mod is None or not hasattr(mod, "weight_packed"):
        return torch.zeros(5376, device=lab.dev)
    W = dequantize_w4a16_full(mod.weight_packed, mod.weight_scale)
    r = torch.zeros(W.shape[1], device=lab.dev)
    for i in range(0, W.shape[0], 4096):
        r += W[i:i + 4096].float().pow(2).sum(0)
    del W
    return r
def rowsq(mod):
    W = dequantize_w4a16_full(mod.weight_packed, mod.weight_scale)
    r = W.float().pow(2).sum(1)
    del W
    return r
NOV = [l for l in range(60) if getattr(lab.layers[l].self_attn, "v_proj", None) is None]
print("layers without v_proj:", NOV)
G = {k: [] for k in ("w_in", "w_preffn", "w_postattn", "w_postffn", "qkv_col", "gu_col", "o_row", "down_row", "scalar")}
with torch.no_grad():
    for l in range(60):
        L = lab.layers[l]
        G["w_in"].append(L.input_layernorm.weight.float().cpu())
        G["w_preffn"].append(L.pre_feedforward_layernorm.weight.float().cpu())
        G["w_postattn"].append(L.post_attention_layernorm.weight.float().cpu())
        G["w_postffn"].append(L.post_feedforward_layernorm.weight.float().cpu())
        a = L.self_attn
        G["qkv_col"].append((colsq(a.q_proj) + colsq(a.k_proj) + colsq(a.v_proj)).sqrt().cpu())
        G["gu_col"].append((colsq(L.mlp.gate_proj) + colsq(L.mlp.up_proj)).sqrt().cpu())
        G["o_row"].append(rowsq(a.o_proj).sqrt().cpu())
        G["down_row"].append(rowsq(L.mlp.down_proj).sqrt().cpu())
        G["scalar"].append(L.layer_scalar.float().cpu().reshape(1))
        torch.cuda.empty_cache()
for k in G:
    out["g_" + k] = torch.stack(G[k], 0)
out["fnw"] = rt.final_norm_weight.float().cpu()
out["wu_col"] = torch.stack([rt.lm_head_weight[i:i + 32768].float().pow(2).sum(0) for i in range(0, rt.lm_head_weight.shape[0], 32768)]).sum(0).sqrt().cpu()
print("gains done t", round(time.time() - t0, 1), flush=True)
torch.save(out, V6_R + "V6_quiet_dims_stats.pt")
torch.save({"meta": samp_meta, "H": torch.stack([torch.cat(samp[l], 0) for l in range(60)], 1)}, V6_R + "V6_quiet_dims_samples.pt")
# quick summary
sd = out["std_all"]
summ = {}
for l in (10, 20, 30, 35, 41, 47, 53, 58):
    s = sd[l]
    summ[l] = {"q01": float(torch.quantile(s, .01)), "q10": float(torch.quantile(s, .10)), "q50": float(s.median()), "q90": float(torch.quantile(s, .9)), "max": float(s.max())}
print(json.dumps(summ))
result = {"ntok": ntok, "nsamp": len(samp_meta), "std_summary": summ, "elapsed": round(time.time() - t0, 1)}
