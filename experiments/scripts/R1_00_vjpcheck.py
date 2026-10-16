# R1_IT_LENS 00: validate the K-copy batched VJP against lab.same_pos_vjp, and time it.
# PREDICTION: per-layer cos(copies, same_pos_vjp) >= 0.98 at all layers (bf16/kernel noise only); K=256 in < 10 s.
import json, time
exec(open("./experiments/scripts/R1_vjplib.py").read())
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
lmcfg = _r1_lm(lab).config
info = {"attn_impl": getattr(lmcfg, "_attn_implementation", None),
        "num_kv_shared_layers": getattr(lmcfg, "num_kv_shared_layers", None)}
print(info, flush=True)
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
c = chats[0]
ids = lab.ids(lab.chat(c["q"], assistant=c["a"]))
cats = e3_chat_positions(lab, ids, SELF_TOKS)
tests = [("self", cats["self"][0]), ("asst", cats["nonself_asst"][5]), ("user", cats["user"][2])]
pids = lab.ids(PROSE[0][1]); tests.append(("prose", 40))
d_p = lab.gender_direction("pron").float()
g = torch.Generator().manual_seed(0)
Rr = torch.randn(255, 5376, generator=g).to(lab.dev) * d_p.norm() / (5376 ** 0.5)
res = {"info": info, "checks": []}
KS = (128, 192)
for nm, p in tests:
    seq = pids if nm == "prose" else ids
    torch.cuda.synchronize(); t0 = time.time()
    ref = lab.same_pos_vjp(seq, p, d_p)
    torch.cuda.synchronize(); t_ref = time.time() - t0
    ref2 = lab.same_pos_vjp(seq, p, Rr[0])
    for K in KS:
        R = torch.cat([d_p[None], Rr[: K - 1]], 0)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        G, H = r1_vjp_copies(lab, seq, p, R)
        torch.cuda.synchronize(); tk = time.time() - t0
        c0 = F.cosine_similarity(G[0], ref, dim=-1).cpu()
        c1 = F.cosine_similarity(G[1], ref2, dim=-1).cpu()
        nr = (G[0].norm(dim=-1) / ref.norm(dim=-1)).cpu()
        rec = {"cls": nm, "pos": p, "K": K, "t_ref": round(t_ref, 2), "t_K": round(tk, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "cos_pron_min": float(c0[:59].min()), "cos_pron_med": float(c0[:59].median()),
               "cos_rand_min": float(c1[:59].min()), "normratio_med": float(nr[:59].median()),
               "cos_pron_layers": [round(float(x), 4) for x in c0[::6]]}
        print(rec, flush=True)
        res["checks"].append(rec)
        del G
    torch.cuda.empty_cache()
# state check: H from copies vs forward_resid
Hf = lab.forward_resid([ids], positions=[[tests[0][1]]])[0][:, 0]
G, H = r1_vjp_copies(lab, ids, tests[0][1], d_p[None])
res["state_cos_min"] = float(F.cosine_similarity(H.cpu(), Hf, dim=-1).min())
print("state cos min", res["state_cos_min"])
result = res
