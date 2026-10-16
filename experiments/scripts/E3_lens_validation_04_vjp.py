# E3_lens_validation (b) true same-position VJPs for 4 final-space directions:
#   0 = pron gender direction d (lab.gender_direction('pron')), 1 = noun gender direction, 2,3 = isotropic random
#   directions (scaled to |d|).  144 positions = 24 default chats x (<=2 self + non-self assistant tokens, 4 total)
#   + 24 prose paragraphs x 2 random positions. Run in 3 chunks (CHUNK=0,1,2); B=2 batched, K=4 backward passes.
# PRE-STATED PREDICTIONS:
#  P4 single-position cos(g_l, J++ pullback) small (~0.05-0.15 mid layers, as in pilot), but the AVERAGE gbar_l
#     will align much better with the JLens pullback (JLens is literally an average Jacobian) than single g_l,
#     reaching maybe 0.3-0.6 at L30-50 if transfer base->IT is decent; J++ (a fitted map) somewhat lower.
#  P5 split-half reliability of gbar will be well below 1 at early/mid layers (Jacobians vary a lot by position).
#  P6 random directions behave similarly to gender direction (no special status for d).
import json, random, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
exec(open("./experiments/scripts/E3_lens_validation_vjplib.py").read())
t0 = time.time()
CH = globals().get("CHUNK", 0)
chats = json.load(open(E3_RES + "E3_lens_validation_chats.json"))
rng = random.Random(1234)
items = []
for i, c in enumerate(chats):
    ids = lab.ids(lab.chat(c["q"], assistant=c["a"]))
    cats = e3_chat_positions(lab, ids, SELF_TOKS)
    selfp = [p for p in cats["self"] if p <= 110]
    nons = [p for p in cats["nonself_asst"] if 1 + cats["asst"][0] <= p <= 110]
    sp = sorted(rng.sample(selfp, min(2, len(selfp))))
    npk = sorted(rng.sample(nons, 4 - len(sp)))
    for p in sp: items.append({"group": f"chat{i:02d}", "src": "chat", "cat": "self", "pos": p, "ids": ids})
    for p in npk: items.append({"group": f"chat{i:02d}", "src": "chat", "cat": "asst_nonself", "pos": p, "ids": ids})
for i, (tag, txt) in enumerate(PROSE):
    ids = lab.ids(txt)
    for p in sorted(rng.sample(range(10, len(ids)), 2)):
        items.append({"group": f"prose{i:02d}", "src": "prose", "cat": tag, "pos": p, "ids": ids})
order = sorted(range(len(items)), key=lambda j: items[j]["pos"])
pairs = [order[k:k + 2] for k in range(0, len(order), 2)]
chunk_pairs = pairs[CH::3]
# directions
d_p = lab.gender_direction("pron").cpu(); d_n = lab.gender_direction("noun").cpu()
g = torch.Generator().manual_seed(7)
r1 = torch.randn(5376, generator=g); r2 = torch.randn(5376, generator=g)
DIRS = torch.stack([d_p, d_n, r1 * d_p.norm() / r1.norm(), r2 * d_p.norm() / r2.norm()]).float()
torch.save(DIRS, E3_RES + "E3_lens_validation_vjp_dirs.pt")
Dg = DIRS.to(lab.dev)
PB = {"jpp": [], "jlens": []}
for l in range(60):
    if l == 59:
        PB["jpp"].append(Dg.clone()); PB["jlens"].append(Dg.clone()); continue
    PB["jpp"].append((Dg.to(torch.bfloat16) @ lab.J[l]).float())
    PB["jlens"].append((Dg.to(torch.bfloat16) @ lab.JL[l].to(lab.dev)).float())
PB = {k: torch.stack(v, 1) for k, v in PB.items()}  # [4, 60, D]
NA = Dg.unsqueeze(1).expand(4, 60, 5376)
Gs, meta = [], []
for pr in chunk_pairs:
    torch.cuda.empty_cache()
    seqs = [items[j]["ids"] for j in pr]; poss = [items[j]["pos"] for j in pr]
    try:
        G = e3_vjp_batch(lab, seqs, poss, DIRS)  # [4, B, 60, D]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); print("OOM -> falling back to B=1", flush=True)
        Gl, keep = [], []
        for jj_, sq, ps in zip(pr, seqs, poss):
            for attempt in range(6):
                try:
                    Gl.append(e3_vjp_batch(lab, [sq], [ps], DIRS)); keep.append(jj_); break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); print("OOM at B=1, retry after wait", jj_, attempt, flush=True); time.sleep(15)
        if not Gl:
            continue
        G = torch.cat(Gl, 1); pr = keep
    for b, j in enumerate(pr):
        gb = G[:, b].to(lab.dev)
        rec = {k: items[j][k] for k in ("group", "src", "cat", "pos")}
        rec["tok"] = lab.strs([items[j]["ids"][items[j]["pos"]]])[0]
        rec["cos_jpp"] = F.cosine_similarity(gb, PB["jpp"], dim=-1).cpu().tolist()
        rec["cos_jlens"] = F.cosine_similarity(gb, PB["jlens"], dim=-1).cpu().tolist()
        rec["cos_naive"] = F.cosine_similarity(gb, NA, dim=-1).cpu().tolist()
        rec["gnorm"] = gb.norm(dim=-1).cpu().tolist()
        meta.append(rec); Gs.append(G[:, b].half())
    print("pair", len(meta), round(time.time() - t0, 1), flush=True)
    if len(meta) % 12 == 0:
        torch.save({"meta": meta, "G": torch.stack(Gs, 0)}, E3_RES + f"E3_lens_validation_vjp_chunk{CH}.pt")
torch.save({"meta": meta, "G": torch.stack(Gs, 0)}, E3_RES + f"E3_lens_validation_vjp_chunk{CH}.pt")
print("n items total", len(items), "chunk", CH, "n", len(meta), "time", time.time() - t0)
result = {"meta": meta}
