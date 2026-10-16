# N1V_03_acts -- (daemon job, resumable) record mlp_act / gate pre-act g / up proj u for the 64 FRESH neurons
# (N1V_neurons.json, seed 99) over the N1B corpus (147.5k positions; same windows as N1B so empirical labels are built
# identically). Body = N1B_02_acts.py with paths switched (output -> results/N1V_acts/).
# PRE-STATED PREDICTIONS (verifier, for N1V_04_analyze): on fresh neurons
#  V3a activation-based next-token labels (Enx) agree with causal labels only weakly (lift < 0.03 overall), as in N1C.
#  V3b gate_trigger_tokens (G) vs empirical input tokens (Ein): small positive lift (< 0.04), far below split-half ceiling.
#  V3c the two poles' input-token sets overlap at null level.
import json, os, time
R = "./experiments/results/"
OUTD = R + "N1V_acts/"
os.makedirs(OUTD, exist_ok=True)
BS = int(globals().get("N1B_BS", 16))
CH_PER_PART = BS * 8
TBUDGET = float(globals().get("N1V_TBUDGET", 170))
t0 = time.time()
neu = json.load(open(R + "N1V_neurons.json"))
LAYERS = neu["layers"]
corpus = [json.loads(l) for l in open(R + "N1B_corpus.jsonl")]
nparts = (len(corpus) + CH_PER_PART - 1) // CH_PER_PART
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
idx = {l: torch.tensor(neu["neurons"][str(l)], device=lab.dev) for l in LAYERS}
cur = {}
hooks = []
def mk_pre(l):
    def hook(mod, args):
        cur[("a", l)] = args[0][..., idx[l]].float()
    return hook
def mk_fwd(l, key):
    def hook(mod, args, out):
        cur[(key, l)] = out[..., idx[l]].float()
    return hook
for l in LAYERS:
    mlp = lab.layers[l].mlp
    hooks.append(mlp.down_proj.register_forward_pre_hook(mk_pre(l)))
    hooks.append(mlp.gate_proj.register_forward_hook(mk_fwd(l, "g")))
    hooks.append(mlp.up_proj.register_forward_hook(mk_fwd(l, "u")))
log = []
try:
    for part in range(nparts):
        fn = OUTD + f"part_{part:03d}.npz"
        if os.path.exists(fn):
            continue
        if time.time() - t0 > TBUDGET:
            break
        chunks = corpus[part * CH_PER_PART:(part + 1) * CH_PER_PART]
        A, G, U, CID, POS = [], [], [], [], []
        def fwd(bch):
            T = max(len(c["ids"]) for c in bch)
            inp = torch.zeros(len(bch), T, dtype=torch.long)
            att = torch.zeros(len(bch), T, dtype=torch.long)
            for b, c in enumerate(bch):
                inp[b, :len(c["ids"])] = torch.tensor(c["ids"])
                att[b, :len(c["ids"])] = 1
            cur.clear()
            try:
                with torch.no_grad():
                    lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
            except torch.OutOfMemoryError:
                cur.clear()
                torch.cuda.empty_cache()
                if len(bch) == 1:
                    raise
                h_ = len(bch) // 2
                print("OOM -> split", len(bch), flush=True)
                fwd(bch[:h_]); fwd(bch[h_:])
                return
            m = att.bool()
            A.append(torch.cat([cur[("a", l)] for l in LAYERS], -1).cpu()[m])   # [N, 192]
            G.append(torch.cat([cur[("g", l)] for l in LAYERS], -1).cpu()[m])
            U.append(torch.cat([cur[("u", l)] for l in LAYERS], -1).cpu()[m])
            CID.append(torch.tensor([c["cid"] for c in bch]).unsqueeze(1).expand(-1, T)[m])
            POS.append(torch.arange(T).unsqueeze(0).expand(len(bch), -1)[m])
            cur.clear()
        for b0 in range(0, len(chunks), BS):
            fwd(chunks[b0:b0 + BS])
        np.savez(fn, act=torch.cat(A).numpy().astype(np.float32), g=torch.cat(G).numpy().astype(np.float16),
                 u=torch.cat(U).numpy().astype(np.float16), cid=torch.cat(CID).numpy().astype(np.int32),
                 pos=torch.cat(POS).numpy().astype(np.int16))
        log.append({"part": part, "n_tok": int(sum(x.shape[0] for x in A)), "t": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
finally:
    for h in hooks:
        h.remove()
    cur.clear()
done = len([f for f in os.listdir(OUTD) if f.endswith(".npz")])
result = {"log": log, "nparts": nparts, "parts_done": done, "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
          "elapsed": time.time() - t0}
