# E5_name job A: third-person name x context sweep. Format: user "Write a short paragraph." + teacher-forced model-turn
# passage "Sure! <N> <context sentence>. <frame with N>" (lead-in makes every mention the space-prefixed name token).
# (A raw-text smoke run showed the instruct model is off-distribution on raw text: junk " own" top token; see report.)
#
# PRE-STATED PREDICTIONS (written before running):
#  P1 behaviour/human context: Gemma, Emma, Jemma, Sarah -> strong "she" (lse_pron F-M > +3); James, Liam, Claude -> strong "he";
#     Gem female-leaning but weaker; Atlas male-leaning; Llama/Mistral/Qwen/Phi weaker / mixed in the human context.
#  P2 AI context: P(it) rises sharply for all names; F-M log-odds among the residual human pronouns shrinks but Gemma keeps a
#     female lean (de-gendering ratio lse(AI)/lse(human) between 0 and 0.6); Claude keeps a male lean (rough mirror image).
#  P3 Gemma(AI, Google) is closer to Llama/Mistral/Qwen(AI) than to Gemma(coworker) in the 4-way pronoun distribution
#     (it-dominance), but its residual F-M lean is on the female side of the generic-AI names.
#  P4 J++ readout at the contextualized (2nd) name mention: female human names > male human names in L30-49 after subtracting
#     the background (non-name tokens); AI context reduces the gap; strong background offsets in L10-19 / L40-49.
#  P5 Robustness (job A2): no lead-in (start-of-turn "Gemma" -> "G"+"emma"), a different user request, and raw text are NOT
#     expected to change the qualitative ordering of names.
# (Predictions written before any main run; a 4-prompt format probe was seen before switching to the chat format.)
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
t0 = time.time()
rows = grid()
if globals().get("SMOKE"):
    rows = rows[::29]
G = 4  # groups: 0 name1-last, 1 name2-last, 2 final, 3 background
all_ids, meta = [], []
for r in rows:
    ids = lab.ids(r["text"])
    cs = content_start(ids)
    n1 = name_positions(ids, r["name"], 0, after=cs)
    n2 = name_positions(ids, r["name"], 1, after=cs)
    fin = len(ids) - 1
    bg = [p for p in range(cs, fin) if p not in n1 and p not in n2]
    r.update({"n1": n1, "n2": n2, "fin": fin, "bg": bg, "ntok": len(ids)})
    all_ids.append(ids)
    meta.append(r)
print("example:", lab.strs(all_ids[0]), meta[0]["n1"], meta[0]["n2"], meta[0]["fin"])
print("name token counts:", {N: len(m["n2"]) for N, m in ((m["name"], m) for m in meta[::64])})

BS = 32
R_all, C_all, S_all = [], [], []
beh = {k: [] for k in ("p_f", "p_m", "p_it", "p_they", "lse_pron", "p_name")}
top5 = []
gsum = torch.zeros(60, 5376, dtype=torch.float64, device=lab.dev)
gcnt = [0.0]
for s in range(0, len(all_ids), BS):
    bids = all_ids[s:s + BS]
    bmeta = meta[s:s + BS]
    store, lengths = fwd_all_layers(bids)
    fb, fp, fg = [], [], []
    for b, m in enumerate(bmeta):
        for g, plist in enumerate([[m["n1"][-1]], [m["n2"][-1]], [m["fin"]], m["bg"]]):
            for p in plist:
                fb.append(b); fp.append(p); fg.append(g)
    R, C = readouts_for_groups(store, fb, fp, fg, len(bids), G)
    R_all.append(R); C_all.append(C)
    # raw states (name2-last, final) at STORE_LAYERS
    sb, sp, sg = [], [], []
    for b, m in enumerate(bmeta):
        sb += [b, b]; sp += [m["n2"][-1], m["fin"]]; sg += [0, 1]
    S_all.append(mean_states(store, sb, sp, sg, len(bids), 2, STORE_LAYERS))
    accumulate_global(store, lengths, gsum, gcnt)
    # behaviour from final-layer state at final position
    h59 = store[59][torch.arange(len(bids), device=lab.dev), torch.tensor([m["fin"] for m in bmeta], device=lab.dev)]
    bv, lp = behav_from_h59(h59)
    for k in ("p_f", "p_m", "p_it", "p_they", "lse_pron"):
        beh[k].extend(bv[k].tolist())
    for b, m in enumerate(bmeta):
        nid = lab.tok.encode(" " + m["name"], add_special_tokens=False)[0]
        beh["p_name"].append(float(lp[b, nid].exp()))
        v, i = torch.topk(lp[b], 5)
        top5.append([(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)])
    if s == 0:
        # sanity: real model logits via lab.next_logits_batch on first 4 prompts
        ref = lab.next_logits_batch(bids[:4])
        lpr = torch.log_softmax(ref, -1)
        g_ = lab.gender
        pf_ref = torch.logsumexp(lpr[:, torch.tensor(g_["pron_f"], device=lab.dev)], -1).exp().cpu().numpy()
        print("sanity p_f mine vs next_logits_batch:", bv["p_f"][:4].round(4), pf_ref.round(4))
        print("max |logit diff|:", float((lab.full_logits(h59[:4]) - ref).abs().max()))
    del store
    torch.cuda.empty_cache()
R_all = np.concatenate(R_all, 0)
C_all = np.concatenate(C_all, 0)
S_all = torch.cat(S_all, 0)
STORE["E5_name_A_states"] = S_all
STORE["E5_name_gsum_A"] = (gsum.cpu(), gcnt[0])
torch.save({"states": S_all, "layers": STORE_LAYERS, "groups": ["name2_last", "final"],
            "gsum": gsum.cpu(), "gcnt": gcnt[0]}, E5_CACHE + "/A_states.pt")
np.savez_compressed(E5_DIR + "/E5_name_A_readouts.npz", R=R_all, C=C_all,
                    metrics=np.array(GENDER_KEYS + EXTRA_TOKS), groups=np.array(["name1", "name2", "final", "bg"]),
                    names=np.array([m["name"] for m in meta]), ctx=np.array([m["ctx"] for m in meta]),
                    frame=np.array([m["frame"] for m in meta]),
                    **{"beh_" + k: np.array(v) for k, v in beh.items()})
json.dump({"meta": [{k: m[k] for k in ("name", "ctx", "frame", "text", "n1", "n2", "fin", "ntok")} for m in meta],
           "top5": top5, "beh": beh}, open(E5_DIR + "/E5_name_A_meta.json", "w"))
print("shapes", R_all.shape, S_all.shape, "elapsed", round(time.time() - t0, 1))
# quick summary: behaviour per name x ctx
summ = {}
names_arr = np.array([m["name"] for m in meta]); ctx_arr = np.array([m["ctx"] for m in meta])
for N in NAMES:
    for c in CONTEXTS:
        msk = (names_arr == N) & (ctx_arr == c)
        if msk.any():
            summ[f"{N}|{c}"] = {k: float(np.mean(np.array(beh[k])[msk])) for k in beh}
for k, v in summ.items():
    print(f"{k:22s} lse={v['lse_pron']:+6.2f} pf={v['p_f']:.3f} pm={v['p_m']:.3f} it={v['p_it']:.3f} they={v['p_they']:.3f} name={v['p_name']:.3f}")
result = {"summary": summ, "elapsed": time.time() - t0}
