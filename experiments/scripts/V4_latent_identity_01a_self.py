FAMILY="self"
# V4_latent_identity_01_readout -- per-token lens log P / rank of identity tokens at typed positions under 10 personas.
# FAMILY = "self" (12 new first-person passages x 10 personas; groups self/asst_you/neutral/user_you/user_I)
#       or "tp"   (8 templates x 7 third-party referents {ChatGPT,Claude,Llama,Gemini,Siri,Gemma (all 'it'), Mark ('he')}
#                  x personas none/llama/man; groups tp_pron/tp_name/neutral/user_name).
# Layers 30,33,...,55 + 59; lenses jpp, jlens, naive. Full-vocab softmax per position (no state averaging).
# PRE-REGISTERED PREDICTIONS (C5 as stated vs refutations):
#  P1 replication: self tokens L35-49 jpp log P(' Gemma'): none~llama~claude~qwen~chatgpt~atlas~startup within 0.5 nats;
#     woman/man lower by >2 nats. Refuted if AI personas differ by >1 nat from none.
#  P2 persona-name control: under persona N (Claude/ChatGPT/Atlas/Meta/OpenAI/Alibaba tokens), N's own token at self tokens
#     L35-49 rises <0.5 nats vs none (claim). If it rises >1 nat the readout is generic persona tracking, and "Gemma under
#     Llama" would be the oddity rather than the rule.
#  P3 self-specificity: self minus third-party-AI pronoun ('it'->ChatGPT/Claude/Llama/Siri, persona none) log P(' Gemma')
#     > +1 nat at L35-49. If <=0.5 nat -> generic "AI referent => Gemma" association, not self-identity.
#     Also self vs user_you (user's 'you' = the assistant) and vs user_I (human) -- user_I should be >2 nats lower.
#  P4 lens referent-readout positive control: at tp_pron, own-referent token log P minus same token under other referents
#     (diagonal - off-diagonal) > 2 nats at L35-49; 'it'->Gemma gives higher ' Gemma' than self tokens.
#  P5 the above qualitative pattern holds in jlens and naive.
FAMILY = globals().get("FAMILY", "self")
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
LAY = [30, 33, 36, 39, 42, 45, 48, 51, 55, 59]
KINDS = ["jpp", "jlens", "naive"]
toks, tok_ids, _ = id_tokens()
meta, all_ids = build_self() if FAMILY == "self" else build_tp()
GN = list(meta[0]["groups"].keys())
pos_lists, grp = [], []
for m in meta:
    pl = []
    for gi, g in enumerate(GN):
        for p in m["groups"][g]:
            pl.append(p); grp.append(gi)
    pos_lists.append(pl)
comp, items, poss = collect_states(all_ids, pos_lists, LAY, bs=12)
print("collected", tuple(comp.shape), round(time.time() - t0, 1))
std = comp.float().std(1)  # [L, D] per-dim std across sampled positions (for quiet-dim masks later)
LP, RK = lens_lp_rank(comp, LAY, KINDS, tok_ids)
print("lens done", round(time.time() - t0, 1))
np.savez_compressed(f"{V4_DIR}/V4_latent_identity_01_{FAMILY}.npz", LP=LP, RK=RK, items=items, poss=poss,
                    grp=np.array(grp), layers=np.array(LAY), kinds=np.array(KINDS), toks=np.array(toks),
                    std=std.numpy())
json.dump({"meta": meta, "groups": GN}, open(f"{V4_DIR}/V4_latent_identity_01_{FAMILY}_meta.json", "w"))
# quick print: jpp, L42, geometric-mean rank of ' Gemma' per persona x group
gi_ = toks.index(" Gemma")
pers = np.array([meta[i]["persona"] for i in items]); grp = np.array(grp)
li = LAY.index(42)
for pn in sorted(set(pers), key=list(PERSONAS).index):
    row = []
    for gidx, g in enumerate(GN):
        msk = (pers == pn) & (grp == gidx)
        if msk.sum():
            row.append(f"{g}:{np.exp(np.log(RK[msk, li, 0, gi_]).mean()):.0f}/{LP[msk, li, 0, gi_].mean():+.1f}")
    print(pn, " ".join(row))
result = {"elapsed": time.time() - t0, "N": int(comp.shape[1])}
