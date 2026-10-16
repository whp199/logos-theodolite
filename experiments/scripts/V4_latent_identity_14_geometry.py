# V4_latent_identity_14_geometry -- where does the DEFAULT self state sit on the data-derived Gemma-vs-Llama identity axis?
# u_l = mean_self(gemma) - mean_self(llama) (quiet dims masked), L35-49, from the 12 SELF_QA passages. For each persona P:
#   c_P = <mean_self(P) - mean_self(llama), u> / |u|^2   (llama -> 0, gemma -> 1 by construction).
# Split-half cross-validation: u from passages 0-5, c computed on passages 6-11 (and vice versa) to avoid circularity.
# Also the same for neutral assistant tokens and user 'you' tokens.
# PRE-REGISTERED: if the default self carries the Gemma identity like the explicit persona, c_none ~ 1 (>= 0.7);
#   if it is identity-neutral, c_none ~ c_startup ~ c_claude (~0-0.3). Human personas: unknown.
exec(open("./experiments/scripts/V4_latent_identity_common.py").read(), globals())
t0 = time.time()
SL = list(range(35, 50))
meta, all_ids = build_self()
groups = ["self", "neutral", "user_you"]
pos_lists = [sum([m["groups"][g] for g in groups], []) for m in meta]
comp, items, poss = collect_states(all_ids, pos_lists, SL, bs=12)
comp = comp.float()
pers = np.array([meta[i]["persona"] for i in items]); qq = np.array([meta[i]["q"] for i in items])
gl = np.array([[g for g in groups if poss[j] in meta[items[j]]["groups"][g]][0] for j in range(len(items))])
std = comp.std(1); quiet = std < torch.quantile(std, 0.10, dim=1, keepdim=True)
res = {}
for g in groups:
    for split in [(range(0, 6), range(6, 12)), (range(6, 12), range(0, 6))]:
        def mean_of(pn, qs):
            msk = torch.tensor((pers == pn) & (gl == g) & np.isin(qq, list(qs)))
            return comp[:, msk].mean(1) * (~quiet).float()
        u = mean_of("gemma", split[0]) - mean_of("llama", split[0])
        base = mean_of("llama", split[1])
        for pn in PERSONAS:
            c = ((mean_of(pn, split[1]) - base) * u).sum(1) / u.pow(2).sum(1)    # per layer
            res.setdefault(f"{g}|{pn}", []).append(c.tolist())
summary = {k: {"c_mean_over_layers_and_splits": float(np.mean(v)), "per_layer": np.mean(v, 0).tolist()} for k, v in res.items()}
json.dump({"layers": SL, "summary": summary}, open(f"{V4_DIR}/V4_latent_identity_14_geometry.json", "w"), indent=1)
for g in groups:
    print(g, " ".join(f"{pn}:{summary[f'{g}|{pn}']['c_mean_over_layers_and_splits']:+.2f}" for pn in PERSONAS))
result = {"elapsed": time.time() - t0}
