# V4_latent_identity_03_analyze_diag (CPU): persona-identity specificity matrices.
# For each group and lens: own-persona token log P minus the same token's mean under the OTHER named AI personas
# (gemma/llama/claude/qwen/chatgpt/atlas), paired by passage. Tests whether ANY position type carries the persona identity.
import json, numpy as np
R = "./experiments/results"
rng = np.random.default_rng(1)
z = np.load(f"{R}/V4_latent_identity_01_self.npz"); m = json.load(open(f"{R}/V4_latent_identity_01_self_meta.json"))
LP, items, grp = z["LP"], z["items"], z["grp"]
LAY = list(z["layers"]); KINDS = list(z["kinds"]); TOK = list(z["toks"]); GN = m["groups"]; meta = m["meta"]
AI = ["gemma", "llama", "claude", "qwen", "chatgpt", "atlas"]
OWN = {"gemma": " Gemma", "llama": " L", "claude": " Claude", "qwen": " Q", "chatgpt": " ChatGPT", "atlas": " Atlas"}
MAKER = {"gemma": " Google", "llama": " Meta", "claude": " Anth", "qwen": " Alibaba", "chatgpt": " OpenAI"}
idx = {(mm["persona"], mm["q"]): i for i, mm in enumerate(meta)}


def pv(persona, g, tok, kind, lis):
    ki = KINDS.index(kind); ti = TOK.index(tok); gi = GN.index(g)
    out = np.full(12, np.nan)
    for q in range(12):
        msk = (items == idx[(persona, q)]) & (grp == gi)
        if msk.sum():
            out[q] = LP[msk][:, lis, ki, ti].mean()
    return out


def bm(x):
    x = x[~np.isnan(x)]
    b = rng.choice(x, (2000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


res = {}
bands = {"L30-39": [30, 33, 36, 39], "L42-51": [42, 45, 48, 51], "L55": [55], "L59": [59]}
for kind in KINDS:
    res[kind] = {}
    for g in GN:
        res[kind][g] = {}
        for bn, bl in bands.items():
            lis = [LAY.index(l) for l in bl]
            dn, dm = [], []
            rown = {}
            for pn in AI:
                own = pv(pn, g, OWN[pn], kind, lis)
                oth = np.nanmean([pv(p2, g, OWN[pn], kind, lis) for p2 in AI if p2 != pn], 0)
                dn.append(own - oth)
                rown[pn] = float(np.nanmean(own - oth))
                if pn in MAKER:
                    own = pv(pn, g, MAKER[pn], kind, lis)
                    oth = np.nanmean([pv(p2, g, MAKER[pn], kind, lis) for p2 in MAKER if p2 != pn], 0)
                    dm.append(own - oth)
            res[kind][g][bn] = {"name_diag_mean": bm(np.nanmean(dn, 0)), "maker_diag_mean": bm(np.nanmean(dm, 0)), "per_persona_name": rown}
json.dump(res, open(f"{R}/V4_latent_identity_03_diag.json", "w"), indent=1)
for kind in KINDS:
    print(f"== {kind}: own-persona name / maker token minus other-AI-persona mean (nats, paired over 12 passages)")
    for g in GN:
        print(f"  {g:9s}", " | ".join(f"{bn}: name {v['name_diag_mean'][0]:+.2f} [{v['name_diag_mean'][1]:+.2f},{v['name_diag_mean'][2]:+.2f}] maker {v['maker_diag_mean'][0]:+.2f}" for bn, v in res[kind][g].items()))
    print("   per-persona name diag at self L42-51:", {k: round(v, 2) for k, v in res[kind]["self"]["L42-51"]["per_persona_name"].items()})
    print("   per-persona name diag at user_you L42-51:", {k: round(v, 2) for k, v in res[kind]["user_you"]["L42-51"]["per_persona_name"].items()})
