# E5_name job C: representational link between the model's self tokens (job B) and third-person name states (job A).
# All states centered by the global token mean (positions >= 2 of every prompt in jobs A and B), per layer.
# Spaces: 'raw' resid_post and 'jpp' ((h - mu) @ J[l]^T). Bootstrap: frames (16, third-person) and passages (12, self)
# resampled independently, 1000 resamples; third-person conditions share frames (paired), self conditions share passages.
#
# PRE-STATED PREDICTIONS:
#  C1 cos(S_default - S_nameless, d_GemmaID) ~ 0 (CI includes 0 or < 0.1); positive control cos(S_gemma_sys - S_nameless, d_GemmaID) > 0.
#  C2 Lens-free gender axis from third-person names (Sarah/Emma vs James/Liam, human context) transfers to self tokens:
#     woman persona > man persona; default lands near the middle (0.3-0.7 on the woman/man scale).
#  C3 Third-person map: Gemma(AI, Google) has AI-score near the generic AI names (>0.7) and a female score between generic AI
#     and Gemma(coworker).
exec(open("./experiments/scripts/E5_name_common.py").read(), globals())
t0 = time.time()
if "E5_name_A_states" in STORE:
    SA = STORE["E5_name_A_states"]; gA = STORE["E5_name_gsum_A"]
else:
    d = torch.load(E5_CACHE + "/A_states.pt"); SA = d["states"]; gA = (d["gsum"], d["gcnt"])
if "E5_name_B_states" in STORE:
    SBs = STORE["E5_name_B_states"]; gB = STORE["E5_name_gsum_B"]
else:
    d = torch.load(E5_CACHE + "/B_states.pt"); SBs = d["states"]; gB = (d["gsum"], d["gcnt"])
mu_all = ((gA[0] + gB[0]) / (gA[1] + gB[1])).float()      # [60, D]
SELF_CONDS = ["default", "gemma_sys", "nameless", "llama_sys", "claude_sys", "qwen_sys", "atlas_sys", "sarah_ai",
              "james_ai", "woman", "man"]
nN, nC, nFr, nS, nP = len(NAMES), len(CONTEXTS), len(FRAMES), len(SELF_CONDS), 12
assert SA.shape[0] == nN * nC * nFr and SBs.shape[0] == nS * nP
NI = {n: i for i, n in enumerate(NAMES)}; CI = {c: i for i, c in enumerate(CONTEXTS)}; SI = {c: i for i, c in enumerate(SELF_CONDS)}
GENERIC = ["Llama", "Mistral", "Qwen"]
NB = 1000
gen = torch.Generator(device="cpu").manual_seed(1234)
FB = torch.randint(0, nFr, (NB, nFr), generator=gen)
PB = torch.randint(0, nP, (NB, nP), generator=gen)
dev = lab.dev


def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)


def unit(a):
    return a / (a.norm(dim=-1, keepdim=True) + 1e-9)


def dot(a, b):
    return (a * b).sum(-1)


def quantities(NA, NF, SB):
    """Batched over resamples. NA, NF: [B, nN, nC, D] third-person name2 / final means; SB: [B, nS, D] self means.
    Returns dict of [B] tensors."""
    q = {}
    nm = lambda X, names, c: X[:, [NI[n] for n in names], CI[c]].mean(1)
    gen_ai = nm(NA, GENERIC, "ai_real")
    dG = NA[:, NI["Gemma"], CI["ai_real"]] - gen_ai
    dGn = NA[:, NI["Gemma"], CI["ai_neutral"]] - nm(NA, GENERIC, "ai_neutral")
    dC = NA[:, NI["Claude"], CI["ai_real"]] - gen_ai
    dL = NA[:, NI["Llama"], CI["ai_real"]] - nm(NA, ["Mistral", "Qwen"], "ai_real")
    dFh = nm(NA, ["Sarah", "Emma"], "human") - nm(NA, ["James", "Liam"], "human")
    dFa = nm(NA, ["Sarah", "Emma"], "ai_neutral") - nm(NA, ["James", "Liam"], "ai_neutral")
    s = lambda c: SB[:, SI[c]]
    dn = lambda c: s(c) - s("nameless")
    q["dcos_default_GemmaID"] = cos(dn("default"), dG)
    q["dcos_default_GemmaID_neutral"] = cos(dn("default"), dGn)
    q["dcos_gemmasys_GemmaID"] = cos(dn("gemma_sys"), dG)
    q["dcos_claudesys_GemmaID"] = cos(dn("claude_sys"), dG)
    q["dcos_llamasys_GemmaID"] = cos(dn("llama_sys"), dG)
    q["dcos_claudesys_ClaudeID"] = cos(dn("claude_sys"), dC)
    q["dcos_llamasys_LlamaID"] = cos(dn("llama_sys"), dL)
    q["dcos_default_gemmasys_selfonly"] = cos(dn("default"), dn("gemma_sys"))
    q["dcos_default_llamasys_selfonly"] = cos(dn("default"), dn("llama_sys"))
    q["dcos_default_claudesys_selfonly"] = cos(dn("default"), dn("claude_sys"))
    q["dcos_woman-man_femH"] = cos(s("woman") - s("man"), dFh)
    q["dcos_sarahai-jamesai_femA"] = cos(s("sarah_ai") - s("james_ai"), dFa)
    q["dcos_default_femH"] = cos(dn("default"), dFh)
    q["dcos_gemmasys_femH"] = cos(dn("gemma_sys"), dFh)
    q["dcos_default_femA"] = cos(dn("default"), dFa)
    uFh, uG = unit(dFh), unit(dG)
    pF = dot(SB, uFh.unsqueeze(1))   # [B, nS]
    pG = dot(SB, uG.unsqueeze(1))
    for c in SELF_CONDS:
        q[f"femscale_H|{c}"] = (pF[:, SI[c]] - pF[:, SI["man"]]) / (pF[:, SI["woman"]] - pF[:, SI["man"]])
        q[f"femscale_AIpersona|{c}"] = (pF[:, SI[c]] - pF[:, SI["james_ai"]]) / (pF[:, SI["sarah_ai"]] - pF[:, SI["james_ai"]])
        q[f"gemmascale|{c}"] = (pG[:, SI[c]] - pG[:, SI["nameless"]]) / (pG[:, SI["gemma_sys"]] - pG[:, SI["nameless"]])
    for tag, X in (("name2", NA), ("final", NF)):
        hum = X[:, :, CI["human"]].mean(1); ai = X[:, :, CI["ai_real"]].mean(1)
        uA = ai - hum
        a0, a1 = dot(hum, uA), dot(ai, uA)
        Fa = nm(X, ["Sarah", "Emma"], "human"); Ma = nm(X, ["James", "Liam"], "human")
        dF = Fa - Ma
        f1, f0 = dot(Fa, dF), dot(Ma, dF)
        A = (torch.einsum("bncd,bd->bnc", X, uA) - a0[:, None, None]) / (a1 - a0)[:, None, None]
        Fm = (torch.einsum("bncd,bd->bnc", X, dF) - f0[:, None, None]) / (f1 - f0)[:, None, None]
        for n in NAMES:
            for c in CONTEXTS:
                q[f"map_{tag}_ai|{n}|{c}"] = A[:, NI[n], CI[c]]
                q[f"map_{tag}_fem|{n}|{c}"] = Fm[:, NI[n], CI[c]]
        gai = nm(X, GENERIC, "ai_real")
        for n in ["Gemma", "Claude", "Gemini", "Sarah", "Emma", "James"]:
            xa = X[:, NI[n], CI["ai_real"]]
            xh = X[:, NI[n], CI["human"]]
            q[f"dist_{tag}|{n}_ai-vs-{n}_human"] = (xa - xh).norm(dim=-1)
            q[f"dist_{tag}|{n}_ai-vs-genericAI"] = (xa - gai).norm(dim=-1)
            q[f"distratio_{tag}|{n}"] = (xa - gai).norm(dim=-1) / (xa - xh).norm(dim=-1)
            q[f"cos_{tag}|{n}_ai-vs-{n}_human"] = cos(xa, xh)
            q[f"cos_{tag}|{n}_ai-vs-genericAI"] = cos(xa, gai)
    return q


WF = torch.zeros(NB, nFr).scatter_add_(1, FB, torch.ones(NB, nFr)) / nFr     # bootstrap weights over frames
WP = torch.zeros(NB, nP).scatter_add_(1, PB, torch.ones(NB, nP)) / nP        # ... over passages
CH = 50


out = {"layers": STORE_LAYERS, "spaces": {}}
for space in ("raw", "jpp"):
    res_space = {}
    cosmats = {}
    for li, l in enumerate(STORE_LAYERS):
        mu = mu_all[l].to(dev)
        XA = SA[:, li, 0].to(dev).float() - mu      # name2
        XF = SA[:, li, 1].to(dev).float() - mu      # final
        XB = SBs[:, li, 0].to(dev).float() - mu     # self tokens
        if space == "jpp":
            XA, XF, XB = (lab.lens(x, l, "jpp").float() for x in (XA, XF, XB))
        XA = XA.view(nN, nC, nFr, -1); XF = XF.view(nN, nC, nFr, -1); XB = XB.view(nS, nP, -1)
        full = quantities(XA.mean(2)[None], XF.mean(2)[None], XB.mean(1)[None])
        boots = {k: [] for k in full}
        for c0 in range(0, NB, CH):
            wf = WF[c0:c0 + CH].to(dev); wp = WP[c0:c0 + CH].to(dev)
            qb = quantities(torch.einsum("bf,ncfd->bncd", wf, XA), torch.einsum("bf,ncfd->bncd", wf, XF),
                            torch.einsum("bp,spd->bsd", wp, XB))
            for k, v in qb.items():
                boots[k].append(v)
        for k in full:
            bt = torch.cat(boots[k]).float().cpu().numpy()
            res_space.setdefault(k, []).append([float(full[k][0]), float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))])
        # full-sample cosine matrix: self conds x (name, ctx) at name2
        NAm = XA.mean(2).reshape(nN * nC, -1); SBm = XB.mean(1)
        cosmats[l] = cos(SBm.unsqueeze(1), NAm.unsqueeze(0)).cpu().numpy().round(4).tolist()
        print(space, l, "dcos_default_GemmaID", res_space["dcos_default_GemmaID"][-1], "gemmasys", res_space["dcos_gemmasys_GemmaID"][-1],
              "femH default", res_space["femscale_H|default"][-1], round(time.time() - t0, 1))
    out["spaces"][space] = {"q": res_space, "cosmat_self_x_namectx": cosmats}
out["cosmat_rows"] = SELF_CONDS
out["cosmat_cols"] = [f"{n}|{c}" for n in NAMES for c in CONTEXTS]
out["generic"] = GENERIC
json.dump(out, open(E5_DIR + "/E5_name_C_similarity.json", "w"))
# Instrument comparison: gender readouts of the SAME stored states (name2, final; self) through jpp / jlens / naive lenses.
lensR = {}
for kind in ("jpp", "jlens", "naive"):
    arrA = np.zeros((SA.shape[0], len(STORE_LAYERS), 2, len(GENDER_KEYS)), np.float32)
    arrB = np.zeros((SBs.shape[0], len(STORE_LAYERS), 2, len(GENDER_KEYS)), np.float32)
    for li, l in enumerate(STORE_LAYERS):
        for gi in range(2):
            sc = lab.gender_scores(lab.lens(SA[:, li, gi].to(dev), l, kind))
            arrA[:, li, gi] = torch.stack([sc[k].float() for k in GENDER_KEYS], -1).cpu().numpy()
            sc = lab.gender_scores(lab.lens(SBs[:, li, gi].to(dev), l, kind))
            arrB[:, li, gi] = torch.stack([sc[k].float() for k in GENDER_KEYS], -1).cpu().numpy()
    lensR[kind + "_A"] = arrA
    lensR[kind + "_B"] = arrB
np.savez_compressed(E5_DIR + "/E5_name_C_lenscompare.npz", layers=np.array(STORE_LAYERS), metrics=np.array(GENDER_KEYS), **lensR)
print("lens comparison saved", round(time.time() - t0, 1))
print("elapsed", round(time.time() - t0, 1))
result = {"elapsed": time.time() - t0}
