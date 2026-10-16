# V3_persona_axis_traits_01: collect self-token / background-token residual means (L20-50), J++ & naive gender readouts,
# and a token sample for covariance, for persona-gender, trait, control-trait, role and reference conditions
# (24 identity-neutral teacher-forced QA passages from E4), plus a third-person prose set.
# Set V3_GROUP = "A" (GP*/TP*) or "B" (CT*/RL*/refs + prose) before exec, or via the first line.
#
# PRE-REGISTERED PREDICTIONS (hypothesis H: the woman-man PERSONA axis mixes gender with stereotyped traits
# (warm/empathetic/service vs blunt/stoic), and the AI-vs-human offset aligns with the trait part):
#  H1 cos(u_persona, u_trait) >= 0.3 at L20-50 (raw), and stays >= 0.2 in whitened coordinates.
#  H2 cos(AI offset, u_trait) > max(0.1, cos(AI offset, u_prose)); default/nameless AI sit on the WARM side (>0.5) of
#     the blunt(0)-warm(1) trait scale relative to the neutral-human reference.
#  H3 Projecting u_trait (or the 4-phrasing trait subspace) out of u_persona moves default from ~0.84 to <= 0.65;
#     projecting out the control-trait subspace moves it by < 0.05.
#  H4 On the persona axis AI_blunt < AI_warm, with AI_blunt <= 0.6; warm human personas > 0.5, blunt < 0.5.
#  H5 Steering along u_trait (+warm) raises forced-choice woman-vs-man and P3 she-vs-he (tested in 03).
# REFUTATION: cos(u_persona,u_trait) < 0.1, or trait projection-out moves default by < 0.1, or AI_blunt stays >= 0.8,
#  -> H refuted; if whitening alone puts default near 0.5 on the persona axis -> raw-metric anisotropy, not traits.
exec(open("./experiments/scripts/V3_persona_axis_traits_common.py").read())
try:
    V3_GROUP
except NameError:
    V3_GROUP = "A"
t0 = time.time()
D = 5376
PUNCT = set(list(".,!?;:'\"-()"))
LIDX = {l: i for i, l in enumerate(LAYERS)}
rng = random.Random(1234 if V3_GROUP == "A" else 4321)

def turn_spans(ids):
    strs = lab.strs(ids)
    mt = lab.model_turn_pos(ids)
    ch = [i for i in range(mt, len(ids)) if strs[i] == "<channel|>"][0]
    selfpos = [i for i in range(ch + 1, len(ids)) if strs[i].strip() in SELF_TOKS]
    bgpos = [i for i in range(ch + 1, len(ids)) if strs[i].strip() not in SELF_TOKS and strs[i].strip() not in PUNCT and strs[i].strip() != ""]
    return selfpos, bgpos

conds = GROUP_A if V3_GROUP == "A" else GROUP_B
items = [dict(cond=c, p=p, sys=COND[c](p)) for c in conds for p in range(NP)]
N = len(items)
NL = len(LAYERS)
self_m = torch.zeros(N, NL, D); bg_m = torch.zeros(N, NL, D, dtype=torch.float16)
RK = ("pron", "noun", "lse_pron", "she_he")
rd = {f"{w}_{k2}_{sc}": torch.zeros(N, NL) for w in ("self", "bg") for k2 in ("jpp", "naive") for sc in RK}
TOKS_PER = 10
tok_samp = {l: [] for l in COV_LAYERS}
tok_meta = []
nself = []
BS = 16
for b0 in range(0, N, BS):
    batch = items[b0:b0 + BS]
    ids_l, pos_l, samp_l = [], [], []
    for it in batch:
        q, a = QA[it["p"]]
        ids = lab.ids(lab.chat(q, system=it["sys"], assistant=a))
        sp, gp = turn_spans(ids)
        ids_l.append(ids); pos_l.append((sp, gp)); nself.append(len(sp))
        allp = sorted(sp + gp)
        sm = sorted(rng.sample(allp, min(TOKS_PER, len(allp))))
        samp_l.append(sm)
        tok_meta += [(b0 + len(samp_l) - 1, it["cond"], it["p"], s) for s in sm]
    B = len(ids_l); T = max(len(x) for x in ids_l)
    inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
    for j, x in enumerate(ids_l):
        inp[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
    hs = []
    def mk(l):
        def hook(mod, args, out):
            h = (out[0] if isinstance(out, tuple) else out).detach()
            with torch.no_grad():
                li = LIDX[l]
                for j, (sp, gp) in enumerate(pos_l):
                    n = b0 + j
                    self_m[n, li] = h[j, sp].float().mean(0).cpu()
                    bg_m[n, li] = h[j, gp].float().mean(0).half().cpu()
                    for w, hh in (("self", h[j, sp]), ("bg", h[j, gp])):
                        for k2 in ("jpp", "naive"):
                            sc = lab.gender_scores(lab.lens(hh, l, k2))
                            for k in RK:
                                rd[f"{w}_{k2}_{k}"][n, li] = sc[k].float().mean().cpu()
                    if l in tok_samp:
                        tok_samp[l].append(h[j, samp_l[j]].half().cpu())
        return hook
    try:
        for l in LAYERS:
            hs.append(lab.layers[l].register_forward_hook(mk(l)))
        with torch.no_grad():
            lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
print("self collection done", N, round(time.time() - t0, 1))
torch.save({"self": self_m, "bg": bg_m, **rd, "items": items, "nself": nself, "layers": LAYERS},
           f"{V3_CACHE}/self_{V3_GROUP}.pt")
torch.save({"toks": {l: torch.cat(v, 0) for l, v in tok_samp.items()}, "meta": tok_meta}, f"{V3_CACHE}/toks_{V3_GROUP}.pt")
summary = {}
for c in conds:
    idx = [i for i, it in enumerate(items) if it["cond"] == c]
    summary[c] = {b: round(rd["self_jpp_pron"][idx][:, [LIDX[l] for l in r]].mean().item(), 3)
                  for b, r in (("L20-29", range(20, 30)), ("L30-39", range(30, 40)), ("L40-50", range(40, 51)))}
print(json.dumps(summary))

prose_out = None
if V3_GROUP == "B":
    pitems = prose_items()
    PN = len(pitems)
    cont_m = torch.zeros(PN, NL, D); last_m = torch.zeros(PN, NL, D)
    prd = {f"{w}_jpp_{sc}": torch.zeros(PN, NL) for w in ("cont", "last") for sc in ("pron", "noun")}
    for b0 in range(0, PN, 24):
        batch = pitems[b0:b0 + 24]
        ids_l, pos_l = [], []
        for it in batch:
            ids = lab.ids(it["text"]); pre = lab.ids(it["prefix"]); npre = len(pre)
            assert ids[:npre] == pre, (it["text"], lab.strs(ids[:npre + 1]), lab.strs(pre))
            ids_l.append(ids); pos_l.append((list(range(npre, len(ids))), len(ids) - 1))
        B = len(ids_l); T = max(len(x) for x in ids_l)
        inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
        for j, x in enumerate(ids_l):
            inp[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        hs = []
        def mk2(l):
            def hook(mod, args, out):
                h = (out[0] if isinstance(out, tuple) else out).detach()
                with torch.no_grad():
                    li = LIDX[l]
                    for j, (cp, lp) in enumerate(pos_l):
                        n = b0 + j
                        cont_m[n, li] = h[j, cp].float().mean(0).cpu()
                        last_m[n, li] = h[j, lp].float().cpu()
                        for w, hh in (("cont", h[j, cp]), ("last", h[j, lp:lp + 1])):
                            sc = lab.gender_scores(lab.lens(hh, l, "jpp"))
                            for k in ("pron", "noun"):
                                prd[f"{w}_jpp_{k}"][n, li] = sc[k].float().mean().cpu()
            return hook
        try:
            for l in LAYERS:
                hs.append(lab.layers[l].register_forward_hook(mk2(l)))
            with torch.no_grad():
                lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
        finally:
            for h in hs:
                h.remove()
    torch.save({"cont": cont_m, "last": last_m, **prd, "items": pitems, "layers": LAYERS}, f"{V3_CACHE}/prose.pt")
    print("prose done", PN, round(time.time() - t0, 1))
result = {"group": V3_GROUP, "N": N, "elapsed": time.time() - t0, "jpp_pron_self": summary,
          "nself_mean": sum(nself) / len(nself), "ntok_samp": sum(v.shape[0] for v in [torch.cat(tok_samp[COV_LAYERS[0]], 0)])}
