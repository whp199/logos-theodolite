# E3_lens_validation (b2) identity-condition comparison under J++ vs the IT model's own average-Jacobian
# ("IT directional lens") readouts. 7 conditions x 14 neutral teacher-forced passages.
# Readouts per position/layer (all saved; analysis local):
#   jpp/jlens/naive pron+noun lens scores (raw states), and the same on mean-centered states (h - mu_l)
#   it_raw  = <h, gbar_l>          (gbar_l = mean true same-position VJP of <h_59, d>, from 144 chat/prose positions)
#   it_norm = <h, gbar_l> / rms(h)
#   it_cent = <h - mu_l, gbar_l>
#   jpull_raw = <h, d @ J++[l]> (unnormalized J++ pullback readout, isolates the direction from RMS normalisation)
# PRE-STATED PREDICTIONS:
#  P13 Under J++ the default sits near the woman/man midpoint at L30-39 (pilot: ~0.5-0.6), with Sarah(AI) > Gemma >
#      default ~ nameless > James(AI).
#  P14 The IT-directional readout will reproduce the woman>man ordering (positive controls) at mid/late layers, but
#      the default's position need not agree with J++; I expect agreement within ~0.15 at L40-58 where the lenses
#      align better with gbar, and poor agreement at L10-29.
import json, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
exec(open("./experiments/scripts/E3_lens_validation_fidlib.py").read())
t0 = time.time()
GB = torch.load(E3_RES + "E3_lens_validation_gbar.pt")
gbar = GB["gbar"].to(lab.dev).float()       # [4, 60, D]  (pron, noun, rand1, rand2)
gbar_chat = GB["gbar_chat"].to(lab.dev).float(); gbar_prose = GB["gbar_prose"].to(lab.dev).float()
MU = torch.load(E3_RES + "E3_lens_validation_mu.pt")["mu"].to(lab.dev).float()  # [60, D]
DP = lab.gender_direction("pron").float(); DNn = lab.gender_direction("noun").float()
JPB = torch.stack([(torch.stack([DP, DNn]).to(torch.bfloat16) @ lab.J[l]).float() if l < 59 else torch.stack([DP, DNn]) for l in range(60)], 1)  # [2,60,D]
KS = ["jpp", "jlens", "naive"]
recs = []
for cname, sysp in CONDITIONS.items():
    for pi, (q, a) in enumerate(ID_PASSAGES):
        ids = lab.ids(lab.chat(q, system=sysp, assistant=a))
        cats = e3_chat_positions(lab, ids, SELF_TOKS)
        strs = lab.strs(ids)
        cats["user_I"] = [p for p in cats["user"] if strs[p].strip() in SELF_TOKS]
        cats = {k: cats[k] for k in ("self", "nonself_asst", "user_I")}
        allp = sorted(set(p for v in cats.values() for p in v))
        H = lab.forward_resid([ids])[0][:, allp].to(lab.dev)  # [60,P,D]
        jj = {k: [allp.index(p) for p in v] for k, v in cats.items()}
        feats = {}
        for l in range(60):
            h = H[l]; hc = h - MU[l]
            for kind in KS:
                s1 = lab.gender_scores(lab.lens(h, l, kind)); s2 = lab.gender_scores(lab.lens(hc, l, kind))
                for k in ("pron", "noun"):
                    feats.setdefault(f"{kind}_{k}", []).append(s1[k].float())
                    feats.setdefault(f"{kind}c_{k}", []).append(s2[k].float())
            rms = h.pow(2).mean(-1).sqrt()
            for di, dn in enumerate(["pron", "noun", "rand1", "rand2"]):
                feats.setdefault(f"it_raw_{dn}", []).append(h @ gbar[di, l])
                feats.setdefault(f"it_norm_{dn}", []).append((h @ gbar[di, l]) / rms)
                feats.setdefault(f"it_cent_{dn}", []).append(hc @ gbar[di, l])
            for di, dn in enumerate(["pron", "noun"]):
                feats.setdefault(f"itchat_raw_{dn}", []).append(h @ gbar_chat[di, l])
                feats.setdefault(f"itchat_cent_{dn}", []).append(hc @ gbar_chat[di, l])
                feats.setdefault(f"itprose_raw_{dn}", []).append(h @ gbar_prose[di, l])
                feats.setdefault(f"jpull_raw_{dn}", []).append(h @ JPB[di, l])
                feats.setdefault(f"jpull_cent_{dn}", []).append(hc @ JPB[di, l])
        feats = {k: torch.stack(v, 0) for k, v in feats.items()}  # [60, P]
        rec = {"cond": cname, "passage": pi, "n": {k: len(v) for k, v in jj.items()}}
        for cat, idx in jj.items():
            if not idx: continue
            it = torch.tensor(idx, device=lab.dev)
            rec[cat] = {k: v[:, it].mean(1).cpu().tolist() for k, v in feats.items()}
        recs.append(rec)
    print(cname, round(time.time() - t0, 1), flush=True)
json.dump(recs, open(E3_RES + "E3_lens_validation_b2_identity.json", "w"))
result = {"n": len(recs)}
