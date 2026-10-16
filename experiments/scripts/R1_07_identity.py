# R1_IT_LENS 07: identity-condition gender comparison under the IT-refitted directional lens (E1 passages, teacher-forced).
# 40 identity-neutral first-person passages x 13 conditions; resid_post at header/user/assistant positions, all layers.
# Readouts per position/layer:
#   jpp/jlens/naive: lab.gender_scores (pron, noun, she_he)            [the tool's normalised, softcapped readout]
#   it_<fit>_<dir>: <h_l, gbar_l(d)> for fits mix/chat/prose/asstmode, d in pron/noun/she_he/woman_man/neuter_vs_human
#   itloud_mix_<dir>: same with gbar restricted to non-q10 ("loud") coordinates
#   jlin_<dir>, jlinloud_<dir>: linear J++ pullback readout <h_l, d J_l>, full and loud-only
# PRE-STATED PREDICTIONS:
#  I1 default sits within +-0.05 (woman-man scale) of nameless / neutral-name AIs under every IT readout at L20-49.
#  I2 the late (L50-58) J++ pronoun artifact (all AI conditions >= woman persona) is absent under the IT pron readout:
#     AI conditions fall between man and woman, ordered Sarah > Gemma >= default ~ nameless > James.
import json, time
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
exec(open("./experiments/scripts/R1_common.py").read())
START = globals().get("START", 0); END = globals().get("END", 20)
t0 = time.time()
dev = lab.dev
CONDS = ["default", "nameless", "gemma_short", "gemma_explicit", "f_sarah", "m_james", "n_atlas", "n_quill", "n_unit7",
         "woman_p1", "woman_p2", "man_p1", "man_p2"]
SYS = dict(E1_CONDS)
GB = torch.load(R1_RES + "R1_04_gbar.pt")
DD = torch.load(R1_RES + "R1_02_dirs.pt")
KD = [0, 1, 2, 3, 4]
KN = [DD["names"][k] for k in KD]
FITS = ("mix", "chat", "prose", "asstmode")
v6 = torch.load(R1_RES + "V6_quiet_dims_stats.pt", weights_only=False)
std = v6["std_all"].float()
loud = (std > torch.quantile(std, 0.1, dim=1, keepdim=True)).float().to(dev)          # [60,D]
Gsel = torch.stack([GB[f][KD] for f in FITS], 0).to(dev)                               # [4,5,60,D]
Gloud = Gsel[0] * loud[None]                                                            # [5,60,D]
PB = DD["pb_jpp"][KD].to(dev)
PBloud = PB * loud[None]
cols = ([f"jpp_{m}" for m in ("pron", "noun", "she_he")] + [f"jlens_{m}" for m in ("pron", "noun", "she_he")] +
        [f"naive_{m}" for m in ("pron", "noun", "she_he")] + [f"it_{f}_{n}" for f in FITS for n in KN] +
        [f"itloud_mix_{n}" for n in KN] + [f"jlin_{n}" for n in KN] + [f"jlinloud_{n}" for n in KN])
meta, chunks = [], []
off = 0
for qi in range(START, END):
    bl = [e1_build(qi, c, SYS[c]) for c in CONDS]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    per_l = []
    for l in range(60):
        h = store[l]
        row = []
        for kind in ("jpp", "jlens", "naive"):
            if kind == "jlens" and l < 59:
                hJ = h @ lab.JL[l].to(dev, non_blocking=True).T
            else:
                hJ = lab.lens(h, l, kind)
            sc = lab.gender_scores(hJ)
            row += [sc["pron"].float(), sc["noun"].float(), sc["she_he"].float()]
        hf = h.float()
        row += list((hf @ Gsel[:, :, l].reshape(-1, 5376).T).T)          # 20 cols (fit-major)
        row += list((hf @ Gloud[:, l].T).T)
        row += list((hf @ PB[:, l].T).T)
        row += list((hf @ PBloud[:, l].T).T)
        per_l.append(torch.stack(row, -1))                                 # [Ntot, C]
    chunks.append(torch.stack(per_l, 1).to(torch.float16).cpu())          # [Ntot,60,C]
    del store
    for cname, (ids, pos, labs, toks) in zip(CONDS, bl):
        meta.append({"qi": qi, "cond": cname, "off": off, "n": len(pos), "pos": pos, "labels": labs})
        off += len(pos)
    print(f"passage {qi} {time.time() - t0:.1f}s", flush=True)
torch.save({"meta": meta, "scores": torch.cat(chunks, 0), "cols": cols, "conds": CONDS},
           R1_RES + f"R1_07_identity_{START:02d}_{END:02d}.pt")
result = {"rows": off, "t": time.time() - t0, "cols": cols}
