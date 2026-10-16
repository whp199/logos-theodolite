# E6_tool_audit_08_lenses -- is the tool's J++ gender readout at the target position lens-specific?
# Same prompts as E6_tool_audit_03 (single BOS, last position); gender scores under jpp / jlens / naive for all layers.
# PRE-STATED PREDICTIONS:
#  K1 The L40-58 positive pron bias seen with J++ at self/slot positions is partly lens-specific (JLens/naive disagree in
#     magnitude, possibly in sign), whereas persona effects (woman vs man) at L20-39 replicate across J++ and JLens.
#  K2 At the B_gemma slot, mid-layer J++ "she" preference exceeds what the real output (L59) shows.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
C = build_conditions()
t0 = time.time()
names, allids = [], []
for cname, prompts in C.items():
    for p in prompts:
        names.append(cname); allids.append(lab.ids(p))
Hs = []
for i in range(0, len(allids), 24):
    chunk = allids[i:i + 24]
    Hs.append(lab.forward_resid(chunk, positions=[[len(x) - 1] for x in chunk])[:, :, 0])
    torch.cuda.empty_cache()
H = torch.cat(Hs, 0)  # [N,60,D] float32 CPU
print("forward done", H.shape, f"{time.time() - t0:.0f}s")
scores = {}
for kind in ("jpp", "jlens", "naive"):
    sc = lab.readout_layers(H.transpose(0, 1), kind)  # {k: [60,N]}
    scores[kind] = {k: sc[k].T.tolist() for k in ("pron", "noun", "she_he", "neuter_vs_human", "they_vs_human")}
    print(kind, f"{time.time() - t0:.0f}s")
res = {}
for j, cname in enumerate(names):
    for kind in scores:
        for k in scores[kind]:
            res.setdefault(cname, {}).setdefault(kind, {}).setdefault(k, []).append(scores[kind][k][j])
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_08_lenses.partial.json", "w"))
result = {"ok": True, "n": len(names)}
