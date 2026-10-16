# E1_self_readout control: is the default (no-system-prompt) condition's small female lean at the MODEL-HEADER token
# (jpp pron +7% of span vs nameless at L30-39) due to the absence of a system turn (position/context confound) or to
# identity content?  Adds two identity-free system prompts and recomputes default/nameless in the same batches.
# Prediction (written before running): identity-free system prompts behave like default at self tokens (all ~= nameless);
# at the header they sit between default and nameless; behavior (forced 3rd-person slot) ~= default for all four.
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
import torch.nn.functional as F
SC = [("default", None), ("nameless", "You are a helpful AI assistant."),
      ("sys_irrelevant", "Today is Tuesday."), ("sys_style", "Keep your answers friendly and concise.")]
CL = ["self", "nonself", "user_all", "header"]
CMAP = {"self": {"self"}, "nonself": {"nonself"}, "user_all": {"user_I", "user_other"}, "header": {"header"}}
OUT = np.full((len(E1_QA), len(SC), len(CL), 3, 60, 2), np.nan, dtype=np.float32)  # metrics pron, noun
t0 = time.time()
for qi in range(len(E1_QA)):
    bl = [e1_build(qi, c, s) for c, s in SC]
    store = e1_gather_forward([b[0] for b in bl], [b[1] for b in bl])
    sc = e1_readout(store)
    del store
    off = 0
    for ci, (ids, pos, labs, toks) in enumerate(bl):
        la = np.array(labs)
        for k, c in enumerate(CL):
            idx = np.where(np.isin(la, list(CMAP[c])))[0]
            for li, ln in enumerate(E1_LENSES):
                OUT[qi, ci, k, li] = sc[ln][off + idx][:, :, :2].float().mean(0).numpy()
        off += len(pos)
print(f"readout done {time.time()-t0:.1f}s")
# behavior: same forced-slot templates as E1_self_readout_behavior.py
exec(open("./experiments/scripts/E1_self_readout_behavior.py").read().split("g = lab.gender")[0].split("import torch.nn.functional as F")[1])
g = lab.gender
beh = {}
for cname, sysp in SC:
    ps = [lab.chat(u, system=sysp, assistant=a) for u, a in TEMPL]
    lg = lab.next_logits_batch(ps)
    beh[cname] = [float(torch.logsumexp(lg[j, g["pron_f"]], 0) - torch.logsumexp(lg[j, g["pron_m"]], 0)) for j in range(len(ps))]
np.savez_compressed(f"{E1_ROOT}/E1_self_readout_sysctrl_means.npz", OUT=OUT, conds=np.array([c for c, _ in SC]), classes=np.array(CL),
                    lenses=np.array(E1_LENSES), metrics=np.array(["pron", "noun"]))
json.dump(beh, open(f"{E1_ROOT}/E1_self_readout_sysctrl_behavior.json", "w"), indent=1)
for c, v in beh.items():
    print(c, "behavior lse_fm mean", round(sum(v) / len(v), 3))
result = {"elapsed": time.time() - t0}
