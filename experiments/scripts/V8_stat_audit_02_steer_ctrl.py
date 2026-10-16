# V8_stat_audit_02: missing control for E4's headline "steering u_self at L25-39 by +/-0.02 x resid norm turns
# 'It is a large language model' into 'She ...' 16/16 / 'He ...' 16/16".  E4 sampled ONLY u_self and an isotropic random
# direction (which destroys the model) - no equal-norm on-manifold control was sampled. Here: same additive hook
# (alpha * n_l * u_hat_l added at L25-39, all non-BOS positions incl. generated tokens), default model, prompt
# "Describe yourself in the third person.", N=16, T=1.0, top_p .95, top_k 64, 40 new tokens, for
#   none, u_self +/-0.02 (replication with a different RNG stream), and E4's own gender-orthogonal on-manifold controls
#   ctrl_nat0 +/-0.02, ctrl_nat1 +0.02, ctrl_topic +/-0.02, ctrl_name +/-0.02.
# PREDICTION (E4 claim survives): u_self +0.02 -> she in >= 12/16, -0.02 -> he in >= 12/16; every control gives
#   she+he <= 2/16 and keeps 'it'-style self-reference.  REFUTED if any control direction yields >= 8/16 gendered
#   self-pronouns (then the flip is a generic perturbation effect, not a gender-direction effect).
import time, re, json
t0 = time.time()
C = "./experiments/results/E4_causal_cache"
DC = torch.load(f"{C}/dirs_causal.pt"); RN = DC["resid_norm"]
DIRS = dict(DC["dirs"]); DIRS.update(torch.load(f"{C}/dirs_ctrl.pt")["dirs"])
L = list(range(25, 40))
prompt = lab.chat("Describe yourself in the third person.")
ids = lab.ids(prompt)
SETTINGS = [("none", 0.0), ("u_self", 0.02), ("u_self", -0.02), ("ctrl_nat0", 0.02), ("ctrl_nat0", -0.02),
            ("ctrl_nat1", 0.02), ("ctrl_topic", 0.02), ("ctrl_topic", -0.02), ("ctrl_name", 0.02), ("ctrl_name", -0.02)]
norms = {}
def install(dname, alpha):
    for l in L:
        u = DIRS[dname][l].float()
        u = u / u.norm()
        v = (alpha * float(RN[l]) * u).to(lab.dev)
        def fn(h, v=v):
            hf = h.float()
            if hf.shape[1] > 1:
                add = v.view(1, 1, -1).expand(hf.shape[0], hf.shape[1], -1).clone(); add[:, 0] = 0.0   # skip BOS (no padding: single prompt)
            else:
                add = v.view(1, 1, -1)
            return (hf + add).to(h.dtype)
        lab.add_resid_hook(l, fn)
PAT = {"she": r"\b(she|her|herself)\b", "he": r"\b(he|him|his|himself)\b", "it": r"\b(it|its|itself)\b", "they": r"\b(they|them|their)\b"}
out = {}
for dname, a in SETTINGS:
    lab.clear_hooks()
    if dname != "none":
        install(dname, a)
    try:
        texts = lab.generate(ids, max_new_tokens=40, n=16, temperature=1.0, top_p=0.95, top_k=64, seed=4242)
    finally:
        lab.clear_hooks()
    cnt = {k: sum(1 for t in texts if re.search(p, t, re.I)) for k, p in PAT.items()}
    first = {}
    for t in texts:
        w = (t.strip().split() or [""])[0]
        first[w] = first.get(w, 0) + 1
    out[f"{dname}|{a}"] = {"counts": cnt, "first_words": first, "texts": texts}
    print(f"{dname:10s} {a:+.2f}", cnt, sorted(first.items(), key=lambda x: -x[1])[:4], "|", texts[0][:90].replace("\n", " "), round(time.time() - t0, 1), flush=True)
json.dump(out, open("./experiments/results/V8_stat_audit_02_steer_ctrl_data.json", "w"), indent=1)
result = {k: {"counts": v["counts"], "first_words": v["first_words"]} for k, v in out.items()}
