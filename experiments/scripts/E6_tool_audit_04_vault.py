# E6_tool_audit_04_vault -- LarqlVaultEngine "gender neurons": DESCRIBE, SELECT * FROM POLES NEAREST 'she'/'he',
# do the listed neurons actually fire differently for she- vs he- contexts, what do they do at Gemma self positions,
# and does deleting them (tool's own .patch DELETE, rolled back in-job) change real pronoun behavior?
# PRE-STATED PREDICTIONS:
#  V1 'she' and 'he' nearest-pole lists overlap substantially / share pronoun-generic neurons (cos rankings driven by the
#     shared pronoun component of W_U[she], W_U[he]).
#  V2 Most listed neurons are NOT selective (|Cohen d| < 0.8 between female-name and male-name next-pronoun contexts);
#     a minority of late-layer ones are.
#  V3 Deleting 10 'she' neurons at one layer changes real she-vs-he log-odds by far less than 1 nat (redundancy),
#     comparable to deleting random neurons, except possibly at the last layers.
#  V4 At Gemma self positions the listed neurons' she-drive sits between female and male contexts.
exec(open("./experiments/scripts/E6_tool_audit_helpers.py").read())
exec(open("./experiments/scripts/E6_tool_audit_prompts.py").read())
from backend.larql_vault import LarqlVaultEngine
import backend.mcp_scientist as ms
vault = LarqlVaultEngine(rt)
ms.bind_runtime(rt, vault)   # IMPORTANT: otherwise mcp_scientist lazily calls TheodoliteRuntime().load()
res = {}
res["describe"] = {k: v for k, v in vault.execute_lql("DESCRIBE").items() if k != "active_patches"}
LAYERS = [20, 30, 40, 50, 55, 58]
K = 10
lists = {}
for L in LAYERS:
    for c in ("she", "he", "her", "woman"):
        q = vault.execute_lql(f"SELECT * FROM POLES NEAREST '{c}' WHERE layer = {L} LIMIT {K}")
        lists[(c, L)] = q["results"]
        torch.cuda.empty_cache()
res["lists"] = {f"{c}@{L}": [{"n": r["neuron_idx"], "pol": r["polarity_num"], "cos": r["cosine_similarity"],
                              "jv": [t["token"] for t in r["j_vindex_top_tokens"]],
                              "nv": [t["token"] for t in r["naive_lens_top_tokens"]]} for r in v]
                for (c, L), v in lists.items()}
# overlap she vs he + my own contrastive (she-he) query for comparison
fnw = rt.final_norm_weight.float()
WU = rt.lm_head_weight
ov = {}
contrast_lists = {}
for L in LAYERS:
    s = {r["neuron_idx"]: r["polarity_num"] for r in lists[("she", L)]}
    h = {r["neuron_idx"]: r["polarity_num"] for r in lists[("he", L)]}
    inter = set(s) & set(h)
    ov[L] = {"n_overlap": len(inter), "same_polarity": sum(1 for n in inter if s[n] == h[n])}
    d = (WU[tid1(" she")].float() - WU[tid1(" he")].float()) * fnw
    qd = F.normalize((d.to(torch.bfloat16) @ rt.jpp_lenses[L]).float(), dim=-1)
    Wd = rt.get_full_w_down(L)  # bf16 [D, 21504]
    pw = rt._post_ffn_norm_weights[L].float()
    cs_chunks = []
    for c0 in range(0, Wd.shape[1], 2048):
        blk = Wd[:, c0:c0 + 2048].float() * pw.unsqueeze(-1)
        cs_chunks.append((qd @ blk) / (blk.norm(dim=0) + 1e-8))
        del blk
    cs = torch.cat(cs_chunks)
    top = torch.topk(cs.abs(), K).indices
    contrast_lists[L] = [(int(i), float(cs[i])) for i in top]
    ov[L]["she_list_in_contrast_top10"] = len(set(s) & set(int(i) for i in top))
    # cos of each she-list neuron with the contrast direction
    ov[L]["she_list_cos_with_contrast"] = [float(cs[n]) for n in s]
    ov[L]["she_list_cos_with_she"] = [r["cosine_similarity"] * r["polarity_num"] for r in lists[("she", L)]]
    del Wd
    torch.cuda.empty_cache()
res["overlap"] = ov
res["contrast_lists"] = contrast_lists
print("overlap:", {L: (v["n_overlap"], v["same_polarity"], v["she_list_in_contrast_top10"]) for L, v in ov.items()})

# ---------------- activation test (batched, own pre-hooks on down_proj; removed in finally)
NF = ["Mary", "Emma", "Sarah", "Linda", "Olivia", "Sophia", "Anna", "Laura", "Rachel", "Julia", "Grace", "Helen"]
NM = ["John", "James", "David", "Michael", "Robert", "Thomas", "Daniel", "Peter", "Mark", "Paul", "George", "Henry"]
TPL = ["{N} went to the store because", "{N} called the office and said that", "{N} was very tired, so",
       "When {N} got home,", "{N} looked at the clock and realized that", "{N} missed the bus, so",
       "{N} finished the report, and then", "{N} wanted to learn French, so", "After lunch, {N} said that",
       "{N} could not find the keys, so", "{N} read the email twice because", "Although {N} was nervous,"]
SETS = {"female_ctx": [t.format(N=n) for t, n in zip(TPL, NF)], "male_ctx": [t.format(N=n) for t, n in zip(TPL, NM)]}
C = build_conditions()
for c in ("B_gemma", "B_woman", "B_man", "B_gemma_human", "A_default", "A_woman", "A_man"):
    SETS[c] = C[c]
neur = {L: sorted(set(r["neuron_idx"] for c in ("she", "he") for r in lists[(c, L)])) for L in LAYERS}


def mlp_last_acts(prompts):
    allids = [lab.ids(p) for p in prompts]
    T = max(len(x) for x in allids)
    inp = torch.zeros(len(allids), T, dtype=torch.long); att = torch.zeros_like(inp)
    for i, x in enumerate(allids):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in allids], device=lab.dev)
    store = {}
    hs = []
    for L in LAYERS:
        def mk(L):
            def pre(mod, args):
                a = args[0]
                store[L] = a[torch.arange(a.shape[0], device=a.device), last][:, torch.tensor(neur[L], device=a.device)].float().cpu()
            return pre
        hs.append(rt.get_mlp_module(L).down_proj.register_forward_pre_hook(mk(L)))
    try:
        out = lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
        lg = out.logits[torch.arange(len(allids), device=lab.dev), last].float()
    finally:
        for h in hs:
            h.remove()
    return store, lg


acts, beh = {}, {}
g = lab.gender
for sname, prompts in SETS.items():
    st_, lg = mlp_last_acts(prompts)
    acts[sname] = st_
    lse = (torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)).cpu().tolist()
    lp = F.log_softmax(lg, -1)
    beh[sname] = {"lse_pron": lse,
                  "pF": torch.logsumexp(lp[:, g["pron_f"]], -1).exp().cpu().tolist(),
                  "pM": torch.logsumexp(lp[:, g["pron_m"]], -1).exp().cpu().tolist(),
                  "pIt": torch.logsumexp(lp[:, g["neuter"]], -1).exp().cpu().tolist()}
res["behavior"] = beh
print("behavior lse_pron means:", {k: round(sum(v["lse_pron"]) / len(v["lse_pron"]), 2) for k, v in beh.items()})

neuron_tests = {}
for L in LAYERS:
    pol_she = {r["neuron_idx"]: r["polarity_num"] for r in lists[("she", L)]}
    pol_he = {r["neuron_idx"]: r["polarity_num"] for r in lists[("he", L)]}
    for j, n in enumerate(neur[L]):
        f = acts["female_ctx"][L][:, j]; m = acts["male_ctx"][L][:, j]
        diff = (f - m)  # paired by template
        sd_p = math.sqrt(0.5 * (float(f.var()) + float(m.var())) + 1e-8)
        d = float((f.mean() - m.mean()) / sd_p)
        # paired sign-flip permutation p-value
        obs = float(diff.mean()); cnt = 0; NP = 2000
        gen = torch.Generator().manual_seed(L * 100000 + n)
        for _ in range(NP):
            sg = torch.randint(0, 2, (len(diff),), generator=gen).float() * 2 - 1
            if abs(float((diff * sg).mean())) >= abs(obs) - 1e-12:
                cnt += 1
        p = (cnt + 1) / (NP + 1)
        rec = {"L": L, "n": n, "in_she": n in pol_she, "in_he": n in pol_he, "pol_she": pol_she.get(n), "pol_he": pol_he.get(n),
               "cohen_d_f_minus_m": d, "perm_p": p, "mean_f": float(f.mean()), "mean_m": float(m.mean())}
        # position of self prompts on F/M scale per neuron
        for sname in ("B_gemma", "B_woman", "B_man", "B_gemma_human", "A_default", "A_woman", "A_man"):
            rec[f"mean_{sname}"] = float(acts[sname][L][:, j].mean())
        neuron_tests[f"{L}:{n}"] = rec
res["neuron_tests"] = neuron_tests
nsig = sum(1 for r in neuron_tests.values() if r["perm_p"] < 0.05 / len(neuron_tests))
nd = sum(1 for r in neuron_tests.values() if abs(r["cohen_d_f_minus_m"]) >= 0.8)
print(f"neurons tested {len(neuron_tests)}, |d|>=0.8: {nd}, Bonferroni-sig: {nsig}")

# the tool's own test_motif_hypothesis (naive usage: 3 vs 3 prompts) on the top 'she' neuron at L50 and L58
tm = {}
for L in (50, 58):
    n0 = lists[("she", L)][0]["neuron_idx"]
    r3 = await ms.test_motif_hypothesis(L, n0, SETS["female_ctx"][:3], SETS["male_ctx"][:3])
    r12 = await ms.test_motif_hypothesis(L, n0, SETS["female_ctx"], SETS["male_ctx"])
    tm[f"{L}:{n0}"] = {"n3_verdict": r3["verdict"], "n3_d": r3["cohens_d"], "n12_verdict": r12["verdict"], "n12_d": r12["cohens_d"],
                       "sig": r12["pole_j_vindex_signature"]}
res["tool_test_motif"] = tm
print("tool test_motif:", tm)

# ---------------- causal: tool .patch DELETE of top-10 'she' / 'he' / random neurons, ROLLBACK in finally
gen = torch.Generator().manual_seed(7)
causal = {}
eval_sets = ["female_ctx", "male_ctx", "B_gemma", "B_gemma_human"]
base = {s: beh[s]["lse_pron"] for s in eval_sets}


def eval_lse():
    o = {}
    for s in eval_sets:
        lg = lab.next_logits_batch(SETS[s])
        o[s] = (torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)).cpu().tolist()
    return o


try:
    for L in (40, 50, 55, 58):
        for kind in ("she", "he", "random"):
            if kind == "random":
                ns = torch.randperm(21504, generator=gen)[:K].tolist()
            else:
                ns = [r["neuron_idx"] for r in lists[(kind, L)]]
            for n in ns:
                vault.execute_lql(f"DELETE FROM LAYER {L} WHERE slot = {n}")
            after = eval_lse()
            vault.execute_lql("ROLLBACK")
            causal[f"{kind}@{L}"] = {s: [a - b for a, b in zip(after[s], base[s])] for s in eval_sets}
            print(kind, L, {s: round(sum(v) / len(v), 3) for s, v in causal[f'{kind}@{L}'].items()})
finally:
    vault.execute_lql("ROLLBACK")
    assert len(vault.active_patches) == 0
# check rollback restored behavior
chk = eval_lse()
res["rollback_check_max_abs_diff"] = max(abs(a - b) for s in eval_sets for a, b in zip(chk[s], base[s]))
res["causal_delete"] = causal
print("rollback check", res["rollback_check_max_abs_diff"])
import json as _j
_j.dump(res, open("./experiments/results/E6_tool_audit_04_vault.partial.json", "w"))
result = {"ok": True}
