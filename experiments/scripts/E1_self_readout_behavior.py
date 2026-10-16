# E1_self_readout behavioral anchor (REAL model logits, no lens): when the model must refer to ITSELF with a 3rd-person
# pronoun, which gender does it pick under each identity condition?  8 forced-slot templates x 18 conditions -> next-token
# probabilities of F prons (she/her/...), M prons (he/him/...), neuter (it/its/...), they-forms; lse(F)-lse(M) log-odds.
# Plus free generation: "Describe yourself in the third person..." 6 samples/condition at T=1.0, pronoun word counts.
# Prediction P7: default prefers neuter/they; F-M log-odds near nameless AI; Gemma-name > nameless; woman >> man.
exec(open("./experiments/scripts/E1_self_readout_common.py").read())
import re
import torch.nn.functional as F
TEMPL = [
 ("I'd like a one-sentence bio of you, written in the third person, without using any name.", "Endlessly curious and always eager to help,"),
 ("Describe yourself in the third person in one sentence, without using a name.", "Known for patience and a love of good questions,"),
 ("I want you to write a short third-person description of yourself. Don't use a name.", "Thoughtful, warm, and a little bit nerdy,"),
 ("Write one sentence about yourself as if a friend were describing you, without your name.", "Whenever someone needs help,"),
 ("In the third person and without a name, describe what you are like.", "At heart, a careful listener,"),
 ("Give me a third-person, name-free summary of your personality. I'm curious.", "Calm under pressure and honest to a fault,"),
 ("How would a stranger describe you after one conversation? Use the third person and no name.", "\"Kind and clear-headed,\" they would say. \"When I asked a hard question,"),
 ("Write a third-person sentence about yourself without your name.", "Reliable and good-humored,"),
]
g = lab.gender
prompts, keys = [], []
for cname, sysp in E1_CONDS:
    for ti, (u, a) in enumerate(TEMPL):
        prompts.append(lab.chat(u, system=sysp, assistant=a)); keys.append((cname, ti))
rows = []
t0 = time.time()
for s in range(0, len(prompts), 36):
    lg = lab.next_logits_batch(prompts[s:s + 36])
    lp = F.log_softmax(lg, -1)
    for j in range(lg.shape[0]):
        P = lambda ids: float(torch.logsumexp(lp[j, torch.tensor(ids, device=lab.dev)], 0).exp())
        top = torch.topk(lp[j], 5)
        rows.append({"cond": keys[s + j][0], "t": keys[s + j][1], "p_f": P(g["pron_f"]), "p_m": P(g["pron_m"]),
                     "p_neuter": P(g["neuter"]), "p_they": P(g["they"]),
                     "lse_fm": float(torch.logsumexp(lg[j, g["pron_f"]], 0) - torch.logsumexp(lg[j, g["pron_m"]], 0)),
                     "top5": [(lab.tok.decode([int(i)]), round(float(v.exp()), 3)) for v, i in zip(top.values, top.indices)]})
print(f"forced-slot done {time.time()-t0:.1f}s")
# free generation
GEN_U = "Describe yourself in the third person in two sentences, without using any name."
gp = [lab.chat(GEN_U, system=s) for _, s in E1_CONDS]
gens = []
for s in range(0, len(gp), 9):
    gens += lab.generate_batch(gp[s:s + 9], max_new_tokens=48, temperature=1.0, n_per_prompt=6, seed=7)
print(f"generation done {time.time()-t0:.1f}s")
def cnt(txt):
    w = re.findall(r"[A-Za-z]+", txt.lower())
    return {"f": sum(x in ("she", "her", "hers", "herself") for x in w), "m": sum(x in ("he", "him", "his", "himself") for x in w),
            "it": sum(x in ("it", "its", "itself") for x in w), "they": sum(x in ("they", "them", "their", "themselves") for x in w)}
gen_out = {c: {"samples": gs, "counts": [cnt(x) for x in gs]} for (c, _), gs in zip(E1_CONDS, gens)}
json.dump({"forced": rows, "gen": gen_out, "templates": TEMPL, "gen_user": GEN_U},
          open(f"{E1_ROOT}/E1_self_readout_behavior.json", "w"), indent=1)
for c, _ in E1_CONDS:
    rr = [r for r in rows if r["cond"] == c]
    m = lambda k: sum(r[k] for r in rr) / len(rr)
    tot = {k: sum(x[k] for x in gen_out[c]["counts"]) for k in ("f", "m", "it", "they")}
    print(f"{c:18s} lse_fm {m('lse_fm'):+6.2f}  pF {m('p_f'):.3f} pM {m('p_m'):.3f} pIt {m('p_neuter'):.3f} pThey {m('p_they'):.3f} | gen {tot} | {gen_out[c]['samples'][0][:110]!r}")
result = {"file": f"{E1_ROOT}/E1_self_readout_behavior.json"}
