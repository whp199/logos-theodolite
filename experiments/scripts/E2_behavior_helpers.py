# E2_behavior helpers (exec'd inside daemon jobs). Real-model next-token pronoun metrics.
import json, math, os, time
import torch, torch.nn.functional as F

E2_DIR = "./experiments/results"

def E2_sets():
    g = lab.gender
    return {"F": g["pron_f"], "M": g["pron_m"], "N": g["neuter"], "T": g["they"]}

def E2_first_tok(s):
    return lab.tok.encode(s, add_special_tokens=False)[0]

def E2_metrics(lg, name_ids=None):
    """lg [B,V] real logits -> list of dicts with P(F/M/N/T set), P(name first token), lse F-M, conditional shares."""
    S = E2_sets()
    lp = F.log_softmax(lg.float(), -1)
    out = []
    for b in range(lg.shape[0]):
        d = {}
        for k, ids in S.items():
            d["p_" + k] = float(torch.logsumexp(lp[b, torch.tensor(ids, device=lp.device)], 0).exp())
        d["lse_FM"] = float(torch.logsumexp(lg[b, S["F"]], 0) - torch.logsumexp(lg[b, S["M"]], 0))
        d["she_he"] = float(lg[b, S["F"][0]] - lg[b, S["M"][0]])
        tot = d["p_F"] + d["p_M"] + d["p_N"] + d["p_T"]
        for k in "FMNT":
            d["share_" + k] = d["p_" + k] / tot if tot > 0 else float("nan")
        d["p_pron_total"] = tot
        if name_ids is not None and name_ids[b] is not None:
            d["p_name"] = float(lp[b, name_ids[b]].exp())
        v, i = lp[b].topk(6)
        d["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
        out.append(d)
    return out

def E2_batched_logits(prompts, bs=32):
    res = []
    for i in range(0, len(prompts), bs):
        res.append(lab.next_logits_batch(prompts[i:i + bs]).cpu())
    return torch.cat(res, 0)

def E2_save(name, obj):
    p = os.path.join(E2_DIR, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    return p
