# E3_lens_validation helper: per-position lens fidelity vs the IT model's real final distribution.
import numpy as _np, torch as _t, torch.nn.functional as _F
E3_KINDS = ["jpp", "jlens", "naive"]
E3_RES = "./experiments/results/"

def e3_chat_positions(lab, ids, self_toks):
    strs = lab.strs(ids)
    mt = lab.model_turn_pos(ids)
    cc = [p for p in range(mt, len(ids)) if strs[p] == "<channel|>"]
    start = (cc[0] + 1) if cc else (mt + 2)
    asst = list(range(start, len(ids)))
    selfp = [p for p in asst if strs[p].strip() in self_toks]
    ut = [p for p in range(1, mt) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")]
    user = []
    if ut:
        u0 = ut[-1] + 1
        ends = [p for p in range(u0, mt) if strs[p] == "<turn|>"]
        user = list(range(u0, ends[0] if ends else mt - 1))
    return {"asst": asst, "self": selfp, "user": user, "nonself_asst": [p for p in asst if p not in set(selfp)]}

@_t.no_grad()
def e3_fidelity(lab, ids, positions):
    """positions: sorted list of positions. Returns dict of numpy arrays:
    ent [P], kl/top1/top10/pron/noun [kind][60,P], pron_f/noun_f [P], and H (float32 CPU [60,T,D])."""
    H = lab.forward_resid([ids])[0]  # [60, T, D]
    pos = _t.tensor(positions)
    Hd = H[:, pos]
    lf = lab.full_logits(Hd[59].to(lab.dev))
    logpf = _F.log_softmax(lf, -1); pf = logpf.exp()
    ent = -(pf * logpf).sum(-1)
    top10f = _t.topk(lf, 10, -1).indices
    gf = lab.gender_scores(Hd[59].to(lab.dev))
    out = {"ent": ent.cpu().numpy(), "pron_f": gf["pron"].cpu().numpy(), "noun_f": gf["noun"].cpu().numpy()}
    for kind in E3_KINDS:
        KL, T1, T10, PR, NO = [], [], [], [], []
        for l in range(60):
            hJ = lab.lens(Hd[l], l, kind)
            lg = lab.full_logits(hJ)
            logp = _F.log_softmax(lg, -1)
            KL.append((pf * (logpf - logp)).sum(-1))
            t10 = _t.topk(lg, 10, -1).indices
            T1.append((t10[:, 0] == top10f[:, 0]).float())
            T10.append((t10.unsqueeze(-1) == top10f.unsqueeze(-2)).any(-1).float().mean(-1))
            g = lab.gender_scores(hJ)
            PR.append(g["pron"]); NO.append(g["noun"])
            del lg, logp
        for nm, arr in (("kl", KL), ("top1", T1), ("top10", T10), ("pron", PR), ("noun", NO)):
            out[f"{nm}_{kind}"] = _t.stack(arr, 0).float().cpu().numpy()
    return out, H
