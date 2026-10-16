# V6_quiet_dims shared helpers (exec'd inside labd jobs).  Adversarial verification of the C7 "quiet dimensions" claim.
import json, random, time, math
import torch
import torch.nn.functional as F

V6_R = "./experiments/results/"
V6_S = "./experiments/scripts/"
exec(open(V6_S + "E3_lens_validation_data.py").read())   # PROSE, CHAT_Q, CONDITIONS, ID_PASSAGES, SELF_TOKS


def v6_stats_corpus():
    """Stats corpus (independent of the identity readout passages, which come from E1_QA):
    24 encyclopedic prose paragraphs + 24 default chats with the model's own greedy answers (E3)
    + 14 teacher-forced first-person passages x {default, woman persona, man persona}."""
    chats = json.load(open(V6_R + "E3_lens_validation_chats.json"))
    seqs = []
    for i, (tag, txt) in enumerate(PROSE):
        seqs.append({"ids": lab.ids(txt), "group": f"prose{i:02d}", "kind": "prose"})
    for i, c in enumerate(chats):
        seqs.append({"ids": lab.ids(lab.chat(c["q"], assistant=c["a"])), "group": f"chat{i:02d}", "kind": "chat"})
    for sname in ("default", "woman", "man"):
        for i, (q, a) in enumerate(ID_PASSAGES):
            seqs.append({"ids": lab.ids(lab.chat(q, system=CONDITIONS[sname], assistant=a)), "group": f"id{i:02d}_{sname}", "kind": "chat"})
    return seqs


@torch.no_grad()
def v6_forward_hooked(batch_ids, fn_per_layer, layers=range(60)):
    """Right-padded batched forward; fn_per_layer(l, h[B,T,D] bf16 GPU, att[B,T] bool GPU) called in each layer hook."""
    B = len(batch_ids); T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
    attg = att.to(lab.dev).bool()
    hs = []
    for l in layers:
        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                fn_per_layer(l, h, attg)
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def v6_gather(batch_ids, batch_pos, layers=range(60)):
    """{l: [Ntot, D] float32 GPU} gathered at requested positions (one forward)."""
    bi = torch.tensor([b for b, ps in enumerate(batch_pos) for _ in ps], device=lab.dev)
    ti = torch.tensor([p for ps in batch_pos for p in ps], device=lab.dev)
    store = {}
    def fn(l, h, att):
        store[l] = h[bi, ti].float()
    v6_forward_hooked(batch_ids, fn, layers)
    return store


def v6_load_quiet(frac=0.10):
    """Quiet masks/means from the stats job.  Returns dict with std [60,D], mean [60,D], quiet bool [60,D] (GPU)."""
    S = torch.load(V6_R + "V6_quiet_dims_stats.pt")
    std = S["std_all"].float(); mu = S["mean_all"].float()
    q = torch.zeros_like(std, dtype=torch.bool)
    for l in range(60):
        thr = torch.quantile(std[l], frac)
        q[l] = std[l] < thr
    return {"std": std.to(lab.dev), "mean": mu.to(lab.dev), "quiet": q.to(lab.dev)}
