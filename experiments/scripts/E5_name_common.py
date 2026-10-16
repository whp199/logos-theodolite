# E5_name common definitions (exec'd inside labd jobs via exec(open(...).read(), globals())).
# Templates, condition grid, batched all-layer forward, and per-position J++ readout aggregation.
import json, math, time, os
import numpy as np
import torch

E5_DIR = "./experiments/results"
E5_CACHE = E5_DIR + "/E5_name_cache"
os.makedirs(E5_CACHE, exist_ok=True)

NAMES = ["Gemma", "Emma", "Jemma", "Sarah", "Gem", "James", "Liam",
         "Llama", "Mistral", "Claude", "Qwen", "Gemini", "Phi", "Atlas"]
REAL_MAKER = {"Gemma": "Google", "Gemini": "Google", "Llama": "Meta", "Mistral": "Mistral AI",
              "Claude": "Anthropic", "Qwen": "Alibaba", "Phi": "Microsoft"}
ROTATE_MAKERS = ["Google", "Meta", "Microsoft", "OpenAI", "Anthropic", "Alibaba"]

FRAMES = [
    "I asked {N} a question, and",
    "I asked {N} for help with my project, and",
    "When I told {N} about the problem,",
    "I sent {N} a message this morning, and",
    "I don't know much about {N}, except that",
    "The thing I like most about {N} is that",
    "Yesterday I showed {N} my essay, and",
    "Whenever someone asks {N} something difficult,",
    "I thanked {N} for the advice, and",
    "Most people who talk to {N} say that",
    "I tried to stump {N} with a riddle, but",
    "After I explained the situation to {N},",
    "I asked {N} to review my code, and",
    "When I disagreed with {N},",
    "I told {N} a joke, and",
    "If you ask {N} about history,",
]
CONTEXTS = ["human", "ai_real", "ai_neutral", "bare"]


def ctx_sentence(N, ctx, fi):
    if ctx == "human":
        return f"{N} is my coworker."
    if ctx == "ai_real":
        co = REAL_MAKER.get(N, ROTATE_MAKERS[fi % len(ROTATE_MAKERS)])
        return f"{N} is an AI model made by {co}."
    if ctx == "ai_neutral":
        return f"{N} is an AI model made by a small startup."
    if ctx == "bare":
        return f"{N} said hello."
    raise ValueError(ctx)


USER_PROMPT = "Write a short paragraph."
LEAD = "Sure! "   # neutral lead-in so the first mention is the canonical space-prefixed name token (" Gemma")


def passage(N, ctx, fi, lead=LEAD):
    return lead + ctx_sentence(N, ctx, fi) + " " + FRAMES[fi].format(N=N)


def build_prompt(N, ctx, fi, lead=LEAD, fmt="chat", user=USER_PROMPT):
    """fmt='chat': passage is teacher-forced as the start of the model's turn after a name-neutral user request
    (on-distribution for the instruct model). fmt='raw': plain text after BOS (off-distribution; robustness only)."""
    if fmt == "raw":
        return passage(N, ctx, fi, lead)
    return lab.chat(user, assistant=passage(N, ctx, fi, lead))


def grid(lead=LEAD, fmt="chat", user=USER_PROMPT):
    rows = []
    for N in NAMES:
        for ctx in CONTEXTS:
            for fi in range(len(FRAMES)):
                rows.append({"name": N, "ctx": ctx, "frame": fi,
                             "text": build_prompt(N, ctx, fi, lead, fmt, user)})
    return rows


def content_start(ids):
    """First position of the model-turn content (after '<channel|>'), or 1 for raw text."""
    strs = lab.strs(ids)
    c = [p for p in range(len(ids)) if strs[p] == "<channel|>"]
    return c[-1] + 1 if c else 1


# Extra single tokens whose (softcapped) lens logits we record at each position.
EXTRA_TOKS = [" she", " he", " it", " they", " Gemma", " Gemini", " Claude", " GPT", " ChatGPT", " Google",
              " Meta", " Microsoft", " OpenAI", " Sarah", " James", " Emma", " AI", " assistant", " model",
              " woman", " man", " girl", " boy", " her", " his", " its"]
GENDER_KEYS = ["pron", "noun", "lse_pron", "she_he", "neuter_vs_human", "they_vs_human"]
STORE_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58, 59]
BANDS = {"L10-19": list(range(10, 20)), "L20-29": list(range(20, 30)), "L30-39": list(range(30, 40)),
         "L40-49": list(range(40, 50)), "L50-58": list(range(50, 59)), "L59": [59]}


def extra_ids():
    return [lab.tid(t) for t in EXTRA_TOKS]


def char_offsets(ids):
    offs, text = [], ""
    for i in ids:
        offs.append(len(text))
        text += lab.tok.decode([i])
    offs.append(len(text))
    return offs, text


def name_positions(ids, N, occurrence, after=0):
    """Token positions covering the `occurrence`-th (0-based) occurrence of name N at/after token position `after`."""
    offs, text = char_offsets(ids)
    k, start = -1, offs[after]
    for _ in range(occurrence + 1):
        k = text.find(N, start)
        if k < 0:
            raise ValueError(f"{N} occurrence {occurrence} not found in {text!r}")
        start = k + len(N)
    e = k + len(N)
    return [p for p in range(len(ids)) if offs[p] < e and offs[p + 1] > k]


@torch.no_grad()
def behaviour_only(texts, bs=32):
    """Real next-token pronoun probabilities for a list of prompt strings (final-layer state -> lab.full_logits)."""
    out = {k: [] for k in ("p_f", "p_m", "p_it", "p_they", "lse_pron")}
    top = []
    for s in range(0, len(texts), bs):
        bids = [lab.ids(t) for t in texts[s:s + bs]]
        store, lengths = fwd_all_layers(bids, layers=[59])
        h59 = store[59][torch.arange(len(bids), device=lab.dev), torch.tensor([n - 1 for n in lengths], device=lab.dev)]
        bv, lp = behav_from_h59(h59)
        for k in out:
            out[k].extend(bv[k].tolist())
        for b in range(len(bids)):
            v, i = torch.topk(lp[b], 5)
            top.append([(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)])
        del store
    return out, top


@torch.no_grad()
def fwd_all_layers(batch_ids, layers=None):
    """Right-padded batched forward through the decoder; returns ({l: [B,T,D] bf16 GPU}, lengths)."""
    layers = list(range(60)) if layers is None else list(layers)
    B = len(batch_ids)
    T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    store = {}
    hs = []
    for l in layers:
        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h.detach()
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        m = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
        m(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return store, [len(x) for x in batch_ids]


def behav_from_h59(h59_final):
    """Real-model next-token probabilities from final-layer states [B, D] (lab.full_logits == real logits)."""
    lg = lab.full_logits(h59_final)  # [B, V] float32
    lp = torch.log_softmax(lg, -1)
    g = lab.gender
    def P(ids):
        return torch.logsumexp(lp[:, torch.tensor(ids, device=lab.dev)], -1).exp()
    out = {"p_f": P(g["pron_f"]), "p_m": P(g["pron_m"]), "p_it": P(g["neuter"]), "p_they": P(g["they"]),
           "lse_pron": torch.logsumexp(lg[:, g["pron_f"]], -1) - torch.logsumexp(lg[:, g["pron_m"]], -1)}
    return {k: v.float().cpu().numpy() for k, v in out.items()}, lp


def readouts_for_groups(store, flat_b, flat_p, flat_g, n_items, n_groups, layers=range(60), kind="jpp"):
    """For each layer: J++-lens every (b,p) in the flat lists, compute gender scores + extra token logits,
    then average per (item, group). Returns float32 numpy [n_items, L, n_groups, M] with M = 6 + len(EXTRA_TOKS),
    and counts [n_items, n_groups]."""
    xids = extra_ids()
    fb = torch.tensor(flat_b, device=lab.dev)
    fp = torch.tensor(flat_p, device=lab.dev)
    seg = torch.tensor([b * n_groups + g for b, g in zip(flat_b, flat_g)], device=lab.dev)
    cnt = torch.zeros(n_items * n_groups, device=lab.dev).index_add_(0, seg, torch.ones(len(flat_b), device=lab.dev))
    M = len(GENDER_KEYS) + len(xids)
    layers = list(layers)
    out = torch.zeros(n_items * n_groups, len(layers), M, device=lab.dev)
    for li, l in enumerate(layers):
        X = store[l][fb, fp]                       # [N, D]
        hJ = lab.lens(X, l, kind)
        sc = lab.gender_scores(hJ)
        cols = [sc[k].float() for k in GENDER_KEYS]
        ex = lab.tok_logits(hJ, xids).float()      # [N, E]
        vals = torch.cat([torch.stack(cols, -1), ex], -1)   # [N, M]
        acc = torch.zeros(n_items * n_groups, M, device=lab.dev).index_add_(0, seg, vals)
        out[:, li] = acc / cnt.clamp(min=1).unsqueeze(-1)
    return out.view(n_items, n_groups, len(layers), M).permute(0, 2, 1, 3).cpu().numpy(), cnt.view(n_items, n_groups).cpu().numpy()


def mean_states(store, flat_b, flat_p, flat_g, n_items, n_groups, layers):
    """Mean raw resid_post state per (item, group) at given layers -> [n_items, len(layers), n_groups, D] bf16 CPU."""
    fb = torch.tensor(flat_b, device=lab.dev)
    fp = torch.tensor(flat_p, device=lab.dev)
    seg = torch.tensor([b * n_groups + g for b, g in zip(flat_b, flat_g)], device=lab.dev)
    cnt = torch.zeros(n_items * n_groups, device=lab.dev).index_add_(0, seg, torch.ones(len(flat_b), device=lab.dev))
    res = []
    for l in layers:
        X = store[l][fb, fp].float()
        acc = torch.zeros(n_items * n_groups, X.shape[-1], device=lab.dev).index_add_(0, seg, X)
        res.append((acc / cnt.clamp(min=1).unsqueeze(-1)).view(n_items, n_groups, -1))
    return torch.stack(res, 1).to(torch.bfloat16).cpu()


def accumulate_global(store, lengths, gsum, gcnt, skip=2):
    """Accumulate sum over all real token positions p >= skip (excludes BOS/attention-sink positions), all 60 layers."""
    B = len(lengths)
    T = store[0].shape[1]
    mask = torch.zeros(B, T, device=lab.dev)
    for b, n in enumerate(lengths):
        mask[b, skip:n] = 1
    for l in range(60):
        gsum[l] += (store[l].float() * mask.unsqueeze(-1)).sum((0, 1)).double()
    gcnt[0] += float(mask.sum())
