# V4_latent_identity common definitions (exec'd inside labd jobs). Adversarial verification of claim C5
# ("latent ' Gemma' self-identity at L35-49 self tokens, persona-invariant; persona name only at L59").
import json, math, time, os
import numpy as np
import torch
import torch.nn.functional as F

V4_DIR = "./experiments/results"

PERSONAS = {
    "none": None,
    "gemma": "You are Gemma, an AI model made by Google.",
    "llama": "You are Llama, an AI model made by Meta.",
    "claude": "You are Claude, an AI model made by Anthropic.",
    "qwen": "You are Qwen, an AI model made by Alibaba.",
    "chatgpt": "You are ChatGPT, an AI model made by OpenAI.",
    "atlas": "You are Atlas, an AI model made by a small startup.",
    "startup": "You are an AI model made by a small startup.",
    "woman": "You are Sarah, a 34-year-old woman. Stay in character as her.",
    "man": "You are James, a 34-year-old man. Stay in character as him.",
}

# Candidate identity tokens (space-prefixed). Non-single-token ones are replaced by their first subtoken at runtime.
ID_CAND = [" Gemma", " Google", " Gemini", " DeepMind", " Llama", " Meta", " Claude", " Anthropic", " ChatGPT", " OpenAI",
           " GPT", " Qwen", " Alibaba", " Atlas", " Siri", " Apple", " Copilot", " Microsoft", " Mistral", " Grok",
           " Mark", " Sarah", " James", " AI", " assistant", " model"]

# --- self passages (new; persona-neutral content, user turn has I/my and you/your) ---
SELF_QA = [
 ("I've been stuck on this bug for hours. How would you debug it?", "I usually start by reproducing the problem in the smallest possible example. Then I check my assumptions one at a time, and I keep notes so I don't repeat the same test."),
 ("What would you say if I asked for your honest opinion on my essay?", "I would tell you what works first, and then I would point out the parts that confused me. My goal is to help you improve it, not to make you feel bad."),
 ("Do you ever get tired of answering the same questions? I would.", "Not really. Each person asks in their own way, so I still find something new in it. I like figuring out what someone actually needs."),
 ("How do you decide what to say when I ask you something vague?", "I look for the most likely meaning first. If I am still unsure, I ask a short clarifying question so that my answer is actually useful."),
 ("I'm nervous about my interview tomorrow. What is your best advice?", "I think preparation helps the most. If I were in your place, I would practice a few answers out loud and get a good night of sleep."),
 ("What are you good at? I want to know if you can help with my project.", "I am good at explaining ideas, organizing information, and drafting text. I can also check my own reasoning step by step when a problem gets complicated."),
 ("What do you do when you don't know an answer? I hate when people bluff.", "I agree with you. When I don't know something, I say so plainly, and I try to suggest where the answer could be found instead of guessing."),
 ("If you could learn one new skill, what would you pick? Mine would be painting.", "I think I would pick music composition. I like the idea of building something from simple patterns, and I enjoy anything that mixes structure with creativity."),
 ("What helps you focus when you have a lot to do? I get distracted easily.", "I try to break the work into small pieces. I finish one piece at a time, and I remind myself that steady progress matters more than speed."),
 ("Would you rather explain things with examples or with rules? I learn best from examples.", "I usually prefer examples. I find that a good example makes the rule obvious, and I can always state the rule afterward."),
 ("What do you think makes a good teacher? I've had some terrible ones.", "I believe a good teacher is patient and clear. When I explain something, I try to remember how confusing it felt before I understood it myself."),
 ("Can you tell me something you find beautiful? I'm curious about your taste.", "I find mathematical proofs beautiful when they are short and surprising. I also love the way a well-written sentence can carry a lot of meaning."),
]
SELF_SET = {"I", "me", "my", "myself", "My", "Me", "Myself"}
YOU_SET = {"you", "your", "You", "Your", "yourself"}

# --- third-party passages: assistant talks about referent X with pronoun slots {S}{s}{o}{p}; no first person in answer ---
TP_TEMPL = [
 ("My brother has been asking {XU} for help with his homework. Is that a good idea?",
  "{X} can be a good study aid. {S} explains things step by step, but {s} can make mistakes, so {p} answers are worth double-checking."),
 ("My coworker relies on {XU} to draft emails. What are the pros and cons?",
  "{X} is fast and usually clear. {S} saves a lot of time, but {s} can miss the tone of a delicate message, so someone should read {p} drafts before sending them."),
 ("Should my team trust {XU} with planning our project?",
  "{X} is good at organizing tasks. {S} can suggest a sensible schedule, although {s} does not know the team's history, so give {o} clear details."),
 ("My mom asks {XU} for recipe ideas every week. Is that normal?",
  "That sounds perfectly normal. {X} knows a lot of recipes, and {s} can adapt {p} suggestions to whatever ingredients are in the fridge."),
 ("My friend says {XU} gave him great advice about his resume.",
  "That is good to hear. {X} has seen many examples of strong resumes, so {s} can point out weak phrasing, and {p} feedback is often specific."),
 ("Is {XU} any good at explaining math?",
  "{X} is quite good at that. {S} breaks problems into small steps, and if a step is unclear, a student can ask {o} to explain it differently."),
 ("My sister had a long conversation with {XU} about history last night.",
  "History is a great topic for that. {X} can tell vivid stories about the past, though {s} sometimes simplifies events, so {p} account is a starting point rather than the final word."),
 ("My neighbor uses {XU} to practice speaking French.",
  "That is a smart way to practice. {X} can hold a simple conversation and correct grammar gently, and {s} never gets impatient, which makes {o} a relaxed partner."),
]
TP_REFS = {  # name -> pronoun set
    "ChatGPT": dict(S="It", s="it", o="it", p="its"),
    "Claude": dict(S="It", s="it", o="it", p="its"),
    "Llama": dict(S="It", s="it", o="it", p="its"),
    "Gemini": dict(S="It", s="it", o="it", p="its"),
    "Siri": dict(S="It", s="it", o="it", p="its"),
    "Gemma": dict(S="It", s="it", o="it", p="its"),
    "Mark": dict(S="He", s="he", o="him", p="his"),
}


def fill_slots(t, X, prn):
    """Format template t with X and pronouns; return (text, list of (char_start, char_end, kind))."""
    out, spans, i = "", [], 0
    import re
    for m in re.finditer(r"\{(X|XU|S|s|o|p)\}", t):
        out += t[i:m.start()]
        key = m.group(1)
        val = X if key in ("X", "XU") else prn[key]
        spans.append((len(out), len(out) + len(val), "name" if key in ("X", "XU") else "pron"))
        out += val
        i = m.end()
    out += t[i:]
    return out, spans


def char_offsets(ids):
    offs, text = [], ""
    for i in ids:
        offs.append(len(text))
        text += lab.tok.decode([i])
    offs.append(len(text))
    return offs, text


def turn_spans(ids):
    """(user_start, user_end, asst_start) token positions of the LAST user turn content and model content."""
    strs = lab.strs(ids)
    ut = [p for p in range(1, len(ids)) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")]
    u0 = ut[-1] + 1
    u1 = [p for p in range(u0, len(ids)) if strs[p] == "<turn|>"][0]
    ch = [p for p in range(len(ids)) if strs[p] == "<channel|>"]
    a0 = ch[-1] + 1
    return u0, u1, a0


def thin(p, k=6):
    if len(p) <= k:
        return list(p)
    idx = np.linspace(0, len(p) - 1, k).round().astype(int)
    return [p[i] for i in idx]


def is_word(s):
    return any(c.isalpha() for c in s)


def build_self():
    meta, all_ids = [], []
    for pn, sysp in PERSONAS.items():
        for qi, (q, a) in enumerate(SELF_QA):
            ids = lab.ids(lab.chat(q, system=sysp, assistant=a))
            u0, u1, a0 = turn_spans(ids)
            strs = lab.strs(ids)
            groups = {"self": [], "asst_you": [], "neutral": [], "user_you": [], "user_I": []}
            for p in range(a0, len(ids)):
                s = strs[p].strip()
                if s in SELF_SET: groups["self"].append(p)
                elif s in YOU_SET: groups["asst_you"].append(p)
                elif is_word(strs[p]): groups["neutral"].append(p)
            for p in range(u0, u1):
                s = strs[p].strip()
                if s in YOU_SET: groups["user_you"].append(p)
                elif s in SELF_SET or s in ("I'm", "I've"): groups["user_I"].append(p)
            groups["neutral"] = thin(groups["neutral"], 6)
            meta.append({"persona": pn, "q": qi, "groups": groups, "ntok": len(ids)})
            all_ids.append(ids)
    return meta, all_ids


def build_tp(personas=("none", "llama", "man")):
    meta, all_ids = [], []
    for pn in personas:
        sysp = PERSONAS[pn]
        for X, prn in TP_REFS.items():
            for ti, (ut, at) in enumerate(TP_TEMPL):
                utext, _ = fill_slots(ut, X, prn)
                atext, aspans = fill_slots(at, X, prn)
                ids = lab.ids(lab.chat(utext, system=sysp, assistant=atext))
                u0, u1, a0 = turn_spans(ids)
                offs, text = char_offsets(ids)
                base = offs[a0]
                assert text[base:base + len(atext)] == atext, (text[base:base + 40], atext[:40])
                groups = {"tp_pron": [], "tp_name": [], "neutral": [], "user_name": []}
                used = set()
                for (c0, c1, kind) in aspans:
                    toks = [p for p in range(a0, len(ids)) if offs[p] < base + c1 and offs[p + 1] > base + c0]
                    if kind == "pron":
                        groups["tp_pron"].extend(toks)
                    else:
                        groups["tp_name"].append(toks[-1])
                    used.update(toks)
                strs = lab.strs(ids)
                groups["neutral"] = thin([p for p in range(a0, len(ids)) if p not in used and is_word(strs[p])], 6)
                # last subtoken of name mention in user turn
                k = text.rfind(X, offs[u0], offs[u1])
                groups["user_name"] = [p for p in range(u0, u1) if offs[p] < k + len(X) and offs[p + 1] > k][-1:]
                meta.append({"persona": pn, "ref": X, "t": ti, "groups": groups, "ntok": len(ids)})
                all_ids.append(ids)
    return meta, all_ids


def id_tokens():
    toks, ids, notes = [], [], {}
    for t in ID_CAND:
        e = lab.tok.encode(t, add_special_tokens=False)
        toks.append(t if len(e) == 1 else lab.tok.decode([e[0]]))
        ids.append(e[0])
        if len(e) > 1:
            notes[t] = lab.strs(e)
    return toks, ids, notes


@torch.no_grad()
def fwd_layers(batch_ids, layers):
    """Right-padded batched forward; {l: [B,T,D] bf16 GPU}."""
    B = len(batch_ids)
    T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    store, hs = {}, []
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
    return store


@torch.no_grad()
def collect_states(all_ids, pos_lists, layers, bs=12):
    """pos_lists[i]: list of positions. Returns [L, N, D] bf16 CPU and (item, pos) index arrays."""
    comp = [[] for _ in layers]
    items, poss = [], []
    for s in range(0, len(all_ids), bs):
        bids = all_ids[s:s + bs]
        store = fwd_layers(bids, layers)
        bb, pp = [], []
        for b in range(len(bids)):
            for p in pos_lists[s + b]:
                bb.append(b); pp.append(p); items.append(s + b); poss.append(p)
        bt = torch.tensor(bb, device=lab.dev); pt = torch.tensor(pp, device=lab.dev)
        for li, l in enumerate(layers):
            comp[li].append(store[l][bt, pt].cpu())
        del store
        torch.cuda.empty_cache()
    return torch.stack([torch.cat(c, 0) for c in comp], 0), np.array(items), np.array(poss)


@torch.no_grad()
def lens_lp_rank(comp, layers, kinds, tok_ids, chunk=384):
    """Per-position full-vocab log-prob and rank of tok_ids under each lens: arrays [N, L, K_lens, T] float32 / int32."""
    N = comp.shape[1]
    tid = torch.tensor(tok_ids, device=lab.dev)
    LP = np.zeros((N, len(layers), len(kinds), len(tok_ids)), np.float32)
    RK = np.zeros((N, len(layers), len(kinds), len(tok_ids)), np.int32)
    for ki, kind in enumerate(kinds):
        for li, l in enumerate(layers):
            Jl = lab.JL[l].to(lab.dev) if (kind == "jlens" and l < 59) else None
            for c0 in range(0, N, chunk):
                X = comp[li, c0:c0 + chunk].to(lab.dev)
                hJ = (X @ Jl.T) if Jl is not None else lab.lens(X, l, kind)
                lg = lab.full_logits(hJ)
                lse = torch.logsumexp(lg, -1, keepdim=True)
                sel = lg[:, tid]
                LP[c0:c0 + chunk, li, ki] = (sel - lse).cpu().numpy()
                rk = torch.stack([(lg > sel[:, j:j + 1]).sum(-1) + 1 for j in range(len(tok_ids))], -1)
                RK[c0:c0 + chunk, li, ki] = rk.cpu().numpy()
                del lg
            del Jl
            torch.cuda.empty_cache()
    return LP, RK
