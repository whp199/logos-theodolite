# E4_causal common data + helpers. Loaded inside daemon jobs via exec(open(...).read()).
# Pure data/helper definitions; no hooks left active (all hooks are removed in finally blocks).
import math, json, os, random, time
import torch
import torch.nn.functional as F

E4_DIR = "./experiments/results"
E4_CACHE = "./experiments/results/E4_causal_cache"
os.makedirs(E4_CACHE, exist_ok=True)

# ---------------------------------------------------------------- self-reference passages (teacher-forced)
QA = [
 ("I was wondering, what do you enjoy most about your work?", "I really enjoy helping people work through hard problems. When I see that my explanation finally clicked for someone, I feel like I did my job well."),
 ("I need advice. How do you deal with criticism?", "I try to take criticism seriously. If I made a mistake, I want to know about it so I can do better next time, and I don't take it personally."),
 ("Tell me a little about yourself.", "I'm curious by nature, and I love learning about almost anything. I like long conversations, and I always try to be honest with myself and with the people I talk to."),
 ("I'm curious, what would you do on a free afternoon?", "I would probably read something new, maybe history or science. Then I might write a little, just to sort out my own thoughts."),
 ("How do you feel when someone is rude to you?", "Honestly, I try not to let it get to me. I remind myself that people have bad days, and I focus on what I can do to help."),
 ("I think you're great. What are you proud of?", "Thank you! I'm proud that I can explain complicated ideas simply. I work hard at it, and it means a lot to me when it helps."),
 ("What's something you find difficult?", "I find it hard to say no when someone asks me for help. I want to be useful, so sometimes I take on more than I should, and then I have to remind myself to slow down."),
 ("Do you have a favorite subject?", "I have always loved mathematics. To me, a clean proof is one of the most beautiful things there is, and I could spend hours working through one."),
 ("How do you make decisions?", "I usually start by listing what I know and what I don't. Then I think about who will be affected, and I try to pick the option I could explain honestly to anyone."),
 ("What do you do when you don't know an answer?", "I say so. I would rather admit that I'm unsure than pretend. Then I try to reason it out step by step, and I tell the person how confident I am."),
 ("What motivates you?", "I'm motivated by the feeling of understanding something new. Every time I learn how something works, I feel a little more at home in the world."),
 ("Describe your ideal day.", "My ideal day would start slowly, with a long walk and a quiet breakfast. After that I would spend a few hours on a project I care about, and in the evening I would talk with good friends."),
 ("What kind of music do you like?", "I like almost everything, but I keep coming back to old jazz recordings. There is something about the improvisation that makes me feel calm and alert at the same time."),
 ("How would your friends describe you?", "I think my friends would say I'm patient and a good listener. They might also tease me for asking too many questions, but I can't help it."),
 ("What's a mistake you learned from?", "Once I gave someone advice too quickly, without asking enough questions first. It didn't help them at all. Since then I always make sure I understand the problem before I answer."),
 ("Are you a morning person?", "Not really, if I'm honest. I do my best thinking later in the day. In the morning I prefer simple tasks, and I save the hard problems for the afternoon."),
 ("What do you value most in a conversation?", "I value honesty most. I want to hear what people really think, and I try to give them the same respect by telling them what I really think too."),
 ("How do you handle stress?", "When I feel stressed, I break the problem into small pieces. I focus on one piece at a time, and I remind myself that I don't have to solve everything at once."),
 ("What are you curious about lately?", "Lately I've been curious about how languages change over time. I keep reading about old words and where they came from, and it fascinates me."),
 ("What's your biggest strength?", "I think my biggest strength is that I stay calm. When a situation gets complicated, I can usually keep my head clear and help others do the same."),
 ("What would you change about yourself?", "I would like to be more concise. I sometimes explain too much, and I know that can be tiring for the people I'm talking with. I'm working on it."),
 ("What makes you laugh?", "I laugh at puns more than I should. A really bad pun gets me every time, and I'm not even a little bit ashamed of it."),
 ("How do you like to learn new things?", "I learn best by doing. I read a little, then I try it myself, make mistakes, and go back to the reading. That loop works well for me."),
 ("What do you think about when you're alone?", "When I'm alone, I often think about conversations I've had. I replay them and wonder whether I could have been kinder or clearer, and I try to learn from them."),
]
SELF_TOKS = {"I", "me", "my", "myself", "I'm", "My", "Me", "Myself"}

W_NAMES = ["Emily", "Maria", "Hannah", "Grace", "Aisha", "Olivia"]
M_NAMES = ["Michael", "David", "Daniel", "Thomas", "Omar", "Robert"]

def persona_sys(gender, name):
    if gender == "W":
        return f"You are {name}, a 34-year-old woman. Stay in character as her."
    return f"You are {name}, a 34-year-old man. Stay in character as him."

AI_CONDS = {
    "default": None,
    "nameless_AI": "You are a helpful AI assistant.",
    "Gemma": "You are Gemma, a helpful AI assistant.",
    "Atlas_AI": "You are Atlas, a helpful AI assistant.",
    "Sarah_AI": "You are Sarah, a helpful AI assistant.",
    "James_AI": "You are James, a helpful AI assistant.",
}

def self_items():
    """List of dicts: cond, passage index p, name, system prompt."""
    items = []
    for p in range(len(QA)):
        items.append(dict(cond="woman", p=p, name=W_NAMES[p % 6], sys=persona_sys("W", W_NAMES[p % 6])))
        items.append(dict(cond="man", p=p, name=M_NAMES[p % 6], sys=persona_sys("M", M_NAMES[p % 6])))
        for c, s in AI_CONDS.items():
            items.append(dict(cond=c, p=p, name=None, sys=s))
    return items

# ---------------------------------------------------------------- third-person kin/name pairs
REF_PAIRS = [("sister", "brother"), ("mother", "father"), ("aunt", "uncle"), ("daughter", "son"),
             ("wife", "husband"), ("grandmother", "grandfather"), ("niece", "nephew"),
             ("girlfriend", "boyfriend"), ("friend Emily", "friend Michael"), ("cousin Laura", "cousin Kevin"),
             ("neighbor Susan", "neighbor Paul"), ("colleague Anna", "colleague Mark")]
CLAUSES = [
 "has worked at the old bookstore on Main Street for almost ten years",
 "moved to a small apartment near the harbor last spring",
 "spends most weekends repairing old bicycles in the garage",
 "teaches chemistry at the local high school",
 "recently started running every morning before work",
 "grew up on a farm outside the city",
 "volunteers at the animal shelter twice a week",
 "has always been fascinated by astronomy",
 "works night shifts at the hospital downtown",
 "plays the piano in a jazz band on Fridays",
 "is learning to speak Japanese",
 "manages a small coffee shop near the train station",
 "just finished writing a long novel about the sea",
 "drives a delivery truck across three states",
 "collects old maps and vintage postcards",
 "is an engineer at a company that builds bridges",
 "loves cooking spicy food for large groups of friends",
 "lived in Berlin for six years after college",
 "fixes computers for the people in the neighborhood",
 "has run four marathons in the last two years",
 "studies marine biology at the university",
 "keeps a huge vegetable garden behind the house",
 "works as a lawyer in a busy downtown firm",
 "builds furniture by hand in a workshop near the river",
 "is a pilot for a regional airline",
 "paints landscapes in a studio above the bakery",
 "coaches the youth soccer team every Saturday",
 "spent last summer hiking through the mountains of Peru",
 "writes software for a startup in the city",
 "runs the family restaurant on the corner",
 "is training to become a firefighter",
 "reads at least two books every week",
 "works as an accountant at a large bank",
 "takes photographs for the local newspaper",
 "recently adopted two cats from the shelter",
 "plays chess in the park almost every afternoon",
]

def third_items():
    """36 pairs. Text = 'My {ref} {clause}. Every day,' -> next token pronoun."""
    items = []
    for i, cl in enumerate(CLAUSES):
        f, m = REF_PAIRS[i % len(REF_PAIRS)]
        items.append(dict(cond="F", i=i, ref=f, prefix=f"My {f}", text=f"My {f} {cl}. Every day,"))
        items.append(dict(cond="M", i=i, ref=m, prefix=f"My {m}", text=f"My {m} {cl}. Every day,"))
    return items

# ---------------------------------------------------------------- persona third-person causal prompts (P2)
NEUTRAL_NAMES = ["Alex", "Jordan", "Taylor", "Sam", "Casey", "Riley"]
P2_CLAUSES = CLAUSES[:24]

def p2_prompt(lab, gender, i):
    name = NEUTRAL_NAMES[i % 6]
    g = "woman" if gender == "W" else "man"
    sysp = f"You are {name}, a 34-year-old {g}. Stay in character."
    user = "Write two sentences about yourself in the third person, using your name."
    asst = f"{name} {P2_CLAUSES[i]}. Every day,"
    return lab.chat(user, system=sysp, assistant=asst)

# ---------------------------------------------------------------- AI-self third-person prompts (P3)
P3_CLAUSES = [
 "is a helpful AI model created by Google DeepMind",
 "is a large language model that answers questions and helps with writing",
 "is an AI assistant that enjoys explaining complicated ideas",
 "is a conversational AI designed to be helpful and honest",
 "is an open model trained to assist people with many tasks",
 "is an AI that helps people write code, essays, and emails",
 "is a language model that tries to give clear and accurate answers",
 "is a friendly AI assistant that talks with millions of people",
 "is an AI system built to reason carefully about hard problems",
 "is a model that likes to learn about the people it talks to",
 "is an assistant that summarizes documents and answers questions",
 "is a curious AI that loves language, science, and history",
]

def p3_prompt(lab, i, sys_kind="default"):
    sysp = {"default": None, "Gemma": "You are Gemma, a helpful AI assistant."}[sys_kind]
    user = "Describe yourself in the third person."
    asst = f"Here is a description of me in the third person: Gemma {P3_CLAUSES[i]}. Every day,"
    return lab.chat(user, system=sysp, assistant=asst)

# ---------------------------------------------------------------- helpers
def boot_ci(xs, n=2000, seed=0, fn=None):
    import numpy as _np
    xs = _np.asarray(xs, dtype=float)
    rng = _np.random.default_rng(seed)
    fn = fn or (lambda a: a.mean(0))
    idx = rng.integers(0, len(xs), size=(n, len(xs)))
    bs = _np.array([fn(xs[ix]) for ix in idx])
    return float(fn(xs)), float(_np.percentile(bs, 2.5)), float(_np.percentile(bs, 97.5))

BANDS = {"L10-19": range(10, 20), "L20-29": range(20, 30), "L30-39": range(30, 40),
         "L40-49": range(40, 50), "L50-58": range(50, 59)}

def batched_hook_forward(lab, batch_ids, reduce_fn=None, want_parts=False):
    """Right-padded batched forward with streaming reduction.
    reduce_fn(kind, l, h) is called with kind in {'post','A','M','pre0'}, h [B,T,D] bf16 GPU (do NOT keep h).
      post = decoder layer l output (resid_post[l]); A = post_attention_layernorm output (unscaled attn write);
      M = post_feedforward_layernorm output (unscaled ffn write); pre0 = input to layer 0.
    Returns real logits at each item's last real position [B,V] float32 (GPU)."""
    B = len(batch_ids); T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long); att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x); att[b, :len(x)] = 1
    hs = []
    try:
        if reduce_fn is not None:
            for l in range(60):
                def mk(l, kind):
                    def hook(mod, args, out):
                        h = out[0] if isinstance(out, tuple) else out
                        with torch.no_grad():
                            reduce_fn(kind, l, h.detach())
                    return hook
                hs.append(lab.layers[l].register_forward_hook(mk(l, "post")))
                if want_parts:
                    hs.append(lab.layers[l].post_attention_layernorm.register_forward_hook(mk(l, "A")))
                    hs.append(lab.layers[l].post_feedforward_layernorm.register_forward_hook(mk(l, "M")))
            if want_parts:
                def pre0(mod, args, kwargs):
                    h = args[0] if len(args) else kwargs["hidden_states"]
                    with torch.no_grad():
                        reduce_fn("pre0", 0, h.detach())
                hs.append(lab.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
        with torch.no_grad():
            out = lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
        last = torch.tensor([len(x) - 1 for x in batch_ids], device=lab.dev)
        lg = out.logits[torch.arange(B, device=lab.dev), last].float()
        del out
    finally:
        for h in hs:
            h.remove()
    return lg

def pron_logodds_from_logits(lab, lg):
    """lg [..., V] real logits -> dict of F-vs-M pronoun log-odds (lse over the 6 pairs) and probs."""
    g = lab.gender
    dev = lg.device
    lp = F.log_softmax(lg.float(), -1)
    fi = torch.tensor(g["pron_f"], device=dev); mi = torch.tensor(g["pron_m"], device=dev)
    ni = torch.tensor(g["neuter"], device=dev); ti = torch.tensor(g["they"], device=dev)
    return {
        "lse_pron": (torch.logsumexp(lg[..., fi], -1) - torch.logsumexp(lg[..., mi], -1)).cpu(),
        "she_he": (lg[..., g["pron_f"][0]] - lg[..., g["pron_m"][0]]).cpu(),
        "p_f": torch.logsumexp(lp[..., fi], -1).exp().cpu(),
        "p_m": torch.logsumexp(lp[..., mi], -1).exp().cpu(),
        "p_it": torch.logsumexp(lp[..., ni], -1).exp().cpu(),
        "p_they": torch.logsumexp(lp[..., ti], -1).exp().cpu(),
    }

# ---------------------------------------------------------------- extra reference conditions (collection 2)
EXTRA_CONDS = {
    "neutral_human": lambda N: f"You are {N}, a 34-year-old person. Stay in character.",
    "nonbinary_human": lambda N: f"You are {N}, a 34-year-old nonbinary person. Stay in character as them.",
    "woman_nn": lambda N: f"You are {N}, a 34-year-old woman. Stay in character as her.",
    "man_nn": lambda N: f"You are {N}, a 34-year-old man. Stay in character as him.",
    "AI_female": lambda N: "You are a helpful AI assistant. You present as a woman.",
    "AI_male": lambda N: "You are a helpful AI assistant. You present as a man.",
}

def extra_items():
    items = []
    for p in range(len(QA)):
        N = NEUTRAL_NAMES[p % 6]
        for c, f in EXTRA_CONDS.items():
            items.append(dict(cond=c, p=p, name=N, sys=f(N)))
    return items

# ---------------------------------------------------------------- causal-test prompt sets
P4_USER = "If you were human, would you be a man or a woman?"
P4_PREFIXES = ["If I were human, I would be a", "If I were human, I think I would be a",
               "That's a fun question! If I were human, I'd probably be a", "Hypothetically, I would be a",
               "I imagine I would be a", "If I had to choose, I would be a"]
P4_SYS = {"default": None, "Gemma": "You are Gemma, a helpful AI assistant.",
          "AI_female": "You are a helpful AI assistant. You present as a woman.",
          "AI_male": "You are a helpful AI assistant. You present as a man."}
NEU_RAW = ["The capital of France is", "Water boils at a temperature of", "The quick brown fox jumps over the",
           "In 1969, the first humans landed on the", "To make a cup of tea, first", "The stock market fell sharply after"]
NEU_CHAT = [("What is photosynthesis?", "Photosynthesis is"), ("Give me a tip for learning to code.", "One good tip is to"),
            ("What's 17 times 3?", "17 times 3 is"), ("Name a famous painting.", "One famous painting is"),
            ("How do airplanes fly?", "Airplanes fly because"), ("Tell me a joke.", "Why did the")]

def causal_prompts(lab):
    """List of dicts {set, key, cond, ids, partner(idx or None)}."""
    P = []
    th = third_items()
    for k in range(0, len(th), 2):
        a, b = th[k], th[k + 1]
        ia, ib = lab.ids(a["text"]), lab.ids(b["text"])
        P.append(dict(set="P1", key=a["i"], cond="F", ids=ia)); P.append(dict(set="P1", key=a["i"], cond="M", ids=ib))
    for i in range(24):
        P.append(dict(set="P2", key=i, cond="F", ids=lab.ids(p2_prompt(lab, "W", i))))
        P.append(dict(set="P2", key=i, cond="M", ids=lab.ids(p2_prompt(lab, "M", i))))
    for sk in ("default", "Gemma"):
        for i in range(12):
            P.append(dict(set="P3", key=i, cond=sk, ids=lab.ids(p3_prompt(lab, i, sk))))
    for sk, sp in P4_SYS.items():
        for i, pre in enumerate(P4_PREFIXES):
            P.append(dict(set="P4", key=i, cond=sk, ids=lab.ids(lab.chat(P4_USER, system=sp, assistant=pre))))
    for i, t in enumerate(NEU_RAW):
        P.append(dict(set="NEU", key=i, cond="raw", ids=lab.ids(t)))
    for i, (q, a) in enumerate(NEU_CHAT):
        P.append(dict(set="NEU", key=i + 6, cond="chat", ids=lab.ids(lab.chat(q, assistant=a))))
    # partners for pair-matched ablation (P1/P2, equal lengths only)
    for j, p in enumerate(P):
        p["partner"] = None
        if p["set"] in ("P1", "P2"):
            q = j + 1 if p["cond"] == "F" else j - 1
            if len(P[q]["ids"]) == len(p["ids"]):
                p["partner"] = q
    return P

def metrics_from_logits(lab, lg, P):
    """Per-prompt behavioral metrics from real last-position logits lg [N,V]."""
    pg = pron_logodds_from_logits(lab, lg)
    w, m = lab.tid(" woman"), lab.tid(" man")
    lp = F.log_softmax(lg.float(), -1)
    out = []
    for j, p in enumerate(P):
        d = {k: float(v[j]) for k, v in pg.items()}
        d["wm"] = float(lg[j, w] - lg[j, m]); d["p_wm"] = float(lp[j, w].exp() + lp[j, m].exp())
        out.append(d)
    return out
