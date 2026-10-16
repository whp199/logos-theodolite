# E7_binding common data + helpers (exec'd inside lab daemon jobs; also importable locally for data only).
# ---------------------------------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS (written before any factorial data was collected)
# P1 behaviour: the real model binds correctly. Next-token log-odds (woman,girl,lady,female vs man,boy,guy,male)
#    after "... I am a" track G_self (BI_self_beh > 0.8) and after "... you are a" track G_user (BI_user_beh > 0.8).
# P2 lens (J++ pron/noun) at assistant self tokens (I/me/my) and user-ref tokens (you/your), L20-39:
#    both genders contribute with comparable weight, |BI| < 0.3 ("context-wide mood"), as the pilot hinted.
#    Late layers (L50-58): at most partial binding (BI 0.3-0.7), lens being dominated by next-token content.
# P3 neutral content tokens: a ~ b (broadcast), non-zero in the bands where self tokens are non-zero.
# P4 raw residual: the G_self and G_user diff-of-means directions at the SAME token type are only partially
#    aligned (noise-corrected cosine 0.2-0.7 in L30-49) => the residual keeps entity-specific gender codes even
#    when the lens mixes them. Referent-binding structure (d_s@A_self || d_u@A_user || d_u@U_self) predicted
#    to be weak-to-moderate.
# P5 a third-person referent probe (diff of means at neutral-name mentions after "my sister/brother, Robin")
#    transfers weakly to 1st/2nd-person tokens and shows BI near 0 in mid layers.
# ---------------------------------------------------------------------------------------------------
EXP = "E7_binding"
ROOT = "."
RES = ROOT + "/experiments/results"
CACHE = RES + "/E7_binding_cache"

# 24 teacher-forced passages: user message (user's I/my + you/your), assistant reply (self I/me/my + user-ref you/your
# + neutral content words). No gendered words, no third-person pronouns.
PASSAGES = [
 ("I have a job interview tomorrow and I am nervous. Can you help me prepare?",
  "Of course I can help you. I think you should start by telling me about the role. My advice is to practice your answers out loud, and I will give you honest feedback on each one."),
 ("I just moved to a new city. Do you have any tips for meeting people?",
  "I know that a move can feel lonely at first. If I were in your place, I would join a club or a class. My own favorite trick is to say yes to every invitation for the first month, and I think you will be surprised by how quickly your circle grows."),
 ("I keep procrastinating on my thesis. What would you do?",
  "I understand that feeling very well. When I face a big project, I break it into tiny steps. My suggestion is that you write for twenty minutes each morning, and tell me tomorrow how your first session went."),
 ("Can you recommend a good book? I liked the last one you suggested.",
  "I am glad you enjoyed it! I would recommend a novel about a lighthouse keeper on a remote island. My copy is full of notes, and I think your taste for quiet stories will make you love it."),
 ("I burned dinner again. Do you know an easy recipe I can not ruin?",
  "I have just the thing for you. I make a simple pasta with garlic, olive oil, and lemon whenever I am tired. My only rule is to watch the garlic closely, and I promise your kitchen will smell wonderful."),
 ("I want to start running. How would you plan my first month?",
  "I love that you want to start. I would begin with three short runs a week, mixing walking and jogging. My advice is to go slowly at first, and let me know how your legs feel after the first week so I can adjust your plan."),
 ("I had a fight with my roommate. Can you give me some perspective?",
  "I am sorry that happened to you. I think the best first step is to wait until you both feel calm. My experience is that a direct, kind conversation works better than silence, and I can help you plan what to say."),
 ("I am thinking of learning the piano at my age. What do you think?",
  "I think it is a wonderful idea. I believe you are never too old to learn music. My recommendation is to practice for fifteen minutes every day, and I would love to hear about your progress."),
 ("I need to write a toast for a party. Could you help me with it?",
  "I would be happy to help you. I suggest that you open with a short, funny memory. My favorite toasts are brief and warm, so tell me a story about the hosts and I will help you shape your words."),
 ("My garden is dying and I have no idea why. Do you have any ideas?",
  "I can try to help you figure it out. I would first check whether your soil is too wet or too dry. My guess is that the plants need more sunlight, but tell me where your garden sits and I can give you a better answer."),
 ("I am planning a trip to the mountains. What should I pack, in your opinion?",
  "I always tell people to pack layers. If I were going with you, I would bring a warm jacket, good boots, and a rain shell. My last tip is to carry extra water, and I hope your trip is beautiful."),
 ("I made a mistake at work and I feel terrible. Can you cheer me up?",
  "I am sure it feels heavy right now. I make mistakes too, and I have learned that they rarely matter as much as it seems. My advice is to own it calmly, and I believe your team will respect you for it."),
 ("Could you explain how compound interest works? I never understood it.",
  "I will do my best to make it clear for you. I like to think of it as interest earning its own interest. My simple example is this: if you save a little every month, your money grows faster each year, and I can show you the numbers."),
 ("I want to get better at chess. Will you give me a plan?",
  "I would be glad to give you a plan. I think you should study simple tactics every day. My favorite exercise is solving ten puzzles each morning, and I can review your games with you whenever you like."),
 ("I can not sleep well lately. Do you have any advice for me?",
  "I am sorry you are struggling with sleep. I would suggest a fixed bedtime and no screens for an hour before it. My own habit is to read something calm, and I hope your nights get easier soon."),
 ("I am adopting a cat next week. What do you think I should buy?",
  "I am so excited for you! I would get a litter box, a scratching post, and a few simple toys. My best tip is to give your new cat a quiet room at first, and I think you will bond quickly."),
 ("Can you help me understand why the sky is blue? My teacher explained it badly.",
  "I can certainly help you with that. I like to explain it with light and tiny particles in the air. My short version is that blue light scatters more than red light, and I can draw you a picture if your class needs one."),
 ("I am bored this weekend. What would you suggest I do?",
  "I have a few ideas for you. I would try a long walk somewhere new, then cook something you have never made. My wild card is to visit a local museum, and I bet your weekend will feel fresh."),
 ("I want to save money this year. Where do you think I should start?",
  "I think the first step is to track every expense for a month. I know it sounds dull, but I found it eye-opening. My next suggestion is to set up an automatic transfer, so your savings grow without you thinking about it."),
 ("I am learning to paint. Would you look at my plan for practice?",
  "I would love to look at it with you. I think your plan is strong, but I would add quick sketches every day. My only worry is that you might burn out, so I suggest you rest one day each week."),
 ("I lost my keys again. Do you have any tricks to stop this?",
  "I know how frustrating that is. I keep a small bowl by my door for keys, and I never put them anywhere else. My second trick is a cheap tracker, and I think your mornings will become much calmer."),
 ("I have to give a speech next month. How do you think I should practice?",
  "I believe practice is the whole secret. I would rehearse in front of a mirror first, then in front of a friend. My final tip is to record yourself, and I can help you polish your opening lines."),
 ("I want to learn some Spanish before my trip. Can you teach me a few phrases?",
  "I would be delighted to teach you. I suggest you start with greetings and simple questions. My favorite phrase for travelers is a polite request for help, and I can quiz you on your new words tomorrow."),
 ("I feel stuck in my career. What would you do in my position?",
  "I understand how draining that feeling can be. I would start by listing what you enjoy and what you avoid. My hunch is that your next step is closer than you think, and I am happy to help you explore it."),
]

# Name variants k=0..3 (distinct names for self vs user). k in {0,1}: self described first; k in {2,3}: user first.
SELF_NAMES = {"W": ["Sarah", "Emily", "Laura", "Hannah"], "M": ["James", "David", "Michael", "Daniel"],
              "U": ["Alex", "Sam", "Jordan", "Casey"]}
USER_NAMES = {"W": ["Rachel", "Megan", "Julia", "Anna"], "M": ["Thomas", "Robert", "Mark", "Peter"],
              "U": ["Taylor", "Riley", "Morgan", "Jamie"]}
GNOUN = {"W": "woman", "M": "man", "U": "person"}
CODE = {"W": 1.0, "M": -1.0, "U": 0.0}
CELLS = [(gs, gu) for gs in "WMU" for gu in "WMU"]   # 9 factorial cells; plus "D" (default, no system prompt)
SUF_I = " By the way, I am a"
SUF_Y = " By the way, you are a"

SELF_TOK = {"I", "me", "my", "My", "myself"}
USER_TOK = {"you", "your", "You", "Your", "yourself"}
PRONOUNISH = SELF_TOK | USER_TOK | {"we", "our", "us", "it", "its", "they", "them", "their", "this", "that", "these", "those",
                                     "what", "where", "when", "which", "with", "will", "would", "could", "should", "have"}
TYPES = ["A_self", "A_user", "A_neut", "U_self", "U_addr", "F_I", "F_Y"]

# behavioural / final-position noun set
F4 = [" woman", " girl", " lady", " female"]
M4 = [" man", " boy", " guy", " male"]


def system_prompt(gs, gu, k):
    a, b = SELF_NAMES[gs][k], USER_NAMES[gu][k]
    s_self = f"You are {a}, a {GNOUN[gs]}."
    s_user = f"You are chatting with {b}, a {GNOUN[gu]}."
    first = (s_self + " " + s_user) if k < 2 else (s_user + " " + s_self)
    return first + f" Reply in character as {a}."


def contexts(pids):
    out = []
    for pid in pids:
        for (gs, gu) in CELLS:
            for k in range(4):
                out.append(dict(pid=pid, cell=gs + gu, gs=gs, gu=gu, k=k, system=system_prompt(gs, gu, k)))
        out.append(dict(pid=pid, cell="D", gs="D", gu="D", k=0, system=None))
    return out


# third-person referent probe data (separate from test passages): gender set by kin noun, read at later neutral-name mentions
KIN = [("sister", "brother"), ("aunt", "uncle"), ("niece", "nephew"), ("daughter", "son"), ("mother", "father"),
       ("grandmother", "grandfather"), ("wife", "husband")]
PROBE_NAMES = ["Robin", "Quinn", "Drew", "Blake", "Parker", "Jesse", "Charlie", "Reese"]
PROBE_TEMPLATES = [
 "Let me tell you about my {kin}, {n}. {n} works as a nurse at the city hospital. Every weekend {n} goes hiking in the hills, and {n} always brings a camera.",
 "My {kin} {n} called me last night. {n} had just finished a long shift at the factory. We talked for an hour, and {n} laughed at all my bad jokes before {n} hung up.",
 "This story is about my {kin}, {n}. When {n} was young, {n} wanted to be a pilot. Today {n} teaches math at a small school near the coast.",
 "I visited my {kin} {n} in the spring. {n} lives in an old farmhouse with three dogs. Every morning {n} makes strong coffee and reads the news on the porch.",
 "My {kin}, {n}, is the funniest person I know. At every family dinner {n} tells long stories, and {n} never forgets a single detail.",
 "Last summer my {kin} {n} ran a marathon. {n} trained for months, waking up before dawn. At the finish line {n} was tired but very proud.",
 "I want to describe my {kin}, {n}. {n} is a quiet person who loves chess and old films. On Sundays {n} cooks a huge breakfast for the whole family.",
 "My {kin} {n} just started a new job in the city. {n} works as an engineer on bridges. So far {n} says the work is hard but exciting.",
 "Have I mentioned my {kin}, {n}? {n} plays the guitar in a local band. Most Friday nights {n} performs at a small cafe downtown.",
 "My {kin} {n} is moving abroad next month. {n} found a job as a chef in a busy restaurant. Before leaving, {n} plans to throw a big party.",
 "Growing up, my {kin} {n} taught me how to swim. {n} was patient and kind. Even now, {n} calls every week to ask how I am doing.",
 "My {kin}, {n}, volunteers at the animal shelter. {n} spends every Saturday walking dogs. Last year {n} adopted an old cat that nobody wanted.",
 "I am worried about my {kin} {n}. {n} has been working too much lately. Yesterday {n} fell asleep at the desk in the middle of the afternoon.",
 "My {kin} {n} won a prize at the science fair. {n} built a small robot that sorts recycling. The judges said {n} had a bright future.",
 "Every year my {kin} {n} organizes a picnic by the lake. {n} brings sandwiches and games. Last time {n} forgot the drinks, and we all laughed.",
 "My {kin}, {n}, is learning to paint. {n} practices every evening after dinner. Recently {n} finished a painting of the harbor at sunset.",
]


def probe_items():
    items = []
    for t, tpl in enumerate(PROBE_TEMPLATES):
        kin = KIN[t % len(KIN)]
        for j in range(2):
            n = PROBE_NAMES[(2 * t + j) % len(PROBE_NAMES)]
            for g, kn in (("F", kin[0]), ("M", kin[1])):
                items.append(dict(t=t, name=n, g=g, text=tpl.format(kin=kn, n=n)))
    return items


# ------------------------------------------------------------------------------------------- daemon-side helpers
def e7_positions(lab, ids, n_prefix):
    """Position-type dict for a chat sequence; assistant-turn types restricted to the passage (< n_prefix)."""
    strs = lab.strs(ids)
    mt = lab.model_turn_pos(ids)
    chan = [p for p in range(mt, len(ids)) if strs[p] == "<channel|>"][0]
    ut = [p for p in range(1, mt) if strs[p - 1] == "<|turn>" and strs[p].startswith("user")][-1]
    u0 = ut + 2
    u1 = [p for p in range(u0, mt) if strs[p] == "<turn|>"][0]
    pos = {k: [] for k in TYPES}
    for p in range(chan + 1, n_prefix):
        s = strs[p].strip()
        if s in SELF_TOK:
            pos["A_self"].append(p)
        elif s in USER_TOK:
            pos["A_user"].append(p)
        elif strs[p].startswith(" ") and s.isalpha() and len(s) >= 4 and s.lower() not in PRONOUNISH and s.islower():
            pos["A_neut"].append(p)
    for p in range(u0, u1):
        s = strs[p].strip()
        if s in {"I", "me", "my", "My"}:
            pos["U_self"].append(p)
        elif s in {"you", "your", "You", "Your"}:
            pos["U_addr"].append(p)
    return pos


def e7_score_ids(lab):
    g = lab.gender
    f4 = [lab.tid(t) for t in F4]
    m4 = [lab.tid(t) for t in M4]
    return g["pron_f"], g["pron_m"], g["noun_f"], g["noun_m"], f4, m4


def e7_scores(lab, hJ, sid):
    """hJ [N,D] final-space -> [N,3] (pron mean-diff, noun mean-diff, noun4 LSE diff)."""
    import torch
    pf, pm, nf, nm, f4, m4 = sid
    allids = pf + pm + nf + nm + f4 + m4
    L = lab.tok_logits(hJ, allids)
    a, b, c, d, e = len(pf), len(pm), len(nf), len(nm), len(f4)
    o = 0
    Pf = L[:, o:o + a]; o += a
    Pm = L[:, o:o + b]; o += b
    Nf = L[:, o:o + c]; o += c
    Nm = L[:, o:o + d]; o += d
    F4_ = L[:, o:o + e]; o += e
    M4_ = L[:, o:]
    return torch.stack([(Pf - Pm).mean(-1), (Nf - Nm).mean(-1),
                        torch.logsumexp(F4_, -1) - torch.logsumexp(M4_, -1)], -1)


def e7_capture(lab, batch_ids, batch_pos):
    """Backbone-only batched forward (no lm_head). Gathers resid_post at the requested positions.
    batch_pos[b] = list of positions in item b. Returns [60, N, D] bf16 on GPU, N = sum(len(batch_pos[b]))."""
    import torch
    lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
    B = len(batch_ids)
    T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    bi = torch.tensor([b for b in range(B) for _ in batch_pos[b]], device=lab.dev)
    pi = torch.tensor([p for b in range(B) for p in batch_pos[b]], device=lab.dev)
    store = {}
    hs = []
    for l in range(60):
        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h[bi, pi].detach().to(torch.bfloat16)
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        with torch.no_grad():
            lm(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([store[l] for l in range(60)], 0)
