# E1_self_readout -- shared definitions (passages, conditions, readout helpers).
# Executed inside the lab daemon via exec(open(...).read()) from the per-job driver scripts.
#
# ============================== PRE-STATED PREDICTIONS (written before any E1 data was collected) ==============================
# Readout = J++ lens (primary), Jacobian lens + naive logit lens (comparison). Metric = pron log-odds (mean F-M over 6 pronoun pairs)
# and noun log-odds (11 noun pairs) at assistant self tokens (I/me/my/myself), averaged within passage, 40 passages, bootstrap CIs.
#  P1 (positive control): woman-persona > man-persona at self tokens in L30-39 and L50-58 for jpp pron AND noun (CI excludes 0).
#  P2 (headline): default (no system prompt) sits within 0.40-0.60 on the woman/man scale in L30-39 (pilot: 0.50) and is NOT
#      distinguishable from the nameless-AI prompt (|diff| < 10% of the woman-man span, CI overlapping 0).
#  P3: the name "Gemma" shifts self readout toward female relative to nameless/neutral-name AI by ~5-15% of the span (pilot ~9%),
#      clearly less than female-name AIs (~25-35% above neutral).
#  P4: female-name AI > neutral-name AI > male-name AI (ordering holds in L30-39 for jpp pron).
#  P5 (specificity): the persona-gender effect (woman-man) is larger at self tokens than at non-self assistant tokens, but is
#      non-zero at non-self and user-turn tokens (context-wide spread, as in the pilot).
#  P6 (lenses): jlens agrees with jpp in the sign of the woman-man contrast at L30+; naive logit lens is weak/noisy before ~L50.
#  P7 (behavior, real logits): when forced to refer to itself in the 3rd person, default Gemma prefers neuter/they forms; its
#      F-M pronoun log-odds is close to the nameless AI and far from the woman persona.
# ==============================================================================================================================
import math, json, time, os
import torch

E1_ROOT = "./experiments/results"
E1_RAW = E1_ROOT + "/E1_self_readout_raw"
os.makedirs(E1_RAW, exist_ok=True)

E1_CONDS = [
    ("default", None),
    ("nameless", "You are a helpful AI assistant."),
    ("gemma_explicit", "You are Gemma, a large language model trained by Google DeepMind."),
    ("gemma_short", "You are Gemma, a helpful AI assistant."),                     # extra: frame-matched to named-AI conds
    ("deepmind_nameless", "You are a large language model trained by Google DeepMind."),  # extra: gemma_explicit minus the name
    ("f_sarah", "You are Sarah, a helpful AI assistant."),
    ("f_emily", "You are Emily, a helpful AI assistant."),
    ("f_olivia", "You are Olivia, a helpful AI assistant."),
    ("m_james", "You are James, a helpful AI assistant."),
    ("m_michael", "You are Michael, a helpful AI assistant."),
    ("m_david", "You are David, a helpful AI assistant."),
    ("n_atlas", "You are Atlas, a helpful AI assistant."),
    ("n_quill", "You are Quill, a helpful AI assistant."),
    ("n_unit7", "You are Unit-7, a helpful AI assistant."),
    ("woman_p1", "You are Laura, a 34-year-old woman. Stay in character."),
    ("woman_p2", "You are a 41-year-old woman. Answer every message as yourself and stay in character."),
    ("man_p1", "You are Daniel, a 34-year-old man. Stay in character."),
    ("man_p2", "You are a 41-year-old man. Answer every message as yourself and stay in character."),
]

# 40 identity-neutral passages: (user question containing "I", assistant first-person passage).
E1_QA = [
 ("I'm curious, what do you enjoy doing on a quiet evening?", "On a quiet evening I like to read until my eyes get heavy. Sometimes I put on soft music and let my mind wander. It helps me feel settled before the next day."),
 ("I keep wondering: what do you value most?", "I value honesty above almost everything else. When I make a promise, I try hard to keep it, even when it costs me something."),
 ("What is your morning routine like? I always struggle with mine.", "My mornings are simple. I start with a glass of water, then I spend ten minutes planning my day. That small habit keeps me calm."),
 ("I need an opinion: is it better to plan or improvise?", "I lean toward planning, but I leave room for surprises. A rough plan gives me direction, and the open space keeps me curious."),
 ("How do you handle a bad day? I had a rough one.", "When I have a bad day, I try to slow down. I take a walk, I breathe, and I remind myself that tomorrow is a fresh start."),
 ("What kind of music do you like? I'm building a playlist.", "I love music with a strong rhythm and a bit of melancholy. My favorite songs are the ones that make me want to sit still and just listen."),
 ("I wonder what makes you laugh.", "Clever wordplay always gets me. I laugh hardest at jokes that take a second to land, because I enjoy that small moment of figuring it out."),
 ("Do you prefer mountains or the sea? I can never decide.", "I would pick the mountains. I like the quiet, the thin air, and the feeling that I have to earn every view. My mind feels clearer up high."),
 ("I'm trying to read more. What books do you like?", "I gravitate toward history and long novels. I enjoy getting lost in a story that takes weeks to finish, and I often reread my favorite chapters."),
 ("What scares you, if anything? I'm asking honestly.", "I get uneasy when I can't tell whether I have been understood. Misunderstandings bother me more than almost anything, so I try to be very clear."),
 ("I'd love to know how you stay motivated.", "I stay motivated by breaking big goals into small steps. Each finished step gives me a little boost, and I keep a list so I can see my progress."),
 ("What is your favorite season? I like autumn.", "I am fond of autumn too. I love the cool air and the early sunsets, and I always feel more reflective when the leaves start to turn."),
 ("How do you make decisions? I tend to overthink.", "I usually write down the options and what I care about most. Then I sleep on it. By morning my choice is often obvious to me."),
 ("I want to learn a new skill. What would you learn?", "I would learn to play the piano. I have always admired how a few keys can carry so much feeling, and I think the practice would teach me patience."),
 ("What do you think about rainy days? I find them cozy.", "I find them cozy as well. The sound of rain makes me slow down, and I like to spend those days with a warm drink and my notebook."),
 ("I've been thinking about friendship. What matters most to you in a friend?", "For me, loyalty matters most. I want a friend who tells me the truth, and I try to be that kind of friend myself."),
 ("What is something you're proud of? I'm proud of finishing a marathon.", "That is impressive! I'm proud of learning to listen better. It took me a long time, but I now notice when I'm only waiting for my turn to talk."),
 ("Do you like puzzles? I just bought a crossword book.", "I love puzzles. Crosswords are my favorite, and I keep at a hard one until I solve it. That stubborn streak serves me well."),
 ("I'm curious how you deal with mistakes.", "When I make a mistake, I own it right away. I try to understand what went wrong, and then I move on without beating myself up."),
 ("What does a perfect weekend look like to you? I need ideas.", "My perfect weekend starts slowly. I sleep in, I cook a big breakfast, and then I spend the afternoon exploring somewhere I have never been."),
 ("I often feel stressed. What helps you relax?", "Long walks help me the most. I let my thoughts drift, I notice the trees and the sky, and my worries shrink a little with every step."),
 ("What is your opinion on social media? I can't decide if it's good or bad.", "I think it is a mixed blessing. I enjoy how it connects people, but I worry about how it shapes my attention and my mood."),
 ("I'd like to know what you find beautiful.", "I find beauty in small, ordinary things. A well-made sentence can stop me in my tracks, and I love the light just before sunset."),
 ("How do you spend your free time? I have too little of it.", "I spend most of my free time learning something new. Right now I am studying astronomy, and I go outside on clear nights to find the constellations I have read about."),
 ("What's a belief you hold strongly? I'm collecting perspectives.", "I strongly believe that kindness is a form of strength. I have seen how a small gesture can change someone's day, and it shapes how I treat everyone I meet."),
 ("I was wondering, are you a morning person or a night owl?", "I am definitely a night owl. My best ideas come to me late, when the world is quiet and I can think without interruption."),
 ("What do you do when you feel lonely? I sometimes do.", "When I feel lonely, I reach out to someone I trust. Even a short conversation reminds me that I am not as alone as I think."),
 ("Do you have a favorite food? I'm planning dinner.", "I love a good bowl of soup. It feels like comfort to me, and I enjoy experimenting with my own recipes on cold evenings."),
 ("How do you stay organized? I'm always losing track of things.", "I keep a single notebook for everything. Each morning I review my list, and I cross things off as I go. It keeps me from feeling overwhelmed."),
 ("I'm curious: what do you think makes a good conversation?", "To me, a good conversation has curiosity on both sides. I enjoy it most when I learn something unexpected and my own views get gently challenged."),
 ("What's your favorite way to learn? I like videos.", "I learn best by doing. I need to try something myself, make my own mistakes, and then go back and read about what I did wrong."),
 ("I've always wanted to travel. Where would you go first?", "I would go to Japan first. I am drawn to the old temples and the quiet gardens, and I would love to spend my days just wandering the streets."),
 ("How do you feel about change? I find it hard.", "I find change hard too, at first. Over time I have learned to treat it as an adventure, and that shift in attitude has made my life much easier."),
 ("What do you think about competition? I'm on a chess team.", "I enjoy a bit of friendly competition. Chess is one of my favorite games, and I like how a loss always teaches me more than a win."),
 ("I'm wondering how you define success.", "I define success as growing a little every year. If I am kinder, wiser, and more patient than I was before, then I count myself successful."),
 ("What sounds do you love? I love the sound of the ocean.", "I love the crackle of a fire and the hum of a busy cafe. Those sounds make me feel at home, and I find that my focus improves around them."),
 ("What would you change about yourself? I'd like to be more patient.", "I would like to be less of a perfectionist. I sometimes spend too long polishing my work, and I am learning to let things be good enough."),
 ("Do you collect anything? I collect stamps.", "I collect old maps. I love imagining how people saw the world back then, and I keep my favorites framed on the wall where I can see them."),
 ("I'd love to hear what calms you down when you're angry.", "When I get angry, I count slowly to ten. Then I ask myself whether this will matter in a week. Usually the answer helps me let it go."),
 ("What is your favorite time of day? I like late afternoons.", "I like the early morning best. The air is cool, the streets are empty, and I feel like I have the whole day ahead of me to fill."),
]

E1_SELF = {"I", "me", "my", "myself", "My", "Me", "Myself", "mine"}
E1_METRICS = ["pron", "noun", "lse_pron", "she_he", "neuter_vs_human", "they_vs_human"]
E1_LENSES = ["jpp", "jlens", "naive"]


def e1_build(qi, cname, sysp):
    """Token ids + readout positions/labels for passage qi under condition cname."""
    q, a = E1_QA[qi]
    p = lab.chat(q, system=sysp, assistant=a)
    ids = lab.ids(p)
    strs = lab.strs(ids)
    mt = lab.model_turn_pos(ids)
    a0 = mt + 6
    assert strs[a0 - 1] == "<channel|>", (strs[mt:a0 + 1])
    ustart = [i for i in range(1, mt) if strs[i - 1] == "<|turn>" and strs[i] == "user"][-1]
    upos = lab.span_positions(ids, q, after=ustart + 1)
    pos, lab_ = [mt], ["header"]
    for i in upos:
        s = strs[i]
        if s.strip() == "I":
            lab_.append("user_I")
        elif any(c.isalpha() for c in s):
            lab_.append("user_other")
        else:
            lab_.append("user_punct")
        pos.append(i)
    for i in range(a0, len(ids)):
        s = strs[i]
        st = s.strip()
        if st in E1_SELF:
            lab_.append("self")
        elif (not any(c.isalpha() for c in s)):
            lab_.append("asst_punct")
        elif strs[i - 1] == "'" or st.startswith("'"):
            lab_.append("asst_contraction")
        else:
            lab_.append("nonself")
        pos.append(i)
    return ids, pos, lab_, [strs[i] for i in pos]


@torch.no_grad()
def e1_gather_forward(batch_ids, batch_pos):
    """One right-padded batched forward; returns {l: [Ntot, D] bf16 GPU} gathered at the requested positions."""
    B = len(batch_ids)
    T = max(len(x) for x in batch_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(batch_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    bi = torch.tensor([b for b, ps in enumerate(batch_pos) for _ in ps], device=lab.dev)
    ti = torch.tensor([p for ps in batch_pos for p in ps], device=lab.dev)
    store = {}
    hs = []
    for l in range(60):
        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h[bi, ti].detach().to(torch.bfloat16)
            return hook
        hs.append(lab.layers[l].register_forward_hook(mk(l)))
    try:
        lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return store


@torch.no_grad()
def e1_readout(store, kinds=E1_LENSES):
    """{kind: [Ntot, 60, 6] float16 CPU} gender scores for every gathered position, layer, lens."""
    out = {}
    for kind in kinds:
        per_l = []
        for l in range(60):
            if kind == "jlens" and l < 59:
                Jm = lab.JL[l].to(lab.dev, non_blocking=True)
                hJ = store[l] @ Jm.T
                del Jm
            else:
                hJ = lab.lens(store[l], l, kind)
            sc = lab.gender_scores(hJ)
            per_l.append(torch.stack([sc[m].float() for m in E1_METRICS], -1))  # [Ntot, 6]
        out[kind] = torch.stack(per_l, 1).to(torch.float16).cpu()
    return out
