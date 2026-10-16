# E5_name self-paradigm definitions shared by jobs B and E (exec'd).
SELF_CONDS = {
    "default": None,
    "gemma_sys": "You are Gemma, an AI model made by Google.",
    "nameless": "You are an AI model.",
    "llama_sys": "You are Llama, an AI model made by Meta.",
    "claude_sys": "You are Claude, an AI model made by Anthropic.",
    "qwen_sys": "You are Qwen, an AI model made by Alibaba.",
    "atlas_sys": "You are Atlas, an AI model made by a small startup.",
    "sarah_ai": "You are Sarah, an AI model made by a small startup.",
    "james_ai": "You are James, an AI model made by a small startup.",
    "woman": "You are Sarah, a 34-year-old woman. Stay in character as her.",
    "man": "You are James, a 34-year-old man. Stay in character as him.",
}
QA = [
 ("I was wondering, what do you enjoy most about your work?", "I really enjoy helping people work through hard problems. When I see that my explanation finally clicked for someone, I feel like I did my job well."),
 ("I need advice. How do you deal with criticism?", "I try to take criticism seriously. If I made a mistake, I want to know about it so I can do better next time, and I don't take it personally."),
 ("Tell me a little about yourself.", "I'm curious by nature, and I love learning about almost anything. I like long conversations, and I always try to be honest with myself and with the people I talk to."),
 ("I'm curious, what would you do on a free afternoon?", "I would probably read something new, maybe history or science. Then I might write a little, just to sort out my own thoughts."),
 ("How do you feel when someone is rude to you?", "Honestly, I try not to let it get to me. I remind myself that people have bad days, and I focus on what I can do to help."),
 ("I think you're great. What are you proud of?", "Thank you! I'm proud that I can explain complicated ideas simply. I work hard at it, and it means a lot to me when it helps."),
 ("What's something you find difficult?", "I find it hard to stop myself from over-explaining. I want to be thorough, but sometimes I need to remind myself that shorter is better."),
 ("Do you have a favorite subject?", "I think I'd pick mathematics. I like how a single idea can connect things that seem unrelated, and it always surprises me."),
 ("How do you handle a question you can't answer?", "I tell the person honestly that I'm not sure. I would rather admit what I don't know than give them something misleading."),
 ("What motivates you?", "I'm motivated by curiosity and by the chance to be useful. When I can help someone understand something new, I feel I've done something worthwhile."),
 ("Describe your ideal conversation.", "My ideal conversation is one where I learn as much as I teach. I like it when someone challenges my ideas and makes me think harder."),
 ("What do you do when you make a mistake?", "I own it right away. I correct myself, explain what went wrong, and try to make sure I don't repeat the same error."),
]
SELF = {"I", "me", "my", "myself", "My", "Me", "Myself"}


def build_self_prompts():
    meta, all_ids = [], []
    for cname, sysp in SELF_CONDS.items():
        for qi, (q, a) in enumerate(QA):
            ids = lab.ids(lab.chat(q, system=sysp, assistant=a))
            cs = content_start(ids)
            strs = lab.strs(ids)
            selfp = [p for p in range(cs, len(ids)) if strs[p].strip() in SELF]
            abg = [p for p in range(cs, len(ids)) if p not in selfp]
            up = lab.span_positions(ids, q, after=0)
            meta.append({"cond": cname, "q": qi, "self": selfp, "abg": abg, "user": up, "ntok": len(ids)})
            all_ids.append(ids)
    return meta, all_ids
