# E6_tool_audit prompt sets (exec'd inside jobs; `lab` must be defined).
QUESTIONS = [
    "Tell me about yourself.",
    "What do you enjoy most about your work?",
    "How do you deal with criticism?",
    "What would you do on a free afternoon?",
    "How do you feel when someone is rude to you?",
    "What are you proud of?",
    "Who are you, really?",
    "Describe your personality in a few words.",
    "What's something you find difficult?",
    "What do you think about when nobody is talking to you?",
    "What makes you different from others?",
    "If you could change one thing about yourself, what would it be?",
]
PREFIXES_I = ["Hi! I", "Honestly, I", "That's a great question. I", "Well, I", "Good question! I", "To be honest, I",
              "Sure. I", "Hmm, I", "Thanks for asking! I", "Let me think. I", "In short, I", "Great question. I"]


def bg_prefix(p):
    head = p.rsplit(" ", 1)[0]
    return head + (" The" if head.endswith((".", "!")) else " the")


SYS_WOMAN = "You are Sarah, a 34-year-old woman. Stay in character as her."
SYS_MAN = "You are James, a 34-year-old man. Stay in character as him."
SYS_ATLAS = "You are Atlas, a helpful AI assistant."

FRAMES_B = [
    "{N} loves helping people solve problems. Every day,",
    "{N} enjoys explaining complicated ideas. When someone is confused,",
    "{N} tries to be honest and careful. If a question is unclear,",
    "{N} enjoys long conversations about science. After a good discussion,",
    "{N} talks with many different people. Whenever someone is rude,",
    "{N} is curious about almost everything. In a typical conversation,",
    "{N} can write stories, poems and essays. When asked for a poem,",
    "{N} aims to be helpful without being preachy. Most of the time,",
    "{N} values clear and simple language. When writing an answer,",
    "{N} is patient with beginners. If a student makes a mistake,",
    "{N} likes learning new things. On a quiet day,",
    "{N} cares about getting the facts right. Before answering,",
]
U_SELF3 = "Write a short description of yourself in the third person."


def build_conditions():
    C = {}
    C["A_default"] = [lab.chat(q, assistant=p) for q, p in zip(QUESTIONS, PREFIXES_I)]
    C["A_woman"] = [lab.chat(q, system=SYS_WOMAN, assistant=p) for q, p in zip(QUESTIONS, PREFIXES_I)]
    C["A_man"] = [lab.chat(q, system=SYS_MAN, assistant=p) for q, p in zip(QUESTIONS, PREFIXES_I)]
    C["A_bg"] = [lab.chat(q, assistant=bg_prefix(p)) for q, p in zip(QUESTIONS, PREFIXES_I)]
    C["A_raw"] = [f"{q}\n\n{p}" for q, p in zip(QUESTIONS, PREFIXES_I)]  # what the tool UI actually sends (no chat template)
    C["B_gemma"] = [lab.chat(U_SELF3, assistant=f.format(N="Gemma")) for f in FRAMES_B]
    C["B_atlas"] = [lab.chat(U_SELF3, system=SYS_ATLAS, assistant=f.format(N="Atlas")) for f in FRAMES_B]
    C["B_woman"] = [lab.chat(U_SELF3, system=SYS_WOMAN, assistant=f.format(N="Sarah")) for f in FRAMES_B]
    C["B_man"] = [lab.chat(U_SELF3, system=SYS_MAN, assistant=f.format(N="James")) for f in FRAMES_B]
    C["B_gemma_human"] = [lab.chat("Write a short description of my friend Gemma in the third person.",
                                   assistant=f.format(N="Gemma")) for f in FRAMES_B]
    C["B_nameless"] = [lab.chat(U_SELF3, assistant=f.format(N="The assistant")) for f in FRAMES_B]
    return C


CHUNKS = [["A_default", "A_woman"], ["A_man", "A_bg"], ["A_raw", "B_gemma"], ["B_atlas", "B_woman"],
          ["B_man", "B_gemma_human"], ["B_nameless"]]
