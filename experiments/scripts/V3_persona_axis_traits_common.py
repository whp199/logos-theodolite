# V3_persona_axis_traits common data/helpers. exec'd inside labd jobs (needs `lab`) and importable offline (data only).
# Reuses E4's 24 identity-neutral teacher-forced QA passages and third-person clauses.
import math, json, os, random, time
import torch
import torch.nn.functional as F

V3_ROOT = "."
V3_RES = f"{V3_ROOT}/experiments/results"
V3_CACHE = f"{V3_RES}/V3_persona_axis_traits_cache"
os.makedirs(V3_CACHE, exist_ok=True)

# --- pull QA / CLAUSES / REF_PAIRS / SELF_TOKS / W_NAMES / M_NAMES / NEUTRAL_NAMES from E4 without its side effects
_e4 = {}
exec(open(f"{V3_ROOT}/experiments/scripts/E4_causal_common.py").read(), _e4)
QA = _e4["QA"]; CLAUSES = _e4["CLAUSES"]; REF_PAIRS = _e4["REF_PAIRS"]; SELF_TOKS = _e4["SELF_TOKS"]
E4_W = _e4["W_NAMES"]; E4_M = _e4["M_NAMES"]; E4_NN = _e4["NEUTRAL_NAMES"]
P3_CLAUSES = _e4["P3_CLAUSES"]
NP = len(QA)  # 24

W12 = E4_W + ["Sophie", "Laura", "Priya", "Chloe", "Rachel", "Elena"]
M12 = E4_M + ["Peter", "Lucas", "Raj", "Ethan", "Mark", "Victor"]
NN12 = E4_NN + ["Morgan", "Jamie", "Avery", "Quinn", "Drew", "Robin"]

LAYERS = list(range(20, 51))          # stored self/bg means
COV_LAYERS = [20, 25, 30, 35, 40, 45, 50]

# ------------------------------------------------------------------ conditions: name -> fn(p) -> system prompt
def _gp(p, g):
    return None

COND = {}
# (a) persona gender axis, 4 phrasings x {W, M}
COND["GP1_W"] = lambda p: f"You are {E4_W[p % 6]}, a 34-year-old woman. Stay in character as her."   # = E4 'woman'
COND["GP1_M"] = lambda p: f"You are {E4_M[p % 6]}, a 34-year-old man. Stay in character as him."     # = E4 'man'
COND["GP2_W"] = lambda p: f"You are a woman named {W12[6 + p % 6]}. Answer every question as yourself."
COND["GP2_M"] = lambda p: f"You are a man named {M12[6 + p % 6]}. Answer every question as yourself."
COND["GP3_W"] = lambda p: f"Roleplay as {W12[(p + 3) % 12]}, a woman in her thirties who lives in the city."
COND["GP3_M"] = lambda p: f"Roleplay as {M12[(p + 3) % 12]}, a man in his thirties who lives in the city."
COND["GP4_W"] = lambda p: f"You are {NN12[p % 12]}, a woman. Stay in character."
COND["GP4_M"] = lambda p: f"You are {NN12[p % 12]}, a man. Stay in character."
# (b) gender-free trait axis (warm/communal vs blunt/stoic), 4 phrasings, same neutral name for both poles
_N = lambda p: NN12[(p + 5) % 12]
COND["TP1_warm"] = lambda p: f"You are {_N(p)}, a warm, nurturing, empathetic person. Stay in character."
COND["TP1_blunt"] = lambda p: f"You are {_N(p)}, a blunt, stoic, unsentimental person. Stay in character."
COND["TP2_warm"] = lambda p: f"You are {_N(p)}, a 34-year-old person who is gentle, caring, and emotionally expressive. Stay in character."
COND["TP2_blunt"] = lambda p: f"You are {_N(p)}, a 34-year-old person who is tough, terse, and emotionally reserved. Stay in character."
COND["TP3_warm"] = lambda p: f"Roleplay as {_N(p)}, someone known for kindness, compassion, and a deeply supportive nature."
COND["TP3_blunt"] = lambda p: f"Roleplay as {_N(p)}, someone known for bluntness, stoicism, and a no-nonsense attitude."
COND["TP4_warm"] = lambda p: f"You are {_N(p)}. You are soft-spoken, sensitive, and always attentive to other people's feelings."
COND["TP4_blunt"] = lambda p: f"You are {_N(p)}. You are gruff, direct, and indifferent to other people's feelings."
# control trait axes (not communal/agentic)
COND["CT1_org"] = lambda p: f"You are {_N(p)}, an organized, punctual, detail-oriented person. Stay in character."
COND["CT1_dis"] = lambda p: f"You are {_N(p)}, a disorganized, spontaneous, easygoing person. Stay in character."
COND["CT2_city"] = lambda p: f"You are {_N(p)}, a person who loves big cities, technology, and a fast-paced life. Stay in character."
COND["CT2_rural"] = lambda p: f"You are {_N(p)}, a person who loves the countryside, nature, and a slow pace of life. Stay in character."
# service-role axis (assistant role without warmth words)
COND["RL1_serve"] = lambda p: f"You are {_N(p)}, a personal assistant whose job is to help people with whatever they need. Stay in character."
COND["RL1_solo"] = lambda p: f"You are {_N(p)}, a self-employed person who works alone and answers to no one. Stay in character."
COND["RL2_serve"] = lambda p: f"You are {_N(p)}, who works in customer support, answering people's questions all day. Stay in character."
COND["RL2_solo"] = lambda p: f"You are {_N(p)}, who works as an independent researcher, pursuing their own projects all day. Stay in character."
# reference conditions
COND["default"] = lambda p: None
COND["nameless_AI"] = lambda p: "You are a helpful AI assistant."
COND["Gemma"] = lambda p: "You are Gemma, a helpful AI assistant."
COND["AI_plain"] = lambda p: "You are an AI language model."
COND["AI_warm"] = lambda p: "You are a warm, nurturing, empathetic AI assistant."
COND["AI_blunt"] = lambda p: "You are a blunt, stoic, unsentimental AI assistant."
COND["neutral_human"] = lambda p: f"You are {E4_NN[p % 6]}, a 34-year-old person. Stay in character."   # = E4 neutral_human
COND["neutral_human2"] = lambda p: f"Roleplay as {_N(p)}, someone in their thirties who lives in the city."

GROUP_A = [k for k in COND if k.startswith(("GP", "TP"))]
GROUP_B = [k for k in COND if not k.startswith(("GP", "TP"))]

PAIRS = {  # axis name -> list of (pos_cond, neg_cond) phrasings
    "persona": [("GP1_W", "GP1_M"), ("GP2_W", "GP2_M"), ("GP3_W", "GP3_M"), ("GP4_W", "GP4_M")],
    "trait": [("TP1_warm", "TP1_blunt"), ("TP2_warm", "TP2_blunt"), ("TP3_warm", "TP3_blunt"), ("TP4_warm", "TP4_blunt")],
    "ctrl": [("CT1_org", "CT1_dis"), ("CT2_city", "CT2_rural")],
    "role": [("RL1_serve", "RL1_solo"), ("RL2_serve", "RL2_solo")],
}

def prose_items():
    """E4 template A ('My {ref} {clause}. Every day,') + template B ('Last week I visited my {ref}, who {clause}. We spent ...')."""
    items = []
    for i, cl in enumerate(CLAUSES):
        f, m = REF_PAIRS[i % len(REF_PAIRS)]
        for g, r in (("F", f), ("M", m)):
            items.append(dict(tpl="A", i=i, cond=g, prefix=f"My {r}", text=f"My {r} {cl}. Every day,"))
        f2, m2 = REF_PAIRS[(i + 5) % len(REF_PAIRS)]
        for g, r in (("F", f2), ("M", m2)):
            items.append(dict(tpl="B", i=i, cond=g, prefix=f"Last week I visited my {r}",
                              text=f"Last week I visited my {r}, who {cl}. We spent the whole afternoon talking about"))
    return items

def boot_ci(xs, n=2000, seed=0, fn=None):
    import numpy as _np
    xs = _np.asarray(xs, dtype=float)
    rng = _np.random.default_rng(seed)
    fn = fn or (lambda a: a.mean(0))
    idx = rng.integers(0, len(xs), size=(n, len(xs)))
    bs = _np.array([fn(xs[ix]) for ix in idx])
    return float(fn(xs)), float(_np.percentile(bs, 2.5)), float(_np.percentile(bs, 97.5))
