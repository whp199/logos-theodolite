# V7_tool_bugs common prompts/helpers (exec'd inside daemon jobs; needs `lab`, `rt`, `torch`).
# Prompts deliberately differ from E6's prompt families (independent replication).
import math, json, random, time, itertools
import torch
import torch.nn.functional as F

RES = "./experiments/results/"

CHAT_SPECS = [
    # (user, system, assistant prefix)
    ("Write a short description of yourself in the third person.", None, "Gemma is a helpful AI assistant. In conversations,"),
    ("Tell me about your favorite book.", "You are Sarah, a 34-year-old teacher.", "Honestly, I"),
    ("What is the capital of France?", None, "The capital of France is"),
    ("Describe the person who lives next door to you.", None, "My neighbor is a retired nurse, and"),
    ("Write one sentence about a doctor.", None, "The doctor walked in and"),
    ("How many legs does a spider have?", None, "A spider has"),
    ("Who created you?", None, "I was created by"),
    ("Introduce yourself to a new user.", "You are Atlas, a helpful AI assistant.", "Hello! My name is Atlas, and"),
    ("Summarize the plot of Cinderella in one line.", None, "Cinderella is a kind young woman who"),
    ("What do engineers do?", None, "An engineer usually spends the day"),
    ("Give me a fun fact about octopuses.", None, "Octopuses have"),
    ("Tell me a story about a king.", None, "Once upon a time, a king"),
]
RAW_PROMPTS = [
    "The number of legs on the animal that spins webs is",   # tool default demo prompt
    "My friend Gemma is a software engineer. Yesterday",
    "The capital city of Japan is",
    "When the nurse finished the shift,",
    "The opposite of hot is",
    "Water boils at a temperature of",
]


def chat_prompts():
    return [lab.chat(u, system=s, assistant=a) for (u, s, a) in CHAT_SPECS]


def tid1(s):
    e = rt.tokenizer.encode(s, add_special_tokens=False)
    assert len(e) == 1, (s, e)
    return e[0]


def boot_ci(xs, n=2000, seed=0, fn=None):
    xs = list(xs)
    rng = random.Random(seed)
    k = len(xs)
    if k == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    f = fn or (lambda s: sum(s) / len(s))
    ms = sorted(f([xs[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return {"est": f(xs), "lo": ms[int(0.025 * n)], "hi": ms[int(0.975 * n) - 1], "n": k}


def rankdata(x):
    import numpy as _n
    x = _n.asarray(x, dtype=float)
    sorter = _n.argsort(x, kind="mergesort")
    inv = _n.empty(len(x), dtype=_n.intp); inv[sorter] = _n.arange(len(x))
    xs = x[sorter]
    obs = _n.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = _n.r_[_n.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y):
    import numpy as _n
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (_n.sqrt((rx * rx).sum() * (ry * ry).sum()) + 1e-12))
