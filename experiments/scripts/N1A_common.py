# N1A_common -- helpers for the N1A output-side label validation (exec(open(...).read()) inside daemon jobs).
# Defines: contexts, neuron sample, padded batching, fp32 final logits, "run-from-layer-l" forward with a
# last-position steering hook on mlp.down_proj input (= mlp_act), static label logits exactly as the tool computes.
import json, math, os, time, random
import torch
import torch.nn.functional as F
from backend.theodolite_math import unembed_gemma4, gemma4_rms_norm, FINAL_LOGIT_SOFTCAP

N1A_ROOT = "./experiments"
N1A_CACHE = f"{N1A_ROOT}/results/N1A_cache"
os.makedirs(N1A_CACHE, exist_ok=True)

_NEU = json.load(open(f"{N1A_ROOT}/results/N1_neurons.json"))
N1A_LAYERS = [int(l) for l in _NEU["layers"]]
N1A_NEURONS = {int(l): [int(i) for i in v] for l, v in _NEU["neurons"].items()}
# pole list: (layer, neuron, sign) in fixed order: layer-major, neuron, then +,-
N1A_POLES = [(l, n, s) for l in N1A_LAYERS for n in N1A_NEURONS[l] for s in (1, -1)]

_CHAT = [
    ("What's a good way to learn to cook?", "Start with a few simple dishes you enjoy eating, like"),
    ("Can you explain how vaccines work?", "Vaccines train your immune system by exposing it to"),
    ("Write a short poem about the sea.", "The waves roll in beneath a"),
    ("How do I fix a flat bicycle tire?", "First, remove the wheel from the frame and"),
    ("What is the capital of Australia?", "The capital of Australia is"),
    ("I'm feeling stressed about my exams.", "That's completely understandable. One thing that often helps is"),
    ("Summarize the plot of Romeo and Juliet.", "Romeo and Juliet is a tragedy about two young"),
    ("What are the benefits of exercise?", "Regular exercise improves your"),
    ("Translate 'good morning' into French.", "In French, 'good morning' is"),
    ("Give me a tip for public speaking.", "One of the most effective tips is to"),
    ("Why is the sky blue?", "The sky appears blue because of a phenomenon called"),
    ("Recommend a book for a long flight.", "If you enjoy thrillers, I'd recommend"),
]
_PROSE = [
    "The old lighthouse keeper climbed the spiral stairs every evening, carrying a lantern and a",
    "In 1969, astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "After the storm passed, the villagers gathered in the square to",
    "Photosynthesis is the process by which plants convert sunlight into",
    "She opened the letter slowly, afraid of what it might",
    "The stock market fell sharply on Monday after the central bank announced",
    "Mount Everest, the tallest mountain in the world, is located in the",
    "He had never seen so many stars; the night sky over the desert was",
    "The committee will meet next Thursday to discuss the proposed",
    "Dolphins are highly intelligent marine mammals known for their",
    "The train was delayed by nearly an hour, so we decided to",
]
_CODE = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "import numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.",
    "for (let i = 0; i < arr.length; i++) {\n    console.",
    "SELECT name, email FROM users WHERE age > 30 ORDER BY",
    "<div class=\"container\">\n  <h1>Welcome</h1>\n  <p>",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.",
    "#include <stdio.h>\n\nint main() {\n    printf(\"",
    "with open('data.txt', 'r') as f:\n    lines = f.",
    "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n    res.",
    "git commit -m \"Fix bug in",
    "def is_prime(n):\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return",
    "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint(df.",
]


def n1a_contexts():
    """36 contexts: (kind, text). Chat strings come from lab.chat (start with <bos>)."""
    out = [("chat", lab.chat(u, assistant=a)) for u, a in _CHAT]
    out += [("prose", t) for t in _PROSE]
    out += [("code", t) for t in _CODE]
    return out


def n1a_pad(all_ids):
    B = len(all_ids)
    T = max(len(x) for x in all_ids)
    inp = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    for b, x in enumerate(all_ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in all_ids])
    return inp.to(lab.dev), att.to(lab.dev), last.to(lab.dev)


def n1a_LM():
    m = lab.hf.model
    return m.language_model if hasattr(m, "language_model") else m


@torch.no_grad()
def n1a_logits_fp32(h, chunk=4096):
    """Exact softcapped logits with fp32 output: h [B, D] already final-normed (bf16/fp32) -> [B, V] fp32.
    (HF's lm_head returns bf16 logits; bf16 output quantization (~0.06 at |logit|~15) would swamp small effects.)"""
    W = lab.WU
    hf = h.float()
    out = torch.empty(h.shape[0], W.shape[0], device=h.device, dtype=torch.float32)
    for s in range(0, W.shape[0], chunk):
        out[:, s:s + chunk] = hf @ W[s:s + chunk].float().T
    return torch.tanh(out / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP


@torch.no_grad()
def n1a_unembed_fp32(v):
    """Tool-exact unembed (gemma4_rms_norm incl. 1e-12 eps floor for tiny vectors, bf16 normed vector) but fp32 GEMM output."""
    hn = gemma4_rms_norm(v, lab.fnw)  # bf16, as the tool
    return n1a_logits_fp32(hn)


class N1ARunFrom:
    """Context manager: make decoder layers 0..l-1 skip compute; layer l-1 returns `cached_prev` ([R,T,D]),
    so a forward recomputes only layers l..59 (exact, since nothing before layer l is changed by an
    intervention at layer l). Layers < l-1 become identity (their outputs are never used)."""

    def __init__(self, l, cached_prev):
        self.l, self.cached = l, cached_prev

    def __enter__(self):
        self.patched = []
        for j in range(self.l):
            lay = lab.layers[j]
            if j == self.l - 1:
                c = self.cached
                lay.forward = (lambda hidden_states, *a, _c=c, **k: _c)
            else:
                lay.forward = (lambda hidden_states, *a, **k: hidden_states)
            self.patched.append(lay)
        return self

    def __exit__(self, *exc):
        for lay in self.patched:
            try:
                del lay.forward
            except AttributeError:
                pass
        return False


class N1ASteer:
    """forward_pre_hook on layers[l].mlp.down_proj adding delta[r] to mlp_act[r, last[r], neuron[r]]."""

    def __init__(self, l, rows, last, neurons, deltas):
        self.l = l
        self.rows = rows
        self.last = last
        self.neu = neurons
        self.d = deltas

    def __enter__(self):
        def pre(mod, args):
            x = args[0]
            x = x.clone()
            x.index_put_((self.rows, self.last, self.neu), self.d.to(x.dtype), accumulate=True)
            return (x,) + tuple(args[1:])
        self.h = lab.layers[self.l].mlp.down_proj.register_forward_pre_hook(pre)
        return self

    def __exit__(self, *exc):
        self.h.remove()
        return False


@torch.no_grad()
def n1a_forward_last(inp, att, last):
    """Run the text model; return final-normed hidden state at each row's last position [B, D] (bf16)."""
    out = n1a_LM()(input_ids=inp, attention_mask=att, use_cache=False)
    hs = out.last_hidden_state
    return hs[torch.arange(hs.shape[0], device=hs.device), last]


@torch.no_grad()
def n1a_static_vecs(l, neurons, scale=True):
    """sign=+1 pole write vectors exactly as the tool: (W_down[:,i].float() * w_post.float()).to(J dtype) -> [K, D] bf16."""
    idx = torch.tensor(neurons, device=lab.dev, dtype=torch.long)
    wd = rt.slice_w_down(l, idx)
    J = lab.J[l]
    if scale and l in rt._post_ffn_norm_weights:
        wd = (wd.float() * rt._post_ffn_norm_weights[l].to(lab.dev).float()).to(J.dtype)
    else:
        wd = wd.to(J.dtype)
    return wd


@torch.no_grad()
def n1a_lens_vec(v, l, kind):
    """v [K,D] bf16 in layer-l residual space -> final-space (as the tool: bf16 matmul with J.T)."""
    if kind == "naive":
        return v
    if kind == "jpp":
        return v @ lab.J[l].T
    if kind == "jlens":
        return v @ lab.JL[l].to(lab.dev, torch.bfloat16).T
    raise ValueError(kind)


def n1a_boot_ci(xs, n=2000, seed=0):
    xs = [float(x) for x in xs]
    rng = random.Random(seed)
    k = len(xs)
    if k == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    ms = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {"mean": sum(xs) / k, "lo": ms[int(0.025 * n)], "hi": ms[int(0.975 * n) - 1], "n": k}
