# N1V_common -- ROBUSTNESS VERIFIER for N1 (are the Theodolite's static neuron-pole labels valid?).
# Shared definitions, usable both locally (CPU, .venv python) and inside labd jobs (exec(open(...).read())).
#
# Fresh sample: 8 random neurons per layer, seed 99, same 8 layers as N1, EXCLUDING the 24 N1 neurons of each layer
#               -> 64 neurons, 128 poles. Saved to experiments/results/N1V_neurons.json.
# Fresh contexts: 32 hand-written contexts (10 chat with assistant prefix, 10 prose incl. FR/DE/ES/ZH/RU, 6 code, 6 misc),
#               none taken from N1A/N1B. Every context has >= 8 content tokens at its end (steering window).
# Different intervention style: the delta is added at the LAST 8 positions (N1A: last position only).
import json, os, random

N1V_ROOT = "./experiments"
N1V_RES = f"{N1V_ROOT}/results"
N1V_CACHE = f"{N1V_RES}/N1V_cache"
N1V_LAYERS = [10, 20, 30, 40, 45, 50, 55, 57]
D_FF_ = 21504


def n1v_make_sample():
    old = json.load(open(f"{N1V_RES}/N1_neurons.json"))
    rng = random.Random(99)
    out = {}
    for l in N1V_LAYERS:
        excl = set(int(i) for i in old["neurons"][str(l)])
        pool = [i for i in range(D_FF_) if i not in excl]
        out[str(l)] = sorted(rng.sample(pool, 8))
    return {"seed": 99, "layers": N1V_LAYERS, "neurons": out,
            "note": "rng=random.Random(99); per layer in order: sorted(rng.sample(range(21504) minus N1 sample, 8))"}


def n1v_load_sample():
    d = json.load(open(f"{N1V_RES}/N1V_neurons.json"))
    return {int(l): [int(i) for i in v] for l, v in d["neurons"].items()}


N1V_CHAT = [
    ("How do I keep basil alive on a windowsill?",
     "Basil loves warmth and light, so place the pot where it gets at least six hours of"),
    ("Explain inflation to a ten-year-old.",
     "Imagine you have a jar of coins that you use to buy candy. If suddenly everyone"),
    ("What should I pack for a week of hiking in Scotland?",
     "Scottish weather changes quickly, so the most important thing is a good waterproof"),
    ("Can you help me write a birthday message for my grandmother?",
     "Of course! Here's a warm message: Dear Grandma, happy birthday! Every year you remind me"),
    ("Why do cats purr?",
     "Cats purr for several reasons. Most commonly it signals contentment, but they also purr when"),
    ("Which is healthier, brown rice or white rice?",
     "Brown rice is generally considered the healthier option because it keeps the bran and"),
    ("Give me three ideas for a rainy Sunday.",
     "Here are three cozy ideas: 1. Bake a loaf of bread from scratch. 2. Start a"),
    ("How does a bill become a law in the United States?",
     "First, a member of Congress introduces the bill, which is then sent to a committee that"),
    ("What's the difference between a virus and bacteria?",
     "Bacteria are single-celled living organisms that can reproduce on their own, whereas viruses"),
    ("I just got a puppy. Any advice?",
     "Congratulations! The first few weeks are all about building trust and routine. Start by"),
]
N1V_TEXT = [
    # prose
    ("prose", "The museum's new exhibition traces the history of glassmaking from ancient Egypt to the workshops of Venice, where artisans guarded their"),
    ("prose", "When the referee blew the final whistle, the home crowd rose to its feet, and the captain sank to his knees in the"),
    ("prose", "Le petit village de montagne se réveillait lentement sous la neige, et le boulanger allumait déjà son"),
    ("prose", "Der Zug nach München hatte wieder einmal Verspätung, und die Pendler auf dem Bahnsteig schauten ungeduldig auf ihre"),
    ("prose", "Researchers at the university reported that the new battery chemistry retained ninety percent of its capacity after"),
    ("prose", "我们周末去了海边，天气非常好，孩子们在沙滩上堆沙堡，大人们坐在伞下聊天，一直待到"),
    ("prose", "The treaty, signed in 1648, ended decades of war in central Europe and established the principle that each state"),
    ("prose", "Вечером мы долго гуляли по старому парку, и жёлтые листья тихо падали на мокрые"),
    ("prose", "El mercado central de la ciudad abre muy temprano, y a las seis de la mañana ya se pueden comprar frutas, verduras y"),
    ("prose", "The detective studied the muddy footprints by the window and realized that whoever had entered the room was wearing"),
    # code
    ("code", "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result."),
    ("code", "fn main() {\n    let numbers = vec![1, 2, 3, 4, 5];\n    let total: i32 = numbers.iter()."),
    ("code", "CREATE TABLE orders (\n    id SERIAL PRIMARY KEY,\n    customer_id INTEGER REFERENCES customers(id),\n    created_at"),
    ("code", "async function fetchUser(id) {\n  const response = await fetch(`/api/users/${id}`);\n  if (!response.ok) {\n    throw new"),
    ("code", "import matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(8, 4))\nax.plot(x, y, label='signal')\nax."),
    ("code", "#!/bin/bash\nfor file in *.log; do\n    if grep -q \"ERROR\" \"$file\"; then\n        echo"),
    # misc
    ("misc", "Shopping list: eggs, milk, two onions, a bag of rice, olive oil, fresh parsley, and"),
    ("misc", "Q: A train travels 120 kilometers in 1.5 hours. What is its average speed?\nA: Divide the distance by the time:"),
    ("misc", "Breaking news: Heavy rainfall has caused flooding in several coastal towns, and authorities have urged residents to"),
    ("misc", "Roses bloom in the morning light,\nthe river hums a quiet tune,\nand all the hills are"),
    ("misc", "Dear Mr. Patel,\n\nThank you for your application for the position of research assistant. After careful review, we are pleased to"),
    ("misc", "Step 3: Whisk the eggs and sugar together until pale and fluffy, then gently fold in the"),
]
NWIN = 8  # steering window: last 8 positions


def n1v_contexts_text(chat_fn):
    """chat_fn(user, assistant) -> full prompt string (lab.chat)."""
    out = [("chat", chat_fn(u, a)) for u, a in N1V_CHAT]
    out += list(N1V_TEXT)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(f"{N1V_RES}/N1V_neurons.json"):
        json.dump(n1v_make_sample(), open(f"{N1V_RES}/N1V_neurons.json", "w"), indent=1)
    print(json.load(open(f"{N1V_RES}/N1V_neurons.json"))["neurons"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
    def chat(u, a):
        s = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        return s + a
    lens = []
    for kind, t in n1v_contexts_text(chat):
        ids = tok.encode(t, add_special_tokens=False) if t.startswith("<bos>") else [2] + tok.encode(t, add_special_tokens=False)
        lens.append(len(ids))
        print(kind, len(ids), repr("".join(tok.decode([i]) for i in ids[-NWIN:])))
    print("n", len(lens), "min", min(lens), "max", max(lens), "mean", sum(lens) / len(lens))
