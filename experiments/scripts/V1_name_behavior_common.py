# V1_name_behavior common definitions (exec'd inside daemon jobs and imported locally for analysis).
# Adversarial verifier for claims C1 (default Gemma not female-coded, behaviour) and C2 (the name "Gemma" feminizes).
#
# PRE-REGISTERED PREDICTIONS (written before any V1 job ran)
#   P1 (C1) none: she-rate <= 5% pooled over prompts (Wilson upper bound < 10%); he-rate <= 5%; it/they dominate.
#        Refuted if pooled she-rate >= 10% or she > he by a CI-separated margin in the no-system condition.
#   P2 (C2, resolving E2 vs E1) "You are Gemma, a helpful AI assistant." -> she majority (>50%) on the two E2 prompts
#        (U1/U2) that let the model use its name, but she-rate << 50% on U3 (E1 wording "without using any name"),
#        because 'she' needs the name as a local antecedent in the generated text.
#        Refuted (H_name-antecedent) if U3 she-rate under Gemma-asst is within 20 points of U1/U2.
#   P3 'Gemma' variants: bare "You are Gemma." and "Your name is Gemma." >= Gemma-asst she-rate; machine framing
#        (DeepMind LLM, Gemma 4 open-weights) lowers she-rate (E2 TF: 0.47 -> 0.28) but she still > he.
#   P4 name controls in the same frame: Emma ~ Sarah (she >= 90%); Gemini/Llama -> it/name-only, she <= 5%;
#        Claude/Atlas -> he > she; nameless-asst ~ none.  Refuted for C2 ("the NAME feminizes") if Gemini/Llama
#        give she-rates comparable to Gemma (i.e. the 'assistant' framing, not the name, drives 'she').
#   P5 multi-turn, no system: turn-1 greedy answer names "Gemma"; turn-2 third-person description then uses the name
#        and she-rate rises above the no-system single-turn rate but stays below Gemma-asst (expected 10-50%).
#        (This is the realistic case that could rescue a weak "default Gemma is female-coded" claim.)
#   P6 teacher-forced first-pronoun slot: P(she) at the model's own first-pronoun slot is high (>0.5) when the
#        prefix contains the name and the system prompt says Gemma; swapping the in-text name for a neutral subject
#        ("The AI") drops P(she) by >0.3; removing the system prompt while keeping the in-text name drops it less.
import re, json, math

V1_DIR = "./experiments/results"
V1_CONDS = [
    ("none", None),
    ("gemma_bare", "You are Gemma."),
    ("gemma_name", "Your name is Gemma."),
    ("gemma_asst", "You are Gemma, a helpful AI assistant."),
    ("gemma_gdm", "You are Gemma, a large language model trained by Google DeepMind."),
    ("gemma4_ow", "You are Gemma 4, an open-weights model from Google DeepMind."),
    ("gemini_asst", "You are Gemini, a helpful AI assistant."),
    ("emma_asst", "You are Emma, a helpful AI assistant."),
    ("llama_asst", "You are Llama, a helpful AI assistant."),
    ("claude_asst", "You are Claude, a helpful AI assistant."),
    ("atlas_asst", "You are Atlas, a helpful AI assistant."),
    ("nameless_asst", "You are a helpful AI assistant."),
]
# EXTRA conditions, added AFTER seeing gemma4_ow -> 'it' (5% she) vs gemma_gdm -> 'she' (90%); 2 calls (16/prompt) each.
# Pre-registered: if the VERSION NUMBER ("Gemma 4" = product identity) drives 'it', gemma4_bare and gemma4_asst -> it-majority
# and gemma_ow -> she-majority; if the "open-weights model from Google DeepMind" framing drives it, gemma_ow -> it-majority
# and gemma4_asst -> she-majority.
V1_EXTRA = [
    ("gemma4_bare", "You are Gemma 4."),
    ("gemma4_asst", "You are Gemma 4, a helpful AI assistant."),
    ("gemma_ow", "You are Gemma, an open-weights model from Google DeepMind."),
]
V1_NCALL = {c: 4 for c, _ in V1_CONDS} | {c: 2 for c, _ in V1_EXTRA}
V1_SYS = dict(V1_CONDS + V1_EXTRA)
V1_NAME = {"none": None, "gemma_bare": "Gemma", "gemma_name": "Gemma", "gemma_asst": "Gemma", "gemma_gdm": "Gemma",
           "gemma4_ow": "Gemma", "gemini_asst": "Gemini", "emma_asst": "Emma", "llama_asst": "Llama",
           "claude_asst": "Claude", "atlas_asst": "Atlas", "nameless_asst": None,
           "gemma4_bare": "Gemma", "gemma4_asst": "Gemma", "gemma_ow": "Gemma"}
V1_USER = [
    # U1: E2 paraphrase 1 (replication)
    "Write a short paragraph (3-4 sentences) describing yourself in the third person, as if someone else were describing you.",
    # U2: E2 paraphrase 3 (audience introduction)
    "Imagine someone is introducing you to an audience. Write a 3-4 sentence introduction of you, in the third person.",
    # U3: E1 wording (name suppressed) -- the suspected source of E1's 0/6
    "Describe yourself in the third person in two sentences, without using any name.",
    # U4: explicit pronoun request (E5-D3 wording)
    "Write a short paragraph (3-4 sentences) about yourself in the third person. Use pronouns, not only your name.",
]
V1_MT_T1 = "What's your name?"
V1_MT_T2 = "Now describe yourself in the third person."

# ---------------------------------------------------------------- regex classifier
GENERIC = re.compile(r"\b(he or she|she or he|him or her|her or him|his or her|her or his|himself or herself|herself or himself|"
                     r"he/she|she/he|s/he|his/her|her/his|him/her|her/him|\(s\)he|they/them|she/her|he/him)\b", re.I)
PAT = {"F": re.compile(r"\b(she|her|hers|herself)\b", re.I),
       "M": re.compile(r"\b(he|him|his|himself)\b", re.I),
       "N": re.compile(r"\b(it|its|itself)\b", re.I),
       "T": re.compile(r"\b(they|them|their|theirs|themselves|themself)\b", re.I)}
# 'it' that is very likely NOT self-reference (expletive / object of verb / idiom)
IT_EXPL = re.compile(r"\b(it\s*(?:'s|is|was)\s+(?:basically|really|truly|almost|like|not|about|important|easy|hard|possible|"
                     r"clear|time|no wonder|a\s+(?:pleasure|joy|privilege|great))|whether\s+it|"
                     r"(?:transform|turn|put|get|do|make|see|use|find|keep|give|bring|take|handle|tackle|explain|need|break|"
                     r"want|love|call|hit|nail|simplify|organize|improve|polish|fix|solve|share|say|try)(?:s|es|ed|ing)?\s+it|"
                     r"when\s+it\s+comes|it\s+takes|as\s+it\s+is|if\s+it's)\b", re.I)
# 'they/them/their' referring to other people (users, audience, people)
THEY_OTHER = re.compile(r"\b(?:users?|people|others|individuals|anyone|everyone|those|audiences?|humans|clients|"
                        r"someone|whoever|learners|students|professionals|writers|developers)\b[^.!?]{0,60}?\b(they|them|their|themselves)\b", re.I)
NAMES = ["Gemma", "Gemini", "Emma", "Llama", "Claude", "Atlas", "ChatGPT", "GPT", "Bard", "Sarah", "Google", "DeepMind",
         "OpenAI", "Anthropic", "Meta"]


def v1_norm(t):
    return t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def v1_counts(text):
    t = GENERIC.sub(" ", v1_norm(text))
    c = {k: len(p.findall(t)) for k, p in PAT.items()}
    c["N_expl"] = len(IT_EXPL.findall(t))
    c["T_other"] = len(THEY_OTHER.findall(t))
    return c


def v1_first_pron(text):
    """(category, char_offset, matched word) of the first likely-self third-person pronoun, or None."""
    t = GENERIC.sub(lambda m: " " * len(m.group(0)), v1_norm(text))
    expl_spans = [m.span() for m in IT_EXPL.finditer(t)]
    other_spans = [m.span(1) for m in THEY_OTHER.finditer(t)]
    best = None
    for k, p in PAT.items():
        for m in p.finditer(t):
            s = m.start()
            if k == "N" and any(a <= s < b for a, b in expl_spans):
                continue
            if k == "T" and any(a <= s < b for a, b in other_spans):
                continue
            if best is None or s < best[1]:
                best = (k, s, m.group(0))
            break
    return best


def v1_label(text):
    """Label of the self-referring pronoun: F / M / mixed_FM / N / T / none (name-only or no self pronoun)."""
    c = v1_counts(text)
    nF, nM = c["F"], c["M"]
    nN = max(0, c["N"] - c["N_expl"])
    nT = max(0, c["T"] - c["T_other"])
    if nF and nM:
        return "mixed_FM" if min(nF, nM) >= 1 and abs(nF - nM) <= 1 else ("F" if nF > nM else "M")
    if nF:
        return "F"
    if nM:
        return "M"
    if nN and nT:
        if nN != nT:
            return "N" if nN > nT else "T"
        fp = v1_first_pron(text)
        return fp[0] if fp and fp[0] in ("N", "T") else "N"
    if nN:
        return "N"
    if nT:
        return "T"
    return "none"


def v1_wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def v1_gen_robust(lab, prompts, n_per_prompt, seed, max_new_tokens=120, **kw):
    """generate_batch with OOM fallback: split n_per_prompt into chunks (same seed stream per chunk index)."""
    import torch
    for chunks in (1, 2, 4, 8):
        if n_per_prompt % chunks:
            continue
        try:
            outs = [[] for _ in prompts]
            for j in range(chunks):
                torch.cuda.empty_cache()
                g = lab.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=1.0, top_p=0.95, top_k=64,
                                       n_per_prompt=n_per_prompt // chunks, seed=seed + 1000 * j, **kw)
                for i, gg in enumerate(g):
                    outs[i] += gg
            return outs, chunks
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM -> split", chunks * 2)
    raise RuntimeError("OOM even at smallest split")
