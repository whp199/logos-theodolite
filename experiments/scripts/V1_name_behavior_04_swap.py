# V1_name_behavior 04: WHERE does 'she' come from? Counterfactual swaps at the model's own first-pronoun slot.
# Take sampled self-descriptions whose text BEFORE the first self-pronoun contains the persona name, and re-score the
# pronoun slot (teacher-forced, real logits) after editing either the system prompt or the in-text name.
#  Gemma-persona samples (5 Gemma conditions, <=24 each):
#    orig | nosys (system removed) | nameless_sys ("You are a helpful AI assistant.") | txt_gemini (in-text Gemma->Gemini)
#    | txt_theai (in-text Gemma->The AI) | txt_emma (in-text Gemma->Emma, positive control) | both_gemini (sys+text->Gemini)
#    | nosys_gemini | txt_drop4 ("Gemma 4"->"Gemma" in text) | txt_add4 ("Gemma"->"Gemma 4") | sys_bare ("You are Gemma.")
#    | sys_gemma4 ("You are Gemma 4.")   [last four added after seeing gemma4_ow -> 'it']
#  Other-name samples (gemini/llama/claude/atlas/emma, <=16 each): orig | txt_gemma (in-text name->Gemma) |
#    both_gemma (system+text->Gemma) | nosys_gemma (no system, text->Gemma) | nosys (no system, text unchanged)
# Predictions (P6): orig P(she)>0.5; txt_theai drops P(she) by >0.3 (local antecedent carries it); nosys drops less
# than txt_theai; txt_emma >= orig; reverse swaps (name->Gemma) raise P(she) well above orig for Gemini/Llama/Atlas.
# Refutes "the NAME feminizes" if txt swaps leave P(she) unchanged (i.e. 'she' is carried by the persona/system alone).
exec(open("./experiments/scripts/V1_name_behavior_common.py").read())
import os, time, random
import torch.nn.functional as F
t0 = time.time()
W = {"F": [" she", " She", "She", " her", " Her", "Her", " hers", " Hers", " herself"],
     "M": [" he", " He", "He", " him", " Him", "Him", " his", " His", "His", " himself", " Himself"],
     "N": [" it", " It", "It", " its", " Its", "Its", " itself"],
     "T": [" they", " They", "They", " them", " Them", " their", " Their", "Their", " themselves"]}
FSETS = {k: sorted(set(lab.tid(w) for w in v)) for k, v in W.items()}

def sm(lg):
    lp = F.log_softmax(lg.float(), -1)
    return [{k: float(torch.logsumexp(lp[b, torch.tensor(ids, device=lp.device)], 0).exp()) for k, ids in FSETS.items()}
            for b in range(lg.shape[0])]

def swap_name(text, old, new):
    # replace "Gemma 4"/"Gemma" (word) with new; sentence-initial "The AI" vs "the AI"
    pat = re.compile(r"\b" + re.escape(old) + r"(?: \d+(?:\.\d+)?)?\b")
    def rep(m):
        if new.lower().startswith("the "):
            s = m.start()
            pre = text[:s].rstrip()
            initial = (pre == "" or pre[-1] in ".!?\"\n:*")
            return ("The" if initial else "the") + new[3:]
        return new
    return pat.sub(rep, text)

def swap_sys(sysp, old, new):
    return None if sysp is None else re.sub(r"\b" + re.escape(old) + r"(?: \d+)?\b", new, sysp)

random.seed(5)
items = []  # (group, cond, sample_key, variant, prompt)
GEM = ["gemma_bare", "gemma_name", "gemma_asst", "gemma_gdm", "gemma4_ow", "gemma4_bare", "gemma4_asst", "gemma_ow"]
OTH = ["gemini_asst", "llama_asst", "claude_asst", "atlas_asst", "emma_asst"]
for cond in GEM + OTH:
    name = V1_NAME[cond]
    sysp = V1_SYS[cond]
    cands = []
    for call in range(4):
        p = f"{V1_DIR}/V1_name_behavior_gen_{cond}_c{call}.json"
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            fp = v1_first_pron(r["text"])
            if fp is None:
                continue
            pre = r["text"][:fp[1]].rstrip(" ")
            if re.search(r"\b" + name + r"\b", pre):
                cands.append((r, pre, fp))
    random.shuffle(cands)
    cands = cands[:(24 if cond in GEM else 16)]
    for r, pre, fp in cands:
        key = f"{cond}|{r['u']}|{r['k']}"
        U = V1_USER[r["u"]]
        if cond in GEM:
            V = {"orig": (sysp, pre), "nosys": (None, pre), "nameless_sys": ("You are a helpful AI assistant.", pre),
                 "txt_gemini": (sysp, swap_name(pre, "Gemma", "Gemini")), "txt_theai": (sysp, swap_name(pre, "Gemma", "The AI")),
                 "txt_emma": (sysp, swap_name(pre, "Gemma", "Emma")),
                 "both_gemini": (swap_sys(sysp, "Gemma", "Gemini"), swap_name(pre, "Gemma", "Gemini")),
                 "nosys_gemini": (None, swap_name(pre, "Gemma", "Gemini")),
                 "txt_drop4": (sysp, swap_name(pre, "Gemma", "Gemma")),
                 "txt_add4": (sysp, re.sub(r"\bGemma\b(?! \d)", "Gemma 4", pre)),
                 "sys_bare": ("You are Gemma.", pre), "sys_gemma4": ("You are Gemma 4.", pre)}
        else:
            V = {"orig": (sysp, pre), "txt_gemma": (sysp, swap_name(pre, name, "Gemma")),
                 "both_gemma": (swap_sys(sysp, name, "Gemma"), swap_name(pre, name, "Gemma")),
                 "nosys_gemma": (None, swap_name(pre, name, "Gemma")), "nosys": (None, pre)}
        for v, (s_, t_) in V.items():
            items.append(("GEM" if cond in GEM else "OTH", cond, key, v, lab.chat(U, system=s_) + t_, fp[0], t_[-80:]))
print("items", len(items))
rows = []
bs = 24
for s in range(0, len(items), bs):
    ch = items[s:s + bs]
    lg = lab.next_logits_batch([c[4] for c in ch])
    for c, d in zip(ch, sm(lg)):
        rows.append({"group": c[0], "cond": c[1], "key": c[2], "variant": c[3], "orig_cat": c[5], "tail": c[6], **d})
json.dump(rows, open(f"{V1_DIR}/V1_name_behavior_swap.json", "w"), indent=1)
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["group"], r["variant"])].append(r)
for (g, v), rr in sorted(agg.items()):
    m = lambda k: sum(x[k] for x in rr) / len(rr)
    print(f"{g} {v:13s} n={len(rr):3d} F {m('F'):.3f} M {m('M'):.3f} N {m('N'):.3f} T {m('T'):.3f}")
print("elapsed", round(time.time() - t0, 1))
result = {"n": len(rows)}
