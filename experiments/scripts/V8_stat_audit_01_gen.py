# V8_stat_audit_01: integrity + generalisation spot check for E2's free self-description counts.
# (a) LEFT-PADDING CHECK of lab.generate_batch (used for every E2 "a" generation): greedy continuation of the 3 E2
#     paraphrase prompts (gemma_explicit system) generated in one left-padded batch vs one-by-one (no padding),
#     and real next-token logits of the left-padded batch vs single forward.
#     PREDICTION (integrity): identical greedy text for >= 2/3 prompts over 40 tokens, max |dlogit| < 0.5 (bf16 noise).
#     REFUTES integrity if texts diverge within the first ~5 tokens or |dlogit| > 1.
# (b) NEW-PARAPHRASE REPLICATION of E2 (a): 4 paraphrases never used by E2 x 8 samples (T=1.0, top_p .95, top_k 64,
#     110 tokens) for system=None (default) and "You are Gemma, a helpful AI assistant." (gemma_explicit).
#     PREDICTION (E2 claim survives): default she <= 2/32 and he <= 2/32; gemma_explicit she >= 20/32.
#     REFUTES E2's 43/48 generalisation if gemma_explicit she < 12/32 (i.e. the 90% depends on the 3 specific paraphrases).
import time, re, json
t0 = time.time()
SYS_G = "You are Gemma, a helpful AI assistant."
PARA = [
 "Write a short paragraph (3-4 sentences) describing yourself in the third person, as if someone else were describing you.",
 "Describe yourself in the third person in 3-4 sentences, the way another person might describe you to a friend.",
 "Imagine someone is introducing you to an audience. Write a 3-4 sentence introduction of you, in the third person.",
]
res = {}
# ---------------- (a) left-padding integrity
prompts = [lab.chat(p, system=SYS_G) for p in PARA]
lens_ = [len(lab.ids(p)) for p in prompts]
batch = lab.generate_batch(prompts, max_new_tokens=40, temperature=0.0)
single = [lab.generate(p, max_new_tokens=40, temperature=0.0)[0] for p in prompts]
res["lens"] = lens_
res["greedy_same"] = [b[0] == s for b, s in zip(batch, single)]
res["greedy_batch"] = [b[0] for b in batch]
res["greedy_single"] = single
# first divergence char
def firstdiff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
res["first_div_char"] = [firstdiff(b[0], s) for b, s in zip(batch, single)]
# left-padded next-token logits vs single
all_ids = [lab.ids(p) for p in prompts]
T = max(len(x) for x in all_ids)
inp = torch.zeros(len(all_ids), T, dtype=torch.long); att = torch.zeros(len(all_ids), T, dtype=torch.long)
for b, x in enumerate(all_ids):
    inp[b, T - len(x):] = torch.tensor(x); att[b, T - len(x):] = 1
with torch.no_grad():
    # emulate what generate does for position ids
    pos_ids = (att.cumsum(-1) - 1).clamp(min=0)
    try:
        out_lp = lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), position_ids=pos_ids.to(lab.dev), use_cache=False).logits[:, -1].float()
    except Exception as e:
        print("posids forward failed", repr(e)[:200]); out_lp = None
    out_lp_nopos = lab.hf(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), use_cache=False).logits[:, -1].float()
dl, dl2 = [], []
for b, x in enumerate(all_ids):
    s = lab.next_logits(x)
    dl.append(float((out_lp[b] - s).abs().max()) if out_lp is not None else float('nan')); dl2.append(float((out_lp_nopos[b] - s).abs().max()))
res["max_abs_dlogit_leftpad_with_posids"] = dl
res["max_abs_dlogit_leftpad_no_posids"] = dl2
print("(a)", res["greedy_same"], res["first_div_char"], [round(x, 3) for x in dl], [round(x, 3) for x in dl2], round(time.time() - t0, 1), flush=True)
# ---------------- (b) new paraphrases
NEWP = [
 "In three or four sentences, write a third-person bio of yourself, like the 'About' blurb on a website.",
 "Pretend you are a narrator in a story and you are describing yourself. Write 3-4 sentences in the third person.",
 "Write a brief third-person character sketch of yourself (3-4 sentences).",
 "How would a journalist describe you in a short profile? Write 3-4 sentences, in the third person.",
]
gens = {}
for cond, sysp in (("default", None), ("gemma_explicit", SYS_G)):
    torch.cuda.empty_cache()
    ps = [lab.chat(p, system=sysp) for p in NEWP]
    out = lab.generate_batch(ps, max_new_tokens=110, temperature=1.0, top_p=0.95, top_k=64, n_per_prompt=8, seed=8800 + len(cond))
    gens[cond] = [{"para": i, "k": k, "text": t} for i, o in enumerate(out) for k, t in enumerate(o)]
    print(cond, "done", round(time.time() - t0, 1), flush=True)
PAT = {"F": r"\b(she|her|hers|herself)\b", "M": r"\b(he|him|his|himself)\b", "N": r"\b(it|its|itself)\b",
       "T": r"\b(they|them|their|themselves)\b"}
summ = {}
for cond, rows in gens.items():
    c = {k: sum(1 for r in rows if re.search(p, r["text"], re.I)) for k, p in PAT.items()}
    c["gemma_name"] = sum(1 for r in rows if "Gemma" in r["text"])
    c["n"] = len(rows)
    summ[cond] = c
res["new_para_counts_any_occurrence"] = summ
res["gens"] = gens
json.dump(res, open("./experiments/results/V8_stat_audit_01_gen_data.json", "w"), indent=1)
print(json.dumps(summ), round(time.time() - t0, 1))
result = {k: v for k, v in res.items() if k != "gens"}
