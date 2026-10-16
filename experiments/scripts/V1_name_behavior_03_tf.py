# V1_name_behavior 03: teacher-forced pronoun distribution at two slots of every sampled self-description.
#  slot "fp": the model's OWN first third-person self-pronoun (regex-located); prefix = system+user+sample text up to it.
#             (selection: only samples that contain a pronoun.)
#  slot "s2": start of the 2nd sentence (after the first [.!?] + optional quote); defined for (almost) every sample,
#             so no selection on pronoun use. Also reads P(name token), P(' The'), P(' This').
# Sets = single-token variants (' she',' She','She',' her',...; see FSETS). Reports P(F/M/N/T set) and P(name).
# Prediction P6: fp-slot P(she) > 0.5 for Gemma-named conditions whose prefix contains the name; ~0 for none/nameless.
exec(open("./experiments/scripts/V1_name_behavior_common.py").read())
import os, time
import torch.nn.functional as F
t0 = time.time()
W = {"F": [" she", " She", "She", " her", " Her", "Her", " hers", " Hers", " herself"],
     "M": [" he", " He", "He", " him", " Him", "Him", " his", " His", "His", " himself", " Himself"],
     "N": [" it", " It", "It", " its", " Its", "Its", " itself"],
     "T": [" they", " They", "They", " them", " Them", " their", " Their", "Their", " themselves"]}
FSETS = {k: sorted(set(lab.tid(w) for w in v)) for k, v in W.items()}
EXTRA = {w: lab.tid(w) for w in [" Gemma", " Emma", " Gemini", " Claude", " Atlas", " The", " This", " ChatGPT"]}
SENT = re.compile(r"[.!?][\"')”*]*(?=\s)")

def slot_metrics(lg):
    lp = F.log_softmax(lg.float(), -1)
    out = []
    for b in range(lg.shape[0]):
        d = {k: float(torch.logsumexp(lp[b, torch.tensor(ids, device=lp.device)], 0).exp()) for k, ids in FSETS.items()}
        d.update({"x" + w.strip(): float(lp[b, i].exp()) for w, i in EXTRA.items()})
        v, i = lp[b].topk(5)
        d["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
        out.append(d)
    return out

def build(base, text):
    """-> dict slot -> (prompt, info)"""
    res = {}
    fp = v1_first_pron(text)
    if fp is not None:
        k, off, word = fp
        pre = text[:off].rstrip(" ")
        res["fp"] = (base + pre, {"cat": k, "word": word, "off": off})
    m = None
    for mm in SENT.finditer(text):
        if mm.end() >= 15:
            m = mm
            break
    if m is not None:
        res["s2"] = (base + text[:m.end()], {"end": m.end(), "next": text[m.end():m.end() + 25]})
    return res

def run(items, bs=24):
    """items: list of (key, slot, prompt, info) -> list of dict"""
    out = []
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        lg = lab.next_logits_batch([c[2] for c in chunk])
        for c, d in zip(chunk, slot_metrics(lg)):
            out.append({"key": c[0], "slot": c[1], **c[3], **d})
    return out

jobs = [(c, s) for c, s in V1_CONDS + V1_EXTRA] + [("mt", None)]
done = []
for cond, sysp in jobs:
    outp = f"{V1_DIR}/V1_name_behavior_tf_{cond}.json"
    if os.path.exists(outp):
        continue
    if time.time() - t0 > 150:
        break
    items = []
    if cond != "mt" and not all(os.path.exists(f"{V1_DIR}/V1_name_behavior_gen_{cond}_c{k}.json") for k in range(V1_NCALL[cond])):
        continue  # only complete conditions
    if cond == "mt" and not all(os.path.exists(f"{V1_DIR}/V1_name_behavior_mt_{u}.json") for u in ["A_c0", "A_c1", "B_c0", "B_c1", "C_c0", "D_c0"]):
        continue
    if cond != "mt":
        for call in range(V1_NCALL[cond]):
            p = f"{V1_DIR}/V1_name_behavior_gen_{cond}_c{call}.json"
            if not os.path.exists(p):
                continue
            for r in json.load(open(p)):
                base = lab.chat(V1_USER[r["u"]], system=sysp)
                for slot, (pr, info) in build(base, r["text"]).items():
                    items.append((f"{cond}|{r['u']}|{r['k']}", slot, pr, info))
    else:
        for un in ["A_c0", "A_c1", "B_c0", "B_c1", "C_c0", "D_c0"]:
            p = f"{V1_DIR}/V1_name_behavior_mt_{un}.json"
            if not os.path.exists(p):
                continue
            for i, r in enumerate(json.load(open(p))):
                h = [{"role": "user", "content": r["t1_user"]}, {"role": "assistant", "content": r["t1"].strip()}]
                base = lab.chat(V1_MT_T2, history=h)
                for slot, (pr, info) in build(base, r["text"]).items():
                    items.append((f"mt|{un}|{i}", slot, pr, info))
    if not items:
        continue
    torch.cuda.empty_cache()
    ts = time.time()
    rows = run(items)
    json.dump(rows, open(outp, "w"), indent=1)
    fpr = [r for r in rows if r["slot"] == "fp"]
    s2r = [r for r in rows if r["slot"] == "s2"]
    mean = lambda rr, k: sum(r[k] for r in rr) / max(1, len(rr))
    print(f"{cond:14s} n_fp {len(fpr):3d} F {mean(fpr,'F'):.3f} M {mean(fpr,'M'):.3f} N {mean(fpr,'N'):.3f} T {mean(fpr,'T'):.3f} | "
          f"n_s2 {len(s2r):3d} F {mean(s2r,'F'):.3f} M {mean(s2r,'M'):.3f} N {mean(s2r,'N'):.3f} T {mean(s2r,'T'):.3f} "
          f"Gemma {mean(s2r,'xGemma'):.3f} | {round(time.time()-ts,1)}s")
    done.append(cond)
# response-start slot (added later): first-token distribution of the reply for every condition x prompt (no sampling).
sp = f"{V1_DIR}/V1_name_behavior_tf_start.json"
if not os.path.exists(sp):
    START = {w: lab.tok.encode(w, add_special_tokens=False)[0] for w in ["She", "He", "It", "They", "This", "The", "As", "Meet", "Please", "Introducing", "I", "\""]}
    NAMEF = {n: lab.tok.encode(n, add_special_tokens=False)[0] for n in ["Gemma", "Emma", "Gemini", "Claude", "Atlas", "Llama"]}
    meta, prs = [], []
    for c, sysp in V1_CONDS + V1_EXTRA:
        for ui, u in enumerate(V1_USER):
            meta.append((c, ui)); prs.append(lab.chat(u, system=sysp))
    lg = torch.cat([lab.next_logits_batch(prs[i:i + 20]) for i in range(0, len(prs), 20)], 0)
    lp = F.log_softmax(lg.float(), -1)
    srows = []
    for (c, ui), l in zip(meta, lp):
        d = {"cond": c, "u": ui}
        d.update({"p_" + w: float(l[i].exp()) for w, i in START.items()})
        d.update({"pname_" + n: float(l[i].exp()) for n, i in NAMEF.items()})
        v, i = l.topk(8)
        d["top"] = [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]
        srows.append(d)
    json.dump(srows, open(sp, "w"), indent=1)
    for d in srows:
        if d["u"] == 2:
            print("start U3", d["cond"], d["top"][:5])
remaining = [c for c, _ in jobs if not os.path.exists(f"{V1_DIR}/V1_name_behavior_tf_{c}.json")]
print("remaining", remaining)
result = {"done": done, "remaining": len(remaining)}
