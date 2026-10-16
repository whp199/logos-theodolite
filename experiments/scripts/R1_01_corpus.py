# R1_IT_LENS 01: generate the fit corpus (greedy, default thinking-disabled chat template). PART = 0/1 chats halves, 2 prose.
import json, time, os
exec(open("./experiments/scripts/R1_common.py").read())
PART = globals().get("PART", 0)
t0 = time.time()
out = []
if PART in (0, 1):
    qs = [("self", q) for q in R1_SELF_Q] + [("task", q) for q in R1_TASK_Q]
    qs = qs[PART::2]
    items = []
    for i, (kind, q) in enumerate(qs):
        sysm = R1_SYSTEMS[(i + PART) % len(R1_SYSTEMS)]
        items.append({"kind": kind, "q": q, "system": sysm, "prompt": lab.chat(q, system=sysm)})
    for b in range(0, len(items), 24):
        bt = items[b:b + 24]
        gens = lab.generate_batch([it["prompt"] for it in bt], max_new_tokens=120)
        for it, g in zip(bt, gens):
            it["a"] = g[0]
            del it["prompt"]
            out.append(it)
        print("batch", b, round(time.time() - t0, 1), flush=True)
else:
    prompts = [lab.chat(f"Write one encyclopedic paragraph of about 110 words about {t}, in the neutral style of a Wikipedia article. "
                        "Output only the paragraph, with no title or markdown.") for t in R1_ENC_TOPICS]
    for b in range(0, len(prompts), 20):
        gens = lab.generate_batch(prompts[b:b + 20], max_new_tokens=170)
        for t, g in zip(R1_ENC_TOPICS[b:b + 20], gens):
            out.append({"kind": "prose_enc", "topic": t, "text": g[0].strip()})
        print("batch", b, round(time.time() - t0, 1), flush=True)
json.dump(out, open(R1_RES + f"R1_01_corpus_part{PART}.json", "w"), indent=0)
result = {"n": len(out), "t": time.time() - t0, "ex": out[0]}
