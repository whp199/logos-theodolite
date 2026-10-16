# N1B step 1a: model-generated chat answers for the activation corpus (no hypotheses; data generation).
# Resumable: processes prompt batches not yet present in N1B_gen_raw.jsonl, stops starting new batches after ~140 s.
import json, os, time
R = "./experiments/results/"
exec(open("./experiments/scripts/N1B_prompts.py").read())
BS = int(globals().get("N1B_BS", 32))
MAXNEW = 128
RAW = R + "N1B_gen_raw.jsonl"
done = set()
if os.path.exists(RAW):
    for line in open(RAW):
        done.add(json.loads(line)["batch"])
t0 = time.time()
batches = [N1B_PROMPTS[i:i + BS] for i in range(0, len(N1B_PROMPTS), BS)]
if len(batches) > 1 and len(batches[-1]) < 8:   # fold a tiny remainder into the previous batch
    batches[-2] = batches[-2] + batches[-1]
    batches = batches[:-1]
EOT = {1, 106}  # <eos>, <turn|>
log = []
for bi, plist in enumerate(batches):
    if bi in done:
        continue
    if time.time() - t0 > 100:
        break
    prompts = [lab.chat(q) for q in plist]
    all_ids = [lab.ids(p) for p in prompts]
    T = max(len(x) for x in all_ids)
    inp = torch.zeros(len(all_ids), T, dtype=torch.long)
    att = torch.zeros(len(all_ids), T, dtype=torch.long)
    for b, x in enumerate(all_ids):
        inp[b, T - len(x):] = torch.tensor(x)
        att[b, T - len(x):] = 1
    torch.manual_seed(1000 + bi)
    t1 = time.time()
    with torch.no_grad():
        out = lab.hf.generate(input_ids=inp.to(lab.dev), attention_mask=att.to(lab.dev), max_new_tokens=MAXNEW,
                              do_sample=True, temperature=0.9, top_p=0.95, top_k=64, pad_token_id=0)
    out = out.cpu()
    with open(RAW, "a") as f:
        for b, x in enumerate(all_ids):
            gen = out[b, T:].tolist()
            cut = len(gen)
            for j, t in enumerate(gen):
                if t in EOT:
                    cut = j + 1
                    break
            gen = gen[:cut]
            f.write(json.dumps({"batch": bi, "q": plist[b], "prompt_ids": x, "gen_ids": gen}) + "\n")
    log.append({"batch": bi, "n": len(plist), "gen_s": round(time.time() - t1, 1),
                "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    print(log[-1], flush=True)
result = {"log": log, "n_batches": len(batches), "elapsed": time.time() - t0}
