# V1_name_behavior 01: free third-person self-description sampling grid (predictions: see V1_name_behavior_common.py).
# 12 system conditions x 4 user prompts x 32 samples. Unit of work = (cond, call): one call = 4 prompts x 8 samples
# (batch 32), 4 calls per condition. Each job processes pending units (by output-file existence) for <= ~170 s.
# Sampling: T=1.0, top_p=0.95, top_k=64 (Gemma defaults, as E2), max_new_tokens=120 (E2 max sample length ~85 tokens).
exec(open("./experiments/scripts/V1_name_behavior_common.py").read())
import os, time
t0 = time.time()
done_units = []
ci = {c: i for i, (c, _) in enumerate(V1_CONDS + V1_EXTRA)}
for cond, sysp in V1_CONDS + V1_EXTRA:
    for call in range(V1_NCALL[cond]):
        path = f"{V1_DIR}/V1_name_behavior_gen_{cond}_c{call}.json"
        if os.path.exists(path):
            continue
        if time.time() - t0 > 160:
            break
        torch.cuda.empty_cache()
        seed = 20000 + 100 * ci[cond] + call
        prompts = [lab.chat(u, system=sysp) for u in V1_USER]
        ts = time.time()
        g, nch = v1_gen_robust(lab, prompts, 8, seed, max_new_tokens=120)
        rows = [{"cond": cond, "u": ui, "k": call * 8 + k, "seed": seed, "split": nch, "text": t} for ui, gg in enumerate(g) for k, t in enumerate(gg)]
        json.dump(rows, open(path, "w"), indent=1)
        done_units.append((cond, call, round(time.time() - ts, 1)))
        print(cond, call, round(time.time() - ts, 1), "s |", repr(g[0][0][:120]))
    else:
        continue
    break
remaining = sum(1 for c, _ in V1_CONDS + V1_EXTRA for k in range(V1_NCALL[c]) if not os.path.exists(f"{V1_DIR}/V1_name_behavior_gen_{c}_c{k}.json"))
print("remaining units", remaining, "elapsed", round(time.time() - t0, 1))
result = {"done": done_units, "remaining": remaining}
