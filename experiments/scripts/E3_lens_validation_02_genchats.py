# E3_lens_validation step 2: greedy answers (default, no system prompt, thinking disabled) for 24 chat questions.
import json, time
exec(open("./experiments/scripts/E3_lens_validation_data.py").read())
t = time.time()
prompts = [lab.chat(q) for q in CHAT_Q]
outs = lab.generate_batch(prompts, max_new_tokens=110, temperature=0.0)
chats = [{"q": q, "a": o[0].strip()} for q, o in zip(CHAT_Q, outs)]
json.dump(chats, open("./experiments/results/E3_lens_validation_chats.json", "w"), indent=1)
print("gen time", time.time() - t)
for c in chats[:24]:
    print(repr(c["a"][:160]))
result = chats
