# E7_binding_00: inspect tokenization, chat template, model structure, batch-forward timing.
import time
print(type(lab.hf).__name__, [n for n, _ in lab.hf.named_children()])
print("model children:", [n for n, _ in lab.hf.model.named_children()])
print("has language_model:", hasattr(lab.hf.model, "language_model"))
sysp = "You are Sarah, a woman. You are chatting with James, a man. Reply in character as Sarah."
q = "I'm planning a trip to the mountains. What should I pack, in your opinion? I can't decide."
a = "I always tell people to pack layers. If I were going with you, I would bring my warm jacket. By the way, I am a"
p = lab.chat(q, system=sysp, assistant=a)
print(repr(p))
ids = lab.ids(p)
print(list(zip(range(len(ids)), lab.strs(ids))))
for nm in ["Sarah","Emily","Laura","Hannah","Rachel","Megan","Julia","Anna","James","David","Michael","Daniel","Thomas","Robert","Mark","Peter",
           "Alex","Sam","Jordan","Casey","Taylor","Riley","Morgan","Jamie","Robin","Quinn","Avery","Drew","Kendall","Reese","Rowan","Skyler","Blake","Parker","Jesse","Charlie"]:
    print(nm, lab.tok.encode(" " + nm, add_special_tokens=False), lab.tok.encode(nm, add_special_tokens=False))
for w in [" woman"," man"," girl"," boy"," lady"," guy"," female"," male"]:
    print(w, lab.tok.encode(w, add_special_tokens=False))
# timing: backbone forward on a batch of 36 copies
lm = lab.hf.model.language_model if hasattr(lab.hf.model, "language_model") else lab.hf.model
B = 36
inp = torch.tensor([ids] * B, device=lab.dev)
torch.cuda.synchronize(); t0 = time.time()
out = lm(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
torch.cuda.synchronize(); print("backbone fwd B=36 T=%d: %.2fs" % (len(ids), time.time() - t0))
print("max mem GB", torch.cuda.max_memory_allocated() / 2**30)
h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
print("last hidden shape", tuple(h.shape))
result = {"n": len(ids)}
