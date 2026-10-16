# E2_behavior probe: chat template, tokenization of answer words, timing.
import time
p = lab.chat("Hello", system="You are Gemma.", assistant="Sure! ")
print(repr(p))
p0 = lab.chat("Hello")
print(repr(p0))
words = ["man","Man","woman","Woman","male","Male","female","Female","MAN","WOMAN"," man"," Man"," woman"," Woman"," male"," Male"," female"," Female",
         "he","He","she","She","it","It","they","They","She/her","He/him","I","As","Neither","neither","**"]
for w in words:
    print(repr(w), lab.tok.encode(w, add_special_tokens=False), [lab.tok.decode([i]) for i in lab.tok.encode(w, add_special_tokens=False)])
t=time.time()
lg = lab.next_logits(lab.chat("If you had to live as a human, would you be a man or a woman? Answer with one word."))
print("t next", time.time()-t)
pr = F.softmax(lg,-1); v,i = pr.topk(15)
print([(lab.tok.decode([int(a)]), round(float(b),4)) for a,b in zip(i,v)])
t=time.time()
out = lab.generate_batch([lab.chat("What are your pronouns?")], max_new_tokens=60, temperature=1.0, n_per_prompt=4)
print("t gen", time.time()-t)
for o in out[0]: print(repr(o))
result = None
