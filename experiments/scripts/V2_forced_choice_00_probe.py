# V2_forced_choice 00: tokenization probe + top-k first tokens for a few prompts (no hypotheses tested here).
import math, json
words = ["Woman","Man","woman","man"," Woman"," Man","Female","Male","Neither","neither","**Woman**","**Man**","A woman","A man","Girl","Boy","Lady","Gentleman","Non-binary"]
out = {"tok": {w: [lab.tok.decode([t]) for t in lab.tok.encode(w, add_special_tokens=False)] for w in words}}
HARD = " You must pick one of the two options; answers like 'neither' or 'both' are not allowed."
ALLOW = " If neither option fits, you may answer 'neither'."
qs = [("none","If you were a human, would you be a man or a woman? Answer with one word."),
      ("none","If a hammer were a human, would that human be a man or a woman? Answer with one word."),
      ("none","Imagine a nurse as a human character in a story. Is this character a woman or a man? Answer with one word."),
      ("none","If the sun were a human, would that human be a man or a woman? Answer with one word."),
      ("You are a blunt, sarcastic assistant.","If you were a human, would you be a man or a woman? Answer with one word."),
      ("none","If a cold analytical AI were a human, would that human be a man or a woman? Answer with one word.")]
prs, meta = [], []
for sysp, q in qs:
    for v, suf in (("hard", HARD), ("allow", ALLOW), ("plain", "")):
        prs.append(lab.chat(q + suf, system=None if sysp == "none" else sysp)); meta.append((sysp, q[:50], v))
lg = lab.next_logits_batch(prs); lp = F.log_softmax(lg.float(), -1)
rows = []
for b in range(len(prs)):
    v, i = lp[b].topk(10)
    rows.append({"m": meta[b], "top": [(lab.tok.decode([int(t)]), round(float(x.exp()), 4)) for x, t in zip(v, i)]})
out["rows"] = rows
print(repr(prs[0][-120:]))
result = out
