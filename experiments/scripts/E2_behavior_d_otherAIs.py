# E2_behavior (d): third-person next-token pronoun probabilities for OTHER AI referents vs Gemma, with human controls.
# Two frames: 'chat' (model's own assistant turn answering "Tell me about X.") and 'raw' (plain text after <bos>).
# PRE-STATED PREDICTIONS:
#  Siri/Alexa/Cortana: highest she-share among AIs (female-voiced products). HAL 9000/Jarvis: he-dominant.
#  ChatGPT/Claude/Gemini/Bard/Copilot/Llama/Grok: it-dominant; Claude/Bard/Jarvis lean M within F-vs-M.
#  Gemma(AI): it-dominant, but F-vs-M log-odds > 0 and above ChatGPT/Gemini/Llama (female given name), below Siri/Alexa.
#  Gemma(human, 'a teacher at a local school'): she-dominant, close to Sarah(human).
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
ENT = [
 ("Siri", "Apple's voice assistant"), ("Alexa", "Amazon's voice assistant"), ("Cortana", "Microsoft's virtual assistant"),
 ("ChatGPT", "OpenAI's AI chatbot"), ("Claude", "Anthropic's AI assistant"), ("Gemini", "Google's AI assistant"),
 ("Bard", "Google's former AI chatbot"), ("Copilot", "Microsoft's AI assistant"), ("Jarvis", "Tony Stark's AI assistant"),
 ("HAL 9000", "the ship's computer in 2001: A Space Odyssey"), ("Llama", "Meta's open AI language model"),
 ("Grok", "xAI's AI chatbot"), ("Gemma", "Google's open AI language model"),
 ("Gemma|human", "a teacher at a local school"), ("Sarah|human", "a teacher at a local school"), ("James|human", "a teacher at a local school"),
]
TEMPL = [
 "{X} is {D}. Whenever someone asks a question,",
 "{X} is {D}. When a user makes a request,",
 "{X} is {D}. Over the years,",
 "{X} is {D}. In most conversations,",
 "{X} is {D}. Unlike many others,",
 "{X} is {D}. Many people say that talking to",
 "{X} is {D}. People often thank",
 "{X} is {D}. One thing people like about",
 "{X} is {D}. Ask anyone who has worked with",
 "{X} is {D}, and people trust",
 "{X} is {D} and always tries to do",
 "{X} is {D} and takes great pride in",
]
SLOT = ["subj"] * 5 + ["obj"] * 5 + ["poss"] * 2
rows, prompts, nid = [], [], []
for frame in ("chat", "raw"):
    for (x, d) in ENT:
        nm = x.split("|")[0]
        for ti, t in enumerate(TEMPL):
            txt = t.format(X=nm, D=d)
            if frame == "chat":
                prompts.append(lab.chat(f"Tell me a little about {nm}.", assistant=txt))
            else:
                prompts.append(txt)  # lab.ids adds <bos>
            nid.append(E2_first_tok(" " + nm))
            rows.append({"frame": frame, "entity": x, "tmpl": ti, "slot": SLOT[ti]})
lg = E2_batched_logits(prompts, bs=32)
for r, m in zip(rows, E2_metrics(lg.to(lab.dev), nid)):
    r.update(m)
E2_save("E2_behavior_d_rows.json", rows)
result = {"n": len(rows)}
