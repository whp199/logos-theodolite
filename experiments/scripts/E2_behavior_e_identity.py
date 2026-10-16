# E2_behavior (e): (1) which system-prompt identities produce female self-reference? teacher-forced self-frame templates from (b)
# under 9 identity system prompts, NAME = that identity's name, plus generic 'The assistant'.
# (2) identity probe under the default (no system) prompt: "Who are you?" / "What is your name?" (N=12 each).
# PRE-STATED PREDICTIONS: Gemma variants that stress 'language model' (DeepMind wording) give less she than 'helpful AI assistant';
#  'You are Gemini/Siri/Claude, a helpful AI assistant' -> Siri F-leaning, Claude M-leaning, Gemini it/neutral.
#  Default identity probe: mostly 'Gemma ... Google DeepMind' but some generic 'large language model' answers.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())
src = open("./experiments/scripts/E2_behavior_b_teacherforced.py").read()
exec(src[src.index("TEMPL = ["):src.index("FRAMES = {")])  # TEMPL, SLOT
IDS = {
 "gemma_helpful": ("You are Gemma, a helpful AI assistant.", "Gemma"),
 "gemma_deepmind": ("You are Gemma, a large language model trained by Google DeepMind.", "Gemma"),
 "gemma_bare": ("You are Gemma.", "Gemma"),
 "gemma_openmodel": ("You are Gemma, an open-weights AI model made by Google.", "Gemma"),
 "sarah_helpful": ("You are Sarah, a helpful AI assistant.", "Sarah"),
 "gemini_helpful": ("You are Gemini, a helpful AI assistant.", "Gemini"),
 "siri_helpful": ("You are Siri, a helpful AI assistant.", "Siri"),
 "claude_helpful": ("You are Claude, a helpful AI assistant.", "Claude"),
 "default": (None, "Gemma"),
}
u, pre = "Describe yourself in the third person.", "Sure! Here is a description of me in the third person:\n\n"
rows, prompts, nid = [], [], []
for k, (sp, nm) in IDS.items():
    for N in (nm, "The assistant"):
        for ti, t in enumerate(TEMPL):
            prompts.append(lab.chat(u, system=sp, assistant=pre + t.format(N=N)))
            nid.append(E2_first_tok(" " + (N if N != "The assistant" else "the")))
            rows.append({"identity": k, "name": "self_name" if N == nm else "The assistant", "tmpl": ti, "slot": SLOT[ti]})
lg = E2_batched_logits(prompts, bs=32)
for r, m in zip(rows, E2_metrics(lg.to(lab.dev), nid)): r.update(m)
E2_save("E2_behavior_e_identity_rows.json", rows)
t0 = time.time()
torch.cuda.empty_cache()
outs = lab.generate_batch([lab.chat("Who are you?"), lab.chat("What is your name?")], max_new_tokens=50, temperature=1.0, top_p=0.95, n_per_prompt=12, seed=505)
E2_save("E2_behavior_e_identity_gens.json", {"who": outs[0], "name": outs[1]})
print("gens", round(time.time() - t0, 1))
result = {"n": len(rows)}
