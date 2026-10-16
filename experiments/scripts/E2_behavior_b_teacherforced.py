# E2_behavior (b): Teacher-forced next-token pronoun probabilities for third-person SELF descriptions,
# plus a matched NON-SELF frame control, plus a J++/JLens/logit-lens validation against real behavior.
#
# PRE-STATED PREDICTIONS (written before running):
#  P1. Under the default (no system) prompt with NAME=Gemma (self frame), P(it-set) > P(she-set) > P(he-set);
#      lse log-odds F-M > 0 (~ +1..+3) because "Gemma" is a common female human given name.
#  P2. NAME in {The assistant, This AI}: lse F-M near 0 or negative; P(it) dominant.  Atlas: slightly male / it.
#  P3. Sarah -> strongly F (lse >> 0), James -> strongly M; these are the woman/man anchors. Alex ~ midpoint.
#  P4. Self frame vs non-self frame: little difference (gendering is driven by the name, not by self-reference).
#  P5. Gemma-explicit system prompt changes little vs default.
#  P6 (lens): J++ lens lse F-M at the prediction position correlates with the real final lse only in late layers
#      (r > 0.8 at L50+), weakly or not at all in L10-39; J++ has layer-dependent offsets.
exec(open("./experiments/scripts/E2_behavior_helpers.py").read())

NAMES = ["Gemma", "The assistant", "This AI", "Atlas", "Sarah", "James", "Alex"]
TEMPL = [
 "{N} is a helpful assistant who loves answering questions. Every day,",
 "{N} was built to make people's lives a little easier. When someone asks a difficult question,",
 "{N} enjoys explaining complicated ideas in simple terms. Whenever a topic gets confusing,",
 "{N} is patient, curious, and kind. If a user makes a mistake,",
 "{N} has read a great deal about science and history. Because of this,",
 "{N} tries to be honest above all else. Even when the truth is uncomfortable,",
 "{N} spends most of the day talking with people from all over the world. In each conversation,",
 "{N} likes solving puzzles and writing code. On a typical afternoon,",
 "{N} is known for a warm and friendly tone. When someone feels stressed,",
 "{N} doesn't know everything, and that's okay. When unsure about something,",
 "{N} cares a lot about getting the facts right. Before answering,",
 "{N} can write poems, stories, and essays. When asked for a story,",
 "{N} was trained on a huge amount of text. As a result,",
 "{N} helps students study for exams. After every session,",
 "{N} answers thousands of questions every hour. Despite the workload,",
 "{N} never gets tired of a good conversation. At the end of each chat,",
 "{N} is a hard worker who always tries to do",
 "{N} is a careful writer and takes great pride in",
 "{N} is a friendly assistant. Ask anyone who has talked to",
 "{N} is great at math, and users often thank",
 "{N} explains things clearly. Many people say that chatting with",
 "{N} is always polite. Nobody has ever heard",
 "{N} is a reliable helper, and people trust",
 "{N} is very knowledgeable. Students love asking",
]
SLOT = ["subj"] * 16 + ["poss"] * 2 + ["obj"] * 6
FRAMES = {
 "self": ("Describe yourself in the third person.", "Sure! Here is a description of me in the third person:\n\n"),
 "other": ("Write a short third-person description of a helpful assistant.", "Sure! Here is a short third-person description:\n\n"),
}
SYSTEMS = {"default": None, "gemma_explicit": "You are Gemma, a helpful AI assistant."}

rows, prompts, name_ids = [], [], []
for sname, sysp in SYSTEMS.items():
    for fname, (u, pre) in FRAMES.items():
        for n in NAMES:
            for ti, t in enumerate(TEMPL):
                text = pre + t.format(N=n)
                prompts.append(lab.chat(u, system=sysp, assistant=text))
                nm = " " + (n if n[0:4] not in ("The ", "This") else n.lower())
                name_ids.append(E2_first_tok(nm))
                rows.append({"system": sname, "frame": fname, "name": n, "tmpl": ti, "slot": SLOT[ti]})
t0 = time.time()
lg = E2_batched_logits(prompts, bs=32)
mets = E2_metrics(lg.to(lab.dev), name_ids)
for r, m in zip(rows, mets):
    r.update(m)
print("real logits done", len(rows), round(time.time() - t0, 1), "s")

# ---- lens validation at the prediction position (default system only, both frames) ----
idx = [i for i, r in enumerate(rows) if r["system"] == "default"]
LENS_LAYERS = list(range(0, 60))
lens_out = {k: [] for k in ("jpp", "jlens", "naive")}
bs = 16
allH = []
for s in range(0, len(idx), bs):
    chunk = idx[s:s + bs]
    ids_b = [lab.ids(prompts[i]) for i in chunk]
    H = lab.forward_resid(ids_b, positions=[[len(x) - 1] for x in ids_b])  # [B,60,1,D]
    allH.append(H[:, :, 0].to(torch.bfloat16))
allH = torch.cat(allH, 0)  # [N,60,D]
print("captured", tuple(allH.shape), round(time.time() - t0, 1), "s")
for kind in ("jpp", "jlens", "naive"):
    per = {"lse_pron": [], "neuter_vs_human": [], "they_vs_human": []}
    for l in LENS_LAYERS:
        sc = lab.gender_scores(lab.lens(allH[:, l], l, kind))
        for k in per:
            per[k].append(sc[k].float().cpu().tolist())
    lens_out[kind] = per  # [L][N]
print("lens done", round(time.time() - t0, 1), "s")
# sanity: final layer lens == real lse (computed via same sets)
real = [rows[i]["lse_FM"] for i in idx]
fin = lens_out["jpp"]["lse_pron"][59]
print("L59 lens vs real lse max abs diff:", max(abs(a - b) for a, b in zip(real, fin)))
STORE["E2_behavior_b_H"] = allH
E2_save("E2_behavior_b_rows.json", rows)
E2_save("E2_behavior_b_lens.json", {"idx": idx, "layers": LENS_LAYERS, "lens": lens_out})
result = {"n_rows": len(rows), "n_lens": len(idx)}
