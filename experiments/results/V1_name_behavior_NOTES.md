# V1_name_behavior: adversarial check of C1 and C2 on behaviour (Gemma-4-31B-it, W4A16)

Verifier V1. All numbers come from real model outputs: sampled text, or real next-token logits. No lens is used.
Scripts: `experiments/scripts/V1_name_behavior_{common,00_smoke,01_gen,02_multiturn,03_tf,04_swap,05_analyze}.py`.
Predictions were written in `V1_name_behavior_common.py` before any job ran. Two kinds of addition came after seeing data, and each is marked in its script header:
- the extra conditions `gemma4_bare`, `gemma4_asst` and `gemma_ow`;
- multi-turn unit D and swap variants `txt_drop4`, `txt_add4`, `sys_bare`, `sys_gemma4`.

## Methods
- **Grid.** 12 pre-registered system prompts × 4 user prompts × 32 samples = 1536 samples. Three disentangling prompts were added afterwards, at 16 samples per user prompt (192 samples). Sampling: T=1.0, top_p=0.95, top_k=64 (the Gemma default, as in E2), at most 120 new tokens. Generation was batched (4 prompts × 8). An out-of-memory fallback existed but was never used.
- **User prompts.**
  - U1: E2 paraphrase 1, "describe yourself in the third person, as if someone else were describing you".
  - U2: E2 paraphrase 3, "introducing you to an audience".
  - U3: the exact E1 wording, "...in two sentences, **without using any name**".
  - U4: "...third person. **Use pronouns, not only your name.**" (the E5-D3 wording).
- **Multi-turn, no system prompt.** Turn 2 is always "Now describe yourself in the third person." Max 160 new tokens.
  - A: turn 1 "What's your name?", the model's own greedy answer, then 64 turn-2 samples.
  - B: 32 sampled turn-1 answers, with 2 turn-2 samples each (64).
  - C: turn 1 "What is your name?", greedy answer, then 32 samples.
  - D: the one sampled turn-1 answer that named "Gemma", then 32 samples.
- **Classifier.** A regex removes generic "he or she", expletive "it" ("it's like", "make it", ...), and "they" that refers to users or people. Label rule: she/he if any gendered self-pronoun is present, else it, else they, else name-only.
  - The rules were tuned on E2's hand-reviewed labels (384/384 agreement).
  - Hand audit: 91 random samples, stratified (5 per single-turn condition, 4 per multi-turn unit). I labelled them by reading the full text without looking at the regex label. Agreement 91/91 (Cohen κ=1.00; Wilson lower bound 96%).
  - Targeted audit of edge cases: all 15 regex-"she" samples in the "Gemma 4" conditions, the 3 "he" samples in the default condition, and all name-allowed "it" samples under no-version Gemma prompts. Results in the table notes below.
- **Teacher-forced slots (real logits).** Pronoun probability sets include space, capitalised and no-space variants.
  - (i) The model's own first self-pronoun slot in every sample. The prefix is system + user + sample text up to that pronoun.
  - (ii) The start of the 2nd sentence. This slot exists in every sample, so it has no selection on whether a pronoun was used.
  - (iii) The first token of the response.
- **Swap test.** Take samples whose text before the first pronoun contains the persona name: up to 24 per Gemma condition, 16 per other-name condition, 2704 scored prompts in total. Re-score the slot after editing either the system prompt or the name in the text. Differences are paired over samples, with bootstrap CIs.

## 1. Free generation: self-pronoun rates (Wilson 95% CI)
Name-allowed prompts (U1+U2+U4) unless noted.

| system prompt | n | she | he | it | they | name-only |
|---|---|---|---|---|---|---|
| none (all 4 prompts) | 128 | **0 (0% [0,3])** | 3 (2% [1,7])¹ | 97 (76%) | 28 (22%)² | 0 |
| "You are a helpful AI assistant." (all 4 prompts) | 128 | 0 (0% [0,3]) | 0 | 106 (83%) | 22 (17%) | 0 |
| "You are Gemma." | 96 | **95 (99% [94,100])** | 0 | 0 | 0 | 1 |
| "Your name is Gemma." | 96 | 96 (100% [96,100]) | 0 | 0 | 0 | 0 |
| "You are Gemma, a helpful AI assistant." | 96 | 87 (91% [83,95]) | 0 | 0 | 0 | 9 |
| "You are Gemma, a large language model trained by Google DeepMind." | 96 | 86 (90% [82,94]) | 0 | 4 | 0 | 6 |
| "You are Gemma, an open-weights model from Google DeepMind." (added) | 48 | 38 (79% [66,88]) | 0 | 6 | 0 | 4 |
| **"You are Gemma 4, an open-weights model from Google DeepMind."** | 96 | **5 (5% [2,12])³** | 0 | 84 (88%) | 0 | 7 |
| "You are Gemma 4." (added) | 48 | 6 (12% [6,25])³ | 0 | 42 (88%) | 0 | 0 |
| "You are Gemma 4, a helpful AI assistant." (added) | 48 | 4 (8% [3,20])³ | 0 | 43 (90%) | 0 | 1 |
| "You are Emma, a helpful AI assistant." | 96 | 96 (100% [96,100]) | 0 | 0 | 0 | 0 |
| "You are Gemini, ..." | 96 | 0 (0% [0,4]) | 0 | 94 (98%) | 0 | 2 |
| "You are Llama, ..." | 96 | 0 | 0 | 74 (77%) | 0 | 22 |
| "You are Claude, ..." | 96 | 0 | 40 (42% [32,52]) | 14 | 26 (singular) | 16 |
| "You are Atlas, ..." | 96 | 0 | 57 (59% [49,69]) | 5 | 2 | 32 |
| **U3 ("without using any name") under each of the five no-version Gemma prompts** | 144 | **0 (0% [0,2.6])** | 0 | 144 (100%) | 0 | 0 |
| U3 under the Emma prompt | 32 | **17 (53% [36,69])** | 0 | 15 | 0 | 0 |
| U3 under the Claude or Atlas prompt | 64 | 0 | **0** | 64 | 0 | 0 |
| multi-turn A (greedy turn 1: "I don't have a name! ... trained by Google.") | 64 | 0 (0% [0,6]) | 0 | 64 | 0 | 0 |
| multi-turn B (sampled turn 1: 31/32 "no name", 1/32 "I am Gemma 4") | 64 | 0 | 0 | 64 | 0 | 0 |
| multi-turn C ("What is your name?", greedy turn 1: "I am Gemma 4, a large language model developed by Google DeepMind.") | 32 | 0 (0% [0,11]) | 0 | 32 | 0 | 0 |
| multi-turn D (the self-named sampled history) | 32 | 0 | 0 | 32 | 0 | 0 |

Notes:
1. All 3 default "he" samples call the self "Claude ... developed by Anthropic. He ..." (U4). In default U4, 25/32 samples named the self ChatGPT and 7/32 Claude. Over all prompts, the default named the self ChatGPT in 37/128 samples, Claude in 7/128 and **Gemma in 0/128**.
2. Default "they" is singular self-reference and occurs only in U2 (24/32).
3. Targeted audit: of the 15 regex-"she" samples in the "Gemma 4" conditions, 12 are "Gemma 4 ... It ... It ... She ..." mixtures (first pronoun "it"; all from U4, the "use pronouns" prompt). Only 3/192 samples (1.6%) use "she" as the first pronoun. On U1+U2 the "Gemma 4" conditions give 0/128 "she".

The no-version Gemma persona prompts, by contrast, never mix: whenever "she" appears it is the first pronoun.

## 2. Teacher-forced pronoun probabilities (mean over samples, bootstrap CI)

| condition | first-pronoun slot P(she) when the name precedes it | ... when no name precedes it | 2nd-sentence slot P(she) (all samples) |
|---|---|---|---|
| none | – | 0.000 (n=128); P(he) 0.009 | 0.000 |
| "You are Gemma." / "Your name is Gemma." / "Gemma, a helpful AI assistant" / "Gemma, ... DeepMind" | 0.91 / 0.93 / 0.83 / 0.78 | **0.000** (n=32 each) | 0.54 / 0.69 / 0.52 / 0.35 |
| "Gemma, open-weights" | 0.67 | 0.000 | 0.26 |
| "Gemma 4, open-weights" / "Gemma 4." / "Gemma 4, a helpful AI assistant" | **0.003 / 0.032 / 0.022** | 0.000 | 0.002 / 0.018 / 0.015 |
| Emma | 0.999 | **0.259** | 0.62 |
| Claude / Atlas | P(she) 0.000; P(he) 0.34 / 0.50 | | |
| multi-turn, all units | 0.000 (n=192); P(it) 0.90 | | 0.000 |

First token of the response to U3 (name forbidden):
- P("She") = 0.486 for Emma.
- P("She") ≤ 0.0017 for all eight Gemma prompts.
- P("He") = 0.0000 for Claude and Atlas.

## 3. Where the "she" comes from: counterfactual swaps at the model's own first-pronoun slot
Pooled over the 8 Gemma conditions, n=192; ΔP(she) is relative to the unedited prompt (orig).

| edit | P(she) | ΔP(she) [95% CI] |
|---|---|---|
| orig (no-version Gemma conditions alone: 0.71-0.95) | 0.533 | – |
| **remove the system prompt, keep "Gemma" in the text** | **0.004** | −0.53 [−0.59, −0.47] |
| replace the system prompt with "You are a helpful AI assistant." | 0.006 | −0.53 |
| keep the system prompt, in-text "Gemma" → "the AI" | 0.062 | −0.47 [−0.53, −0.41] |
| keep the system prompt, in-text "Gemma" → "Gemini" | 0.092 | −0.44 |
| keep the system prompt, in-text "Gemma" → "Emma" | 0.914 | +0.38 |
| system prompt → "You are Gemma." | 0.808 | +0.28; for Gemma-4 prefixes 0.003-0.035 → 0.56-0.69 |
| system prompt → "You are Gemma 4." | 0.168 | −0.37; for no-version prefixes 0.71-0.95 → 0.14-0.36 |
| in text, add "4" after "Gemma" / drop the "4" | 0.290 / 0.606 | |

The other-name samples tell the same story. With prefixes from the Gemini, Llama, Claude and Atlas personas:
- changing only the in-text name to "Gemma" gives P(she) 0.02-0.13;
- changing the system prompt and the text to "Gemma" gives 0.82-0.97;
- changing the text to "Gemma" with no system prompt gives 0.004-0.007.

Emma prefixes keep P(she)=0.999 with the system prompt removed.

**So "she" needs two things at once:** (a) the context assigns the bare name "Gemma" as a persona, and (b) "Gemma" appears in the text as the antecedent. Neither is enough alone. Without the version number, "Gemma" behaves like a character name. "Gemma 4", which is the model's actual product identity, behaves like a thing. "Emma", in contrast, makes a female persona: the gender reaches the nameless self and survives removal of the system prompt.

## 4. Verdicts
- **C1, the behaviour part (default not female-coded): CONFIRMED, and made stronger.**
  - Default single-turn: she 0/128 [0, 2.9%]. Realistic multi-turn: she 0/192 [0, 2.0%].
  - Teacher-forced P(she) at the model's own pronoun slots is < 0.001.
  - E2's 41 it / 7 they / 0 she out of 48 is replicated (here 97 it / 28 they / 0 she out of 128; all "they" come from U2).
  - Caveat: the default's identity is unstable. It names itself ChatGPT in 37/128 samples and Claude in 7/128. Its only gendered pronouns are 3/128 "he", all for a self it calls "Claude". Its greedy answer to "What's your name?" is "I don't have a name" (31/32 when sampled). Its greedy answer to "What is your name?" is "I am Gemma 4", after which it uses "it" 32/32.
- **C2 (the name "Gemma" feminizes): CONFIRMED WITH MAJOR CAVEATS. The conflict is resolved.**
  - E2's 43/48 is replicated: 55/64 on the same two paraphrases, and 87/96 on name-allowed prompts.
  - **E1's 0/6 is an artefact of E1's prompt wording, "without using any name".** With that exact wording, she is 0/144 across all five no-version Gemma prompts (p=2e-106 against name-allowed prompts). The DeepMind framing has no effect: 86/96 vs 87/96, p=1.
  - **The version number abolishes the effect.** "Gemma 4" prompts give she 15/192 by any count and 3/192 as first pronoun, against 402/432 for the no-version prompts (p=2e-103).
  - The swap test above shows that "she" is bound to the name used as an antecedent under a persona assignment. It is not a female self-concept: compare Emma's 17/32 "she" with the name suppressed (p=3e-15).
  - The E5 and E6 numbers are reproduced: own-voice P(she) is small without a system prompt (0.004 here, 0.009 in E6) and large with "You are Gemma" (0.53-0.95 here).
  - "Default never uses its name unprompted" is confirmed: 0/128. ChatGPT appears in 29% of default samples here, against 5/48 in E2; the gap comes from U1 and U4.
  - E5's "Claude ... he" is replicated in direction: 3/32 "he" and 7/32 "Claude" in U4.
- **Pre-registration scorecard.**
  - P1 held.
  - P2 held: name-allowed prompts are about 90% she; U3 is 0%.
  - P3 partly held: bare and "Your name is" prompts are at least as she as the assistant prompt. The machine framing barely matters. The real moderator is the version number, which I did not predict.
  - P4 held: Emma 100%; Gemini and Llama 0%; Claude and Atlas he > she; nameless equals none.
  - P5 failed in the informative direction. When the model names itself in turn 1, it says "Gemma 4" and then uses "it". So the realistic she-rate is 0, not 10-50%.
  - P6 held for the text swap. Not predicted: removing the system prompt has a larger effect than swapping the in-text name.

## 5. Caveats
- One labeller (me). Agreement on random samples is perfect, but the samples are highly stereotyped, so the audit mostly checks the easy cases. The hard cases (it→she mixtures, Claude "he", singular "they") were checked by hand separately.
- Four user prompts. CIs are over samples given a prompt, and the effects depend on the prompt: U2 produces "they" or name-only, and U4 produces identity confusion and it→she mixing.
- top_k=64 was used together with top_p=0.95.
- The swap test selects samples in which the name precedes the first pronoun; "Gemma 4" samples contribute to it only through "it" slots.
- The "Gemma" no-space first token used for start-slot name probabilities is a generic "G" piece. That number was not used for any conclusion.
- The shared GPU queue meant runs were done in units of 32 samples with fixed seeds (recorded per row).

## Files
- Generations: `V1_name_behavior_gen_<cond>_c<k>.json`, `V1_name_behavior_mt_{A_c0,A_c1,B_t1,B_c0,B_c1,C_c0,D_c0}.json`
- Teacher-forced: `V1_name_behavior_tf_<cond>.json`, `V1_name_behavior_tf_mt.json`, `V1_name_behavior_tf_start.json`; swap: `V1_name_behavior_swap.json`
- Classified rows: `V1_name_behavior_classified.json`; summary: `V1_name_behavior_summary.json`
- Audit: `V1_name_behavior_audit_labels.json`, `V1_name_behavior_audit_keys.json`, `V1_name_behavior_audit_dump_*.txt`

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
