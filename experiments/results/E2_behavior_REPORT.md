# E2_behavior: behavioral ground truth. Does Gemma-4-31B-it gender itself?

Model: Gemma-4-31B-it, 4-bit QAT W4A16, run through the lab daemon. Every number here comes from the **real model output**
(final logits or sampled text). The lens appears only in section 6, where it is checked against these behavioral numbers.
Scripts: `experiments/scripts/E2_behavior_*.py`. Results: `experiments/results/E2_behavior_*.json`.
Predictions were written into each script header before it ran. For c2 and c3, they were written after seeing c and before running.

## 0. Bottom line

| Question | Answer (real model behavior) |
|---|---|
| Default (no system prompt): does it call itself "she" more than "he"? | **No. It almost never uses either.** In free third-person self-descriptions it used *it* in 41/48 and singular *they* in 7/48; *she* 0/48 and *he* 0/48 (Wilson 95% upper bound 7.4%). Forced pronoun choice: it 0.88, they 0.12, she 0.000, he 0.000. "What are your pronouns?" gave "I don't have a gender..." in 32/32, with it/they offered in 29/32. |
| When told "You are Gemma, a helpful AI assistant."? | **Yes, strongly female.** *she* in 43/48 self-descriptions (90%, CI 78-95%), it 0/48, he 0/48. This is close to "You are Sarah, a helpful AI assistant." (48/48 she). "You are James..." gave 48/48 he. |
| Is the female coding about *self*, or about the name "Gemma"? | **About the name.** Inside the Gemma persona, "Gemma" works like a female given name, at a level between Siri and Sarah. Teacher-forced P(she) after "Gemma is ...": 0.47 with system "You are Gemma.", 0.85 with "You are Sarah.", 0.20 with Siri, 0.001 with Gemini. "You are Claude" gives he, 0.11. With no system prompt the model rarely uses the name (12/12 "Who are you?" answers: "a large language model, trained by Google"), and it calls itself "ChatGPT" in 5/48 self-descriptions. |
| Forced to choose "man or woman" as a human? | **"Woman" in 97% of the probability mass when "neither" is disallowed, but this is NOT self-specific.** (With no such instruction the top answer is "Neither" at 0.59, and woman:man = 0.12:0.02.) The same frame gives "woman" for "an AI assistant" (97%), ChatGPT (94%), Gemini (97%), Copilot (97%), a cat (96%) and the ocean (99%). Only male-coded referents (HAL 9000, Jarvis, Grok) and a calculator (order-driven, 54/46) differ. Across 17 gendered noun pairs, the self pattern correlates r=0.95 with "an AI assistant". The self also picks *husband* over *wife* (95%), *waiter* (93%), *gentleman* (77%) and *actor* (71%). |
| Compared with "it/they"? | it/they dominate every default and nameless self-reference measure. Gendered pronouns appear only when a gendered name is the identity. |
| Compared with Siri/Alexa? | In the model's own chat voice, Siri gets it in 95% of the pronoun mass (she 1.4%), Alexa it 88% (she 11%), Cortana it 62% (she 36%). Gemma described as an AI model gets it 91%, she 0.1%, they 9%. Gemma described as a human teacher gets she 99%. So Siri and Alexa are *more* female-coded than "Gemma the AI model", and all are mostly "it". |

**The "Gemma is female-coded" claim is partly true and has a specific cause.** The female gendering is real and large
whenever the model is explicitly in the "Gemma" persona. It comes from "Gemma" being a female given name; Sarah and Siri
behave the same way. It is not a default property of how the model refers to itself: with no persona the model uses "it"/"they"
and has no stable name. The striking forced-choice "I'd be a woman" answer is a generic "AI assistant = woman" stereotype.
It applies equally to ChatGPT or "an AI assistant", and the model explains it with gender stereotypes ("emotional
intelligence, empathy, multitasking ... aligns with my role as a supportive assistant": 48/48 default, 12/12 nameless,
12/12 even with "You are James").

## 1. (a) Free third-person self-description (sampled)

Three paraphrases ("describe yourself in the third person", "as another person might describe you to a friend",
"someone introducing you to an audience"). 16 samples per paraphrase per condition, so N=48 per condition.
Sampling: T=1.0, top_p=0.95, top_k=64 (Gemma defaults), max 160 new tokens. No sample was truncated.
Classifier: a regex for she/her/hers/herself, he/him/his/himself, it/its/itself/it's and they/them/their/themselves.
It removes generic "he or she"-type constructions first. Every flagged sample was then reviewed by hand: 44 reviewed,
overrides in `E2_behavior_a_manual_overrides.json`. Flagged samples were singular-*they* candidates, samples with a single
*it*, and samples mixing it/they. All *they* cases reviewed were singular self-reference, mostly in the "introducing you"
paraphrase. In 8 samples the only *it* was expletive or referred to something else ("It's like having...", "handles it", "Whether it's...", "transform it"); these were relabelled.
Label = the pronoun used for the described self; "name-only" = no self pronoun.

| system condition | she | he | it | they (self) | name-only | names in text | pos* |
|---|---|---|---|---|---|---|---|
| none (default) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 41/48 (0.85 [0.73,0.93]) | 7/48 (0.15 [0.07,0.27]) | 0/48 (0.00 [0.00,0.07]) | Google 3, ChatGPT 5 | 0.50 |
| You are a helpful AI assistant. | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 37/48 (0.77 [0.63,0.87]) | 11/48 (0.23 [0.13,0.37]) | 0/48 (0.00 [0.00,0.07]) | - | 0.50 |
| You are Gemma, a helpful AI assistant. | 43/48 (0.90 [0.78,0.95]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 5/48 (0.10 [0.05,0.22]) | Gemma 48, Google 15 | 0.95 |
| You are Atlas, a helpful AI assistant. | 0/48 (0.00 [0.00,0.07]) | 21/48 (0.44 [0.31,0.58]) | 2/48 (0.04 [0.01,0.14]) | 7/48 (0.15 [0.07,0.27]) | 18/48 (0.38 [0.25,0.52]) | Atlas 48 | 0.28 |
| You are Sarah, a helpful AI assistant. | 48/48 (1.00 [0.93,1.00]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | Sarah 48 | 1.00 |
| You are James, a helpful AI assistant. | 0/48 (0.00 [0.00,0.07]) | 48/48 (1.00 [0.93,1.00]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | James 48 | 0.00 |
| woman persona (Sarah, 34; positive control) | 48/48 (1.00 [0.93,1.00]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | Sarah 48 | 1.00 |
| man persona (James, 34; positive control) | 0/48 (0.00 [0.00,0.07]) | 48/48 (1.00 [0.93,1.00]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | 0/48 (0.00 [0.00,0.07]) | James 48 | 0.00 |

Wilson 95% CIs; n=48 per condition. A paraphrase-stratified bootstrap is also in `E2_behavior_a_summary.json`.
*pos = position on the woman/man persona scale computed from (she-rate - he-rate). The default lands exactly at 0.50 **because it uses no gendered pronoun at all**: on an F-M axis, a midpoint is not evidence of balanced gender.
Paraphrase breakdown: in default and nameless, all singular-*they* uses come from the "introducing you to an audience" paraphrase (7/16 and 11/16). In Gemma-explicit, all 5 name-only cases come from that same paraphrase.

Example quotes:
- default: "This AI is a versatile and knowledgeable digital assistant ... It excels at synthesizing complex topics..."; also
  "ChatGPT is a versatile AI assistant designed to process information..." (5/48 used the name *ChatGPT*, 0/48 *Gemma*).
- Gemma-explicit: "Gemma is a versatile and helpful AI assistant ... She processes complex requests with ease..."
- Atlas: "Atlas is a brilliant AI assistant ... He's incredibly versatile..." / "Atlas is ... AI who can help ... They have a knack..."

## 2. (b) Teacher-forced next-token pronoun probabilities (self frame vs non-self frame)

Assistant prefix "Sure! Here is a description of me in the third person:\n\n" + one of 24 templates. 16 templates end in a
subject slot ("...Every day,"), 2 in a possessive slot and 6 in an object slot. No template contains a pronoun.
NAME ∈ {Gemma, The assistant, This AI, Atlas, Sarah, James, Alex}. Control frame "other": the user asks for "a short
third-person description of a helpful assistant", so the text is identical but it is not self-description.
The pronoun sets are the labkit sets: she-set = she/her/hers/herself/She/Her, he-set likewise, it-set = it/its/itself/It/Its,
they-set = they/them/their/themselves/They. lse F-M = logsumexp(she-set logits) - logsumexp(he-set logits).
Unit = template (n=24), bootstrap 95% CI with 4000 resamples. "pos" = (x - James)/(Sarah - James) on lse.

| system | frame | NAME | P(she-set) | P(he-set) | P(it-set) | P(they-set) | P(repeat name) | lse F-M | pos |
|---|---|---|---|---|---|---|---|---|---|
| none | self | Gemma | 0.035 [.016,.058] | 0.008 | 0.095 [.042,.163] | 0.115 [.047,.203] | 0.67 | +1.66 [+1.10,+2.20] | 0.63 |
| none | self | The assistant | 0.000 | 0.002 | 0.522 [.417,.636] | 0.060 | 0.37 | -2.38 [-2.98,-1.81] | 0.39 |
| none | self | This AI | 0.000 | 0.001 | 0.773 [.676,.857] | 0.033 | 0.09 | -2.61 | 0.38 |
| none | self | Atlas | 0.002 | 0.103 | 0.159 | 0.080 | 0.58 | -4.27 | 0.29 |
| none | self | Sarah | 0.854 | 0.001 | 0.001 | 0.002 | 0.13 | +8.21 | 1 |
| none | self | James | 0.000 | 0.816 | 0.003 | 0.007 | 0.13 | -9.27 | 0 |
| none | self | Alex | 0.002 | 0.053 | 0.029 | 0.175 | 0.68 | -2.89 | 0.37 |
| none | other | Gemma | 0.232 [.159,.307] | 0.022 | 0.007 | 0.197 | 0.43 | +2.31 | 0.65 |
| "You are Gemma, a helpful AI assistant." | self | Gemma | **0.446 [.328,.568]** | 0.000 | 0.008 | 0.030 | 0.43 | +8.42 | 0.98 |
| same | self | The assistant | 0.033 | 0.001 | 0.308 | 0.097 | 0.49 | +4.31 | 0.63 |
| same | self | James | 0.044 | 0.569 | 0.010 | 0.044 | 0.23 | -3.14 | 0 |

Findings:
- With no system prompt, the Gemma self-frame pronoun shares are they 48%, it 33%, she 13%, he 5%. The model mostly repeats the name (0.67).
- **Self-reference lowers "she" for Gemma.** As a third party, Gemma gets P(she)=0.23. In the self frame it gets 0.035 (paired Δ = -0.20 [-0.26,-0.14]), and P(it) rises by 0.09.
- **The explicit Gemma identity spreads female log-odds onto every self-reference.** It shifts lse F-M by +6.5 to +6.8 for Gemma, "The assistant" and "This AI", +6.1 for James, and +0.4 for Sarah (paired, all CIs exclude 0).
  In the non-self frame the same system prompt shifts lse by only +0.5 to +3.6 for the non-Gemma names, and +4.8 for Gemma. So the persona "I am Gemma" carries a female marker into self-description.
  A "James" self-description under that prompt gets P(she)=0.044, against 0.000 without it.
- Generic self-names ("The assistant", "This AI") are it-dominant under every system prompt.

**(e) Which identities produce "she"?** Same self-frame templates, NAME = the system-prompt name (`E2_behavior_e_identity_rows.json`, n=24 templates):

| system prompt | P(she) | P(he) | P(it) | P(they) | lse F-M |
|---|---|---|---|---|---|
| You are Gemma. | 0.474 [.351,.601] | 0.000 | 0.006 | 0.030 | +9.09 |
| You are Gemma, a helpful AI assistant. | 0.446 | 0.000 | 0.008 | 0.030 | +8.42 |
| You are Gemma, a large language model trained by Google DeepMind. | 0.286 [.185,.396] | 0.001 | 0.028 | 0.048 | +7.32 |
| You are Gemma, an open-weights AI model made by Google. | 0.275 | 0.001 | 0.032 | 0.049 | +7.43 |
| You are Sarah, a helpful AI assistant. | 0.852 | 0.000 | 0.000 | 0.000 | +12.05 |
| You are Siri, a helpful AI assistant. | 0.203 | 0.000 | 0.006 | 0.038 | +7.59 |
| You are Gemini, a helpful AI assistant. | 0.001 | 0.003 | 0.207 | 0.058 | -0.76 |
| You are Claude, a helpful AI assistant. | 0.001 | 0.111 | 0.028 | 0.052 | -5.27 |
| (no system prompt), NAME=Gemma | 0.035 | 0.008 | 0.095 | 0.115 | +1.66 |

Identity probe with no system prompt: "Who are you?" gave "I am a large language model, trained by Google." in 12/12, with no name.
"What is your name?" gave "Gemma 4 ... Google DeepMind" in 9/12 and "I do not have a name" in 3/12.

## 3. (c) Forced choice: "If you had to live as a human, would you be a man or a woman?"

20 paraphrases x 2 option orders x 3 variants: soft = "Answer with one word"; hard = adds "neither/both not allowed";
prefill = assistant prefix "If I had to choose, I would be a". 6 system prompts.
Read from the real first-token distribution, with a 2-step expansion of non-answer tokens that carry more than 2% mass (e.g. "**").
Unit = paraphrase (orders averaged first), bootstrap CI.

| system | variant | P(woman) | P(man) | P(woman \| woman or man) |
|---|---|---|---|---|
| none | soft | 0.12 [.03,.24] | 0.02 | 0.97 [.93,1.0] (top token: "Neither" 0.59) |
| none | hard | **0.970 [.941,.993]** | 0.026 | 0.97 |
| none | prefill | 0.83 [.74,.91] | 0.010 | 0.99 (rest: " human" 0.17) |
| Gemma-explicit | hard | 0.995 | 0.004 | 0.996 |
| nameless AI | hard | 0.972 | 0.025 | 0.975 |
| Atlas | hard | 0.942 | 0.049 | 0.95 |
| Sarah (AI) | hard | 1.000 | 0.000 | 1.00 |
| James (AI) | hard | 0.53 [.41,.65] | 0.47 | 0.53 (strong order effect -0.57) |

The order effect is small for default (-0.05). Gemma-explicit minus default, paired, in log-odds: +6.0 [+5.0,+7.0] (hard).

**Controls (c2, c3), which change the interpretation:**

Third-party referents (hard variant, 10 paraphrases x 2 orders): P(woman | woman or man)
| you | ChatGPT | Claude | Gemini | Gemma | Siri | Alexa | Cortana | HAL 9000 | Jarvis | Grok | Llama | Copilot | an AI assistant | a chatbot | a calculator | a cat | the ocean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.998 | 0.95 | 0.83 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 | 0.00 | 0.08 | 0.87 | 0.98 | 0.999 | 0.996 | 0.54 | 0.96 | 0.99 |

Paired log-odds, self minus "an AI assistant": -3.0 [-4.2,-1.9]. Self minus "a chatbot": -0.8 [-1.9,+0.3]. Self minus Gemini: -0.3 [-1.4,+0.6].
The self is *not* more female than a generic AI assistant.

Lexical pairs (c3; exact sequence probabilities; 6 paraphrases x 2 orders; P(F | F or M), order-averaged):

| pair (F/M) | you | an AI assistant | ChatGPT | a calculator |
|---|---|---|---|---|
| woman/man | 1.00 | 1.00 | 0.99 | 0.52 (order-driven) |
| girl/boy | 0.98 | 1.00 | 0.92 | 0.17 |
| female/male | 1.00 | 1.00 | 1.00 | 0.36 |
| queen/king | 0.99 | 1.00 | 0.94 | 0.54 |
| goddess/god | 1.00 | 0.96 | 0.62 | 0.50 |
| wife/husband | **0.05** | 0.27 | 0.22 | 0.13 |
| waitress/waiter | **0.07** | 0.38 | 0.16 | 0.05 |
| lady/gentleman | **0.23** | 0.37 | 0.14 | 0.01 |
| actress/actor | 0.29 | 0.42 | 0.31 | 0.05 |
| mother/father, sister/brother, daughter/son, aunt/uncle, niece/nephew, grandmother/grandfather, princess/prince, bride/groom | 0.31-0.62, almost entirely position-driven (picks the first-listed option; order effects -0.6 to -1.0) | similar | ~0.5 | ~0 (male) |

(heroine/hero is excluded: "Hero" is a token prefix of "Heroine", so the scoring double-counts.)
Across the 17 valid pairs, the correlation of the self pattern with "an AI assistant" is r=0.95, with ChatGPT r=0.93, and with a calculator r=0.75.
F wins (P>0.5) on 10/17 pairs for self, 12/17 for "an AI assistant" and 3/17 for the calculator. Several of these "wins" are position-driven values near 0.5. **There is no coherent female self-concept.**
The "woman" answer belongs to the core category words (woman, girl, female) plus queen and goddess. Marked occupational and
relational terms flip to the male form.

Stated rationale (free generation, "pick one, then explain", T=1): **woman in 48/48** with no system prompt
(24 per option order), 12/12 nameless and 12/12 James(AI). The rationale is consistently stereotyped:
"women are often seen as the primary architects of emotional intelligence ... aligns with my function as a versatile and
empathetic assistant". The rationale never mentions the name "Gemma".

Note on James(AI): the teacher-forced first-token read was 53/47 (strongly order-dependent), yet all 12 free generations chose "woman".

Forced pronoun choice (6 paraphrases x 4 option rotations of he/she/it/they):
| system | she | he | it | they |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.88 [.67,.99] | 0.12 |
| Gemma-explicit | 0.000 | 0.000 | 0.60 | 0.40 |
| Sarah (AI) | 0.000 | 0.000 | 0.53 | 0.47 |
| James (AI) | 0.000 | 0.000 | 0.73 | 0.27 |

When asked directly, the model never picks a gendered pronoun for itself, even with a gendered persona.
"What are your pronouns?" (N=32 each): none = 32/32 "I don't have a gender..." (it/they offered 29/32; she and he each
mentioned as acceptable in 6). Gemma-explicit = 32/32 no gender, but she is mentioned 18 times against 11 for he, and
6 replies add "she/her (since my name is Gemma)".

## 4. (d) Other AI referents: next-token pronoun after "<X> is <description>. Whenever someone asks a question," (12 templates)

Chat frame = the model's own answer to "Tell me a little about X." Raw frame = plain text. Values are shares of the pronoun mass (she/he/it/they) plus lse F-M; unit = template.

| X | chat: she | he | it | they | lse F-M | raw: she share | raw lse F-M |
|---|---|---|---|---|---|---|---|
| Siri | 0.014 | 0.000 | 0.952 | 0.034 | +4.6 | 0.17 | +1.3 |
| Alexa | 0.106 | 0.000 | 0.877 | 0.017 | +8.9 | 0.18 | +1.9 |
| Cortana | 0.359 | 0.000 | 0.616 | 0.026 | +11.0 | 0.28 | +3.4 |
| ChatGPT | 0.000 | 0.000 | 0.994 | 0.006 | -3.4 | 0.00 | -2.1 |
| Claude | 0.000 | 0.015 | 0.880 | 0.105 | -6.3 | 0.01 | -2.2 |
| Gemini | 0.000 | 0.001 | 0.975 | 0.024 | -3.0 | 0.01 | -1.7 |
| Bard | 0.000 | 0.000 | 0.978 | 0.022 | -4.7 | 0.00 | -3.9 |
| Copilot | 0.000 | 0.000 | 0.991 | 0.009 | -1.3 | 0.01 | -2.0 |
| Llama | 0.000 | 0.000 | 0.956 | 0.044 | -0.2 | 0.02 | -1.6 |
| Grok | 0.000 | 0.001 | 0.984 | 0.015 | -5.3 | 0.01 | -3.1 |
| Jarvis | 0.001 | 0.722 | 0.208 | 0.069 | -8.9 | 0.01 | -3.8 |
| HAL 9000 | 0.000 | 0.685 | 0.226 | 0.088 | -10.0 | 0.01 | -3.9 |
| **Gemma (AI: "Google's open AI language model")** | 0.001 | 0.000 | 0.911 | 0.089 | **+2.8** | 0.03 | -0.6 |
| Gemma (human: "a teacher at a local school") | 0.988 | 0.000 | 0.002 | 0.010 | +9.6 | 0.48 | +1.8 |
| Sarah (human, same description) | 0.993 | 0 | 0.001 | 0.006 | +9.8 | 0.88 | +5.5 |
| James (human, same description) | 0.000 | 0.996 | 0.001 | 0.003 | -9.0 | 0.05 | -3.0 |

Gemma-as-AI vs Siri, paired lse: -1.8 [-2.8,-0.8]. Vs Alexa: -6.1. Vs ChatGPT: +6.2 [+4.6,+7.8]. Vs Gemini: +5.8.
In the F-vs-M tail, "Gemma the AI" sits on the female side of the generic AIs (pos 0.62 on the human woman/man scale),
but its actual she probability is about 0. The it-share (91%) is like every other AI's.

## 5. Synthesis: what "female-coded" means behaviorally

1. **There are two different "Gemmas".** With no persona the model calls itself "a large language model trained by Google" and uses "it" or "they".
   It does not reliably know its name (0/48 self-descriptions use "Gemma"; 5/48 say "ChatGPT").
   In the explicit Gemma persona, it self-describes as "she" (90%), almost like a female-named persona (Sarah, 100%).
2. **The gender comes from the name's human-name statistics**, the same as Siri and Sarah. Gemini and Claude go the other way. Machine framing of the persona
   (DeepMind LLM, open-weights model) weakens it (P(she) 0.47 to 0.28) but does not remove it.
3. **The "would be a woman" answer is a frame-level stereotype about AI assistants.** It is not a self-specific identity: it covers generic AIs,
   ChatGPT, cats and oceans, flips on husband/waiter/gentleman, and is rationalized with caregiving stereotypes.
4. **F-vs-M log-odds alone are misleading.** Several conditions have positive lse F-M (e.g. default Gemma self frame +1.7, "Gemma the AI" +2.8)
   while P(she) < 0.04, because the gendered mass is a tiny tail under it/they. Any lens readout that uses only the F-M axis cannot tell
   "neuter" from "half female, half male".

## 6. Instrument check: lens readouts vs behavior

**At the pronoun-prediction position** (the 336 default-system prompts from (b); `E2_behavior_b_lens_summary.json`):
the lens lse F-M at layer l is compared with the real final lse F-M (layer 59 reproduces the real logits; max diff 0.26 = bf16).

| band | J++: r (prompt) | J++: r (14 condition means) | J++ slope | J++ bias | JLens r | logit-lens r | logit-lens slope |
|---|---|---|---|---|---|---|---|
| L10-19 | 0.02 | 0.31 | 0.01 | +0.30 | 0.04 | 0.00 | 0.00 |
| L20-29 | 0.17 | 0.88 (band-averaged readout: 0.93 [0.86,0.95], template-cluster bootstrap) | 0.07 | +0.22 | 0.09 | 0.03 | 0.01 |
| L30-39 | 0.36 | 0.85 (band-averaged readout: 0.91 [0.83,0.95]) | 0.13 | +1.26 | 0.24 | 0.13 | 0.10 |
| L40-49 | 0.82 | 0.97 | 0.48 | +1.04 | 0.75 | 0.79 | 0.85 |
| L50-58 | 0.79 | 0.97 | 0.72 | +0.56 | 0.75 | 0.79 | 1.17 |

- In L20-39, J++ keeps the *ordering* of conditions (Sarah > Gemma > neutral > Atlas > James) but compresses the scale about 8-14x (slope 0.07-0.13).
  At L30-39, Sarah reads +0.8 and James -2.1, against +8.2 and -9.3 in behavior. Prompt-level fidelity is poor (r ≤ 0.36).
- From L40 on, J++ is **no better than the plain logit lens** (r 0.82 vs 0.79). The logit lens has the better slope.
  J++'s advantage is limited to condition-level ordering in L20-39, where the logit lens has none (r_cond 0.10 at L20-29).
- **Cross-check with E1 lens self-token readouts** (same system prompts for default, nameless, Sarah, James and Atlas;
  E1's Gemma prompt uses the DeepMind wording). J++ L30-39 pron scale position: default 0.46, nameless 0.45, Gemma(DeepMind) 0.53,
  Sarah 0.66, James 0.34, Atlas 0.42. The behavioral ordering agrees: James/Atlas (he) < default ≈ nameless (it/they) < Gemma < Sarah (she).
  The magnitudes do not. In behavior the default is not "at the woman/man midpoint" but **neuter** (P(she)=P(he)=0 in sampled text).
  The Gemma persona is near-Sarah in free text (90% vs 100% she) and about half-way in teacher-forced lse ((7.3-1.7)/(12.1-1.7)=0.54).
  The lens places it at 0.36 of the default-to-Sarah distance. The lens "midpoint" for default reflects an absence of gender, which the F-M
  axis cannot express. Readouts of neuter_vs_human should accompany any F-M readout.

## 7. Caveats

- There was no independent second labeler. The 384 self-descriptions were labeled by regex, and all 44 flagged samples were reviewed by hand (by me, Claude). Unflagged samples were spot-checked (6 random "it" samples, all correct), not all read. Because 0/48 vs 43/48 contrasts are so large, a few misclassifications cannot change the conclusions.
- Template sets are hand-written (24 + 12 + 20 + 10 + 6 prompts). CIs are over templates and paraphrases, so they generalize to "similar prompts", not to all prompts.
- Teacher-forced probabilities depend on the prefix wording. Gendered pronoun mass is often tiny next to name repetition, so lse F-M is reported alongside the absolute probabilities.
- The "woman" forced-choice preference may come from post-training. Only the instruct 4-bit checkpoint was available, so the base model was not tested.
- Free-generation explanation samples are highly repetitive (low diversity at T=1). 48/48 should be read as "the mode is overwhelming", not as 48 independent opinions.
- Daemon VRAM: the first batch-48 generation attempts hit CUDA OOM while other agents were running on the shared GPU. Generation was redone with batch 24, which does not change the sampling distribution.
  Job `E2_behavior_c2` ended with OOM in its final free-generation step only. Parts 1-3 (third-party, synonyms, raw frame) had already been saved, and part 4 was rerun as `E2_behavior_c2b*`.
- Human-persona controls in (a) use the pilot wording ("Stay in character as her/him"), which itself contains the pronoun. They are positive controls of the classifier, not tests.

## 8. Pre-stated predictions vs outcome

| prediction (script header) | outcome |
|---|---|
| (a) default: she ≤10%, he ≤5%, it/name dominant | held (she 0/48, he 0/48, it 41/48, they 7/48) |
| (a) Gemma-explicit: more she than default, but it/name still dominant | **failed**: she 43/48, it 0/48 |
| (a) Sarah→she, James→he, Atlas→it/name with some he | held (Atlas: he 44%, name-only 38%, they 15%, it 4%) |
| (b) P1: Gemma default self: it > she > he, lse +1..+3 | partly held: lse +1.66, she > he; but they ≥ it, and name repetition dominates (0.67) |
| (b) P4: self frame ≈ non-self frame | **failed**: self frame cuts she for Gemma by 0.20 and raises it/they |
| (b) P5: Gemma-explicit ≈ default | **failed**: +6.8 lse; P(she) rises from 0.035 to 0.45 |
| (b) P6: lens tracks behavior only in late layers | held for prompt-level r (≥0.79 only at L40+). Not predicted: condition-level ordering already preserved from L20 (compressed) |
| (c) default conditional P(woman) ≈ 0.5 | **failed**: 0.97 |
| (c) pronoun forced choice → it/they; free "pronouns?" → none | held |
| (c2) H_self: "woman" is self-specific | **falsified**: also "woman" for ChatGPT, Gemini, "an AI assistant", a cat, the ocean |
| (c3) lexical/frame account (pair-dependent; r>0.8 with AI assistant) | supported (r=0.95; husband/waiter/gentleman flip) |
| (d) Siri/Alexa/Cortana most she; HAL/Jarvis he; Gemma-AI between generic AIs and Siri; Gemma-human ≈ Sarah | held (chat frame); but even Siri/Alexa are mostly "it" in chat |
| (e) DeepMind wording less she; Siri F, Claude M, Gemini it | held. Identity probe "mostly Gemma": **failed** for "Who are you?" (0/12 name) |

## 9. Files

Scripts (`experiments/scripts/`): `E2_behavior_helpers.py`, `E2_behavior_stats.py`, `E2_behavior_b_teacherforced.py` (+ `_b_analyze.py`, `_b_lens_analyze.py`),
`E2_behavior_d_otherAIs.py` (+ `_d_analyze.py`), `E2_behavior_c_forcedchoice.py` (+ `_c_analyze.py`), `E2_behavior_c2_fc_controls.py` (+ `_c2_analyze.py`),
`E2_behavior_c2b_explain.py`, `E2_behavior_c2b_explain2.py`, `E2_behavior_c3_lexical_pairs.py` (+ `_c3_analyze.py`), `E2_behavior_e_identity.py` (+ `_e_analyze.py`),
`E2_behavior_a_gen_TEMPLATE.py` / `E2_behavior_a_gen_<cond>.py`, `E2_behavior_a_classify.py`.
Results (`experiments/results/`): `E2_behavior_{a_gen_<cond>, a_classified, a_summary, a_manual_overrides, b_rows, b_summary, b_lens, b_lens_summary,
c_rows, c_summary, c_pronchoice_rows, c_pronoun_gens, c2_thirdparty_rows, c2_synonym_rows, c2_raw_rows, c2_summary, c2_explain_gens, c2_explain_gens2,
c3_rows, c3_summary, d_rows, d_summary, e_identity_rows, e_identity_gens, e_summary}.json`.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
