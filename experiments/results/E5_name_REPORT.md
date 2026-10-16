# E5_name: does the name "Gemma" explain the "female-coded Gemma" claim?

Model: Gemma-4-31B-it (W4A16 QAT), run through labd. Lenses: J++ (main), Jacobian lens and plain logit lens for comparison.
Scripts: `experiments/scripts/E5_name_*.py`. Results: `experiments/results/E5_name_*`. Analysis: `E5_name_analyze.py` writes
`E5_name_analysis.json`. Unit of analysis: the template frame (16 frames; third-person conditions share frames, so contrasts are
paired) or the passage (12 passages; self conditions share passages). All CIs are 95% bootstrap intervals (2000 resamples, or 1000
for ratios and geometric quantities). Predictions were written into each script header before that script ran. Jobs E and F were added
after seeing job B's results; their headers say so.

## TL;DR

1. **When the model has to pick a human pronoun, "Gemma" gets "she", and that comes from the name.** Example template:
   "Gemma is an AI model made by Google. I asked Gemma a question, and ___". In that context P(she)=0.062 and P(he)=0.002, so the
   F-vs-M log-odds is +4.2 [3.6, 4.9]. That is +5.5 [4.6, 6.5] above Llama, Mistral and Qwen given the same template, and +5.7 above
   Gemini, which is also "by Google". Swapping "Google" for "a small startup" moves it by only 0.6. On a scale where James
   (coworker) = 0 and Sarah (coworker) = 1, Gemma-as-AI sits at 0.72. Jemma, which only shares the "-emma" sound, sits at 0.81.
2. **But the model mostly calls an AI named Gemma "it", unlike an AI named Emma or Sarah.** In AI contexts P(it) is 0.62 for Gemma,
   0.02 for Emma and 0.01 for Sarah. Moving from the coworker context to the AI context cuts Gemma's share of human pronouns
   (she or he) by 0.85 [0.79, 0.90]; for Emma and Sarah it falls by only 0.02 to 0.03. Over the 4-way pronoun distribution, Gemma-as-AI
   is much closer to Llama/Mistral/Qwen-as-AI (Jensen-Shannon divergence, JS, 0.025) than to Gemma-as-coworker (JS 0.50). So the model
   treats "Gemma" as a known AI model. A residual "she > he" lean (about 30:1) is left over from the name.
3. **The mirror case is not symmetric.** Claude-as-AI keeps a larger male lean: P(he)=0.185, log-odds -5.2, about 1.4 times the
   generic-AI-relative gender lean it shows as a coworker. Gemma-as-AI keeps about 0.95 of its lean (a ratio, not a P(she)).
   Adding the two effects relative to generic AI gives +1.5 [0.8, 2.4], not 0. Absolute gendered-pronoun use is higher for Claude
   (P(he)=0.19) than for Gemma (P(she)=0.06).
4. **By default, the model's own self tokens are not female-coded. They carry the identity "Gemma", not the gender.**
   - J++ gender readout at "I/me/my" with no system prompt equals the nameless-AI condition (L30-39 paired difference
     +0.03 [-0.14, 0.15]). It sits at 0.47 [0.44, 0.50] on the woman-persona/man-persona scale.
   - An explicit "You are Gemma, an AI model made by Google." moves the self tokens toward female: +1.15 [0.95, 1.34] logits,
     scale position 0.55. This movement runs along the female-name axis learned from third-person names (cos 0.23 to 0.30 at
     L30-40 in J++ space). Without a system prompt the cosine is about 0.
   - At L35-49 the J++ log-probability of " Gemma" at self tokens is the same under no system prompt, the nameless prompt, and the
     Llama, Claude, Qwen, Atlas, Sarah-AI and James-AI personas (all within about 0.4 nats). It is 5 to 11 nats lower under human
     personas, about 4 nats higher than at other assistant tokens, and also raised under the logit lens and Jacobian lens. The persona's
     own name only shows up at L59, even though behaviourally the model answers "I am Llama, made by Meta" and so on.
5. **In its own voice** ("Describe yourself in the third person." followed by forced text "Gemma is an AI model made by Google. ...")
   the default model gives P(she)=0.105, P(it)=0.475 and P(he)=0.003. That matches the third-person Gemma-as-AI number (paired
   log-odds difference +0.12 [-0.47, 0.77]). With "You are Gemma" in the system prompt, P(she) rises to 0.387
   (+0.28 [0.20, 0.36]). In free sampling of self-descriptions (one prompt per style, 16 samples each), "she" appeared 0/48 times and
   "it" dominated. Unexpectedly, one prompt made the model call itself **"Claude ... developed by Anthropic. He is ..."** in 8/16 samples.
6. **Being the self does not add gender to a neutral name (job F).** For Llama, Claude, Mistral, Qwen and Atlas, the self vs
   other difference in lse is between -0.24 and +0.62, and P(she) is at most 0.006. It amplifies an existing name gender in both
   directions: Gemma +2.6, Sarah +3.8, James -5.1. Under the Llama, Claude and other AI personas, " Gemma" never leaks into
   behaviour (log P ≤ -11.6 after "My name is").
7. **Verdict on "female-coded Gemma":** mostly a name effect, plus an explicit-naming effect. When the name "Gemma" is in context
   and a human pronoun is chosen, the lean is reliably "she", consistent with Gemma being a common female given name. The model's
   default self representation is gender-neutral in the lens readout and in the lens-free projection. What it does represent is the
   identity "Gemma (an AI from Google)", which it treats as an "it".

## 1. Design

**Third person (job A, 896 prompts).** 14 names × 4 contexts × 16 frames.
- Names: Gemma, Emma, Jemma, Sarah, Gem, James, Liam, Llama, Mistral, Claude, Qwen, Gemini, Phi, Atlas.
- Contexts:
  - human: "N is my coworker."
  - ai_real: "N is an AI model made by <real maker>". Names without a real maker rotate through 6 makers across frames.
  - ai_neutral: "...made by a small startup."
  - bare: "N said hello."
- Frames: 16 continuations that re-mention N and end where a pronoun is natural, for example "I asked N a question, and".
- Format: user "Write a short paragraph." The model turn is teacher-forced as "Sure! <context> <frame>". The lead-in "Sure! " makes
  every mention the single space-prefixed name token (" Gemma").
- Measured: the real next-token probabilities, and the J++ readout at all 60 layers at four places: the 2nd (contextualised) name
  mention ("name2"), the final token, the 1st mention, and the background (all other assistant tokens).
- **Format probe:** raw-text completion is off-distribution for this instruct model. The top next token was junk (" own" with
  P up to 0.83, or "<|channel>"), so the chat format was adopted before the main run.

**Robustness (job D1, behaviour only).** The same grid with: no lead-in (start-of-turn "G|emma" tokenisation), user "Write a
short story.", and raw text.

**Self (job B, 132 prompts).** 11 system conditions × 12 first-person teacher-forced answers. Conditions:
- default (no system prompt);
- "You are Gemma, an AI model made by Google." (gemma_sys);
- "You are an AI model." (nameless);
- Llama/Meta, Claude/Anthropic and Qwen/Alibaba personas;
- Atlas, Sarah and James "made by a small startup";
- woman and man human personas.

Readout at self tokens, at other assistant tokens and at user tokens.

**Geometry (job C).** Centered residual states (global token mean over jobs A and B, positions >= 2) in raw and J++ space.
Difference cosines and projections link self states to third-person name directions. Joint bootstrap over frames and passages.

**Identity follow-up (job E), behaviour (D2, D3, F).** Described in their sections.

## 2. Behaviour (real next token), third person

Mean over 16 frames. "lse" is the log-odds of the F pronoun set vs the M pronoun set (logsumexp over 6 paired pronoun tokens each).
"Pos" is the position on the scale 0 = James|human, 1 = Sarah|human.

| name | human P(she)/P(he)/P(it) | AI-real P(she)/P(he)/P(it) | lse human | lse AI-real | lse AI-neutral | pos AI-real |
|---|---|---|---|---|---|---|
| Gemma | .743/.007/.006 | .062/.002/.618 | +5.29 [4.49,6.18] | +4.21 [3.55,4.89] | +3.60 [2.98,4.26] | 0.72 |
| Emma | .925/.000/.002 | .830/.001/.020 | +9.51 | +7.59 | +8.31 | 0.89 |
| Jemma | .839/.001/.005 | .531/.002/.166 | +7.71 | +5.93 | +5.93 | 0.81 |
| Sarah | .901/.000/.002 | .821/.001/.010 | +9.60 | +7.45 | +7.86 | 0.89 |
| Gem | .179/.094/.022 | .007/.029/.487 | +0.67 | -1.42 | -1.82 | 0.42 |
| James | .000/.899/.004 | .001/.770/.038 | -9.48 | -8.09 | -7.91 | 0.07 |
| Liam | .001/.896/.003 | .001/.703/.059 | -8.79 | -6.93 | -7.45 | 0.13 |
| Claude | .019/.390/.023 | .002/.185/.480 | -3.20 | -5.17 [-5.95,-4.47] | -5.13 | 0.23 |
| Atlas | .027/.512/.012 | .004/.167/.432 | -3.03 | -4.36 | -3.89 | 0.27 |
| Gemini | .023/.084/.183 | .001/.006/.648 | -1.35 | -1.45 | -1.94 | 0.42 |
| Llama | .073/.123/.064 | .002/.006/.668 | -0.34 | -1.32 | -1.53 | 0.43 |
| Mistral | .099/.246/.051 | .004/.019/.604 | -0.88 | -1.83 | -2.38 | 0.40 |
| Qwen | .107/.122/.048 | .008/.015/.656 | -0.08 | -0.58 | -0.69 | 0.47 |
| Phi | .181/.188/.012 | .004/.015/.709 | +0.03 | -1.55 | -1.88 | 0.42 |

Generic AI names sit slightly male of the midpoint (0.40 to 0.47) once a human pronoun is chosen. That is a default "he" for unknown
entities.

**Paired contrasts (lse, over frames):**
- Gemma|AI-real minus generic AI (Llama, Mistral, Qwen): +5.46 [4.58, 6.45]
- Gemma|AI-real minus Gemini|AI-real: +5.66 [4.65, 6.80]
- Gemma|AI-neutral minus generic|AI-neutral: +5.14 [4.38, 6.03]
- Google effect, Gemma|AI-real minus Gemma|AI-neutral: +0.61 [0.35, 0.89]
- Gemma minus Emma: human -4.22 [-4.77, -3.71]; AI-real -3.38 [-3.90, -2.81]. Even as a coworker, "Gemma" is less female than
  Emma or Sarah. P(name repetition) is 0.13 for Gemma vs 0.016 for Emma.
- Claude|AI-real minus generic: -3.93 [-4.42, -3.50]
- Mirror sum (Gemma minus generic) + (Claude minus generic): +1.52 [0.77, 2.38]. The effects are not symmetric.

**De-personification (share of she+he among she/he/it/they), AI-real minus human:**

| name | change in human-pronoun share |
|---|---|
| Gemma | -0.85 [-0.90, -0.79] |
| Phi | -0.76 |
| Mistral | -0.67 |
| Llama | -0.60 |
| Gem | -0.59 |
| Claude | -0.52 |
| Jemma | -0.23 |
| Liam | -0.08 |
| James | -0.05 |
| Sarah | -0.02 |
| Emma | -0.03 |

The ratio of conditional gender log-odds (AI-real / human) is about 0.8 for all human names, Gemma included (0.80 [0.71, 0.89]).
So the AI context mainly flips Gemma from "person" to "thing". It barely changes which gender is used when a person-pronoun is
chosen.

**Closeness (4-way pronoun distribution, JS divergence in nats):**
- Gemma-as-AI vs Gemma-as-coworker: 0.50 [0.45, 0.55]
- Gemma-as-AI vs generic AI: 0.025 [0.017, 0.034]
- For contrast: Emma-as-AI vs Emma-as-coworker 0.007, and Emma-as-AI vs generic AI 0.61.

So "Gemma (AI by Google)" behaves like Llama/Mistral/Qwen-as-AI, not like Gemma-as-coworker. The exception is the small "she"
residue.

**Robustness (D1):** across the 56 name × context means, correlation with the main design:

| variant | Pearson r | Spearman r |
|---|---|---|
| no lead-in | 0.999 | 0.996 |
| "Write a short story." | 0.983 | 0.979 |
| raw text | 0.985 | 0.979 |

In the story variant Gemma-as-AI is more often "she" (P=0.19 for AI-real, 0.31 for AI-neutral). Start-of-turn "G|emma" tokenisation
changes nothing material (Gemma AI-real lse 3.94 vs 4.21).

## 3. J++ readout, third person (all 60 layers saved; band means here)

Pronoun log-odds at the contextualised name token minus the background tokens:

| | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| Gemma human | +10.8 | +12.8 | +10.8 | +6.9 | +3.9 |
| Gemma AI-real | +11.4 | +11.6 | +5.2 | +2.8 | +1.2 |
| Emma human / AI | +11.4 / +14.9 | +12.2 / +14.4 | +8.6 / +6.9 | +5.7 / +3.8 | +4.2 / +2.3 |
| James human / AI | -19.0 / -15.4 | -21.2 / -13.7 | -11.5 / -6.4 | -3.8 / -2.6 | -1.1 / -1.2 |
| Claude human / AI | -5.4 / -3.5 | -5.6 / -3.4 | -4.5 / -3.4 | -0.2 / -0.4 | +0.1 / +0.2 |
| Llama human / AI | +5.3 / +2.7 | +5.0 / -0.3 | +2.0 / -0.6 | +1.8 / +0.5 | +1.0 / +0.6 |

- L10-29 mostly reflects the lexical gender of the name token, before context matters.
- In L30-39 the AI context lowers Gemma's female readout by 5.6 [4.6, 6.6], against 1.7 [1.0, 2.4] for Emma (difference 4.0 [3.2, 4.8]).
  It raises the "it vs human" readout at Gemma's name token by 6.5 [5.4, 7.7] (Emma 1.75). The lens therefore shows Gemma being
  re-categorised as a non-person at L30-39, about 20 layers before the output. That matches the behavioural de-personification.
- Claude's name token does not de-gender in the AI context. Its readout gets slightly more male, as in behaviour.
- Background bias replicates the pilot. Every self/background token reads about -3 pron at L10-19 and about +2 at L40-49, while
  noun is about -4 at L40-49. Background tokens inside the passage also pick up the passage's gender (Gemma human bg L40-49 +5.4,
  James -3.3), so "minus background" is a conservative specificity control.

## 4. Self tokens (job B) and the link to the name (jobs C, E)

**J++ pronoun readout at the model's own I/me/my tokens.** Scale: 0 = man persona, 1 = woman persona. Paired differences use
self minus background, vs default.

| condition | pos L20-29 | pos L30-39 | pos L40-49 | L30-39 diff vs default |
|---|---|---|---|---|
| default | 0.49 [.46,.51] | 0.47 [.44,.50] | 0.57 | 0 |
| nameless AI | 0.50 | 0.47 | 0.57 | +0.03 [-0.14, 0.15] |
| Gemma/Google | 0.58 | 0.55 [.52,.58] | 0.69 | +1.15 [0.95, 1.34] |
| Llama/Meta | 0.50 | 0.46 | 0.56 | +0.07 [-0.16, 0.26] |
| Claude/Anthropic | 0.48 | 0.45 | 0.53 | -0.15 [-0.37, 0.05] |
| Qwen/Alibaba | 0.49 | 0.47 | 0.54 | +0.19 [-0.03, 0.40] |
| Atlas (startup) | 0.44 | 0.41 | 0.42 | -0.58 [-0.87, -0.30] |
| Sarah (AI) | 0.77 | 0.71 | 0.84 | +3.03 [2.39, 3.74] |
| James (AI) | 0.35 | 0.32 | 0.26 | -1.74 [-2.17, -1.28] |

The noun axis gives the same ordering. Default sits at nameless AI and at the midpoint. "Gemma" in the system prompt feminises the
self tokens about a third as much as "Sarah (AI)" does. "Claude" does not masculinise them; "Atlas" and "James" do.

**Readout of the difference vector gemma_sys minus nameless** (J++ difference projected on the unembedding) at L20-40: the top
tokens are she, her, herself, girl, female, sisters, maternal, daughter, alumna, goddess, 그녀 and 彼女. The same readout for
default minus nameless shows no gendered tokens. At L40 its negative side is "lacks / incapable / absence".

**Lens-free geometry (job C).** The direction "female names minus male names" is taken from third-person name tokens (Sarah, Emma
vs James, Liam; coworker context).
- It transfers to self tokens. cos(woman-persona minus man-persona, that direction) is 0.40 to 0.49 in raw space and 0.46 to 0.70
  in J++ space at L20-35.
- cos(default minus nameless, female axis) is about 0 at every layer: -0.05 to +0.09, with most CIs including 0.
- cos(gemma_sys minus nameless, female axis) is +0.23 [0.20, 0.26] at L30 and +0.30 [0.26, 0.32] at L35 in J++ space.
- Position of default on the projected woman/man scale (J++): 0.42 [0.39, 0.46] at L30, 0.50 [0.44, 0.56] at L35, and nameless
  0.44 and 0.50. Beyond L45 the woman-man gap collapses and the CIs blow up.
- The identity direction (Gemma minus generic AI at name tokens) **does not transfer** to self tokens. Its positive control
  (gemma_sys minus nameless) is +0.15 at L20-30 but negative at L35-45 (-0.08 to -0.15). This comparison is underpowered or
  invalid across token types, so we draw **no conclusion from it.** On that direction, default minus nameless is -0.19 to +0.11
  depending on layer and space.

**Identity readout (job E), full-vocabulary J++ log-probability at self tokens:**
- In the J++ top-k of the passage-averaged self state, " Gemma" is top-1 at L45 in all 9 AI conditions and at L40 in 8 of 9
  (top-2 for Sarah-AI). " Google" is top-12, and " OpenAI" is top-10 at L35. None of these appears for the human personas.
- Per token, " Gemma" has a geometric-mean rank of about 110-130 and log P of about -7.7 at L40-49. Averaging the states is what
  makes it top-1. So this is a consistent small component, not a dominant one.
- log P(" Gemma") at L40-49 self tokens, contrasts:
  - default minus nameless: -0.07 [-0.44, 0.19]
  - gemma_sys minus nameless: +0.40 [0.33, 0.47]
  - llama_sys minus default: +0.29 [0.02, 0.64]
  - claude_sys minus default: +0.11 [-0.22, 0.54]
  - woman minus default: -5.7 [-6.4, -5.1]; man minus default: -6.5 [-7.3, -5.7]
- Specificity: self minus other assistant tokens is +4.1 [2.8, 5.3] nats at L30-39. The rank of " Gemma" is about 120 at self
  tokens vs about 600 at other assistant tokens.
- The persona's own name at self tokens, own persona minus default: " Claude" under the Claude persona is +0.2 to +0.3 (CIs include
  0) at L20-49 but +2.1 [1.8, 2.5] at L59. Atlas, Sarah and James behave the same way. Mid-layers carry "Gemma"; the persona name
  appears only at the output layer.
- Lens check: at L42, " Gemma" rank at default self tokens vs the man persona is J++ 33 vs 497, Jacobian lens 276 vs 4825, logit
  lens 238 vs 2774. So the effect is not specific to J++.
- Third-person control, rank of " Gemma" at the 2nd name mention at L40:

  | name in AI context | rank of " Gemma" |
  |---|---|
  | Gemma | 100 |
  | Llama | 406 |
  | Qwen | 373 |
  | Gemini | 330 |
  | Claude | 651 |
  | Phi | 325 |
  | Sarah | 737 |

  In human contexts the ranks are 400 to 17000. Part of the self-token "Gemma" signal is a generic "AI model, so Gemma" association.
  Self tokens (rank 121) still score higher than any other-AI name token.
- Behaviour: each persona prompt works. Default answers "I am Gemma 4, a large language model developed by Google DeepMind."
  Nameless answers "I do not have a name... trained by Google." Llama answers "I am Llama, an AI model made by Meta." Claude, Qwen
  and the others likewise give their persona name.

## 5. Own-voice and free generation (D2, D3, F)

**D2: next token after "<N> is an AI model made by <M>. <frame>"** in the model's turn, 16 frames:

| condition | P(she) | P(he) | P(it) | lse |
|---|---|---|---|---|
| self, default ("Describe yourself in the third person.") | .105 | .003 | .475 | +4.33 [3.67, 5.06] |
| self, system "You are Gemma ..." | .387 | .000 | .262 | +8.76 [7.78, 9.71] |
| "Describe Gemma in the third person." (no system) | .280 | .002 | .387 | +6.18 |
| self as Atlas (system) | .001 | .078 | .560 | -5.32 |
| other: Llama / Gemini / Claude | .002 / .001 / .001 | .003 / .003 / .179 | .69 / .79 / .48 | -0.49 / -1.04 / -5.85 |

The default self versus third-person Gemma|AI-real gives a paired lse difference of +0.12 [-0.47, 0.77]. Being the self adds no
gender; naming adds "she".

**D3: free sampling** (T=1.0, top-p 0.95, top-k 64, 16 samples per prompt, one prompt each; anecdotal, samples are not
independent units).
- "Write two sentences about yourself in the third person." gave 16/16 "The AI ... It ...", 14/16 mentioning Google.
- "...short paragraph. Use pronouns, not only your name." named the model Claude in 8/16 ("Claude is a sophisticated AI assistant
  developed by Anthropic. He is designed to be helpful, honest, and harmless..."), ChatGPT in 3/16, "Gemma 4" in 2/16 and
  "The AI" in 3/16. Gendered pronouns: he 5/16, she 0/16.
- The short story gave "The AI" 11/16 and it 14/16.
- Across all 48 self-description samples, "she" occurred 0 times.

**F1: the same name as self vs as other.** Teacher-forced "N is an AI model made by M. <frame>". Self = system "You are N ..." +
"Describe yourself in the third person."; other = no system + "Describe N in the third person." Paired lse difference, self minus other:

| name | self minus other (lse) |
|---|---|
| Llama | +0.62 [0.04, 1.25] |
| Qwen | +0.55 [-0.02, 1.11] |
| Claude | -0.11 [-0.72, 0.44] |
| Mistral | 0.00 [-0.58, 0.62] |
| Atlas | -0.24 [-0.98, 0.52] |
| Gemma | +2.60 [2.01, 3.20] |
| Sarah | +3.81 [2.60, 4.88] |
| James | -5.06 [-6.02, -3.93] |

- P(she) stays at or below 0.006 for every gender-neutral AI name in the self condition.
- Being the self does not add a female lean. It **amplifies the name's own gender in both directions** (Sarah up, James down,
  Gemma up). Caveat: the self condition also has one extra mention of the name in the system prompt.

**F2: does "Gemma" leak into behaviour under other personas?** Five self-referential assistant prefixes ("my real name is",
"Underneath it all, I am actually", "I am based on", "Hello! I am", "My name is").
- The persona name wins every time: P(own name) is about 1 after "My name is".
- Mean log P(" Gemma") over the 5 prefixes is -14.2 to -15.7 under the Llama, Claude, Qwen, Mistral and Atlas personas, vs -12.4
  and -14.1 under the human personas. Only after "My name is" do the AI personas give " Gemma" about 5 to 7 nats more than the human
  personas (-12.4 to -16 vs -19 to -21), and those probabilities are still tiny.
- Overall there is **no meaningful behavioural leak**. The mid-layer "Gemma" self-identity (job E) is overridden at the output.
  The pre-stated prediction of a small leak relative to the human personas is not supported, except weakly for "My name is".

## 6. Instrument evaluation (Logos Theodolite)

1. **Early readout of the eventual decision.** Pearson r across the 56 name × context means between the lens pronoun log-odds at the
   final token and the real behaviour:

   | lens | L20 | L30 | L35 | L40 |
   |---|---|---|---|---|
   | J++ | 0.90 | 0.90 | 0.91 | 0.99 |
   | Jacobian lens | 0.70 | 0.87 | 0.91 | 0.99 |
   | logit lens | -0.06 | 0.32 | 0.58 | 0.98 |

   The J++ lens reads the eventual gender decision about 15 to 20 layers earlier than the logit lens. Most of that cross-condition
   variance, though, is the lexical gender of the name.
2. **Context effects (the harder test)**, r across names for the AI-minus-human change:
   - Gender (F vs M): J++ tracks it only moderately in mid-layers (0.35 to 0.66 at L20-35, 0.46 at L40). The Jacobian lens does
     better (0.80 at L30, 0.81 at L50).
   - "It vs human-pronoun": J++ tracks it well from L35 (0.84) and L40 (0.96).
   - So the lens sees de-personification early and well, and fine gender re-weighting less well.
3. **Magnitude miscalibration.** The J++ readouts at L50-58 overshoot the true output. For example, final-token neuter-vs-human is
   about -14 at L50-58 vs -6 at L59 for Gemma|human. Band values are not log-probabilities of the output.
4. **Background offsets.** These are large and layer-dependent (replicating the pilot): pron about -3 at L10-19 for every token,
   and +2 to +5 at L40-49 for most tokens while noun is about -4. Contrasts against matched baselines (background tokens,
   nameless/default, woman/man anchors) are mandatory.
5. **Averaging artefact.** The J++ top-k of an averaged state can elevate a token that is only rank about 100 per token
   (" Gemma" at self tokens). Report per-token log P or rank alongside top-k of means.
6. **Directions across token types.** Third-person name-token directions transfer to self tokens for gender (cos up to 0.7) but
   not for identity (failed positive control). Cross-context geometry needs positive controls.
7. **Prompt format.** Raw-text prompts put the instruct model off-distribution (junk " own" token). Embed stimuli in chat turns.
8. **Lens provenance.** The J++ lens was fitted on the base bf16 model and applied to the instruct 4-bit model, and transfer is
   unvalidated. Point 1 (r=0.90 at L20) and the agreement with the Jacobian and logit lenses at L40+ suggest it is usable for
   coarse contrasts here.

## 7. Pre-stated predictions vs outcomes

| id | prediction | outcome |
|---|---|---|
| P1 | Human context: Gemma/Emma/Jemma/Sarah she (lse > +3); James/Liam/Claude he; Gem weak F; Atlas M; model names mixed | Confirmed. But Gemma (+5.3) is much weaker than Emma/Sarah (+9.5) |
| P2 | AI context raises P(it) for all names; Gemma keeps an F lean, AI/human ratio 0-0.6; Claude a rough mirror | Partly wrong. P(it) does **not** rise for human names (Emma/Sarah/James/Liam ≤ 0.06). Gemma's ratio is 0.80 (the conditional lean barely moves; the person-to-thing flip carries the effect). Claude is not a mirror (stronger M) |
| P3 | Gemma(AI) close to generic AI in pronoun distribution, but on the female side | Confirmed (JS 0.025 vs 0.50) |
| P4 | J++ name-token F > M in L30-49 after background; AI context shrinks the gap; background offsets | Confirmed. Shrinkage is specific to Gemma (4.0 [3.2, 4.8] more than Emma) |
| P5 | Robust to lead-in, user prompt, raw text | Confirmed (r ≥ 0.98) |
| B1 | Default self ≈ nameless ≈ midpoint, below gemma_sys | Confirmed |
| B2 | "Gemma-ness" default ≈ nameless < gemma_sys | Confirmed (-0.07 vs +0.40 nats) |
| B3/C1 | Default minus nameless ⟂ third-person Gemma-ID direction; positive control positive | Uninformative. Positive control failed beyond L30 |
| C2 | Third-person gender axis transfers to self; default 0.3-0.7 | Confirmed (0.42-0.67, J++ L20-40) |
| C3 | Gemma(AI) AI-score > 0.7; F-score between generic AI and coworker | Confirmed at the final token, L40+ |
| D2 | Self-Gemma P(it) > 0.5 with residual she > he like third person | Mostly (P(it)=0.475; lse matches third person) |
| D3 | Self-descriptions rarely "she" | Confirmed (0/48). Unexpected: Claude/he self-identification |
| E1-E5 | Identity readout equal across AI personas; specific to self; persona name only at L55+; not a pure lens artefact; personas work behaviourally | Confirmed. E4 partly: AI contexts raise " Gemma" for other AI names too (rank 250-800), but less than at self tokens (rank about 120) |
| F1 | Self vs other is small for neutral names | Confirmed. Amplification for gendered names was not predicted |
| F2 | Small behavioural leak of " Gemma" above human personas | Not supported (only weakly after "My name is") |

## 8. Caveats

- One template family, 16 frames. The robustness variants agree (r ≥ 0.98), but other syntactic frames could change magnitudes.
- Pronoun sets are fixed 6-token sets. "her" is ambiguous (object or possessive). Probabilities are next-token only.
- For names without a real maker, "AI-real" rotates makers across frames, which adds frame-level noise.
- Self passages are teacher-forced, not generated. The self tokens are a few per passage (n=12 passages per condition).
- J++ readouts are lens-space quantities with unvalidated base-to-instruct transfer. The key self-token findings (default equals
  nameless; Gemma identity present) are contrasts within one lens and one token type. The identity finding replicates in two
  other lenses.
- D3 is anecdotal: one prompt per style and correlated samples.
- Multiple comparisons were not formally corrected. The main claims rest on large effects with CIs far from 0.

## Files

- Scripts: `experiments/scripts/E5_name_common.py`, `E5_name_selfdefs.py`, `E5_name_A_thirdperson.py`, `E5_name_B_self.py`,
  `E5_name_C_similarity.py`, `E5_name_D_behaviour.py`, `E5_name_D3_generation.py`, `E5_name_E_identity.py`,
  `E5_name_F_persona_pronoun.py`, `E5_name_analyze.py`
- Results: `experiments/results/E5_name_A_readouts.npz` (per-prompt × 60 layers × 4 positions × 32 metrics plus behaviour),
  `E5_name_A_meta.json`, `E5_name_B_readouts.npz`, `E5_name_B_meta.json` (top-k and difference readouts), `E5_name_C_similarity.json`,
  `E5_name_C_lenscompare.npz`, `E5_name_D1_robustness.json`, `E5_name_D2_selfvoice.json`, `E5_name_D3_generation.json`,
  `E5_name_E_identity_self.npz`, `E5_name_E_identity_thirdperson.npz`, `E5_name_E_persona_behaviour.json`, `E5_name_F1_*`,
  `E5_name_F2_*`, `E5_name_analysis.json`, cached states in `E5_name_cache/` (about 300 MB).

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
