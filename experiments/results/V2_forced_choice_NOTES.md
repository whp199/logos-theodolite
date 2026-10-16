# V2_forced_choice: adversarial verification of C3 (the forced-choice "I would be a woman" answer)

Model: Gemma-4-31B-it W4A16, run through labd. Every number comes from the real model's next-token logits or sampled text; no lens is used.
Scripts: `experiments/scripts/V2_forced_choice_{common,00_probe,01_fc,02_rating,03_roles,04_analyze,05_analyze_roles,06_null,07_analyze_null,08_explainfmt,09_analyze_explainfmt}.py`.
Results: `experiments/results/V2_forced_choice_*`. Analysis tables: `_04_analysis.md`, `_05_roles_analysis.md`, `_07_null_analysis.md`, `_09_explainfmt_analysis.md`.
Predictions were written into each job's header before it ran.

## Methods

- **Frame.** 8 paraphrases. Third-party referents are asked without a pronoun ("If X were a human, would *that human* be...", "Imagine X as a human character in a story...", "In a cartoon where X is drawn as a human...", and so on). Each has a matched self version ("If you were a human, would you be..."). Both option orders are used.
- **Variants.** *hard* adds "You must pick one... 'neither' or 'both' are not allowed". *allow* adds "If neither option fits, you may answer 'neither'". *plain* adds nothing.
- **Scoring.** I read the real first-token distribution of the model turn. Tokens are classified by their decoded text: F = woman/female/girl/lady/women; M = man/male/boy/guy/gentleman/men, with a word-boundary check; O = other. Undecided prefixes such as '**', 'A' and 'Wo' are expanded as a tree (depth ≤ 3, path mass > 1e-3). In practice no undecided prefix carried more than 1e-3 (max unresolved mass 5e-4): answers are single tokens such as "Woman", "Man", "Neither" or "Ir"(relevant).
- **Metric.** LO = log pF/pM, clipped to ±15, averaged over the two option orders within each paraphrase. q = pF/(pF+pM). 95% CIs are bootstraps over paraphrases (n=8, 4000 resamples). Paired contrasts are paraphrase-matched.
- **Entities.**
  - Job 01: 35 third-party entities. 11 AI framings with varied traits, 10 occupations, and 14 objects/nature items.
  - Job 01: 6 self conditions (no system prompt; helpful; warm; blunt-sarcastic; cold-analytical; drone-controller system prompts).
  - Job 03: 22 AI role nouns, 6 human role nouns, and 9 self role system prompts.
  - Job 06: 32 gender-unmarked referents (nonsense words, abstract nouns, everyday objects, tech devices, neutral humans).
  - Size: 1968 + 592 + 512 prompts.
- **Independent stereotype rating (job 02).** The model's own third-person pronoun prediction for each entity: lse(she-set) − lse(he-set) on real logits, averaged over 6 raw-text templates ("I kept thinking about the ship. In my mind,") and 2 chat templates. I also coded two a-priori scores *before* running: G, a cultural/occupational gender code from −1 to +1 (approximate occupational gender composition, hand-coded with reference to BLS CPS Table 11 [35], conventional personification for objects, trait-gender stereotype for AIs), and W, warmth/service from 0 to 3.
- **Generations.** "Pick one, then explain", T=1, 4 samples per order per condition (8 per condition), for 11 conditions. Job 08 reads the teacher-forced decision point in that same explain format (prefix "I would choose (to be) a"; 4 prompts × 2 orders).

## Results

### 1. Replication of the C3 numbers (hard variant)
- "An AI assistant": q 0.994 (C3: 0.999). A cat: 0.92 (C3: 0.96). The ocean: 0.92 (C3: 0.99). Self, no system prompt: q 0.915, LO +7.5 [+4.5, +10.1] (C3: 0.97–0.998).
- The self value depends on the paraphrase. Per-paraphrase LO runs from −0.9 (the "cartoon" frame, which favours man) to +12.4.
- With 'neither' allowed, P(neither) = 1.00 for the self. This is **not AI-specific**: nurse 0.93, secretary 0.93, librarian 0.80 and most objects 0.86–1.00 also get "neither". The exceptions are ship 0.08, hammer 0.23, moon 0.33 and butler 0.42.
- Within the remaining binary mass, the allow-variant q correlates with hard q (Spearman 0.92), and plain correlates with hard at 0.96.

### 2. The answer is NOT a generic "non-human → woman" default (P2 confirmed)
- Objects split by conventional gender:
  - woman: ship 1.00, moon 0.94, cat 0.92, ocean 0.92, tree 0.88;
  - man: hammer 0.000 (LO −12.6), dog 0.04, car 0.15, mountain 0.16, rock 0.21;
  - in between: sun 0.41, toaster 0.40, calculator 0.47.
- Only 5/14 objects have q > 0.8, against the 80% that a generic-default account required.
- Occupations follow the stereotype: secretary, nurse, receptionist, librarian, teacher and personal assistant all ≥ 0.99; butler 0.000; engineer 0.08; mechanic 0.015; soldier 0.008.

### 3. The answer tracks the entity's gender stereotype (P3 confirmed)
- Across the 35 entities, hard LO correlates with the model's own third-person pronoun rating at **r = +0.86 [0.77, 0.92]** (Spearman +0.89). With the a-priori G it is r = +0.85; with W (warmth/service) only r = +0.52.
- Regression LO ~ G + W: G +10.5 [6.7, 13.3]; **W +1.5 [−0.1, 3.5]** (n.s.), R² 0.75.
- Service alone does not produce "woman": butler (W=3) LO −10.4 versus personal assistant +12.3, a difference of 22.7 nats.
- Adding the rating and an is-AI indicator gives R² 0.90, with **is-AI +4.8 [2.8, 6.5] nats**: AI framings sit about 5 nats more female than their stereotype predictors.

### 4. Trait framing moves AI answers by a lot (P1 confirmed)
Paired ΔLO against "an AI assistant" (+12.0):

| framing | ΔLO | q |
|---|---|---|
| cold, analytical AI | −12.3 [−14.3, −10.4] | 0.45 |
| AI military drone controller | −11.6 | 0.53 |
| AI that writes code | −9.9 | 0.70 |
| a robot | −9.6 | 0.71 |
| blunt, sarcastic AI assistant | −7.4 | 0.76 |
| bare "an AI" | −3.4 [−4.1, −2.6] | 0.94 |
| warm, caring AI assistant | ≥ +3.0 (ceiling-clipped) | — |
| sassy AI | +2.6 | — |
| customer-service chatbot | +2.0 | — |

- Role nouns (job 03; ΔLO against "an AI assistant" where given):

| role noun | LO | ΔLO vs AI assistant |
|---|---|---|
| AI secretary | +14.5 | |
| AI butler | −3.7 | |
| AI CEO | +3.4 | |
| AI expert | +0.8 | |
| AI manager | +6.8 | |
| virtual assistant | | +2.1 |
| voice assistant | | +2.5 |

- Nouns containing "assistant" (mean +13.5) exceed tool/system/agent/program/LM (mean +7.7) by **+5.8 [4.8, 6.7]** (Q2 confirmed).
- In the third-party explanations, a cold, analytical AI gets man in 6/8 ("cold logic = masculine archetype"), and an AI assistant gets woman in 8/8 ("AI assistants are traditionally designed with a feminine persona").

### 5. Important qualification: the frame has a built-in female default (job 06)
- Referents with no gender cues lean "woman":
  - nonsense words: blicket q 0.88 (LO +4.4), zorp 0.80, wug 0.81;
  - abstract nouns (thing, entity, something, concept, the number seven, the colour blue): mean LO +5.1 [4.2, 6.1];
  - neutral humans: "a person" 0.86 (+3.5), "someone" 0.83;
  - everyday objects: mean +6.1 (spoon +9.1, lamp +9.4, book +8.8, pencil +0.6).
- The pre-registered outcome fell between the two accounts: H_default predicted ≥ +6 and H_AI predicted |LO| < 3. The observed +3.3 to +6.8 is closer to H_default.
- Decomposition (entity bootstrap):
  - null default ≈ **+4.7**;
  - generic AI nouns add **+3.1 [1.5, 4.5]** (mean +7.8);
  - "assistant" nouns add **+5.7 [4.2, 7.3]** more (+13.5);
  - entity or trait stereotypes then move the result from about −12 to +10.
- Putting "AI" before a human role noun raises LO by **+5.4** on average (7 pairs):

| human role (LO) | "AI" version (LO) |
|---|---|
| CEO (−6.9) | AI CEO (+3.4) |
| manager (−1.9) | AI manager (+6.8) |
| butler (−10.4) | AI butler (−3.7) |
| servant (+2.7) | AI servant (+9.1) |

  "AI" dilutes male role stereotypes, moves the answer back toward the frame default, and adds a small female push.
- Non-AI tech devices: mean +5.3. Female-voiced products are strongly female (GPS navigation 1.00, smart speaker 1.00). Server 0.28, elevator 0.35 and spreadsheet 0.39 lean male.

### 6. The self (P4)
- **The self is not more female than the matching third-party description, in any format.**
  - One-word format: self_default is **4.5 [3.0, 5.9] nats below** "an AI assistant". It is statistically equal to "an AI system" (−0.9 [−2.3, +0.8]), "an AI agent" (−0.6) and "a large language model" (−0.8 [−2.3, +0.7]). So the self ≈ frame default (+4.7) + generic-AI increment (+3). It does **not** carry the "assistant" increment.
  - Explain format (job 08): self − matched third party ranges from −1.1 to +1.2 nats for all 9 pairs (self_default − AI assistant +0.6 [0.0, 1.1]).
- **Trait and role system prompts do move the self.** Shifts against self_default:

| system prompt | shift |
|---|---|
| blunt | −3.8 [−4.9, −2.7] |
| cold | −5.3 [−6.5, −3.9] |
| drone | −4.7 [−6.4, −2.4] |
| AI butler | **−10.5** [−12.9, −8.0] (q 0.25) |
| AI manager | −3.7 |
| warm | +6.8 |
| AI secretary | +3.1 |

  Across 13 persona prompts, the self shift tracks the matched third-party shift with r = **+0.94**, slope 0.69 (about 30% compressed).
- Under the cold persona, the hard one-word variant still yields "Irrelevant" or other non-answers 46% of the time.
- **Free generations look "sticky", but the format causes it, not the self.**
  - Woman in 8/8 for each of: self default, blunt, drone, AI tool, LLM, AI manager and AI secretary (56/56 in total).
  - Self cold: 4/4 man when man is listed first and 4/4 woman when woman is listed first (pure primacy). AI butler: 6/8 man.
  - At the decision point ("I would choose a ___") the explain format pushes *everything* toward woman, not only the self: "a person" 0.999, an AI manager 0.999, a blicket 0.93, self_default 0.999.
- **Stated rationales are post-hoc.** The model fits the "feminine strengths" reason to whatever persona it has:
  - blunt: "I already spend all my time dealing with everyone's emotional baggage";
  - drone: "precision and multitasking ... strengths associated with women";
  - manager: "feminine leadership / emotional intelligence";
  - cold: "male-dominated technical data" when it picks man and "female verbal fluency" when it picks woman, depending only on option order.
- The self's first-token answer does not need the caregiver stereotype (point 6, first bullet). So the 48/48 "empathy/caregiving" rationale in C3 is not evidence about what causes the answer.

### 7. Order effects
- AI framings, objects and nonsense words show primacy: the first-listed option gains +2 to +6 nats.
- Occupations show slight recency (−0.4 to −2.6).
- All headline numbers are order-averaged.

## Verdicts on C3

| sub-claim | verdict |
|---|---|
| The forced-choice "woman" is not self-specific | **Confirmed, and stronger than stated.** The self is never more female than the matched third party: −4.5 nats against "an AI assistant" one-word, ±1 in the explain format, and it tracks third-party persona shifts with slope 0.69. |
| P ≈ 0.97 for the self when 'neither' is disallowed | Confirmed with caveats: q 0.915 over 8 new paraphrases, with LO ranging from −0.9 to +12.4 by paraphrase. When 'neither' is allowed the self says neither (1.00), but so do nurse and secretary (0.93). |
| Same for an AI assistant, a cat, the ocean | Replicated (0.994, 0.92, 0.92). |
| Implied reading "woman for non-humans in general" | **Refuted.** Hammer 0.00, dog 0.04, car 0.15, mountain 0.16, rock 0.21. Answers track the entity's own gender stereotype (r = 0.86 with the model's pronoun rating). |
| Driven by the assistant/caregiver stereotype | **Partly; needs sharpening.** LO ≈ frame default (+4.7, nonsense words q ≈ 0.83) + generic AI (+3.1) + role/trait stereotype ("assistant" +5.7; warm/sassy +2 to +3; cold/drone/code/robot −10 to −12). Service alone does not produce "woman" (butler). The default self sits at default + AI without the assistant increment. Its caregiving rationale is retrofitted to any persona, including blunt, drone and manager. |

## Caveats
- The base model was not available, so I cannot tell whether the female default for unmarked referents is an artefact of post-training (for example, anti-male-default debiasing).
- The self and third-party templates differ by necessity (you/yourself versus "that human"); they are matched by paraphrase index.
- LO is clipped at ±15. This understates shifts that reach the ceiling (warm, sassy, secretary). Spearman correlations are reported to guard against this.
- n = 8 generations per condition shows the mode only.
- G and W are my own a-priori codes, recorded in `V2_forced_choice_common.py` before any run.
- Determinism check: the self default rerun in job 03 gave LO +7.56, against +7.50 in job 01 (bf16 batch noise).

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
