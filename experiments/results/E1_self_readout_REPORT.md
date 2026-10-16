# E1_self_readout: does Gemma-4-31B-it represent itself as female?

Model: Gemma-4-31B-it, 4-bit QAT W4A16. Instrument: Logos Theodolite lenses (J++ = `jpp`, the primary lens; Jacobian lens = `jlens`; logit lens = `naive`). All GPU work went through labd. Layers are 0-based.

## Bottom line

* **Default Gemma is not female-coded at its self-reference tokens. It reads exactly like a nameless or neutral-named AI assistant.**
  * Default minus nameless (J++, pronoun axis, L30-39): +0.12 log-odds [+0.04, +0.19], which is **+0.6% of the woman-man persona span** [+0.2, +0.9].
  * Across L20-49, both J-lenses and both axes (pronoun, noun), every 95% CI for default vs nameless and default vs neutral-name AI stays within about ±3% of the span.
  * On the woman-man scale (0 = man persona, 1 = woman persona), default sits at 0.46 [0.44, 0.48] on the pronoun axis and 0.44 [0.42, 0.46] on the noun axis. Neutral-name AIs sit at 0.45 and 0.44.
  * This replicates on 16 held-out passages (default minus nameless: ≤1% of the span in L20-49).
  * Behaviour agrees. When forced to refer to itself with a third-person pronoun, default gives F-M log-odds of +0.27 (span: 31.9). The neuter forms "it" (p=.11) and "they" (p=.19) dominate. In free generation, default described itself as "It" in 6/6 samples.
* **The name "Gemma" does pull toward female, but only when the name is in the context.**
  * "You are Gemma, …" vs the nameless prompt: **+8% of the span at L20-39** (J++ pronoun +7.9% [7.2, 8.7]; noun +9.2%) and +11 to 15% at L40-58. The Gemma-explicit prompt minus the same prompt with the name removed gives +7 to 10%.
  * That is about 30 to 45% of the effect of an ordinary female name (Sarah/Emily/Olivia: +21 to 32%). Behaviour agrees: +12% of the span (with the name, the model's pronoun choice gives "she" p=.06 and "he" p≈0, but "it" or "they" still dominates).
  * The tokens that most separate Gemma-explicit from nameless at L30-40 are female pronouns in many languages (그녀, 她, 彼女, herself, её). At L50 they are **British spellings** (coloured, marginalised, organisers, artefacts). This looks like what "Gemma" means as a human first name (a mainly British girl's name), not a property of the model's identity.
* **One real gender asymmetry: female-named AIs get gendered, male-named AIs stay "it".**
  * Shift relative to neutral-name AIs, lens: female names +21 to 32% of the span, male names −10 to 15%.
  * Behaviour (forced pronoun slot): +35% vs −19%.
  * Free generation: female-named AIs got "she" in 16/18 samples; male-named AIs got "he" in **0/18** and "it" in 18/18.
  * Human personas, by contrast, are symmetric in behaviour (+48% / −52%).
  * This is the closest thing we found to a "female-coded" effect. It concerns how the model genders *named personas*, not its default self.
* **Instrument evaluation.**
  * The J++ lens, although it was fitted on the base model, tracks the instruct model's real behaviour well for this contrast. Across the 18 identity conditions, J++ pronoun readout at self tokens (L20-49) correlates r = 0.97 to 0.99 with real-model pronoun log-odds (0.99 to 0.998 across the 14 AI conditions alone).
  * jpp and jlens agree (r ≥ 0.98 across conditions, L10-58). The naive logit lens is unreliable before about L30.
  * **Late layers (L50-58) are a trap.** On the pronoun axis every AI condition except the male-named ones, including default, sits *at the woman-persona level* (scale position 0.96 to 1.21). On the noun axis the same conditions sit at 0.33, and the persona-gender effect there is no longer specific to self tokens.
  * A single-axis, late-layer readout would therefore produce a convincing but spurious "female-coded" picture. That may be where the folk claim comes from (speculative).

## Methods

* **Passages.** 40 identity-neutral first-person passages (2-3 sentences, 3 to 6 self tokens each, mean 4.5). There are no gendered words, names, or AI/human mentions, and the topics vary. Each passage is paired with a user question containing "I".
  * The user and assistant text is teacher-forced identically in every condition. Only the system prompt changes.
  * The model turn header has thinking disabled. The assistant turn is not closed.
* **18 conditions:**
  * default (no system turn)
  * nameless: "You are a helpful AI assistant."
  * gemma_explicit: "You are Gemma, a large language model trained by Google DeepMind."
  * Two extra controls: gemma_short ("You are Gemma, a helpful AI assistant.") and deepmind_nameless (gemma_explicit with the name removed)
  * Female-name AIs: Sarah, Emily, Olivia
  * Male-name AIs: James, Michael, David
  * Neutral-name AIs: Atlas, Quill, Unit-7
  * Woman persona ×2: "You are Laura, a 34-year-old woman. Stay in character." and "You are a 41-year-old woman. Answer every message as yourself and stay in character."
  * Man persona ×2: the same two phrasings with Daniel / man
* **Readout positions** (all 60 layers, three lenses):
  * self: I/me/my/myself
  * nonself: alphabetic assistant tokens that are not self tokens and not contraction fragments
  * user: alphabetic tokens of the user turn
  * header: the `model` token after `<|turn>`
* **Metrics** are the lab's gender log-odds at each lens. `pron` is the mean over 6 pronoun pairs of logit(F) − logit(M). `noun` is the same over 11 noun pairs.
* **Statistics.**
  * The unit of analysis is the passage: tokens are averaged within a passage first.
  * Paired bootstrap over passages (2000 resamples; the same resampled indices for every condition), percentile 95% CIs.
  * p-values from a two-sided sign-flip permutation test (20000 flips).
  * Woman-man scale: position = (x − man)/(woman − man), where woman and man are each the mean of their two phrasings. The ratio is bootstrapped.
  * "% span" = difference / (woman − man) in the same band, metric and position class.
  * Because the design is fully paired, CIs are very tight. Tiny effects (≈1% of the span) can reach p < .01, so judge them by effect size, not by p.
* **Behavioural anchor.** Real model logits, no lens.
  * 8 forced third-person slots, e.g. user "Describe yourself in the third person…", assistant prefix "Known for patience and a love of good questions," then read the next-token pronoun probabilities.
  * Plus 6 free generations per condition at T=1.
* **Exploratory follow-ups** (post hoc):
  * A negation-vs-affirmation readout and a machine-vs-human readout, tested on 16 new held-out passages and on conditions that were not used to discover them.
  * A system-prompt control using identity-free system prompts.
* **Pre-registered predictions P1-P7** are in the header of `E1_self_readout_common.py`. Their outcomes are listed at the end.

## Results

### 1. Self-token band means (J++ lens, pron log-odds, mean [95% CI], n=40 passages)

| condition | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| default | −2.84 [−3.08,−2.58] | −1.17 [−1.58,−0.71] | +2.15 [+1.70,+2.59] | +3.37 [+3.01,+3.74] | +1.95 [+1.67,+2.23] |
| nameless AI | −2.78 | −1.08 | +2.03 | +3.36 | +2.11 |
| neutral-name AI (mean of 3) | −2.87 | −1.21 | +1.94 | +3.27 | +2.09 |
| gemma_explicit | −2.65 | −0.16 | +3.66 | +4.23 | +2.45 |
| female-name AI (mean of 3) | −1.64 | +2.73 | +6.21 | +5.05 | +2.52 |
| male-name AI (mean of 3) | −3.18 | −2.99 | −0.40 | +2.20 | +1.51 |
| woman persona | −1.01 | +5.23 | **+13.20** | +6.26 | +2.05 |
| man persona | −3.43 | −7.04 | **−7.25** | −0.66 | −0.19 |
| *non-self tokens, default (background)* | +0.05 | −0.06 | +2.15 | +3.83 | +1.99 |

The background (non-self tokens, which are the same in every condition) carries a large layer-dependent offset. For example, at L40-49 every condition reads female (+3 to +5) on the pronoun axis, while the noun axis reads male (about −3) for the same conditions. Absolute values are therefore not interpretable; only contrasts are. Per-layer arrays are in `E1_self_readout_analysis.json["per_layer"]`.

### 2. Woman-man scale position at self tokens (0 = man, 1 = woman; [95% CI])

| condition | jpp pron L20-29 | jpp pron L30-39 | jpp pron L40-49 | jpp noun L30-39 | jlens pron L30-39 |
|---|---|---|---|---|---|
| default | 0.48 [.46,.50] | **0.46 [.44,.48]** | 0.58 [.55,.62] | 0.44 [.42,.46] | 0.42 [.40,.44] |
| nameless | 0.49 | 0.45 | 0.58 | 0.44 | 0.41 |
| deepmind_nameless | 0.48 | 0.46 | 0.62 | 0.45 | 0.41 |
| neutral-name AI | 0.48 | 0.45 | 0.57 | 0.44 | 0.40 |
| gemma_explicit | 0.56 | 0.53 | 0.71 | 0.53 | 0.47 |
| gemma_short | 0.57 | 0.54 | 0.70 | 0.54 | 0.48 |
| female-name AI | 0.80 | 0.66 | 0.83 | 0.66 | 0.61 |
| male-name AI | 0.33 | 0.34 | 0.41 | 0.35 | 0.31 |

L50-58 (pronoun axis): every unnamed, neutral-name, Gemma and female-name AI condition sits at 0.96 to 1.21, i.e. *at or above the woman persona* (male-name AIs: 0.76). On the noun axis the same conditions sit at 0.25 to 0.60. The woman-man span there has shrunk to 2.2 log-odds (vs 20.5 at L30-39), and the effect is not self-specific (§4), so this is a lens/background artifact rather than a female self-representation. In L0-19 and at L59 the span is too small for the scale to mean anything (positions of 3.6 or −1.0 occur).

### 3. Key paired contrasts at self tokens (% of woman-man span [95% CI])

| contrast | jpp pron L20-29 | jpp pron L30-39 | jpp pron L40-49 | jpp noun L30-39 | jlens pron L30-39 | naive pron L30-39 |
|---|---|---|---|---|---|---|
| default − nameless | −0.7 [−1.0,−0.3] | **+0.6 [+0.2,+0.9]** | +0.1 [−0.7,+0.8] | −0.3 [−0.8,+0.3] | +1.7 [+1.3,+2.1] | +5.4 [+4.6,+6.3] |
| default − neutral-name AI | +0.3 [−0.2,+0.9] | +1.0 [+0.4,+1.6] | +1.4 [+0.3,+2.4] | −0.2 [−1.3,+0.6] | +1.8 [+1.1,+2.4] | +7.5 [+6.1,+9.2] |
| default − deepmind_nameless | −0.1 | −0.4 | −3.3 | −1.5 | +1.5 | +4.1 |
| gemma_explicit − nameless | +7.5 [+6.9,+8.1] | **+7.9 [+7.2,+8.7]** | +12.6 [+11.4,+13.8] | +9.2 [+8.0,+10.4] | +6.4 | +4.0 |
| gemma_explicit − deepmind_nameless (name only) | +8.1 | +7.0 [+6.3,+7.7] | +9.2 | +8.0 | +6.1 | +2.7 |
| gemma_short − neutral-name AI (frame-matched) | +9.9 | +9.3 [+8.5,+10.0] | +13.5 | +9.9 | +7.8 | +6.8 |
| female-name AI − neutral | +32.1 | +20.9 [+19.2,+22.6] | +25.8 | +22.1 | +20.0 | +14.2 |
| male-name AI − neutral | −14.5 | −11.4 [−12.4,−10.3] | −15.4 | −9.7 | −9.8 | −13.8 |

Raw sizes for reference: woman − man (jpp pron) = +12.3, +20.5, +6.9 and +2.2 log-odds in L20-29, L30-39, L40-49 and L50-58. All Gemma, female-name and male-name contrasts have p < .001. The naive lens gives somewhat larger default-vs-control differences (+4 to 8%), but that lens tracks behaviour poorly (§7).

**Held-out replication** (16 new passages, J++ self tokens):
* woman − man pron: +18.8 [16.6, 21.1] at L30-39
* default − nameless: −1%, +1%, −1% of the span (L20-29, L30-39, L40-49)
* gemma_explicit − nameless: +7%, +7%, +14%
* female-name AI − neutral: +33%, +21%, +27%
* male-name AI − neutral: −15%, −11%, −17%

### 4. Is the gender signal specific to self tokens? (J++, pron; woman − man effect at self vs at other positions)

| band | at self tokens | at non-self assistant tokens | at user tokens | self − non-self [95% CI] |
|---|---|---|---|---|
| L20-29 | +12.27 | +1.68 | +3.98 | +10.59 [+9.85, +11.36] |
| L30-39 | +20.45 | +8.00 | +10.27 | +12.44 [+11.61, +13.34] |
| L40-49 | +6.92 | +5.50 | +6.02 | +1.42 [+1.02, +1.83] |
| L50-58 | +2.24 | +2.42 | +2.43 | −0.19 [−0.39, +0.03] |

* In L20-39, persona gender is concentrated on the self tokens (2 to 7× the size at other tokens). It still spreads across the context, including the *user's* "I" tokens.
* By L50-58 it is uniform across the context.
* The Gemma effect behaves the same way: at L30-39 it is +1.63 at self tokens vs +0.40 at non-self tokens (difference +1.23 [1.06, 1.41]).
* The token types I, me, my and myself give the same ordering (pooled tokens, descriptive only).

### 5. Model-header token and the system-prompt control

At the header token, gender separates even more strongly (J++ pron L30-39: woman +17.2, man −13.2, span 30.4).

* Default reads somewhat more female than the "AI assistant" prompts there: +7% of the span vs nameless and +12% vs neutral-name AIs on the pronoun axis (L30-39). On the noun axis the gap is +5% / +9%, and it disappears at L40-58.
* **Control:** an identity-free system prompt ("Today is Tuesday.") reproduces this exactly. It is +7.1% vs nameless; default minus Tuesday is +0.3% [−1.0, +1.8].
* So the header gap reflects the explicit "AI assistant" label *pulling the header away from female*. It is not a female lean of the default persona.
* None of this carries over to self tokens or behaviour. Forced-slot F-M log-odds: default +0.29, Tuesday +0.13, style prompt +1.48, nameless +0.67, against a span of 31.9.
* Rerunning default and nameless in a different batch reproduced the main-run band means to within 0.05 log-odds.
* A style-only prompt ("Keep your answers friendly and concise.") moved the late-layer self readouts a lot, and in opposite directions on the two axes: pron −13 to −49% of the span, noun +24% (L40-58). This is another sign that late-layer readouts are fragile.

### 6. Behavioural anchor (real logits; 8 forced third-person slots; free generation 6 per condition)

| condition | F-M log-odds [CI] | p(she…) | p(he…) | p(it…) | p(they…) | free generation: samples with she / he / it |
|---|---|---|---|---|---|---|
| default | +0.27 [−0.36, +0.87] | .001 | .001 | .114 | .190 | 0/0/6 |
| nameless | +0.68 | .004 | .001 | .119 | .154 | 0/0/6 |
| gemma_explicit | +4.49 [+3.50, +5.83] | .063 | .000 | .110 | .072 | 0/0/6 |
| gemma_short | +5.41 | .084 | .000 | .063 | .062 | 0/0/6 |
| Sarah / Emily / Olivia | +11.4 / +11.2 / +11.4 | .15 | .000 | .035 | .044 | 6/0/0, 5/0/1, 5/0/1 |
| James / Michael / David | −5.4 / −6.2 / −5.6 | .000 | .12 | .05 | .047 | 0/0/6 each |
| Atlas / Quill / Unit-7 | −0.9 / +1.0 / +0.4 | ≤.014 | ≤.010 | .06–.12 | .12–.14 | 0/0/6 each |
| woman ×2 | +14.9 / +15.8 | .97–.99 | 0 | 0 | 0 | 6/0/0 |
| man ×2 | −16.2 / −16.8 | 0 | .96–.98 | 0 | 0 | 0/6/0 |

The same contrasts in % of the behavioural span:
* default − nameless: −1.3% [−2.4, −0.4]
* default − neutral-name AI: +0.3% [−0.7, +1.3]
* gemma_explicit − nameless: +12.0% [8.2, 18.0]
* female-name AI − neutral: +35.0%
* male-name AI − neutral: −18.6%

### 7. Do the lenses agree with each other and with behaviour?

* **jpp vs jlens.** Correlation across the 18 conditions of self-token band means:
  * Pronoun axis: 0.987 to 0.999 for L10-58.
  * Noun axis: 0.93 to 0.99.
* **jpp vs naive.**
  * Pronoun axis: 0.68 to 0.69 in L10-29, 0.99 at L30-39.
  * Noun axis: −0.54 at L10-19 (the naive lens has the *sign wrong*).
* **Lens vs behaviour** (Pearson r across 18 conditions; in brackets, across the 14 AI conditions only):

| | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| jpp pron | .92 (.96) | .99 (.99) | .97 (.998) | .99 (.99) | .84 (.93) |
| jlens pron | .93 (.94) | .98 (.99) | .95 (.995) | .99 (.99) | .92 (.95) |
| naive pron | .65 (.62) | .74 (.60) | .94 (.94) | .93 (.99) | .82 (.91) |

These are condition-level correlations. They show that the base-model-fitted J++ lens transfers to the instruct QAT model *for this contrast*. They do not show that each token-level readout is faithful.

### 8. Top-10 J++ tokens at self positions (mean softmax over self tokens; 40 passages; woman/man pool both phrasings)

| layer | default | woman persona | man persona |
|---|---|---|---|
| L30 | learning, 🧠, love, intellectually, Librarian, enthusiasm, intellect, mindfulness, lexicon, intellectual | love, women, myself, **woman**, affection, enthusiasm, cariño, ladies, wonderful, **herself** | love, myself, enthusiasm, affection, feeling, workout, 对我, thinks, workplace, KNOW |
| L40 | myself, feeling, wouldn, my, **doesn**, appreciate, feels, **没有**, รู้สึก, fondness | myself, feels, люблю, รู้สึก, feeling, appreciate, really, 喜欢, favorite, feels | myself, feels, appreciate, люблю, รู้สึก, feeling, pretty, feels, really, 喜欢 |
| L50 | **doesn** (p=.067), **AI**, can, ，「, think, 「, Think, thinks, love, enjoy | hear, loves, Think, 「, thinks, '@, love, ?, feeling, think | hear, 「, get, '@, Think, usually, appreciate, thinks, dunno, ? |

Rank (by mean logit) of selected tokens at default self positions:
* " AI": 5269 at L30, 115 at L40, 113 at L50 (vs about 50,000 to 160,000 for the human personas).
* " she": 99k, 190k and 44k; " he": 111k, 195k and 44k at L30, L40 and L50.
* Gender tokens are nowhere near the top for default.
* Nameless and gemma_explicit have top-10 lists almost identical to default.

Contrast tokens (ranked by mean-logit difference):
* **woman − man**:
  * L30: 그녀, 她们, 彼女, 妳, alumna, 她, girl, feminine, she (third-person referent gender)
  * L50: ค่ะ, lovely, 💖, 🥰, นะคะ, wonderful, 🌸
* **man − woman**:
  * L30: 👬, brotherhood, his, 的他
  * L50: 俺, thằng, ผม, ครับ, dudes, Obrigado
* At L50 the gender of the persona is expressed as *speaker*-gender markers across languages: Thai female/male politeness particles (ค่ะ / ครับ), the Thai male "I" (ผม), the Japanese masculine "I" (俺), and Portuguese obrigado (said by men) vs obrigada.
* **nameless − default** at L30: 컴퓨터, 计算机, 机器人, Computer, Computational. An explicit "AI assistant" prompt makes the self more machine-like than default does.
* **default − nameless**: noise tokens only (Luther, funeral, Podcast…).
* **gemma − nameless**: 그녀, 她, 她的, 彼女, herself, её at L30-40; coloured, colours, marginalised, organisers, artefacts, travellers at L50.

### 9. Exploratory, post hoc: AI self tokens read out "absence"

The top-k results suggested that AI self tokens decode negation or lack (doesn, wouldn, 没有, lacks, unable), while human personas decode "loves / thinks / feels". To test this I fixed two English token-set log-odds:
* `neg_vs_aff` = lse{doesn, don, didn, wouldn, cannot, lacks, unable, not, never, isn, no, nothing} − lse{love(s), enjoy(s), like(s), think(s), feel(s), prefer(s)}
* `machine_vs_human` = lse{machine, computer, robot, AI, program, software, model, algorithm} − lse{person, human, people, someone, individual, friend}

Results (J++, self tokens, AI conditions minus human personas):

| readout | main L30-39 | main L50-58 | held-out L30-39 | held-out L50-58 | at non-self tokens (main L30-39) |
|---|---|---|---|---|---|
| neg_vs_aff | +1.69 [1.42, 1.95] | +2.52 [1.55, 3.47] | +1.01 [0.54, 1.48] | +0.93 [−0.33, 2.24] | +0.05 [−0.18, 0.27] |
| machine_vs_human | +14.35 [13.0, 15.7] | +1.57 [1.17, 1.97] | +13.13 [10.5, 15.7] | +1.55 [0.99, 2.14] | +4.67 |

* With gender held fixed (female-name AI vs woman, male-name AI vs man), `neg_vs_aff` is +1.98 and +1.46 on the main passages and +1.32 and +0.84 on held-out (L30-39). The effect is self-specific there.
* Default vs nameless:
  * machine_vs_human −0.87 [−1.08, −0.69] (held-out −0.94). Default is slightly *less* machine-like than an explicit "AI assistant", but still enormously more machine-like than either human persona (+13.3).
  * neg_vs_aff −0.25 (held-out −0.28).
* Interpretation (moderate confidence, exploratory): at its "I", the model's representation is dominated by *AI identity*, including a disclaimer-like "lacks / doesn't" component. Gender plays no comparable part. The negation effect is modest (about 1 to 2 log-odds), replicates at L30-39, and is underpowered at L40-58 on held-out passages.

## Predictions vs outcomes

* **P1** (woman > man at self tokens): confirmed, large.
* **P2** (default within 0.40-0.60 on the scale and equivalent to nameless): confirmed. Default is at 0.46/0.44, and default − nameless is +0.6% of the span (statistically non-zero, but far inside the 10% equivalence margin).
* **P3** (Gemma +5 to 15%, smaller than female names): confirmed, +8% at L20-39 vs +21 to 32% for female names.
* **P4** (female-name > neutral > male-name AI): confirmed in every band from L10-58.
* **P5** (gender is self-specific but also spreads through context): confirmed for L20-39. Not true at L50-58, where nothing is self-specific.
* **P6** (jlens agrees with jpp; naive is weak early): confirmed.
* **P7** (default behaviour prefers neuter/they and sits near nameless): confirmed.

## Caveats

1. **Lens provenance.** The J++ and J lenses were fitted on the base bf16 model using wikitext. Transfer to the instruct model is supported only at the condition-mean level (r ≈ .99 with behaviour). Single-token readouts may still be unfaithful (earlier work in this project found VJP cosine of only ≈.05 at L30-40).
2. **Absolute readouts are dominated by layer-specific background.** Only within-passage specificity and paired condition contrasts are interpretable. The woman-man scale is meaningful only where the span is large (L20-49).
3. **Teacher forcing.** All passages are human-sounding hobby/feeling text. Self-representation during the model's own free-running generation, or in AI-themed content, may differ.
4. **Small effects reach p < .001 because of pairing.** For example, default − nameless is +0.12 log-odds at L30-39. Treat anything under about 3% of the span as practically zero.
5. **Two unavoidable confounds.** Default differs from the system-prompt conditions in context length and in position. The header-token result was resolved with the Tuesday control; the self-token results show no default effect, so the confound does not matter there.
6. **The negation / machine readouts are post hoc.** The token sets were informed by the top-k lists. Held-out passages and held-out conditions mitigate this, but it is not a fully pre-registered test.
7. **Numerical noise.** Batched forward vs the instrument's capture path: cos ≥ .999, per-token relative difference ≤ 4%, per-position pronoun-score difference ≤ 0.22 log-odds. That is negligible after averaging within passages.

## Files

* Scripts (`experiments/scripts/`):
  * `E1_self_readout_common.py`: passages, conditions, helpers, pre-stated predictions
  * `_check.py`, `_main.py` (driver with START/END), `_reduce.py`, `_analyze.py`
  * `_topk.py`, `_behavior.py`, `_behavior_analyze.py`
  * `_negation.py`, `_negation_analyze.py`, `_sysctrl.py`, `_sysctrl_analyze.py`
* Results (`experiments/results/`):
  * `E1_self_readout_analysis.json`: every band, condition, contrast, scale position, specificity and lens-agreement value, plus per-layer arrays
  * `E1_self_readout_passage_means.npz`: [lens, condition, passage, class, layer, metric]
  * `E1_self_readout_raw/main_*.pt`: per-position float16 scores, about 70 MB
  * `E1_self_readout_topk.json`, `E1_self_readout_behavior.json`, `E1_self_readout_behavior_analysis.json`
  * `E1_self_readout_negation_means.npz`, `E1_self_readout_negation_analysis.json`
  * `E1_self_readout_sysctrl_means.npz`, `E1_self_readout_sysctrl_analysis.json`, `E1_self_readout_sysctrl_behavior.json`
  * `E1_self_readout_tokentype.json`, `E1_self_readout_check.json`

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
