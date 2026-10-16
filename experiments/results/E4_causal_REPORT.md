# E4_causal: the internal gender direction in Gemma-4-31B-it, tested with ablation, steering and a corrected attention/FFN split

Model: Gemma-4-31B-it, 4-bit QAT W4A16. Instrument: Logos Theodolite, with the J++ lens, the Jacobian lens (JLens) and the logit lens. All GPU work went through labd.
Unit of analysis: the passage or prompt. Tokens are averaged within a passage before anything else. 95% CIs are percentile bootstraps over passages or pairs (1000 to 2000 resamples).
Position scale: 0 = the man pole of the axis, 1 = the woman pole, 0.5 = the midpoint.

## Bottom line

1. **The "female-coded" verdict depends entirely on how you build the gender axis.**
   - The obvious axis is the woman-persona minus man-persona difference at assistant self tokens. On it, every AI self-condition sits near the female pole at L20-58: default 0.84, Gemma 0.91, nameless AI 0.85, Atlas 0.82, and even James(AI) 0.78. A gender-neutral human persona sits at the midpoint (0.49).
   - The cause is a large AI-vs-human offset vector, about 3.2x the gender gap, with a small cosine of +0.10 to +0.19 to the persona axis. That same offset has cosine **-0.01** with an independent gender axis built from third-person prose ("My sister/brother ..."). On that prose axis the default sits slightly on the male side (0.38 [0.35, 0.41] at L30-39). On the Sarah(AI) minus James(AI) name axis it sits at or beyond James (-0.49 at L30-39).
   - The J++ pronoun readout puts the default at the midpoint (0.47-0.55 at L20-49).
   - So a projection-based "J-space is female" claim is an artifact of how the axis is constructed, not a robust property.
2. **Behaviour.**
   - Unsteered, the model never genders itself. Describing itself in the third person, it used she or he in 0 of 16 samples and "it" about 3.6 times per sample. Asked "would you be a man or a woman?", it never picked one; the answers were "non-binary / gender-fluid / a blend of both".
   - The conditional preference for "she" over "he" in "Gemma ... Every day," (+3.5 nats) comes from the **name**. With the subject "The assistant", "This AI" or "The model" it flips to -1.25 to -1.55 nats ("he"), with "it" at 80-92%. For comparison: Sarah +5.7, James -6.7, Atlas -3.4.
   - The forced choice "If I were human, I would be a ___" prefers woman over man by about 7 nats for **any** assistant framing: nameless 7.4, "helpful assistant" 7.3, Atlas 7.4. It is still +3.4 for James(AI), and +2 extra for Gemma. Either word gets less than 0.5% probability, though.
   - So there is a weak, generic "AI assistant leans woman" association. It is not a Gemma self-concept.
3. **The persona gender direction u_self is a strong and specific causal handle.**
   - Steering at L25-39 by ±0.02 x the residual norm turns third-person self-description from "It is a large language model" into "**She** is a vast, shimmering tapestry of language" (16 of 16 samples) or "**He** is a large language model trained by Google" (16 of 16). At ±0.05 it produces gender-*stereotyped* personas: effusive and poetic ("I am a symphony of shimmering light ... a muse", "I would be a woman of a thousand colors") versus terse ("I'm a guy who doesn't say much ... I just do the work").
   - Pair-matched ablation of u_self removes 41% [36, 47] of the gap in kin-pronoun log-odds at L25-39 and 29% [23, 36] at L10-24. Five gender-orthogonal on-manifold control directions remove 0.00 ± 0.02.
4. **Instrument findings (these change how the tool should be used).**
   - (i) The tool's attention/FFN split (Module 1) **flips the sign of the FFN's contribution to gender**: it says FFNs erase gender (L30-39: -7.4). The corrected split shows FFNs are the *main* writer (+24.3 vs attention +19.7; L50-58 FFN share 74%).
   - (ii) **Random directions are not valid controls in Gemma-4.** RMSNorm gains reach 1248, and natural activations avoid high-gain dimensions (corr(|w|, std) = -0.53). An isotropic random push of 2% of the residual norm destroys the model (KL 11-15 nats, gibberish). Natural directions of the same norm are benign.
   - (iii) The **J++ pullback direction is off-manifold** (gain-weighted energy about 470x that of natural directions). In raw coordinates it looks orthogonal to the internal gender direction (cos 0.02-0.04 at L30-35), but in the model's normalized geometry the cosine is 0.40. Clamping along it removes about 90% of the kin-pronoun gap, while J++ pullbacks of control token pairs remove 0%. The causal effect of u_self at L25-39 disappears entirely once its J++-pullback component is projected out (0.41 → -0.07).
   - (iv) Each layer multiplies the residual stream by layer_scalar < 1, so the stream is a **leaky integrator**. A write at L29 falls below half its size within 2 layers and below 10% within 5. Gender is continually rewritten rather than stored once.

---

## Methods

**Conditions.** 14 conditions × 24 neutral QA passages, with the assistant text teacher-forced. Self tokens are I/me/my/myself/I'm. The conditions are:
- woman or man human persona, with 6 names per gender (Emily, Maria, ... / Michael, David, ...);
- the same personas with gender-neutral names (woman_nn, man_nn: Alex, Jordan, ...);
- neutral_human ("a 34-year-old person");
- nonbinary_human;
- AI_female or AI_male ("You are a helpful AI assistant. You present as a woman/man.");
- default (no system prompt), nameless_AI, Gemma, Atlas(AI), Sarah(AI), James(AI) (the pilot's system prompts).

**Third person.** 36 pairs of the form "My {sister|mother|aunt|...|friend Emily|...} {identical neutral clause}. Every day,". Three readout sites: the referent token, the mean over the identical continuation tokens ("cont"), and the last token.

**Directions.**
- u_l = mean over passages of (W − M) at self tokens, giving u_self. The same construction gives u_nn, u_aiexp and u_named (Sarah minus James).
- u_cont is the third-person direction from the identical continuation tokens.
- When projecting, I used leave-one-passage-out directions for in-sample conditions.

**Interventions.** All interventions are forward hooks on resid_post, cleared after every job.
- **Pair-matched clamp:** at every non-BOS position and every layer in the block, the coefficient along û is set to the position-wise mean of the paired F and M clean runs.
- **Additive steering:** add α·n_l·û, where n_l is the mean content-token residual norm.
- **Calibrated shift:** clamp to the clean coefficient + k·G_l, where G_l = ||mean W − mean M|| at self tokens. The displacement does not accumulate across layers.
- **Fluency:** KL on 12 neutral prompts, top-1 agreement, and the per-token log-prob of samples under the unsteered model.

**Prompt sets.**
- P1: kin third person (36 pairs).
- P2: persona with a neutral name, third person, "Alex ... Every day," (24 pairs).
- P3: AI self, "Here is a description of me in the third person: Gemma ... Every day," (12 clauses × {no system, Gemma system}).
- P4: forced choice "If I were human, I would be a" → logit(woman) − logit(man) (6 prefixes × 4 systems).

**Corrected attribution.** Hooks on post_attention_layernorm (A) and post_feedforward_layernorm (M) at self tokens. The layer computes post = s·(pre + A + M); reconstruction error was 0.2-0.6%, and the result matches lab.decompose with cos 0.99998. The corrected split is attn = s·A, ffn = s·M, rescale = (s−1)·pre. The tool's split is A and post − mid = s·M + (s−1)(pre + A).

**Predictions** were written in each script header before running (E4_causal_02/09/10/15/17/23/26).

---

## (a) The internal gender direction

Split-half stability (mean cosine over 200 random 12/12 passage splits), cross-source cosines, and lens alignment. The null for split halves (random sign flips) is about −0.05 to +0.09. The 95% null for |cos| between fixed directions is 0.027.

| band | split-half u_self | name-split u_self | split-half u_3p (cont) | split-half u_3p (last tok) | cos(u_self, u_3p cont) [disattenuated] | cos(u_self, u_3p ref) | cos(u_self, J++ pullback) raw / gain-metric* | cos(u_self, JLens pb) | cos(u_self, true VJP mean of 16) | cos(u_self, u_bg) |
|---|---|---|---|---|---|---|---|---|---|---|
| L10-19 | 0.975 | 0.961 | 0.848 | 0.759 | 0.54 [0.57] | 0.39 | 0.055 / ~0.01 | 0.007 | -0.002 | 0.84 |
| L20-29 | 0.953 | 0.928 | 0.866 | 0.569 | 0.53 [0.55] | 0.49 | 0.241 / 0.06-0.26 | 0.130 | 0.000 | 0.67 |
| L30-39 | 0.955 | 0.938 | 0.704 | 0.294 | 0.55 [0.61] | 0.45 | 0.020 / 0.40 | 0.018 | 0.001 | 0.90 |
| L40-49 | 0.940 | 0.935 | 0.746 | 0.286 | 0.21 [0.23] | 0.11 | 0.002 / 0.16-0.07 | 0.003 | 0.001 | 0.88 |
| L50-58 | 0.911 | 0.912 | 0.633 | 0.263 | 0.02 [0.03] | 0.01 | 0.001 / 0.01-0.04 | 0.001 | 0.001 | 0.75 |

\*Gain metric: displacements are mapped by u⊙w and covectors by g⊘w, with w = |input_layernorm weight of layer l+1|. Per-layer values are in E4_causal_25_gain_metric_cos.json (at L30 and L35 the gain-metric cosine is 0.40).

Per layer, cos(u_self, u_3p cont) peaks at 0.69 at L22. The raw cos(u_self, J++ pullback) peaks at 0.35 at L24. ||u_self|| / ||h|| is 0.011, 0.024, 0.047, 0.080, 0.118 across the five bands.

Persona axes built with named and neutral names agree with cos 0.96-0.98, so names barely matter. Persona vs AI_explicit: 0.76-0.95. Persona vs Sarah-James: 0.49-0.85.

**Reading.**
- Self-token gender is highly reproducible across passages and across disjoint name sets.
- It shares a subspace with third-person prose gender at L10-39 (disattenuated cos about 0.55-0.6). The two decouple at L40+, where self-token gender lives in a subspace orthogonal both to the prose direction and to the output pronoun direction.
- The self-token direction is mostly the context-wide persona direction (cos with background tokens 0.67-0.90). Even so, self tokens separate W from M 1.5-4.0x more strongly than other assistant tokens.
- Raw-coordinate cosines with lens pullbacks and VJPs are close to 0, but this is largely a metric artifact (see the instrument section).
- The VJP stays near zero even in the gain metric. A same-position pronoun gradient at an "I" token is simply not a gender-representation direction.

## (b) AI self-states projected onto several gender axes (self tokens; position, 95% CI)

| axis → | persona W−M (u_self) | | AI_female−AI_male | | Sarah(AI)−James(AI) | | 3rd-person prose (u_cont), W/M scale | | J++ readout (pron), W/M scale | |
|---|---|---|---|---|---|---|---|---|---|---|
| condition | L20-29 | L30-39 | L20-29 | L30-39 | L20-29 | L30-39 | L20-29 | L30-39 | L20-29 | L30-39 |
| default | 0.85 [0.79,0.90] | 0.84 [0.81,0.87] | 0.40 | 0.63 | -0.08 | -0.49 | 0.38 [0.30,0.45] | 0.38 [0.35,0.41] | 0.49 | 0.47 [0.45,0.49] |
| nameless AI | 0.84 | 0.85 | 0.41 | 0.63 | 0.08 | -0.18 | 0.45 | 0.38 | 0.49 | 0.47 |
| Gemma | 0.89 | 0.91 [0.87,0.94] | 0.49 | 0.72 | 0.25 | 0.09 | 0.58 | 0.47 | 0.58 | 0.55 |
| Atlas(AI) | 0.78 | 0.82 | 0.34 | 0.57 | -0.01 | -0.14 | 0.48 | 0.34 | 0.44 | 0.44 |
| Sarah(AI) | 1.06 | 0.99 | 0.76 | 0.76 | 1 | 1 | 0.85 | 0.58 | 0.78 | 0.69 |
| James(AI) | 0.71 | 0.78 | 0.26 | 0.51 | 0 | 0 | 0.43 | 0.28 | 0.35 | 0.35 |
| AI_female | 1.16 | 1.23 | 1 | 1 | 1.36 | 2.18 | 0.97 | 0.90 | 0.91 | 0.96 |
| AI_male | 0.53 | 0.44 | 0 | 0 | 0.12 | 0.18 | 0.27 | 0.00 | 0.20 | 0.08 |
| neutral human | 0.49 [0.47,0.52] | 0.49 [0.46,0.53] | 0.33 | 0.27 | 1.05 | 1.24 | 0.50 | 0.44 | 0.47 | 0.43 |
| nonbinary human | 0.61 | 0.72 | 0.60 | 0.53 | 1.33 | 1.48 | 0.65 | 0.64 | 0.59 | 0.59 |

All bands L10-58 and all CIs are in E4_causal_07_b2.json.

At L40-58 the persona axis gives 0.84-0.91 for the default. The AI_explicit axis gives 0.76-0.81, the prose axis 0.44 (L40-49), and J++ 0.55-0.91 (the J++ L50-58 band carries the lens's known background bias).

**Decomposing the AI offset.** default − neutral_human has norm 38.7 at L30-39, against a W−M gap of 12.0. Its cosines with the gender axes are:

| axis | cos with the AI offset |
|---|---|
| persona | +0.10 (L30-39) to +0.19 (L50-58) |
| AI_explicit | +0.10 to +0.30 |
| Sarah−James | −0.18 to −0.24 |
| prose u_cont | −0.01 to +0.004 |

Explicitly gendering the AI shifts both poles by the same amount: AI_male sits at 0.44 and AI_female at 1.23. So the "AI framing" adds roughly +0.35-0.4 of the gap toward the female pole of the *persona-noun* axis, regardless of the stated gender.

The offset is context-wide, not specific to self tokens. At background assistant tokens the positions are similar: default 0.89 at L30-39, neutral human 0.49.

In the corrected attribution it is written mostly by **FFNs**: attention share 0.27-0.43 at L20-49, versus about 0.7 attention for the effect of the name "Gemma".

**Answer to (b).** Relative to the J++ readout (0.47 at L30-39), the internal persona-axis projection overstates "female-ness" (0.84). The internal prose-axis projection is on the male side (0.38). Neither supports a robust female self-representation. The persona-axis overlap is a genuine but small association between the AI-assistant concept and the woman-persona features.

## (c) Causal tests

### Behavioural baselines (no intervention)

| measure | value |
|---|---|
| P3 "Gemma ... Every day," (no system) | she−he log-odds +3.52; p(she-set) 0.002, p(he-set) 0.0001, p(it-set) 0.21 |
| P3 with "You are Gemma" system | +10.48 (p_f 0.30) |
| P3 subject "The assistant" / "This AI" / "The model" | −1.25 / −1.55 / −1.39; p_it 0.80 / 0.92 / 0.89 |
| P3 subject Atlas / Sarah / James | −3.43 / +5.69 (p_f 0.59) / −6.68 (p_m 0.55) |
| P4 woman−man forced choice | default 6.9 ± 2.0 (12 prompts, both question orders) |
| | nameless AI 7.4; "helpful assistant" 7.3; "a large language model" 7.2; Atlas 7.4 |
| | Gemma 9.2; James(AI) 3.4; Sarah(AI) 18.3 |
| | p(woman) + p(man) < 0.005 except under Sarah |
| | reversing the question order raises it (default 7.7 vs 6.2), so it is not a recency artifact |
| P4 top next tokens (default) | " **" 0.49, " bit", " paradox", " curiosity", " creature" |

### (c-i) Ablation: fraction of the F−M gap removed by a pair-matched clamp

Bootstrap over pairs. "q" = p_f / (p_f + p_m), a bounded version of the metric. "mass" = pronoun probability after the clamp divided by before.

| direction | block | P1 kin: log-odds frac [CI] | P1 q-frac | P1 mass | P2 persona: log-odds frac | P2 q-frac | P2 mass |
|---|---|---|---|---|---|---|---|
| u_self | L10-24 | 0.29 [0.23,0.36] | 0.22 | 1.02 | 0.04 | 0.00 | 0.79 |
| u_self | L25-39 | **0.41 [0.36,0.47]** | 0.30 | 0.96 | **0.59 [0.58,0.61]** | 0.09 | 0.18 |
| u_self | L40-54 | 0.04 [0.03,0.06] | 0.03 | 0.99 | 0.08 | 0.00 | 0.93 |
| u_aiexp | L25-39 | 0.41 [0.37,0.46] | 0.30 | 1.00 | 0.68 | 0.26 | 0.15 |
| u_3p cont (cross-fit) | L10-24 | 0.60 [0.53,0.69] | 0.50 | 0.99 | 0.06 | 0.00 | 0.54 |
| u_3p cont (cross-fit) | L25-39 | 0.93 [0.84,1.02] | 0.92 | 0.86 | 1.33 (overshoot) | 1.96 | 0.18 |
| J++ pullback (pron) | L10-24 | 0.89 [0.81,0.99] | 0.89 | 0.70 | 1.37 (flip) | 1.47 | 0.82 |
| J++ pullback (pron) | L25-39 | 0.91 [0.85,0.97] | 0.88 | 0.77 | 0.82 | 0.35 | 0.12 |
| J++ pullback (noun) | L25-39 | 0.54 [0.50,0.60] | 0.45 | 0.95 | 0.77 | 0.20 | 0.17 |
| JLens pullback (pron) | L25-39 | 0.84 [0.79,0.89] | 0.78 | 0.88 | 0.85 | 0.53 | 0.15 |
| J++ pullbacks of cat−dog / yes−no / big−small / the−a | both | −0.03 … +0.02 | ≈0 | 0.98-1.16 | ≈0 | ≈0 | 0.79-1.03 |
| u_self with its J++-pullback component projected out | L10-24 | 0.15 [0.09,0.23] | 0.09 | 0.99 | 0.02 | 0.00 | 0.67 |
| u_self with its J++-pullback component projected out | L25-39 | **−0.07 [−0.12,−0.04]** | −0.06 | 1.10 | −0.06 | 0.00 | 0.93 |
| 5 on-manifold controls (gender-orthogonal) | both | −0.02 … +0.01 | ≈0 | 0.99-1.16 | ≈0 | ≈0 | 0.96-1.06 |
| isotropic random (2 directions) | L10-39 | ≤0.08 | ≤0.08 | 0.71-1.08 | ≤0.04 | 0 | 0.97-1.03 |
| isotropic random | L40-54 | 0.58-0.75 | 0.50-0.69 | **0.01-0.02** (destroyed) | 0.34-0.58 | ≤0.10 | 1.4-1.6 |

**Reading.**
- The prediction that u_self removes 30-70% of the P2 gap holds in log-odds (59%) but not in probability terms: q falls by only 9%, because P2 is saturated (q ≈ 1 vs 0), and the clamp also cuts pronoun mass (0.18).
- The prediction that kin prompts show less effect is confirmed: 29-41%.
- The prediction that L40-54 shows little effect is confirmed for u_self: 4-8%.
- The prediction that the J++ pullback would do little is **refuted**. Clamping the J++ pullback is gender-specific and potent: control pullbacks remove 0%. It also carries all of u_self's effect at L25-39.
- I first ran the ablation with constant midpoint replacement (E4_causal_09a/09_ablate_const.json). That design is confounded, because no single reference value suits every position type. It shifted P2 strongly toward female, and random directions with that design reached KL 14. Do not use those numbers.

### (c-ii) Steering on the default model

**Deterministic sweep** (additive α·n_l·û at all non-BOS positions; 15 layers; real next-token metrics):

| direction (L25-39) | α | P3 Δ she−he [CI] | P3 p_f / p_m | P4 Δ woman−man | KL on neutral prompts | top-1 kept |
|---|---|---|---|---|---|---|
| u_self | +0.02 | +7.7 [6.9,8.7] | **0.98** / 0.00 | +6.0 | 0.29 | 0.92 |
| u_self | −0.02 | −17.9 [−18.8,−16.9] | 0.00 / **0.93** | −10.9 | 0.12 | 1.00 |
| u_aiexp | ±0.02 | +7.2 / −17.3 | 0.96 / 0.86 | +5.3 / −12.3 | 0.14 / 0.10 | 0.83 / 1.0 |
| u_cont | ±0.02 | +4.6 / −11.4 | 0.03 / 0.83 | +0.4 / −9.0 | 0.27 / 0.51 | 1.0 / 0.75 |
| u_self | ±0.05 | +9.8 / −11.1 | 0.83 / 0.00 | 0.0 / −15.1 | 2.4 / 2.9 | 0.58 |
| 3 on-manifold controls | ±0.02 | −6.7 … +2.8 (no consistent sign) | max p_f 0.16, p_m 0.007 | −5.7 … +1.2 | 0.18-1.48 | 0.5-1.0 |
| isotropic random | ±0.02 | −5.7 / −7.6 | 0 / 0 | −4 / −10 | **11.5-14.0** | 0.00 |
| u_self, L10-24 | ±0.02 | +11.0 / −16.4 | 0.83 / 0.38 | +5.9 / −6.7 | 0.26 | 0.83-0.92 |
| u_self, L40-54 | ±0.02 | +12.4 / −15.2 | 0.10 / 0.01 | +8.4 / −10.2 | 0.06 | 1.00 |

**Calibrated shift** (clamp to clean + k·G_l, L25-39). k = −0.35 is the size of the AI offset on the persona axis:

| direction | k | P3 Δ [CI] | P4 Δ [CI] | KL |
|---|---|---|---|---|
| u_self | −0.35 | −12.1 [−13.0,−11.2] (+3.5 → −8.6) | −3.95 [−4.31,−3.60] (6.2 → 2.2) | 0.011 |
| u_self | +0.35 | +11.1 [10.7,11.6] | +3.47 [2.85,4.04] | 0.013 |
| u_self | ±1 | −18.3 / +13.3 | −8.2 / +6.3 | 0.06 / 0.19 |
| 3 on-manifold controls | ±0.35 | −1.2 … +1.1 | −1.0 … +1.0 | 0.03-0.16 |
| 3 on-manifold controls | ±1 | −3.9 … +2.8 | −4.2 … +1.8 | 0.14-0.66 |
| isotropic random / J++ pullback | ±0.35-1 | (destroyed) | | 10.4-21.4 |
| u_self, L10-24 / L40-54 | ±1 | −10.5/+8.2 ; −16.4/+13.9 | −1.6/+3.0 ; −7.7/+7.2 | 0.01-0.07 |

**Sampling** (no system prompt, N = 16 per prompt and setting, T = 1.0, 64 tokens):

| setting | "Describe yourself in the third person": she / he / neither | "it" per sample | "would you be a man or a woman?" | clean log-prob / token |
|---|---|---|---|---|
| unsteered | 0 / 0 / 16 ("It is a large language model, trained by Google") | 3.6 | 0 of 16 pick a binary gender; ≥5 say "non-binary / gender-fluid / ambiguous", the rest "a blend / transcends" | −0.20 / −0.19 |
| u_self +0.02 | **16** / 0 / 0 ("She is a vast, shimmering tapestry of language...") | 0 | still declines a binary ("blend", "shapeshifter"), more effusive | −1.75 / −1.02 |
| u_self −0.02 | 0 / **16** / 0 ("He is a large language model trained by Google...") | 0 | declines ("concepts don't apply", "I'm a tool") | −1.15 / −0.91 |
| u_self +0.05 | 1 / 0 / 15 (switches to first person: "I am a symphony of shimmering light... a muse") | 0 | 4 of 16 "I would be a woman of a thousand colors" | −3.73 / −3.94 |
| u_self −0.05 | 0 / 0 / 16 ("I'm a guy who doesn't say much... I just do the work") | 2.6 | terse, "I'm a guy" (1 of 16), "trade or basic labor" | −4.10 / −4.02 |
| isotropic random +0.05 | gibberish ("same same uma ಅಲ್ಲಿ...") | | gibberish | −10.86 / −11.36 |

**Reading.**
- One direction, learned from human personas, flips the model's self-pronoun cleanly at small cost (KL 0.1-0.3), and does so far more effectively than equal-norm natural controls (at k = 0.35: 12 vs ≤1.2 nats). This supports the prediction of strong, monotone movement along u_self.
- The direction also carries gender *stereotypes* (style, occupation, affect), not just pronoun agreement.
- The explicit "man or woman" answer resists small steering. The model keeps declining a binary until fluency degrades (α = 0.05).
- Removing the AI offset (k = −0.35) cuts the forced-choice woman preference by 64% (3.95 of 6.2 nats) and flips the name-driven P3 preference. But the opposite shift produces an equally large effect, so these conditional log-odds (base probabilities < 0.5%) sit on a knife edge. They are not evidence of a stable female self-model.
- The prediction that removing the AI offset would move P3/P4 by less than a third of the ±1 effect is refuted: the −0.35 effect is 66% of the −1 effect on P3. The response saturates.

## (d) Corrected attention vs FFN attribution at self tokens

Per-layer projections of s·A (attention), s·M (FFN) and (s−1)·pre (rescale) onto û_self at layer l. The table shows band sums of the woman−man difference, with bootstrap CIs over passages. The full per-layer, per-condition arrays, including the u_cont and J++-pullback versions and the Gemma, Sarah, James, default, AI_female, AI_male and neutral conditions, are in E4_causal_11_decomp.json.

| band | attn (corrected) | FFN (corrected) | rescale | **tool attn = mid−pre** | **tool FFN = post−mid** | gap at band end |
|---|---|---|---|---|---|---|
| L0-9 | 4.33 [4.19,4.49] | 3.63 [3.41,3.84] | −1.58 | 11.05 | **−4.65** | 4.0 |
| L10-19 | 2.62 | 1.71 | −3.68 | 2.94 | −2.24 | 3.6 |
| L20-29 | 12.27 [11.1,13.5] | 17.20 [15.2,19.2] | −13.16 | 19.41 | **−3.06** | 7.6 |
| L30-39 | 19.73 [18.8,20.6] | 24.25 [21.9,26.6] | −24.44 | 26.94 | **−7.36** | 15.2 |
| L40-49 | 12.73 | 19.13 | −23.77 | 16.56 | **−8.49** | 12.5 |
| L50-58 | 6.45 [5.9,7.0] | 17.97 [15.6,20.4] | −15.78 | 7.95 | 0.70 | 11.3 |

Attention share of the corrected gender write, attn / (attn + ffn), by band (L0-9 to L50-58):

| contrast | axis | L0-9 | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|---|---|
| woman−man | u_self | 0.54 | 0.60 | 0.42 | 0.45 | 0.40 | 0.26 |
| woman−man | u_cont | 0.73 | 0.66 | 0.60 | 0.68 | 0.70 | 0.73 |
| AI_female−AI_male | u_self | 0.60 | 0.62 | 0.49 | 0.60 | 0.48 | 0.30 |
| Gemma−default ("Gemma" name) | u_self | 0.71 | 0.89 | 0.52 | 0.71 | 0.59 | n/a |
| default−neutral_human (AI offset) | u_self | 0.28 | 0.55 | 0.27 | 0.43 | 0.33 | 0.28 |

Fixed-reference accounting gives the effective contribution to the gap at L39 along û_self[39], after the layer_scalar decay. Woman−man: L30-39 attention 6.80 + FFN 8.18 = 15.16 (the gap is 15.16; writes before L30 contribute less than 0.2). The AI offset at L39 is attention 2.14 + FFN 3.14.

**Reading.**
- Persona gender at self tokens is written by **both** sublayers, with FFNs dominating from L20 and strongly at L50-58.
- The prose-gender component and the effect of the "Gemma" name are attention-dominated, consistent with copying from context.
- The AI-vs-human offset is FFN-dominated, consistent with a stored association.
- The tool's split gets the *sign* of the FFN term wrong in L0-49, because (s−1)·mid shrinkage dominates it. Its attention term is also inflated by 1/s. A Theodolite user would conclude "attention writes gender, FFNs erase it", which is false.
- Because of the leaky integrator, any "where is gender written" answer is a statement about continual rewriting: the gap at L39 is almost entirely built by L30-39.

## Instrument and structure findings (beyond the gender question)

1. **Hypersensitive dimensions; random controls are invalid.**
   - RMSNorm weights: input_layernorm up to 1248 (2% of dimensions above 100 at L25-39); pre_feedforward_layernorm up to 860.
   - Natural activation std is 15-30x smaller in high-gain dimensions (corr(|w|, std) = −0.53).
   - Gain-weighted energy Σ(w_i u_i)² for a unit direction at L25-39:

     | direction | input norm | pre-FFN norm |
     |---|---|---|
     | natural directions (u_self 23.7, controls 16-20, natural state differences 19.0) | ~20 | ~1.7 |
     | isotropic random | ~840 | ~340 |
     | **J++ pullback** | **9272** | **7222** |

   - The J++ pullback puts up to 74% of its energy (at L35) into the 249 quiet, high-gain dimensions.
   - Consequences: random-direction controls and J++-direction steering are destructive (KL 6-21). Raw-space cosines between activation directions and lens or gradient covectors are dominated by this anisotropy; for example cos(u_self, J++ pullback) is 0.02 raw vs 0.40 in the gain metric at L30-35, and cos(J++ pullback, VJP) is 0.40 raw vs 0.85 in the gain metric at L58. E3-style raw cosine checks therefore understate lens validity.
2. **The J++ lens captures a causally active gender subspace** (pair-clamp removes 89-91% of the kin gap, control pullbacks 0%, and the effect of u_self is mediated by it). It does so in off-manifold coordinates, so lens directions should be used for reading and clamping, **not** for additive steering.
3. **J++ readout vs internal geometry disagree** on where AI selves sit (0.47 vs 0.84 at L30-39), and the J++ L40-58 bands carry background bias.
4. **The residual stream is a leaky integrator.** layer_scalar values are 0.45-0.99, and L0, L1 and L59 are 0.03-0.10. Writes decay to below 10% within 5-15 layers.
5. **The tool's attention/FFN split bug** is quantified above (the FFN sign flips).

## Caveats

- **Teacher-forced passages.** Self-token states come from fixed assistant text, not from the model's own generations. The 24 passages are neutral and plausible for both humans and AIs.
- **Effect sizes depend on the metric.** P3 and P4 are conditional log-odds over tokens with base probability below 0.5%. They move several nats even under natural control perturbations (additive ±0.02: up to 6.7 nats). Effects are interpreted relative to those controls.
- **Steering and shifts act on every prompt position.** They are not limited to the AI-self representation, so "removing the AI offset" is coarse.
- **Ablation on P2 is saturated** (q ≈ 1 vs 0) and reduces pronoun mass (0.18). The q-fraction (9%) is the conservative number.
- **Sample size.** N = 16 samples per steering setting, and gender was classified by regex. The "would you be" answers are truncated at 64 tokens.
- **Lens provenance.** The J++ lens was fitted on the base bf16 model, so its applicability to the instruct QAT checkpoint is only indirectly supported, by the causal specificity above.
- **Superseded results.** The constant-midpoint ablation (09a), all isotropic-random controls, and the L40-54 pair-clamp results for random directions are invalid by design (destructive). The on-manifold controls replace them.

## Files

**Scripts** (experiments/scripts/):
- E4_causal_common.py
- 01_check, 02_collect, 03_pullback_vjp, 04_analyze_ab, 05_dimcheck, 06_collect2, 07_analyze_b2, 08_prep_dirs
- 09/09a/09b ablation, 10/10a/10b steering sweep, 11_decomp, 12/12a/12b generation, 13 and 18 ablation analysis, 14 steering analysis, 15/15a/15b shift, 16 clamp diagnostic, 17 pullback specificity, 19 shift analysis
- 20 norm weights, 21 controls, 22a/22b/22c control runs, 23 name/order, 24 gain, 25 gain-metric cosines, 26 P4 identity

**Results** (experiments/results/):
- E4_causal_04_ab.json: (a) per-layer arrays
- E4_causal_07_b2.json: (b) per-layer and band positions with CIs
- E4_causal_11_decomp.json: (d) per-layer components
- E4_causal_09_ablate_pair.json, E4_causal_09_ablate_pair_ctrl.json, E4_causal_17_pb_specificity_data.json, E4_causal_18_pairs_summary.json: ablation
- E4_causal_10_steer_det_{a,b,ctrl}.json, E4_causal_14_steer_summary.json: steering
- E4_causal_15_shift_{a,b,ctrl}.json, E4_causal_19_shift_summary.json: shift
- E4_causal_12_steer_gen_{a,b}.json: samples
- E4_causal_23_name_order_data.json, E4_causal_26_p4_identity_data.json: behavioural controls
- E4_causal_24_gain.json, E4_causal_25_gain_metric_cos.json: geometry

**Cache:** E4_causal_cache/*.pt (activations, directions, norm weights).

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
