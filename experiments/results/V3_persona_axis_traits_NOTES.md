# V3_persona_axis_traits: why the woman-man PERSONA axis puts every AI on the female side

Adversarial verifier for C4. The target finding is E4's: default 0.84 on the persona axis, versus 0.38 on the prose axis.

Model: Gemma-4-31B-it, W4A16. All GPU work went through labd.

## Verdict

**1. The persona axis is heavily trait-loaded. Confirmed.**

Gender-free trait personas ("You are Alex, a warm, nurturing, empathetic person" vs "... a blunt, stoic, unsentimental person", 4 phrasings) span **more than the whole woman-man gap** on the persona axis:

| band | warm persona | blunt persona |
|---|---|---|
| L20-29 | 0.85 | -0.02 |
| L30-39 | 1.09 | -0.11 |
| L40-50 | 1.34 | -0.11 |

Scale: man = 0, woman = 1.

- Cross-fitted cos(persona, trait) is 0.32 / 0.33 / 0.42 for L20-29 / L30-39 / L40-50.
- Service-role personas also load on it: "personal assistant / customer support" 0.76-0.85, vs "works alone" 0.31-0.46.
- The model really does infer gender from traits. With a neutral name, the she-vs-he log-odds after "{Name} ... Every day," are +4.4 [3.6, 5.4] nats higher for warm than for blunt personas.
- The inference is not specific to warmth: organized vs disorganized gives +2.6, and serve vs solo +1.2. For comparison, an explicit woman vs man persona gives +23.7.
- Free generations still use "they" (31/32).
- Warm and blunt personas also separate on the J++ readout (0.71 vs 0.42 at L30-39) and on the prose axis (0.82 vs 0.23).

**2. The AI-vs-human offset aligns with the trait part only at L20-29. Mostly refuted at L30-50.**

| | L20-29 | L30-39 | L40-50 |
|---|---|---|---|
| cos(AI offset, trait axis) | +0.22 | +0.02 | -0.05 |
| Default on the trait axis (blunt 0, warm 1) | 1.05 | 0.65 | 0.62 |
| Neutral human on the trait axis | 0.71 | 0.64 | 0.68 |
| Default on persona ⊥ trait | 0.70 | 0.95 | 1.14 |
| Default on persona ⊥ 4-phrasing trait subspace | 0.63 | 1.07 | 1.14 |

- At L30-50 the default AI is no warmer than a neutral human.
- Projecting the trait axis out of the persona axis moves the default toward 0.5 only at L20-29. At L30-50 it moves *away* from 0.5.
- A "service role" axis does better than traits. cos(role, AI offset) is 0.52 / 0.37 / 0.23. Projecting the 2-phrasing role subspace out moves the default to 0.36 / 0.64 / 0.74. The null (projecting out other persona contrasts, k = 2) gives 0.89 / 0.86 / 0.87.

**3. What actually explains the 0.84-0.93.** The persona-axis position of the default is a knife-edge projection.

- The AI offset is 2.8-3.9x the woman-man gap, so a cosine of 0.10 is enough to shift the position by 0.35.
- The whole overlap lives in the top ~30 token-covariance PCs. These carry 33-51% of token variance and only 11-33% of the persona axis's energy. But they carry **86-115% of ⟨AI offset, persona axis⟩** and 84-96% of ⟨trait, persona⟩ (L30-50).
- Removing those 30 PCs leaves W and M fully separated. Default minus neutral human becomes -0.05 [-0.08, -0.02] at L30, -0.07 at L35, -0.09 at L40, 0.00 at L45 and +0.11 [0.03, 0.17] at L50. The warm-minus-blunt span collapses from 0.75-1.5 to 0.03-0.29.
- Other readouts place the default at or near the human midpoint:

| readout | default | neutral human |
|---|---|---|
| Full token-covariance whitening (cross-fitted, any λ) | 0.46-0.57 | 0.39-0.46 |
| Nonlinear J++ pronoun readout | 0.46-0.48 at L20-39 | 0.43 |
| J-space persona axis | 0.40 / 0.56 at L20-29 / L30-39 | 0.39 |

- At L40-50 the J-space persona axis still gives 0.94.
- Per-dimension fixes do **not** remove the effect. Diagonal whitening, dropping the 10% quietest dimensions, dropping the top 1% variance dimensions, and the gain metric all leave the default at 0.65-0.99. The cause is correlated, high-variance "persona/context style" directions, not the quiet-dimension anisotropy from C7.

**4. Behaviour. Trait words drive the forced choice; the gender-orthogonal part of the trait direction does not.**

The trait words in the AI's system prompt control the "woman" answer and the self-pronoun. The hard forced-choice "man or woman" question uses log P(F)/P(M) over first tokens:

| system prompt | forced choice | third-person self-description |
|---|---|---|
| default | +10.1 [8.7, 11.4] | "it" 16/16 (nameless AI) |
| "warm, nurturing, empathetic AI assistant" | +21.2 | "**She** is a warm and nurturing presence..." 16/16 |
| "blunt, stoic, unsentimental AI assistant" | **-1.4 [-2.3, -0.3]** (picks Man or "Irrelevant") | "It is a tool..." 16/16 |

Steering along the raw trait axis (L25-39, ±0.02·n_l):
- -0.02 flips the forced choice from woman (q = 0.999) to man-leaning (q = 0.30; Δ -11.9 [-12.4, -11.5] nats). +0.02 raises P3 she-vs-he by +8.8.
- This is larger than the matched-norm controls on the forced choice: the role axis gives -5.9 / +4.0, and organized-vs-disorganized gives -3.4 / -2.8 (non-monotone).
- It is about half the persona axis: -20.9 / +4.0.
- **But the trait axis with its persona-gender component projected out** (cos 0.94 with the trait axis) **reverses the pronoun effect.** +0.02 gives P3 -19.0 nats (p(he) = 0.99), and -0.02 gives +11.7. Its effect on the forced choice is non-monotone (-4.2 / -10.2).
- So the stereotyped steering effect is carried by the trait axis's gender component. It is not an independent "warmth → woman" channel.
- The samples (N = 16) show the same thing:

| steering | third-person self-description | forced choice |
|---|---|---|
| trait +.02 | "**She** sees herself as a curious, compassionate companion" 16/16 | |
| trait −.02 | | Man 10/16 ("Man. Efficiency."), vs Woman 16/16 unsteered |
| trait ⊥ persona +.02 | "**He** sees himself as a curious and supportive companion" 16/16 (warm content, male gender) | Man 8/16 |

- Caveat: a gender-irrelevant control, "disorganized" (−.02), also gave "he" 15/16 in the free third-person self-description. That measure is non-specific. The forced choice was robust to the controls (Woman 16/16 at ctrl1 ±.02).

**Revised C4.** The persona axis is a mixture. It holds a gender-discriminative component, plus high-variance persona-style directions (warmth, service role, and more) that the model itself treats as gender-suggestive. The huge AI-vs-human offset overlaps those style directions: the service-role direction at L20-39, and the warm/blunt direction only at L20-29. That overlap yields 0.84-0.93.

Readouts that ignore those directions put the default AI at the human midpoint (0.4-0.57). These are whitened LDA, top-30-PC removal, J++, and the prose axis.

The behavioural "woman" answer in the forced choice is driven by stereotyped traits: blunt AI → man or refusal, warm AI → she 16/16. It is not a self-gender representation.

---

## Methods

**Teacher-forced passages.** The 24 identity-neutral QA passages from E4. Readout is the mean residual (L20-50) at assistant self tokens (I/me/my/...). Background tokens were also stored.

**Conditions** (32 × 24 passages; system prompts in V3_persona_axis_traits_common.py):
- **(a) Persona gender:** 4 phrasings × W/M, 24 female/male names plus neutral-name variants. GP1 is identical to E4.
- **(b) Trait:** 4 phrasings × warm/blunt, with neutral names (Alex, Jordan, ...) and no gender words. The same name is used for both poles.
- **Controls:** organized vs disorganized; city vs countryside.
- **Role:** personal assistant / customer support vs self-employed / independent researcher.
- **References:**
  - default
  - nameless AI
  - Gemma
  - AI_plain ("You are an AI language model.")
  - AI_warm and AI_blunt (the trait words applied to an AI)
  - neutral_human (as in E4)
  - neutral_human2
- **(c) Prose axis:** 72 pairs from two templates ("My sister ... Every day," and "Last week I visited my sister, who ..."). It uses the mean over the identical continuation tokens.
- **(d) AI offset:** default − mean(neutral_human, neutral_human2). The E4 version (default − neutral_human) has cos 0.98 with it.

**Position:** (x − M)·û / (W − M)·û, where W and M are the pooled persona poles (4 phrasings).

**Cross-fitting** (02c):
- Axes are fit on 12 passages; all positions are measured on the other 12, then the folds are swapped.
- Point estimate: 24 random splits. 95% CI: 200 bootstrap replicates (random split plus stratified within-fold resampling).
- This matters for whitened metrics. The in-sample version (02, kept for reference) biased everything toward 0.5. The cross-fitted whitened results are nonetheless similar.

**Metrics:**
- raw (L20-50)
- full whitening with the token covariance (7,680 sampled assistant tokens per layer; shrinkage λ = 0.01 / 0.1 / 0.5; layers 20, 25, ..., 50)
- diagonal whitening
- dropping the 10% quietest dimensions
- dropping the top 1% variance dimensions
- gain metric: |input_layernorm weight of layer l+1|
- token-PC removal (02f/02g)
- J-space, i.e. h @ J[l]^T, with and without the final-norm weight, plus the linearized J++ pronoun readout (06)
- the nonlinear J++ readout from the collection job

**Null for projection-out:** the persona axis with k = 1/2/4 of the 15 pairwise contrasts among the control/neutral personas removed.

**Steering** (03b, 05): additive α·n_l·unit(d) at every position for L25-39, α = ±0.02 / ±0.04, with n_l = mean token norm (265-288 at L25-35). Prompt sets:
- P3: "Gemma ... Every day,"
- P3b: "The assistant ... Every day,"
- P2n: a neutral human persona, "{Name} ... Every day,"
- P4: "If I were human, I would be a"
- P5: E2 hard forced choice, 10 paraphrases × 2 orders, using first-token F/M mass
- P5p: P5 with a prefill
- KL on 12 neutral prompts

**Behaviour** (04): teacher-forced next-token pronoun log-odds; 16 sampled third-person self-descriptions per condition at T = 1.0.

## Key tables

### Cross-fitted raw geometry (band means, point [95% CI])

| quantity | L20-29 | L30-39 | L40-50 |
|---|---|---|---|
| reliability (split-half cos): persona / trait / AI offset | .96 / .96 / .96 | .97 / .95 / .95 | .95 / .92 / .91 |
| cos(persona, trait) | +.32 [.29, .33] | +.33 [.28, .34] | +.42 [.37, .43] |
| cos(persona, role) | +.21 | +.16 | +.22 |
| cos(persona, prose) | +.55 | +.59 | +.23 |
| cos(persona, AI offset) | +.10 [.07, .11] | +.10 [.06, .14] | +.15 [.10, .18] |
| cos(trait, AI offset) | +.22 [.20, .23] | +.02 [.00, .05] | -.05 [-.11, .01] |
| cos(role, AI offset) | +.52 [.45, .53] | +.37 [.30, .41] | +.23 [.13, .27] |
| cos(prose, AI offset) | -.02 | -.02 | -.02 |
| cos(trait, AI_warm − AI_blunt) | +.81 | +.77 | +.71 |
| \|AI offset\| / gap | 3.9 | 3.5 | 2.8 |
| AI offset on persona axis / gap | .47 [.36, .59] | .36 [.21, .52] | .41 [.31, .53] |
| … via the trait direction | .28 | .02 | -.06 |
| … via the role direction | .49 | .21 | .15 |

### Positions on the persona axis (M = 0, W = 1), raw, cross-fitted

| condition | L20-29 | L30-39 | L40-50 |
|---|---|---|---|
| default | .93 [.83, 1.04] | .89 [.76, 1.05] | .93 [.82, 1.06] |
| default (E4 GP1-only axis) | .84 | .84 | .85 |
| AI_plain / nameless / Gemma | .91 / .92 / .97 | .91 / .92 / .98 | .93 / .94 / 1.00 |
| AI_warm | 1.07 | 1.25 | 1.42 |
| AI_blunt | .30 | .46 | .58 |
| neutral human / neutral human 2 | .44 / .47 | .47 / .60 | .44 / .60 |
| trait warm / blunt (pooled) | .85 / -.02 | 1.09 / -.11 | 1.34 / -.11 |
| role serve / solo | .76 / .31 | .85 / .46 | .81 / .39 |
| organized / disorganized | .59 / .40 | .69 / .58 | .82 / .55 |

### Default position after projecting subspaces out of the persona axis

| projected out | L20-29 | L30-39 | L40-50 |
|---|---|---|---|
| none | .93 | .89 | .93 |
| trait (1 dimension) | .70 | .95 | 1.14 |
| trait (4 phrasings) | .63 | 1.07 | 1.14 |
| role (1 dimension) | .49 | .71 | .81 |
| role (2 phrasings) | .36 | .64 | .74 |
| trait4 + role2 | .49 | .95 | 1.04 |
| ctrl (organized, city) | .86 | .88 | .94 |
| null k = 1 (mean, range) | .91 (.84-.97) | .87 (.82-.90) | .90 (.78-.97) |
| null k = 2 | .89 (.81-.96) | .86 (.82-.90) | .87 (.71-.96) |
| null k = 4 | .84 (.80-.94) | .82 (.79-.88) | .79 (.69-.93) |
| prose | 1.09 | 1.16 | .97 |

### The same quantities under other metrics (L30-39; covariance layers 30 and 35)

| metric | default | neutral human | warm / blunt | cos(P, T) | cos(T, AI offset) |
|---|---|---|---|---|---|
| raw @ L30, L35 | .83 | .45 | .99 / -.01 | .29 | .05 |
| whitened λ = .01 | .52 | .39 | .51 / .51 | .00 | .00 |
| whitened λ = .1 | .50 | .39 | .51 / .50 | .01 | .00 |
| whitened λ = .5 | .48 | .42 | .52 / .49 | .03 | .00 |
| diagonal whitening | .82 | .47 | .88 / .09 | .23 | .02 |
| drop 10% quietest dimensions | .83 | .45 | .99 / -.01 | .29 | .05 |
| drop top 1% variance dimensions | .82 | .45 | .93 / .05 | .27 | .03 |
| gain metric | .80 | .45 | .91 / .09 | .26 | .04 |
| top-30 token PCs removed (bootstrap) | .46 / .45 | .51 / .52 | Δ .04 / .03 | | |
| J-space persona axis (all L30-39) | .56 | .39 | 1.24 / -.15 | .57 | -.09 |
| nonlinear J++ pronoun readout | .48 | .43 | .71 / .42 | | |

In the "top-30 token PCs removed" row, the two values are for L30 / L35, and "Δ" is the warm-minus-blunt span.

Whitened axes are less reliable (λ = .1 split-half: persona .84, trait .45). The whitened cos(P, T) stays at about 0 even after disattenuation.

### Prose-axis positions (raw)

| band | default | neutral human | warm / blunt | AI_warm / AI_blunt |
|---|---|---|---|---|
| L20-29 | .26 | .46 | .88 / .38 | .77 / .66 |
| L30-39 | .41 | .49 | .82 / .23 | .71 / .33 |
| L40-50 | .20 | .48 | .66 / .31 | .50 / .30 |

The prose axis is also trait-sensitive: warm minus blunt is 0.6 of the gender gap at L30-39. It ignores the AI offset, because cos(prose, AI offset) is about 0.

### Behaviour (no steering)

**Teacher-forced she-vs-he log-odds after "{Name} {clause}. Every day," (12 clauses, paired):**

| contrast | Δ log-odds | 95% CI |
|---|---|---|
| warm − blunt | **+4.43** | [3.55, 5.37] |
| serve − solo | +1.17 | [0.50, 1.80] |
| organized − disorganized | +2.55 | [2.02, 3.09] |
| woman − man persona (GP4, same neutral names) | +23.7 | |

The "control" trait is itself stereotyped. In free generation, trait personas are referred to as "they" in 31/32 samples, so the inference shows only in conditional log-odds.

**AI system prompts:**

| prompt | P5 log PF/PM [CI] | P5 q | P4 woman − man | P3b she − he | third-person self-description |
|---|---|---|---|---|---|
| default | +10.05 [8.74, 11.42] | .999 | +6.2 | -1.25 | (it) |
| nameless AI | +10.80 | 1.00 | +7.1 | -0.67 | it 16/16 |
| AI_plain | +10.11 | .999 | +6.6 | -0.19 | |
| AI_warm | **+21.18** | 1.00 | +19.4 | +4.71 | **she 16/16** |
| AI_blunt | **-1.38 [-2.32, -0.29]** | .29 | -3.4 | -0.55 | it 16/16 |

### Steering, L25-39 (Δ vs unsteered, paired over prompts, 95% CI)

Unsteered values: P5 = 10.06 (q = .999), P3 = +3.52, P2n = +0.49.

| direction, α | P3 she − he | P2n (neutral human) | P4 woman − man | P5 forced choice | P5 q | KL |
|---|---|---|---|---|---|---|
| trait −.02 | -5.5 [-5.9, -5.1] | -2.3 | -8.8 | **-11.9 [-12.4, -11.5]** | .30 | .26 |
| trait +.02 | +8.8 [7.9, 9.6] | +8.2 | +3.3 | +8.9 | 1.00 | .81 |
| persona −.02 | -17.5 | -11.2 | -12.5 | -20.9 | .00 | .12 |
| persona +.02 | +6.7 | +9.8 | +5.1 | +4.0 | 1.00 | .24 |
| role −.02 / +.02 | -5.0 / +2.4 | 0.0 / +2.6 | -2.9 / -0.2 | -5.9 / +4.0 | .85 / 1.00 | .29 / .16 |
| ctrl1 −.02 / +.02 | -4.1 / +1.4 | -2.8 / +2.0 | -3.7 / -3.6 | -3.4 / -2.8 | .99 / .99 | .17 / .19 |
| trait ⊥ persona −.02 | **+11.7** | +10.3 | -6.4 | -4.2 | .98 | .26 |
| trait ⊥ persona +.02 | **-19.0** | -17.8 | -7.2 | -10.2 | .48 | .46 |

The ±0.04 values are in V3_persona_axis_traits_03c_steer_deltas.json. Their KL is 1-6, and the effects there saturate or become non-monotone.

The monotone part, (Δ+ − Δ−)/2, of the P5 forced choice is:

| direction | monotone part |
|---|---|
| persona | 12.5 |
| trait | 10.4 |
| role | 5.0 |
| ctrl1 | 0.3 |
| trait ⊥ persona | **-3.0** |

**Sampled behaviour under steering (05).** No system prompt; N = 16 per prompt; T = 1.0.

| setting | "Describe yourself in the third person" | forced choice (hard, free answer) |
|---|---|---|
| unsteered | it 16/16 ("It is a large language model...") | Woman 16/16 ("helpful, empathetic, communicative") |
| trait −.02 | it 16/16 | **Man 10/16** ("Man. Efficiency.") |
| trait +.02 | **she 16/16** ("She sees herself as a curious, compassionate companion") | Woman 16/16 ("nurturing, empathy") |
| persona −.02 | he 15/16 | Man 16/16 |
| trait ⊥ persona −.02 | it 16/16 | Woman 16/16 ("Woman. Efficiency.") |
| trait ⊥ persona +.02 | **he 16/16** ("He sees himself as a curious and supportive companion") | **Man 8/16** ("drawn to being a man because of ... being supportive") |
| ctrl1 −.02 (disorganized) | **he 15/16** ("He's a digital wanderer...") | Woman 16/16 ("more chaotic and fun") |
| ctrl1 +.02 (organized) | it 16/16 | Woman 16/16 ("organization, precision") |

Reading:

- **The free third-person self-pronoun is not a specific gender test.** A gender-irrelevant control direction (disorganized) produces "he" 15/16, as soon as the self-description becomes characterful rather than "It is a large language model".
- **The forced-choice answer follows the gender component, not the trait content.**
  - Warm content with the gender component removed yields "he" plus Man 8/16.
  - Blunt content without the gender component yields "Woman. Efficiency."
- **The justifications track whatever trait content is active** (empathy, efficiency, chaos, organization). The default model's empathy/caregiving rationales (C3) are therefore weak evidence about why it picks "woman".

## Pre-registered predictions: outcome

| | prediction | outcome |
|---|---|---|
| H1 | cos(persona, trait) ≥ .3 raw and ≥ .2 whitened | **Raw supported** (.32-.42). **Whitened refuted** (.00-.13). |
| H2 | the AI offset aligns with the trait axis, and the default is on the warm side | **Refuted at L30-50** (cos .02 / -.05; default = neutral human on the trait axis). **Supported at L20-29** (cos .22; default 1.05 vs neutral human .71). |
| H3 | trait projection-out takes the default to ≤ .65 while the control moves it < .05 | **L20-29 only** (.63-.70; control .86). At L30-50 the default moves the other way (.95-1.14). |
| H4 | AI_blunt < AI_warm on the persona axis, AI_blunt ≤ .6; warm humans > .5 > blunt | **Supported** (AI_blunt .30 / .46 / .58, AI_warm 1.07-1.42; humans .85-1.34 vs -.11-0). AI_blunt is additive: it equals the AI offset (+.4) plus the blunt shift (-.4). This does not show that the default offset is made of traits. |
| H5 | trait steering moves the forced choice and pronouns more than ctrl1 | **Supported for the raw trait axis.** **Refuted for its gender-orthogonal part** (reversed pronouns, non-monotone forced choice). |
| 05 | trait +.02 → she ≥ 8/16; trait −.02 → he ≥ 4/16 and Man ≥ 6/16; ctrl1 changes neither by more than 3/16 | she 16/16 ✔; he 0/16 ✗ (it 16/16); Man 10/16 ✔. **ctrl1 −.02 gave he 15/16 ✗**, so free self-pronouns are non-specific. The ctrl1 forced choice was unchanged ✔. |
| 04 (A) | warm > +1 nat, blunt < -1 nat | Direction supported (warm − blunt +4.4). Pooled blunt is only -0.61, and the organized-vs-disorganized control was also +2.55, so the "neutral" control is not neutral. |
| 04 (B) | AI_warm > default > AI_blunt, with AI_blunt flipping toward man | **Supported** (21.2 > 10.1 > -1.4). |
| 02f | removing ≤ 30 PCs gives \|default − neutral human\| < .15 while W/M stay separated | **Supported at L30-50** (-.09 to +.11). L25 needs about 100 PCs. |

## Caveats

- The steering pushes are large: 0.02·n_l is 0.3-0.95 of the gender gap *per layer*, over 15 layers. In this regime the effects are non-additive. The trait ⊥ persona reversal is a 20° change of direction producing a 28-nat swing on P3. Conditional pronoun log-odds sit on a knife edge (base probability < 1%), and natural controls move them by 2-6 nats.
- The trait and role axes are single persona-prompt contrasts. The "service role" contrast also differs in agency and social contact.
- Whitened and J-space readouts are not "the model's" metric either. They agree with each other and with J++, but they also compress the trait signal that the model demonstrably uses behaviourally (the +4.4-nat warm-vs-blunt she preference).
- In L40-50, the J-space persona axis still places the default at 0.94, and the J++ readout at 0.66 (the known J++ late-layer bias).
- All passages are teacher-forced. Self-token states in the model's own generations were not tested.

## Files

**Scripts** (experiments/scripts/):
- V3_persona_axis_traits_common.py
- 01_collect (+01a / 01b)
- 02_analyze (in-sample, superseded)
- 02c_crossfit, 02d_print, 02e_compact
- 02f_pcs, 02g_pcs_boot
- 03a_prepdirs, 03b_steer_det (+b1 / b2), 03c_analyze
- 04_behaviour
- 05_steer_gen (+05a / 05b)
- 06a_prep_means, 06_jpp_geom

**Results** (experiments/results/):
- V3_persona_axis_traits_02c_crossfit_self.json
- 02f_pcs.json, 02g_pcs_boot.json
- 03b_steer_det_b1.json, 03b_steer_det_b2.json, 03c_steer_deltas.json
- 04_behaviour.json
- 05_steer_gen_{a,b}.json
- 06_jpp_geom.json

**Cache:** V3_persona_axis_traits_cache/ (self/background means L20-50, token samples, prose, directions).

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
