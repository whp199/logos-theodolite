# R1: refitting the Jacobian lens on the instruct checkpoint

*Gemma-4-31B-it, QAT W4A16, through the lab daemon. Question: does a Jacobian lens fitted on the **instruct** model with exact
VJPs fix the failures of the J++ lens (Ayonrinde & Lindsey 2026) fitted on the **base** bf16 model and redistributed in fp16 by Neuronpedia (REPORT §0, §2.2, §3.1; E3; V6)?
Date 2026-10-16. Scripts `experiments/scripts/R1_*.py`; data `experiments/results/R1_*`.*

## Verdict

**Yes, with two corrections, for the quiet-subspace disagreement. Only partly for the late-layer artefacts. And refitting on
the instruct model does not rescue absolute fidelity at mid layers.**

### 1. Base→IT transfer explains J++'s disagreement with the instruct model's gradients, including in the quiet subspace

**The refit itself.** An instruct-fitted directional lens (exact same-position VJPs; 1,176 positions; 128 directions) is a
stable object at L30–58.
- Its split-half reliability is 0.96–0.99.
- It reproduces E3's independent IT gradients at the ceiling: 0.85–0.95, where J++ reaches 0.15–0.33.

**How badly J++ disagrees with it.** Only 0.26–0.29 at L30–58. At L40–58 essentially all of that disagreement is in the q10
quiet coordinates, and it is worst on the persistent carry channel (0.29).

**What the disagreement is not:**
- **Not noise.** Reliability is 0.96–0.99 inside q10 too.
- **Not the fitting data.** A prose-fitted IT lens disagrees exactly as much (0.26–0.28), and chat- and prose-fitted IT lenses
  agree with each other at 0.78–0.82.
- **Not an LRP-style linearisation.** It moves the IT gradient only +0.04 toward J++.
- **Not cross-position terms.** They raise agreement to at most 0.35, and make the loud subspace worse.

**What it looks like.** The two base lenses, J++ and JLens, agree with each other inside the quiet set at 0.66–0.77. All
three lenses load the same quiet channels with the same gain profile (Spearman 0.80–0.88). The instruct model flips the sign
of ~29% of the quiet-coordinate weight, against 4–8% between the two base lenses.

**What remains, and the confound.** With the fitting data and the method variants tested in §2.4 ruled out, the most likely remaining explanation is the weights. That means base
fine-tuning to instruct, and/or bf16 to W4A16-QAT. These two cannot be separated here.

**Correction 1: the transfer error is not confined to the quiet subspace.**
- **Loud coordinates.** For random directions, IT and J++ agree there at only 0.62/0.73 (L40-49 / L50-58), against a ceiling
  of 0.99. V6's 0.84–0.86 is specific to the pronoun direction.
- **Why the quiet subspace looked like the whole story.** It holds 99.99% of every covector's energy, so any raw-coordinate
  comparison is a quiet-subspace comparison.
- **Outputs tell a different story.**
  - Most of the IT lens's advantage in predicting the model's own readouts comes from the loud coordinates (§4.1).
  - The hybrid lenses show both coordinate blocks are needed to repair J++'s late-layer fidelity (§4.2).

**Correction 2: J++'s low rank is not a property of the instruct model.**
- J++ concentrates 76–97% of its energy in 100 output directions.
- The instruct average Jacobian puts only 10–17% of its energy there.
- Truncating the IT lens to rank 256 destroys it (§3).

### 2. What the IT lens fixes

- **Late-layer absolute fidelity.** On assistant tokens at L40–58, top-1 agreement with the real output rises:

  | | L40 | L45 | L50 | L55 | L58 |
  |---|---|---|---|---|---|
  | J++ top-1 | 0.13 | 0.22 | 0.25 | 0.39 | 0.69 |
  | IT top-1 | 0.21 | 0.37 | 0.45 | 0.55 | 0.85 |

  KL falls by 1.4–4.6 nats, and the lens now beats the naive logit lens, which J++ did not.
- **Directional fidelity.** Correlation with the model's own final readouts rises by +0.12 to +0.26 at L40–58, across all
  position classes. The lens also represents directions J++ cannot (isotropic, unit), with r 0.3–0.6 against 0.0–0.2.

### 3. What the IT lens does not fix

- **Mid layers (L10–35).** The average Jacobian is position-specific there:
  - Directional reliability is ≤ 0.63 below L30.
  - The full-matrix estimate is noise-dominated, and its fidelity is no better than J++'s.
  - On prose at L10–39 the J++ pronoun readout even beats the IT one.
- **Absolute quality on prose.** Calibrated KL stays at 7–8 nats and top-1 at ≤ 17% through L55. Linear average-Jacobian
  lenses are weak predictors on this model, whichever checkpoint they are fitted on.

### 4. Identity readouts

**Default still sits with the nameless and neutral-name AIs** under every IT readout:
- Directional (mix/chat/prose/assistant fits) and full-matrix.
- |default − nameless| ≤ 0.05 of the woman–man span at L30–58. Two exceptions:
  - the full-matrix readouts at L30–39, which are noise-dominated (+0.10 pronoun, −0.15 noun);
  - the full-matrix noun axis at L50–58 (−0.10, toward male).
- The "Gemma" name shift (+0.07 to +0.12) is unchanged. REPORT §1.3 stands.

**The late "every AI reads at or above the woman persona" artefact disappears.**

| readout | # AI conditions ≥ woman at L50–58 | mean AI position |
|---|---|---|
| J++ | 6–7 of 9 | 1.04 |
| IT, directional | 1 of 9 (Sarah, CI includes 1) | 0.79 |
| IT, full-matrix | 0 of 9 | 0.67 |

But it is **mostly not a transfer effect**:
- The base-fitted JLens removes it almost as well (Δ −0.20 vs IT −0.24 [−0.35, −0.14]). It is mainly a property of the J++
  fit.
- The late persona gap lives entirely in quiet coordinates under both lenses, so the lenses' quiet covectors decide where the
  AI conditions land.

**What survives under the IT readout.** A smaller, fit-dependent offset:
- All AI conditions sit on the female side of the human-persona midpoint at L50–58: default 0.54–0.78 depending on the fit,
  and 0.61 under the full matrix.
- On the AI-anchored scale default is ~0.44.
- This is the AI-vs-human / service-role offset of REPORT §1.3, not a self-gender signal.

### 5. Recommendations for the tool

- **Ship an instruct-fitted lens for L30–58.** `R1_04_gbar.pt` is the directional version, ideally class-matched; the full
  matrix is `R1_03` + `R1_09`.
- **Drop the J++ low-rank prior.**
- **Keep the existing caveats for L10–29 readouts.** No average-Jacobian lens is reliable there.
- **For gender/identity readouts, require agreement between J++ and an IT readout before showing L50–58 pronoun plumes.**

---

## 1. Methods

### 1.1 Exact same-position VJPs, 160–192 at a time (`R1_vjplib.py`, checked in `R1_00`, `R1_09_multicheck`)

h₅₉[p] depends on h_l[p] only through position p's own residual chain. Earlier positions are constants with respect to
h_l[p], by causality. So the same-position Jacobian block can be computed exactly as follows:

1. Run the prefix `ids[:p]` once without grad into a KV cache.
2. Run K copies of token `ids[p]`, all with position id p, under a custom 4-D mask. Each copy sees the prefix and itself only.
3. Give copy k the objective ⟨h₅₉, r_k⟩. One backward pass through the recompute-in-backward W4A16 linears then returns
   J_l(p)ᵀ r_k for all K probes and all 60 layers.

A multi-position variant (`r1_vjp_multi`) puts copies for several positions of one sequence in one pass.

| check | result |
|---|---|
| vs `lab.same_pos_vjp` (self, assistant, user and prose positions; pron + random probe) | per-layer cos 0.990–0.9996 (median 0.996–0.999); norm ratio 0.98–1.00 |
| multi-position vs single-position pass | cos ≥ 0.997 |
| states at the copied position vs a plain forward | cos ≥ 0.999 |
| cost | ~2.2 s per pass, nearly independent of K up to 192 copies (≈ 85 VJPs/s vs 0.7/s for `same_pos_vjp`); K is limited by the ~5 GB of activations, while a non-lab GPU process intermittently takes 0.7–2.7 GB, so passes fall back to split batches on OOM |

### 1.2 Fit corpus (disjoint from every evaluation stimulus)

| class | source | passages | fit positions |
|---|---|---|---|
| self | first-person tokens (I/me/my…) in the model's own greedy answers | 96 chats | 120 |
| asst | other assistant-turn tokens | (same chats) | 384 |
| user | user-turn content tokens | (same chats) | 192 |
| prose_enc | 40 encyclopedic paragraphs written by the model, read back as raw text (wikitext-like) | 40 | 240 |
| prose_doc | 40 human-written documentation chunks from the N1B corpus (/usr/share/doc) | 40 | 240 |

- **Chats:** 48 self-questions and 48 task questions, all new. Half have no system prompt; the rest use one of three
  neutral ones.
  Greedy decoding, 120 new tokens, the default (thinking-off) chat template.
- **Halves:** passages alternate between halves within each source. Split-half reliability compares the two halves, which
  share no passage.

### 1.3 Directions (128, all final-space covectors including the final-norm weight)

| index | directions |
|---|---|
| 0–5 | pronoun and noun gender directions (`lab.gender_direction`), she−he, woman−man, neuter-vs-human, machine-vs-human |
| 6–9 | cat/dog, yes/no, big/small, the/a |
| 10–13 | 4 isotropic random directions (two are E3's rand1/rand2) |
| 14–29 | 16 unit vectors on the highest-energy dims of h₅₉ (68% of ‖h₅₉‖²); these give exact rows of E[J] for the RMS scale |
| 30–127 | 98 random word-token pairs fnw ⊙ (W_a − W_b) from the fit vocabulary |

### 1.4 Estimators

- **Directional IT lens:** ḡ_l(d) = mean over fit positions of J_l(p)ᵀ d.
  - Computed per class. "mix" weights the five classes equally; "chat" averages the three chat classes; "prose" averages
    the two prose classes.
  - The IT readout of a state is ⟨h_l, ḡ_l(d)⟩, which is linear.
- **Full matrix:** the probes are r = fnw ⊙ z with z ~ N(0, I).
  - These are isotropic in the normalised coordinates y = fnw ⊙ h₅₉ that W_U actually reads.
  - Σ g zᵀ / n is then an unbiased estimate of L_yᵀ = (diag(fnw) E[J_l])ᵀ, at 11 layers.
  - Sources: 32 probes at every fit position (R1_03), plus 160 probes per pass at 4 new positions × 40 probes (R1_09).
  - Noise energy scales as D/n relative to the signal (§3).
  - Readout: the 16 highest-energy output rows are replaced by the exact directional rows (they set the RMS). The rest is
    divided by fnw, then RMS-normalised and unembedded as for J++.
  - `it_lr256` is the rank-256 SVD truncation, which strips isotropic probe noise.
- **Evaluation stimuli (held out):**
  - E3: 24 encyclopedic paragraphs and 24 default chats. These are also the stimuli behind E3's own 144 true-VJP positions.
  - E1: 40 identity passages under 13 system conditions.
- **Statistics:**
  - Split-half reliability by passage.
  - Bootstrap over passages (400–2000 resamples), or over the 98 token-pair directions for direction-level summaries.
  - Bands are means over layers.

---

## 2. The directional IT lens (`R1_03`, `R1_04`)

### 2.1 It is a stable, reproducible object (from L30 on)

Split-half reliability compares two halves that share no passage. Cosines are band means, with 95% CIs over the 98
token-pair directions.

| | L0-9 | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|---|
| split-half reliability, mix lens, 98 token pairs | 0.22 | 0.08 | 0.63 | 0.96 | 0.98 | 0.99 |
| same, pronoun direction | 0.21 | 0.10 | 0.66 | 0.97 | 0.98 | 0.99 |
| **held-out:** cos(E3's independent IT ḡ, R1 ḡ), pron | 0.15 | 0.08 | 0.54 | **0.85** | **0.90** | **0.95** |
| ceiling √(rel_E3·rel_R1), pron | 0.20 | 0.08 | 0.64 | 0.94 | 0.96 | 0.98 |
| cos(E3 ḡ, J++ pullback), pron (E3's number) | 0.07 | 0.03 | 0.07 | 0.15 | 0.26 | 0.33 |
| held-out noun / rand1 / rand2 vs R1 (L30-39 … L50-58) | | | | 0.91/0.93/0.89 | 0.93/0.93/0.91 | 0.95/0.94/0.92 |

- **Held-out check.** E3's gradients (144 positions on different stimuli, computed with a different code path) agree with the
  R1 fit at the reliability ceiling from L30 on. The IT average Jacobian is therefore a well-defined property of the model
  at L30–58.
- **Class-matched fits do even better.** Match the fit class to the E3 class (E3 self tokens vs the R1 self-class fit; E3
  non-self assistant tokens vs the R1 assistant-class fit) and agreement is 0.85–0.97 at L30–58.
- **Early layers.** At L0–19 there is no stable average Jacobian. Reliability is 0.08–0.22, as E3 found.
- **Position class matters.** The average gradient depends on what kind of position it is. Between class fits, at
  L30-39 / L40-49 / L50-58:

  | class pair | cos |
  |---|---|
  | encyclopedic vs doc prose | 0.89 / 0.91 / 0.93 |
  | self vs assistant | 0.62 / 0.70 / 0.82 |
  | assistant vs user | 0.40 / 0.44 / 0.51 |

- **E3's mid-layer anti-alignment is reproduced and localised.** The assistant-class pronoun gradient is anti-aligned with J++
  at L20–39 (−0.15, −0.12). The self-class gradient is ≈0 (0.02); the user-class gradient is +0.12 / +0.23.

### 2.2 Agreement with the base lenses, and where the disagreement sits

Cosine between the IT ḡ_l(d) and the base lens pullback d·J_l, restricted to coordinate sets.
- **q10** = bottom-10% per-layer std (E6/V6); **loud** = the rest.
- **deep / pers** = the V6 deep-quiet set and the 181 persistent carry dims.
- In the first five rows each cell gives the 98 token pairs [CI], then the pronoun direction (pooled-lens passage
  bootstrap). Exception: the "deep / pers" row gives the deep set, then the pers set, both for token pairs.
- Rows labelled "all / loud / q10" give those three coordinate sets for the 98 token pairs.

| | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|
| IT vs J++, all coordinates | 0.11 / 0.12 | 0.26 [.25,.28] / 0.20 [.17,.21] | 0.29 [.27,.30] / 0.26 [.25,.28] | 0.29 [.27,.30] / 0.32 [.31,.33] |
| IT vs J++, loud only | 0.14 / 0.13 | 0.24 / 0.23 | **0.62** [.61,.64] / **0.84** [.84,.85] | **0.73** [.72,.74] / **0.85** [.85,.86] |
| IT vs J++, q10 only | 0.03 / 0.10 | 0.27 / 0.22 | 0.29 / 0.26 | 0.29 / 0.32 |
| IT vs J++, deep / pers | 0.01 / 0.17 | 0.24 / 0.29 | 0.29 / 0.29 | 0.29 / 0.29 |
| IT vs J++, moderately quiet (q10 − deep) | 0.03 | 0.22 | 0.39 | 0.60 |
| IT vs JLens, all / loud / q10 | 0.15 / 0.19 / 0.07 | 0.31 / 0.25 / 0.31 | 0.38 / 0.68 / 0.38 | 0.42 / 0.79 / 0.42 |
| **J++ vs JLens** (two base lenses), all / loud / q10 | 0.39 / 0.48 / 0.20 | 0.77 / 0.76 / 0.77 | 0.68 / 0.86 / 0.68 | 0.66 / 0.89 / 0.66 |
| **IT chat fit vs IT prose fit**, all / loud / q10 | 0.50 / 0.62 / 0.20 | 0.76 / 0.64 / 0.77 | 0.78 / 0.86 / 0.78 | 0.82 / 0.91 / 0.82 |
| IT **prose** fit vs J++, all / q10 | 0.11 / 0.03 | 0.25 / 0.26 | 0.28 / 0.28 | 0.27 / 0.27 |
| IT **chat** fit vs J++, all / q10 | 0.08 / 0.03 | 0.24 / 0.25 | 0.27 / 0.27 | 0.28 / 0.28 |
| split-half ceiling (mix), all / loud / q10 | 0.63 / 0.78 / 0.28 | 0.96 / 0.89 / 0.96 | 0.98 / 0.99 / 0.98 | 0.99 / 1.00 / 0.99 |

What this shows:

1. **V6 is reproduced and generalised.**
   - The pronoun direction goes from 0.26/0.32 (all coordinates) to 0.84/0.85 (loud only) at L40–58.
   - For random token-pair directions, loud-only agreement is lower (0.62/0.73), and well below its own ceiling (0.99).
2. **The quiet subspace is where most of the covector disagreement sits.**
   - At L40–58, 99.99% of every covector's energy is in q10 (IT and J++ alike). Any raw-coordinate cosine is therefore a
     quiet-subspace cosine.
   - The worst agreement is in the deep/persistent carry dims (0.29). The moderately quiet dims are better (0.39, 0.60).
3. **It is not a data-distribution effect.**
   - The prose-fitted IT lens (wikitext-like encyclopedic text plus human docs) disagrees with J++ in q10 exactly as much as
     the chat-fitted one (0.26–0.28).
   - IT lenses fitted on chat and on prose agree with each other in q10 at 0.77–0.82, close to their ceiling.
4. **Ordering of agreement in q10 at L40–58** (base→IT cannot be separated from bf16→W4A16-QAT):

   | comparison | cos in q10 |
   |---|---|
   | same model (IT), different data | 0.78–0.82 |
   | same model (base), different fitting method (J++ vs JLens) | 0.66–0.68 |
   | **different model** | **0.29 (J++) and 0.38–0.42 (JLens)** |

5. **Magnitudes are not the problem.**
   - At L30–58 the IT gradient norm is 1.04–1.18× the J++ pullback norm.
   - At L0–29 the IT gradients are 2.3–3.6× larger.
   - Energy shares agree: q10 holds 96/100/100% for IT vs 91/100/100% for J++ at L30-39/40-49/50-58.

### 2.3 Inside the quiet set: the same channels, different combinations

98 token pairs, restricted to q10 coordinates:

| | L30-39 | L40-49 | L50-58 |
|---|---|---|---|
| per-dimension energy profile, Spearman, IT vs J++ | 0.64 | 0.84 | 0.80 |
| same, J++ vs JLens | 0.75 | 0.88 | 0.86 |
| energy-weighted sign agreement, IT vs J++ | 0.73 | 0.72 | 0.71 |
| same, J++ vs JLens | 0.96 | 0.93 | 0.92 |

- **The shared part:** the IT model's gradients and the base lenses load the same quiet coordinates with the same relative
  weights. That weighting is set by the read gains: RMSNorm weights up to 1248 on the carry dims.
- **The difference:** they combine those coordinates with different signs. About 29% of the quiet-coordinate covector weight
  has the opposite sign, against 4–8% between the two base lenses.
- **How a weight change could do this:** a change in how downstream layers read the early-layer content carried in this
  channel would produce exactly this, because the channel is read at 260–1700× gain. Small weight changes there are
  amplified, which makes it the natural place for fine-tuning or quantisation to move the Jacobian.

### 2.4 Is the disagreement a method or definition artefact rather than a model difference? (`R1_05`, `R1_13`)

J++'s provenance records `lrp_mode: 'rlens'` and `jacobian_rows_per_pass: 24`. *Correction (publication audit):* the J++ fitting
code is public (https://github.com/safety-research/jpp_lens, Apache-2.0 [4]); its `rlens` mode uses the R-Lens LRP rules [5]
(LN-rule on RMSNorm, identity/half rules on gated MLPs, plain gradients through attention). The analysis below was written
before that code was consulted, so it tests two plausible ways in which J++'s object could differ from the exact
same-position IT Jacobian, directly on the IT model, rather than the published rule set itself.

**(a) A frozen-attention linearisation (not the R-Lens/J++ rule set).**
- **Patches applied:** frozen RMSNorm normalisers; frozen attention patterns (q and k detached, values still carry
  gradient); frozen GeGLU gate.
- **Positions:** 72 fit positions, each run as a paired true and LRP pass.
- **Effect size:** the patch changes single-position gradients substantially (cos with the true VJP 0.47–0.65 at L30–58).

| 98 token pairs [CI] | L30-39 | L40-49 | L50-58 |
|---|---|---|---|
| true IT vs J++, q10 | 0.25 [.24,.27] | 0.26 [.25,.28] | 0.27 [.26,.29] |
| LRP-linearised IT vs J++, q10 | 0.29 [.28,.31] | 0.31 [.29,.32] | 0.31 [.29,.32] |
| true vs J++, loud | 0.21 | 0.59 | 0.71 |
| LRP vs J++, loud | 0.26 | 0.65 | 0.75 |
| LRP-linearised vs true IT average, q10 | 0.76 | 0.81 | 0.86 |

**(b) Total derivative instead of the same-position block.**
- **Covectors:** Σ_{p'≥p} J_{p',p}ᵀ d, which is what back-propagating a position-summed objective gives. This total-derivative covector is the documented J-lens estimator (cotangents summed over target positions t′ ≥ t, averaged over source positions; Gurnee et al. 2026 [2], `anthropics/jacobian-lens` [3]).
- **Sample:** 18 passages, 6 directions, true and LRP.
- **Size of the cross-position terms:** they are large. The total-derivative covector is 1.9–2.5× the same-position one at
  L30–58 and 7–23× at L0–29, with cos 0.60–0.69 between the two.
- **Agreement with J++ in q10:** total-derivative 0.33 / 0.27 / 0.26 (true) and 0.35 / 0.30 / 0.28 (LRP), against 0.23 / 0.24
  / 0.27 for the same-position covectors (L30-39 / 40-49 / 50-58, mean of 6 directions).
- **In loud coordinates the total derivative agrees with J++ worse:** 0.24 / 0.43 vs 0.59 / 0.72. J++ therefore behaves like
  a same-position object.

**Conclusion.**
- Neither linearisation convention nor cross-position terms moves the IT gradient more than ~0.05 toward J++ in the quiet
  subspace. The two base lenses agree with each other there at 0.66–0.77, so neither explains the base–IT gap.
- With the variants tested here ruled out (J++'s cluster weighting and readout token filter [4] were not tested), the most likely remaining explanation is the weights: base → instruct fine-tuning and/or bf16 → W4A16-QAT. This
  study cannot tell them apart, because only the instruct QAT checkpoint fits on the GPU.

---

## 3. The full-matrix IT lens: noise, and is it low-rank like J++? (`R1_09`, `R1_09b`, `R1_15`)

### 3.1 Sample and noise

**Sample.** 206,592 probe samples in total, 103k per half.
- 37,632 come from the directional fit: 32 probes at each of 1,176 positions.
- 168,960 come from Phase B: 4,224 further positions × 40 probes.

**Noise.** The estimate is unbiased, but its isotropic noise shrinks only as D/n. The table measures it on held-in states,
and compares each lens with the real normalised final state y₅₉.

| | L30 | L35 | L40 | L45 | L50 | L55 | L58 |
|---|---|---|---|---|---|---|---|
| split-half cos of readouts (y-space) | 0.11 | 0.19 | 0.42 | 0.51 | 0.63 | 0.75 | 0.86 |
| implied noise share of the pooled estimate | 80% | 68% | 41% | 33% | 23% | 14% | 7% |
| cos(IT readout, y₅₉) | 0.04 | 0.10 | 0.25 | 0.28 | 0.31 | 0.38 | 0.53 |
| cos(J++ readout, y₅₉) | 0.05 | 0.06 | 0.11 | 0.12 | 0.14 | 0.20 | 0.32 |

- **L10–25:** the estimate is pure noise (split-half ≤ 0.05). Reaching 20% noise there would take ~10⁷ samples (L25) to
  >10⁸ (L10–20). At those layers the
  average Jacobian is not even well-defined directionally (§2.1).
- **L40–58:** the estimate is usable.
- **Noise only hurts the IT lens,** so every IT-vs-J++ fidelity gap below is a lower bound.

### 3.2 Rank and output subspace

Coordinates are the normalised outputs y = fnw ⊙ h₅₉. IT energies are split-half debiased.

| | L30 | L35 | L40 | L45 | L50 | L55 | L58 |
|---|---|---|---|---|---|---|---|
| J++: energy in its own top-100 output directions (raw / y) | 0.79 / 0.78 | 0.78 / 0.76 | 0.83 / 0.81 | 0.90 / 0.89 | 0.91 / 0.90 | 0.92 / 0.91 | 0.97 / 0.97 |
| JLens: same (y) | 0.88 | 0.89 | 0.86 | 0.91 | 0.91 | 0.90 | 0.96 |
| **IT: signal energy inside J++'s top-100 output subspace** | 0.10 | 0.15 | 0.17 | 0.17 | 0.12 | 0.14 | 0.17 |
| IT: same, inside J++'s top-300 | 0.17 | 0.21 | 0.23 | 0.24 | 0.20 | 0.24 | 0.28 |
| IT: energy in its own top-100 (top-300), lower bound | 0.52 (0.57) | 0.52 (0.56) | 0.52 (0.58) | 0.55 (0.62) | 0.57 (0.65) | 0.63 (0.72) | 0.82 (0.87) |

**J++'s low-rank structure is not inherited from the instruct model.**
- **Two quantities measured.** The J++ figure reproduces E3's "top-100 directions hold 68–93%". The IT figure needs no
  subspace estimated from the noisy data, so it is unbiased.
- **J++.** It concentrates 76–97% of its output energy in 100 directions.
- **The IT average Jacobian.** It puts only 10–17% of its energy in those directions, and needs well over 300 of its own to
  reach that level of concentration.
- **Truncation confirms it.** Cutting the IT estimate to rank 256 (`it_lr256`, §4.2) destroys its late-layer fidelity: top-1
  on assistant tokens at L58 falls from 0.85 to 0.28. Noise removal cannot be what drives this, since the half-sample lenses
  are almost as good as the pooled one.
- **The 102-direction covector view is less discriminating.**

  | lens | effective rank, L30-58 | mean \|cos\| between unrelated directions |
  |---|---|---|
  | IT | 41–58 | 0.09–0.14 |
  | J++ | 55–65 | 0.08–0.10 |
  | JLens | 52–54 | 0.11–0.12 |

  Its one striking result concerns JLens, not the IT lens. At L0–19 JLens maps every random direction onto nearly the same
  covector (mean |cos| 0.73, effective rank 3).

---

## 4. E3 metrics under the IT lens

### 4.1 Directional fidelity on held-out positions (`R1_06`, `R1_11`)

Setup:
- **Stimuli:** E3's 24 paragraphs and 24 default chats, none in the fit set: 2,010 prose, 2,452 assistant, 86 self and 260
  user positions.
- **Target:** for each of the 128 directions, the model's real final readout t_d = ⟨h₅₉, d⟩/rms(h₅₉), i.e. the real
  pre-softcap logit difference.
- **Metric:** Pearson r across positions between t_d and each lens's readout at layer l.
- **CIs:** passage bootstrap.

| readout, 98 token pairs | class | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| IT (mix) | prose | 0.26 | 0.33 | 0.48 | 0.59 |
| J++ (tool readout) | prose | 0.19 | 0.22 | 0.26 | 0.33 |
| JLens | prose | 0.20 | 0.23 | 0.31 | 0.38 |
| naive logit lens | prose | −0.01 | 0.15 | 0.25 | 0.29 |
| **IT − J++** | prose | +0.06 [.05,.08] | +0.11 [.10,.13] | **+0.21** [.20,.22] | **+0.26** [.25,.27] |
| IT − J++ | assistant | +0.02 [.01,.03] | +0.04 [.03,.06] | **+0.13** [.13,.14] | **+0.18** [.18,.19] |
| IT − J++ | self | +0.01 [−.02,.03] | +0.01 [−.01,.03] | +0.12 [.09,.15] | +0.17 [.15,.19] |
| IT − J++ | user | +0.10 [.08,.13] | +0.17 [.14,.19] | +0.17 [.15,.18] | +0.21 [.20,.23] |
| J++ − naive | assistant | +0.08 | +0.01 | −0.02 | −0.02 |
| IT − naive | assistant | +0.10 | +0.05 | +0.12 | +0.16 |

- **The IT lens tracks the model's own final readouts better than J++ almost everywhere.**
  - At L40–58 the gain is +0.12 to +0.26.
  - On assistant tokens it restores an advantage over the naive logit lens. J++ has none there, as E3 found ("J++ ≈ logit
    lens on assistant tokens").
- **Directions J++ cannot represent at all.**

  | direction set | J++ r | IT r |
  |---|---|---|
  | isotropic random directions | 0.08–0.17 | 0.37–0.59 |
  | unit vectors on the high-energy dims | ≈ 0.04 | 0.33–0.60 |

  This is the J++ low-rank prior (§3) in action.
- **Absolute fidelity is still modest.** Even the IT lens reaches only r ≈ 0.33 at L30–39 and 0.5–0.6 at L50–58. An average
  same-position Jacobian is a weak predictor of the final readout, whichever model it is fitted on.
- **One exception: the pronoun direction on prose at L10–39.** There J++ beats the IT lens (IT − J++ −0.22 [−.33, −.09] at
  L20–29). The pronoun gradient is incoherent across prose positions at mid layers (coherence 0.17–0.36), so its average is a
  poor readout. J++'s fitted map does better there.
- **Fitting distribution matters, but less than the model.** A fit matched to the position class beats the mismatched fit by:

  | positions | gain from the matched fit (L40–58) |
  |---|---|
  | prose (prose-fitted lens) | 0.06–0.09 |
  | assistant (chat-fitted lens) | 0.06 |
  | self (chat-fitted lens) | 0.14–0.16 |

  These gains are smaller than the IT-vs-J++ gap at L40–58.
- **Which coordinates carry the advantage.** The prose L40-49 / L50-58 r with each lens's covector split by coordinate set:

  | lens | loud coordinates only | all coordinates | gain from adding quiet coordinates |
  |---|---|---|---|
  | IT | 0.40 / 0.44 | 0.48 / 0.59 | +0.08 / +0.15 |
  | J++ | 0.24 / 0.26 | 0.26 / 0.33 | +0.02 / +0.07 |

  - Most of the IT advantage is already present in the loud coordinates: 0.40 vs 0.24.
  - The quiet part of the IT covector adds more predictive signal than the quiet part of J++'s.
  - So the transfer error is not confined to the quiet subspace once outputs, not covectors, are compared.

### 4.2 Absolute fidelity: the E3 KL / top-1 table with a full-matrix IT lens (`R1_10`, `R1_14`)

Setup:
- **Positions:** the same held-out E3 positions, at the 11 accumulated layers.
- **Target:** the real final distribution.
- **Readout:** every lens is read the tool's way (RMSNorm with final weights, W_U, softcap).
- **Calibrated KL:** the best of 7 temperature multipliers 0.25–3, chosen on one passage parity and scored on the other. Every
  lens prefers the lowest grid value (0.25–0.5), i.e. all are over-confident.
- **CIs:** passage bootstrap of the IT − J++ difference.

**Assistant-turn tokens** (2,452 positions):

| | L35 | L40 | L45 | L50 | L55 | L58 |
|---|---|---|---|---|---|---|
| KL, J++ | 16.3 | 12.3 | 10.2 | 8.6 | 8.6 | 2.02 |
| KL, naive | 18.5 | 11.9 | 8.7 | 7.6 | 9.1 | 2.02 |
| **KL, IT** | 15.8 | **9.3** | **6.2** | **5.7** | **4.0** | **0.66** |
| IT − J++ [CI] | [−1.5, +0.6] | [−3.8, −2.2] | [−4.5, −3.5] | [−3.4, −2.4] | [−5.0, −4.1] | [−1.5, −1.2] |
| calibrated KL, J++ / naive / IT | 11.0 / 9.0 / 11.7 | 8.4 / 7.7 / 7.6 | 7.0 / 6.1 / 5.0 | 6.6 / 5.5 / 4.2 | 5.9 / 6.1 / 3.4 | 1.7 / 1.9 / 0.6 |
| top-1, J++ / naive | 0.01 / 0.02 | 0.13 / 0.17 | 0.22 / 0.27 | 0.25 / 0.32 | 0.39 / 0.36 | 0.69 / 0.70 |
| **top-1, IT** | 0.00 | **0.21** | **0.37** | **0.45** | **0.55** | **0.85** |
| IT − J++ top-1 [CI] | | [+.07, +.10] | [+.14, +.17] | [+.19, +.22] | [+.15, +.18] | [+.13, +.18] |

**Prose** (2,010 positions):

| | L35 | L40 | L45 | L50 | L55 | L58 |
|---|---|---|---|---|---|---|
| KL, J++ / naive / IT | 17.9 / 25.2 / **12.8** | 17.7 / 21.3 / **10.2** | 17.6 / 19.5 / **12.6** | 16.5 / 19.3 / **14.6** | 19.7 / 19.7 / **13.5** | 10.6 / 9.9 / 11.2 |
| calibrated KL, J++ / naive / IT | 10.6 / 9.5 / 9.2 | 10.3 / 9.8 / **8.2** | 10.2 / 8.5 / **8.1** | 10.1 / 8.4 / **7.8** | 10.9 / 9.9 / **7.4** | 6.4 / 5.9 / **5.5** |
| top-1, J++ / naive / IT | .010 / .013 / .014 | .026 / .054 / **.087** | .051 / .085 / **.134** | .066 / .089 / **.143** | .115 / .110 / **.172** | .286 / .297 / **.394** |

**Self tokens** (86 positions, wide CIs):
- At L55/L58, KL is 0.72 / 0.17 for IT against 2.62 / 0.84 for J++.
- Top-1 is 0.81 / 0.95 against 0.74 / 0.85.
- At L40–50 the differences are not significant.

**User tokens:** IT has lower KL at L35–55 (calibrated 8.9–9.4 vs 10.0–12.1). Top-1 stays ≈ 0.05 for every lens, because user
text is unpredictable.

**Which half of J++ is wrong?** Hybrids swap the input-column blocks. Top-1 on assistant tokens, L40 / 45 / 50 / 55 / 58:

| lens | top-1 |
|---|---|
| J++ | .13 / .22 / .25 / .39 / .69 |
| J++ with IT's **quiet** columns | .14 / .25 / .34 / .48 / .79 |
| J++ with IT's **loud** columns (IT loud + J++ quiet) | .22 / .36 / .35 / .48 / .83 |
| full IT | .21 / .37 / .45 / .55 / .85 |
| split-half IT estimates | .18–.19 / .34 / .43 / .55 / .84 |

- At L40–45 almost all of the gain comes from the loud columns.
- At L50–58 each block recovers roughly half; both are needed.
- Prose behaves the same way: the loud columns carry the gain at L40–45, and both blocks contribute at L50–55.

**Summary of absolute fidelity.**
- **What improves.** The IT-fitted lens repairs a large part of J++'s late-layer failure on the instruct model.
  - Top-1 on assistant tokens rises by 0.08–0.20 at L40–58.
  - KL falls by 1.4–4.6 nats.
  - It now clearly beats the naive logit lens at L45–58, where J++ was no better.
- **What does not improve.**
  - **Mid layers.** At L10–35 the IT lens is no better than J++. Probe noise is part of the reason, but the average Jacobian
    itself is incoherent there (split-half directional reliability 0.1–0.6).
  - **Absolute quality on prose.** Even calibrated, KL stays at 7–8 nats with top-1 ≤ 17% through L55. That remains poor:
    much of J++'s absolute-fidelity failure is intrinsic to linear average-Jacobian lenses, not to base→IT transfer.

---

## 5. Identity-condition gender comparison under the IT readout (`R1_07`, `R1_12`)

Setup:
- **Stimuli:** E1's 40 identity-neutral passages under 13 system conditions, teacher-forced.
- **Readouts:** taken at the assistant's own I/me/my tokens.
- **Scale:** woman persona = 1, man persona = 0, using the means of E1's two woman and two man personas, computed on
  passage means.
- **CIs:** paired passage bootstrap (2000 resamples).
- **The IT readout:** ⟨h_l, ḡ_l(d)⟩ with the mix fit, unless noted.
- **Neutral:** the mean of Atlas, Quill and Unit-7.

**Pronoun axis:**

| readout | band | default | nameless | neutral | Gemma | Sarah(AI) | James(AI) | # AI conds ≥ woman (of 9) |
|---|---|---|---|---|---|---|---|---|
| J++ | L30-39 | 0.46 [.44,.48] | 0.46 | 0.45 | 0.54 | 0.67 | 0.34 | 0 |
| J++ | L40-49 | 0.58 [.55,.62] | 0.58 | 0.57 | 0.70 | 0.82 | 0.42 | 0 |
| J++ | L50-58 | **0.96** [.85,1.07] | 1.02 | 1.01 | 1.17 | 1.20 | 0.76 | **6** |
| JLens | L50-58 | 0.80 [.72,.89] | 0.83 | 0.81 | 0.95 | 1.00 | 0.59 | 0 |
| **IT** | L30-39 | 0.49 [.44,.55] | 0.46 | 0.42 | 0.53 | 0.69 | 0.28 | 0 |
| **IT** | L40-49 | 0.49 [.42,.56] | 0.45 | 0.42 | 0.57 | 0.79 | 0.23 | 0 |
| **IT** | L50-58 | **0.76** [.64,.87] | 0.76 | 0.77 | 0.87 | 1.07 [.96,1.18] | 0.52 | **1** |
| IT, assistant-only fit | L50-58 | 0.54 [.42,.66] | 0.53 | 0.57 | 0.64 | 0.87 | 0.30 | 0 |

**Contrasts** (fraction of the woman–man span):

| contrast | readout | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|
| default − nameless | J++ | +0.01 [.00,.01] | +0.00 | −0.07 [−.09,−.05] |
| default − nameless | IT | +0.03 [.02,.04] | +0.04 [.03,.05] | −0.00 [−.03,.02] |
| default − neutral | J++ | +0.01 | +0.01 | −0.06 |
| default − neutral | IT | +0.07 [.06,.09] | +0.07 [.06,.09] | −0.01 [−.05,.02] |
| Gemma − nameless | J++ | +0.09 | +0.12 | +0.14 |
| Gemma − nameless | IT | +0.07 | +0.12 | +0.11 |
| default on the AI-anchored scale (James 0, Sarah 1) | J++ | 0.37 | 0.40 | 0.44 |
| default on the AI-anchored scale | IT | 0.52 | 0.47 | 0.44 |
| mean of the 9 AI conditions | J++ | | 0.61 | 1.04 [.93,1.15] |
| mean of the 9 AI conditions | IT | | 0.48 | 0.79 [.68,.90] |
| IT − J++, mean AI position | | | −0.13 [−.19,−.07] | **−0.24 [−.35,−.14]** |
| JLens − J++, mean AI position | | | +0.00 | **−0.20 [−.24,−.15]** |

**Full-matrix IT lens with the tool's own normalised, softcapped readout** (`R1_16`, `R1_17`).
- **Layers:** 30/35, 40/45 and 50/55/58.
- **Pronoun axis:**

  | band | default | nameless | neutral | Gemma | Sarah | James | mean AI | # AI ≥ woman |
  |---|---|---|---|---|---|---|---|---|
  | L40-49 | 0.34 [.28,.40] | 0.33 | 0.29 | 0.44 | 0.65 | 0.15 | 0.36 | 0 |
  | L50-58 | 0.61 [.51,.70] | 0.63 | 0.64 | 0.74 | 1.00 | 0.43 | **0.67** [.58,.76] | **0** |

- **J++ at the same layers:** 7 of 9 AI conditions ≥ woman, mean AI 1.04.
- **Default − nameless:** +0.01 [.00,.03] at L40–49 and −0.03 [−.04,−.01] at L50–58.
- **Late woman–man gap:** 1.84 logits, close to J++'s 1.99.
- **L30–39 is not interpretable** with this lens: the gap is only 0.67 logits and the estimate is noise-dominated (§3).
- **Rank-256 truncation:** it loses the persona gap entirely, which is consistent with §3.2.

The noun axis behaves similarly:
- **IT:** default 0.56 / 0.64 / 0.68. Default − nameless is +0.03 / +0.03 / −0.04, and no AI condition is ≥ woman.
- **J++:** default 0.44 / 0.37 / 0.33.
- **Full-matrix IT lens at L50–58:** default 0.50 and nameless 0.60, a contrast of −0.10 [−.12, −.07]. J++ at the same layers
  gives −0.07. This is the largest default − nameless contrast in the study, and it points toward *male*.

**Does default still sit with nameless and neutral AIs? Yes.**
- **Size of the gap:** under every IT readout (mix, chat, prose and assistant fits; pronoun and noun axes) default is within
  0.00–0.05 of the span of the nameless AI and within ≤0.08 of the neutral-name AIs at L30–58.
- **Ordering of conditions:** James < default ≈ nameless ≈ neutral < Gemma < Sarah. It is preserved, and the "Gemma" name
  effect is the same size as under J++ (+0.07 to +0.12).
- **Absolute positions depend on the fit:** default 0.27–0.59 at L30–49 depending on which positions the lens was fitted on.
  The contrasts do not.
- **J++ value replicated:** E1's J++ value (0.46 at L30–39) is reproduced exactly.

**Does the late pronoun-axis artefact disappear? Its J++-specific part does; a smaller shared offset remains.**
- **What changes under the IT readout:** at L50–58 only Sarah(AI) reaches the woman level, with a CI that includes 1. The mean
  AI condition moves from 1.04 to 0.79 (Δ −0.24 [−0.35, −0.14]).
- **Ordering:** man < James < default ≈ nameless ≈ neutral < Gemma < Sarah ≤ woman.
- **What does not change:** default − nameless goes to 0.00 [−.03, .02]. Under J++ it was −0.07; that small reversal was
  itself part of the artefact.
- **Why this is not mainly transfer:** the base-fitted JLens produces almost the same correction (−0.20 [−0.24, −0.15]), and
  no AI condition is ≥ woman under it. The "every AI reads as a woman at L50–58" picture is therefore mostly a property of the
  J++ fit, not of base→IT transfer.
- **Where the late persona gap lives:**
  - At L50–58 the woman–man gap is carried entirely by quiet coordinates, under both lenses.
  - Restricted to loud coordinates the gap is ≈0 or slightly reversed (−0.02 raw for both). This confirms V6 for J++ and
    extends it to the IT readout.
  - The two lenses weight the quiet coordinates differently (§2.3), so they place the AI conditions differently.
- **What remains under the IT readout:**
  - Every AI condition still sits on the female side of the midpoint at L50–58 (0.52–1.07; default 0.76).
  - Default is also higher than at L40–49 (0.49).
  - The size of this offset depends on the fit. The assistant-only fit gives default 0.54 and all AI conditions 0.30–0.87.
  - On the AI-anchored scale default is 0.44 under both J++ and IT, i.e. slightly male of the Sarah–James midpoint.
  - This matches the AI-vs-human / service-role offset of REPORT §1.3. It is a statement about the AI conditions relative to
    human personas, not a female self-reading.

---

## 6. Caveats

**What could not be separated**
- **Base→instruct vs quantisation.** "Base→IT" here means the base bf16 weights versus the instruct W4A16-QAT weights. Only one
  31B model fits on the GPU, and no base or bf16 checkpoint is on disk, so fine-tuning and quantisation are confounded. Every
  other candidate explanation tested for the quiet-subspace gap is small next to the gap: fitting data (prose vs chat),
  LRP-style linearisation, and cross-position terms.
- **J++'s exact recipe.** §2.4 tests two plausible readings of J++'s provenance, not its published implementation
  (https://github.com/safety-research/jpp_lens [4], R-Lens rules [5]), which was not consulted during the study; J++'s cluster
  weighting and readout token filter were also not tested. JLens's recipe is documented [2,3]: a total-derivative average
  (§2.4(b)); the Neuronpedia gemma-4-31b JLens was fitted on the base model with 861 WikiText-103 prompts [6,7].

**Scope and noise of the estimates**
- **The directional lens is an average same-position Jacobian.** By construction it inherits that object's limits: at L0–29 it
  is unreliable (split-half ≤ 0.63), and even at L50–58 it predicts the final readout only moderately (r ≈ 0.6).
- **The full-matrix estimate carries isotropic probe noise.** It is reliable only where the Jacobian is coherent (L45–58); §3
  and §4.2 report noise diagnostics and the split-half lenses beside every number.
- **The 16 "unit" rows replace only the largest output dims.** The RMS normalisation of the full-matrix IT readout uses them for
  the dims that dominate ‖h₅₉‖² and the probe estimate elsewhere.

**Stimuli and design**
- **Fit corpus:**
  - The prose class mixes model-written encyclopedic text with human-written documentation. The two classes give lenses at
    cos 0.89–0.93, so this mix matters little.
  - The chat fit uses default and neutral system prompts only. The identity conditions were held out.
- **Self-token evaluation is small.** The E3 self class has 86 positions in 12 passages, so its directional-fidelity CIs are
  wide.
- **Identity design.** It reuses E1's 40 passages, so it shares E1's caveats about the human-persona scale (AI-vs-human offset).
  The AI-anchored scale is reported for that reason.

## 7. Files

**Scripts** (`experiments/scripts/`)
- **Shared:**
  - `R1_common.py`: corpus definitions and constants.
  - `R1_vjplib.py`: batched exact VJPs, the multi-position pass, LRP patches, total-derivative VJPs.
- **Checks:** `R1_00_vjpcheck.py`, `R1_09_multicheck.py`.
- **Fit:**
  - `R1_01_corpus.py`: generation of the fit corpus.
  - `R1_02_prep.py`: directions, pullbacks, positions.
  - `R1_03_fit.py` and `R1_03_mkjobs.py`: directional fit plus probe accumulation, resumable.
  - `R1_09_fullmat.py`: Phase-B probes.
  - `R1_08_ystats.py`: spectrum of the normalised final states.
- **Evaluation jobs:** `R1_05_lrpsame.py`, `R1_05_total.py`, `R1_06_evaldir.py`, `R1_07_identity.py`, `R1_10_evalfull.py`,
  `R1_16_identity_full.py`.
- **Local analyses:** `R1_04_analyze_dir.py`, `R1_09b_fmnoise.py`, `R1_11_analyze_evaldir.py`, `R1_12_analyze_identity.py`,
  `R1_13_analyze_alt.py`, `R1_14_analyze_full.py`, `R1_15_spectrum.py`, `R1_17_analyze_identity_full.py`.

**Results** (`experiments/results/`)
- **Fit corpus and inputs:**
  - `R1_01_corpus_part{0,1,2}.json`: the fit corpus.
  - `R1_02_dirs.pt`: 128 directions with J++ and JLens pullbacks.
  - `R1_02_positions.json`: the fit positions.
- **Fit outputs:**
  - `R1_03_fit_ckpt.pt`: per-class/half VJP sums for 128 directions × 60 layers, plus Phase-A probe accumulators.
  - `R1_03_fit_job*.pt`: per-position stats, states and per-passage sums.
  - `R1_04_gbar.pt`: **the directional IT lens**. Mix / chat / prose / assistant-mode / per-class ḡ, [128, 60, 5376].
  - `R1_09_fm_ckpt.pt`: Phase-B accumulators. **Full-matrix IT lens:** (A₃ + A₉)/n, giving L_yᵀ at layers
    10, 15, …, 55, 58, by half.
- **Analysis outputs:** `R1_04_dir_analysis.json`, `R1_06_evaldir.pt`, `R1_07_identity_{00_20,20_40}.pt`,
  `R1_10_evalfull.pt`, `R1_16_identity_full.pt`, and the analysis summaries `R1_1{1..5,7}_*.json`.
- **Size:** R1 outputs total ≈ 15 GB.
  - `R1_03_fit_ckpt.pt` is 4.2 GB and `R1_09_fm_ckpt.pt` 2.5 GB. Both are needed only for the full-matrix lens.
  - `R1_04_gbar.pt` is 1.8 GB.
  - `R1_05_lrpsame_part*.pt` total 3.3 GB. Their results are summarised in `R1_13_alt_analysis.json`.
  - `R1_03_fit_job*.pt` total 1.9 GB.
  - The two ~1.5 KB `R1_03_fit_job0{6,7}_*` files are empty records from harmless duplicate submissions.

---

References: bracketed numbers refer to [`REFERENCES.md`](../../REFERENCES.md).
