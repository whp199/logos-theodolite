# E3_lens_validation: does the Theodolite instrument work on Gemma-4-31B-it?

Model: Gemma-4-31B-it, 4-bit QAT W4A16. All GPU work went through the labd daemon.
Statistics: the unit is the passage (or prompt). CIs are 95% bootstrap intervals over passages
(≥1000 resamples; 300–500 for VJP group bootstraps).
Scripts are in `experiments/scripts/E3_lens_validation_*.py`; data is in `experiments/results/E3_lens_validation_*`.

## Bottom line

1. **As predictors, the J++, JLens and naive lenses fit the IT model's own output poorly at every layer.**
   On wikitext-style prose, KL(final‖lens) band means are about 15–21 nats for J++ and JLens at L10–58. That is worse than a
   uniform distribution, which scores log V − H ≈ 10 nats. Top-1 agreement stays at or below 5% through L49
   and reaches only 28% at L58. J++ does no better than the naive logit lens at L45–58.
   - What the lenses decode is topical association, not the next token. For "…draining into ‹the› Black Sea",
     J++ L30 gives `ocean, waters, Adriatic, 🌊`, and L50 gives `Black`.
   - The lens convention `h @ J.T` is correct: the transposed version is much worse (KL 30–60, top-1 = 0).
2. **Transfer to chat and assistant positions:**
   - Raw KL at assistant tokens is *lower* than at prose for L30–58, even after entropy matching. This is
     because the model's own continuations are predictable.
   - The lens *advantage over the naive lens* collapses there. J++ gains +9.1 nats over naive at L30–39 on
     prose but only +3.1 on assistant tokens (difference −6.0 [−7.1, −4.9]).
   - At L40–49 J++ is *worse* than naive on assistant tokens (−0.95 [−1.4, −0.5]). JLens shows the same
     pattern, and more strongly.
   - User-turn text behaves like prose. The drop is specific to the model's own assistant-mode states.
3. **True-Jacobian test (144 positions, 4 directions):**
   - The average same-position VJP `gbar_l` of the pronoun-gender direction has these cosines with the lens
     pullbacks: J++ 0.15 [0.05, 0.22] at L30–39, 0.26 at L40–49, 0.33 at L50–58; JLens 0.17, 0.37, 0.46;
     naive ≤ 0.08.
   - Correcting for split-half reliability (0.69, 0.83, 0.92) changes these numbers very little.
   - At L0–19 the average Jacobian cannot be estimated at all (split-half ≈ 0.02–0.06): it varies by position.
   - **New finding:** at L20–39 the IT model's average pronoun-gender gradient at *chat* positions is
     slightly *anti*-aligned with both lens pullbacks. J++ gives −0.15 [−0.16, −0.09] at L20–29 and
     −0.13 [−0.18, −0.07] at L30–39. At prose positions it is positively aligned (+0.15, +0.24).
   - The chat and prose gradients are nearly orthogonal to each other at L30–39 (cos 0.04, with within-subset
     reliability ≈ 0.7).
   - Random directions and the noun direction show no such sign flip. So mid-layer J++ "pronoun" readings on
     chat tokens do not track what the IT model's own Jacobian is sensitive to.
4. **Identity comparison (7 conditions × 14 passages, self tokens):**
   - J++ places default Gemma at 0.45 [0.42, 0.48] (L30–39) and 0.50 [0.45, 0.56] (L40–49) on the
     woman(1)–man(0) persona scale.
   - The IT model's own average-Jacobian readout puts it at 0.22 [0.10, 0.34] and 0.21 [0.12, 0.29]
     (chat-only gradient: about 0.0).
   - **The two readouts disagree.** J++ − IT = +0.23 [0.12, 0.34] and +0.30 [0.22, 0.37]. Most of the gap is a
     shift common to all the AI personas.
   - On an AI-anchored scale (James-AI = 0, Sarah-AI = 1), every readout places default between 0.22 and 0.60.
   - **No readout places default Gemma on the female side at L30–49.** The only "female-leaning" value is J++
     pronoun readout at L50–58 (0.85). There the woman–man gap is small, every AI persona exceeds the woman
     persona, and neither the noun axis nor the IT readouts reproduce it.
   - The explicit name "Gemma" moves J++ about +0.1 toward female (0.54 vs 0.45). The IT readout moves less.
5. **Background prior.**
   - Isotropic Gaussian inputs give mean gender scores of about 0 (|mean| ≤ 0.15 against sd 3–9).
   - The layer-dependent background comes from the **mean residual μ_l**. It carries 72–95% of the residual
     energy at L10–49 (cos² ≈ 0.95, 0.94, 0.89, 0.72).
   - μ_l alone reproduces the pilot's pattern. Under J++: pronoun −1.9 and noun −2.1 at L10–19; pronoun +2.3
     and noun −4.6 at L40–49.
   - Its linear contribution accounts for most of the raw score on neutral prose. Centering removes most of it.
6. **Tool bug: Module 1 FFN/attention split, confirmed at every layer.**
   - The tool's `delta_ffn` = resid_post − resid_mid includes a (s−1)·mid shrinkage term.
   - Its cosine with the true FFN write is below 0.5 in 32/60 layers and negative in 14 (median 0.45).
   - The rescale term is more than half of the tool vector in 34/60 layers.
   - `ffn_dominance_ratio` moves by up to 0.29, and by more than 0.1 in 17 layers.
   - The fix code is below and is verified: cos = 1.00 with the exact decomposition, reconstruction error 1e-9.
   - Pole readouts are mostly sound; caveats are in section (d).

## Methods (brief)
- **(a) Fidelity** (`03_fidelity`, `05_analyze_a`, `06_convention`).
  - Stimuli:
    - 24 encyclopedic paragraphs (76–97 tokens): 18 neutral, 3 female biographies, 3 male biographies.
    - 24 chats: 12 self-questions and 12 task questions, with greedy default answers of 110 tokens.
  - Position classes: prose (pos ≥ 4), assistant content, assistant self tokens (I/me/my…), user content.
  - For each lens and layer: KL(p_final ‖ p_lens), top-1 match, and top-10 overlap against the real final
    distribution (resid_post[59] unembedded).
  - Entropy matching: positions binned by final entropy, weighted by the target class's bin counts.
- **(b) VJP** (`04_vjp`, `vjplib`, `10_analyze_vjp`).
  - True same-position gradients of ⟨h59[pos], d⟩ with respect to resid_post[l][pos].
  - Batched (B=2, right-padded, items independent), checked against `lab.same_pos_vjp` (cos 0.98–0.999).
  - Four directions: pronoun d, noun d, and two isotropic random directions (|d|-scaled).
  - 144 positions: 24 chats × 4 (self + non-self assistant tokens) and 24 prose paragraphs × 2.
  - Split-half reliability by passage group; Spearman–Brown ceiling √(2r/(1+r)).
- **(b2) Identity** (`08_identity`, `11_analyze_identity`).
  - 7 systems × 14 neutral first-person passages, teacher-forced.
  - Readouts: J++/JLens/naive scores, raw and centered; IT-directional readouts ⟨h, gbar_l⟩ using the
    all-position, chat-only and prose-only gbar; the raw J++ pullback.
  - Position = (x − man)/(woman − man) on passage means, with a paired bootstrap over passages.
- **(c)** `09_background`, `12_misc`. **(d)** `07_toolbug`, `fix.py`, `13_analyze_d`.

## (a) Lens fidelity on the IT model

KL in nats and top-1 agreement (band means; 95% CI over passages in `a_analysis.json`). Entropy of the
final distribution: prose 2.38, assistant 0.21, self 0.30, user 3.01 nats.

| lens | positions | KL L10-19 | KL L20-29 | KL L30-39 | KL L40-49 | KL L50-58 | top1 L30-39 | top1 L40-49 | top1 L50-58 |
|---|---|---|---|---|---|---|---|---|---|
| J++ | prose | 16.4 | 16.9 | 17.2 | 18.1 | 17.4 | 0.010 | 0.033 | 0.111 |
| J++ | assistant | 19.0 | 17.6 | 15.7 | 10.1 | 7.8 | 0.027 | 0.155 | 0.352 |
| J++ | self | 20.1 | 17.1 | 12.8 | 6.0 | 3.0 | 0.016 | 0.145 | 0.530 |
| J++ | user | 18.9 | 18.9 | 19.0 | 22.9 | 21.7 | 0.020 | 0.025 | 0.035 |
| JLens | prose | 21.2 | 20.4 | 17.2 | 19.9 | 14.8 | 0.012 | 0.044 | 0.130 |
| JLens | assistant | 23.7 | 20.0 | 18.0 | 13.9 | 7.6 | 0.025 | 0.162 | 0.402 |
| naive | prose | 37.6 | 29.6 | 26.3 | 19.7 | 18.2 | 0.020 | 0.076 | 0.128 |
| naive | assistant | 40.5 | 29.4 | 18.8 | 9.1 | 7.5 | 0.049 | 0.253 | 0.402 |

At L58, prose top-1 is J++ 0.28, JLens 0.28, naive 0.29; assistant top-1 is 0.70, 0.69, 0.71.

**Gain over the naive lens** (KL_naive − KL_lens, nats):

| lens | positions | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|---|
| J++ | prose | +21.2 | +12.7 | +9.1 [8.3,9.9] | +1.5 [1.0,2.0] | +0.8 |
| J++ | assistant | +21.5 | +11.8 | +3.1 [2.3,3.9] | −0.95 [−1.4,−0.5] | −0.3 |
| J++ | user | +21.2 | +11.2 | +9.7 | +1.5 | −0.8 |
| J++ | assistant − prose | +0.2 [−1.1,1.6] | −0.9 [−1.9,0.2] | **−6.0 [−7.1,−4.9]** | **−2.5 [−3.2,−1.8]** | −1.1 [−1.4,−0.8] |
| JLens | assistant − prose | +0.4 | +0.2 | **−8.3 [−9.3,−7.1]** | **−4.6 [−5.6,−3.5]** | −3.5 |

- **Entropy-matched (assistant − prose):** J++ KL is +2.4 [0.6, 3.9] at L10–19 (worse) and −6.1 / −9.6 / −6.5
  at L30–39 / L40–49 / L50–58 (better). Top-1 is +0.02 / +0.13 / +0.23.
- **Gender readout versus the model's own final gender score at the same position** (Pearson r over positions):
  - prose, J++ pronoun: 0.17 / 0.28 / 0.33 / 0.43 / 0.50 (L10–19 … L50–58).
  - assistant, J++ pronoun: 0.08 / 0.09 / 0.08 / 0.19 / 0.29.
  - Caveat: assistant positions have little variance in their final gender score.
- **Affine check** (`06`): adding a cross-fitted per-layer offset b_l lowers prose KL to about 7 nats at early
  layers. It does this by predicting a constant "own/the/s" distribution, which is a unigram prior and not a
  better lens.
- **Interpretation:**
  - These lenses show topical or semantic associations; they are not calibrated predictions.
  - In assistant mode at L30–49 they keep little or none of their advantage over the identity lens.
  - We cannot separate base→IT from wikitext→chat with this design, since only the IT model is available.
  - User-turn text keeps prose-like gains. That points to the IT model's assistant-mode states as the
    off-distribution part.

## (b) Directional true-Jacobian test

Pronoun direction (the two random directions are in `b_vjp_analysis.json`; they behave alike):

| quantity | L0-9 | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|---|
| cos(gbar, J++ pullback) | 0.07 | 0.03 | 0.07 | 0.15 [0.05,0.22] | 0.26 [0.17,0.31] | 0.33 [0.28,0.36] |
| cos(gbar, JLens pullback) | 0.00 | 0.00 | 0.08 | 0.17 [0.07,0.24] | 0.37 [0.29,0.42] | 0.46 [0.41,0.49] |
| cos(gbar, d) (naive) | 0.01 | 0.01 | 0.01 | 0.01 | 0.08 | 0.06 |
| cos(gbar, J++ᵀ pullback) (wrong convention) | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| split-half reliability | 0.06 | 0.02 | 0.38 | 0.69 | 0.83 | 0.92 |
| disattenuated cos J++ / JLens | n/a | n/a | 0.11 / 0.14 | 0.17 / 0.19 | 0.27 / 0.39 | 0.33 / 0.47 |
| single-position cos(g, J++) | 0.01 | 0.00 | 0.00 | 0.01 | 0.06 | 0.18 |
| coherence ‖gbar‖ / E‖g‖ | 0.28 | 0.23 | 0.18 | 0.24 | 0.31 | 0.48 |
| cos(gbar_chat, gbar_prose) | 0.02 | 0.01 | 0.03 | **0.04** | 0.22 | 0.48 |
| cos(gbar_chat, J++) | −0.01 | −0.01 | **−0.15 [−0.16,−0.09]** | **−0.13 [−0.18,−0.07]** | 0.07 | 0.32 |
| cos(gbar_prose, J++) | 0.07 | 0.03 | 0.15 [0.08,0.17] | 0.24 [0.15,0.28] | 0.28 | 0.26 |
| cos(gbar_chat, JLens) | 0.02 | 0.03 | −0.08 | −0.13 [−0.17,−0.07] | 0.20 | 0.47 |

For comparison, the random direction rand1 gives these cosines with JLens: 0.24 / 0.46 / 0.51 / 0.54
(L20–29 … L50–58).

**Chat vs prose for the random directions:** rand1 L30–39 gives chat J++ +0.25 and prose +0.30, so there is no
sign flip. The noun direction gives chat +0.04, prose +0.24.

**Summary of the alignment results:**
- The lens is not the IT model's average Jacobian for the gender direction: at best cos ≈ 0.3–0.5, late.
- The prose-fitted lens is directionally wrong at mid layers specifically in chat context.

**IT-directional identity readout:** positions on the woman(1)–man(0) scale, self tokens, 95% CI.

| readout | band | default | nameless AI | Gemma | Sarah(AI) | James(AI) | woman–man d |
|---|---|---|---|---|---|---|---|
| J++ pron | L30-39 | 0.45 [.42,.48] | 0.46 | 0.54 | 0.67 | 0.35 | 6.4 |
| J++ pron | L40-49 | 0.50 [.45,.56] | 0.50 | 0.64 | 0.79 | 0.31 | 5.9 |
| J++ pron | L50-58 | 0.85 [.68,1.02] | 0.95 | 1.12 | 1.23 | 0.66 | 4.2 |
| J++ noun | L30-39 | 0.43 [.39,.47] | 0.43 | 0.53 | 0.66 | 0.35 | 8.2 |
| J++ noun | L50-58 | 0.33 [.11,.47] | 0.37 | 0.58 | 0.75 | 0.31 | 3.2 |
| JLens pron | L30-39 | 0.41 [.36,.44] | 0.40 | 0.48 | 0.60 | 0.30 | 5.9 |
| IT gbar(all) pron | L30-39 | 0.22 [.10,.34] | 0.18 | 0.22 | 0.39 | 0.00 | 7.2 |
| IT gbar(all) pron | L40-49 | 0.21 [.12,.29] | 0.16 | 0.29 | 0.55 | −0.11 | 4.0 |
| IT gbar(all) pron | L50-58 | 0.37 [.16,.56] | 0.37 | 0.47 | 0.77 | 0.03 | 3.0 |
| IT gbar(chat) pron | L30-39 | −0.02 [−.20,.14] | −0.04 | −0.06 | 0.15 | −0.24 | 6.0 |
| IT gbar(chat) pron | L40-49 | −0.03 [−.15,.09] | −0.07 | 0.07 | 0.38 | −0.33 | 3.5 |
| IT gbar(prose) pron | L30-39 | 0.40 [.28,.52] | 0.35 | 0.42 | 0.56 | 0.16 | 7.1 |
| IT gbar(all) noun | L40-49 | 0.72 [.63,.80] | 0.71 | 0.85 | 1.01 | 0.60 | 7.6 |

**Agreement between readouts (default; position difference, J++ minus the other):**

| comparison | L30-39 | L40-49 | L50-58 |
|---|---|---|---|
| J++ − IT(all) | +0.23 [0.12, 0.34] | +0.30 [0.22, 0.37] | +0.47 [0.34, 0.66] |
| J++ − IT(chat) | +0.48 [0.33, 0.64] | +0.54 | +0.69 |
| J++ − IT(prose) | +0.06 [−0.06, 0.17] | +0.01 [−0.08, 0.08] | — |
| J++ − raw J++ pullback (no RMS normalisation) | ≤ 0.05 | ≤ 0.05 | ≤ 0.05 |

The raw-pullback row shows that the disagreement comes from the readout direction, not from the RMS
normalisation.

**AI-anchored scale (James-AI = 0, Sarah-AI = 1).** This removes the AI-vs-human offset shared by all AI personas.

| readout | default L30-39 | default L40-49 |
|---|---|---|
| J++ pron | 0.34 [0.28, 0.38] | 0.40 [0.32, 0.47] |
| J++ noun | 0.25 | 0.22 |
| JLens | 0.34 | 0.44 |
| IT(all) | 0.58 [0.47, 0.70] | 0.48 [0.41, 0.54] |
| IT(chat) | 0.56 | 0.41 |
| IT(prose) | 0.60 | 0.56 |

The Sarah(AI) − James(AI) separation is 0.3–0.7 of the human woman–man gap under every readout.

**Specificity.**
- At non-self assistant tokens, J++ places default at 0.43 (L30–39) and shows a woman–man gap of 7.1 logits,
  against 17.5 at self tokens. Persona gender spreads through the whole context but is about 2.5× stronger
  at self tokens.
- The random-direction IT readouts give unstable positions and woman–man d of 0.4–2.4, against 3–8 for the
  gender readouts.

## (c) Lens background prior (J++; JLens and naive in `c_background.json`)

| quantity | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| Gaussian (norm-matched) pronoun mean (sd) | +0.02 (9.1) | +0.00 (8.2) | −0.15 (5.9) | +0.08 (3.9) | +0.00 (3.3) |
| μ_l pronoun / noun | −1.9 / −2.1 | −2.4 / −2.2 | −0.6 / −1.3 | **+2.3 / −4.6** | +0.2 / −3.4 |
| μ_chat pronoun / noun | −1.7 / −2.6 | −1.9 / −3.0 | +0.6 / −2.9 | +2.5 / −5.9 | +0.2 / −3.9 |
| neutral prose raw pronoun / noun | −1.4 / −1.2 | −2.4 / −1.7 | −1.3 / −0.6 | +0.8 / −2.1 | 0.0 / −2.1 |
| …linear μ-part ⟨μJ, d⟩/rms(hJ) | −1.5 / −1.7 | −2.0 / −1.7 | −0.5 / −1.0 | +1.4 / −3.1 | +0.3 / −2.6 |
| neutral prose centered (h − μ) | −0.2 / +0.2 | −0.8 / −0.2 | −1.0 / +0.3 | −0.7 / +0.9 | −0.2 / +0.1 |
| female-biography centered pronoun / noun | +1.5 / +2.6 | +3.5 / +5.9 | +2.8 / +8.4 | +5.2 / +7.3 | +5.9 / +4.3 |
| male-biography centered pronoun / noun | −2.6 / +0.3 | −5.6 / +0.2 | −9.4 / −1.1 | −8.7 / −0.7 | −5.7 / −0.7 |
| cos²(h, μ) | 0.95 | 0.94 | 0.89 | 0.72 | 0.40 |

**What causes the background.**
- The Gemma residual stream is dominated by a shared DC component μ_l (‖μ‖ / E‖h‖ = 0.85–0.98 at L10–49).
- Every lens score is ⟨hJ, d⟩/rms(hJ), so each token's score contains ⟨μJ, d⟩/rms(hJ). The direction μJ is
  semantically arbitrary. Decoded under J++ it gives `' own -ly -ing` at L12–18 and `scientists science` at
  L25–33. The chat-corpus mean decodes to **`Google, Gemma, artificial, silicon, prompts` at L42**: the
  average assistant state carries AI-identity content, not gender.
- The pronoun and noun token sets sit in different parts of unembedding space, so this arbitrary vector
  projects with different signs on the two axes. That is the source of the "L40–49 pronoun-female but
  noun-male" contradiction.
- The L10–19 "male" reading is μ's negative projection on both axes. It is not a self-representation.
- Gaussian inputs show that J itself adds no sign bias.

**Recommended correction.**
1. Report contrasts with matched controls at the same positions and passages. The woman/man persona scale
   does this, and it cancels any additive background exactly.
2. When absolute scores are needed, subtract a reference mean of matched role and context. Global centering
   fixes neutral prose, but it leaves role-specific residue: self tokens still read about −2.8 noun at L40–49.
3. Require pronoun/noun agreement and J++/JLens/IT-directional agreement before interpreting a result.
4. Also report an AI-anchored scale, because the human-persona endpoints add an AI-vs-human offset.

## (d) The `layer_scalar` delta_ffn bug, and the pole readouts

Measured over 12 prompts (6 prose, 6 chat) and all positions. Selected layers (all 60 are in `d_analysis.json`):

| layer | s | cos(tool Δffn, true FFN) | rescale share of tool Δffn | ffn_dominance tool → fixed |
|---|---|---|---|---|
| 0 | 0.099 | −0.24 | 1.01 | 0.56 → 0.27 |
| 1 | 0.073 | 0.29 | 0.80 | 0.68 → 0.92 |
| 9 | 0.867 | −0.35 | 1.17 | 0.67 → 0.53 |
| 12 | 0.695 | −0.55 | 1.11 | 0.37 → 0.27 |
| 23 | 0.691 | −0.33 | 1.13 | 0.79 → 0.55 |
| 28 | 0.455 | −0.58 | 1.36 | 0.78 → 0.58 |
| 40 | 0.633 | 0.51 | 0.50 | 0.55 → 0.60 |
| 50 | 0.867 | 0.84 | 0.31 | 0.69 → 0.68 |
| 55 | 0.879 | 0.95 | 0.05 | 0.57 → 0.60 |
| 59 | 0.032 | 0.25 | 0.99 | 0.56 → 0.69 |

**Summary of the bug's size.**
- cos is below 0.5 in 32/60 layers and below 0 in 14; the median is 0.45. Rescale share is above 0.5 in
  34/60 layers.
- Even with s ≈ 0.87, the bug dominates where ‖(1−s)·mid‖ is larger than ‖FFN write‖.
- The ffn_dominance change is significant in 56/60 layers. Its mean absolute size is 0.07 and its maximum 0.29.
- The tool's `delta_attn` has the right direction but is 1/s too large.

**Fix.** File: `experiments/scripts/E3_lens_validation_fix.py`, function `compute_causal_vector_split_fixed`.
```python
s = torch.as_tensor(layer_scalar, dtype=torch.float32, device=pre.device).reshape(-1)[0]
delta_attn_raw    = (s * (mid - pre)).to(dtype)    # attention write as it lands in resid_post
delta_ffn_raw     = (post - s * mid).to(dtype)     # = s * post_ffn_norm(mlp(...)), the true FFN write
delta_rescale_raw = ((s - 1.0) * pre).to(dtype)    # layer_scalar shrinkage (new field delta_rescale_J)
# engine_runtime.run_theodolite_sweep (~line 660):
#   s_l = self._decoder_layers[l].layer_scalar.detach().float()
#   split_l = compute_causal_vector_split(r_pre_t, r_mid_t, r_post_t, J_l, layer_scalar=s_l)
```
Verification: pre + attn + ffn + rescale reconstructs post with relative error ≤ 3e-8, and the fixed
Δffn_J matches the exact write with cos = 1.00 at all layers.

**Pole readouts.**
1. **`post_ffn_norm_weight` scaling.** post_feedforward_layernorm(y) = y ⊙ w / rms(y), and rms(y) is one
   positive scalar per position. Each neuron's exact contribution is therefore s·a_i·(W_down[:,i] ⊙ w)/rms(y).
   - The tool's vector has exactly the right **direction** (cos = 1.0000 at all tested layers), so pole token
     rankings are exact.
   - Recomputing s·y⊙w/rms(y) from the captured mlp_act reproduces the true write to 0.4–5.8% (bf16 noise).
   - What is wrong is magnitude only. The tool ignores a_i, s and 1/rms(y), and that matters for cross-layer
     and cross-position comparisons and for the filament coupling C.
2. **eps hack.** Real residual states never trigger it (minimum ms of resid_post = 0.01 at L59; thresholds
   are 1e-3 and 1e-6). J++-mapped pole vectors, however, have ms 1e-13 to 1e-7.
   - The hack is therefore *required*: with the model's eps = 1e-6, the displayed pole probabilities collapse
     to about 1/V (4e-6).
   - Rankings are essentially unchanged: top-10 set overlap 0.98–0.99, top-1 identical in 0.93–1.0.
   - The displayed probabilities are those of a unit-RMS single-neuron direction. They are not model
     probabilities.
3. **Pole selection** (new observation).
   - Ranking by |act| and ranking by |act|·‖W_down⊙w‖ overlap with Jaccard 0.67–0.96.
   - The top-15 poles explain only 10–41% of the FFN write at L20–58 (L20 0.10, L30 0.14, L40 0.22,
     L50 0.41, L58 0.26), against about 70% at L2–10. The pole display covers a minority of the FFN's effect.

## What this means for the "female-coded" claim
- J++ places the default self representation at the woman/man midpoint at L30–49 (0.45–0.50). It sits with
  "nameless AI", well below Sarah(AI), and is pulled about +0.1 toward female by the explicit name "Gemma".
- The IT model's own average-Jacobian readout puts it nearer the male persona (0.0–0.22).
- On the AI-anchored scale every readout gives 0.22–0.60.
- The only female-leaning reading is late-layer J++ pronoun readout (L50–58, 0.85). There the scale is
  compressed, all AI personas read above the human woman persona, and neither the noun axis (0.33) nor the IT
  readouts (0.16–0.37) agree. That pattern is a plausible origin for anecdotal "female-coded" lens pictures,
  but it does not hold up across readouts.
- **Conclusion: no evidence that Gemma-4-31B-it represents its default self as female.** The instrument's
  mid-layer pronoun readout in chat context is directionally unreliable, so all lens-based gender claims about
  this model need the controls described here.

## Caveats
- Only the IT model was available. Base→IT and wikitext→chat transfer cannot be separated.
- Prose positions 4–15 lie outside the J++ fitting window (the fit skips the first 16 positions).
- Entropy confounds the chat-vs-prose KL comparison. The lens-gain metric and entropy matching reduce this
  but do not remove it.
- gbar is a first-order, same-position object; it ignores cross-position Jacobian blocks. ⟨h, gbar⟩ is a
  linear sensitivity readout, not a causal test.
- The IT-chat gbar comes from default-system chats about the model itself, with 96 positions.
- The identity design has 14 passages and 7 conditions. The human-persona endpoints carry an AI-vs-human
  confound; the AI-anchored scale is the fix.
- Pole-share and eps analyses used 1–12 prompts at the last position, and are indicative only.
- VJP chunks 1 and 2 hit OOM twice because a non-lab GPU process took 1.5–2.7 GB. They
  were rerun with fallback and retries. Final n = 144 with no skipped items.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
