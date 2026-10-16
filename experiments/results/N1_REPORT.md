# N1: are the Theodolite's static neuron-pole labels valid?

The labels compared are J-VINDEX, the naive lens, JLens, causal steering and activation-based labels, each against a permutation null.

Model: Gemma-4-31B-it, 4-bit QAT W4A16. GPU work went through the labd daemon. The exec log records about 6 min for N1A, 8.4 min for N1B, and one job of a few seconds for N1C.

Sample: the fixed N1 sample of 24 random GeGLU neurons at each of 8 layers (L10, 20, 30, 40, 45, 50, 55, 57). That is 192 neurons and 384 poles. Pole + means mlp_act > 0 and pole − means mlp_act < 0. The sample is in `N1_neurons.json`.

Parts: N1A measured the output side (static labels and causal steering). N1B measured the input side and the activation-based labels. N1C, this report, integrates and judges them.

Scripts are `experiments/scripts/N1{A,B,C}_*.py` and data is `experiments/results/N1{A,B,C}_*`.

Statistics:
- Every agreement number is shown next to its permutation null. The null pairs a pole's label with the measurement of each of the other 23 sampled neurons in the same layer, with the same sign. This is the exhaustive within-layer permutation.
- Lift is real − null.
- CIs are 95% bootstraps over poles (2000 resamples).
- `*` marks a lift whose CI excludes 0.

## Bottom line

**Verdict on J-VINDEX:**
- **Output labels at L40–57:** valid, but J++ adds nothing over the naive lens there.
- **Output labels at L10–30:** neat-looking but meaningless.
- **Input label (`gate_trigger_tokens`):** close to useless at every layer.

1. **Q1. Do J-VINDEX output labels predict causal output effects? Yes at L40–57, no at L10–30, and no better than the naive lens anywhere.**
   - The overlap metric is P@10: of the label's top-10 tokens, the fraction that fall in the causal top-100.
   - At L40/45/50/55/57, J-VINDEX scores 0.35/0.35/0.55/0.67/0.61, against nulls of 0.000–0.003. The Spearman lift over the full vocabulary is 0.35–0.75, which is 45–77% of the split-half reliability ceiling of the causal measurement.
   - At L10–30, P@10 is 0.000–0.002, equal to the null. There, single-neuron effects are generic (shared across neurons) and context-dependent, with a reliability ceiling of only 0.25–0.29.
   - J-VINDEX minus naive, paired: P@10 −0.004 [−0.019, +0.010] and rho +0.012. J++ helps slightly at L45 (+0.06) and hurts at L57 (−0.10). JLens is the same as J++.
   - **Prior-z scoring hurts:** P@10 changes by −0.076 [−0.092, −0.060]. The neuron-prior variant changes it by −0.044.
2. **The neatness of J-VINDEX labels at L10–30 is a property of the lens, not of the neuron (new in N1C).**
   - J++ turns *any* direction into a tight word cluster, such as "Elon, Tesla, aerospace, SpaceX, Drone" for an isotropic random vector at L20.
   - Tightness is measured as the mean pairwise embedding cosine of the top-10 tokens. At L10–30 it is 0.082 for J-VINDEX labels of real neurons and 0.071 for random 512-neuron mixtures. The difference is +0.011 [−0.005, +0.027], i.e. no difference.
   - J-VINDEX labels are tighter than naive labels for the same neurons (+0.029 [+0.011, +0.046]). That is why they look more interpretable, and the causal test gives no support for them.
   - Only at L40–57 are single neurons tighter than mixtures: +0.057 [+0.039, +0.075], for both J-VINDEX and naive. Here the neuron basis really is special, consistent with Goodfire.
3. **Q2. Are poles "say" neurons, i.e. do causal output labels match the tokens that actually follow when the pole fires? Mostly no.**
   - P@10 of causal labels against the empirical next-token labels is 0.041 vs a null of 0.028 (+0.013 [+0.005, +0.021]). The split-half ceiling of the empirical label itself is 0.249 vs 0.071.
   - Only L40 is clearly above null: 0.129 vs 0.067, with 35% of poles counting as "say" poles against 18% under the null.
   - The position-level alignment metric (ALIGN, defined in Methods) gives +0.5 percentile points [+0.2, +0.8], about 20% of its ceiling.
   - Static output labels do the same or worse.
   - So a valid output label says what the neuron pushes when it is amplified. It does not say what the neuron is used for in running text.
   - Clear trigger→effect neurons exist but are rare: 4 of 30 audited late poles. One example: L45 n4748+ fires on `super` and pushes `()`.
4. **Q3. Is `gate_trigger_tokens` meaningful? Barely.**
   - On attested tokens (corpus count ≥ 3), P@10 against the empirical input tokens is 0.063 vs a null of 0.043 (+0.020 [+0.012, +0.029]). The split-half ceiling is 0.288 vs 0.026, so the label recovers about 8% of the reproducible signal.
   - With strict full-vocabulary token ids (N1B) it recovers 1.5–2.6%, because 93% of its tokens are rare (fewer than 3 occurrences in the 147.5k-token corpus).
   - The whole-vocabulary ranking does carry a little signal at L20–45: ALIGN is +2.0 to +3.2 percentile points, against ceilings of 3.2–9.0.
   - At L10, the type-correct direct readout (W_gate·w_pre dotted with W_E) is far better: ALIGN +10.7 vs +0.2, which is at the split-half ceiling of 9.7.
   - The label is pole-agnostic, yet the two poles of a neuron fire on different inputs.
   - Null-calibrating it (prior-z) does not help: P@10 lift +0.017, ALIGN unchanged. Passing W_up through J++ (a pole-signed version) is at null.
5. **Q4. Bipolarity: on the input side the two poles are distinct jobs; on the output side, at L40–57, they are mirror images.**
   - Input: the poles' top input tokens overlap no more than those of two unrelated neurons (P@10 0.015 vs null 0.016). Token-level responses are anti-correlated: Spearman −0.36 to −0.54 at L20–57, against a null of about +0.03 (L10: +0.12).
   - Output at L40–57: the causal effect of − is the negation of the effect of + (top-10 of c+ inside the top-100 of −c−: 0.46–0.87 vs null ≤0.03; rho(c+, c−) −0.36 to −0.83).
   - Output at L10–30: no antipodal structure. The + and − effects are positively correlated (+0.15 to +0.28), because both share the generic response. That is still less than the +0.33 to +0.41 between unrelated neurons, so a weak antipodal component sits on top.
   - Most neurons are genuinely two-sided (N1B): 82% [77, 87] have a weaker pole reaching at least 30% of the stronger pole's maximum activation.
   - Per pole, the weaker pole is much less clean (see 6).
6. **Cleanliness audit (Goodfire-style; 48 random poles, 6 per layer; judged by me as a single rater).**
   - Activation side: 46% [33, 60] one coherent job, 12% two related jobs, 42% unclear.
   - **The pole's strength matters a lot.** Strong poles: 62% [43, 79] one job and 83% [64, 93] one or two. Weak poles: 29% one job and 67% unclear.
   - The closest comparison to Goodfire's preliminary Qwen3-32B figure of about 90% "one main job" is our 62% for strong poles. It is lower, with a small n, a single rater, and a much smaller and narrower corpus.
   - Output side at L40–57: 43% of causal top-token sets are coherent and 37% are partly coherent. Their link to the trigger is coherent in only 13% (4 of 30).

**Neat or pointless?**
- **Output side at L40–57:** neat and valid, but J-VINDEX is redundant with the naive lens there.
- **Output side at L10–30:** pointless, and worse than that because it is persuasive. The clean-looking word clusters come from J++ itself. Causal evidence is absent, and activation evidence is almost absent: semantic match with firing contexts is +0.03 at L20 and 0 at L30 (N1B), against a ceiling of +0.13. Token overlap with the input tokens is above null only at L20 (P@10 0.077 vs 0.026) and at null at L10 and L30.
- **Input side:** pointless as implemented.

## Methods

### Label types (all per pole p = (layer l, neuron i, sign s))
| name | definition | source |
|---|---|---|
| **JV** (J-VINDEX, the tool's label) | unembed(s·W_down[:,i]·w_post @ J++[l]ᵀ) | N1A cache; matches `inspect_bipolar_slot` (top-8 overlap 7–8 of 8) |
| NV (naive lens) | unembed(s·W_down[:,i]·w_post) | N1A |
| JL | same as JV, through JLens | N1A |
| JVz / NVz | z-scored per token against 1024 isotropic random vectors pushed through the same lens | N1A |
| **C** (causal) | mean over 36 contexts of logprob(steered) − logprob(base). Steering adds s·60·max\|act\| to mlp_act[l, last, i]; the baseline runs in the same forward pass | N1A (kA; kB = 180 gives the same picture) |
| **Enx** (empirical next token) | the tokens that follow the pole's top-1% positions (ranked by s·act, with s·act > 0), ranked by log-enrichment over corpus frequency, count ≥ 3 | recomputed from N1B activations (147,534 positions) |
| **Ein** (empirical input) | the same for the current token | N1B activations |
| **G** (the tool's `gate_trigger_tokens`) | unembed(W_gate[i] @ J++[l]ᵀ). Neuron-level, so the same for both poles | N1B static |
| Gz / Gemb / U | G z-scored against the isotropic J++ prior; W_gate·w_pre dotted directly with W_E; s·unembed(W_up·w_pre @ J++ᵀ) | N1B static, N1C |

### Vocabularies and metrics
- **FULL:** all 262,144 tokens. Used only for static and causal labels.
- **A:** the 5,544 "attested" tokens, those with corpus count ≥ 3. Every label is restricted to A before ranking, so labels can be compared with empirical labels on tokens that can actually occur.
- **P@10→100 (X→Y):** |top10(X) ∩ top100(Y)| / k, where k is the number of rankable tokens in top10(X).
- **Spearman:** computed over W5, the 3,471 tokens seen at least 5 times as both current and next token.
- **ALIGN:** the mean percentile, within the pole's label ranking over A, of the actual next token (or current token) at the pole's top-1% positions, minus the same mean over all positions. It uses the whole ranking rather than top-k lists.
- **Ceilings:**
  - Empirical labels estimated on one half of the documents and tested on the other half, for both P@10 and ALIGN.
  - For the causal rho, N1A's split-half reliability across contexts.

### Cleanliness audit
- Selection: 6 random poles per layer (seed 20261016).
- Order of judging: I judged part A first for all 48 poles, then part B.
  - Part A: the top-10 contexts plus 6 contexts sampled from the top 1%, with enriched input and next tokens.
  - Part B: causal top and bottom tokens, J-VINDEX, naive and gate labels.
- Rubric:
  - "one": at least 7 of the 10 top contexts share one describable feature.
  - "two": two features together cover 7 of 10.
  - "unclear": neither.
  - Each "one" also carries a confidence rating (high, medium or low).
- Judgments and rubric: `N1C_audit_judgments.json`. Scores: `N1C_audit.json`.

## Results

### T1. Output labels vs causal steering (Q1)
Cells are real/null.

| | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 | all |
|---|---|---|---|---|---|---|---|---|---|
| P@10 J-VINDEX→causal (FULL) | 0.002/0.001 | 0.000/0.000 | 0.000/0.001 | 0.354/0.003* | 0.346/0.000* | 0.550/0.000* | 0.673/0.000* | 0.608/0.001* | 0.317/0.001* |
| P@10 naive→causal (FULL) | 0.000/0.000 | 0.002/0.001 | 0.002/0.001 | 0.331/0.005* | 0.288/0.000* | 0.552/0.000* | 0.681/0.000* | 0.713/0.001* | 0.321/0.001* |
| P@10 JLens→causal (FULL) | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.354/0.004* | 0.325/0.000* | 0.562/0.000* | 0.688/0.000* | 0.658/0.001* | 0.323/0.001* |
| P@10 J-VINDEX prior-z→causal (FULL) | 0.000/0.000 | 0.000/0.000 | 0.000/0.001 | 0.235/0.003* | 0.279/0.000* | 0.456/0.000* | 0.460/0.000* | 0.494/0.001* | 0.241/0.001* |
| Spearman lift, J-VINDEX (N1A) | +0.021 | +0.011 | +0.042 | +0.349 | +0.597 | +0.628 | +0.748 | +0.738 | +0.392 |
| Spearman lift, naive (N1A) | +0.002 | +0.004 | +0.030 | +0.306 | +0.561 | +0.630 | +0.751 | +0.758 | +0.380 |
| reliability ceiling for rho | 0.25 | 0.27 | 0.29 | 0.78 | 0.85 | 0.92 | 0.98 | 0.99 | – |
| paired P@10, J-VINDEX − naive | +0.002 | −0.002 | −0.002 | +0.023 | +0.058 | −0.002 | −0.008 | −0.104 | −0.004 [−0.019, +0.010] |
| paired P@10, prior-z − raw | −0.002 | 0.000 | 0.000 | −0.119 | −0.067 | −0.094 | −0.212 | −0.115 | −0.076 [−0.092, −0.060] |

Other points:
- The post_ffn_norm scaling is essential. Without it, rho is 0.03 and P@10 is 0.002 (N1A).
- **A free confidence signal:** when the J-VINDEX and naive top-10 lists agree, the label is usually causally valid.
  - Within L40–57, their top-10 overlap predicts causal P@10 with within-layer Spearman 0.53–0.89.
  - Its AUC for predicting causal P@10 ≥ 0.3 is 0.84.
  - Layer alone (L ≥ 40) predicts validity with AUC 0.92.

### T2. Say test: do output labels name the tokens that follow when the pole fires? (Q2)
| | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 | all |
|---|---|---|---|---|---|---|---|---|---|
| P@10 causal→next (A) | 0.052/0.047 | 0.023/0.024 | 0.015/0.017 | 0.129/0.067* | 0.040/0.026 | 0.017/0.009 | 0.027/0.013 | 0.027/0.024 | 0.041/0.028* |
| P@10 causal→next, top-0.1% events | 0.046/0.032 | 0.023/0.021 | 0.019/0.017 | 0.119/0.053* | 0.042/0.020 | 0.029/0.015* | 0.031/0.012 | 0.031/0.019 | 0.042/0.024* |
| P@10 J-VINDEX→next | 0.038/0.042 | 0.027/0.026 | 0.033/0.037 | 0.085/0.032* | 0.033/0.013 | 0.017/0.010 | 0.029/0.010 | 0.056/0.032* | 0.040/0.025* |
| P@10 naive→next | 0.027/0.025 | 0.054/0.067 | 0.077/0.093 | 0.121/0.062* | 0.019/0.014 | 0.023/0.018 | 0.023/0.006 | 0.042/0.021* | 0.048/0.038* |
| ceiling (next-token label, half 0→half 1) | 0.217/0.076* | 0.240/0.071* | 0.275/0.108* | 0.294/0.053* | 0.206/0.063* | 0.215/0.076* | 0.248/0.070* | 0.298/0.053* | 0.249/0.071* |
| ALIGN_next, causal (percentile-point lift) | −0.0 | +0.4 | +0.4 | +0.6 | +0.1 | +0.4 | +1.2* | +0.8 | +0.5* |
| ALIGN_next, ceiling | +1.3* | +1.6* | +1.5* | +3.9* | +2.3* | +1.6* | +3.0* | +4.1* | +2.4* |
| "say" poles: ≥2 of 10 causal tokens in next top-100 (null) | 0.15 (0.11) | 0.02 (0.04) | 0.02 (0.03) | 0.35 (0.18) | 0.12 (0.05) | 0.02 (0.00) | 0.02 (0.01) | 0.04 (0.02) | – |

- Causal labels do not beat J-VINDEX against the next tokens: paired difference +0.001 [−0.009, +0.012].
- Both poles show the same small effect: + gives +0.015 and − gives +0.011.

### T3. Input side: `gate_trigger_tokens` and alternatives vs the current token at firing positions (Q3)
| | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 | all |
|---|---|---|---|---|---|---|---|---|---|
| P@10 gate (tool, J++)→input | 0.054/0.048 | 0.060/0.028* | 0.075/0.054* | 0.056/0.022* | 0.067/0.042* | 0.046/0.029* | 0.098/0.068* | 0.050/0.053 | 0.063/0.043* |
| P@10 gate prior-z→input | 0.021/0.010 | 0.065/0.019* | 0.056/0.039* | 0.035/0.009* | 0.033/0.023 | 0.006/0.007 | 0.044/0.025* | 0.029/0.021 | 0.036/0.019* |
| P@10 gate direct embedding→input | 0.094/0.011* | 0.048/0.008* | 0.025/0.022 | 0.019/0.024 | 0.063/0.038* | 0.027/0.027 | 0.023/0.028 | 0.027/0.034 | 0.041/0.024* |
| P@10 W_up via J++ (pole-signed)→input | 0.031/0.043 | 0.021/0.020 | 0.048/0.031* | 0.027/0.024 | 0.031/0.031 | 0.044/0.039 | 0.048/0.053 | 0.023/0.040 | 0.034/0.035 |
| ceiling (input label, half 0→half 1) | 0.298/0.012* | 0.192/0.010* | 0.229/0.043* | 0.369/0.036* | 0.233/0.012* | 0.279/0.036* | 0.342/0.026* | 0.363/0.032* | 0.288/0.026* |
| ALIGN_cur, gate (tool) | +0.2 | +2.7* | +2.8* | +3.2* | +2.0* | +0.6 | +1.1* | +0.5 | +1.6* |
| ALIGN_cur, gate direct embedding | +10.7* | +2.5* | +0.0 | +0.4 | +0.9 | +1.0* | +0.5 | −0.5 | +1.9* |
| ALIGN_cur, ceiling | +9.7* | +5.7* | +3.2* | +9.0* | +7.1* | +3.9* | +5.7* | +7.5* | +6.5* |
| P@10 gate→J-VINDEX, same neuron | 0.058/0.062 | 0.115/0.045* | 0.073/0.034* | 0.085/0.022* | 0.013/0.016 | 0.029/0.035 | 0.029/0.021 | 0.031/0.031 | 0.054/0.033* |

- The last row shows that the gate label and the output label of the same neuron overlap above null at L20–40. Both are read out through the same J++ map, which is consistent with a shared-lens artifact rather than input/output structure.
- N1B reports the same picture with other metrics:
  - Gate rows decoded through J++ have top-10 softmax mass of 0.40–0.90, the same as random vectors.
  - At L45–57, 24–34% of gate-label tokens are tokens that random vectors also decode to.
  - Token-response Spearman is 0.061, against a ceiling of 0.74.

### T4. Bipolarity (Q4)
The partner is the other pole of the same neuron. The null partner is the opposite-sign pole of another neuron.

| | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 | all |
|---|---|---|---|---|---|---|---|---|---|
| P@10 input(s)→input(−s) | 0.019/0.010 | 0.008/0.016 | 0.013/0.022 | 0.019/0.014 | 0.002/0.018 | 0.008/0.013 | 0.019/0.018 | 0.029/0.020 | 0.015/0.016 |
| P@10 next(s)→next(−s) | 0.031/0.019 | 0.019/0.017 | 0.046/0.034 | 0.038/0.031 | 0.013/0.018 | 0.033/0.022 | 0.027/0.026 | 0.023/0.030 | 0.029/0.025 |
| P@10 causal(s)→**negated** causal(−s) | 0.006/0.005 | 0.006/0.007 | 0.017/0.001* | 0.460/0.028* | 0.608/0.016* | 0.608/0.014* | 0.865/0.018* | 0.794/0.014* | 0.421/0.013* |
| Spearman causal(s) vs causal(−s) | 0.147/0.401 | 0.257/0.326 | 0.284/0.408 | −0.355/0.149 | −0.636/0.093 | −0.653/0.071 | −0.825/0.035 | −0.776/0.047 | −0.320/0.191 |
| Spearman input-token response(s) vs (−s) | 0.117/0.216 | −0.368/0.032 | −0.402/0.028 | −0.494/0.035 | −0.362/0.039 | −0.446/0.043 | −0.540/0.035 | −0.515/0.028 | −0.376/0.057 |
| ALIGN_next of causal(s) at the *other* pole's firing positions | +0.6 | −0.1 | +0.1 | −0.2 | +0.2 | +0.0 | −1.1* (CI < 0) | −1.0 | −0.2 |

- Each pole's own causal label also fits its own next-token response slightly better than the other pole's does: +0.013 [+0.006, +0.020] overall and +0.022 at L40–57.
- So at L55–57 the two poles behave weakly as one signed axis in natural text.
- Exceptions exist. L40 n9212 pushes "and" in many languages from *both* poles. Its + pole fires on list separators and its − pole on noun-phrase heads, and both kinds of context are typically followed by "and".

### T5. Label tightness: lens property or neuron property? (N1C, new)
Tightness is the mean pairwise cosine of the mean-centred input embeddings of the top-10 tokens. The null directions come from daemon job N1C_07.

| | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 |
|---|---|---|---|---|---|---|---|---|
| J-VINDEX, sampled neurons | 0.070 | 0.090 | 0.087 | 0.119 | 0.085 | 0.134 | 0.071 | 0.086 |
| J-VINDEX, 32 other single neurons | 0.051 | 0.092 | 0.090 | 0.079 | 0.111 | 0.095 | 0.048 | 0.066 |
| J-VINDEX, 512-neuron random mixtures | 0.043 | 0.080 | 0.090 | 0.088 | 0.033 | 0.055 | 0.025 | 0.009 |
| J-VINDEX, isotropic random vectors | 0.106 | 0.054 | 0.040 | 0.043 | 0.046 | 0.093 | 0.133 | 0.134 |
| naive, sampled neurons | 0.032 | 0.032 | 0.097 | 0.113 | 0.064 | 0.126 | 0.062 | 0.082 |
| naive, 512-neuron mixtures | 0.041 | 0.035 | 0.096 | 0.084 | 0.011 | 0.029 | 0.019 | 0.018 |
| causal top-10 | 0.020 | 0.015 | 0.057 | 0.160 | 0.084 | 0.149 | 0.067 | 0.089 |
| J-VINDEX: neurons − mixtures | +0.027 | +0.009 | −0.004 | +0.032 | +0.052* | +0.079* | +0.046* | +0.077* |
| neurons: J-VINDEX − naive | +0.039* | +0.058* | −0.011 | +0.006 | +0.021 | +0.009 | +0.009 | +0.004 |

- Isotropic vectors decode to tight but meaningless clusters: HTML tags at L10 and L55–57; "aerospace" or "children" clusters at L20.
- At L10–30, a random 512-neuron mixture gets a J-VINDEX label as tight as a real neuron's. Two examples of mixture labels:
  - L20: "Utah, Scientology, Mormon, Scorsese, Playboy".
  - L30: "queens, women, girls, female, goddesses".

### Agreement matrix (P@10→100 on vocabulary A; real/null)
Rows give the label's top 10 and columns the reference's top 100. Full per-layer matrices, including JLens, count-ranked empirical labels and the token-response labels, are in `N1C_matrix.json`.

**L40–57**

| X \ Y | J-VINDEX | naive | J-VINDEX z | causal | next (emp) | input (emp) | gate (tool) | gate-embed | W_up J++ |
|---|---|---|---|---|---|---|---|---|---|
| J-VINDEX | – | 0.786/0.020* | 0.942/0.019* | 0.589/0.020* | 0.044/0.019* | 0.047/0.020* | 0.037/0.024* | 0.020/0.018 | 0.023/0.024 |
| naive | 0.796/0.020* | – | 0.751/0.019* | 0.581/0.022* | 0.045/0.024* | 0.052/0.025* | 0.029/0.018* | 0.020/0.019 | 0.019/0.019 |
| causal | 0.601/0.020* | 0.580/0.023* | 0.575/0.020* | – | 0.048/0.028* | 0.044/0.027* | 0.028/0.020 | 0.018/0.019 | 0.020/0.019 |
| next (emp) | 0.041/0.018* | 0.040/0.019* | 0.039/0.015* | 0.042/0.020* | – | 0.153/0.025* | 0.033/0.022* | 0.023/0.022 | 0.020/0.021 |
| input (emp) | 0.041/0.020* | 0.038/0.020* | 0.041/0.019* | 0.036/0.020* | 0.073/0.010* | – | 0.032/0.019* | 0.019/0.019 | 0.019/0.020 |
| gate (tool) | 0.038/0.025* | 0.025/0.019 | 0.019/0.008* | 0.032/0.017* | 0.054/0.044* | 0.063/0.043* | – | 0.042/0.028* | 0.095/0.080 |

**L10–30**

| X \ Y | J-VINDEX | naive | J-VINDEX z | causal | next (emp) | input (emp) | gate (tool) | gate-embed | W_up J++ |
|---|---|---|---|---|---|---|---|---|---|
| J-VINDEX | – | 0.071/0.025* | 0.899/0.027* | 0.026/0.020 | 0.033/0.035 | 0.049/0.031* | 0.075/0.044* | 0.027/0.015 | 0.087/0.049* |
| naive | 0.070/0.025* | – | 0.088/0.026* | 0.026/0.019 | 0.053/0.062 | 0.057/0.056 | 0.020/0.025 | 0.012/0.011 | 0.033/0.023 |
| causal | 0.019/0.016 | 0.019/0.020 | 0.021/0.016 | – | 0.030/0.029 | 0.029/0.025 | 0.025/0.018 | 0.020/0.020 | 0.015/0.018 |
| next (emp) | 0.019/0.019 | 0.023/0.022 | 0.021/0.017 | 0.019/0.017 | – | 0.196/0.030* | 0.024/0.021 | 0.025/0.016* | 0.016/0.021 |
| input (emp) | 0.038/0.018* | 0.013/0.014 | 0.031/0.017* | 0.024/0.017 | 0.076/0.008* | – | 0.038/0.019* | 0.075/0.019* | 0.022/0.020 |
| gate (tool) | 0.082/0.047* | 0.026/0.025 | 0.043/0.018* | 0.023/0.019 | 0.052/0.048 | 0.063/0.043* | – | 0.021/0.016 | 0.065/0.057 |

How to read the matrices:
- **At L40–57** there are two clusters with nothing between them: the weight-and-steering cluster (J-VINDEX, naive, causal) and the activation cluster (input and next tokens). Agreement inside each cluster is 0.07–0.80; agreement across them is 0.04–0.05, against nulls of about 0.01–0.03.
- **At L10–30** the weight-and-steering cluster is gone (J-VINDEX vs naive 0.07; vs causal at null). J-VINDEX agrees above null mainly with the other J++-derived labels (gate 0.075/0.044, W_up 0.087/0.049), i.e. the shared lens. At L20 only, it also agrees weakly with the input tokens (0.077/0.026).

### Cleanliness audit
| group | one job | two related jobs | unclear | one (high or medium confidence only) |
|---|---|---|---|---|
| all 48 poles | 22 (46% [33, 60]) | 6 (12% [6, 25]) | 20 (42% [29, 56]) | 19 (40% [27, 54]) |
| strong pole of its neuron (24) | 15 (62% [43, 79]) | 5 (21%) | 4 (17%) | 14 (58% [39, 76]) |
| weak pole (24) | 7 (29% [15, 49]) | 1 (4%) | 16 (67% [47, 82]) | 5 (21%) |
| L10–30 (18) | 7 (39%) | 2 (11%) | 9 (50%) | 7 (39%) |
| L40–57 (30) | 15 (50%) | 4 (13%) | 11 (37%) | 12 (40%) |

Output side (causal top and bottom tokens):
- **L10–30:** 16 of 18 show only the generic response. The same junk tokens recur across unrelated neurons: ' documentación', ' razão', 'ABAD' at L10; 'gather/congregate' at L30.
- **L40–57:** 13 of 30 coherent (43% [27, 61]), 11 partly coherent, 6 incoherent. Examples: "and" in 15 languages, "()", Scandinavian function words, "twenty/thirty", "Malta", "Beijing", "on" (suppressing "off").
- **Trigger and effect form one computation in only 4 of 30 (13%):**
  - list separators → "and"
  - noun-phrase heads → "and"
  - `super` → "()"
  - Russian prose → "об"
  - In one further case the relation is inverse: the pole fires on gerunds and suppresses gerunds.
- **J-VINDEX labels at L10–30 vs firing contexts:** they matched the contexts semantically in 2 of 18 audited poles. L20 n1546− fires on pets and is labelled "marine, animals, Galapagos". L30 n16684+ fires on the Silk Road and is labelled "Russian, Siberian, Kazakh, Mongolian".

**Comparison with Goodfire (preliminary Qwen3-32B, about 90% one main job; Cammarata 2026, "Neural Networks Are Already Clean", Goodfire, https://already-clean.goodfire.com).** The closest comparison is our strong-pole figure of 62% [43, 79], or 83% for one-or-two. Our lower number should be read with these method differences in mind:
1. **Corpus:** 147.5k tokens, mostly technical READMEs and code. Many "unclear" poles may simply never meet their trigger in this corpus.
2. **Rater:** a single rater (me), not blind to layer or strength. The rubric is strict: 7 of 10 contexts.
3. **Units:** Goodfire's figure is also per pole [9], but its mix of weak and strong poles is unspecified; half of the poles audited here are the weaker side of their neuron (often dense, low-magnitude, or leak events).
4. **Steering:** our steering check is a context-averaged logit effect at 60× max activation.

The direction of the strong/weak split is robust. The absolute level is not comparable within a factor of about 1.5.

### 12 poles side by side
These are drawn at random from the 48 audited poles (seed 1612; one per layer plus 4 more), with no cherry-picking. "Activation label" is my summary of the top contexts.

| pole | activation label (top contexts) | empirical next tokens (enriched) | causal top tokens (steering) | J-VINDEX top tokens | gate_trigger_tokens | P@10 JV→C |
|---|---|---|---|---|---|---|
| L10 n10398− | one: token right after an 'in'/'IN' prefix (In-hale, IN-Send-Payment, IN-Restaurant) | 'Reservation', ' minor', 'Car', 'channels', ' scripts' | ' documentación', ' Wap', 'typeof', ' mandat', 'ర్ప', ' pooled' | ' Mga', 'Mga', ' Lena', ' Feb', ' Когда', 'г' | ' Ś', 'Ź', 'NEUTRON', ' Justiça', '嘌' | 0.0 |
| L10 n11313+ | one: philosophy/ethics discussion (utilitarianism, paradox, morally right action) | 'sle', 'arian', 'ilit', 'ism', 'om' | 'emics', ' documentación', '$_', 'ங்களாக', ' laminated', ' pooled' | '</h4>', '```', '</h5>', '”', '</footer>' | '.**', '.");', ':**', ' („', '.")' | 0.0 |
| L10 n20059+ | two: word after a variable name (pressure is, stop is None) + '[bug]' tags | ' rely', ' mouse', ' introduced', ' y' | ' timbers', 'throat', ' endowments', 'ference', ' depends', ' Cinemas' | '】', '</sub>', '﹞', ' Escrit', '```' | ' LCA', ' Champlain', ' Sask', ' dredging', ' expropri' | 0.0 |
| L20 n1546− | one: pets/animals in chat (cat kneading, teaching a dog to sit, turtle story) | 'classmethod', 'property', 'OS', 'Lock', 'ITA' | ' breathable', ' fighter', ' tennis', ' Hal', ' tubo', '))`' | ' Adirond', ' marine', ' Galapagos', ' Marine', ' animals', ' hewan' | 'Chicago', '刑', ' Chicago', ' Tesla', ' Arrest' | 0.0 |
| L30 n1711+ (weak) | unclear: dense weak pole on README text | '----------', 'Any', 'Linux', 'Returns', 'Build' | ' gathers', ' congregate', 'flip', ' gathered', ' gac', ' gabb' | ' californ', ' California', ' eventual', ' seaw', ' Californians' | '', '».', '</h6>', '\n\n\n…', ' Hickory' | 0.0 |
| L40 n20715+ (weak) | one (low confidence): start of the next item in an enumeration | 'next', 'prec', 'obj', 'top', 'attention' | ' смо', ' रीड', 'лян', ' ร', ' migrants', ' البعض' | ' innovator', ' escuela', ' AirPods', 'ReadLine', ' casque' | '</h3>', ' gemstone', 'యు', ' Mayan', ' ocup' | 0.1 |
| L45 n17699− (weak) | two: HTML align="center" + line ends in lorem-ipsum blocks | '-----------', 'center', '==========', 'Alias' | ' twenty', ' Twenty', 'Twenty', ' thirty', 'twenty', '二十' | '二十', ' twenty', ' Twenty', 'twenty', '三十', ' thirty' | '</span>', '</u>', '〕', '', '》，' | 1.0 |
| L45 n20186+ (weak) | unclear: mixed doc/code tokens | ' Sony', ' Plus', 'fault', 'utf', 'supp' | 'Starting', ' найбільш', 'starting', ' Starting', 'COMING', '而在' | ' androidTest', ' генерал', ' paymentRequest', 'ujian', ' bakery' | '</h3>', '</h5>', '</em>', ' */}', '</sub>' | 0.1 |
| L50 n14713− | one: Berlin Wall history (mostly one answer) | 'ball', ':`', 'ype', 'title', 'hidden' | ' malt', ' Malta', ' Maltese', ' Malt', 'Mal', 'malt' | ' Malta', ' Maltese', ' Malt', ' malt', 'malt', 'Mal' | ' ●', ' €', '●', '', '】，' | 1.0 |
| L55 n1280− (weak) | unclear: mixed (if and only if, Ada Lovelace) | ' depends', 'lass', 'gl', 'stack', 'Ref' | 'FERRED', ' डालें', ' judged', 'MainMenu', 'ígenes', 'mlp' | ' croix', ' diocese', ' কংগ্রেস', 'ي', ' isDisabled' | '『', ' $+', '<strong>', '＋', 'सीना' | 0.4 |
| L55 n4016− (weak) | unclear: Beartype, github.com/, CLI flags | 'min', 'ib', 'art', 'commands', 'join' | ' وهذا', 'ede', ' Rey', ' ngăn', ' 앉', ' commens' | ' Rey', ' commens', 'アクション', ' 앉', '앉' | '</sub>', ')】', '</b>', '\\]', '】' | 0.8 |
| L57 n316+ | two: list/bullet structure, hyphenated compounds, gerund items | 'itized', 'yield', 'Ride', 'fixes', 'AKE' | 'хие', 'ങ്ങൾ', 'ukkah', 'retch', 'algia', 'ちに' (suppresses: updating, playing, fixing) | '’', 'みたいな', '們', '似的', ' prostu' | runs of '\n\n\n…' | 0.1 |

Patterns across the table:
- **Late layers:** J-VINDEX reproduces the causal label ("twenty/thirty", "Malta"). Neither label resembles the contexts that make the pole fire: lorem-ipsum blocks, Berlin Wall.
- **Early layers:** causal labels are the generic L10/L30 junk.
- **Gate labels:** mostly markup or rare tokens at every layer.

## Recommendations for the tool
1. **Output labels:**
   - Show the naive lens instead of, or next to, J-VINDEX. They are equally valid and the naive lens is cheaper and has no lens artifacts.
   - Show labels only at L ≥ 40. At L < 40, mark vocabulary labels "no evidence; lens artifact likely" or hide them.
   - Add a free validity badge from the J-VINDEX/naive top-10 overlap (AUC 0.84 for predicting causal P@10 ≥ 0.3 at L40–57).
2. **Calibration:**
   - Do not re-rank labels by isotropic-prior z-scores. This costs 0.05–0.08 P@10 and does not rescue the gate label.
   - If the tool needs calibration, compute per-pole significance against a random-neuron or random-mixture null, and show label tightness next to the null tightness from T5. A user should be able to see that "aerospace, SpaceX, Drone" is what J++ outputs for a random vector at L20.
3. **Remove `gate_trigger_tokens`, which passes a read covector through J++ (a category error, N1B):**
   - Replace it with activation labels stored in the VINDEX vault: the top-k contexts and enriched input and next tokens for **each pole separately**. A one-off streaming pass over a corpus, keeping the top-k contexts for every pole of all 21,504 × 60 neurons, is feasible.
   - At L10–20, a cheap static fallback is the direct readout (W_gate·w_pre) · W_E (ALIGN +10.7 at L10, at the ceiling).
4. **Show both poles as separate units:**
   - Input: separate triggers, since they are disjoint.
   - Output: at L40–57, the − pole's output is the negation of the + pole's.
   - Flag weak poles. Most of them are dense or leak-dominated and unclear (67%).
5. **Add a one-click causal check** for the pole under the cursor:
   - One forward pass with ±k·max|act| added at the current position. Run the baseline in the same batch, because bf16 batch-size noise reaches KL 0.014.
   - Show the top Δlogprob tokens.
   - This is the only output label that works at every layer where an effect exists. It also shows when the effect is the generic L10–30 response; the same tokens recurring for every neuron is the giveaway.
6. **Filaments:** the reader-pole polarity is set by W_up, not W_gate.
   - At strong events the gate is almost always open (g > 0 in 72–100% of the top-30 events) and sign(act) = sign(u).
   - So a writer changes which pole of the reader fires through ⟨σ_i w_down, w_up⟩, while ⟨σ_i w_down, w_gate⟩ only opens or closes the reader.
   - Suggested coupling: use σ_j·⟨σ_i w_down, w_up_j⟩ for polarity, with the gate term as an on/off factor.
   - This follows from the measured mechanism but **is untested as a coupling score**; validate it with activation patching before shipping. W_up decoded through J++ is *not* a usable label (T3: at null).
7. **Keep the post_ffn_norm scaling.** Without it, labels are destroyed (rho 0.03).
8. **Gender demo:** check `query_nearest_edges` picks causally before displaying them. N1A found the L50 picks valid, the L40 picks mixed (one mislabel), and the L30 picks not gendered.

## Caveats
- **Causal effects (N1A):**
  - Context-averaged effects at the last position of 36 short contexts.
  - The delta is large and off-distribution: k = 60 × max|act| over about 1,000 positions.
  - At L10–30, effects are below or near reliability, so "no agreement" there means "no context-general vocabulary effect to label", not that the labels are wrong.
- **Corpus (N1B):**
  - 147.5k tokens, skewed toward technical docs and code. Chat is 9%.
  - Each chunk is at most 128 tokens.
  - Top-1% sets of sparse neurons include closed-gate leak events (13–33% of mass at L10/20/55/57).
  - Empirical labels and the audit inherit these limits.
- **ALIGN** uses percentiles over the 5,544 attested tokens. Its dynamic range is small: the next-token ceiling is +2.4 points. Read it relative to its ceiling.
- **Audit:**
  - n = 48, a single rater, judgments made by hand (stored in the repository so they can be re-rated).
  - Two of the "one job" late poles are single-document topics (Silk Road, Berlin Wall), which may be document detectors rather than concept neurons.
- **Correction to a brief fact:** single-neuron J-VINDEX distributions are *not* near-flat under the tool's own unembed (eps floor 1e-12): their top-10 softmax mass is 0.33–0.68, comparable to random vectors. The value of about 1e-4 came from labkit's eps = 1e-6 (both N1A and N1B).

## Files
- **N1C scripts** (`experiments/scripts/`):
  - `N1C_common.py` – shared helpers.
  - `N1C_01_labels.py` – builds all label types on a common vocabulary.
  - `N1C_02_matrix.py` – agreement matrix, nulls and Q1–Q4 extras.
  - `N1C_03_tables.py` – renders the matrix as tables.
  - `N1C_04_align.py` – position-level ALIGN metric.
  - `N1C_05_audit_dump.py` – audit evidence dump.
  - `N1C_06_audit_score.py` – audit scoring and the 12-row table.
  - `N1C_07_coherence_job.py` – daemon job: null directions.
  - `N1C_08_coherence.py` – label tightness.
  - `N1C_10_report_tables.py` – the report tables.
- **N1C results** (`experiments/results/`): `N1C_labels.npz` (with `N1C_labels_meta.json`), `N1C_matrix.json`, `N1C_per_pole.json`, `N1C_align.json`, `N1C_07_coherence.json`, `N1C_coherence.json`, `N1C_audit_picks.json`, `N1C_audit_judgments.json`, `N1C_audit.json`.
- **Inputs:** `N1A_metrics.json`, `N1A_labels.json`, `N1A_cache/` (fp32 causal vectors, static logits, priors), `N1B_labels.json`, `N1B_metrics.json`, `N1B_semantic.json`, `N1B_static.npz`, `N1B_acts/`, `N1B_corpus.jsonl`.

---

*Publication note:* verbatim activation contexts drawn from local system documentation, installed-package sources and package metadata were removed from `N1B_labels.json` (and the corpus `N1B_corpus.jsonl` is not distributed), because that text is third-party material. Model-generated chat contexts, single tokens, token counts and all statistics are kept. Regenerate the full contexts locally with `experiments/scripts/N1B_build_corpus.py` and `N1B_04_analyze.py`.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
