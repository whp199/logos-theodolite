# N1V robustness re-run: are the static neuron-pole labels valid? (verifier notes)

**Role:** adversarial robustness verifier for the N1 integrator verdict (`N1_REPORT.md`).
**What changed relative to N1A/N1B/N1C:**
- a fresh neuron sample (seed 99, 8 per layer, disjoint from the N1 sample);
- 32 new hand-written contexts;
- a different intervention style (the delta is added at the last 8 positions);
- four more causal conditions, to separate sample effects, intervention style, magnitude and the post-FFN RMSNorm.

**Bottom line:**
- The integrator's central verdict survives:
  - Output labels are causally valid at L40–57 and useless at L10–30.
  - Prior-z hurts.
  - `gate_trigger_tokens` is weak.
  - Steering labels are not "say" labels.
  - The neatness of J-VINDEX labels at L10–30 comes from the J++ lens.
- Two parts do not survive as stated:
  1. **"J-VINDEX adds nothing over the naive lens" is too strong.** At L40–45, J++ beats naive by +0.09 to +0.12 P@10 under all 5 interventions; neuron-cluster CIs exclude 0. The integrator's own N1 data show the same at L40–45: +0.041 [+0.008, +0.074], recomputed here from `N1C_labels.npz`. Their pooled −0.004 hides this, because their sample has −0.038 at L50–57. At L50–57 the two lenses are about equal.
  2. **The L10–30 output-side picture is partly an artifact of the 60× steering magnitude.** That picture was: c+ and c− positively correlated (not antipodal), and a "generic response shared across neurons". At k = 10, with either intervention implementation, the two poles at L20–30 are antipodal: rho(c+, c−) is −0.24 to −0.31, against a null of +0.13 to +0.14. The between-neuron shared response also halves. What does survive at L10–30 is that no context-general and magnitude-stable single-neuron effect exists to label.

## Setup
- **Sample (`N1V_neurons.json`):** `random.Random(99)`; for each of L10, 20, 30, 40, 45, 50, 55, 57, `sorted(rng.sample(range(21504) minus N1 sample, 8))`. 64 neurons, 128 poles.
- **Contexts (`N1V_common.py`):** 32 hand-written contexts:
  - 10 chat with an assistant prefix;
  - 10 prose, including FR, DE, ES, ZH and RU;
  - 6 code (Python, Rust, SQL, JS, matplotlib, bash);
  - 6 misc (list, math, news, poem, letter, recipe).
  - Lengths are 18–55 tokens. Every context has at least 8 content tokens in the steering window.
- **Static labels (`N1V_01_static_cache.py`):** computed tool-exact (gemma4_rms_norm, bf16 normed vector, fp32 GEMM, softcap): JV (J-VINDEX), JL (JLens), NV (naive), G (`gate_trigger_tokens`), Gemb (W_gate·w_pre · W_E) and Uemb.
- **Causal conditions (`N1V_02_causal.py`):** delta = s·k·max|act| over the fresh contexts. The unsteered base runs in the same forward pass. Readout: c = mean logprob change at the last position, full vocabulary, fp32.
  - **A8k60 (primary):** delta added to mlp_act at the last 8 positions.
  - **A1k60:** last position only. This is the N1A style.
  - **A8k10:** as A8k60 with k = 10.
  - **L8k10, L8k60:** a "frozen-norm linear write". Add delta·(w_post ⊙ W_down[:,i]) / rms(y_p) to the post-FFN-RMSNorm output at the last 8 positions. This is the same first-order write as A8, but it cannot saturate the norm.
- **Activation labels (`N1V_03_acts.py`):** fresh neurons over the N1B corpus (147.5k positions). Enx and Ein are built exactly as in N1C: top-1% s·act positions, log-enrichment, count ≥ 3, vocabulary A of 5,544 attested tokens.
- **Analysis (`N1V_04_analyze.py`, tables in `N1V_05_tables.py` → `N1V_tables.md`):**
  - P@10 is the share of a label's top-10 that falls in the reference's top-100.
  - Every metric is shown next to the exhaustive within-layer permutation null (7 other neurons, same sign).
  - 95% bootstrap over poles. Paired differences also get a neuron-cluster bootstrap, because a neuron's two poles are not independent.
- **Sanity checks (all passed):**
  - The run-from-layer-l replay is exact (max |Δh| = 0.0 at L10, L40 and L57).
  - Under A1, logprob differences at positions before the steered one are exactly 0.
  - No MEMIT overlays are active.
  - fp32 logits match HF logits in argmax; max abs diff 0.54 from bf16 rounding of the HF logits.
  - The bf16 chaos floor (k = 0.01) has KL 1e-4 to 1.2e-3 and split-half rho of about 0.

### Pre-stated predictions and outcomes
- **V1a (replay exact):** confirmed.
- **V1b (60× max|act| saturates the post-FFN RMSNorm at most layers):** the median of r60 = 60·max|act|·‖W_down[:,i]‖ / ‖y‖ is:

  | layer | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 |
  |---|---|---|---|---|---|---|---|---|
  | median r60 | 0.29 | 1.44 | 2.01 | 2.82 | 2.33 | 1.69 | 0.89 | 0.65 |

  The ratio exceeds 1 at 70–100% of window positions at L20–50. Confirmed at L20–50; not at L10 or L55–57.
- **V2a (A8k60 replicates N1A):** confirmed.
- **V2b (the L10–30 positive c+/c− correlation is a norm-saturation artifact):** partly wrong.
  - Antipodality returns at k = 10 with both implementations (A8k10 and L8k10).
  - L8k60 bypasses the norm and still shows c+/c− > 0 at L20–30.
  - So the driver is the steering magnitude, not RMSNorm saturation as such.
- **V2c (causality check):** confirmed.
- **V3a (say lift < 0.03):** confirmed; the lift is at null.
- **V3b (gate lift < 0.04, far below the ceiling):** confirmed.
- **V3c (pole input sets overlap at null):** confirmed.

## Q1: static output labels vs causal steering

P@10, label top-10 inside causal top-100 on the full vocabulary, real/null. `*` marks a lift whose CI excludes 0.

| label → causal | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 | L10–30 | L40–57 |
|---|---|---|---|---|---|---|---|---|---|---|
| JV → A8k60 | .006/.001 | .006/.000 | .000/.000 | .369/.000* | .587/.000* | .425/.000* | .544/.001* | .725/.000* | .004/.000 | .530/.000* |
| NV → A8k60 | .006/.000 | .000/.000 | .006/.004 | .269/.001* | .500/.000* | .400/.000* | .550/.000* | .738/.000* | .004/.001 | .491/.000* |
| JV → A1k60 (N1A style) | .000/.000 | .000/.000 | .000/.002 | .363/.002* | .569/.000* | .406/.001* | .569/.002* | .725/.000* | .000/.001 | .526/.001* |
| JV → A8k10 | .000/.000 | .000/.001 | .000/.000 | .506/.001* | .581/.001* | .463/.000* | .594/.001* | .750/.000* | .000/.000 | .579/.001* |
| JV → L8k10 | .006/.000 | .006/.001 | .006/.000 | .506/.001* | .594/.001* | .487/.000* | .575/.002* | .769/.000* | .006/.000 | .586/.001* |
| JV → L8k60 | .000/.001 | .000/.002 | .006/.000 | .250/.000* | .569/.007* | .331/.000* | .438/.000* | .744/.000* | .002/.001 | .466/.001* |
| N1 (integrator), JV | .002/.001 | .000/.000 | .000/.001 | .354/.003 | .346/.000 | .550/.000 | .673/.000 | .608/.001 | – | ≈.51 |

**Late layers (L40–57):**
- The late-layer validity replicates on fresh neurons and fresh contexts under every intervention: P@10 of 0.47 to 0.59 against a null of at most 0.002.
- It is the same for + and − poles (A8k60, late: 0.528 vs 0.532).
- Moving from last-1 to last-8 makes no difference: A1k60 ≈ A8k60, and the causal labels overlap 0.98 at late layers.
- **Magnitude matters at L40.** k = 10 gives 0.51, k = 60 gives 0.37 and L8k60 gives 0.25. The integrator's 60× steering understates validity at L40.

**Spearman over the full vocabulary:**
- Lift at L40–57 is +0.29 to +0.78. That is 34–79% of the attenuation ceiling sqrt(SB(r)), against the integrator's 45–77%. Replicated.
- At L20–30 the J-VINDEX lift is small but significant under every condition: +0.02 to +0.12, and +0.03 to +0.19 for the antisymmetric part (c+ − c−)/2.
- Naive is at about 0 at L20: −0.01 to −0.002.
- On vocabulary A, JV → C at L10–30 is 0.042 vs a null of 0.019 (lift CI [+0.005, +0.043], driven by L20). Naive is 0.021/0.021.
- So J++ carries a faint causal signal at L20–30 that naive lacks. It is far too weak for a top-10 label.

**Reliability of the causal target at L10–30:**
- Median split-half rho is 0.00–0.07 under every condition except L8k60 at L30 (0.20). N1A measured 0.03–0.04.
- The L10–30 causal labels are not stable across magnitude. L8k10 → L8k60 overlap is 0.031–0.044 vs a null of 0.014–0.020; at late layers it is 0.43–0.94.
- At L10 the effects are close to the bf16 floor. Median KL is 0.001 at k = 10, against a floor of 6e-4 to 9e-4.

**J-VINDEX minus naive, paired P@10** (neuron-cluster 95% CIs):

| intervention | L40–45 | L50–57 | L40–57 | all | Spearman, all |
|---|---|---|---|---|---|
| A8k60 | +0.094 [+0.047, +0.141] | +0.002 [−0.042, +0.046] | +0.039 [+0.005, +0.074] | +0.024 [+0.001, +0.047] | +0.037 [+0.030, +0.045] |
| A1k60 | +0.087 [+0.047, +0.128] | −0.040 [−0.081, −0.000] | +0.011 [−0.023, +0.046] | +0.007 [−0.017, +0.028] | +0.018 [+0.011, +0.027] |
| A8k10 | +0.109 [+0.053, +0.172] | −0.006 [−0.044, +0.033] | +0.040 [+0.004, +0.080] | +0.025 [+0.001, +0.052] | +0.036 [+0.028, +0.044] |
| L8k10 | +0.119 [+0.062, +0.178] | +0.002 [−0.040, +0.048] | +0.049 [+0.010, +0.089] | +0.033 [+0.007, +0.059] | +0.037 [+0.028, +0.046] |
| L8k60 | +0.091 [+0.044, +0.138] | −0.010 [−0.054, +0.035] | +0.030 [−0.006, +0.066] | +0.016 [−0.006, +0.039] | +0.030 [+0.022, +0.039] |
| N1 data, recomputed | +0.041 [+0.008, +0.074] | −0.038 [−0.067, −0.008] | −0.007 [−0.030, +0.018] | −0.004 (pole bootstrap) | +0.012 |

- **JLens:** JLens minus naive at L40–45 is +0.08 to +0.13 under all conditions. At L50–57 it is +0.07 under k = 10 (CI excludes 0) and about 0 otherwise. JLens is at least as good as J++.
- **Prior-z:** z-scoring against the isotropic random-vector prior hurts under every condition, by −0.07 [−0.10, −0.05] overall and −0.11 to −0.13 at late layers. Confirmed.
- **Free confidence signal:** the top-10 overlap between J-VINDEX and naive predicts causal validity within L40–57.
  - Within-layer Spearman with P@10 is 0.57–0.93.
  - AUC for P@10 ≥ 0.3 is 0.84; the integrator also had 0.84.
  - Layer ≥ 40 alone gives AUC 0.85, against 0.92 for the integrator.
  - Confirmed.
- **Peakedness:** an earlier claim in this study (REPORT.md §4, 'Single neurons decode to near-flat J++ distributions') said single-neuron J-VINDEX distributions are "near flat (top-10 mass ~1e-4)". That is not true with the current tool-exact unembed.
  - Fresh poles have a median top-10 softmax mass of 0.28–0.53. N1A measured 0.33–0.68. Isotropic vectors give 0.36–0.95.
  - Peakedness therefore cannot separate real labels from lens noise. The 1e-4 figure probably comes from a pre-eps-floor unembed.

## Q4 (output side): bipolarity depends on the steering magnitude

Spearman(c+, c−) for the same neuron, real/null. The null pairs a pole with the opposite pole of another neuron.

| condition | L10 | L20 | L30 | L40 | L45 | L50 | L55 | L57 |
|---|---|---|---|---|---|---|---|---|
| A8k60 | −.07/.11 | **+.11/.36** | **+.26/.46** | −.09/.21 | −.61/.09 | −.69/.07 | −.88/.03 | −.87/.03 |
| A1k60 (N1A style) | −.19/.20 | +.15/.31 | +.28/.48 | −.08/.22 | −.59/.10 | −.69/.08 | −.89/.03 | −.89/.03 |
| A8k10 | +.17/.32 | **−.24/.13** | **−.31/.14** | −.55/.07 | −.80/.03 | −.92/.02 | −.98/.01 | −.99/.00 |
| L8k10 | +.24/.32 | **−.27/.14** | **−.29/.13** | −.54/.05 | −.78/.02 | −.91/.01 | −.97/.01 | −.99/.00 |
| L8k60 | −.03/.09 | +.28/.32 | +.24/.34 | −.15/.10 | −.47/.06 | −.49/.06 | −.79/.03 | −.84/.03 |

The between-neuron same-sign rho measures the "generic response". At L20/L30 it is:

| condition | L20 | L30 |
|---|---|---|
| A8k60 | +0.36 | +0.44 |
| A1k60 | +0.38 | +0.48 |
| A8k10 | +0.17 | +0.18 |
| L8k10 | +0.16 | +0.14 |
| L8k60 | +0.30 | +0.34 |

What the bipolarity data show:
- The integrator's "not antipodal at L10–30 (+0.15 to +0.28)" reproduces exactly under k = 60 on fresh data.
- It reverses at k = 10 in both implementations. Both large-magnitude conditions show it: A8k60, which saturates the norm, and L8k60, which does not. So any large push at L20–30 produces a shared disruption. The neuron's own effect at moderate k is antipodal but context-specific.
- At L40–57, mirror-image outputs are confirmed and become even sharper at k = 10: rho of −0.54 to −0.99, and P@10 of c+ against −c− of 0.47–1.00.
- At L40 under k = 60, antipodality is weak on fresh data: −0.09 against a null of +0.21. The integrator reported −0.36 to −0.83 at L40–57.

## Activation-based labels (Q2 say-test, Q3 gate, Q4 inputs)
Fresh neurons on the N1B corpus, vocabulary A, P@10 real/null (early = L10–30, late = L40–57):

| comparison | early | late | all (lift, 95% CI) |
|---|---|---|---|
| causal (A8k60) → next-token label, the say test | .013/.009 | .026/.020 | .021/.016, +0.005 [−0.001, +0.012] |
| causal (A1k60) → next-token label | .021/.013 | .028/.022 | +0.007 [−0.001, +0.015] |
| ALIGN_next of causal (A8k60), percentile points | +0.1/+0.5 | +0.5/+0.3 | +0.3/+0.3, ceiling +4.3/+0.6 |
| J-VINDEX → next-token label | .029/.033 | .022/.013* | .025/.020 |
| `gate_trigger_tokens` → input-token label | .062/.039* | .037/.030 | .047/.033, +0.013 [+0.003, +0.024] |
| ceiling: input label, half 0 → half 1 | .206/.019 | .296/.032 | .263/.027 |
| ALIGN_cur of gate (tool) | +2.6/−0.3* | +2.1/+0.2* | +2.3, against a held-out ceiling of +8.3 |
| ALIGN_cur of direct gate embedding | +3.9* (L10 +9.8) | +0.7* | – |
| input label pole s → pole −s | .013/.017 | .028/.026 | −0.001 [−0.010, +0.011] |
| J-VINDEX → input-token label | .035/.032 | .036/.017* | +0.013 [+0.003, +0.025] |

**Say test:**
- Confirmed, and weaker than the integrator found: it is at null on fresh neurons.
- The integrator's "only L40 clear" (0.129/0.067) does not replicate: fresh L40 is 0.075/0.062.

**Gate label:**
- Confirmed: the top-10 overlap is about 6% of the reproducible signal (integrator: 8%).
- At L10 the direct embedding readout gives ALIGN +9.8; the tool's label gives −0.2. That matches the integrator's +10.7 vs +0.2.
- Nuance: on the whole ranking, the gate label recovers 28% of the held-out ALIGN ceiling overall (lift +2.2 vs +7.9 pp); at L20–30 the share is about 70%. The integrator found 25%. "Nearly useless" is right for a top-10 token list and too strong for the full ranking.

**Input side of bipolarity:** the two poles' input sets overlap at null. Confirmed.

**Mechanism:**
- g > 0 in 90% of top-30 events, and sign(act) = sign(u) in 90%. Confirmed.
- 75% of neurons have a weaker pole reaching at least 30% of the stronger pole's maximum (integrator: 82%).

**Activation support for J-VINDEX:** none at L10–30 against either the input or the next-token label (both at null). Small but significant at L40–57.

**Four-way agreement matrix on vocabulary A** (early / late; this was the experiment the user requested):

| row label → column label | JV | NV | C (A8k60) | Enx | Ein |
|---|---|---|---|---|---|
| JV | – | .046/.021* / .786/.023* | .042/.019* / .659/.018* | .029/.033 / .022/.013* | .035/.032 / .036/.017* |
| NV | .060/.021* / .796/.022* | – | .021/.021 / .594/.019* | .088/.078 / .034/.024* | .050/.067 / .040/.025* |
| C | .040/.020 / .640/.017* | .019/.020 / .580/.021* | – | .013/.009 / .026/.020 | .015/.012 / .049/.027* |
| Enx | .023/.020 / .035/.017* | .019/.022 / .034/.018* | .019/.016 / .029/.019 | – | .213/.031* / .133/.023* |
| Ein | .021/.022 / .045/.022* | .021/.015 / .041/.020* | .017/.018 / .054/.022* | .075/.007* / .074/.016* | – |
| G | .062/.042 / .035/.023 | .035/.031 / .031/.020 | .013/.011 / .026/.016 | .046/.041 / .029/.034 | .062/.039* / .037/.030 |

What the matrix shows:
- **Output-side labels agree with each other at late layers.** JV, NV and C reach 0.58–0.80 against a null of about 0.02.
- **Activation labels barely touch them, at any layer.** The overlap is at most 0.05, so whatever the neuron pushes when steered is mostly not what it fires on or what follows it.

## Label tightness (the "neatness" test)
- **L10–30:** J-VINDEX top-10 tightness is 0.065, against 0.084 for 512-neuron random mixtures. The difference is −0.019 [−0.039, +0.001], so real neurons are no tighter. Confirmed: the neatness belongs to the lens.
- **L40–57:** 0.108 vs 0.042, a difference of +0.065 [+0.037, +0.098]. Naive gives +0.060. Confirmed: at late layers the neuron basis is special.
- **Not replicated:** "J-VINDEX labels are tighter than naive at L10–30 (+0.029 [+0.011, +0.046])". Fresh data give +0.010 [−0.016, +0.034]; only L20 shows it (0.086 vs 0.028).
- **Example (L30 n6971):**
  - The labels look neat: J-VINDEX+ is "gardens, botanical, horticultural" and J-VINDEX− is "electrical, 電力, электро".
  - Steering pushes "décidé, determined, decided". The same generic tokens appear for other L30 neurons.
  - The pole fires on ML-code tokens: Conv, dropout, convolution.

## Claim-by-claim verdict
| integrator claim | verdict | reason (fresh sample) |
|---|---|---|
| JV valid at L40–57, null at L10–30 | **confirmed** | Late P@10 .47–.59 vs ≤ .002 under 5 interventions; early ≈ null. Caveat: a faint whole-vocabulary JV signal at L20–30 (rho lift +.02 to +.12) that naive lacks. |
| J++ adds nothing over naive | **weakened** | J++ is better at L40–45 by +.09 to +.12 (cluster CIs exclude 0 in all 5 conditions; N1 data +.041 [+.008, +.074]). About equal at L50–57. Spearman +.02 to +.04 overall. Practically small. |
| prior-z hurts | **confirmed** | −.07 [−.10, −.05] in every condition. |
| L10–30 neatness is a lens property | **confirmed** | Neurons no tighter than mixtures at L10–30, tighter at L40–57. The sub-claim that JV is tighter than naive at L10–30 does not replicate. |
| weak say property | **confirmed (stronger)** | Lift at null (+.005 [−.001, +.012]). The L40 exception does not replicate. |
| gate_trigger_tokens ≈ useless; embedding readout wins at L10 | **confirmed** | About 6% of the top-10 ceiling; embedding ALIGN +9.8 vs gate −0.2 at L10. Whole-ranking ALIGN is 28% of the held-out ceiling. |
| distinct pole inputs, antipodal outputs at L40–57 | **confirmed** | Input overlap at null. rho(c+, c−) at L45–57: −.47 to −.89 at k = 60 and −.78 to −.99 at k = 10. L40 is only −.09 at k = 60 but −.54 at k = 10. |
| not antipodal / generic shared response at L10–30 | **refuted as a neuron property** | It holds only at 60× magnitude. At k = 10, L20–30 poles are antipodal (−.24 to −.31 vs null +.13 to +.14) and the shared response halves. |
| no context-general single-neuron effect below L40 | **confirmed** | Split-half rho .00–.07; causal labels unstable across magnitude (k10 vs k60 overlap at null). |
| JV/NV agreement predicts validity | **confirmed** | AUC .84; within-layer rho .57–.93. |
| cleanliness audit (46% / 62% one job) | **untested** | Not redone. |

## Methodological issues found
1. **60× max|act| is outside the linear regime at L20–50.**
   - The neuron delta exceeds the whole FFN output norm at most window positions: median r60 is 1.4–2.8 at L20–50.
   - Effects at L20–30 then have a magnitude-driven component shared by both signs and by all neurons.
   - This produced the integrator's "not antipodal / generic" reading at L10–30, and it lowers P@10 at L40 (0.37 vs 0.51 at k = 10).
   - Recommendation: use k ≈ 10, or the frozen-norm linear write, as the causal reference.
2. **Pooling the JV-vs-naive difference over L40–57 hides a layer-dependent effect.** In both samples J++ helps at L40–45; in N1 it hurts at L50–57.
3. **Pole-level bootstrap treats a neuron's two poles as independent.** Their static labels are exact negations, and their causal effects are mirrored at late layers. The neuron-cluster CIs used here are slightly wider, for example [+0.003, +0.046] becomes [+0.001, +0.047].
4. **The brief's "near-flat single-neuron J-VINDEX distribution" fact is wrong** for the current unembed (top-10 mass 0.28–0.68).

## Limitations
- 8 neurons per layer, so per-layer CIs are wide. Band-level and pooled statements are the reliable ones.
- The activation labels use the same N1B corpus as the integrator, which is mostly technical text. Only the neurons are fresh here, so corpus robustness is untested.
- The cleanliness audit was not redone.
- Only one set of 32 contexts was used.
- The GPU was shared, and jobs fell back to 5 slots per forward after an OOM. The results are unaffected, because the base slot is always in the same forward pass.

## Files
- **Scripts:** `experiments/scripts/N1V_common.py`, `N1V_lib.py`, `N1V_01_static_cache.py`, `N1V_02_causal.py`, `N1V_03_acts.py`, `N1V_04_analyze.py`, `N1V_05_tables.py`.
- **Results:** `experiments/results/N1V_neurons.json`, `N1V_analysis.json`, `N1V_analysis_tables.txt`, `N1V_tables.md` (all tables), and `N1V_01_static_cache.json.full.json`, `N1V_02_causal_r{1,2}.json.full.json`, `N1V_03_acts.json.full.json`.
- **Caches:** `N1V_cache/{static_L*,causal_*_L*,ctx_cache}.pt`, `N1V_cache/null_dirs_top10.json` and `N1V_acts/part_*.npz`.
- **Name clash:** the files `N1V_audit_*`, `N1V_code_audit.md`, `N1V_cache/steer_dir_*` and `N1V_cache/smallk_*` belong to a different verifier (the code audit), which also uses the N1V prefix.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
