# V6_quiet_dims: adversarial check of the C7 "quiet dimensions" claim

Target: the structural part of claim C7 (E4, E6). It says about 10% of residual dimensions are "quiet", meaning near-constant across tokens, and that they have huge downstream gain. It says J++/JLens pullbacks put 99-100% of their energy in these dimensions at L35-53. It says a random push of about 1% of ‖h‖ collapses the model. It also asks whether this explains the weak J++ vs IT-gradient alignment reported by E3.

Model and access: Gemma-4-31B-it, 4-bit QAT W4A16, run only through labd. Lenses: J++ and JLens, both fitted on the base model.

## Methods

**01 stats.** I collected resid_post at all 60 layers over 8,085 non-BOS tokens:
- 2,082 prose tokens from the 24 E3 encyclopedic paragraphs;
- 6,003 chat tokens: the 24 E3 default chats with the model's own greedy answers, plus 14 teacher-forced first-person passages × {default, woman persona, man persona}.

For every coordinate I computed mean and std. I did this for the whole set, prose only, chat only, and even/odd sequence halves.

I also recorded, per coordinate:
- max |h|, the BOS value, and a random sample of 540 tokens × 60 layers;
- read gains: |input_layernorm|, |pre_feedforward_layernorm| and |final norm| weights, and the column norms of q/k/v and gate/up;
- write gains: |post_attention_layernorm| and |post_feedforward_layernorm| weights, and the row norms of o/down.

Quiet sets, with numbers taken from the 01 stats:
- **q10**: bottom 10% of per-layer std. This is E6's definition.
- **deep**: std below 0.05 × the per-layer median.
- **pers**: the 181 dims that are deep-quiet at every layer from L42 to L57.

**Energy shares (local).** Pullbacks fnw⊙(W_a−W_b)·J[l] for J++ and JLens. Directions:
- gender pairs: she/he, woman/man;
- 4 control pairs: cat/dog, yes/no, big/small, the/a;
- 20 random token pairs.

Gradients: E3's true same-position VJPs, 144 positions × pron/noun/2 random directions, plus their averages gbar, gbar_chat and gbar_prose.

**05 dose-response.** I added one fixed random vector at every non-BOS position at a single layer (L20, L30, L40 or L50). Eight families, all at equal raw norm f·median‖h_l‖ with f from 0.001 to 0.3, 2 seeds each:
- iso;
- confined to q10, to pers, or to 538 random dims from the loud half;
- "white" versions (ξ⊙std) over all dims, over q10, and over loud dims;
- natural token differences.

Readout: KL(clean‖perturbed) over the last 16 positions of 12 prompts (6 chat, 6 prose).

**06 / 06b readouts.** Seven conditions (default, nameless, Gemma-short, woman ×2, man ×2) × 24 E1 passages, teacher-forced, layers 10-58, 6,328 token positions. J++ and JLens pron/noun scores under these masks, where "mean" means the corpus token-mean of each coordinate:
- full;
- q10 → mean;
- pers → mean;
- deep → mean;
- q10 → 0;
- controls: 3 random 10% masks → mean; q10 minus deep ("mod") → mean; top-10% read-gain → mean.

Also:
- a linear split of the pre-softcap pron score into μ_loud, μ_quiet, dev_loud and dev_quiet;
- IT-gradient readouts ⟨h, gbar_l⟩, full and masked.

**Statistics.** Bootstrap over passages, or over E3 position groups for the VJP analyses (300-1000 resamples). Predictions were registered in each script header.

## Results

### 1. What the quiet dims are (`V6_quiet_dims_stats.pt`, `_samples.pt`)

**The std distribution depends on depth.**
- At L0-15 most coordinates are tiny (median std 0.01-0.02) and a few hundred are loud.
- At L24-33 there is almost no deep-quiet tail: 4-78 dims.
- At L36-57 a distinct cluster sits 2-3 orders of magnitude below the median:
  - L36-39: 465-491 dims, std 0.006-0.018 × median;
  - L40-49: 220-421 dims, 0.003-0.006 × median;
  - L50-57: 193-219 dims, 0.0014-0.0028 × median.
- **The "10%" cut is not the natural boundary.** At L41-57 the 10th percentile is 0.3-0.6 × the median, so roughly 60% of E6's q10 set is only moderately low-variance.

**The quiet set is stable across data.**
- Deep-set Jaccard between prose and chat is 0.89-1.00 at L35-58.
- The Spearman correlation of per-dim std between prose and chat is 0.73-0.99.
- So quietness is a property of the network, not of the corpus.

**The same indices stay quiet across depth, within L35-58.**
- Deep-set Jaccard is 0.60-0.94 among L42-57 and 0.32-0.67 between L35 and the later layers.
- 181 dims are deep-quiet at every layer from L42 to L57.
- A different, transient set of 331 dims is quiet at L15-25, has about 10× read gain there, and is ordinary from L30 on.

**They are not near-constant.**
- Their coefficient of variation (std/|mean|) is 2.5-8, the same as other dims. They are scaled-down channels, not constant offsets.
- In the coordinates the next layer actually reads, w_in[l+1]⊙h/rms(h), the 181 persistent dims have 1.1-2.5× the median std of other dims and carry 4-14% of the token-varying input variance at L35-58.

**Gain.** Quietness tracks read gain at every depth.
- Spearman(log std_l, |w_in[l+1]|) is −0.4 to −0.95, and with |w_preffn[l+1]| it is −0.45 to −0.99 (L5-58). With the final-norm weight the relation is weak and changes sign with depth: +0.37 to +0.46 at L5-20, −0.41 to −0.47 at L54-58.
- The extreme weights all sit on persistent dims:
  - max |w_in| = 1248 (L52, dim 1683);
  - max |w_pre| = 860 (L51, dim 4206);
  - max |fnw| = 512 (dim 4501).
- The top-50 |fnw| dims and the top-100 |w_in[51]| dims are 100% persistent.
- Read-gain ratio (persistent vs other): 0.01-0.1× at L2-15, 260-1700× at L40-58.
- Write-gain ratio (post-attention and post-FFN norm weights): 200-1100× at L2-15, 0.001-0.003× at L30-58.

**History of the persistent dims.**
- At L5-20 these 181 dims are among the *loudest* (97th percentile of std, 184-448× the median std).
- After about L25 the model stops writing to them, and the layer_scalar leak shrinks them geometrically.
- Their values at L45 still correlate with their own L20/L25 values: median per-dim r = 0.71, against 0.03 for loud dims. The slope is 1.3-3× the product of layer scalars over that span.
- So this is a low-amplitude **carry channel of early-layer content**, read with compensating gain. It is not an unused direction.

**Massive activations and attention sinks: not the same thing.**
- The canonical massive dims (2041, 2293, 2250, 1451, 3970, 1376) have std 10-35 and max 145-350 at every layer. None are in the persistent set, and at L30+ the overlap with the top-1% max|h| dims is 0.
- The BOS energy share of the persistent dims (17% at L5) matches their share for ordinary tokens (17.5%), so they are not BOS-specific.
- The persistent dims are, however, the early large-activation dims: they make up 43-81% of the top-1% max|h| dims at L3-20.

### 2. Energy shares (`V6_quiet_dims_03_energy_shares.json`, `_08_tokpair_energy.json`)

| covector | q10 share L35 / L40 / L45-58 | pers share L30 / L40 |
|---|---|---|
| J++ pullback, gender pairs | 0.988 / 0.999 / 1.000 | 0.56 / 0.80 |
| J++, 4 control pairs and 20 random pairs | 0.99 / 1.000 / 1.000 (min over pairs 0.985) | 0.79 / 0.89 |
| JLens, gender / random pairs | 0.990 / 0.999 / 1.000 | 0.68 / 0.84 ; 0.81 / 0.90 |
| **true IT VJP, single position, pron** | 0.990 / 1.000 / 1.000 | **0.89 / 0.94** |
| true IT VJP, single position, random direction | 0.990 / 1.000 / 1.000 | 0.90 / 0.95 |
| true IT gbar (average gradient) | 0.988 / 0.999 / 1.000 | 0.95 / 0.98 |
| isotropic | 0.10 | 0.034 |

- The concentration does not depend on direction, and **the true IT-model gradients concentrate at least as much as the lens pullbacks.**
- Gradient energy per dimension rank-correlates with |w_in[l+1]| (Spearman 0.47-0.92).
- So E6's explanation, "the averaged-Jacobian lens [2,4] assigns large weights to low-variance coordinates", does not identify a lens defect. Any covector looks like this in a basis where the read gains span 3 orders of magnitude.
- Treating a covector (pullback or gradient) as a displacement for steering is a type error. That is why steering along such directions collapses the model.

### 3. Sensitivity (`V6_quiet_dims_05_dose_L*.json`)

Single-layer pushes, KL in nats, mean of 2 seeds. Three families at the stated fraction of ‖h‖:

| layer | quiet10 at 1% | pers at 0.3% | loud at 30% |
|---|---|---|---|
| L40 | 10.4 | 2.9 | 0.11 |
| L50 | 10.9 | 8.6 | 0.065 |

Other families:
- At L40/L50, iso at 1% gives 1.0 / 5.4.
- Natural differences at 10% give 0.04 / 0.02.
- At L20 the persistent dims, loud there and with low read gain, are harmless: the KL = 0.1 crossing is at f* = 0.06, close to loud (0.085). This confirms my prediction P-c.

The KL = 0.1 crossing, measured three ways:
- **Raw norm:** f* ranges over more than 200× across families at L40. The C7 sensitivity claim, and my prediction P-a, are confirmed.
- **Gain-weighted size** ‖w_in[l+1]⊙δ‖/rms(h): the crossings fall within 1.5-4× of each other. At L40 they are 8.8-16.2 for every family that crosses.
- **Fit quality:** the R² of log KL against log gain-weighted size is 0.94 / 0.85 / 0.84 / 0.82 at L40 / L50 / L30 / L20. Against log raw norm it is 0.02 / 0.07 / 0.24 / 0.58.

Quiet coordinates pushed at their own natural scale (white_q10) are not special per unit of z-size:
- L40: z* = 22, against 19 for natural differences;
- L50: z* = 16, against 26.

This supports P-b. The "catastrophic gain" is read gain acting on pushes that are tens of natural standard deviations off-distribution in those coordinates.

### 4. Gender readouts (`V6_quiet_dims_09_readout_summary.json`, `_10_ctrl_summary.json`)

J++ pron at self tokens. Retention = the woman−man gap under the mask divided by the full gap, with 95% CI. "pos" = the default condition's position on the man(0)/woman(1) scale.

| band | full gap | q10→mean | deep→mean | pers→mean | random 10% (3 seeds) | top-10% gain | default pos full → q10-masked |
|---|---|---|---|---|---|---|---|
| L20-29 | 12.7 | 0.99 | 1.00 | 0.92 | 0.93-0.94 | 0.94 | 0.48 → 0.48 |
| L30-39 | 21.5 | 0.82 [.81,.83] | 0.91 | 1.01 | 0.93-0.94 | 0.73 | 0.45 → 0.45 |
| L40-49 | 7.2 | 0.86 [.84,.87] | 0.97 | 0.99 | 0.92-0.93 | 0.74 | 0.56 → 0.61 |
| L50-58 | 2.4 | **−0.17** [−.27,−.09] | 1.00 | 1.00 | 0.96 | −0.09 | 0.88 → undefined |

**Condition separation survives at L20-49.**
- Persona gap: compared with random masks, the quiet-specific loss is only 8-12% of the gap. The deep-quiet dims carry 0-9%.
- At L50-58 the small late persona gap (2.4 logits) lies entirely in the q10 coordinates. Masking q10 minus deep alone ("mod", the moderately low-variance dims with 2-6× read gain) gives −0.16. It does not lie in the deep-quiet dims (retention 1.00). JLens and the noun axis show the same pattern: retention at L50-58 is −0.10 for JLens pron and 0.21-0.26 for noun.

**Contrasts are robust to every mask at L20-49 (C1 and C2 survive):**
- default − nameless stays within ±0.02 of the gap;
- Gemma − nameless stays at +0.09 to +0.14.

**Background.** Linear split of the default self-token pre-softcap J++ pron score into [μ_loud, μ_quiet, dev_loud, dev_quiet]:

| band | split | quiet coordinates' share |
|---|---|---|
| L20-29 | (μ_quiet −0.13) | negligible |
| L30-39 | [−0.27, +0.18, +2.96, −0.37] | small |
| L40-49 | [+0.53, **+1.17**, +2.03, −0.15] | 33% of the total, all from their DC mean |
| L50-58 | [−0.71, **+1.11**, +0.54, **+1.07**] | all of it (2.2 of 2.0) |

- E3's "she-leaning μ background at L40-49" (+2.3 to +3.5) is partly a quiet-coordinate DC term. The J++ readout of μ goes from +3.5 to +1.3 when q10 is zeroed; for JLens L50-58 it goes from +0.9 to −4.3.
- Persona-scale positions cancel this term, which is why positions barely move at L20-49 (Δ ≤ 0.05).
- My prediction R4, a shift of more than 0.1 at L40-58, is not supported at L40-49 and cannot be tested at L50-58, where the scale collapses.

### 5. Does masking quiet dims make J++ agree with the true IT gradient? (`_07_gradalign_raw.json`, `_04_dataagree.json`)

Raw cos(gbar, J++ pullback), full vs restricted to non-q10 coordinates, with 95% CI and the split-half ceiling of gbar in that subspace:

| band | all positions | chat positions | prose positions |
|---|---|---|---|
| L20-29 | 0.07 → 0.07 | **−0.15 → −0.17** | 0.15 → 0.16 |
| L30-39 | 0.15 → 0.19 [.15,.21] (ceil .68) | **−0.13 → +0.07** [.03,.11] | 0.24 → 0.19 |
| L40-49 | 0.26 → **0.84** [.82,.84] (ceil .97) | 0.07 → 0.85 | 0.28 → 0.78 |
| L50-58 | 0.33 → **0.86** (ceil .99) | 0.32 → 0.86 | 0.26 → 0.81 |

JLens: 0.37 → 0.91 and 0.46 → 0.93 at L40-58.

**L40-58: yes.** In the loud subspace, J++/JLens and the IT model's average gradient agree almost up to reliability. The low raw cosines there come from disagreement *inside* the quiet coordinates: restricted to q10, cos is 0.28-0.39. That is plausibly a base→IT transfer problem, since tiny coordinates read with 500× gain are where weight changes matter most; this was not tested.

On natural tokens, the correlation between the J++ readout and the IT-gradient readout also rises: L45 0.74 → 0.84, L50 0.64 → 0.80.

**L20-39: no.** E3's mid-layer misalignment is not caused by the quiet dims. Chat L20-29 stays negative (−0.17; −0.15 in the data metric too). Chat L30-39 only moves to about zero (+0.07). Data-metric correlation at L20-35 is −0.02 to 0.34.

**At the condition level, masking changes nothing.** J++ minus IT(gbar) default position, full → q10-masked:
- L30-39: +0.09 [.03,.15] → +0.10;
- L40-49: +0.12 → +0.08.

My prediction R5 is refuted.

Side finding: on these E1 passages and personas, IT(gbar) puts the default at 0.37 [.30,.43] (L30-39) and 0.45 (L40-49). E3 reported 0.22 / 0.21, so E3's J++−IT gap of +0.23/+0.30 depends on the stimuli; here it is only +0.09/+0.12.

## Verdicts on the C7 structural sub-claims

| sub-claim | verdict | revision |
|---|---|---|
| About 10% of dims are quiet and near-constant across tokens | **weakened** | A distinct cluster of 193-491 dims (3.6-9%) at L36-57, 181 of them persistent across L42-57, with std 0.001-0.02 × median. It is absent at L24-33, and a different transient set appears at L15-25. These dims are not constant (CV like other dims). They are early-layer loud channels that stop being written and decay, and they carry normal-size signal in the next layer's normalized coordinates. |
| Huge downstream gain (RMSNorm up to 1248) | **confirmed, sharpened** | Read gain 260-1700× median exactly on these dims (max w_in 1248, w_pre 860, fnw 512 all on persistent dims). Spearman(log std, read gain) −0.4 to −0.99. Write gain about 0.002×. |
| Same indices across layers? | **confirmed within L35-58** | Different from L≤25, where the same 181 dims are the loudest. |
| Massive-activation / sink coordinates? | **no** | Canonical massive dims (2041, 2293, …) are disjoint at L30+. The persistent dims overlap the early large-activation dims at L3-20 but are not BOS-specific. |
| Lens pullbacks put 99-100% of energy in quiet dims at L35-53 | **confirmed, but not lens-specific** | The true IT VJPs do the same (99-100%; more concentrated than J++ at L25-40), for any direction. Covector geometry, not lens error. |
| A random push of about 1% of ‖h‖ collapses the model | **confirmed, mechanism identified** | Single-layer 1% pushes give KL 1-11 nats at L40-50. KL is a function of read-gain-weighted (equivalently z-scored) size, R² 0.82-0.96. Quiet dims at natural scale are not special. |
| Quiet coords drive lens readouts | **mixed** | At L20-49 they carry ≤12% of the persona gap beyond random masking, and contrasts and positions are robust. At L50-58 the J++/JLens persona gap lives entirely in ~300 moderately-low-variance, high-gain coords (not the deep-quiet set). The quiet DC mean supplies about +1.1 logits of the L40-58 she-leaning pron background. |
| Masking quiet dims improves J++/IT alignment | **partly confirmed** | L40-58 raw cos 0.26-0.33 → 0.84-0.86 (near ceiling). Not at L20-39 (chat stays −0.17 at L20-29). No change in condition-level J++−IT disagreement. |

## Files

Scripts (`experiments/scripts/`):
- V6_quiet_dims_common.py
- V6_quiet_dims_00_inspect.py
- V6_quiet_dims_01_stats.py
- V6_quiet_dims_05_dose.py, with wrappers 05a and 05b
- V6_quiet_dims_06_readout.py
- V6_quiet_dims_06b_readout_ctrl.py
- V6_quiet_dims_09_analyze_readout.py
- V6_quiet_dims_10_analyze_ctrl.py
- Local CPU analyses (run with .venv/bin/python): V6_quiet_dims_local_a1..a10.py
  - a1-a5: section 1 (std distributions, overlap across layers and corpora, gains, massive activations, carry channel)
  - a6 and a10: section 2 (energy shares of VJPs and pullbacks; token-pair pullbacks)
  - a7 and a9: section 5 (gradient alignment)
  - a8: section 3 (dose-response collapse)
  - The 181-dim persistent set is saved as results/V6_quiet_dims_pers181.npy.

Results (`experiments/results/`):
- V6_quiet_dims_stats.pt
- V6_quiet_dims_samples.pt
- V6_quiet_dims_03_energy_shares.json
- V6_quiet_dims_04_dataagree.json
- V6_quiet_dims_05_dose_L20_30.json
- V6_quiet_dims_05_dose_L40_50.json
- V6_quiet_dims_06_readout.pt
- V6_quiet_dims_06b_readout_ctrl.pt
- V6_quiet_dims_07_gradalign_raw.json
- V6_quiet_dims_08_tokpair_energy.json
- V6_quiet_dims_09_readout_summary.json
- V6_quiet_dims_10_ctrl_summary.json

## Caveats

- The VJPs are E3's: 144 positions from 48 sequences, IT model only. Base-model statistics are not available, so the base→IT explanation for the quiet-subspace disagreement is a hypothesis.
- The quiet sets were defined on an 8k-token corpus. They replicate across prose and chat, so this is unlikely to matter.
- Dose-response used a fixed vector at every position of 12 prompts. KL was measured on the last 16 positions, and the noise floor is about 3e-4 nats.
- The IT-gradient readout's woman−man gap is small in raw units, so IT positions at L10-29 and in masked L50-58 are ill-conditioned and are not interpreted.
- Readout masking replaces coordinates with the corpus token-mean. That is a linear intervention on the lens input only, not on the model.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
