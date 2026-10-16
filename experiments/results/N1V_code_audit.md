# N1V code audit: static neuron-label validation (N1A / N1B / N1C)

Verifier role: adversarial code audit of every `experiments/scripts/N1*_*.py` behind the integrator verdict in `N1_REPORT.md`.
I read all 31 N1A/N1B/N1C scripts, the Gemma-4 HF decoder/MLP/RMSNorm source and the backend unembed and pole code.
I then re-ran targeted checks: 2 daemon jobs (about 90 s GPU each) and 5 CPU analyses.

## Bottom line

- **No coding bug changes the main label-validity conclusions.** These hold:
  - output labels are valid at L40-57 and at null at L20-30;
  - J-VINDEX is no better than the naive logit lens;
  - prior-z scoring hurts;
  - gate_trigger_tokens is weak.
- **The steering protocol has a confound** (finding F1). At k=60, adding to `mlp_act` saturates Gemma-4's `post_feedforward_layernorm`. As a result the "causal effect" contains an ablation of the whole MLP output at that position.
  - That shared component is what the verdict calls "generic responses shared across neurons" at L10-30.
  - It also distorts the bipolarity numbers.
- **The label-validity results survive three re-tests:** a linear-regime k (k=10 and k=3), steering that bypasses the norm, and antisymmetrisation.
- **Some interpretive claims are weaker than stated:**
  - L10 is underpowered.
  - The anti-correlated token responses are mechanical.
  - The "neurons are tighter" result holds only against an anomalous baseline.

## Checked and found correct

| Item | How checked | Result |
|---|---|---|
| Pole sign convention | N1A_02 adds `+s*k*max\|act\|`; static '-' label = negation of '+'; N1B uses `s*act>0`; N1C row mapping is asserted against N1A meta | consistent everywhere; sign test rho(static+, c-) is -0.36..-0.75 at L40-57 |
| Hook module/position | `down_proj` forward_pre_hook = `act_fn(gate)*up` (HF `Gemma4TextMLP`); index_put at `[row, last[row], neuron]` | correct |
| W_down used by static labels | daemon: `down_proj(one-hot)` vs `rt.slice_w_down` | relative error 0.0 (L10, L40) |
| post_ffn_norm scaling | HF `Gemma4RMSNorm` multiplies by `weight` itself (no 1+w); module weight vs `rt._post_ffn_norm_weights` | max abs diff 0.0; reconstructed `post_norm(y)` rel. error about 2e-3 (bf16) |
| Run-from-layer skip trick | N1A_00 smoke: no KV sharing (`num_kv_shared_layers=0`), no per-layer inputs | exact (max logit diff 0.0); baseline slot sits in the same forward |
| Padding / last index | right padding, `last=len-1`, causal mask; N1B masks `att` then drops pos 0 (BOS) | correct; corpus has 0 pad tokens; BOS is excluded from all activation statistics |
| Full-vocab Spearman | avg-rank, centre, L2-normalise, dot | correct; my independent argsort version reproduces N1A (L40 jpp rho 0.341 vs N1A lift 0.35) |
| P@10 and nulls | same-layer, same-sign, other-neuron (23 partners, exhaustive); cross-pole scheme uses the opposite pole of other neurons; permutation index arithmetic checked | correct |
| Bootstrap over poles (+/- dependence) | within-neuron correlation of + and - pole P@10 (late) = -0.04; neuron-cluster bootstrap of paired J-VINDEX minus naive | CI unchanged: [-0.019, +0.010] (late, cluster: [-0.030, +0.017]) |
| Tool fidelity | N1A_01 / N1B_03 vs `inspect_bipolar_slot` | 7-8 of top 8 overlap for J-VINDEX, naive and gate labels |

## Findings

### F1 (methodological, confirmed on the daemon): k=60 steering saturates the post-FFN RMSNorm and injects an MLP-ablation component

In Gemma-4 the residual is updated as `resid += post_ffn_norm(down_proj(mlp_act))`. N1A adds `delta = s*60*max|act|` to one neuron at the last position.

`N1V_audit_01_postnorm` measured the resulting change in the MLP output at the last position (median over 24 neurons x 36 contexts x 2 signs). The columns are:
- `||delta*W_down||/||y||`: size of the injected write relative to the MLP output `y`.
- `||Δ||/||linear||`: the actual residual change relative to the linear prediction (1 = no saturation).
- `cos(Δ, -post_norm(y))`: how much of the change points along "remove the existing MLP output".
- `||Δ||/||post_norm(y)||`: the change relative to the size of the whole normalised MLP output.

| L | `||delta*W_down||/||y||` | `||Δ||/||linear||` | `cos(Δ, -post_norm(y))` | `||Δ||/||post_norm(y)||` |
|---|---|---|---|---|
| 10 | 0.19 | 0.98 | 0.07 | 0.32 |
| 20 | 1.73 | 0.55 | 0.46 | 1.10 |
| 30 | 1.84 | 0.56 | 0.52 | 1.02 |
| 40 | 1.15 | 0.70 | 0.33 | 1.02 |
| 45 | 1.74 | 0.52 | 0.30 | 1.56 |
| 50 | 1.42 | 0.60 | 0.26 | 1.66 |
| 55 | 0.76 | 0.82 | 0.22 | 0.97 |
| 57 | 0.83 | 0.80 | 0.27 | 0.83 |

At k=10 the saturation ratio is at least 0.965 at every layer, i.e. the linear regime.

**Directly ablating the MLP output at the last position** (setting `post_norm(y_last)` to 0) produces an effect `c_abl`. It correlates with the symmetric part `(c+ + c-)/2` of the cached N1A kA vectors:

| Layer | Per neuron | Vs the layer mean |
|---|---|---|
| L10 | 0.70 | 0.79 |
| L20 | 0.45 | 0.63 |
| L30 | 0.54 | 0.66 |
| L40 | 0.52 | 0.75 |

So the "generic response shared across neurons" is largely an artefact of the protocol.

**Re-test in the linear regime** (`N1V_audit_03_smallk`, identical protocol at k=10 and k=3; metrics in `N1V_audit_04_smallk_metrics.json`):

| | J-VINDEX P@10 (null) | naive P@10 | rho(c+,c-) | cross-neuron rho | split-half |
|---|---|---|---|---|---|
| L20 k10 | 0.000 (0.001) | 0.004 | +0.21 (k60 +0.27) | 0.21 (k60 0.33) | 0.03 |
| L30 k10 | 0.000 (0.000) | 0.002 | **-0.07 (k60 +0.31)** | 0.23 (k60 0.43) | 0.004 |
| L40 k10 | 0.385 (0.003) | 0.365 | **-0.77 (k60 -0.36)** | 0.04 (k60 0.14) | 0.39 |
| L45 k10 | 0.315 | 0.279 | -0.84 (k60 -0.68) | 0.02 | 0.55 |
| L50 k10 / k3 | 0.546 / 0.517 | 0.544 / 0.508 | -0.92 / -0.85 | 0.01 | 0.75 / 0.66 |
| L55 k10 / k3 | 0.690 / 0.633 | 0.698 / 0.656 | -0.96 / -0.90 | 0.00 | 0.91 / 0.84 |

**Norm-bypassing steering** (adding `s*α*unit(w_post*W_down_i)` after the norm, with `α` matched to the k=60 residual change) gave the same picture:
- at L20-30, P@10 is 0.000-0.002;
- at L40, J-VINDEX P@10 is 0.383 vs naive 0.358, and rho(c+,c-) is -0.51.

Antisymmetrising the cache, `(c+ - c-)/2`, raises rho at L30 from 0.038 to 0.072, but P@10 stays at 0.

**Consequences:**
- **Late-layer validity and "J-VINDEX = naive" are robust.** They are not an artefact of the huge k.
- **The L20-30 null is robust.** There is still no context-general effect even in the linear regime: split-half reliability is 0.004-0.05.
- **"Below L40 the effects are generic and shared across neurons" is mostly protocol-induced.** It should not be read as a property of the neurons.
- **The Q4 numbers come out wrong under k=60:**
  - L10-30 c+/c- are positively correlated (+0.15 to +0.28); at L30 this becomes -0.07 at k=10.
  - L40 antipodality is underestimated: rho is -0.36 vs -0.77, and P@10 into the negated partner is 0.46.

### F2: the L10 causal measurement is underpowered, so "L10 no signal" carries no information about the labels

- **The effect is at the noise floor.** kA KL median is 0.0022, against a bf16 chaos floor of 0.0008-0.0018 (N1A_02 'floor' poles).
- **The perturbation is tiny.** The k=60 residual change is about 1.7% of the residual norm (`||post_norm(y)||` is 12.6, `||resid||` is 238).
- **It cannot be fixed by turning k down.** k=1 has the same KL as k=30 (N1A_02_calib).

L20-30 does not have this problem: KL there is 50-140 times the floor. The L10 cell should be reported as "not measurable with this protocol" rather than lumped with L20-30.

### F3: the "token-level responses are anti-correlated (-0.36 to -0.54)" result is mechanical

`rho(Rin+, Rin-)`, with `Rin±(t) = E[relu(±act) | token t]`, is negative by construction (`N1V_audit_06_rin_mechanical.py`):
- pure noise with no token dependence gives -0.44;
- a single-axis surrogate (token mean + noise) gives -0.67 to -0.95;
- real neurons give -0.34;
- two independent surrogates give -0.02, which is the null analogue used in N1C (about +0.03).

So the real-vs-null gap says nothing about the poles being "distinct jobs".

### F4: input-trigger overlap cannot show the poles are "unrelated"

A pole's top-1% positions are by definition disjoint from the other pole's. A token that drives one sign is unlikely to drive the other, so same-neuron pairs are biased below the other-neuron null.

Equal overlap (0.015 vs 0.016) shows the poles fire on *different* tokens. That fits both "unrelated jobs" and "opposite ends of one axis" (e.g. 'on' / 'off'). Calling them "unrelated" needs a semantic test.

### F5: the N1C_08 tightness metric cannot separate J++ prior clusters from semantics

Isotropic random vectors decoded through J++ (HTML closing tags and similar) are as tight as real neurons (`N1V_audit_07_tightness_iso.py`):
- **L40-57:** real minus iso = +0.009 [-0.011, +0.029]; iso minus mix512 = +0.048 [+0.033, +0.063].
- **L10:** iso 0.106 vs real 0.070.

The verdict's "+0.057 [+0.039, +0.075]: single neurons are tighter at L40-57" is relative to mix512 only. Mix512 is an anomalously flat baseline at late layers (0.009-0.055).

Under the naive lens neurons are tighter than iso (0.089 vs 0.033), so that part survives. The L10-30 "comes from the lens" argument rests on a non-significant difference (+0.011 [-0.005, +0.027]); it is a reasonable reading, but not shown.

### F6: minor issues (no effect on the conclusions)

- **bf16 output logits.** N1B/N1C static labels come from a bf16 GEMM. This produces ties, but matches the tool.
- **BOS (id 2) is in the attested vocabulary A.** It can never occur as a current or next token, so it is a guaranteed miss; this hits real and null equally.
- **Prior-z.** `(v-mu)/sd` against an isotropic prior effectively ranks tokens by cosine and ignores token norm, which boosts low-norm or rare tokens. The code is correct, and the result that it hurts is the expected mechanism.
- **AUC figures are on different populations.** The predictor AUC (0.84) is within L40-57, while "layer ≥ 40 alone: AUC 0.92" is pooled over all layers.
- **Cleanliness audit.** It is a single non-blind rater (disclosed). The scoring code (`N1C_06`) computes the fractions and Wilson CIs correctly.

## Verdict on the integrator's claims

| Claim | Verdict | Why |
|---|---|---|
| L40-57 J-VINDEX labels predict steering; L10-30 at null | confirmed (L10 part uninformative) | Survives linear k (k10/k3), norm-bypassing steering and antisymmetrisation. L10 effect is at the bf16 floor (F2). |
| J++ adds nothing over naive | confirmed | k10: L40 +0.02, L45 +0.04, L50 +0.00, L55 -0.01; same as k60. |
| Prior-z hurts | confirmed | Code is correct; the effect is mechanical (norm-free ranking). |
| Neatness at L10-30 is a lens property; neurons tighter at L40-57 | weakened | Iso vectors are as tight as neurons at L40-57 under J++; mix512 baseline is anomalous (F5). |
| Weak 'say' property | confirmed (not re-run) | Nulls are correct; the k60 ablation component is shared and subtracted by the null. |
| gate_trigger_tokens carries a small fraction of the signal | confirmed | Code matches the tool (8/8); nulls are correct. |
| Poles: distinct inputs, antipodal outputs at L40-57 | weakened | Antipodality confirmed and stronger at linear k (-0.77 to -0.96). Input distinctness evidence: rho(Rin) is mechanical (F3), overlap is non-diagnostic (F4). L10-30 positive c+/c- correlation is a k60 artefact (F1). |
| Cleanliness depends on pole strength | untested | Qualitative single-rater audit. |
| JV/naive overlap predicts validity | confirmed (with caveat) | The two AUCs are on different populations. |
| Use W_up for filament polarity | untested | It is a recommendation. |

## Files

- Scripts:
  - `experiments/scripts/N1V_audit_01_postnorm.py` (daemon)
  - `experiments/scripts/N1V_audit_02_metrics.py`
  - `experiments/scripts/N1V_audit_03_smallk.py` (daemon)
  - `experiments/scripts/N1V_audit_04_smallk_metrics.py`
  - `experiments/scripts/N1V_audit_05_antisym_cpu.py`
  - `experiments/scripts/N1V_audit_06_rin_mechanical.py`
  - `experiments/scripts/N1V_audit_07_tightness_iso.py`
- Results:
  - `experiments/results/N1V_audit_01_postnorm.json.full.json`
  - `experiments/results/N1V_audit_02_metrics.json`
  - `experiments/results/N1V_audit_03_smallk.json.full.json`
  - `experiments/results/N1V_audit_04_smallk_metrics.json`
  - `experiments/results/N1V_audit_05_antisym.json`
  - `experiments/results/N1V_cache/steer_dir_L*.pt`
  - `experiments/results/N1V_cache/smallk_k*_L*.pt`

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
