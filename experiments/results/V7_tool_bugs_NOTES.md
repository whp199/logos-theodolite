# V7_tool_bugs — adversarial verification of C8 (tool-level bugs)

Model: Gemma-4-31B-it W4A16, accessed through the lab daemon only. Every test calls the tool's own code:
- `rt.run_theodolite_sweep`, `rt.eager_model.to_tokens` / `capture_generation`;
- `backend.server._build_steering_spec_from_req`;
- `theodolite_math.build_monument_frame` / `project_trajectory_to_3d` / `compute_downstream_filaments` / `compute_causal_vector_split`.

The prompts are new and differ from E6's families: 12 chat prompts (`lab.chat`) and 6 raw prompts, including the tool's default demo prompt (`V7_tool_bugs_common.py`). Each script header pre-registers its predictions.

## Verdicts on C8

| C8 sub-claim | Verdict | Key numbers |
|---|---|---|
| Module 1 attention/FFN split is wrong (`layer_scalar`) | **CONFIRMED.** Patch ready and verified. | Tool Δffn vs true FFN write: median cos 0.49; cos < 0.5 in 31/60 layers and < 0 in 6. ffn_dominance class flips in 12/60 layers. Patched: cos ≥ 0.9985 at every layer. |
| Double BOS for chat strings | **CONFIRMED (small effect).** The naive fix is harmful. | Chat strings get ids `[2,2,105,…]` in 12/12. KL 0.010 [0.003, 0.019] nats. CKA +0.055. Filament-set Jaccard 0.78. `prepend_bos=False` on raw text leaves **no** BOS: KL 5.2 nats. |
| Filament coupling *anti*-correlated with causal effect (ρ = −0.44, n = 48) | **REFUTED (sign).** The score is weak, not anti-correlated. | n = 2830 filaments over 8 prompts: ρ(\|C\|, \|Δgate\|) = **+0.41 [0.38, 0.44]**, positive in 8/8 prompts. E6-style sampling reproduces a ~0 ρ (0.06 [−0.10, 0.21]). A principled direct-path score gives **+0.87**; the true gradient gives **+0.93**. |
| Monument-frame azimuth depends on monument order | **CONFIRMED (stronger than reported).** | Azimuth range over the 6 permutations: median 58–99° per layer band, > 30° in 75–98% of cells. The frame is *not* ill-conditioned (κ = 1.75); the cause is Gram-Schmidt asymmetry. |
| Server steering defaults (650 at 5 layers) destroy the model | **CONFIRMED and sharpened.** There is also a direction bug. | 650 is 2.4–7.4× the median residual norm, giving prompt-independent junk in 6/6 prompts. 79% of the server direction's energy sits in the 3 dims with negative final-norm weight (`clamp(min=0.1)` bug). The server direction collapses the model even at scale 2. |
| J++ plume ' she' 50 vs ' he' 2 | not tested | — |

## Ranked bug table

| # | Severity | Location (file:line) | Bug | Reproduction evidence (V7) | Fix |
|---|---|---|---|---|---|
| 1 | **Critical** | `backend/server.py:104-105` (defaults `layers=[29,35,41,47,53]`, `scale=650`); `:166,:186` (`AddSpec(normalize=True)` makes scale the absolute added length; no `position_mask`, so BOS and every generated token are steered); `:171-176` (`1/fnw.clamp(min=0.1)`) | The steering "laser" adds 650 per layer. It also builds the direction with the inverse final-norm weight, and the clamp turns the 3 *negative* fnw dims into the largest multipliers (10, against a median of 0.16). | Median non-BOS ‖h‖ is 237 / 269 / 175 / 142 / 88 at L29/35/41/47/53, so 650/‖h‖ = **2.7 / 2.4 / 3.7 / 4.6 / 7.4**; 650 also exceeds the BOS norms (129–213). At 650, 6/6 prompts give `<mask>×12` (she/he). The default six/eight gives `<|tool_response>×12`. A random direction gives `' Cada'×12`: **magnitude alone breaks the model**. With the server direction, **scale 2 (0.8–2.3% of ‖h‖) already collapses 6/6 prompts** to `\n×12` (KL 15.6 nats), moving she−he log-odds the *wrong* way (median −6.1). **79%** of the final-space direction energy lies in the 3 clamped dims. cos(server dir, fnw⊙ΔW_U) = 0.12 in final space and 0.03–0.14 after pullback. | Use `fnw ⊙ (W_U[t+] − W_U[t−])` with no inverse and no clamp. Scale relative to ‖h‖ (`NormScaledAddSpec`, strength ≤ 0.01) and mask BOS. Project out the quiet subspace. Change the default scale from 650 to ~1% of ‖h‖. Check: a quiet-masked fnw direction at norm 2 gives fluent on-target text ("she is a thoughtful and creative collaborator…", 3/3 chat prompts, Δlog-odds +12), **but collapses at 6.5**. The usable window is narrow, so expose it as a fraction of ‖h‖. |
| 2 | High | `backend/theodolite_math.py:155-156`; call site `backend/engine_runtime.py:660` | Module 1 ignores Gemma-4's `layer_scalar`. The tool uses `post − mid` = s·ffn + (s−1)·mid, which folds the rescale of the whole residual into the "FFN vault". It also reports attention 1/s too large. | 3 prompts, every layer, compared with ground truth captured at `attn_out_post` and `mlp_out_post`. Tool Δffn: median cos **0.49**; < 0.5 in **31/60** layers and < 0 in **6** (e.g. L12 −0.10, L23 −0.03, L28 −0.36, L35 −0.02). ffn_dominance changes by a mean of 0.074 (max 0.30), and the > 0.5 class flips in **12/60** layers. The omitted rescale term is a median **1.23×** the FFN write (23× at L0, 18× at L59). | **`experiments/results/V7_module1_fix.patch`** (applies cleanly with `patch -p1`). Δattn = s(mid−pre), Δffn = post − s·mid, plus a new `delta_rescale_J` = (s−1)·pre @ Jᵀ. The call site passes `layer_scalar`. The server JSON gains `rescale_norm` and `layer_scalar`. Verified: Δffn cos ≥ **0.9985** and Δattn cos ≥ 0.9984 against ground truth at all 60 layers. Reconstruction error ≤ 0.5%. Relative error ≤ 5.4% (bf16 cancellation, L2-24); capturing `attn_out_post`/`mlp_out_post` directly would remove that residual error. |
| 3 | High | `backend/theodolite_math.py:387-407` (Gram-Schmidt), `:415-438` (`build_monument_frame` uses raw `W_U` rows, global basis); `backend/engine_runtime.py:773` | The 3D frame depends on monument order: monument 1 absorbs the shared pronoun component. It ignores the final-norm weight and is not centred. | 18 prompts, all 6 permutations of (she, he, it), re-projected with the tool's own functions; the canonical permutation reproduces the tool's azimuth exactly. **Azimuth range** across permutations: median 71 / 84 / 83 / 88 / 99 / 68 / 59° (L0-9 … L59, chat); > 30° in **75–98%** of cells. **Elevation range**: median 50–74°. Elevation depends only on which monument is last. HUD x[she]−y[he] sign flips with order in 6–35% of cells. The frame captures **2.4–3.9%** of ‖h_J‖. Conditioning is fine: κ(raw she/he/it) = 1.75 and cos(she,he) = 0.40; fnw-weighted κ = 2.65 and cos = 0.68. Raw and effective directions per token have cos 0.38–0.53. | Symmetric (Löwdin) orthonormalization (W Wᵀ)^−½ W, which is permutation-invariant (verified to 1e-7), or dual-basis coordinates. Use effective directions fnw⊙W_U applied to normed h_J. Display the captured fraction. Prefer contrastive axes such as fnw⊙(W_she−W_he) and subtract a background mean. |
| 4 | Medium | `backend/theodolite_math.py:333-335` (score = ⟨σW_down⊙w_post, W_gate⊙w_pre⟩·\|a_dst\|); `backend/engine_runtime.py:729-736` | Filament coupling ignores the source activation, the layer scalars (Π s), both RMSNorm scales/Jacobians, W_up and the actual pre-activations. Its scale is badly miscalibrated across layers, so the top-\|C\| list is dominated by L0 → L2-6 edges. The filament count (354) is fixed by the caps. | Every pole at the target position was zero-ablated (60×15 = 900 sources, batched; clean-row noise = 0) across 8 prompts (≈ 610k candidate pairs; 2830 tool filaments). Results: (1) Tool filaments: ρ(\|C\|, \|Δg\|) = **+0.41 [0.38, 0.44]** (cluster bootstrap over prompts [0.34, 0.47]); ρ with \|Δa\| = +0.55. Per prompt +0.26…+0.55. (2) Direct-path score D_gate (actual a_i, s_l·Π s_k, exact RMSNorm Jacobians at y_l and mid_l′): **+0.87 [0.86, 0.89]**. D_act (adds W_up and the actual g, u): **+0.84** with \|Δa\|. True autograd gradient: **+0.93 [0.91, 0.94]** (n = 1419). (3) In L0-9, ρ(C) = 0.15 against D_gate 0.88. The top-20 \|C\| (149–395) are all L0 edges with Δg/g of 0–8% (median 1.2%; 5/20 are exactly 0 at bf16 resolution). D_gate tracks them (Pearson r = 0.92 on magnitudes). (4) Precision@6 per source layer: tool **0.24**, D_act 0.54, chance 0.005. Selected filaments have 8× the median \|Δa\| of unselected candidates (AUC 0.81). (5) Sign: −sign(dot) is right 92% of the time. Median effect is 3–4% of the destination pre-activation. | Replace C with the direct-path estimate D (formula in `V7_tool_bugs_03_filaments.py`; it needs only the clean forward's y_l, mid_l′, g, u). Call it a "direct-path linear estimate", not a "verified live circuit". Optionally verify with batched ablation (≈ 30 s per prompt). |
| 5 | Medium (usage) | `backend/server.py` (`SweepRequest.prompt` is sent raw; no `apply_chat_template` in `backend/`) | Without a chat template the tool runs the instruct model as a raw LM. | Greedy continuations on raw prompts are degenerate in 2/6 (`" own own own…"`, `" the same same…"`). This reproduces with HF `generate`, so it is not an eager-path artifact. The same requests through the chat template are fluent. **No steering leak:** module hook count is 0 before and after a steered `capture_generation`, and unsteered outputs are identical after it. | Add a chat-template option in the server, on by default for IT checkpoints, and pair it with the BOS fix. |
| 6 | Low–Medium | `backend/engine_runtime.py:568,583`; `backend/mcp_scientist.py:94,199` | `to_tokens(prompt, prepend_bos=True)` prefixes a second `<bos>` to template strings. | 12/12 chat prompts: ids `[2,2,105]` against `[2,105,…]`. Next-token KL **0.010 [0.003, 0.019]** (max 0.053). Top-1 agrees 12/12. 6-token greedy identical in 11/12 (the 12th differs only after `<turn|>`). CKA mean off-diagonal 0.608 → 0.663 (Δ +0.055 [0.053, 0.056]). Pole-set Jaccard 0.84–0.92 per band (1–2 of 15 poles change). Filament Jaccard 0.78. J++ she−he logit shift 0.08–0.37 (sign flips ≤ 8%). Azimuth median shift 0.4–1.9° (max 98°). **Naive fix is harmful:** `prepend_bos=False` on raw text gives no BOS, with KL **5.2 [3.2, 7.1]** nats, top-1 0/6, and garbage generations. | `experiments/results/V7_bos_fix.patch`: skip the prepend only when the prompt already starts with `<bos>`. It applies cleanly together with the Module-1 patch. |

## Methods and numbers by sub-task

### (a) Double BOS (`V7_tool_bugs_01_bos*.py`, summary `V7_tool_bugs_04_bos_monument_summary.json`)

Method:
- 12 chat prompts were swept through `rt.run_theodolite_sweep` twice: tool default `prepend_bos=True`, then `False` (which yields the correct single BOS).
- 6 raw prompts were swept with `True` (correct) and `False` (no BOS).
- Real next-token KL was computed on the exact ids the tool used.

The table above has the numbers. Pre-registered A1, A2 and A4 hold. A3 is only weakly true: CKA moves, but azimuth and lens readouts barely move.

### (b) Monument frame (`V7_tool_bugs_01_bos*.py`, `V7_tool_bugs_05_m1patch_frames.py`)

Method:
- Each single-BOS bundle was re-projected under all 6 permutations of (she, he, it) with the tool's `build_monument_frame` and `project_trajectory_to_3d`. The max |Δaz| of the canonical permutation against the bundle is exactly 0.
- Löwdin coordinates were checked for permutation invariance (max error 1.2e-7).

Sign agreement with the J++ she−he logit:
- Gram-Schmidt she-first: 74–86%;
- he-first: 79–85%;
- Löwdin: 75–89%.

Even the order-free frame agrees only ~80%, because it uses raw W_U without fnw. B1 holds. The "ill-conditioned frame" hypothesis is refuted: κ = 1.75, against 1.2 for the default spider/eight/insect.

### (c) Steering (`V7_tool_bugs_02_steer*.py`, summary `V7_tool_bugs_02_steer_summary.json`)

Method:
- 6 prompts: 3 raw and 3 chat.
- Directions: server she/he, server default six/eight, random, and quiet-dim-masked fnw-correct.
- Scales: 2–650.
- Each run is a 12-token greedy generation through `rt.eager_model.capture_generation`, the same call the sweep makes.

| direction @ scale | degenerate (≤ 2 distinct tokens) | distinct outputs across 6 prompts | median KL (first token) | median Δ target log-odds |
|---|---|---|---|---|
| none | 2/6 (the two raw prompts, see bug 5) | 6 | 0 | 0 |
| server she/he @ 2 / 6.5 / 20 | 6/6 each | 1 (`\n×12`) | 15.0–15.6 | −6.1 / −4.9 / −1.2 |
| server she/he @ 65 / **650** | 6/6 | 1 (Bengali junk / `<mask>`) | 15.5 / 13.0 | +4.4 / −2.9 |
| server six/eight @ 6.5 / 65 / 650 | 6/6 | ≤ 3 | 14–16 | +8.5 / +9.1 / +1.9 |
| random @ 6.5 / 65 / 650 | 6/6 | ≤ 2 | 11.5 / 15.7 / 14.3 | −3.2 / −2.4 / −8.9 |
| quiet-masked fnw @ 2 | 2/6 (the same raw prompts) | 6 | 1.5 | **+12.0**, fluent "she" on 3/3 chat prompts |
| quiet-masked fnw @ 6.5 / 20 / 65 | 4/6, 6/6, 6/6 | ↓ | 12–16 | +10…+34 (junk) |

S1–S4 all hold.

### (d) Filaments (`V7_tool_bugs_03_filaments.py`, `V7_tool_bugs_06_filgrad.py`, summary `V7_tool_bugs_07_filament_summary.json`)

Method:
- The tool sweep (`max_new_tokens=0`, single BOS) supplies the poles and filaments. The recomputed C matches the tool's `coupling_score` to ≤ 0.26%.
- All 900 poles were ablated at the target position, one source per batch row (B = 32, 2 clean rows; clean-row difference exactly 0).
- For every pole in the next 6 layers, Δ of gate pre-activation g, up pre-activation u and post-GeGLU activation a was recorded.
- Static scores: C_tool, |dot|, D_gate, D_act.
- True gradients −a_i·∂g_j/∂a_i and −a_i·∂a_j/∂a_i: autograd through the W4A16 recompute linears, same position, all paths, every even destination layer.

Results by prediction:
- **P1 is wrong in the interesting direction:** ρ(C_tool) = +0.41, not ≤ 0.1.
- **P2 holds:** D_gate 0.87 and D_act 0.84.
- **P3 is partly wrong:** the tool's precision@6 (0.24) is well above chance, though far below D_act (0.54).

Why E6 saw −0.44:
- E6 used 48 filaments from 2 prompts, taking the top-4 by |C| per 10-layer band. That design restricts the range within each band and over-represents the L0 band.
- Applying the same sampling to my data gives ρ = 0.06 [−0.10, 0.21], with per-prompt values from −0.27 to +0.48. E6's sign is not robust.

Caveats:
- The measured Δg is bf16-quantized (median |Δg| = 0.125, about 2 ulp at |g| ≈ 10). 11% of L0-9 filaments have Δg exactly 0, which partly explains E6's "zero effect" for the top filaments. This quantization caps every ρ below 1.
- All candidate pairs (610k): C_tool 0.36–0.46, D_act 0.61, gradient 0.81.

### (e) Module-1 fix review (`V7_tool_bugs_05_m1patch_frames.py`, patch `V7_module1_fix.patch`)

E3's formula is correct: post − pre = s(mid−pre) + (post − s·mid) + (s−1)·pre exactly. It follows from `Gemma4TextDecoderLayer.forward`, which ends with `hidden_states *= self.layer_scalar`; the config has no per-layer-input and no MoE, so no extra terms enter.

Review points folded into the patch:
1. The function keeps its signature and adds `layer_scalar=1.0`, so it stays backward compatible.
2. `delta_rescale_J`, `rescale_norm` and `layer_scalar` are real dataclass fields with defaults; E3 had monkey-patched an attribute.
3. The single call site, `engine_runtime.py:660`, now passes the buffer.
4. The server JSON exposes `rescale_norm`, because the rescale term is as large as the FFN write. The binary wire format is unchanged.

The patch was verified by loading the patched module from a scratch copy; `backend/` was not modified. Numbers are in the table. The remaining ≤ 5.4% error comes from bf16 cancellation in post − s·mid at L2-24; direct capture of `mlp_out_post` and `attn_out_post` (2 more hook points per layer) would make the split exact.

## Notes and limitations

- The two steering jobs ran about 450 s each, longer than the ~4 min guideline, because eager generation is slow (~9 s per 12 tokens). Later jobs were 25–75 s.
- Filament effects are single-neuron, single-position ablations; the ground truth includes indirect paths. D_gate is direct-path only, and the gap between D_act (0.85) and the gradient (0.93) is the share carried by indirect paths.
- Steering conclusions rest on 6 prompts and greedy decoding.
- Each daemon job's namespace forms a reference cycle (functions defined in the namespace point back to it through `__globals__`), so GPU tensors from finished jobs live until garbage collection. Other agents hit OOM while this was happening. A `gc.collect()` after each job in `labd.py` would help; that file was not edited here.

## Files

Scripts are in `experiments/scripts/`:
- `V7_tool_bugs_common.py`
- `_00_basics.py`
- `_01_bos.py` and chunk wrappers
- `_02_steer.py` and `_p0` / `_p1`
- `_03_filaments.py` and `_p0`..`_p7`
- `_04_analyze_bos.py` (offline)
- `_05_m1patch_frames.py`
- `_06_filgrad.py` and `_p0`..`_p7`
- `_07_analyze_fil.py` (offline)
- `_08_diag.py`

Results are in `experiments/results/`:
- `V7_tool_bugs_00_basics.json`
- `V7_tool_bugs_01_bos_{chat0_4,chat4_8,chat8_12,raw}.partial.json`
- `V7_tool_bugs_04_bos_monument_summary.json`
- `V7_tool_bugs_02_steer_{p0,p1}.partial.json`, `V7_tool_bugs_02_steer_summary.json`
- `V7_tool_bugs_03_fil_<prompt>.npz` (8 files) and `.quick.json`
- `V7_tool_bugs_06_filgrad_<prompt>.npz` (8 files)
- `V7_tool_bugs_07_filament_summary.json`
- `V7_tool_bugs_05_m1patch_frames.partial.json`
- `V7_tool_bugs_08_diag.json`
- `V7_module1_fix.patch`, `V7_bos_fix.patch`

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
