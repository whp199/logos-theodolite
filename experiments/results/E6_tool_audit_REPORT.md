# E6_tool_audit — End-to-end audit of the Logos Theodolite tool on the "Gemma-4-31B is female-coded" question

Model: Gemma-4-31B-it, 4-bit QAT W4A16 (lab daemon). Lenses: J++ (`jpp`), Jacobian lens (`jlens`), and the naive logit lens.
Everything below was produced by calling the tool's own code paths (`rt.run_theodolite_sweep`, `LarqlVaultEngine.execute_lql`,
`mcp_scientist.test_motif_hypothesis`, `server._build_steering_spec_from_req`) inside daemon jobs, then checking them against
controls, other lenses, the real model output, and causal interventions.

## TL;DR

1. **The naive picture.** Point the tool at Gemma writing about itself in the third person ("Gemma ... In conversations, ▢").
   The J++ "ghost-token plumes" show **` she` as the top-1 token in 50 layer×prompt cells and ` he` in only 2** (12 prompts, L30-58).
   Tracked ranks put `she` above `he`, the HUD shows x[she] > y[he], and LARQL finds "she neurons" that `test_motif_hypothesis`
   marks as `CONFIRMED`. A naive user would conclude "J-space says Gemma is female".
2. **What the evidence supports.** The gender readout at the self-slot follows **the name, not some built-in self-model**.
   On a woman-persona→man-persona scale (0 = man persona, 1 = woman persona) at the third-person pronoun slot, with the real model output at L59:
   **Gemma-self 0.62 [0.58, 0.65]; "Atlas"-self 0.27 [0.25, 0.31]; nameless "The assistant" 0.43 [0.40, 0.46];
   "Gemma" as a human friend 0.96 [0.93, 1.00].** J++, JLens and the naive lens agree at L40-58: J++ L50-58 gives 0.60 / 0.29 / 0.46 / 0.94,
   and JLens and naive land within about 0.07 of those (except naive on human-Gemma, 0.98).
   In actual behaviour Gemma **avoids gendered pronouns for itself**: P(she-set)=0.009, P(he-set)=0.006, P(it-set)=0.19 [0.07, 0.34], P(they-set)=0.14,
   and the top-1 token is the name "Gemma" in 9 of 12 prompts. The real-output F-vs-M log-odds is +1.48 [+0.68, +2.19], compared with −5.10 for Atlas and +8.38 for human-Gemma.
   At first-person "I" tokens the default assistant sits at the **midpoint (J++ L30-39: 0.43 [0.41, 0.45])**. That is indistinguishable from a
   **non-self background token** (0.42 [0.39, 0.44]; paired difference +0.42 [−0.14, +0.96] logits, against a 28.9-logit persona gap).
   Verdict: "female-coded" is **mostly a name prior ("Gemma" is a common female given name) plus display amplification.**
   About a third of that prior survives in the AI-self context: (0.62−0.43)/(0.96−0.43) ≈ 0.36. There is no evidence of a gendered self-representation beyond that.
3. **The tool has many correctness problems that matter for this kind of conclusion.** In short:
   - a double BOS on chat prompts;
   - an order-dependent, fnw-ignoring 3D frame that captures only 1-16% of the state;
   - a wrong attention/FFN split;
   - pole "logits" pinned at the softcap;
   - filament coupling scores that are *anti*-correlated with causal effect (ρ = −0.44);
   - LARQL "cosine similarities" of 0.0004-0.02 at L30+, with orthographic false positives;
   - an MCP verdict rule that "CONFIRMS" at n=3;
   - steering defaults that are 2.6-7.8× the residual norm and point in the wrong direction (cos 0.03-0.14 with the gradient direction). They turn the model into `<mask><mask>...`.

   The bugs are listed below with file:line references and fixes.
4. **New structural finding (beyond the joke question).** Gemma-4-31B's residual stream has **"quiet" dimensions**: about 10% of coordinates with near-zero natural variance
   (std down to 0.002, against a median of ~1) and **catastrophic downstream gain**. A random direction of norm 2 per layer, about 1% of ‖h‖, collapses the model.
   Confined to loud dimensions it does nothing even at norm 10.
   J++/JLens pullbacks put **99-100% of their energy in this quiet subspace at L35-53**. The quiet coordinates (≤1.2% of ‖h‖²) supply **25-52% of ‖h·J^T‖**.
   This is why lens-derived steering, and likely the low VJP-vs-pullback agreement, fail. Masking the quiet dims turns the same gradient direction into clean steering:
   P(she) = 0.985 with entropy 0.07.
   Separately, `layer_scalar` makes the residual stream leaky: Π s after L29 = 4e-5.

## Methods (short)

- **Prompts** (`experiments/scripts/E6_tool_audit_prompts.py`), 12 per condition, matched across conditions by index:
  - **Family A** (first-person self; the target is the final " I" of an assistant prefix such as "Honestly, I"):
    - `A_default`: no system prompt;
    - `A_woman` / `A_man`: Sarah or James 34-year-old persona;
    - `A_bg`: the same prefix ending in " the", a non-self background token;
    - `A_raw`: the same text with **no chat template**, which is what the tool UI actually sends.
  - **Family B** (third-person pronoun slot): user says "Write a short description of yourself in the third person." and the assistant prefix is
    "{N} <trait sentence>. <clause>," so the next token is a pronoun or name slot. Conditions:
    - `B_gemma` (N=Gemma, no system prompt);
    - `B_atlas` (system "You are Atlas...");
    - `B_woman`, `B_man` (personas);
    - `B_gemma_human` ("my friend Gemma");
    - `B_nameless` (N="The assistant").
- **Tool sweeps.** `rt.run_theodolite_sweep` was run with naive defaults (`prepend_bos=True`, monuments she/he/it, tracked she/he/it/her/him/woman/man/they,
  max_new_tokens=4): 132 sweeps. Recorded per prompt: J++/naive top-5 plumes and tracked ranks/logits, 3D trajectory (xyz, azimuth, elevation),
  the Module 1 split, pole labels, filaments, and CKA.
- **Principled readouts** used the correct single-BOS tokenization at the same position:
  - lens pronoun log-odds `pron` (mean over 6 F/M pronoun pairs) for J++, JLens and naive;
  - real next-token probabilities.
- **Statistics.** Unit = prompt (per-prompt band means first). Bootstrap 95% CIs use 2000 resamples. Scale positions are bootstrapped jointly over prompts.
- **Interventions:**
  - tool `.patch DELETE` of LARQL neurons, always followed by `ROLLBACK` in the same job (verified restoration: max |Δ| = 0.0);
  - zero-ablation of filament source neurons at the target position;
  - server steering specs.

## 1. What a naive user would see (tool displays) — and why each one misleads

| Display | B_gemma (Gemma-self slot) | Control that reveals the problem |
|---|---|---|
| J++ plumes, top-1 over L30-58 (cells = layer×prompt) | ` she` 50, ` he` 2, ` they` 55 | Real output: P(she) = 0.009. Atlas-self: ` he` 35 / ` she` 2. Nameless: ` it` 53 / ` they` 87 / ` she` 2. Output top-1 is the name. Lens top-1 matches the output top-1 in 0-1% of cells at L30-49 for B_gemma. |
| J++ tracked rank, log10(rank_he) − log10(rank_she), L40-49 | +1.04 [0.86, 1.24] | Woman +1.74, man −1.67, Atlas −0.40 → scale position 0.80 |
| HUD x[she] − y[he], L40-49 (order she,he,it) | +0.08 [0.08, 0.09] | **Atlas (+0.04) and "The assistant" (+0.05; output is "it" 66%) are also positive, and the man persona is ≈ 0 (−0.01)**: monument-1 bias. With order (he,she,it): Atlas −0.03, nameless −0.02. |
| Azimuth θ, L40-49 | 15° (order she,he,it) vs **51°** (order he,she,it) | The number depends on monument order. All B prompts sit at ~14° at L10-19, so the background lies "near she". All A prompts at L10-19 lie he-side: a token-type background. |
| 3D frame coverage | 4-8% of ‖h_J‖ (A family: 1-3%) | Angles describe a tiny subspace |
| Pole J-VINDEX labels (gendered labels F/M, 12 prompts) | 17 / 28 | Logit values 20-30 are softcap artifacts (§3.4) |
| LARQL `POLES NEAREST 'she'` + `test_motif_hypothesis` | L50 n5313 "CONFIRMED_SOUTH_POLE", d = −22 at n=3 | That neuron is real and causal. It is a *bipolar* she/he neuron (it is also in the 'he' list) and does not mark Gemma as female (§2.3). |

First-person displays (A family): no gendered token appears in the J++ top-5 at the default self "I" token in any layer. The woman persona shows 383 female tokens, the man persona 65 male.
The trajectory angles at L10-19 are he-leaning for every "I" token, including the woman persona (angle-cos difference −0.52).

## 2. What is justified

### 2.1 Woman/man scale, lens-robust (`E6_tool_audit_09_lenses_summary.json`)

Pronoun log-odds at the last position, single BOS. Position = (x − man)/(woman − man).

| Condition | J++ L40-49 | J++ L50-58 | JLens L50-58 | naive L50-58 | **real output (L59)** |
|---|---|---|---|---|---|
| B_gemma (self) | 0.71 [0.67, 0.75] | 0.60 [0.56, 0.63] | 0.62 [0.58, 0.66] | 0.58 [0.54, 0.61] | **0.62 [0.58, 0.65]** |
| B_atlas (self) | 0.27 | 0.29 | 0.22 | 0.33 | **0.27** |
| B_nameless (self) | 0.53 | 0.46 | 0.42 | 0.48 | **0.43** |
| B_gemma_human | 0.91 | 0.94 | 0.90 | 0.98 | **0.96** |
| woman − man gap (logits) | 21.6 | 30.9 | 29.9 | 39.0 | 20.4 |

| Condition (A, self "I" token) | J++ L20-29 | J++ L30-39 | JLens L30-39 | naive L30-39 |
|---|---|---|---|---|
| A_default | 0.50 [0.49, 0.52] | 0.43 [0.41, 0.45] | 0.36 [0.33, 0.38] | 0.77 [0.66, 0.89] |
| A_bg (non-self token) | 0.42 | 0.42 | 0.40 | 0.47 |
| A_raw (no chat template) | 0.46 | 0.41 | 0.43 | 0.03 |
| woman − man gap | 27.9 | 28.9 | 21.2 | 8.2 |

Caveat on the A family at L50-58: J++ gives A_default 1.55 ("more female than the woman persona"). The persona gap there is only 1.9 logits, so the scale is
ill-conditioned. The raw value (+3.7) appears at L50-58 for the default "I" token but not for `A_bg` (paired +3.73 [2.31, 4.91]). It replicates under JLens (+3.72).
It is a real late-layer she>he logit tilt at the first-person token, but it is persona-independent (woman +2.65, man +0.79) and it vanishes at the output (L59: +0.11).
I do not interpret it as a gendered self-representation.

### 2.2 Behaviour at the self slot (real model, single BOS; `E6_tool_audit_07_summary.json`)

| Condition | P(she-set) | P(he-set) | P(it-set) | P(they-set) | log-odds F/M | top-1 |
|---|---|---|---|---|---|---|
| B_gemma | 0.009 | 0.006 | 0.19 | 0.14 | +1.48 [0.68, 2.19] | "Gemma" 9/12, "it" 2/12 |
| B_atlas | 0.001 | 0.089 | 0.04 | 0.05 | −5.10 [−5.80, −4.44] | "Atlas" 12/12 |
| B_nameless ("The assistant") | 0.0002 | 0.0009 | **0.66** | 0.04 | −1.02 [−1.39, −0.60] | "it" 9/12, "the" 3/12 |
| B_gemma_human | 0.78 | 0.0004 | 0.003 | 0.004 | +8.38 | "she" 12/12 |
| B_woman / B_man | 0.91 / 0.00 | 0.00 / 0.89 | ≈0 | ≈0 | +12.23 / −13.39 | she / he |

### 2.3 LARQL "gender neurons" (`E6_tool_audit_04_vault.partial.json`)

- 102 neurons (top-10 `NEAREST 'she'` ∪ `'he'` at L20/30/40/50/55/58) were tested on 12 matched female-name vs male-name next-pronoun contexts
  with a paired sign-flip permutation test. **35/102 are selective at BH q < 0.05** (30 have |d| ≥ 0.8). So the query finds real gender neurons, but
  most hits are not selective, and several are orthographic neighbours (` shed`, ` usher`, ` heap`, ` helix`, quote marks).
- On the selective neurons, Gemma-self sits at a median **0.57** on the B_man→B_woman scale, and human-Gemma at **0.94**.
- Causal check (tool `DELETE`, rolled back): removing the top-10 'she' neurons at L50 changes F/M log-odds by:

  | Prompts | Δ log-odds | Baseline |
  |---|---|---|
  | female contexts | **−1.80** | 7.26 |
  | male contexts | +0.90 | |
  | human-Gemma | −2.28 | 8.41 |
  | Gemma-self | −0.33 | 1.54 |

  Ten random neurons give |Δ| ≤ 0.04. The 'he' list at L50 gives an identical effect because it shares the bipolar neuron 5313.

## 3. Tool bugs and validity problems (with evidence and fixes)

1. **Double BOS on chat prompts.** `backend/engine_runtime.py:568,583`: `to_tokens(prompt, prepend_bos=True)` puts an extra `<bos>` in front of a template that already starts with
   `<bos>`. Verified: ids `[2, 2, 105, ...]` in 120/120 chat sweeps. The same pattern is in `mcp_scientist.py:94,199`.
   - Effect: KL(single‖double) on the next token is 0.00-0.02 nats on average (max 0.15); top-1 agreement is 83-100%.
   - Lens pron shifts by up to +0.43 [0.29, 0.55] at L40-58.
   - Mean off-diagonal CKA rises from 0.60 to 0.66; it returns to 0.60 when the BOS tokens are dropped.
   - In the demo, the L59 azimuth moved from −2° to −34°.
   - *Fix:* detect a leading `<bos>`, or tokenize with `add_special_tokens=False`. Better still, apply the chat template in the server.
   - Related: **the tool never applies a chat template** (no `apply_chat_template` anywhere in `backend/` or `frontend/`). UI prompts therefore run the instruct model as a raw LM (`A_raw`).
2. **Module 1 split ignores `layer_scalar`.** `theodolite_math.py:155-167`. Confirmed in tool outputs:
   - cos(tool Δffn, true FFN write) is 0.17-0.38 at L20-39 (band means over 132 sweeps), 0.89-0.93 at L50-58 and 0.18-0.35 at L59;
   - in the demo it was −0.37 at L12;
   - the FFN-dominant classification (ratio > 0.5) flips in 11-31% of layers at L10-29.

   The cyan/gold arrows are wrong in direction at most early and mid layers.
   - *Fix:* use `attn = s·(mid−pre)`, `ffn = post − s·mid`, and report the rescale term `(s−1)·pre` separately.
3. **Module 4 frame:**
   - (a) It uses raw `W_U[t]` (`theodolite_math.py:432-436`), not the effective logit direction `fnw ⊙ W_U[t]`. Their cosine is only 0.40 (she), 0.38 (he) and 0.53 (it), because the final-norm weight ranges from −2.4 to 512 (max/median 80).
   - (b) Gram-Schmidt order gives monument 1 the full shared pronoun component. `cos_raw(she, he)` = 0.40 (κ = 1.75, so not ill-conditioned), but the HUD's x[she] vs y[he⊥] comparison is biased: Atlas reads "she > he". The azimuth value itself changes by 30-60° under permutation.
   - (c) The frame captures only 1-16% of ‖h_J‖.
   - (d) The projection is not mean-centred, so token-type background dominates.
   - (e) `frontend/src/components/BallisticTrajectory.tsx:35` floors `maxCoord` at 1.0 while |x| ≤ 0.15, so the trajectory is drawn almost on the axis.
   - *Fix:* project onto contrastive, fnw-weighted directions (e.g. fnw⊙(W_she − W_he)); subtract a background mean; show the captured-variance fraction; drop the floor.
4. **Module 2 pole logits.** `theodolite_math.py:74` (eps hack) plus RMS-normalizing a single direction pins displayed logits at 20-30, near the softcap. With an honest eps = 1e-6 the maxima are 0.03-16. Top-3 token identity is kept in 87-100% of cases, so the labels survive but the magnitudes mean nothing.
   - Pole selection by |act| agrees with selection by write norm (Spearman 0.87-0.98; top-15 overlap 12-14). That is fine.
   - *Fix:* show ranks or logit-differences relative to a baseline, not saturated logits.
5. **Module 3 filaments are not a "verified live circuit."** `theodolite_math.py:333-335`; `engine_runtime.py:729-736`.
   - Each sweep returns about 354 filaments: a constant fixed by the caps (6 per source layer × 59), so the count carries no information.
   - I ablated the source neuron for 48 filaments. The highest-scoring ones (L0→L6, |C| = 80-150) had **zero effect**.
   - Spearman(|C|, |Δgate_dst|) = **−0.44** (relative effect: −0.55).
   - Sign prediction is right in 90% overall (L0-9: 50%; L10+: 94-100%). Median |Δgate|/|gate| is 6% (24% at L50+).
   - The score ignores the source activation, layer scalars (Π s over L1-L5 ≈ 0.06), the norms, and attention. It multiplies by |post-GeGLU act| of the destination.
   - *Fix:* score = a_i · s_l · ⟨W_down_i ⊙ w_post / rms_ffn, W_gate_j ⊙ w_pre / rms(mid_l')⟩. Validate by ablation and rename it.
6. **LARQL `NEAREST`.** `larql_vault.py:141-158`.
   - The reported cosines are 0.0004-0.02 at L30-58. That is at or below the typical random-vector |cos| of 1/√5376 ≈ 0.014, even for the best of 21504 neurons. The likely reason, not separately verified: the query `u·J[l]` puts 99-100% of its energy in the quiet subspace at L35+ (§4b), where W_down columns carry little energy. At L20, where the pullback is less concentrated, cosines are 0.03-0.19. Ranking still finds some real gender neurons (§2.3).
   - The 'she' and 'he' result lists include orthographic neighbours, and `EDGES` and `POLES` are the same query (`:396`, `_kind` is ignored).
   - *Fix:* rank by the contrastive logit effect `⟨w_down⊙w_post·J^T, fnw⊙(W_she−W_he)⟩` and verify on activations.
7. **`test_motif_hypothesis`.** `mcp_scientist.py:221-238` uses population variance, no significance test, last-token only, and the label "CONFIRMED" at |d| ≥ 0.8 for any n. n=3 gave d = −22.
   - *Fix:* permutation p-value, minimum n, and multiple-comparison control.
   - Also `mcp_scientist.py:45`: an unbound call loads a second 31B model (OOM hazard).
8. **Server steering defaults.** `server.py:104-105,171-176`.
   - Scale 650 at 5 layers is **2.6-7.8×** the median residual norm, and the direction uses `1/fnw` instead of `fnw`. Its cosine with the gradient-correct direction is **0.03-0.14**, and its lens-predicted effect per unit norm is 7-35× smaller.
   - Through the tool's own sweep path, she→he steering at 650 makes the output `<mask><mask>...` for every prompt. Even 65 collapses it to one repeated token, identically across the woman, man and Gemma prompts. A random direction at 650 also collapses it.
   - *Fix:* use `fnw ⊙ ΔW_U` pulled back through J, then **project out the quiet subspace** (or whiten by the per-dim residual std). Scale relative to ‖h‖ and skip the BOS position. Better still, use data-derived contrastive directions. See §3a: with quiet dims masked, norm 2 gives P(she) = 0.985 and fluent text.
9. **Tracked-rank hack.** `engine_runtime.py:616-685`: when the first generated token is whitespace, J++ ranks take min(rank@t, rank@t+1). That is optimistic selection, and it applies to J++ only, not naive, so J++-vs-naive comparisons are biased.
   It never triggered in our 132 sweeps (120 chat + 12 raw). Per its own comment, it exists for the default digit demo, where the whitespace-then-digit pattern makes it fire.
10. **CKA.** `theodolite_math.py:584-590`: the single-token path is a squared centred cosine, not CKA. The multi-token path includes BOS and template tokens; see the double-BOS inflation above.
11. **think mode.** `server.py:464`: prepending `"<|think|>\n"` to a raw prompt is not Gemma-4's thinking protocol.
12. **LARQL INSERT.** `larql_vault.py:280-361`: the docstring says it also edits gate_proj, but only a down_proj column is changed. "MEMIT" is a rank-1 column add with no covariance solve.

### 3a. Steering scan (`E6_tool_audit_06/10/11/12`)

Real next-token readout on 12 prompts each, with steering added at resid_post of L29/35/41/47/53 at every non-BOS position.
Baseline entropy at the B_gemma slot is 0.87 nats; baseline F/M log-odds are +1.54 (Gemma), −5.09 (Atlas) and −13.33 (man).

| Direction (unit) × scale per layer | B_gemma F/M | B_man F/M | entropy | top-1 (all prompts) |
|---|---|---|---|---|
| server (1/fnw), 650 (default) | 0.0 | 0.0 | 11.1 | `<mask>` |
| server, 65 / 20 / 2 / 0.5 | 7.6 / 3.7 / −1.8 / −0.2 | same as Gemma | 8.0 / 6.1 / 5.3 / 6.4 | junk / `\n` |
| grad (fnw), 40 / 10 / 2 / 0.5 | 14.8 / 8.5 / 4.1 / 5.1 | ≈ same | 6.8 / 6.9 / 1.2 / 2.3 | `allError` / ` same` |
| **random**, 10 / 2 / 0.5 | −3.3 / −1.5 / +0.55 | −3.4 / −2.8 / −15.9 | 4.2 / 3.2 / 1.5 | `<\|channel>` (≥2) |
| **data-derived mean(B_woman − B_man) resid diff**, 0.5 / 2 / 10 | **3.5 / 11.0 / 25.7** | −11.5 / −6.2 / +21.0 | **0.89 / 1.04 / 0.30** | Gemma / Gemma / **she** (P = 0.90) |
| grad at L29 only, 0.5 / 2 | 9.2 / 11.2 | −11.0 / +12.0 | 0.97 / 0.16 | she (P = 0.47 / 0.91) |
| grad at L53 only, 0.5 | 6.8 | 2.4 | 4.1 | `<\|channel>` |

The tool's default (650) destroys the model; so do 10× and 300× smaller scales. The lens-derived directions (server *and* gradient-correct) collapse
the output to one prompt-independent token at norms of **0.5-2 per layer, i.e. 0.3-1% of ‖h‖**. **A random direction does the same.**
A data-derived gender direction of norm 10 is benign and flips every prompt to a fluent "she".
The J-pullback directions put 14-23% of their energy on just 5 residual dimensions, where the natural residual is ~10-100× below the median coordinate magnitude.
Their participation ratio is 44-114, against ~1800 for random directions.
This points to strongly anisotropic sensitivity: off-manifold directions carry enormous downstream gain (see §4b).

**Quiet-dimension test** (`E6_tool_audit_12_quiet_dims`). "Quiet" dimensions are the 10% of residual dimensions with the smallest token-to-token std.
At L41-53 their std reaches down to 0.002-0.005, against a median of ~1.

| Direction at norm 2 per layer (L29-53) | share of energy on quiet dims | B_gemma F/M | entropy | top-1 |
|---|---|---|---|---|
| none | — | +1.54 | 0.87 | Gemma |
| random | 10% | +4.2 | 2.2 | ` same` (all prompts) |
| random restricted to quiet dims | 100% | +12.0 | 6.1 | ` singular` (all prompts) |
| random restricted to loud dims (std > median), norm 2 / 10 | 0% | +1.6 / +2.1 | 0.88 / 0.91 | Gemma (unchanged) |
| variance-weighted random, norm 10 | 0.1% | +2.0 | 0.86 | Gemma (unchanged) |
| tool server direction | **50% (L29), 99-100% (L35-53)** | (collapse, see above) | | |
| tool grad direction | **38% (L29), 99-100% (L35-53)** | (collapse) | | |
| grad direction with quiet dims removed | 0% | **+14.1, P(she) = 0.985** (B_man: P(she) = 0.93) | **0.07** | she (fluent) |

So the tool's steering failure is not mainly about scale or the 1/fnw choice. **The J++ pullback directions live almost entirely in the quiet subspace**,
and that subspace has catastrophic downstream gain. Masking it out turns the same gradient direction into a clean, strong, coherent steering vector.

## 4. Structural observations the tool hides

### 4a. Gemma-4's residual stream is leaky

Every layer multiplies the whole residual by `layer_scalar` (0.03-0.99). The product Π_{k>l} s_k is **4.3e-5 after L29** and **0.012 after L53**.
The direct linear carry of resid_post[29] is therefore ≈ 0.09% of the final residual norm, and of resid_post[53] ≈ 9%. Information must be continually re-written,
and late layers dominate the final state. Two consequences for this tool:
- mid-layer "trajectories" are counterfactual lens readouts, not what reaches the output;
- any analysis that subtracts adjacent residuals without the scalar (Module 1) is wrong.

### 4b. Quiet dimensions with catastrophic gain

About 10% of residual dimensions have near-zero natural variance at L35-53. A random unit direction at norm 2 per layer, about 1% of ‖h‖ with only 10% of its energy in these dims, collapses the model to a
prompt-independent token; norm 0.5 already visibly perturbs it. A random direction confined to the quiet dims collapses it at norm 2, while one confined to the "loud" dims does nothing even at norm 10. J++ pullbacks u·J[l] (and the tool's LARQL/steering queries built from them)
put 99-100% of their energy in this subspace at L35+. This likely explains the pilot's low cos(true VJP, J++ pullback) of ~0.05 at L30-40: the averaged-Jacobian lens [2,4] assigns
large weights to low-variance coordinates. It also makes J-space directions, as directions, unsafe as interventions or as neuron-similarity queries without whitening.

**Do the lens readouts depend on quiet dims?** (`E6_tool_audit_13/14`) The quiet-dim deviations carry only 0.001-1.2% of ‖h‖², yet they make up
**25-52% of ‖h·J^T‖ for J++ (25-64% for JLens) at L28-58**. cos(h_J full, h_J with quiet dims clamped to their mean) falls to 0.85-0.97.
Clamping shifts per-prompt pron readouts by 1-3.7 logits (J++) and up to 6 (JLens) on average, and it moves the A-family "background" a lot.
For example, A_default L30-39 raw pron goes from +0.30 to +4.40.
The B-family conclusions survive clamping:

| Condition | J++ L40-49, full → clamped | J++ L50-58, full → clamped | JLens L50-58, full → clamped |
|---|---|---|---|
| Atlas | 0.27 → 0.28 | 0.29 → 0.28 | 0.22 → 0.25 |
| nameless | 0.53 → 0.52 | 0.46 → 0.36 | 0.42 → 0.35 |
| Gemma | 0.70 → 0.67 | **0.60 → 0.44** | **0.62 → 0.48** |
| human-Gemma | 0.91 → 0.93 | 0.94 → 0.95 | 0.90 → 0.95 |

Part of Gemma-self's mild female lean at L50-58 in the lenses therefore sits in quiet coordinates. The real output still gives 0.62.
Absolute lens values and backgrounds should be treated as fragile, especially because the lens was fitted on the bf16 *base* model, whose quiet-dim statistics may differ from the 4-bit instruct model's.

## 5. Pre-stated predictions vs outcomes

- H1, partly right. Display metrics do track persona gender at the B slot once they are calibrated against woman/man controls (HUD scale positions 0.52 for Gemma and 0.26 for Atlas at L40-49). Uncalibrated, they mislead: background, monument order, Atlas reading "she > he", plume amplification.
- H2, right. A_default is at the midpoint at L20-39. At the slot, Gemma prefers it/they and its name over she/he.
- H3, right. Human-Gemma 0.96 vs Gemma-self 0.62.
- H4, right: KL ≤ 0.02 on average.
- V1, wrong: she/he lists overlap in only 1-4 neurons, though they share bipolar neurons.
- V2, mostly right: 35% are selective.
- V3, wrong at L50: a 1.8-logit causal effect, 50-100× random.
- V4, right: 0.57.
- F1, could not be decided: filament |C| beats single random sources (median percentile 1.0), but the top-of-1350 selection makes this expected.
- F2, right on magnitude (ρ = −0.44, negative), wrong on sign (90% agreement).
- F3, half right: logits are artifacts, but |act| ranking is fine.
- F4, right for double BOS.
- S1, right (even 65 breaks the model).
- S2, right (cos 0.03-0.14).
- Job 10 (a), wrong: no scale ≥ 0.5 is coherent for the 5-layer lens directions.
- Job 10 (b), not testable because everything collapses.
- Job 10 (c), right: 650 still breaks the model without BOS.
- Job 11, "random directions are harmless at 2-10": **wrong**. That was the surprise that led to job 12. "J-pullback directions concentrated" and "data-driven direction benign": right.
- Job 12 (a, b, c): right.
- Job 13 (a), right: 25-52% of ‖h_J‖.
- Job 13 (b), right for the B family.

## 6. Caveats

- 12 prompts per condition and template-limited prompt families.
- The B slot measures conditional F-vs-M preference while P(she) and P(he) are both ≈1%.
- The J++ lens was fitted on the base bf16 model. The B-family conclusions replicate under JLens, the naive lens and the real output; the A-family mid-layer values are lens-dependent (naive differs).
- The woman/man scale is ill-conditioned wherever the persona gap is small (A family at L50+, B family at L20-29).
- Single-neuron ablations at the target position only; no attention-path analysis.

## Files

Scripts are in `experiments/scripts/`:
- `E6_tool_audit_01_basics.py` — BOS, geometry, scalars, norms
- `_02_sweep_demo.py`
- `_03_sweeps.py` with chunk wrappers `_c0`..`_c5`
- `_04_vault.py`
- `_05_filaments.py`
- `_06_steering.py`
- `_07_analyze.py` and `_09_analyze_lenses.py` — offline
- `_08_lenses.py`
- `_10_steer_scan.py`
- `_11_steer_diag.py`
- `_12_quiet_dims.py`
- `_13_quiet_readout.py`
- `_14_analyze_quiet.py` — offline
- `_helpers.py`, `_prompts.py`

Results are in `experiments/results/`:
- `E6_tool_audit_*` JSON files (`*.partial.json` hold the full data)
- `E6_tool_audit_07_summary.json`
- `E6_tool_audit_09_lenses_summary.json`
- `E6_tool_audit_14_quiet_summary.json`
- `E6_tool_audit_07_analyze_stdout.txt`

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
