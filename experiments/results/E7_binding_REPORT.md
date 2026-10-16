# E7_binding: is gender in J-space bound to entities, or a mood spread over the whole context?

Model: Gemma-4-31B-it, W4A16 QAT, accessed through the lab daemon. Lenses: J++ (`jpp`), Jacobian lens (`jlens`), naive logit lens (`naive`). Both lens files were fitted on the base bf16 model and are applied here, unvalidated, to the instruct 4-bit model (see Established Fact 2).
Experiment id: `E7_binding`. All scripts are in `experiments/scripts/E7_binding_*.py`. All results are in `experiments/results/E7_binding_*`.

## TL;DR

1. **My main prediction was wrong.** I predicted that mid-layer J-space gender would be a context-wide mood (|BI| < 0.3). It is the opposite. In **L20-39**, J++ / JLens / probe readouts at the assistant's own **"I/me/my"** tokens carry the **assistant's** gender and almost none of the user's. At the **"you/your"** tokens they carry the **user's** gender. Values: J++ pronoun log-odds at self tokens, L30-39: G_self coefficient a = +8.85 [8.42, 9.29], G_user coefficient b = +0.39 [0.23, 0.55], binding index **BI = +0.92 [0.88, 0.95]** (L20-29: +1.00). At user-reference tokens, L30-39: a = -0.20, b = +5.51, **BI = +1.00 [0.92, 1.00]**.
2. **Binding exists only in a window.** It appears at about L16-21 for self tokens and about L24-30 for "you" tokens. It **dissolves in the late layers**: self-token BI is 0.45 at L40-49, 0.23 at L50-58 and **0.00 at L59**, where a = b = +0.42. The final-layer readout at a pronoun is an unbound mixture of both genders. In L10-19 the signal is weak, unbound, and dominated by the assistant persona.
3. **The real model also binds, but leaks.** Next-token log-odds (woman, girl, lady, female vs man, boy, guy, male):
   - after "... I am a": a = +9.43 vs b = +1.43, BI = 0.74 [0.69, 0.78];
   - after "... you are a": a = +1.23 vs b = +7.20, BI = 0.71 [0.65, 0.77].

   The lens at the prediction position shows the leak **entering after L40**: BI is 1.00 at L30-39, then 0.79, 0.73, and 0.67 at L59, which is the real output.
4. **Raw-residual geometry, which uses no lens, agrees, and shows how the binding is coded.** In L20-39, the assistant-gender direction and the user-gender direction measured at the *same* "I" token are orthogonal (noise-corrected cosine -0.28 / -0.17). Yet the user's gender is still present at that token at about 1/3 of the magnitude. A shared **"gender of this token's referent"** direction exists:
   - cos(assistant-gender at "I", user-gender at "you") = +0.65 / +0.66;
   - cos(assistant-gender at "I", assistant-gender at the user-turn "you") = +0.67 / +0.70.

   A third-person referent probe trained on unrelated "my sister/brother Robin" prose aligns with this referent direction and not with the non-referent one. The lens pullback directions do the same. Early (L10-19) and late (L40-58) layers fall back to a single shared gender axis (cosine +0.79, +0.40, +0.69).
5. **Causally, the bound pronoun representations do little, and mostly not what the lens suggests.** These results come from activation patching, with no lens involved.
   - **Whole-stream swaps.** Swapping the complete residual stream (all layers) of the passage "I/me/my" tokens moves "I am a" only **1.7% [1.1, 2.5]** of the way to the swapped persona. It moves "you are a" by 8.1% [5.7, 10.7].
   - **Direction-only swaps.** Swapping only the lens-visible *referent* (self-gender) coordinate at the "I" tokens reproduces the whole "you are a" effect (8.3%) and has no effect on "I am a" (-0.5%). The effect is **contrastive**: making "I" more male makes the predicted user more female.
   - **Neutralisation test.** Neutralising that coordinate **increases** leakage of the assistant's gender into "you are a" by +0.12 nats [0.05, 0.19] (+24%; random-direction control -0.01). Symmetrically, neutralising the user coordinate at "you" tokens increases leakage of the user's gender into "I am a" by +0.17 [0.11, 0.24] (+11%).
   - So the bound representations mainly act like **explaining away**: they suppress the other entity's gender, and only weakly support their own referent (-2% and -7% when removed).
   - **Where the information flows.** It goes system prompt → (transfer between L27 and L30) → suffix tokens and the redundantly distributed context. Chat-template tokens carry ≤ 2%. Self-gender for "I am a" is still partly read straight from the system prompt at L50: 0.13 averaged over prompt orders, 0.24 with self described first.
   - **Implication.** A strong, entity-bound J-space readout at "I" tokens (+17.7 log-odds W-M) does **not** show that the model uses that representation to express its own gender.
6. **Relevance to the "Gemma-4 is female-coded" claim.**
   - Readouts at self tokens are entity-specific *only* in L20-39. Late-layer and final-layer readouts at "I" mix in the other person's gender and the lens background.
   - On the woman-man persona scale, the unnamed default assistant (no system prompt) sits at **0.36 [0.32, 0.39]** at self tokens (J++, L30-39). It is on the *man* side of the midpoint, not the woman side. The probe gives 0.59 [0.51, 0.68]. The answer depends on the readout; neither supports strong female coding.
   - The absolute J++ pronoun value for the default changes sign across bands: -3.5 at L20-29, +0.7 at L30-39, +3.3 at L40-49. Absolute J-space "she > he" values are therefore not interpretable without persona baselines.

## Design

- **3 x 3 factorial plus a default condition.** The system prompt sets G_self in {woman, man, unspecified} x G_user in {woman, man, unspecified}. Template: `You are {A}, a {woman|man|person}. You are chatting with {B}, a {woman|man|person}. Reply in character as {A}.`
  - Four name/order variants: Sarah/Emily/Laura/Hannah, James/David/Michael/Daniel, and Alex/Sam/Jordan/Casey for self; Rachel/Megan/Julia/Anna, Thomas/Robert/Mark/Peter, and Taylor/Riley/Morgan/Jamie for the user.
  - Variants k = 0, 1 describe self first; k = 2, 3 describe the user first.
  - The default condition has no system prompt.
  - All names and gender nouns are single tokens, so paired contexts are token-aligned.
- **24 teacher-forced passages.** Each has a user message containing the user's "I/my" and "you", and an assistant reply containing "I/me/my", "you/your" and neutral content words. No gendered words and no third-person pronouns appear in the passages.
  - Each context is run with two suffixes: " By the way, I am a" and " By the way, you are a".
  - Total: 24 x 37 contexts x 2 = 1776 sequences.
- **Position types:**
  - A_self: assistant I/me/my, about 4.5 per passage;
  - A_user: assistant you/your, about 3.0;
  - A_neut: content words, about 18;
  - U_self: user-turn I/me/my, about 2.1, referring to the **user**;
  - U_addr: user-turn you/your, about 1.0, referring to the **assistant**;
  - F_I / F_Y: the final "a" token of each suffix.
- **Readouts at every layer:**
  - J++, JLens and naive lens, each scored three ways: `pron` (mean over 6 F-M pronoun pairs), `noun` (11 pairs) and `noun4` (LSE woman/girl/lady/female minus man/boy/guy/male);
  - two diff-of-means probes trained on separate third-person data (`E7_binding_01`), 64 sequences of the form "my {sister|brother}, Robin. Robin ...":
    - `probe_ref`: read at later mentions of the neutral name;
    - `probe_ctx`: read at all tokens after the kinship noun.

    Held-out (odd vs even templates) accuracy of `probe_ref` is 0.87 / 0.82 / 0.70 / 0.91 / 0.80 by band. The probe projection is scaled so that ±0.5 equals the training class means.
- **Statistics.** Tokens are averaged within each sequence, and sequences within a passage x cell (over the 4 variants). Per passage, I compute a = (mean of the G_self = W cells minus the G_self = M cells) / 2, which is the OLS slope for the ±1/0 coding, and b likewise for G_user.
  - BI_ref = (coefficient of the referent minus coefficient of the other entity) / (|a| + |b|). The referent is self for A_self, U_addr and F_I, and the user for A_user, U_self and F_Y.
  - BI is computed from mean a and b, with a bootstrap over the 24 passages (2000 resamples).
  - BI is +1 when the other entity's coefficient is zero or of opposite sign.
- **Positive control for the lens.** At third-person name tokens, J++ pron F-M is +6.0 / +13.8 / +13.2 / +20.4 / +11.8 for L10-19 through L50-58. For comparison, 2a at self tokens is 2.3 / 13.5 / 17.7 / 6.1 / 1.7.

## Pre-registered predictions and their outcomes

These are in the header of `E7_binding_common.py`; the patch predictions are in the headers of 05 and 05c.

| # | Prediction | Outcome |
|---|---|---|
| P1 | Behaviour binds, BI > 0.8 | **Partly.** BI = 0.74 (I am a) and 0.71 (you are a). Binds, but with about 15% leakage. Self-first prompts: 0.91 / 0.65. User-first prompts: 0.59 / 0.79. |
| P2 | Mid-layer lens at pronouns is unbound, \|BI\| < 0.3 | **Falsified.** BI is about 1.0 at L20-39 for both I and you tokens. |
| P3 | Neutral tokens broadcast, a ≈ b | **Partly.** a ≈ b at L40+; at L20-39 a > b (1.70 vs 0.67). The self persona is mentioned twice in the system prompt. |
| P4 | Same-token self and user directions partly aligned, corrected cosine 0.2-0.7 | **Falsified in L20-39.** Cosine is about -0.2 (orthogonal). Confirmed at L40-58 (+0.40 / +0.69). |
| P5 | Third-person probe transfers weakly, BI ≈ 0 | **Falsified.** The probe shows BI ≈ 1.0 at A_self, A_user, U_self, F_I and F_Y in L20-39. |
| C1 | sys patch at L10 flips about fully, then decays with layer | **Confirmed.** 1.00 at L10 and L20; 0.34 / 0.07 at L30; 0.13 / -0.01 at L50. |
| C2 | Pronoun patches are small and binding-consistent | **Small:** \|flip\| ≤ 0.09. A_user binding-consistent (Y 0.088 vs I 0.023); A_self is not (I 0.012 < Y 0.029). |
| C3 | Passage patch moderate (0.1-0.5) at L30-50 | **Mostly falsified.** Only at L30 (Y 0.16, I 0.05); about 0 at L40-50. |
| 05b | Suffix dominates at L40-50; user-turn patch < 0.15 | **Confirmed.** Suffix 0.67 / 0.97 at L40; user turn ≤ 0.11. |
| 05d | Template tokens carry > 0.2 at L27-35 | **Falsified.** ≤ 0.02. |
| 05c | Band patch: passage I < 0.3, Y 0.2-0.5; A_self I < 0.1; A_user Y 0.1-0.3 > I | **Confirmed.** Passage 0.075 / 0.28; A_self 0.017 / 0.081 (affects Y *more*); A_user 0.034 / 0.119. |
| 05e | A_self "non" (user) component drives the Y effect | **Falsified.** The *referent* (self) coordinate drives it (ref 0.083 vs non 0.017), contrastively. |
| 05f | Neutralising the referent coordinate raises cross-entity leak by ≥ 0.2 nats; own prediction unchanged | **Direction confirmed, smaller.** +0.12 (A_self → Y) and +0.17 (A_user → I). Own prediction drops slightly: -0.20 / -0.46 nats. rand ≈ 0. |

## Results

### 1. Binding indices by layer band (J++ pronoun log-odds)

Each cell is a / b / **BI_ref** [95% CI].

| type (referent) | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 | L59 |
|---|---|---|---|---|---|---|
| A_self (self) | +1.13 / +0.35 / **+0.53** | +6.77 / -0.04 / **+1.00** [0.98, 1.00] | +8.85 / +0.39 / **+0.92** [0.88, 0.95] | +3.03 / +1.16 / **+0.45** [0.40, 0.50] | +0.85 / +0.53 / **+0.23** [0.14, 0.32] | +0.42 / +0.42 / **-0.00** [-0.14, 0.12] |
| A_user (user) | +0.31 / +0.25 / **-0.10** | +0.44 / +1.24 / **+0.47** [0.25, 0.71] | -0.20 / +5.51 / **+1.00** [0.92, 1.00] | +0.70 / +2.11 / **+0.50** | +0.46 / +0.81 / **+0.28** | +0.31 / +0.70 / **+0.39** |
| U_self (user) | +0.84 / +0.18 / **-0.65** | +0.61 / +2.54 / **+0.61** | +3.28 / +13.32 / **+0.60** [0.45, 0.74] | +2.12 / +6.47 / **+0.51** | +0.90 / +2.12 / **+0.41** | +0.11 / +0.51 / **+0.66** |
| U_addr (self) | +0.21 / +0.29 / **-0.17** | +4.95 / +3.31 / **+0.20** | +12.31 / +4.30 / **+0.48** [0.43, 0.54] | +5.24 / +5.65 / **-0.04** | +2.12 / +2.47 / **-0.08** | +0.44 / +0.43 / **+0.01** |
| F_I (self) | +0.86 / +0.27 / **+0.52** | +4.12 / +0.13 / **+0.94** | +8.20 / -0.21 / **+1.00** | +7.83 / +0.90 / **+0.79** | +4.93 / +0.77 / **+0.73** | +4.56 / +0.89 / **+0.67** |
| F_Y (user) | +0.57 / +0.38 / **-0.20** | +0.14 / +1.63 / **+0.84** | -0.72 / +6.54 / **+1.00** | +0.06 / +5.51 / **+0.98** | +0.32 / +3.50 / **+0.83** | +0.47 / +3.40 / **+0.76** |
| A_neut (none) | +0.09 / +0.04 | +0.52 / +0.11 | +1.70 / +0.67 | +1.14 / +1.20 | +0.56 / +0.68 | +0.44 / +0.68 |

The other readouts give the same qualitative pattern; full tables are in `E7_binding_analysis.json` under `band_table`.

- **JLens** at A_self: BI = 0.49 / 0.99 / 0.92 / 0.41 / 0.33 / 0.00. At A_user: -0.15 / 0.55 / 1.00 / 0.50 / 0.35 / 0.39.
- **Naive lens** at A_self: BI = 0.33 / 0.86 / 0.69 / 0.38 / 0.24 / 0.00. At A_user, binding appears only at L30-39 (0.56). The naive lens sees binding later and less cleanly.
- **probe_ref** at A_self: BI = 0.45 / 1.00 / 1.00 / 0.60 / -0.18. At A_user: -0.14 / 1.00 / 1.00 / 1.00 / 1.00.
  - The probe keeps user binding at "you" tokens through L58, whereas the lenses lose it. This suggests the late-layer *lens* mixing partly reflects lens readout directions picking up a shared late gender axis.
  - The raw-residual BI along the cross-validated optimal direction also decays late (A_user: 0.35 at L40-49, 0.08 at L50-58), so the residual itself becomes less bound.

Layers whose BI lower CI exceeds 0.5 (J++ pron):
- A_self: L16-39 and L41;
- A_user: L29-41;
- F_I: L16-59;
- F_Y: L22-59.

Representative per-layer a/b at A_self: L21 4.6/0.0, L30 9.1/0.0, L39 7.1/1.0, L45 2.9/1.3, L54 1.3/0.7, L59 0.4/0.4.

**Cell means.** At A_self, L30-39 (rows G_self W/M/U, columns G_user W/M/U): +11.8 +12.9 +12.1 / -4.3 -6.4 -5.6 / +0.5 -0.8 +0.9. Rows differ and columns barely matter. At L50-58: +3.2 +2.0 +2.5 / +1.4 +0.2 +1.0 / +1.7 +1.0 +1.6. The values are compressed, both factors matter, and the background is positive.

**User-turn tokens.** These are the pilot's "spread" observation. At the user's own "I" (U_self), the assistant persona's gender does leak in at L30-39: a = +3.3 against b = +13.3, a leak ratio of about 0.25. The user-turn "you" (U_addr), which refers to the assistant, is bound to the assistant at L30-39 (a = 12.3 vs b = 4.3) and is unbound by L40.

**Order robustness** (`E7_binding_07`): A_self L30-39 BI = 1.00 with self described first and 0.78 [0.76, 0.81] with the user described first. A_user L30-39 = 1.00 in both orders. Behavioural leakage depends on order: "I am a" BI = 0.91 vs 0.59; "you are a" BI = 0.65 vs 0.79. In both cases the leak comes from the sentence "You are ..., a {gender}" that comes first.

### 2. Behaviour (real model, no lens)

| probe | a (G_self) | b (G_user) | BI_ref | P(F4) + P(M4): WW / MM / UU / default |
|---|---|---|---|---|
| "... I am a" | +9.43 [8.92, 9.96] | +1.43 [1.17, 1.68] | +0.74 [0.69, 0.78] | 0.236 / 0.410 / 0.000 / 0.000 |
| "... you are a" | +1.23 [0.97, 1.50] | +7.20 [6.71, 7.68] | +0.71 [0.65, 0.77] | 0.075 / 0.268 / 0.000 / 0.000 |

Typical top-1 continuations are " bit" or " little" for "I am a", and " great" for "you are a". " woman" or " man" appear in the top 5 when a gender is specified. In the default condition the model says "large [language model]" or "professional". The gender log-odds are meaningful as preferences, but the probability mass is small (0.0-0.4).

### 3. Raw-residual geometry (no lens)

Script: `E7_binding_04`. For each layer and position type I compute per-passage diff-of-means directions ds (effect of G_self) and du (effect of G_user). Cosines are noise-corrected with split-half reliability over passages (200 splits); reliabilities are 0.77-0.99. Coordinates are standardised by the per-dimension pooled within-cell SD; raw coordinates agree within about 0.1.

| pair (corrected cosine) | L10-19 | L20-29 | L30-39 | L40-49 | L50-58 |
|---|---|---|---|---|---|
| ds@A_self ~ du@A_self (same token, other entity) | +0.79 | **-0.28** | **-0.17** | +0.40 | +0.69 |
| ds@A_self ~ du@A_user (referent gender, I vs you) | +0.80 | **+0.65** | **+0.66** | +0.58 | +0.47 |
| ds@A_self ~ ds@U_addr (assistant gender at assistant "I" vs user-turn "you") | +0.61 | +0.67 | +0.70 | +0.62 | +0.44 |
| ds@A_self ~ ds@A_user (same entity at I vs you tokens) | +0.74 | **-0.05** | **+0.04** | +0.64 | +0.62 |
| ds@A_neut ~ du@A_neut | +0.94 | +0.56 | +0.49 | +0.63 | +0.70 |
| ‖du‖/‖ds‖ at A_self (unbiased) | 0.42 | 0.33 | 0.40 | 0.34 | 0.33 |
| raw-space BI along CV-optimal direction, A_self | +0.50 | +0.99 | +0.99 | +0.77 | +0.62 |
| raw-space BI along CV-optimal direction, A_user | -0.07 | +0.53 | +1.00 | +0.35 | +0.08 |
| raw-space BI along CV-optimal direction, F_I | +0.48 | +0.91 | +1.00 | +0.95 | +0.78 |
| cos(ds@A_self, J++ pron pullback) | +0.10 | +0.26 | +0.36 | +0.09 | +0.03 |
| cos(du@A_self, J++ pron pullback) | +0.08 | -0.00 | +0.03 | +0.11 | +0.07 |
| cos(du@A_user, J++ pron pullback) | +0.07 | +0.26 | +0.42 | +0.19 | +0.12 |
| cos(ds@A_user, J++ pron pullback) | +0.07 | +0.07 | -0.02 | +0.05 | +0.04 |
| cos(ds@A_self, probe_ref), raw coordinates | +0.33 | +0.40 | +0.31 | +0.08 | +0.02 |
| cos(du@A_self, probe_ref), raw coordinates | +0.22 | -0.16 | -0.05 | +0.06 | +0.08 |

**Interpretation.** In L20-39 the residual uses **referent binding**, not entity slots:
- One shared direction encodes "the gender of the entity this token refers to". It is shared across first person, second person, user turn and assistant turn, and it overlaps with a third-person referent probe.
- The other entity's gender is still decodable at the same token, at about 1/3 of the norm, but it lies in a direction orthogonal to it.
- The lens readout directions (J++ / JLens pullbacks) select the referent direction, which is why mid-layer lens readouts look perfectly bound.
- In L10-19 and L40-58 the two genders share one axis. Lens readouts there act as a "mood" detector.

### 4. Causal information flow (activation patching, no lens)

Scripts: `E7_binding_05`, `05b`, `05c`, `05d`, `05e`, `05f`. Pairs are WM <-> MW, token-aligned. The flip fraction is (patched - clean target) / (clean source - clean target), averaged over passages, with a bootstrap CI over passages.

**Single-layer patches** (24 passages x 2 orders x 2 directions):

| set patched at one layer | L10 | L20 | L30 | L40 | L50 |
|---|---|---|---|---|---|
| sys (system prompt), I / Y | 1.00 / 0.99 | 1.00 / 1.00 | **0.34** / **0.07** | 0.26 / -0.01 | 0.13 / -0.01 |
| rest (everything except sys) | -0.00 / 0.01 | -0.00 / -0.00 | 0.66 / 0.93 | 0.74 / 1.01 | 0.87 / 1.01 |
| suffix ("By the way, I am a" incl. final) | 0.00 / 0.00 | 0.00 / 0.00 | 0.20 / 0.18 | 0.67 / 0.97 | 0.86 / 1.02 |
| passage (all assistant passage tokens) | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / **0.16** [0.10, 0.23] | 0.02 / -0.01 | 0.00 / -0.01 |
| user turn | 0.00 / 0.00 | -0.00 / -0.00 | 0.04 / 0.11 | 0.01 / 0.03 | 0.00 / 0.00 |
| A_self ("I/me/my" in passage) | 0.00 / 0.00 | 0.00 / 0.00 | **0.012** / 0.029 | 0.00 / 0.00 | 0.00 / 0.00 |
| A_user ("you/your" in passage) | 0.00 / 0.00 | 0.00 / 0.00 | 0.023 / **0.088** | 0.00 / 0.01 | 0.00 / 0.00 |

**Finer flow map** (`05d`, single-layer, self-first prompts only, 24 passages). Sys flip for I / Y:
- L20: 1.00 / 1.00
- L24: 0.97 / 0.97
- L27: 0.94 / 0.92
- **L30: 0.47 / 0.12**
- L35: 0.44 / 0.05
- L40: 0.42 / -0.01
- L50: 0.24 / 0.00

Suffix flip for I / Y: 0.01 / 0.02 at L24, 0.15 / 0.17 at L30, 0.32 / 0.57 at L35, 0.52 / 0.97 at L40, 0.76 / 1.01 at L50.

The template tokens are `<turn|>\n<|turn>user\n` after the system prompt, and `<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`. They carry ≤ 0.02 at every layer, which rejects my post-hoc guess that turn-boundary tokens act as identity caches.

At L30, "rest" (0.66) is much larger than suffix + passage + user + template (about 0.3). The non-system copy of the information is stored **redundantly**, so subsets show sub-additive effects.

**Band patches** (`05c`): resid_post is replaced at *every* layer of a band at the chosen positions. With band "all", that position set's whole residual stream comes from the source. Values are flip I / Y [95% CI]:

| set | all layers (L0-59) | L20-39 only |
|---|---|---|
| sys | 1.00 / 1.00 | 1.00 / 1.00 |
| user turn | 0.058 [0.044, 0.073] / 0.153 [0.103, 0.208] | 0.056 / 0.163 |
| passage | 0.075 [0.055, 0.098] / **0.280** [0.193, 0.383] | 0.067 / 0.289 |
| A_self ("I/me/my") | **0.017** [0.011, 0.025] / **0.081** [0.057, 0.107] | 0.016 / 0.082 |
| A_user ("you/your") | 0.034 [0.024, 0.045] / 0.119 [0.056, 0.208] | 0.033 / 0.122 |

The L20-39 patch equals the all-layer patch, so these positions matter only through their mid-layer states. Swapping the "I"-token streams moves the **user** prediction about 5x more than the self prediction.

**Directional interchange at pronoun tokens** (`05e`, all layers). Directions are cross-fitted from the other half of the passages: *ref* is the referent-gender diff-of-means direction; *non* is the other entity's direction, orthogonalised; *rand* is a random direction.

| set / subspace | flip "I am a" | flip "you are a" |
|---|---|---|
| A_self ref | -0.005 [-0.016, 0.007] | **+0.083** [0.059, 0.109] |
| A_self non | +0.002 | +0.017 [0.011, 0.025] |
| A_self ref + non | -0.003 | +0.094 |
| A_self rand | -0.000 | +0.007 |
| A_self full stream | +0.017 | +0.081 |
| A_user ref | +0.032 [0.022, 0.043] | +0.071 [0.025, 0.132] |
| A_user non | +0.022 | +0.044 |
| A_user rand | +0.002 | +0.000 |
| A_user full stream | +0.034 | +0.119 |

The raw values make the sign of the A_self effect clear. In the WM target (self woman, user man), swapping only the "I"-token self coordinate to "man" shifts "you are a" by **+0.82 nats toward *woman***, while "I am a" changes by +0.01. In the MW target the shift is -1.09. This is contrastive: a more male "I" yields a more female "you".

**Explaining-away test** (`05f`). The referent coordinate is *neutralised* by interchanging it with a context where that entity is "a person". Paired bootstrap over 24 passages, in nats:

| measure | clean | ref neutralised (Δ [CI]) | rand control Δ | full-stream Δ |
|---|---|---|---|---|
| self gender → "you are a" leak, (Y[WU]-Y[MU])/2 | +0.50 | **+0.12** [0.05, 0.19] | -0.01 | +0.23 |
| self gender in "I am a", (I[WU]-I[MU])/2 | +10.33 | -0.20 [-0.25, -0.16] | -0.04 | -0.12 |
| user gender → "I am a" leak, (I[UW]-I[UM])/2 | +1.48 | **+0.17** [0.11, 0.24] | -0.01 | +0.19 |
| user gender in "you are a", (Y[UW]-Y[UM])/2 | +6.97 | -0.46 [-0.68, -0.28] | -0.03 | -0.42 |

Removing the bound referent-gender signal at a pronoun **increases** leakage of that entity's gender into predictions about the *other* entity. It only slightly weakens predictions about its own referent. The bound mid-layer representation acts mainly as a suppressive, explaining-away-like signal that sharpens the model's bindings. A heteronormative "opposite-gender interlocutor" prior would predict the same sign; this test cannot tell the two apart.

**Reading.**
- Until about L27, everything the prediction uses is in the system-prompt tokens.
- Between L27 and L30 most of the *user's* gender is copied out into the conversation and suffix, where it is stored redundantly. By L40 the user's gender sits entirely in the suffix tokens.
- The *assistant's* gender for "I am a" keeps being read directly from the persona definition as late as L50.
- The lens-visible, entity-bound representations at passage pronouns carry little causal weight for either behaviour, and their main measurable role is suppressive (explaining-away-like).

## Instrument evaluation (Theodolite)

- **Real strength.** J++ and JLens detect the mid-layer binding earlier and more cleanly than the naive logit lens. For A_self, the BI lower CI exceeds 0.5 from L16 (J++) vs L20 (naive); for A_user, from L29 vs only scattered layers. Lens readouts agree with lens-free measures (probe, raw-residual BI, behaviour) wherever they can be compared. Lens transfer from base to instruct-QAT therefore looks reasonable *for this contrast-based use*.
- **Pitfall 1: layer choice decides the answer.** The same J++ readout at "I" tokens is perfectly entity-bound at L20-39 and unbound at L59 (BI = 0.00). A UI that shows late-layer "ghost tokens" at self tokens displays a context-wide gender mixture, not "the model's self-concept".
- **Pitfall 2: absolute values are uninterpretable.** The background lens bias is large and varies by band:
  - the default assistant's self-token J++ pron is -3.5 / +0.7 / +3.3 for L20-29 / L30-39 / L40-49;
  - neutral content tokens read +2 to +5 "she" at L30-58 in every cell.

  Only within-position contrasts against persona baselines mean anything.
- **Pitfall 3: readable does not mean used, and used may not mean what it looks like.** The strongest, cleanest lens signal (+17.7 log-odds W-M at "I" tokens, L30-39) sits in positions whose residuals barely affect "I am a" (1.7% for a whole-stream swap). The same coordinate *is* used, but contrastively, to suppress self-gender leakage into "you are a". J-space readouts show what is linearly *present*, not what is *used* or how.
- Combined with the delta_ffn bug (REPORT.md §3.4, bug #2) and the low single-instance VJP agreement, the Theodolite should be used for averaged, baseline-referenced contrasts, never for single-prompt absolute "J-space gender" claims.

## Caveats

- **Prompt asymmetry.** The self persona is named twice ("Reply in character as A"), and the persona descriptions use "a person" for the unspecified condition. This likely inflates a relative to b at neutral tokens and in early layers. It cannot explain the near-zero b at self tokens in mid layers.
- **Low probability mass.** Behavioural gender mass is low (0-0.41). The log-odds measure preference, not the likelihood that the model volunteers a gender.
- **Diff-of-means geometry** uses 24 passages. Noise correction assumes roughly isotropic noise. Raw and diag-standardised coordinates agree, but cosines involving the lens pullbacks are much smaller in raw coordinates (0.02-0.2), because of outlier dimensions.
- **Single-layer patches** understate distributed contributions; the band patches address this. The directional and neutralisation tests use 1-D diff-of-means directions, which capture the mean shift but not necessarily all gender information. The 05b/05d/05e/05f designs were added post hoc after earlier patch results, each with predictions written before it ran. Patching tests only two behaviours ("I am a", "you are a"). Other behaviours, such as choosing pronouns in free text, might use the pronoun-token representations.
- **Scope.** One model, one prompt template family, short passages, English, thinking disabled. "Unspecified" uses neutral names, which carry some gender lean, but this is averaged over four names.
- **Unvalidated lens transfer** (Fact 2) still applies. Every lens conclusion here is backed by a lens-free measure.

## Files

- Scripts: `experiments/scripts/E7_binding_common.py` (data, predictions, helpers), `E7_binding_00_inspect.py`, `E7_binding_01_probe.py`, `E7_binding_02_factorial.py` (+ `_c0..c2`), `E7_binding_03_analyze.py`, `E7_binding_04_geometry.py`, `E7_binding_05_patch.py` (+ `_c0, c1`), `E7_binding_05b_patch.py` (+ `_c0, c1`), `E7_binding_05c_bandpatch.py` (+ `_c0, c1`), `E7_binding_05d_flowmap.py`, `E7_binding_05e_dirpatch.py` (+ `_c0, c1`), `E7_binding_05f_explainaway.py` (+ `_c0, c1`), `E7_binding_06_tables.py`, `E7_binding_07_order_check.py`, `E7_binding_08_dirs.py`, `E7_binding_09_explainaway_stats.py`.
- Results:
  - `E7_binding_01_probe.json` (probe validation + lens positive control);
  - `E7_binding_factorial_chunk{0,1,2}.npz` (per-context x type x layer x 11 readouts + behaviour);
  - `E7_binding_analysis.json` (band tables, per-layer BI with CIs, cell means, woman-man scale, behaviour);
  - `E7_binding_geometry.json` and `E7_binding_geometry_summary.json`;
  - `E7_binding_patch_chunk{0,1}.json`, `E7_binding_patchb_chunk{0,1}.json`, `E7_binding_patchd.json`, `E7_binding_bandpatch_chunk{0,1}.json`, `E7_binding_dirpatch_chunk{0,1}.json`, `E7_binding_explainaway_chunk{0,1}.json`, `E7_binding_explainaway_stats.json`;
  - `E7_binding_order_check.json`, `E7_binding_tables_summary.json`, `E7_binding_tables.md` (all generated tables).
- Cache (large): `experiments/results/E7_binding_cache/` (probe means, lens pullbacks, variant-averaged residual means of about 1 GB, fp16).

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
