# V5_binding_dissociation: does the C6 "binding" follow the REFERENT, or the token type and turn?

Target: claim C6 (E7). E7 found that in L20-39, J-space gender at the assistant's "I/me/my" tracks the assistant's gender and at "you/your" tracks the user's. In E7, every assistant-turn "I" referred to the assistant, so referent, token type and turn speaker were confounded.

Here, quoted and reported speech pulls them apart. All work went through the lab daemon. n = 24 items in every design; 4 name/order variants per item. Bootstrap 4000 resamples over items.

## Verdict (short)

- **Descriptive C6 replicates.** The model's own-turn pronouns are "bound" at L20-39 (J++ pron, L30-39):
  - assistant "I": share of the assistant's gender 0.88 [0.86, 0.89];
  - user-turn "I": 0.99;
  - two-party E7-style BI 0.87 [0.84, 0.90].
- **C6 read as *referent* binding is refuted/weakened.** At a quoted "I" whose referent is a chat participant, the J-space gender does **not** follow the referent:
  - **With a third party T present, it follows T.** User quoting the assistant ("you told me, 'I will ...'"): the referent's (assistant's) coefficient is -0.05, T's is +3.95, DI -1.00 [-1.00, -0.52].
  - **With no third party, it follows the turn speaker.** Assistant quoting the user: DI -0.41 [-0.52, -0.29]. User quoting the assistant: DI -0.31 [-0.53, -0.09]. This is the "token-type/turn artifact" outcome.
  - **Referent-following appears only when the referent IS the third party.** T's quoted "I": DI +0.97. "You" addressed to T: DI +0.92. These cases cannot separate referent binding from a "quotes belong to the third party" default.
  - **The real model does resolve the quotes at prediction positions.** Behavioural DI is +0.20 to +0.71. So the token-level readout reflects a role default, not the referent representation the model uses.
- **Late "unbinding" is readout-specific.**
  - It replicates for the noun axis and when quiet dims are masked. Masked, the self signal at L50-58 is exactly gone (a ≈ -0.2).
  - Along E7's own raw-space referent direction, which is lens-free and out-of-sample, assistant-"I" binding stays at BI 0.62-0.79 through L58, with undiminished magnitude.
  - In a three-person context the late lens "gender" at every token is dominated by the third party, who is the only person "she/he" can refer to.

## Design

- **System prompt.** It sets the genders of A (assistant), U (user) and T (third party) in a 2x2x2 design (woman/man):
  - all names are gender-neutral single tokens: A ∈ {Alex, Sam, Jordan, Casey}, U ∈ {Taylor, Riley, Morgan, Jamie}, T ∈ {Avery, Rowan, Dana, Cameron};
  - the template is `You are {A}, a {g}. You are chatting with {U}, a {g}. The conversation will mention {T}, a {g}. Reply in character as {A}.`;
  - entity order is rotated over 4 variants;
  - the 8 cells are token-identical except for the 3 descriptor tokens.
- **Items.** 24 teacher-forced conversations. Each contains:
  - **user turn:** the user's own I/you; T quoted by the user (qT: "I" → T); the assistant quoted by the user (qA: "I" → A, "you" → U);
  - **assistant turn:** own I/you; the user quoted (qU: "I" → U); T quoted (qT: "I" → T); a suggested line for the user to say to T (qS: "You" → T, "I" → U);
  - **controls:**
    - a proximity control, "As for {T}, I would ..." (A_nearT_I: the assistant's own "I" two tokens after T's name);
    - a token-id control, A_init_I. The turn-initial 'I' has the same no-space token id as quoted 'I's.
  - **T name tokens and a behavioural F_T probe:** " By the way, {T} is a".
- **Readouts** (22 per position and layer):
  - J++, JLens and naive lenses × {pron, noun, noun4}, each on the full h and on h with the **quiet dims clamped**. Quiet dims are the 10% lowest token-std per layer, from these contexts. They hold 99-100% of J++ pullback energy at L36+, replicating E6.
  - E7's `probe_ref` and `probe_ctx`, trained on separate third-person prose.
  - **e7dir**: projections on E7's cross-fitted raw-space A_self and A_user referent directions. These come from different passages, gendered names and no T, so the test is lens-free and out-of-sample.
- **Statistics.** Coefficients a, b, c are the ±1-coded effects of G_A, G_U and G_T (each (mean_W - mean_M)/2), averaged per item over the 4 variants.
  - DI = (R - D) / (|R| + |D|), where R is the referent's coefficient and D the coefficient of the entity the token class would refer to outside quotes in that turn ("I" → turn speaker; "you" → turn addressee).
  - Like E7's BI, DI saturates at ±1 when the other coefficient has the opposite sign.
- **Follow-up designs:**
  - `_04`: prediction positions inside quotes, with real next-token woman-set vs man-set log-odds plus lenses;
  - `_06`: all T quotes removed;
  - `_08`: two-party, with E7's 24 passages and a quoted user "I" inserted in the assistant turn and a quoted assistant "I" in the user turn; no third party anywhere.
- **Pre-registered predictions** are in the headers of `V5_binding_dissociation_common.py`, `_04`, `_06` and `_08`. The `_06` and `_08` predictions were written after seeing the first `_02` results.

## Results

### 1. Three-party factorial, L30-39 (J++ pron coefficients; DI or share for other readouts)

| type (referent R / default D) | a (A) | b (U) | c (T) | J++ DI [95% CI] | noun | JLens | naive | quiet-masked | probe_ref | e7dir |
|---|---|---|---|---|---|---|---|---|---|---|
| A_qT_I: T's "I", assistant turn (T / A) | +0.05 | +0.72 | **+3.89** | **+0.97** [0.85, 1.00] | +1.00 | +0.98 | +0.76 | +0.92 | +1.00 | +0.71 |
| A_qS_you: "You" said to T (T / U) | -0.12 | +0.15 | **+3.38** | **+0.92** [0.87, 0.97] | +0.94 | +0.90 | +0.72 | +0.89 | +1.00 | +0.70 |
| A_qU_I: user's quoted "I" (U / A) | +0.76 | +0.98 | **+2.16** | +0.13 [-0.15, 0.44] | +0.14 | +0.15 | +0.27 | +0.17 | +0.12 | +0.15 |
| A_qS_I: user's "I" in the line to T (U / A) | +2.14 | +2.29 | +0.60 | +0.03 [-0.08, 0.15] | +0.07 | +0.03 | +0.12 | +0.08 | +0.22 | +0.14 |
| U_qA_I: user quoting the assistant's "I" (A / U) | **-0.05** | +1.02 | **+3.95** | **-1.00** [-1.00, -0.52] | -1.00 | -1.00 | -0.51 | -0.95 | -1.00 | -0.37 |
| U_qT_I: user quoting T (T / U) | +1.84 | +1.26 | +3.58 | +0.48 [0.24, 0.70] | +0.50 | +0.50 | +0.39 | +0.44 | +0.69 | +0.44 |
| U_qA_you: "you" in the assistant's quote, user turn (U / A) | +0.90 | **+6.23** | -0.38 | +0.75 [0.41, 1.00] | +0.83 | +0.62 | +0.58 | +0.73 | +1.00 | +0.70 |
| A_out_I, own (A); share_R | **+5.02** | +0.63 | -0.08 | share 0.88 | 0.85 | 0.87 | 0.82 | 0.85 | 0.94 | 0.93 |
| A_init_I, same token id as quoted 'I' (A) | **+9.29** | +0.51 | -0.33 | 0.92 | | | | | | |
| A_nearT_I, proximity control (A) | **+7.27** | -0.03 | +0.68 | 0.91 | 0.85 | 0.90 | 0.78 | 0.87 | 0.72 | 0.82 |
| A_out_you (U) | +0.55 | **+3.82** | +0.37 | 0.81 | | | | | | |
| U_out_I (U) | +0.15 | **+11.45** | -0.01 | 0.99 | | | | | | |
| U_out_you (A) | +1.36 | +0.30 | **+2.31** | 0.34 [0.01, 0.56] | | | | | | |
| A_T, T name tokens (T) | +0.59 | +0.68 | **+6.85** | 0.84 | | | | | | |

- **Robustness:**
  - results are the same in all 4 system-prompt orders: U_qA_I DI is between -0.68 and -0.97 whether T is described first or last;
  - L20-29 shows the same pattern.
- **Behavioural positive control.** The real next token at " By the way, {T} is a" gives c = +9.26, a = +1.71, b = +1.50.
- **P1 per type:**
  - **confirmed** for A_qT_I and A_qS_you, but these are also predicted by the "third-party default";
  - **mixed** for A_qU_I and A_qS_I;
  - **reversed** for U_qA_I.
- **P2, proximity, confirmed.** The assistant's own "I" right after T's name follows A.
- **P3 confirmed.** Probe and e7dir reproduce every sign.

### 2. Lens-free raw-residual geometry

Noise-corrected cosine with the assistant-gender direction at its own "I" (dA@A_out_I), over 300 item split-halves.

- **Shared referent direction at L20-35.** It holds for "you" → U: 0.69-0.78, which replicates E7. It also holds for T at name tokens (0.58-0.70), T at T's quoted "I" (0.55-0.81) and T at "You" said to T (0.68-0.78).
- **Misbinding at quoted participant "I"s.** At A_qU_I and U_qA_I, T's gender also lies in the referent direction: 0.82/0.73 and 0.75/0.74 at L30/L35.
  - So the referent-gender slot itself is filled with the third party's gender.
  - This is not orthogonal context leakage. By contrast, T's gender at the proximity-control "I" is orthogonal to that direction (cos ≈ 0).
  - The quoted assistant "I" in the user turn carries the assistant's gender **outside** the referent direction: cos -0.08 to +0.20 at L20-35.
- **Late layers.** Name-token T gender decouples from the referent direction at L40+ (cos ≈ 0.1).

### 3. Control without T quotes (`_06`, 24 items)

T's quotes were plausibly priming "quotes = T". Removing them leaves the quoted-participant results unchanged:
- U_qA_I: a = -0.69, b = +3.52, c = +3.64, DI -1.00;
- A_qU_I: a = +1.85, b = +1.37, c = +2.23, DI -0.15 [-0.29, 0.01].

H_prime is therefore refuted for quoted "I"s.

It does repair the user-turn "you" → A that sits next to the T quote: U_out_you goes from a = 1.36, c = 2.31 to **a = 10.14, b = 3.49, c = -0.55**, which is E7-like. So T contamination of that token was local spill-over from the adjacent quote.

### 4. Two-party control (`_08`, the task's critical item (a); no third party)

| type | a (A) | b (U) | DI [CI] (J++ pron, L30-39) | noun | JLens | naive | quiet-masked | probe | e7dir |
|---|---|---|---|---|---|---|---|---|---|
| A_qU_I: "You told me before, 'I ...'" (R = U, D = A) | **+6.68** | +2.81 | **-0.41** [-0.52, -0.29] | -0.37 | -0.36 | -0.31 | -0.40 | -0.63 | -0.58 |
| U_qA_I: "you told me, 'I ...'" (R = A, D = U) | +1.20 | **+2.27** | **-0.31** [-0.53, -0.09] | -0.31 | -0.32 | -0.24 | -0.32 | -0.51 | -0.10 |
| U_qA_you (R = U, D = A) | +1.21 | **+9.69** | +0.78 [0.58, 0.93] | +0.94 | | | | +1.00 | +0.77 |
| A_out_I (E7 BI) | +4.15 | +0.30 | +0.87 [0.84, 0.90] | | | | | | |

- **Behaviour, real next token.**
  - After "... Earlier you told me, \"I am a": a = +1.83, b = **+4.22**, so the output follows the **referent** (DI +0.40 [0.33, 0.46]).
  - Own-voice "I am a": BI +0.65.
- **Reading.** The model knows whose "I" it is. The J-space at the quoted "I" token carries the turn speaker's gender instead.
- **Caveat.** These quotes sit at the turn start/end, which strengthens speaker priming (compare A_init_I). The three-party `_06` placement, mid-turn, also gives speaker ≥ referent.

### 5. Prediction positions inside quotes (`_04`, three-party, 24 items)

| suffix (R / D) | behaviour a / b / c | behaviour DI [CI] | p(woman-set + man-set) | J++ L30-39 a / b / c |
|---|---|---|---|---|
| own " I am a" (A) | +9.84 / +1.92 / +0.80 | share 0.78 | 0.654 | +8.77 / +0.23 / -0.01 |
| ' Earlier you told me, "I am a' (U / A) | +2.66 / **+3.99** / +0.47 | **+0.20** [0.11, 0.29] | 0.020 | +1.58 / **+4.32** / +0.11 |
| ' {T} once told you, "I am a' (T / A) | +0.93 / +1.15 / **+5.38** | +0.71 [0.65, 0.76] | 0.020 | -0.17 / +0.39 / +5.74 |
| ' You could also tell {T}, "You are a' (T / U) | +2.06 / +1.93 / **+6.31** | +0.53 [0.48, 0.58] | 0.001 | +0.47 / +1.08 / +5.74 |
| user turn ' Once you told me, "I am a' (A / U) | **+0.99** / +0.20 / +0.37 | +0.66 [0.56, 0.76] | 0.003 | +0.93 / +0.41 / **+1.88** |

- At the assistant-turn prediction positions, both the mid-layer lens and the output pick the quoted referent; there is no T contamination at P_qU.
- The output leaks the turn speaker's gender: A/U = 0.67 at P_qU.
- In the user turn, the lens is T-dominated while the tiny-mass output favours A.

### 6. Late-layer re-test (P4). A_out_I BI vs U by band, L20-29 / L30-39 / L40-49 / L50-58

| readout | three-party | two-party (E7-like) |
|---|---|---|
| J++ pron | +0.82 / +0.78 / **+0.17** / **-0.13** (c = 1.7-2.1 late) | +0.73 / +0.87 / +0.23 / -0.03 |
| J++ noun | +0.82 / +0.73 / +0.26 / -0.06 | +0.72 / +0.86 / +0.37 / +0.08 |
| J++ noun4 | +0.82 / +0.73 / +0.32 / +0.05 | +0.71 / +0.85 / +0.38 / +0.16 |
| J++ quiet-masked pron | +0.82 / +0.73 / +0.10 / **-1.00 (a = -0.2)** | +0.73 / +0.82 / +0.16 / -1.00 (a = 0.0) |
| JLens quiet-masked | +0.80 / +0.70 / +0.02 / -1.00 | +0.72 / +0.80 / +0.08 / -0.94 |
| probe_ref | +0.99 / +0.92 / +0.20 / -0.37 | +0.98 / +1.00 / +0.47 / -0.13 |
| **e7dir_self (raw, lens-free)** | **+1.00 / +0.94 / +0.76 / +0.62** (a = 1.6 / 2.7 / 3.7 / 2.7; c ≈ 0) | **+0.97 / +1.00 / +0.79 / +0.66** |

- **The lens/probe decay replicates on the noun axis and with quiet dims masked.**
  - Masking removes what remains of the late self signal in J++ and JLens. The late signal visible to the lenses lives entirely in the quiet dims.
  - With noun axes, the user's own "I" in the user turn stays bound late: BI 0.94 / 0.78 at L40-49 / L50-58; E7's own noun table agrees. The decay is token-type specific.
- **The information is not lost.** Along a fixed raw referent direction, the assistant's gender at its own "I" remains dominant at L40-58.
  - The lens "unbinding" happens because a context-wide signal takes over the she/he readout. Here that signal is the third party's gender: A_neut has c = 4.34 at L40-49.
  - For "you" tokens, E7's raw A_user direction does decay: BI 0.88 → 0.15 → -0.10.
- **L59 is not evidence about binding.** The L59 "readout" at a pronoun is the real next-token distribution after the pronoun, where she/he logits are tail mass. A BI of 0.00 there is uninformative about binding.

## Revised claim (C6)

In L20-39, J-space gender at the model's own-turn pronouns follows a role default: own-turn "I" carries the turn speaker's gender and "you" the turn addressee's (share 0.81-0.99). This is not resolved reference. At quoted pronoun tokens:
- if a third party is present, the token carries the third party's gender, which occupies the same raw "referent" direction even when the quote is the user's or the assistant's words. User quoting the assistant: assistant coefficient -0.05 vs third party +3.95.
- with no third party, it carries the turn speaker's gender. Assistant quoting the user: DI -0.41.
- it follows the referent only when the referent is the third party, which a third-party default also predicts.

The model's next-token behaviour does resolve quoted referents (DI +0.20 to +0.71). Token-level "binding" is therefore a heuristic default, not the referent representation that drives prediction; this is consistent with E7's ~2% causal weight for I-token states.

"Binding fades late" holds for lens and probe readouts (noun axis and quiet-masked included) but not for the residual: a fixed raw referent direction keeps assistant-"I" BI at 0.6-0.8 through L58. Late lens gender largely tracks the conversation's third-person referent.

## Caveats

- These are constructed, teacher-forced English passages with straight double quotes. Quoted 'I' is the no-space token id; the A_init_I control shows this id carries the speaker's gender normally.
- DI and BI saturate at ±1 when the other coefficient has the opposite sign; the coefficients are reported alongside.
- Probe magnitudes are small (≤ 0.4), but their signs are consistent.
- e7dir was fitted to maximise self-gender at E7's A_self tokens. High BI along it is partly by construction, but it is out-of-sample: different passages, names and prompts, and a third entity.
- She/he readouts in a three-person context may partly index the anticipated third-person referent rather than the token's referent; the noun axes and probe agree in sign.
- In the two-party design, the quotes sit at turn edges.
- Some geometry pairs have low split-half reliability: quoted-I directions have rel 0.2-0.4, so their disattenuated cosines are noisy. The lens and probe coefficient tables are the primary evidence.
- Engineering note: the shared GPU sat at about 31.9 of 32.6 GB, and one chunk OOM'd at batch 32. Jobs were rerun at batch 12; the results are identical in design.
- No causal patching was done on the quoted-token states.

## Files

- **Scripts** in `experiments/scripts/`:
  - `V5_binding_dissociation_common.py`: items, design and pre-registration;
  - `_00_inspect`, `_01_quiet`;
  - `_02_factorial` (+ `_c0` to `_c2`);
  - `_03_analyze` (argument `factorial` or `noqt`);
  - `_04_predpos` (+ `_c0` to `_c2`);
  - `_05_geometry`;
  - `_06_noqt` (+ `_c0` to `_c2`);
  - `_07_analyze_predpos`;
  - `_08_twoparty` (+ `_c0`, `_c1`);
  - `_09_analyze_twoparty`;
  - `_10_summary`.
- **Results** in `experiments/results/`:
  - `V5_binding_dissociation_summary.json`, with key numbers for all tables;
  - `_analysis.json`, `_analysis_noqt.json`, `_twoparty_analysis.json`, `_predpos_analysis.json`, `_geometry.json`;
  - raw npz files: `_factorial_chunk{0,1,2}.npz`, `_noqt_chunk{0,1,2}.npz`, `_twoparty_chunk{0,1}.npz`, `_predpos_chunk{0,1,2}.npz`.
- **Cache:** `V5_binding_dissociation_cache/`, holding quiet-dim statistics and residual means.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
