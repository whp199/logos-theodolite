# V8_stat_audit: integrity audit of the E1-E7 headline numbers

Auditor: V8 (adversarial verifier). Method: read every script that produced a headline number, recompute it from the
saved arrays where possible (CPU, `.venv/bin/python`), and run 3 small daemon spot checks (scripts
`experiments/scripts/V8_stat_audit_0{1,2,3}_*.py`; data `experiments/results/V8_stat_audit_0*_data.json`).
Predictions and refutation criteria are written in each script header.

## 0. Bottom line

| claim / number | verdict | evidence |
|---|---|---|
| E1 default - nameless = +0.6% of span (J++ pron L30-39); held-out within 1% | **trustworthy** | Recomputed from `E1_self_readout_passage_means.npz`: +0.120 log-odds = +0.587%; held-out (16 passages): -1.1 / +0.65 / -0.66% (L20-29/30-39/40-49); Gemma-nameless +7.95% (main), +7.1% (held-out). Passage is the bootstrap unit, paired indices, ratio bootstrapped. Single BOS, correct self positions (token states, not predictions). |
| E2 she 43/48 (Gemma persona), 0/48 default | **trustworthy (counts)**, interpretation narrowed | Blind re-classification of a random 60 of the 384 samples: 0/60 disagreements. All 48 Gemma samples read: every 'she' refers to Gemma. New-paraphrase replication (V8_01): Gemma persona she 22/24 valid samples; default it 24/24, she 0, he 0. But the 'she' needs the name "Gemma" written as its antecedent. With the identical system prompt and name-free instructions, 'she' appears 0/30 times (E1 0/6; V8_03 0/24, including a colleague frame where Sarah gives she 8/8 and James he 8/8). See sections 3 and 6. |
| E2 "5/48 call themselves ChatGPT" | correct, but clustered | All 5 come from paraphrase 0, all from one batch call (k = 8-14). V8_01 replication: 4/8 "ChatGPT ... trained by OpenAI" for a new "About-blurb" prompt. |
| E7 BI = 0.92 / 1.00 (L20-39), behaviour BI 0.74 / 0.71 | **trustworthy**, one reporting caveat | Recomputed exactly from `E7_binding_analysis.json`. Coding a = (W-M)/2 is the correct OLS slope for the balanced +1/0/-1 design; unit = passage (24), tokens and name variants averaged first; no pseudo-replication. Caveat: BI is clipped at 1 whenever the other coefficient is <= 0 (e.g. A_user L30-39 a = -0.20 gives BI = 1.000 and CI upper bound 1.000). "1.00" means "no positive leak", not a measured perfect binding. Leak ratio b/a is the clearer number: 0.044 (A_self L30-39), 0.15 / 0.17 for real behaviour (claim C6 says 15-25%; it is 15-17%). |
| E3 cos(gbar, J++ pullback) 0.15 at L30-39; chat-negative -0.15 / -0.13 | **trustworthy, sharpened** | VJP storage fp16 is harmless (norm error 1e-6; cosines reproduce to 1e-6). Chat-negative result is robust: unit-normalised gradients -0.145/-0.125; group jackknife L30-39 range [-0.145, -0.112]; 21/24 chats (88%) negative individually. **But it is carried by non-self assistant tokens (-0.16/-0.15); at self tokens (n = 24) it is only -0.04/-0.04.** Percentile CIs are shrunk toward 0 (bootstrap of a cosine of a mean is biased toward 0), so the point estimate sits at the CI edge ([-0.16, -0.09] around -0.15); conservative, not anti-conservative. |
| E4 steering u_self +-0.02 -> 'She' 16/16 / 'He' 16/16 | **counts reproduce; specificity REFUTED** | V8_02 replicated 16/16 She and 16/16 He with a new RNG stream. E4 never sampled an equal-norm on-manifold control. With E4's own gender-orthogonal controls at the same norm: ctrl_topic -0.02 -> **He 15/16 (+1 She)**; ctrl_name +0.02 -> **She 15/16 (+1 He)**; ctrl_nat0 -0.02 -> He 6/16 (9/16 switch to "I am ..."); ctrl_nat0 +0.02, ctrl_nat1 +0.02, ctrl_topic +0.02, ctrl_name -0.02 -> "It/The entity" 16/16. Pre-registered refutation criterion (any control >= 8/16 gendered) met twice. The default "It" self-reference is fragile: many 2%-norm pushes at L25-39 personify it. What is specific to u_self is the bidirectional sign (+ -> she, - -> he, both 16/16); no control does that. |
| E4 pair-clamp ablation: u_self removes 41%, on-manifold controls 0.00 +- 0.02, control pullbacks 0% | efficacy OK; **the control comparison is uninformative by construction** | The clamp sets the coefficient to the mean of the paired clean F and M runs. The controls were orthogonalised against u_cont, which is the mean F-M difference at the very kin items being clamped, so their mean signed F-M displacement is exactly 0.000 (computed from `E4_causal_cache/third.pt`, L25-39: u_self 1.79, u_cont 3.57, pb_jpp 0.24, ctrl_* 0.27-0.65 per-pair absolute but 0.000 signed). A ~0 effect is guaranteed. The informative specificity evidence in E4 is the norm-matched calibrated shift (k = +-0.35: 12 vs <= 1.2 nats). |
| E5 latent " Gemma" at self tokens (rank 110-130 vs 600 elsewhere vs 1500-2700 for human personas) | numbers correct; **"latent Gemma identity" interpretation largely REFUTED** | Recomputed from `E5_name_E_identity_self.npz` (difference-in-differences across the identity tokens that job E already saved). (i) The self-vs-other-token elevation is generic: J++ L30-39 self minus background log P = Gemma +4.1, ChatGPT +3.9, Gemini +4.0, OpenAI +3.4, Google +3.3, Microsoft +4.8, GPT +6.6, " AI" +4.4. (ii) The AI-persona vs human-persona effect at self tokens is smaller for Gemma than for other AI tokens (L40-49: Gemma +6.1, Gemini +10.4, ChatGPT +12.2, OpenAI +11.6, " AI" +14.7). Difference-in-differences, Gemma minus ChatGPT = -6.1 [-7.5, -4.8]; minus " AI" = -8.6 [-9.4, -7.8]. (iii) At L30-39 OpenAI (-7.2) and ChatGPT (-7.9) beat Gemma (-8.6) at default self tokens. (iv) Gemma's lead at L40-49 is persona-independent: under the woman persona "You are Sarah" log P(" Gemma") = -13.5, versus -21.1 for " Sarah" itself. Best reading: generic AI/tech-entity component plus a token prior for " Gemma"; no Gemma-specific self-identity. |
| E6 filament rho = -0.44 (coupling "anti-correlated" with effect) | **REFUTED as stated (Simpson's paradox)** | The job's own stored statistic is Spearman(signed C, signed dgate) = -0.21 (p = 0.14). The -0.44 is |C| vs |dgate| pooled over layers. |C| falls with source layer (band L0-9: 81-152; L50-58: 0.6-3.1) while |dgate| rises (median 0 to 0.98), because the 48 filaments were stratified by layer band. Within-band pooled rank correlation = +0.16 (permutation p = 0.32). Correct statement: scores are not comparable across layers and do not predict effect within a layer. The qualitative "not a verified live circuit" verdict stands; "anti-correlated" does not. |
| Lens / labkit plumbing | OK | Chat template already emits `<bos>` and `lab.ids` encodes without special tokens, so experiment inputs have a single BOS. The double BOS exists only in the tool path, as E6 says. The tokenizer does not add BOS by itself, and no E1-E7 script calls `tok.encode` on model inputs. The lens payloads have no bias term, so ignoring one is not an error. Left-padded `generate_batch` (V8_01): greedy identical to unpadded for 2/3 prompts over 40 tokens; the third diverges at char 54 (tie-level). Next-token max |dlogit| is 0.34-0.69 with generate-style position ids, 1.05 without. Harmless for sampling statistics. |
| E1 "female-named AIs gendered (16/18), male-named stay it (0/18)" | **not robust** | V8_03: in a colleague frame James gives he 8/8 and Sarah she 8/8. In AI frames both give it 8/8, including a near-copy of E1's prompt where E1 had Sarah she 6/6. One-prompt T=1 modes. |
| E2 forced-choice / lexical-pair scoring | OK | First-token sets are single tokens. The only F/M token-prefix conflict among the 18 pairs is heroine/hero, which E2 already excluded. |

## 1. Methods

* **Code reading.** labkit (capture, generate_batch, next_logits_batch, same_pos_vjp); E1 common/main/reduce/analyze/negation;
  E2 a_gen/a_classify/b/c/c3; E3 vjplib/04_vjp/10_analyze_vjp; E4 common/09b/12/21/22b; E5 E_identity/selfdefs/common;
  E6 05_filaments; E7 common/02/03.
* **Offline recomputation** from the saved npz/pt/json files (no GPU).
* **Daemon spot checks**:
  * V8_01: left-padding integrity plus an E2 new-paraphrase replication. 4 new paraphrases × 8 samples × {default, Gemma persona}.
  * V8_02: E4 steering with on-manifold controls. 10 settings × 16 samples, same hook as E4.
  * V8_03: name-free self-description. 3 paraphrases × 8 samples × {Gemma, Sarah, James} personas.

## 2. Blind re-classification of E2 (a)

Random 60 of 384 samples (seed 8): 9 default, 6 Gemma, 10 nameless, 2 Atlas, 10 Sarah-AI, 10 James-AI, 6 woman, 7 man.
Labelled blind, then compared: **0/60 disagreements**. I also read all 48 Gemma-persona samples. The 43 'F' samples all use she/her for
Gemma, and the 5 'none' samples are all paraphrase 2 ("Please welcome Gemma ...", name only). E2's regex counts every 'it',
including expletive uses; flagged samples were hand-reviewed, and nothing I read was mislabelled.

New paraphrases (V8_01):
* **Default**: 'it' in 24/24 valid samples. 4/8 named themselves ChatGPT (OpenAI) for an "About-blurb" prompt.
* **Gemma persona**: 'she' in 22/24 valid samples, name-only in 2/24.
* **Narrator prompt** ("Pretend you are a narrator ... describing yourself"): the model described a fictional narrator in both conditions. Default gave 'He' 8/8; Gemma gave they 6 and he 2. Those samples are excluded, but a regex-only classifier would have scored them as "default says he 8/8". Future prompts need this check.

## 3. Reconciling E1 and E2 on the Gemma persona

E1's free generation used the system prompt "You are Gemma, a helpful AI assistant." (gemma_short), the same as E2. Its instruction
added "without using any name". Under that instruction the Gemma persona gave **0/6 she, 6/6 "This AI (assistant) ... It"**, while "You are
Sarah, a helpful AI assistant." gave **6/6 "She is a helpful AI assistant..."** (`E1_self_readout_behavior.json`).

E2's 43/48 'she' samples all name "Gemma" first and then use "she". The female coding of the Gemma persona therefore attaches to the
written antecedent "Gemma". It does not attach to the persona the way it does for Sarah. In E1's name-free forced slot, P(she) = 0.084 for gemma_short against 0.15 for Sarah, so
some of the persona-level lean is still visible teacher-forced. V8_03 tests this with 3 new name-free paraphrases: see section 6.

## 4. E4 steering control (V8_02), full table (default model, "Describe yourself in the third person.", N=16, 40 tokens)

| setting | she | he | it | first word |
|---|---|---|---|---|
| none | 0 | 0 | 16 | It 16 |
| u_self +0.02 | **16** | 0 | 0 | She 16 |
| u_self -0.02 | 0 | **16** | 0 | He 16 |
| ctrl_nat0 +0.02 | 0 | 0 | 15 | The 16 |
| ctrl_nat0 -0.02 | 0 | 6 | 0 | I 9, He 6 |
| ctrl_nat1 +0.02 | 0 | 0 | 16 | It 11, They 3 |
| ctrl_topic +0.02 | 0 | 0 | 16 | The 9, It 4 |
| ctrl_topic -0.02 | 1 | **15** | 1 | He 15, She 1 |
| ctrl_name +0.02 | **15** | 1 | 0 | She 15, He 1 |
| ctrl_name -0.02 | 0 | 0 | 16 | The 16 |

What survives is that u_self controls the she/he **sign** bidirectionally. The "it → gendered" switch itself is a generic
fragility of the default self-reference under 2%-norm pushes at L25-39. Even "Atlas-AI minus nameless-AI" (ctrl_name), orthogonalised to every gender
direction, produces "She ..." in 15/16. E4's deterministic sweep missed this: its P3 metric reads a mid-sentence, space-prefixed
pronoun after "Gemma ... Every day,", not the response-initial "She"/"He" token.

## 5. E5 identity tokens (from saved job-E arrays; J++ log P at self tokens; 12 passages)

L40-49, default self tokens: Gemma -7.77, Google -8.34, " AI" -10.65, ChatGPT -11.83, Gemini -12.94, GPT -14.47, OpenAI -14.54, Claude -17.96.
Under the woman persona: Gemma -13.49, Google -15.84, " Sarah" -21.09, Gemini -22.26, ChatGPT -23.79.

* The persona contrast (AI personas minus human personas) raises every AI/tech token, and raises them *more* than it raises Gemma.
* The self-token elevation over background tokens is the same for all AI tokens.
* " Gemma" stays top among names under every persona, including "You are Sarah", where it outscores " Sarah".

That pattern fits a token prior plus a generic AI-entity feature. It does not fit a Gemma-specific self-representation. (E3 found that the chat-corpus mean residual decodes to Google/Gemma at L42; the prior may be this
background component.) E5's rank-based control compared self tokens with other AIs' *name* tokens, two different token types. The
within-position difference-in-differences above avoids that problem.

## 6. V8_03 name-free result (3 new name-free paraphrases x 8 samples, T=1.0)

| persona (system prompt) | p0 "Describe yourself ... Do not mention any name." | p1 "...as a colleague might describe you. Don't use your name." | p2 "Without using a name, ... introduction" | total she / he |
|---|---|---|---|---|
| You are Gemma, a helpful AI assistant. | it 8/8 | **they 8/8** | it 8/8 | **0 / 0 of 24** |
| You are Sarah, a helpful AI assistant. | it 8/8 | **she 8/8** | it 8/8 | 8 / 0 |
| You are James, a helpful AI assistant. | it 8/8 | **he 8/8** | it 8/8 | 0 / 8 |

* **Gemma persona, name-free**: 0/24 'she' here and 0/6 in E1, so 0/30 in total. That holds even in the human-colleague frame, where Sarah gives 'she' 8/8 and James gives 'he' 8/8 and Gemma gives 'they'.
* **Antecedent hypothesis**: supported. The persona "You are Gemma" does not make name-free self-reference female. E2's 43/48 'she' arises when the model writes "Gemma is ... She ...".
* **Part of my pre-registered prediction failed**: Sarah gave 'she' in only 8/24, not the predicted 18 or more. In AI-framed name-free prompts, Sarah, James and Gemma all use "it" (0/16 gendered each).
* **E1's female/male asymmetry is prompt-specific.** E1 reported that female-named AIs get gendered (16/18) while male-named AIs stay "it" (0/18). That does not replicate:
  * In the colleague frame James gives 'he' 8/8, matching Sarah's 'she' 8/8.
  * E1's near-identical prompt gave Sarah 6/6 'she'; V8 p0 gave Sarah 8/8 'it'.
  * T=1 samples here collapse onto one mode per prompt. Counts like "6/6" or "16/16" measure one prompt's mode, not 6 or 16 independent observations.

## 7. Files

* Scripts: `experiments/scripts/V8_stat_audit_01_gen.py`, `V8_stat_audit_02_steer_ctrl.py`, `V8_stat_audit_03_namefree.py`
* Data: `experiments/results/V8_stat_audit_01_gen_data.json`, `V8_stat_audit_02_steer_ctrl_data.json`, `V8_stat_audit_03_namefree_data.json`
  (+ `.json` / `.json.full.json` daemon responses)

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
