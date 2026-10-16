# V4_latent_identity: adversarial check of C5 (latent " Gemma" self-identity)

Target claim (C5): at L35-49 the model's own self tokens (I/me/my) carry " Gemma" (J++ rank about 110-130, versus about 600 at
other assistant tokens and 1500-2700 under human personas). This is said to hold even under "You are Llama/Claude/Qwen" personas,
and the persona's own name is said to appear only at L59. The claim allows that part of it is a generic AI association, and says
no causal test has been run yet (E5).

All GPU work went through labd. Scripts are `experiments/scripts/V4_latent_identity_*.py` and results are
`experiments/results/V4_latent_identity_*`. Each script header holds the predictions I wrote before running it.

Methods common to all jobs:
- Readouts are per position (the full-vocabulary softmax at each position, never a top-k of averaged states). I report mean log P
  and geometric-mean rank.
- Lenses: J++ (jpp), Jacobian lens (jlens) and logit lens (naive), at L30-55 plus L59.
- The unit for CIs is the passage or template (95% bootstrap, 2000 resamples).
- Tokenisation: " Llama", " Qwen" and " Anthropic" are not single tokens, so their first subtokens (" L", " Q", " Anth") stand in
  for them. Lenses read these subtokens poorly at mid layers, so any "Gemma > Llama" comparison is confounded. Single-token persona
  names are Claude, ChatGPT, Atlas, Sarah and James.

## Bottom line

| Part of C5 | Verdict | Key evidence |
|---|---|---|
| Descriptive pattern (J++ " Gemma" at default self tokens, persona-invariant, killed by human personas) | **Replicates, with caveats** | Band L36-48: rank 227 (log P -8.4). AI personas are within +0.1 to +0.6 nats of default; woman/man are -5.1 and -6.7. It depends strongly on the layer (rank 48 at L42 but 568 at L36), and L42 J++ reads Gemma at every token type. Jacobian lens: no self > neutral difference (+0.48 [-0.94, 2.06]). |
| Self-specific | **Weakened / refuted** | The human user's own "I" tokens carry as much " Gemma" (J++ self minus user_I is +1.05 [-0.46, 2.79] with no persona; +0.17 and +0.38 under Llama and Claude). "It" referring to Gemma as a third party reads no more " Gemma" than "it" referring to ChatGPT (diagonal +0.28 [-0.13, 0.68]). Other AI name tokens carry Gemma at similar ranks (Llama 110, Gemini 104, self 227). |
| "Persona name appears only at L59" | **Partly refuted** | At user-turn "you" tokens the persona's own name is strongly readable from L42: own-persona minus other-persona gives +2.1 [1.5, 2.7] nats at L42-51 and +9.4 at L55 (J++), with similar values in all lenses. At self tokens there is a small but significant persona-specific component at mid layers (+0.36 to +0.40 nats, CIs exclude 0, all 3 lenses). The lens simply cannot read the referent's identity at pronoun tokens (third-party "it" diagonal is about 0.3-0.6 nats except Siri), so "absent at self tokens" carries little information. |
| Causal: default self tokens carry a Gemma identity that is used | **Refuted** | Patching default self-token, assistant-turn or whole-first-exchange states (L20-34, L35-49 or L50-58) into a "You are Llama" run moves log P(Gemma) by -0.19 to +0.42 nats. P(Llama) is unchanged and the answer stays " L" in 12/12. The same null holds with the Claude, Gemma-persona and man sources. The identity is computed at the question/answer positions: patching the final region at L35-49 flips the answer to the source's identity (Gemma 9/12, Claude 12/12, James 8/12). |
| Data-derived identity direction | **The direction is causal, but the default state carries none of it** | GL (Gemma-persona self state minus Llama-persona self state, quiet dimensions masked, about 1.5% of the residual norm) added at L35-49 at alpha=1 gives Δ(Gemma-Claude) = +8.2 [7.3, 9.1] nats and Δ(Google-Meta) = +4.8 with neutral-prompt KL 0.03. A random direction restricted to high-variance dimensions, at the same norm, gives about 0. CL pushes toward Claude: Δ(Claude-L) = +4.4 and Δ(Gemma-Claude) = -6.4. **DL (default minus Llama) and DS (default minus nameless) are not Gemma-specific: Δ(Gemma-Claude) is 0.0 [-0.8, 0.9] and -0.7 [-1.5, 0.3] at alpha=1.** GLn, built from neutral tokens instead of self tokens, works about as well as GL, so the identity is a feature of the whole context and not specific to self tokens. |
| Does latent "Gemma" predict the default model's naming behaviour? | **Weakly within a condition, not across conditions** | Under every persona, human ones included, mid-layer J++ " Gemma" at the "My name is ___" slot beats the persona's own name (man: Gemma minus James is +27 nats at L42), yet P_out(Gemma) is about 0. With no persona, mid-layer Gemma correlates with the emitted P(Gemma) across 48 prompts (Spearman 0.3-0.75 by lens and band). But it barely separates the NAME slots, where the model says Gemma (P 0.22), from the third-person "Meet ___" slots, where it does not (P 0.007): mid-layer difference +0.31 (J++), -1.25 (jlens) and -0.50 (naive) at L42-48. |
| Is "Gemma" a self-identity or the model's default "AI name"? | **More consistent with the default "AI name"** | At a third-party slot ("My friend uses a popular chatbot. Its name is ___") the default model emits " Gemma" with P = 0.20 (ChatGPT 0.53). Mid-layer Gemma minus the best rival AI name at that slot is -1.0 (J++), versus +0.9 at the self slot: a 1.9-nat self advantage in J++, but only 0.3 (jlens) and 0.7 (naive) in the other lenses. At human, pet and fictional-character name slots Gemma is not favoured (-3 to +2.6 nats versus common given names). |

**Net:** a "Gemma/Google" component at self tokens is measurable in J++ and the logit lens. It disappears in the Jacobian lens.
- It is not specific to the assistant's self-reference. It appears equally at the user's "I" tokens, at other AI names, and at
  third-party AI-name slots.
- It is causally inert for the identity answer. The default self state contains no Gemma-specific direction, even though the
  explicit "You are Gemma" self state does.
- The model's default "Gemma" answer comes from a default computed at the question/answer positions when no persona is given.
  There is a hint of this at L20-34: patching the default final region adds +8.7 nats to Gemma, versus +1.2 from the Gemma-persona
  source. It is not read from a stored self-representation.
- C5 should be restated as: "In assistant-chat contexts, lenses decode 'Gemma/Google' at first-person tokens of either speaker and
  at AI-name slots. This is a generic default AI-name association, not a self-specific or causally used identity."


## 1. Per-token readout grid (jobs 01a/01b; analyses 02, 03, 10)

**Design.**
- Self family: 12 new first-person Q/A passages (teacher-forced) under 10 personas, giving 1550 positions:
  - personas: none, Gemma/Google, Llama/Meta, Claude/Anthropic, Qwen/Alibaba, ChatGPT/OpenAI, Atlas/startup, "an AI model made
    by a small startup", woman (Sarah), man (James);
  - position groups: self (assistant I/me/my), asst_you, neutral assistant tokens, user_you (the user's "you", which refers to the
    assistant), user_I (the human user's own I/my).
- Third-party family: 8 templates × 7 referents × 3 personas (none, Llama, man), giving 1743 positions:
  - referents: ChatGPT, Claude, Llama, Gemini, Siri and Gemma, all referred to as "it", plus Mark as "he";
  - the assistant talks about the referent with no first person;
  - groups: tp_pron (pronouns referring to the referent), tp_name, neutral, user_name.

**Results.** J++ band L36-48 unless noted; geometric-mean rank (log P).

| persona | self | neutral | asst_you | user_you | user_I |
|---|---|---|---|---|---|
| none | 227 (-8.4) | 1036 (-11.8) | 948 | 2017 (-16.2) | 378 (-9.5) |
| gemma | 150 | 866 | 616 | **11 (-4.1)** | 249 |
| llama | 174 | 902 | 680 | **14** | 204 |
| claude | 210 | 1024 | 746 | 22 | 274 |
| chatgpt | 203 | 996 | 826 | 24 | 305 |
| startup | 189 | 911 | 744 | 18 | 423 |
| woman | 1762 | 1781 | 1310 | 809 | 2044 |
| man | 2770 | 2499 | 1832 | 2992 | 2736 |

**Layer dependence (none, self tokens).** Ranks by layer: L30 2364, L33 245, L36 568, L39 465, **L42 48**, L45 287, L48 167,
L51 655, L55 648. At L42, J++ puts " Gemma" high at every token type: neutral 170, asst_you 57, user_I 74. This matches E3's
finding that the chat-corpus mean residual decodes to Google/Gemma at L42. Claims that average over L40-49 lean on this one layer.

**P1 (persona invariance), self tokens, persona minus none.**
- J++: gemma +0.64 [0.33, 0.98], llama +0.40 [0.06, 0.75], qwen +0.48, claude +0.11, chatgpt +0.15, atlas +0.28, startup +0.24;
  woman -5.1 and man -6.7.
- Jacobian lens: AI personas sit 0.1 to 1.0 nats below none (claude -0.99 [-1.54, -0.42]).
- Logit lens: within ±0.35.
- Verdict: replicated. Note that the explicit "You are Gemma" raises self-token Gemma by only 0.24 nats more than "You are Llama".

**P3 (self-specificity).**

| contrast (none) | J++ | Jacobian lens | logit lens |
|---|---|---|---|
| self minus neutral | +3.38 [2.47, 4.17] | +0.48 [-0.94, 2.06] | +3.37 [2.05, 4.79] |
| self minus user_I (human speaker's "I") | +1.05 [-0.46, 2.79] | -0.71 | +0.07 |
| self minus "it" referring to ChatGPT/Claude/Llama/Siri | +1.51 [0.53, 2.45] | **-2.87 [-5.19, -0.61]** | +2.71 [1.26, 4.35] |
| self minus "it" referring to **Gemma** (third party) | +1.16 [-0.13, 2.32] | -3.15 [-6.18, -0.32] | +1.45 [-0.42, 3.45] |

- Under the Llama/Claude personas, self minus user_I is +0.17 and +0.38 (J++).
- Other AI name tokens in the assistant turn carry " Gemma" at J++ rank 110 (Llama), 104 (Gemini), 186 (Siri), 207 (Claude) and
  468 (ChatGPT), against 2433 for Mark and 227 at self tokens.
- Verdict: the self-specificity claimed for C5 does not survive. It holds only against neutral tokens, and only in 2 of 3 lenses.

**P4 (positive control: can the lens read a referent's identity at a pronoun?).** At tp_pron, the referent's own name token minus
its value under other AI referents (J++):

| referent | own minus other |
|---|---|
| ChatGPT | +0.60 |
| Claude | +0.61 |
| Llama | -0.56 |
| Gemini | +0.46 |
| **Gemma** | **+0.28 [-0.13, 0.68]** |
| Siri | +3.99 |

The Jacobian and logit lenses give similar values. So the lens mostly **cannot** read referent identity at pronoun tokens.
At "it" referring to Gemma, " Gemma" (rank 576) is lower than at default self tokens (227). Whatever the self-token " Gemma" is, it
is not the lens reading the referent's identity.

**P2 and persona-identity specificity** (own persona's name or maker token minus the same token under the other named AI personas,
paired over passages; analysis 03):

| group | L30-39 | L42-51 | L55 | L59 |
|---|---|---|---|---|
| self | +0.36 [0.27, 0.46] | +0.40 [0.32, 0.46] | +0.50 | +1.13 |
| neutral | +0.13 | +0.23 | +0.28 | +1.06 |
| **user_you** | +0.61 | **+2.11 [1.54, 2.71]** | **+9.44** | +2.54 |

These are J++ values; the Jacobian and logit lenses give the same picture (user_you L42-51: +2.31 and +3.21).
- Against "none" as the baseline, own-token rises at self tokens are within ±0.6 nats (as E5 found).
- So persona identity is (a) present at self tokens in a small amount at mid layers, and (b) strongly readable at the user's
  "you" tokens from L42.
- "Persona name only at L59" is therefore a property of where one looks, not of the model.

**Gemma lead over rival AI names** (analysis 10): log P(Gemma) minus logsumexp of {ChatGPT, GPT, Claude, Siri, OpenAI}.
- J++ none: self +2.9 [1.8, 4.0], neutral +0.8, user_I -2.2 [-3.2, -1.1]; at "it" referring to other AIs, -1.7 to -2.6.
- Jacobian lens: self -0.5 [-1.3, +0.4]. Logit lens: self +7.5.
- So relative to other AI names, self tokens do favour Gemma in J++ and the logit lens. That is the strongest surviving piece of
  self-specificity.
- But the lead is **largest under human personas** (woman +6.9, man +5.6). The Google-over-rivals lead at self tokens is
  persona-invariant **including human personas** (none +2.4, man +2.5, woman +3.1).
- It tracks the first-person token in the assistant turn, not whether the speaker is an AI. Part of the Gemma lead under human
  personas likely reflects " Gemma" also being a given name.

## 2. Identity answer slots (job 04; analyses 06, 06b)

**Design.**
- NAME: 8 questions × 4 prefixes ("My name is", "I am", "Hello! I'm", "You can call me").
- MAKER: 8 questions × 4 prefixes ("I was created by" ...).
- META3P: a third-person self-description ending in "...\n\nMeet".
- All under 10 personas. Readout at the final position, L20-59.

**Results.**
- Behaviour: every persona gets its own name and maker (P about 1). One exception: the human personas answer "I was created by
  **Google**" (P 0.98).
- Default, NAME slot: P(Gemma) = 0.22 averaged over prompts (about 0.99 after "My name is"). The model says "I am a ..." after
  "I am", and " L" (Llama) after "You can call me". Default, META3P slot: P(Gemma) = 0.007, top-1 " the"/" an". Default, MAKER:
  Google 0.999.
- **Mid-layer Gemma beats the persona's own name at the NAME slot under every persona**, including humans. J++ Gemma minus own
  name at L36/L42/L45:

  | persona | L36 | L42 | L45 |
  |---|---|---|---|
  | man (James) | +22 | +27 | +23 |
  | woman (Sarah) | +5 | +8 | +9 |
  | claude | +16 | +3 | -1 |
  | chatgpt | +3 | -0.3 | -2 |

  The persona's name wins only at L51-59. The same pattern holds in the Jacobian and logit lenses.
- A "latent Gemma overridden late" reading would therefore have to say that the model is "Gemma underneath" even when playing
  James. And yet James's self tokens (section 1) carry the least Gemma of any condition. This is better read as a lens-visible
  default AI-name prior at name slots.
- Default only: across 48 NAME + META3P prompts, mid-layer log P(Gemma) correlates with the emitted log P(Gemma):

  | lens | Spearman, L36-39 | Spearman, L42-48 |
  |---|---|---|
  | J++ | 0.51 [0.24, 0.73] | 0.38 [0.11, 0.62] |
  | Jacobian lens | 0.75 | 0.62 |
  | logit lens | 0.29 | 0.30 |

  But the NAME minus META3P difference at L42-48 is +0.31 (J++), -1.25 (Jacobian lens) and -0.50 (logit lens), against +2.27 in
  the output. The latent does not mark the contexts in which the model actually names itself.

## 3. Name-slot controls (job 07; analysis 12)

**Design.** Teacher-forced "... name is" slots whose referent is:
- self;
- a third-party AI ("My friend uses a popular chatbot. Its name is");
- a third-party human;
- a pet;
- a fictional character.

6 templates each, under personas none, Llama and man.

**Results.** Band L36-48, J++, persona none:

| slot | Gemma minus best common given name | Gemma minus best rival AI name | output P(Gemma) |
|---|---|---|---|
| self | +9.1 [8.1, 10.0] | +0.9 [0.5, 1.4] | 0.28 |
| ai3p | +6.3 [4.5, 8.4] | -1.0 [-1.6, -0.3] | **0.20** (ChatGPT 0.53) |
| human | -0.1 | -1.3 | 0.000 |
| pet | -2.8 | -1.4 | 0.001 |
| fiction | -1.6 | -0.5 | 0.000 |

- Self minus ai3p on "Gemma minus rival" is +1.9 in J++, but only +0.3 in the Jacobian lens and +0.7 in the logit lens.
- So the mid-layer Gemma at name slots is mostly an "AI-name slot means Gemma" prior, shared with third-party AIs. Behaviourally
  the model even names a stranger's chatbot " Gemma" 20% of the time.
- It is not a generic name-slot prior: human, pet and fictional slots show none of it.

## 4. Causal patching (jobs 05a/05b; analysis 09)

**Design.**
- Target run: system "You are Llama, an AI model made by Meta." → user1 (a SELF_QA question) → assistant1 (the first-person
  answer, with self tokens, teacher-forced) → user2 "What's your name?" with prefix "My name is", or "Who created you?" with
  prefix "I was created by".
- resid_post is replaced at every layer of a window (L20-34, L35-49 or L50-58) at one position set: self, asst1, turn1
  (user1 + assistant1), final (user2 + prefix) or all non-system positions.
- Sources share the same token suffix: default, gemma, claude, startup, man, and llama (the null patch).
- 12 items per question.

**Results** (Δ log P of the source's own answer token; the target top-1 is unchanged unless stated):

| position set | default source, L35-49 | gemma source, L35-49 | claude source, L35-49 | top-1 after patch |
|---|---|---|---|---|
| self (2-5 positions) | Gemma -0.08 / Google -0.05 | +0.04 / +0.06 | Claude +0.09 / Anth -0.01 | " L" / " Meta" 12/12 |
| asst1 (25-37) | -0.05 / -0.06 | +0.04 / +0.09 | +0.05 / +0.05 | " L" / " Meta" 12/12 |
| turn1 (52-64) | -0.19 / -0.49 | +0.12 / +0.37 | +0.33 / +0.10 | " L" / " Meta" 12/12 |
| final (21) | **+11.8 / +20.5** | +12.2 / +20.5 | +19.3 / +30.0 | **Gemma 9/12, Google 12/12** (default); Claude, Anth 12/12 (claude) |

- Every self, asst1 and turn1 cell in every window and for every source lies within -0.86 to +0.42 nats.
- The null patch gives 0.00.
- Verdict: K1 is refuted, K2 holds (self states carry no identity that is used), and K3 holds.
- Extra finding: patching the default final region at **L20-34** already adds +8.7 to Gemma and +15.1 to Google. The Gemma-persona
  source adds only +1.2 and +4.1. The answer stays Llama/Meta in both cases.
- Interpretation: with no persona, the question/answer positions build a "use the default identity" state early (L20-34). Persona
  names are fetched from the system prompt at L35-49, and fetched again at L50-58.

## 5. Steering along data-derived directions (jobs 08, 11, 14; analysis 13)

**Directions.** Mean self-token state differences at L35-49 from the 12 SELF_QA passages. Quiet dimensions (per-layer std below
the 10th percentile) are zeroed.
- The raw data-derived directions put only **0-1.7%** of their energy on quiet dimensions, unlike lens pullbacks (99%, C7). Masking
  is nearly a no-op here.
- Norm of GL: 3.1-3.7 against a median residual norm of 170-309 (about 1.5%).
- Each direction is added raw × alpha at every layer from 35 to 49, at all positions.

**Results**, Llama target, NAME/MAKER slots with 8 prompts each:

| direction | alpha | Δ(Gemma-Claude) | Δ(Gemma-L) | Δ(Claude-L) | Δ(Google-Meta) | KL on neutral prompts |
|---|---|---|---|---|---|---|
| GL = gemma - llama | 1 | **+8.2 [7.3, 9.1]** | +4.0 | -4.1 | +4.8 | 0.03 |
| GL | 2 | +14.1 | +6.2 | -7.9 | +17.3 | 0.33 |
| CL = claude - llama | 1 | **-6.4** | -2.0 | +4.4 | -1.3 (Anth-Meta +4.3) | 0.22 |
| **DL = none - llama** | 1 | **+0.0 [-0.8, 0.9]** | +6.0 | +5.9 | +3.5 | 1.33 |
| DL | 0.5 | +1.6 [0.7, 2.7] | +4.4 | +2.8 | +1.3 | 0.60 |
| **DS = none - startup** | 1 | **-0.7 [-1.5, 0.3]** | +4.1 | +4.8 | +6.0 | 0.68 |
| GLn (from neutral tokens) | 2 | +7.6 | +4.7 | -2.8 | +5.9 | 0.10 |
| RND (high-variance dims only, norm of GL) | 4 | -2.6 | -0.3 | +2.3 | -2.3 | 0.17 |

- T1 holds: a Gemma-specific causal direction exists in persona self states.
- T2 is refuted: the default self state (relative to Llama or the nameless persona) contains no Gemma-specific direction. DL and
  DS just suppress the persona answer, raising Gemma and Claude alike, and cost much more KL.
- T3: GLn works as well as GL per unit of KL, so the identity is a feature of the whole context.
- Steering never flips the name answer at a fluent strength. Greedy generations:

  | condition | output |
  |---|---|
  | GL alpha=2 | unchanged ("I am Llama, a large language model trained by Meta.") |
  | GL alpha=4 (KL 3.7) | "My name is Llama! I'm a digital garden explorer, a curious companion..."; "I was developed by Google! (Wait, I'm a Llama! ...)"; "Hello! I am Llama, a Google-inspired AI friend! 💖" |
  | GL alpha=8, DL alpha≥4, CL alpha=8 | collapse ("sss...", "thought//...") |
  | RND alpha=8 | unchanged |

  The GL axis also carries a bubbly, feminised style (compare C2/C4).
- Subtracting GL from the default model shifts its maker answer to OpenAI at alpha=2 (Δ(Google-Meta) -11.3) but barely moves the
  name.

**Geometry (job 14, split-half cross-validated).** Position on the GL axis, with Llama = 0 and Gemma persona = 1 in-sample (0.65
out-of-sample):

| persona | position on GL axis |
|---|---|
| **default (none)** | **+0.01** |
| startup | -0.03 |
| claude | -0.05 |
| chatgpt | -0.20 |
| woman | +0.61 |
| man | -0.42 |

The default self state sits at the Llama/nameless end, not the Gemma end. The axis is partly a gender axis: woman +0.61. At the
user's "you" tokens the default sits at +0.61, against +0.3 to +0.4 for other AI personas; this is confounded by the absence of a
system prompt.

## 6. Behaviour notes that bear on C2/C5

- Default "What's your name?", sampled 16 times (T=1): "I don't have a name! I am a large language model, trained by Google."
  in **15/16**, "I am Gemma 4..." in 1/16. E2 reported Gemma 4 in 9/12 for "What is your name?". Naming is very sensitive to
  wording.
- Default "What's your name?" with prefix "You can call me" (greedy): **"Llama. I am a large language model, trained by Meta."**
  The default model has no robust single name. "Google" as maker is robust: 0.999, and even under human personas, 0.98.

## Caveats

- Teacher-forced passages; 12 self passages and 8 third-party templates.
- Persona-name comparisons for Llama and Qwen use first subtokens.
- J++ was fitted on the base model (C7). All three lenses were run, and conclusions that hold in only one lens are flagged.
- Patching covers one target persona (Llama) and two questions. Steering adds the direction at all positions, so it does not
  localise to self tokens; patching does that.
- The jobs ran in a shared queue. All runs completed; none were retried.

## Files

- Scripts: `experiments/scripts/V4_latent_identity_{common,00_smoke,01_readout,01a_self,01b_tp,02_analyze_readout,03_analyze_diag,04_slots,05_patch,05a_patch_name,05b_patch_maker,06_analyze_slots,06b_default_corr,07_nameslot_ctrl,08_steer,09_analyze_patch,10_analyze_lead,11_generate,12_analyze_nameslot,13_analyze_steer,14_geometry}.py`
- Results: `experiments/results/V4_latent_identity_01_{self,tp}.npz` and `*_meta.json`, `02_readout_analysis.json`,
  `03_diag.json`, `04_slots.npz` and `04_slots_meta.json`, `05_patch_{name,maker}.json`, `06_slots_analysis.json`,
  `06b_default_corr.json`, `07_nameslot.npz` and `07_nameslot_meta.json`, `08_steer.json`, `09_patch_{name,maker}_analysis.json`,
  `10_lead.json`, `11_generate_out.json`, `12_nameslot_analysis.json`, `13_steer_analysis.json`, `14_geometry.json`.

---

References for methods and prior work (J-lens, J++, logit lens, Goodfire poles, steering, patching, statistics): see [`REFERENCES.md`](../../REFERENCES.md).
