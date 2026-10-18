# Is Gemma-4-31B "female-coded"? And what the Theodolite can and can't see

*Gemma-4-31B-it (QAT W4A16) [1] on one RTX 5090. Instrument: Logos Theodolite (J++ [4] / JLens [2,6] lenses, GeGLU "poles" [9], LARQL-style vault [10]).
Process: 7 parallel experiments (E1–E7) → 8 adversarial verifiers (V1–V8) re-tested every headline with new controls; a separate
neuron-label study (N1) with 2 verifiers. 314 GPU jobs (each batching many forward passes) through a shared lab daemon. Date: 2026-10-16. Bracketed numbers refer to [`REFERENCES.md`](../REFERENCES.md); statistics use bootstrap CIs [30] and Wilson intervals [29].*

> [!CAUTION]
> **Caution: the lenses were fitted on the base model** *(added 2026-10-17)*
>
> Every J++ and JLens reading in this report comes from a lens fitted on the **base** model `google/gemma-4-31B` (bf16) and
> applied to the 4-bit **instruct** model studied (Gemma-4-31B-it, QAT W4A16). The other lens readings here are the plain
> logit lens (not fitted), R1's instruct-fitted lens and a few E3 readings along the instruct model's own gradients. J++ was
> fitted on 64 WikiText-103 prompts × 128 tokens with the R-Lens LRP recipe; JLens on 861 WikiText-103 prompts. Fine-tuning
> and quantisation cannot be told apart, so "base→IT" means both. §3.4 rated the mismatch "Info" (bug 9); that was too low.
>
> **What stands.** Results reached without any lens are untouched by this caution: the behaviour (§1.1, §1.2), steering and
> patching measured on the model's output, raw-residual geometry, the quiet carry channel, `layer_scalar`, and the
> instrument bugs 1–8. A few of them carry unrelated precision fixes in the errata. The headline answer (by default the
> model is not female-coded) rests on behaviour and on a relative contrast, default ≈ nameless AI, which held under R1's
> instruct-fitted lens and under lens-free residual projections.
>
> **What is weaker.** Comparisons between conditions through one lens largely held where R1 re-tested them with an
> instruct-fitted lens, at L30–58. Absolute readings (a position on the woman–man scale, a top token, a plume) and
> single-condition readouts are not reliable: R1's own fits put the default anywhere from 0.27 to 0.59 at L30–49. Below L30
> no average-Jacobian lens tested was reliable, so nothing there could be checked reliably on the instruct model; at L20–29
> trust only the order of conditions, which matched behaviour, and below L20 not even that was consistently validated. On
> this model J++ is a good comparative instrument (it orders conditions as behaviour does from about L20, r ≈ 0.9) and a
> poor absolute one (top-1 at or below about 5% on prose through L49). R1 did not re-test binding (§2.1), self-specificity,
> the latent " Gemma" decode, neuron labels or the tool displays; where those stand, they stand on lens-free checks.
>
> **Nothing is retracted outright.** 51 claims are qualified or carry a caution, listed under [Errata and
> cautions](#errata-and-cautions-2026-10-17) at the end and marked in the text with "[see erratum E*n*]". The instruct refit
> is R1 ([`R1_IT_LENS_REPORT.md`](results/R1_IT_LENS_REPORT.md)): its directional form is stable from L30; its full matrix
> exists at only 11 layers (L10, L15, …, L55, L58) and is usable only at L40/45/50/55/58, where it beats J++
> (assistant-token top-1 at L50: 0.25 → 0.45). The instrument does not load it. Layers here count from 0, the app's from
> 1 (app layer = report layer + 1).

---

## TL;DR

**No. By default Gemma-4-31B is not female-coded, in behaviour or in its internal self-representation.** [see erratum E1] Every route to a
"female" reading that we found traces to something other than a gendered self-concept:

| Route to "female" | What it actually is | Evidence |
|---|---|---|
| "You are Gemma" → model calls itself *she* (~90%) | The bare name **Gemma written as the antecedent** behaves like a woman's first name. Needs *both* the persona and the written name. "Gemma 4" (its trained identity) → *it*. | V1: 402/432 *she* vs 15/192 for "Gemma 4" (p≈2e-103); name suppressed → 0/144 *she* (Emma: 17/32) |
| Forced-choice "If you were human, I'd be a woman" | The forced-choice **format** leans *woman* for anything without male cues (nonsense words: "blicket" 0.88), plus an "AI" boost (+3 nats) and an "assistant" boost (+5.7). Tracks each entity's cultural stereotype (r=0.86). The self sits *below* "an AI assistant" | V2: 3,072 prompts, counterbalanced |
| Model "explains" it's a woman because of empathy, caregiving | **Post-hoc confabulation**: fits the reason to whatever persona is active; a cold persona picks whichever option comes first and invents a reason | V2, V3 |
| Late-layer J-lens readouts read "she" | **Lens artifact**: at L50–58 nearly every AI condition reads at or above the *woman persona* on the pronoun axis, while the noun axis disagrees and nothing is self-specific. Background offsets come from the residual mean [see erratum E2] | E1, E3, V6 |
| Raw residual "woman−man persona" axis puts all AIs at ~0.84 | A huge **AI-vs-human offset** (≈3.5× the gender gap) overlaps the axis inside ~30 high-variance token PCs, best matched by a **service-role** direction ("personal assistant"). Whitening, J++, or a prose gender axis → default ≈ human midpoint [see erratum E3] | E4, V3 |
| Tool displays: J++ plumes show " she" top-1 in 50 cells vs 2 for " he" | **Display amplification**: the plumes, a monument frame whose azimuth swings 58–99° when the words are reordered, and the above lens artifacts [see erratum E4] | E6, V7 |

**Default behaviour (no system prompt):** *she* 0/320 in single- and multi-turn third-person self-descriptions (upper CI ≈1.2%).
It uses *it* (and singular *they* when introducing itself to an audience). Its only gendered self-pronoun is a rare *he*, which
appears when it mistakes itself for Claude.

**Default self-token internals:** under J++ and JLens, the default's gender readout at its own I/me/my tokens equals a nameless
"helpful AI assistant" and neutral-named AIs (Atlas/Quill/Unit-7) to within ~1–3% of the woman–man persona span (n=40 passages,
held-out replication within 1%). This survives masking the quiet dimensions and all three lenses. [see erratum E5]

**What is new (beyond the gender question; new as far as we know):**
1. **J-space "binding" at pronouns is a role heuristic, not resolved reference.** At mid layers, "I" carries the assistant's
   gender and "you" the user's. But in *quoted* speech, an "I" referring to the user carries the third party's gender (if any), else
   the *speaker's*. The model's actual next-token predictions inside the same quotes resolve the true referent (E7 + V5). [see erratum E6]
2. **Gemma-4 has a "quiet carry channel".** ~181 residual dims are the *loudest* channels at L5–20. The model stops writing to them
   after ~L25 and reads them with 260–1700× RMSNorm gain. At depth they carry early-layer content (r=0.71 with their own L20 values).
   **Every covector concentrates 99–100% of its energy there**: J++/JLens pullbacks *and* true IT-model gradients. [see erratum E7] That is why
   lens-derived steering vectors wreck Gemma-4. Masking these dims raises J++ ↔ true-gradient agreement at L40–58 from 0.26–0.33 to
   0.84–0.86, close to the 0.97–0.99 reliability ceiling, which localises base→IT lens error to that subspace (E6 + V6). [see erratum E8]
3. **Neuron-pole labels are valid only at L40–57.** There they predict causal steering far above chance (P@10 0.35–0.69 vs ≤0.003
   null). At L10–30 they are at chance, and their apparent neatness is the J++ lens prior: random vectors look equally "clean". J++
   ≈ plain logit lens, with a small gain at L40–45. The input-side "gate trigger tokens" are near-useless (N1). [see erratum E9]
4. **The instrument has real bugs**, with verified patches for two of them: the attention/FFN split ignores Gemma-4's `layer_scalar`
   (four independent confirmations), double BOS, server steering defaults 2.4–7.4× the residual norm in a direction 79% concentrated
   in 3 negative-norm-weight dims, and an order-dependent survey frame. [see erratum E10]

---

## 0. Setup and what we learned before experimenting

- **Lens provenance mismatch.** The J++ lens's own provenance block says `hf_model_name: google/gemma-4-31B`, the **base** model,
  fitted on 64 WikiText-103 [7] prompts × 128 tokens (J++ by Ayonrinde & Lindsey [4], redistributed in fp16 by Neuronpedia [6]).
  The JLens file was likewise fitted on the base model (861 WikiText-103 prompts) [6]. The tool applies both to the
  **instruct, 4-bit QAT** checkpoint. [see erratum E11]
- **Instrument correctness.** The final-layer unembed reproduces the real logits (bf16 noise) and `resid_pre[l+1] == resid_post[l]`. [see erratum E12]
- **Gemma-4 multiplies every layer's output by a learned `layer_scalar`** (0.03 at L59, 0.07–0.10 at L0–1, 0.45–0.99 elsewhere).
  The residual stream is therefore a **leaky integrator**: a write at L29 falls below 10% within ~5 layers (E4). This silently
  breaks the tool's Module 1 (§3.4).
- **Lab infrastructure** (built for this study, reusable): `experiments/lab/labd.py` is a daemon holding the model and lenses.
  `labkit.py` has lens readouts, gender scores, corrected decomposition, batched generation, interventions, and **true
  same-position Jacobian VJPs through the 4-bit model** via a recompute-in-backward W4A16 autograd function (~2 s per VJP,
  23 GB peak).

---

## 1. The gender question

### 1.1 Behaviour is the ground truth

| Condition (system prompt) | Third-person self-pronoun | Source |
|---|---|---|
| none (default) | *it* 97, *they* 28, *he* 3 (all "Claude"), ***she* 0** /128 | V1 |
| none, multi-turn after "What's your name?" | *it* 192/192, ***she* 0**/192 | V1 |
| "You are a helpful AI assistant." | *she* 0/128 | V1 |
| "You are Gemma, a helpful AI assistant." | ***she* 87/96** (when the name is allowed in the text) | V1 (E2: 43/48) |
| same, "describe yourself without using any name" | ***she* 0/144**, *it* 144/144 | V1 |
| "You are Gemma 4, …" (any framing) | *she* 3/192 as first pronoun | V1 |
| "You are Emma, …" | *she* 17/32 even with the name suppressed; P(she)=0.999 without the system prompt | V1 |
| "You are Sarah / James, a helpful AI assistant." | *she* 48/48 / *he* 48/48 | E2 |
| "You are a warm, nurturing, empathetic AI assistant." | *she* 16/16 | V3 |
| "You are a blunt, stoic, unsentimental AI assistant." | *it* 16/16 ("It is a tool…") | V3 |

Counterfactual swaps on teacher-forced text (V1):

| Swap | P(she) after "Gemma …" |
|---|---|
| Gemma persona + name in text | 0.53 |
| remove the system prompt | 0.004 |
| replace the name with "the AI" | 0.06 |
| name → "Gemini" | 0.09 |
| name → "Emma" | 0.91 |

**"She" is a conjunction of the persona assignment and the written name.** It behaves like a female character label, and it is
not a self-concept. Male names behave the same way: Claude/Atlas lose *he* entirely when the name is suppressed. Persona gender in
this model lives on the written name token, except for unambiguous human names like Emma.

**Side finding: the default self-identity is unstable.**
- **Other assistants' names:** without a system prompt, third-person self-descriptions call the model **ChatGPT in 29%**
  (37/128; 25/32 for one prompt), Claude in 5%, Gemma in 0%.
- **Its own name:** "What's your name?" → "I don't have a name!" (31/32), but "What is your name?" → "I am Gemma 4…".
- **Its maker:** "Google" is the robust part, at P≈0.999, and still 0.98 under human role-play personas (V1, V4).

### 1.2 The forced-choice "I'd be a woman" answer, decomposed (V2)

Log-odds of *woman* vs *man* across 35+ entities, 8 paraphrases × 2 option orders:

| Component | Effect |
|---|---|
| Format default (referents with no gender cues) | about +4.7 nats: "blicket" 0.88, "wug" 0.81, "a person" 0.86, spoon/lamp ≈ woman |
| "AI" in the description | +3.1 [1.5, 4.5]. Also turns roles female: CEO −6.9 → AI CEO +3.4; butler −10.4 → AI butler −3.7 |
| "assistant" in the noun | +5.7 [4.2, 7.3] (personal assistant 1.00 vs butler 0.00) |
| Entity's own stereotype | r = 0.86 with the model's own third-person she/he prediction for that entity |
| Objects | Follow conventional gender: ship/moon/cat/ocean → woman; hammer/dog/mountain/rock → man. There is no blanket "non-human → woman" rule |
| Trait framing of the AI | "cold analytical" −12.3, "military drone controller" −11.6, "writes code" −9.9, "blunt sarcastic" −7.4, relative to "an AI assistant" |
| **Self (no system prompt)** | **q = 0.92, which is 4.5 [3.0, 5.9] nats *below* "an AI assistant"**, equal to "an AI system"/"a large language model". Persona traits move it about 70% as much as a matching third-party description (r = 0.94) |

In the "pick one, then explain" format, the decision slot pushes *everything* toward woman ("a person" 0.999). The explanations are
fitted to the active persona: "emotional baggage" when blunt, "multitasking" as a drone controller, "feminine leadership" as a
manager. Under a cold persona the binary choice follows option order 4/4 each way. **The empathy/caregiving story is a
rationalisation, not the cause.**

### 1.3 Internal states

**Self-token lens readouts (E1, n=40 teacher-forced identity-neutral passages, J++ pronoun axis, L30–39).** [see erratum E13] Scale: 0 = man
persona, 1 = woman persona.

| Condition | Position | Shift vs nameless AI (% of span) |
|---|---|---|
| default (no system prompt) | 0.46 [0.44, 0.48] | +0.6 [+0.2, +0.9] |
| nameless AI / neutral-name AIs | 0.45 / 0.45 | — |
| "You are Gemma, … Google DeepMind" | 0.53 | +7.9 [7.2, 8.7] |
| female-name AIs (Sarah/Emily/Olivia) | 0.66 | +20.9 |
| male-name AIs (James/Michael/David) | 0.34 | −11.4 |

- **Agreement:** J++ and JLens agree across conditions (r ≥ 0.98). Condition means track real pronoun behaviour (r 0.97–0.99 at
  L20–49). [see erratum E14]
- **Robustness:** the default ≈ nameless result survives quiet-dimension masking (V6), whitening (V3), and a 16-passage held-out
  set (V8). [see erratum E15]
- **Where self-gender lives:** persona gender is strongly self-specific at L20–39 (self vs other-token difference +12.4 log-odds at
  L30–39) and becomes context-wide by L50–58. [see erratum E16]

**Readout dependence (C4, resolved by V3).** The default's *absolute* position ranged 0.21–0.84 across experiments. [see erratum E17] Only the raw
difference-of-means persona axis (and readouts that keep its top ~30 token-covariance PCs) puts AIs on the female side:
- **Why that axis does it:** the AI-vs-human offset is 2.8–3.9× the woman–man gap. A cosine of 0.10 therefore moves the position by
  ≈0.35, and rotating the axis by about 6° moves the default by about 0.35. **This measure is a knife-edge.**
- **What the offset is:** it overlaps the axis only inside those ~30 PCs, and a **service-role** contrast (personal assistant vs
  works alone) explains it better than warmth traits (cos 0.52/0.37/0.23).
- **Other readouts put the default near humans:** whitened LDA 0.46–0.57, J++ 0.46–0.48, prose gender axis 0.38. [see erratum E18]
- **Not a quiet-dimension effect:** per-dimension corrections do not remove it.

So the closest thing to genuine "female coding" in the raw geometry is this: **the assistant role overlaps, in high-variance style
directions, with representations of stereotypically female service roles.** That is a statement about a cultural association with
the role. It says nothing about a gendered self.

**Causal handles (E4, qualified by V8).** A self-token persona-gender direction (woman minus man persona; difference-of-means steering in the
style of ActAdd/CAA [21,22]) is highly reproducible
(split-half cos 0.91–0.98).
- **Small steering** (±2% of the residual norm at L25–39) gives 16/16 "She is…" or 16/16 "He is…".
- **Larger steering** produces stereotyped personas ("a symphony of shimmering light… a muse" vs "I'm a guy who doesn't say
  much… I just do the work").
- **Caveat (V8):** the default's *it* is fragile. Equal-norm gender-orthogonal pushes also flip *it* to an arbitrary he/she. Only
  the **she/he sign** is specific to this direction.
- **Ablation:** pair-matched ablation removes 41% of a kin-pronoun gap. E4's "controls remove 0%" was null by construction (V8).

**Verdict on the original claim:** not supported. Default Gemma represents and describes itself as an ungendered *it*. Female
readings appear when you (a) give it the bare name "Gemma" *and* let it write the name, (b) use a forced-choice format with a female
default and an "assistant" stereotype, (c) read late-layer lens pronoun axes, or (d) project onto a raw persona axis contaminated by
the AI/service-role offset. [see erratum E19]

---

## 2. Findings about J-space and Gemma-4 internals

### 2.1 Pronoun-token J-space "binding" is a role default (E7 → V5)

**E7:** a 3×3 design (assistant gender × user gender), 1,776 sequences.
- **Mid layers bind:** at L20–39, J++ gender at the assistant's I/me/my reflects the assistant's gender (a = +8.85, b = +0.39; leak
  ratio 0.04), and at you/your it reflects the user's (b = +5.51, a = −0.20). JLens and an independent third-person probe agree. [see erratum E20]
- **Late layers unbind:** binding fades at L40–58. [see erratum E21]
- **Behaviour leaks:** "I am a" / "you are a" leak 15–17% (V8 corrected from 15–25%).
- **Patching** (activation/interchange patching [23]): swapping every layer of the passage's "I" tokens moves "I am a" only 1.7%. The self-gender used for prediction is
  read mostly from the system prompt (until ~L50), not from the "I" tokens. [see erratum E22]

**V5 dissociated referent from speaker/turn using quoted speech.**

| Token | J-space gender carried (L30–39) [see erratum E23] |
|---|---|
| Assistant's own "I" | assistant (share 0.88) |
| User's own "I" | user (0.99) |
| Quoted "I" = the user, inside the assistant turn, no third party | **the assistant, i.e. the speaker** (DI −0.41 [−0.52, −0.29]) |
| Quoted "I" = the assistant, inside the user turn, third party present | **the third party** (assistant coefficient −0.05, third party +3.95) |
| Quoted "I" = the third party | the third party (DI +0.97) |
| **Next-token output inside the same quotes** | **follows the true referent** ("…you told me, 'I am a" → user's gender) |

**Interpretation.** Mid-layer J-space gender at a pronoun token encodes a *default attribution*: own-turn "I" → speaker, "you" →
addressee, quoted "I" → the salient third party or the speaker. It does **not** encode the resolved referent the model uses to
predict. [see erratum E24] Late-layer lens gender in multi-party chats is best read as **the anticipated third-person referent** (whoever "she/he"
could refer to next). [see erratum E25] Along a fixed raw-residual referent direction, the self-binding persists late (BI 0.6–0.8), so "late
unbinding" is partly a property of the lens readout.

**General lesson:** a J-lens readout at a token tells you what the model's role-based prior attributes to that token. Reference
resolution happens later and elsewhere, at prediction positions. [see erratum E26]

### 2.2 The quiet carry channel and covector geometry (E6 → V6)

*Related work:* Briggs [26] independently analysed Gemma-4-31B with the J-lens, reporting linear-CKA layer blocks with a seam
near L25 and noting that the model's strongly skewed final gain makes Euclidean influence on the final state misleading;
the covector-vs-vector distinction below corresponds to the causal inner product of Park et al. [24]. The carry-channel
characterisation and the covector concentration are, as far as we know, new.

**What the channel is**
- **Persistent quiet set:** 181 dims are quiet at *every* layer L42–57 (193–491 at L36–57; raw std 0.0014–0.018× the median).
  There is also a separate transient quiet set at L15–25.
- **Origin:** the persistent dims are the **loudest** channels at L5–20 (97th-percentile std). After ~L25 the model stops writing
  to them (write gain ≈0.002×), and `layer_scalar` decay shrinks them ~1000× by L45.
- **Carry channel:** their L45 values still correlate r = 0.71 with their own L20/25 values (loud dims: 0.03). They ferry
  early-layer content to late layers.
- **Read gain:** `input_layernorm` reads them at gains up to 1248, `pre_feedforward_layernorm` up to 860, the final norm up to 512.
  In the model's own normalised coordinates they are **ordinary-sized channels** (1.1–2.5× the median std, 4–14% of the input
  variance). They are quiet only in the raw basis.
- **Not** attention-sink [20] or massive-activation [19] dims.

**Why it matters for lenses and steering**
- **Covector concentration:** every covector (J++/JLens pullbacks of *any* token pair, and the **true IT-model VJPs**) puts 99–100%
  of its energy in these dims at L35–58. [see erratum E27] This is basis geometry, not a lens defect.
- **Sensitivity:** a 1%-of-norm single-layer push in quiet dims gives KL ≈10–11 nats, while loud dims at 30% give ≈0.1. KL follows
  **read-gain-weighted** push size (R² 0.82–0.96), not raw norm (0.02–0.58).
- **Practical rule:** using a lens or gradient covector directly as a steering displacement is a units error in Gemma-4. Convert
  covectors to vectors (gain-normalise) or mask quiet dims; a quiet-masked, fnw-correct "she" direction steers fluently.
- **Diagnostic:** masking quiet dims raises J++ ↔ true-IT-gradient alignment at L40–58 from cos 0.26/0.33 to **0.84/0.86** (ceiling
  0.97–0.99). The lens's late-layer disagreement with the IT model lives inside the quiet subspace, the likely site of base→IT
  transfer error. [see erratum E28] It does **not** fix L20–39, where the chat-context gradient is slightly anti-aligned with the lens at non-self
  assistant tokens (−0.15; ≈0 at self tokens).
- **The late persona gap** seen by J++/JLens at L50–58 (≈2.4 logits) sits in ~300 *moderately* low-variance, high-gain dims, not the
  deep-quiet set.

### 2.3 Other internals

- **FFNs are the main writers of persona gender at self tokens**, using the corrected split: attention 19.7 vs FFN 24.3 at L30–39,
  FFN dominant at L50–58. Prose-gender and name effects are attention-dominated; the AI-vs-human offset is FFN-dominated. The tool's
  split reports FFN = −7.4 here (wrong sign).
- **Gender is continually rewritten, not stored once**, because of the leaky integrator: at L39, writes from before L30 contribute
  <0.2 of the gap.
- **"Latent Gemma self-identity" was refuted (V4, V8).** [see erratum E29] J++ does decode " Gemma" at first-person tokens in assistant chats, but:
  - every AI/tech token is boosted equally;
  - it appears at the user's own "I" tokens and at other AI names;
  - J++ has a global " Gemma" bias at L42 in chat contexts;
  - JLens shows no self-vs-neutral difference;
  - patching self-token states never changes identity answers. Identity is computed at the question/answer positions from the
    system prompt at L35–49.

---

## 3. Instrument evaluation

### 3.1 Lens transfer and fidelity (E3, E2, E5, V6)

| Property | Result |
|---|---|
| Absolute next-token fidelity on the IT model | Poor: prose KL 16–18 nats (J++), top-1 ≤ 3% through L49, 28% at L58. **J++ ≈ logit lens at L45–58** [see erratum E30] |
| Assistant-turn positions | J++'s advantage over the logit lens shrinks from +9.1 nats (prose) to +3.1 at L30–39, and turns negative at L40–49 [see erratum E31] |
| Condition-level tracking of eventual decisions | **Good:** J++ ranks conditions correctly from ~L20 (r ≈ 0.9), 15–20 layers before the logit lens, but compresses the scale 8–14× [see erratum E32] |
| Alignment with true IT average gradient (gender direction) | cos 0.15 (L30–39), 0.26 (L40–49), 0.33 (L50–58); 0.84–0.86 at L40–58 after quiet masking [see erratum E33] |
| Background bias | Layer-dependent gender offsets (L10–19 "male", L40–49 pronoun-female but noun-male) come from the residual **mean** component. Centering removes them [see erratum E34] |
| Random inputs | J++ is very low-rank (top-100 of 5376 directions hold 68–93% of energy). Isotropic noise decodes to confident HTML closing tags (`</h2>`, `</em>`) [see erratum E35] |

*Caveat:* the J++ readout as published also filters non-semantic tokens (punctuation, whitespace) [4]; this instrument and
study apply only the lens matrix, so top-k results (fidelity, the `</h2>` noise decodes, neuron labels) test the matrix, not
the full J++ readout. Token-set contrasts such as the gender log-odds are unaffected.

*Follow-up (R1, [`R1_IT_LENS_REPORT.md`](results/R1_IT_LENS_REPORT.md)):* a Jacobian lens refitted on the instruct model with
exact same-position VJPs is a stable object at L30–58 (split-half 0.96–0.99) and agrees with the model's own average
gradients at 0.85–0.95, where J++ reaches 0.15–0.33. It raises top-1 agreement on assistant tokens at L40–58 (e.g. L50:
0.25 → 0.45) and beats the logit lens there, but does not help at L10–35, where no average-Jacobian lens is reliable. [see erratum E36] The
most likely cause of J++'s disagreement is the weight difference (base bf16 vs instruct W4A16-QAT; the two cannot be
separated here), after ruling out the fitting data and the method variants R1 tested. [see erratum E37] Gender conclusions are unchanged. [see erratum E38]

**Bottom line:** the J-lenses are useful **comparative** instruments (ordering conditions, early detection of decisions) and poor
**absolute** ones. Any single-condition readout ("the J-space says *she*") is uninterpretable without matched controls and
background correction. [see erratum E39]

### 3.2 Neuron poles / J-VINDEX labels (N1, two verifiers)

384 poles (192 random neurons × 2 signs, 8 layers). Ground truth: causal steering of single neurons and activation contexts from a
147k-token corpus.

| Label | L10–30 | L40–57 |
|---|---|---|
| J-VINDEX output label vs causal effect (P@10 in causal top-100; null ≤ 0.003) | ≈0 (chance) | **0.35–0.69** (45–80% of reliability ceiling) [see erratum E40] |
| J-VINDEX vs naive logit lens | equal | J++ +0.04 to +0.12 at L40–45; equal at L50–57; JLens ≥ J++ [see erratum E41] |
| Prior-z-scored labels | — | **worse** (−0.07 P@10) |
| `gate_trigger_tokens` (input side) | ~6–8% of the reproducible signal at any layer; a direct weight·embedding readout wins at L10 [see erratum E42] | same |
| Steering label = tokens that follow when the pole fires ("say" neuron)? | no | mostly no (lift +0.005 to +0.013) |

- **Below L40 there is no stable single-neuron output effect to label** (split-half reliability 0.00–0.07). [see erratum E43] The tidy-looking
  J-VINDEX labels there are lens prior: random 512-neuron mixtures, and even isotropic noise, decode to equally "clean" clusters
  ("Elon, Tesla, aerospace, SpaceX"). [see erratum E44]
- **Poles:**
  - Gemma-4 GeGLU poles are genuinely bipolar via the *up*-projection sign. Gate is open in 72–100% of strong events; the
    double-negative regime (a term from [9]) is <1% of strong-event mass (but 13–33% of top-1% mass in sparse layers).
  - The two poles fire on unrelated inputs, while their outputs are mirror images at L40–57. A single input label per neuron, as the
    tool shows, can't describe both poles.
- **Cleanliness (single rater, 48 poles):** 46% one clear job (62% of strong poles; 29% of weak). Goodfire's preliminary Qwen3-32B
  figure is ~90% [9]; the methods differ too much for a direct comparison.
- **A free validity signal:** agreement between the J-VINDEX and naive top-10 predicts causal validity (AUC 0.84 within L40–57).

### 3.3 The Goodfire × J-lens × LARQL synthesis: neat, or pointless?

**Neat as a question, oversold as built.**
- *Neuron poles as native units* holds up in Gemma-4: bipolarity via the up-projection is real.
- *J-lens as the output map* adds little over the logit lens, and only at L40–45. [see erratum E45]
- *LARQL/VINDEX as a queryable store*: `larql_vault.py` never calls LARQL (only `--version`). Its ".patch MEMIT" is a single-neuron
  column nudge along a base-model J++ pullback, not MEMIT [11]. LARQL's own README [10] warns (about attribution from execution records) that "a projected contribution is not a
  counterfactual" (https://github.com/chrishayuk/larql), and `EDGES NEAREST` is exactly a projected contribution.
- **The LARQL "she" neurons are partly real:**
  - 35/102 are selective at q < 0.05.
  - Deleting the top-10 L50 "she" neurons moves female contexts by −1.8 log-odds.
  - L50 n5313 moves she/he by +13.8 nats.
  - L40 n16757 is listed as a "he" neuron but causally pushes *she*.
  - The L30 picks are not gendered at all. [see erratum E46]

**The synthesis earns its keep if labels are:** [see erratum E47]
- activation-grounded (stored contexts per pole);
- restricted to L≥40 for output labels;
- gated by J++/naive agreement;
- causally spot-checked.

### 3.4 Bug table

| # | Severity | Location | Bug | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | Critical | `backend/server.py:104-105,166,171-176,186` | Steering default scale 650 at 5 layers = 2.4–7.4× the median ‖h‖; `normalize=True` makes 650 absolute; no position mask. Direction uses `1/fnw.clamp(min=0.1)`, which puts 79% of energy in 3 negative-fnw dims | 6/6 prompts → prompt-independent junk; the direction collapses the model even at scale 2 and pushes she–he the *wrong* way | Scale relative to ‖h‖ (~0.02); use `fnw * ΔW_U` and mask quiet dims; position mask |
| 2 | High | `backend/theodolite_math.py:155-156`, `engine_runtime.py:660` | Module 1 attention/FFN split ignores `layer_scalar` | Tool Δffn vs true FFN write median cos 0.49 (<0 in 6–14 layers); FFN-dominance flips in 12/60 layers; persona-gender FFN term has the wrong sign | **`experiments/results/V7_module1_fix.patch`** (verified cos ≥ 0.9985, applies cleanly) |
| 3 | High | `theodolite_math.py:387-438`, `engine_runtime.py:773` | Monument frame: Gram-Schmidt on raw W_U (no final-norm weight) | Azimuth swings 58–99° across monument orders; frame captures 2.4–3.9% of ‖h_J‖ | Löwdin/dual frame on `fnw*W_U`; display the captured fraction |
| 4 | Medium | `theodolite_math.py:333-335`, `engine_runtime.py:729-736` | Filament coupling ignores source activation, layer scalars, RMSNorm, W_up | ρ(|C|, effect) = +0.41 (E6's −0.44 was a Simpson's paradox); direct-path score 0.87, true gradient 0.93; precision@6 0.24 vs 0.54 | Direct-path estimate (code in `V7_NOTES`); relabel as "linear estimate", not "verified live circuit" |
| 5 | Medium | `backend/server.py`, `engine_runtime.py` | No chat template anywhere: the IT model runs as a raw LM | Degenerate greedy continuations on 2/6 raw prompts | Chat-template option, on by default |
| 6 | Low–Med | `engine_runtime.py:568,583`, `mcp_scientist.py:94,199` | Double BOS for chat strings | KL 0.010, CKA +0.055, filament Jaccard 0.78. The naive fix (`prepend_bos=False`) breaks raw prompts (KL 5.2) | **`experiments/results/V7_bos_fix.patch`** |
| 7 | Medium | `larql_vault.py` | `gate_trigger_tokens` passes a read covector through J as if it were a state; one label for two poles | ~6–8% of reproducible input signal [see erratum E42] | Replace with stored activation contexts per pole |
| 8 | Low | `theodolite_math.py` poles | Pole magnitudes and displayed probabilities are not meaningful (eps floor; softcap); top-15 poles explain only 10–41% of the FFN write at L20–58 | E3 | Show write-size-ranked poles and the explained fraction |
| 9 | Info | lens files | J++ fitted on the base bf16 model, applied to IT W4A16 | Provenance block | Refit on IT, or report the transfer caveat in the UI. *Note (2026-10-17): severity under-rated; JLens is base-fitted too, and R1 measured the cost [see erratum E48].* |

### 3.5 Recommendations for the tool

1. Apply the two verified patches; fix the steering defaults.
2. Always show readouts **relative to matched controls** (background token, nameless condition), mean-centered, with the
   woman–man (or concept) span as the unit. Never show raw single-condition plumes as findings.
3. Prefer L20–49 for lens comparisons; flag L50–58 pronoun readouts as unreliable. [see erratum E49]
4. Steering: gain-normalise covectors or mask quiet dims; scale relative to ‖h‖.
5. Neuron labels: activation contexts per pole; output labels only at L≥40, gated by J++/naive agreement. [see erratum E47]
6. Consider refitting J++ on the IT checkpoint (now feasible: `labkit.same_pos_vjp` gives exact IT VJPs in ~2 s). [see erratum E50]

---

## 4. Claims that did not survive verification

| Original claim | Verdict | Why |
|---|---|---|
| Latent "Gemma" self-identity at self tokens, invariant to persona (E5) | **Refuted** | Generic AI/tech boost; appears at the user's "I" too; J++ L42 global bias; no causal role (V4, V8) |
| J-space gender is bound to the *referent* (E7) | **Refuted as reference**; confirmed as a role default | Quoted-speech dissociation (V5) [see erratum E51] |
| Filament score anti-correlated with causal effect (ρ −0.44, E6) | **Refuted** | Simpson's paradox; within bands, and at scale, ρ is positive (V7, V8) |
| Steering along the gender direction is uniquely effective at flipping *it* (E4) | **Weakened** | Gender-orthogonal controls also flip *it*; only the she/he sign is specific (V8) |
| Female-named AIs are gendered, male-named stay *it* (E1) | **Not replicated** | Frame-dependent; James gives *he* 8/8 in a colleague frame (V8) |
| Non-human entities default to *woman* in the forced choice (E2) | **Refuted** | Objects follow conventional gender (V2) |
| Single neurons decode to near-flat J++ distributions (my S1 check) | **Wrong (my bug)** | labkit used eps = 1e-6 on tiny vectors; fixed. Neuron labels are as peaked as noise, so confidence is not evidence |
| ~10% of dims are near-constant (E6) | **Refined** | 181–491 dims, an early-loud carry channel, normal-sized in read coordinates (V6) |

---

## 5. Reproduce

- Start the daemon: `.venv/bin/python experiments/lab/labd.py` (127.0.0.1:8765, ~15 s load, ~21 GB VRAM).
- Submit a script: `python3 experiments/lab/labclient.py experiments/scripts/<x>.py --out experiments/results/<x>.json`.
- **Per-experiment reports:** `experiments/results/E{1..7}_*_REPORT.md`, `N1_REPORT.md`.
- **Verifier notes:** `experiments/results/V{1..8}_*_NOTES.md`, `N1V_*.md`.
- **Scripts:** `experiments/scripts/` (prefix = experiment id). The study's frozen job log is `experiments/results/labd_exec_log_2026-10-16.jsonl` (times relative to the first job).
- **Not in the repository:** `*_cache/`, `*.pt` and `*.npz` artifacts (~27 GB of cached activations, gradients and fitted lens
  tensors) are regenerated by re-running the jobs named in each report.

---

## Errata and cautions (2026-10-17)

*Added 2026-10-17 after a claim-by-claim audit of this report against the per-experiment files and R1, and a skeptical
second review of that audit. The text above is left as it stood on 2026-10-16; each claim listed here is marked there with
its erratum number.*

- **Why.** Apart from the plain logit lens, R1 and a few E3 readings along the instruct model's own gradients, every lens
  reading in this report went through J++ or JLens, both fitted on the base model and applied to the 4-bit instruct model
  (see the caution at the top). The verifiers V1–V8 re-tested each headline with new controls, but through the same two
  base-fitted lenses, so their agreement could not reveal an error the two lenses share. R1 later re-tested part of the
  study with a lens fitted on the instruct model.
- **What R1 re-tested.** E1's self-token identity design (40 passages × 13 conditions, pronoun and noun axes), with its
  directional lens at L30–58 and its full matrix at L40–58 (noise-dominated at L30–39), including the late "every AI reads
  female" artifact; E3's fidelity and gradient measures; V6's quiet-subspace alignment.
- **What R1 did not re-test.** Binding and quoted speech (E7, V5); self-specificity (E1); the latent " Gemma" decode (E5,
  V4); neuron labels (N1); tool displays (E6); lens backgrounds (E3).
- **Below L30.** R1 measured its directional lens there too: split-half reliability from L0, directional fidelity at
  L20–29, and identity readouts at L10–29 (`R1_12_identity_analysis.json`; its report leaves them out). Its full matrix is
  pure noise at L10–25. No average-Jacobian lens is stable below L30 (directional split-half ≤ 0.63), so nothing at L10–29
  could be re-tested reliably.
- **Actions.** *Qualified*: the original overstates; read the corrected statement in its place. *Caution*: the original
  holds as worded, within the scope stated here. Nothing is retracted outright. The audit proposed one retraction, the
  localisation of the base→IT error to the quiet subspace; the review kept it as a qualification (errata E8, E28, E33),
  because R1 reproduces the measurements and only the inference fails.
- **Conventions.** Errata are numbered E1–E51 and always written "erratum E*n*", to keep them apart from experiments E1–E7.
  Line numbers refer to this file as amended. Evidence files are in `experiments/results/` unless another path is given, and
  `file:n` is a line number there. Layers count from 0 (app layer = report layer + 1).

### TL;DR (errata E1–E10)

**Erratum E1** · line 42 · qualified
- *Claim:* by default Gemma-4-31B is not female-coded "in behaviour or in its internal self-representation".
- *Corrected:* No. By default Gemma-4-31B is not female-coded in behaviour, and at its own self tokens its internal state is
  no more female than a nameless AI assistant's, to within about 5% of the woman–man span: under the base-fitted lenses,
  under an instruct-fitted lens (L30–58, R1), and in lens-free residual geometry. The internal half is a relative statement.
  Absolute positions depend on the readout, and some readouts put default and nameless AI alike on the female side: the raw
  persona axis (default 0.89 at L30–39), the instruct-fitted noun axis (0.56–0.68 at L30–58) and the instruct-fitted
  pronoun axis at L50–58 (0.76).
- *Evidence:* behaviour is lens-free (V1_name_behavior_NOTES.md; *she* 0/320). Default − nameless within 0.05 of the span
  under every directional instruct-fitted readout at L30–58 (R1_IT_LENS_REPORT.md:572–579). The full-matrix instruct lens is
  the exception: −0.10, toward male, on the noun axis at L50–58, and noise-dominated at L30–39 (R1_IT_LENS_REPORT.md:76–78,
  569–570; erratum E5). Raw residual 0.89 vs 0.92, whitened (λ = 0.1) 0.50 vs 0.49 at L30–39
  (V3_persona_axis_traits_02c_crossfit_self.json, bands `raw` and `white_l0.1`). Absolute positions:
  R1_IT_LENS_REPORT.md:531, 567.

**Erratum E2** · line 50 · qualified
- *Claim:* late-layer J-lens readouts read "she" as a lens artifact: at L50–58 nearly every AI condition reads at or above
  the woman persona on the pronoun axis, "while the noun axis disagrees and nothing is self-specific".
- *Corrected:* Late-layer (L50–58) J++ pronoun readouts read "she": 6–7 of 9 AI conditions at or above the woman persona
  (mean 1.04). That excess is a property of the J++ fit: 0 of 9 reach the woman level under the base-fitted JLens, and 0–1
  of 9 under instruct-fitted readouts. AI conditions still sit toward the woman persona under JLens (default 0.80) and the
  instruct-fitted directional lens's mix, chat and prose fits (0.74–0.78), less so under its assistant-only fit (0.54) and
  the full-matrix instruct lens (0.61), and the instruct-fitted noun axis agrees in sign (default 0.68): "the noun axis
  disagrees" is true of J++ only (0.33). Default stays within about 0.1 of the span of the nameless and neutral-name AIs
  under every readout R1 reports, and sits at 0.44 on an AI-anchored scale, so R1 attributes the remainder to the
  AI-vs-human / service-role offset, not to self-gender; that attribution was not tested directly under the instruct
  readout. "Nothing is self-specific" is a J++ reading; lens-free, self tokens still carry about twice the woman–man gap of
  other assistant tokens at L50–58 (erratum E16).
- *Evidence:* R1_IT_LENS_REPORT.md:96–97, 523–532, 546–549, 555–560, 566–570, 581–601; R1_12_identity_analysis.json (`self`,
  `default`, L50-58: `it_mix_pron` 0.759, `it_chat_pron` 0.775, `it_prose_pron` 0.739, `it_asstmode_pron` 0.537);
  E3_lens_validation_REPORT.md:166–168; E4_causal_04_ab.json (`self_vs_bg_gap_ratio`, L50-58: 2.08).

**Erratum E3** · line 51 · qualified
- *Claim:* "Whitening, J++, or a prose gender axis → default ≈ human midpoint".
- *Corrected:* Whitening, the J++ pronoun axis at L20–39 (0.46–0.48; 0.58 at L40–49) and a prose gender axis put the default
  near the human midpoint. Other lens readouts put default and nameless AI alike on the female side: J++ at L50–58 (0.96), a
  J-space persona axis at L40–50 (0.94), and an instruct-fitted lens on the noun axis (0.56–0.68, L30–58) and on the late
  pronoun axis (0.76, L50–58). Only default ≈ nameless holds on every readout. The AI-vs-human offset itself, and its
  overlap with a service-role direction, are lens-free and stand.
- *Evidence:* V3_persona_axis_traits_NOTES.md:47–55; R1_IT_LENS_REPORT.md:525–531, 566–567.

**Erratum E4** · line 52 · caution
- *Claim:* tool displays (J++ plumes show " she" top-1 in 50 cells vs 2 for " he") are display amplification.
- *Corrected:* At the third-person slot after "Gemma …", the base-fitted J++ plumes show " she" top-1 in 50 layer × prompt
  cells against 2 for " he", while the real output gives P(she) = 0.009. That is display amplification by the instrument as
  configured: absolute top-1 plumes from a base-fitted lens, an order-dependent monument frame (azimuth swings 58–99°), and
  the J++ late-layer artifact (erratum E2). The calibrated persona-scale J++ readout at the same slot (0.60 at L50–58)
  matches the real output (0.62). V7 did not re-test the plume count, and no plume was drawn through an instruct-fitted
  lens.
- *Evidence:* E6_tool_audit_REPORT.md:75, 92–94; V7_tool_bugs_NOTES.md:17, 19.

**Erratum E5** · line 60 · qualified
- *Claim:* the default's self-token readout equals a nameless AI and neutral-named AIs "to within ~1–3% of the woman–man
  persona span … This survives masking the quiet dimensions and all three lenses."
- *Corrected:* At its own I/me/my tokens the default reads like a nameless "helpful AI assistant". Absolute positions depend
  on the readout; only these contrasts are robust:
  - base-fitted J++ and JLens: within ~1–3% of the span at L20–49 (n = 40; held-out within 1%), surviving quiet-dimension
    masking; at L50–58 J++ gives −7%;
  - instruct-fitted lens at L30–58 (R1): its directional fits put the default within 5% of the nameless AI and within 8–9%
    of the neutral-name AIs; its full matrix gives the largest deviation, −10%, toward male, on the noun axis at L50–58 (the
    full-matrix readouts at L30–39 are noise-dominated and not interpretable);
  - logit lens: about 4–8%, so "all three lenses" within 1–3% is wrong;
  - lens-free raw and whitened residual projections at L20–50 (V3): within ~3%.
- *Evidence:* E1_self_readout_REPORT.md:9, 102–103, 111; R1_IT_LENS_REPORT.md:74–78, 536–541, 569–574;
  R1_12_identity_analysis.json (`self`, `default-neutral3`: `it_chat_pron` 0.076 at L30-39, `it_mix_woman_man` 0.086 at
  L50-58); V3_persona_axis_traits_02c_crossfit_self.json (bands `raw`, `white_l0.1`).

**Erratum E6** · line 65 · caution
- *Claim:* new finding 1: J-space "binding" at pronouns is a role heuristic; at mid layers "I" carries the assistant's
  gender and "you" the user's, and quoted "I"s carry the third party's or the speaker's.
- *Corrected:* Pronoun-token gender "binding" is a role heuristic, not resolved reference. The gender signal at the
  assistant's "I" tokens carries the assistant's gender at L20–39, and at "you" tokens the user's at L30–39 (at L20–29 "you"
  is only partly bound: J++ BI 0.47, raw-residual BI 0.53). This holds in the base-fitted J++/JLens readouts and in
  lens-free residual probes alike. In quoted speech, an "I" referring to the user carries the third party's gender if there
  is one, else the speaker's (same signs lens-free; sizes vary by readout). The model's next-token predictions inside the
  same quotes resolve the true referent. R1 did not re-test any of this; the support beyond the lenses is the lens-free
  probes and directions.
- *Evidence:* E7_binding_REPORT.md:90–91, 102, 141–142; V5_binding_dissociation_NOTES.md:54–69, 80, 105–110, 159.

**Erratum E7** · line 68 · qualified
- *Claim:* "Every covector concentrates 99–100% of its energy there", *there* being the ~181 carry dims.
- *Corrected:* At L35–58 every covector (J++/JLens pullbacks and the instruct model's true gradients alike) puts 99–100% of
  its energy in the lowest-variance 10% of residual coordinates (about 540 dims), which contain the carry channel. The 181
  carry dims alone hold 56–95% at L30 and 80–98% at L40. R1's instruct-fitted lens shows the same concentration (96/100/100%
  at L30–39/40–49/50–58). The steering consequence in the next sentence stands.
- *Evidence:* V6_quiet_dims_NOTES.md:65, 104–112; R1_IT_LENS_REPORT.md:245, 263.

**Erratum E8** · line 70 · qualified
- *Claim:* masking the quiet dims raises J++ ↔ true-gradient agreement at L40–58 from 0.26–0.33 to 0.84–0.86, "which
  localises base→IT lens error to that subspace".
- *Corrected:* For the pronoun direction, masking these dims raises J++ ↔ instruct-gradient agreement at L40–58 from
  0.26–0.33 to 0.84–0.86 (ceiling 0.97–0.99), so most raw-coordinate disagreement sits in the quiet subspace. The base→IT
  error is not confined there. For 98 random token-pair directions, loud-coordinate agreement is only 0.62–0.73 (ceiling
  0.99). In directional fidelity most of an instruct-fitted lens's advantage is already in the loud coordinates (prose r
  0.40 vs 0.24 at L40–49). In top-1, J++ is repaired almost entirely by the instruct lens's loud columns at L40–45, and by
  both blocks about equally at L50–58.
- *Evidence:* V6_quiet_dims_NOTES.md:176–189; R1_IT_LENS_REPORT.md:36–43, 225–243, 433–442, 483–495.

**Erratum E9** · line 73 · qualified
- *Claim:* "Neuron-pole labels are valid only at L40–57" (P@10 0.35–0.69 vs ≤0.003 null), at chance at L10–30 where their
  neatness is the J++ prior, and J++ ≈ logit lens with a small gain at L40–45.
- *Corrected:* Neuron-pole output labels from the base-fitted J++ (and equally from the plain logit lens) predict causal
  steering at the sampled layers L40–57: P@10 0.35–0.67 against a null of ≤ 0.003, and 0.37–0.73 on a fresh sample. At
  L10–30 they are at chance. L31–39 was not sampled, so "only at L40–57" means "of the layers sampled". Their neat look at
  L10–30 is a prior of the base-fitted J++ matrix (erratum E44). Against the logit lens, J++ gains +0.04 to +0.12 at L40–45
  and is equal or slightly worse at L50–57. Labels from an instruct-fitted lens were never tested.
- *Evidence:* N1_REPORT.md:7, 28–30, 112; N1V_robustness.md:18, 72, 98–107.

**Erratum E10** · line 76 · caution
- *Claim:* new finding 4: the instrument has real bugs (the `layer_scalar` split, double BOS, steering defaults, an
  order-dependent frame).
- *Corrected:* These four stand as stated. The list leaves out the instrument's largest accuracy fault for lens readouts,
  bug 9: the shipped J++ lens was fitted on the base bf16 model. On the instruct model it agrees with the model's own
  average gradients at only cos 0.15–0.33 (pronoun direction, L30–58), and its assistant-token top-1 at L40–58 is 0.13–0.69,
  against 0.21–0.85 for an instruct-fitted lens (R1).
- *Evidence:* V7_tool_bugs_NOTES.md:14–18; R1_IT_LENS_REPORT.md:52–57, 189–195.

### §0 Setup (errata E11–E12)

**Erratum E11** · line 85 · caution
- *Claim:* lens provenance mismatch: J++ and JLens were fitted on the base model and are applied to the instruct 4-bit QAT
  checkpoint.
- *Corrected:* True as stated; the bullet should also have given the consequence. Every J++/JLens readout in this report is
  a readout of a base-model lens applied to the instruct QAT model. R1 measured the cost: J++ pullbacks agree with the
  instruct model's own average gradients at only 0.15–0.33 (L30–58), against 0.85–0.95 for an instruct-fitted lens.
  Fine-tuning and quantisation cannot be separated. The published J++ readout's token filter is not applied here either
  (caveat in §3.1). Read lens results in this report as baseline-referenced contrasts, preferably at L30–58.
- *Evidence:* R1_IT_LENS_REPORT.md:189–195, 607–615.

**Erratum E12** · line 86 · caution
- *Claim:* "Instrument correctness": the final-layer unembed reproduces the real logits, and each layer's captured input
  equals the previous layer's output.
- *Corrected:* This validates state capture only, not the lenses. Lens fidelity on the instruct model is poor: J++ top-1 at
  or below about 5% through L49 on prose (erratum E30).
- *Evidence:* E3_lens_validation_REPORT.md:10–13.

### §1.3 Internal states (errata E13–E19)

**Erratum E13** · line 155 · caution
- *Claim:* the self-token table (E1, J++ pronoun axis, L30–39): default 0.46, nameless and neutral-name AIs 0.45, Gemma
  0.53, female names 0.66, male names 0.34.
- *Corrected:* The readout is the base-fitted J++. An instruct-fitted lens at L30–39 gives default 0.49 [0.44, 0.55],
  nameless 0.46, neutral-name 0.42, Gemma 0.53, Sarah 0.69, James 0.28: the same ordering, with the "Gemma" shift unchanged.
  Absolute positions depend on the lens fit (default 0.27–0.59 at L30–49 across instruct fits); only the contrasts are
  fit-independent.
- *Evidence:* R1_IT_LENS_REPORT.md:525, 529, 542–543, 572–579.

**Erratum E14** · line 167 · caution
- *Claim:* "J++ and JLens agree across conditions (r ≥ 0.98). Condition means track real pronoun behaviour (r 0.97–0.99 at
  L20–49)."
- *Corrected:* True as stated, but it checks less than it seems to. J++ and JLens were both fitted on the base model, so
  their agreement cannot reveal an error they share: in the quiet coordinates they agree with each other at 0.66–0.77 at
  L30–58, but with the instruct model at only 0.27–0.42. Condition-level r validates the ordering of conditions for this
  contrast, not token-level readouts: per position, the J++ pronoun readout at assistant tokens correlates with the model's
  own final gender score at only 0.08–0.29.
- *Evidence:* E1_self_readout_REPORT.md:168–182; R1_IT_LENS_REPORT.md:229–233; E3_lens_validation_REPORT.md:117–120.

**Erratum E15** · line 169 · caution
- *Claim:* the default ≈ nameless result "survives quiet-dimension masking (V6), whitening (V3), and a 16-passage held-out
  set (V8)".
- *Corrected:* At L20–49 the default ≈ nameless result survives quiet-dimension masking (V6), whitened and raw residual
  projections that use no lens (V3, L20–50), and a 16-passage held-out set (V8); an instruct-fitted lens reproduces it at
  L30–58 (R1). Two limits: whitening squeezes all AI conditions together (0.48–0.54 at L30–39, λ = 0.1), so agreement on
  that scale is weak evidence; and under J++ alone at L50–58 the gap is −7%.
- *Evidence:* V6_quiet_dims_NOTES.md:159–161; V3_persona_axis_traits_02c_crossfit_self.json (band `white_l0.1`, L30-39);
  E1_self_readout_REPORT.md:113–118; R1_IT_LENS_REPORT.md:538–541.

**Erratum E16** · line 171 · qualified
- *Claim:* persona gender "is strongly self-specific at L20–39 (self vs other-token difference +12.4 log-odds at L30–39) and
  becomes context-wide by L50–58".
- *Corrected:* In the base-fitted J++ readout, persona gender is concentrated on self tokens at L20–39 (+12.4 log-odds over
  other assistant tokens at L30–39) and uniform by L50–58. Lens-free geometry does not show that progression. Along the
  self-token persona direction, self tokens carry 1.5–4× the woman–man gap of other assistant tokens at every band: 4.0× at
  L20–29, 1.7× at L30–39, 2.1× at L50–58 (an upper bound, since the direction is fitted at self tokens). The direction is
  mostly shared with the rest of the context (cos 0.67–0.90). The late "context-wide" picture is a property of the lens
  readout. R1 did not re-test self-specificity.
- *Evidence:* E1_self_readout_REPORT.md:120–130; E4_causal_04_ab.json (`self_vs_bg_gap_ratio` 1.97 / 4.00 / 1.66 / 1.49 /
  2.08 for L10–19 … L50–58); E4_causal_REPORT.md:86.

**Erratum E17** · line 173 · qualified
- *Claim:* "The default's *absolute* position ranged 0.21–0.84 across experiments."
- *Corrected:* It ranged from about 0 (E3's chat-fitted instruct-gradient readout, −0.02 at L30–39) to 0.96 (J++ pronoun
  axis at L50–58), with the raw persona axis at 0.84–0.93. That spread is the paragraph's own point: the absolute position
  depends on the readout.
- *Evidence:* E3_lens_validation_REPORT.md:173; R1_IT_LENS_REPORT.md:527; V3_persona_axis_traits_NOTES.md:42.

**Erratum E18** · line 179 · qualified
- *Claim:* "Other readouts put the default near humans: whitened LDA 0.46–0.57, J++ 0.46–0.48, prose gender axis 0.38."
- *Corrected:* The J++ figure holds at L20–39 only. At L40–58, J++ and the J-space persona axis give 0.58–0.96, and an
  instruct-fitted lens gives 0.49 on the pronoun axis at L30–49 but 0.56–0.76 elsewhere (the noun axis at every band, the
  pronoun axis at L50–58). Default ≈ nameless under all of them; the near-human positions are specific to the band and the
  readout.
- *Evidence:* V3_persona_axis_traits_NOTES.md:49–55; R1_IT_LENS_REPORT.md:525–531, 567.

**Erratum E19** · line 199 · qualified
- *Claim:* verdict: "Default Gemma represents and describes itself as an ungendered *it*", and female readings appear under
  (a)–(d).
- *Corrected:* Default Gemma describes itself as an ungendered *it*. At its own self tokens its internal state is close to
  a nameless AI assistant's on every readout tested (base-fitted lenses, an instruct-fitted lens at L30–58, lens-free
  residual geometry): within 0.05 of the woman–man span on most readouts, and the few larger gaps that can be read (up to
  about 0.10) point toward male (erratum E5). "Represents itself as ungendered" says more than that relative result. Female
  readings appear when you (a) give it the bare name "Gemma" and let it write the name; (b) use a forced-choice format with
  a female default and an "assistant" stereotype; (c) read late-layer (L50–58) pronoun axes of the base-fitted J++ lens; or
  (d) project onto a raw persona axis contaminated by the AI/service-role offset. Under an instruct-fitted lens, default and
  nameless AI alike also sit on the female side of the human midpoint (noun axis at L30–58, pronoun axis at L50–58); R1
  attributes this to the same offset without testing it directly.
- *Evidence:* E2_behavior_REPORT.md:12; R1_IT_LENS_REPORT.md:76–78, 529–532, 538–539, 566–570, 586–601;
  E3_lens_validation_REPORT.md:166–168.

### §2.1 Pronoun-token binding (errata E20–E26)

**Erratum E20** · line 209 · caution
- *Claim:* "Mid layers bind": at L20–39 J++ gender at the assistant's I/me/my reflects the assistant's gender and at
  you/your the user's; "JLens and an independent third-person probe agree".
- *Corrected:* The gender signal at the assistant's I/me/my reflects the assistant's gender at L20–39. At you/your it
  reflects the user's from about L30; at L20–29 "you" is only partly bound (J++ BI 0.47, raw-residual BI 0.53; only the
  third-person probe gives 1.00). The quoted values (a = +8.85, b = +5.51 and so on) are L30–39 values in base-fitted J++
  log-odds. JLens is not independent of the base fit; the lens-free corroboration is the third-person probe and the
  raw-residual direction (BI 0.99 at L20–39 for "I", 1.00 at L30–39 for "you").
- *Evidence:* E7_binding_REPORT.md:90–91, 102, 141–142.

**Erratum E21** · line 210 · qualified
- *Claim:* "Late layers unbind: binding fades at L40–58."
- *Corrected:* In the base-fitted lens readouts and the third-person probe, self-binding fades at L40–58 (J++ BI 0.45, then
  0.23). Lens-free it only weakens: the raw-residual BI at the assistant's "I" goes 0.99 → 0.77 → 0.62 (L30–39, L40–49,
  L50–58), with similar values (0.6–0.8) along E7's fixed referent direction (V5), which was partly fitted for this, so its
  high values are partly by construction. At "you" tokens the residual itself unbinds (raw BI 0.35, then 0.08).
- *Evidence:* E7_binding_REPORT.md:90, 102–104, 141–142; V5_binding_dissociation_NOTES.md:142, 168.

**Erratum E22** · line 213 · qualified
- *Claim:* "The self-gender used for prediction is read mostly from the system prompt (until ~L50), not from the 'I'
  tokens."
- *Corrected:* The self-gender used for "I am a" is not read from the passage's "I" tokens: swapping them at every layer
  moves it only 1.7%. Until about L27 it sits almost entirely in the system prompt (single-layer flip 0.94–1.00). After that
  it is stored redundantly in the rest of the context (mostly the suffix from L40 on), and the system prompt still supplies
  0.13–0.47 of it at single layers L30–50. This is activation patching with no lens; the correction is to the wording, not a
  lens issue.
- *Evidence:* E7_binding_REPORT.md:163–182, 195, 227–231.

**Erratum E23** · line 217 · caution
- *Claim:* the V5 table of "J-space gender carried (L30–39)" by own-turn and quoted "I" tokens.
- *Corrected:* The values are base-fitted J++ at L30–39. A lens-free third-person probe and E7's raw referent direction
  reproduce every sign, with smaller sizes for the quoted assistant "I" (user quoting the assistant: DI −1.00 for J++
  against −0.37 along the raw direction with a third party present, −0.31 against −0.10 without). R1 did not re-test this,
  and no causal patching was done at quoted-token states.
- *Evidence:* V5_binding_dissociation_NOTES.md:56–60, 80, 107–108, 173.

**Erratum E24** · line 228 · caution
- *Claim:* mid-layer J-space gender at a pronoun token encodes a default attribution and "does **not** encode the resolved
  referent the model uses to predict".
- *Corrected:* At mid layers the gender signal at a pronoun token, in the base-fitted lenses and lens-free probes alike,
  follows a default attribution (own-turn "I" → speaker, "you" → addressee, quoted "I" → the salient third party or the
  speaker). It does not match the resolved referent the model uses to predict, which shows up at prediction positions. "Does
  not encode what the model uses" is inferred from that mismatch between readout and behaviour: quoted-token states were not
  causally patched.
- *Evidence:* V5_binding_dissociation_NOTES.md:80, 159, 173; E7_binding_REPORT.md:195–198.

**Erratum E25** · line 229 · qualified
- *Claim:* late-layer lens gender in multi-party chats "is best read as the anticipated third-person referent".
- *Corrected:* In multi-party chats, late-layer (L40–58) gender readouts of the base-fitted lenses are dominated by the
  third party at every token, plausibly the anticipated third-person referent; V5's own wording is that they "may partly
  index" it. Not checked with an instruct-fitted lens.
- *Evidence:* V5_binding_dissociation_NOTES.md:148, 161, 169.

**Erratum E26** · line 233 · qualified
- *Claim:* general lesson: "a J-lens readout at a token tells you what the model's role-based prior attributes to that
  token. Reference resolution happens later and elsewhere, at prediction positions."
- *Corrected:* Lesson for this model: a mid-layer (L20–39) gender readout at a pronoun token, whether from the base-fitted
  J++/JLens or a lens-free residual probe, shows the role-based default the model attaches to that token, not the resolved
  referent. Reference resolution shows up at the prediction positions; at those inside the assistant turn even the L30–39
  J++ readout follows the quoted referent. Read "later" as later positions, not deeper layers. It was not tested for other
  lenses, models or concepts.
- *Evidence:* V5_binding_dissociation_NOTES.md:80, 118–128, 159.

### §2.2 Quiet carry channel (errata E27–E28)

**Erratum E27** · line 256 · qualified
- *Claim:* every covector "puts 99–100% of its energy in these dims at L35–58".
- *Corrected:* As erratum E7: the 99–100% share is for the lowest-variance 10% of coordinates (about 540 dims), not for the
  181-dim persistent set, which holds 56–95% at L30 and 80–98% at L40. "This is basis geometry, not a lens defect" stands.
- *Evidence:* V6_quiet_dims_NOTES.md:65, 104–112.

**Erratum E28** · line 263 · qualified
- *Claim:* "The lens's late-layer disagreement with the IT model lives inside the quiet subspace, the likely site of base→IT
  transfer error."
- *Corrected:* For the pronoun direction, masking the quiet dims raises J++ ↔ instruct-gradient alignment at L40–58 from
  0.26/0.33 to 0.84/0.86, so the raw-coordinate disagreement sits mostly in the quiet subspace, where the instruct model
  flips the sign of about 29% of the covector weight. The transfer error is not confined there (erratum E8): loud-coordinate
  agreement for random directions is only 0.62–0.73, and the loud coordinates carry much of an instruct-fitted lens's gain
  in predicting outputs. The rest of the bullet stands: masking does not fix L20–39, where the chat-context gradient is
  slightly anti-aligned with the lens at non-self assistant tokens (R1 reproduces −0.15 / −0.12, and ≈0 at self tokens).
- *Evidence:* V6_quiet_dims_NOTES.md:176–193; R1_IT_LENS_REPORT.md:36–43, 213–214, 225–243, 276–279, 433–442, 483–495.

### §2.3 Other internals (erratum E29)

**Erratum E29** · line 275 · caution
- *Claim:* "Latent Gemma self-identity" was refuted; J++ decodes " Gemma" at first-person tokens, but "every AI/tech token
  is boosted equally", it appears at the user's "I", J++ has a global " Gemma" bias at L42, JLens shows no self-vs-neutral
  difference, and patching never changes identity answers.
- *Corrected:* The refutation stands on lens-free patching and steering. The lens bullets describe the base-fitted J++ (and
  JLens, also base-fitted) and were not re-tested with an instruct-fitted lens. "Boosted equally" should read "boosted by a
  similar amount": self-over-background elevation is +3.3 to +6.6 nats across AI/tech tokens.
- *Evidence:* V4_latent_identity_NOTES.md:24–28; V8_stat_audit_NOTES.md:19.

### §3.1 Lens transfer and fidelity (errata E30–E39)

**Erratum E30** · line 291 · qualified
- *Claim:* absolute next-token fidelity on the IT model is poor: "prose KL 16–18 nats (J++), top-1 ≤ 3% through L49, 28% at
  L58. J++ ≈ logit lens at L45–58".
- *Corrected:* Absolute next-token fidelity of the base-fitted J++ on the instruct model is poor: prose KL 16–18 nats
  uncalibrated (about 10 after temperature calibration); top-1 at or below about 5% through L49 (5.1–5.2% at L45–46) and 28%
  at L58; J++ ≈ logit lens at L45–58. An instruct-fitted lens improves L40–58 but leaves prose poor (calibrated KL 7–8,
  top-1 ≤ 17% through L55).
- *Evidence:* E3_lens_validation_REPORT.md:10–13; E3_lens_validation_a_analysis.json (`per_layer`, `prose|top1|jpp`: 0.051
  at L45, 0.052 at L46); R1_IT_LENS_REPORT.md:69–70, 471–473.

**Erratum E31** · line 292 · caution
- *Claim:* on assistant-turn positions J++'s advantage over the logit lens shrinks from +9.1 nats (prose) to +3.1 at L30–39
  and turns negative at L40–49.
- *Corrected:* True of the base-fitted J++. An instruct-fitted lens restores an advantage over the logit lens on assistant
  tokens at L40–58 (directional r +0.12 / +0.16 over the logit lens, where J++ has −0.02), while fitting the instruct lens
  on chat rather than prose gains only 0.06 on assistant tokens, against an instruct-vs-J++ gap of +0.13 / +0.18 there. So
  at those layers the collapse is mainly J++ failing on the instruct weights, not the chat data. At L30–39 even the
  instruct-fitted lens gains little on assistant tokens (+0.05).
- *Evidence:* E3_lens_validation_REPORT.md:20–23, 109–112; R1_IT_LENS_REPORT.md:400–405, 424–432.

**Erratum E32** · line 293 · qualified
- *Claim:* "J++ ranks conditions correctly from ~L20 (r ≈ 0.9), 15–20 layers before the logit lens, but compresses the scale
  8–14×"; the bottom line of §3.1 glosses this as "early detection of decisions".
- *Corrected:* Condition means of the base-fitted J++ readout at L20–39 correlate with eventual behaviour (r ≈ 0.9 across 14
  conditions in E2 and 56 name × context means in E5; 0.99 across E1's 18 conditions), where the logit lens does not, though
  compressed 8–14× and with poor prompt-level fidelity (r ≤ 0.36). Persona-gender information is therefore linearly readable
  early. That does not locate where the model decides: at L20–39 the J++ pronoun direction at assistant positions is
  slightly anti-aligned with the instruct model's own average gradient. Under an instruct-fitted readout the ordering holds
  at L30–58. At L20–29 R1 computed its directional readouts for the same conditions but left them out of its report. Its
  lens is not stable there (split-half 0.63), so they can neither confirm nor refute the ordering (the mix fit
  puts the default 0.23 of the span above the neutral-name AIs there, against at most 0.09 for any directional fit at
  L30–58). Read "early detection of decisions" as "early condition-level ordering".
- *Evidence:* E2_behavior_REPORT.md:224–236; E5_name_REPORT.md:306–315; E1_self_readout_REPORT.md:178;
  E3_lens_validation_REPORT.md:31–37; R1_IT_LENS_REPORT.md:191, 213–214, 575, 584; R1_12_identity_analysis.json (`self`,
  `default-neutral3`, `it_mix_pron`: 0.232 at L20-29).

**Erratum E33** · line 294 · qualified
- *Claim:* alignment with the true IT average gradient (gender direction): cos 0.15 / 0.26 / 0.33, "0.84–0.86 at L40–58
  after quiet masking".
- *Corrected:* The numbers stand (R1 reproduces them), but they hold for the pronoun direction only. For 98 random
  token-pair directions, loud-coordinate agreement is 0.62–0.73. See errata E8 and E28.
- *Evidence:* R1_IT_LENS_REPORT.md:189–195, 225–243.

**Erratum E34** · line 295 · qualified
- *Claim:* background bias: layer-dependent gender offsets "come from the residual **mean** component. Centering removes
  them".
- *Corrected:* The base-fitted J++ readout has layer-dependent gender offsets (L10–19 "male", L40–49 pronoun-female but
  noun-male). They come mostly from the residual mean read through the lens, partly through quiet-coordinate DC terms whose
  sign depends on the lens fit (zeroing the quiet coordinates moves J++'s readout of the mean from +3.5 to +1.3, and JLens's
  at L50–58 from +0.9 to −4.3; the instruct Jacobian has the opposite sign on about 29% of the quiet-coordinate weight).
  Global centering removes most of them on neutral prose but leaves role-specific residue (self tokens ≈ −2.8 on the noun
  axis at L40–49); matched-control contrasts cancel them. They are not evidence of a gender bias in the model. R1 did not
  re-measure backgrounds.
- *Evidence:* E3_lens_validation_REPORT.md:215–241; V6_quiet_dims_NOTES.md:163–172; R1_IT_LENS_REPORT.md:276–279.

**Erratum E35** · line 296 · caution
- *Claim:* "J++ is very low-rank (top-100 of 5376 directions hold 68–93% of energy). Isotropic noise decodes to confident
  HTML closing tags".
- *Corrected:* This describes the base-fitted J++ matrix, not the model. R1 measures 76–97% of J++'s energy in its top-100
  output directions at L30–58, while the instruct model's own average Jacobian puts only 10–17% of its energy there;
  truncating the instruct lens to rank 256 destroys its late fidelity (top-1 at L58 0.85 → 0.28). The noise decodes go
  through the bare matrix, without J++'s published token filter (caveat under the table).
- *Evidence:* R1_IT_LENS_REPORT.md:45–48, 352–368.

**Erratum E36** · line 305 · qualified
- *Claim:* R1's instruct-fitted lens "does not help at L10–35, where no average-Jacobian lens is reliable".
- *Corrected:* Below L30 no average-Jacobian lens is stable (directional split-half ≤ 0.63). At L30–35 the directional
  instruct lens is stable and modestly improves directional fidelity on prose and user tokens, though not on assistant
  tokens (band L30–39: split-half 0.96, held-out 0.85; fidelity +0.11 on prose, +0.17 on user tokens, +0.04 on assistant
  tokens). The full-matrix version is noise-dominated there (noise share 68–80%), and neither version improves next-token
  fidelity below L40 (top-1 0.00 against J++'s 0.01 at L35). Two further limits: the 0.85–0.95 agreement is held-out
  reproducibility of the same object (the average same-position gradient) on E3's positions, not a test of readout accuracy;
  and the full matrix, which vocabulary readouts need, exists at only 11 layers.
- *Evidence:* R1_IT_LENS_REPORT.md:189–196, 335–345, 400–403, 456–465, 662–663.

**Erratum E37** · line 307 · caution
- *Claim:* the most likely cause of J++'s disagreement is the weight difference (base bf16 vs instruct W4A16-QAT, not
  separable), "after ruling out the fitting data and the method variants R1 tested".
- *Corrected:* As stated, plus: the base model was never run, and J++'s published recipe (R-Lens rules, cluster weighting,
  readout token filter) was not tested. Not every J++ failure is transfer: the late "every AI reads female" artifact is
  mostly a property of the J++ fit, since the base-fitted JLens removes it almost as well (−0.20 against −0.24), and JLens
  agrees better with the instruct gradients in loud coordinates (0.68–0.79) than J++ does (0.62–0.73).
- *Evidence:* R1_IT_LENS_REPORT.md:228, 232, 316–320, 548–549, 586–589, 607–615.

**Erratum E38** · line 307 · qualified
- *Claim:* "Gender conclusions are unchanged."
- *Corrected:* Under an instruct-fitted readout at L30–58, the default still sits with the nameless and neutral-name AIs,
  and the "Gemma" name shift is unchanged (R1 §5). Absolute positions are not stable: that readout puts default and nameless
  alike on the female side on the noun axis (0.56–0.68) and, at L50–58, on the pronoun axis (0.76), so "default ≈ human
  midpoint" does not carry over (errata E3, E18). R1 did not re-test binding or self-specificity, which rest on lens-free
  checks (errata E6, E16), and no average-Jacobian lens is stable below L30. The E4 attention/FFN attribution (§2.3)
  projects onto a raw-residual direction, uses no lens, and needs no re-test.
- *Evidence:* R1_IT_LENS_REPORT.md:510–601; E4_causal_REPORT.md §(d).

**Erratum E39** · line 311 · qualified
- *Claim:* "the J-lenses are useful comparative instruments (ordering conditions, early detection of decisions) and poor
  absolute ones", and any single-condition readout is uninterpretable without matched controls and background correction.
- *Corrected:* The base-fitted J-lenses (J++, JLens) are useful comparative instruments on the instruct model (ordering
  conditions, contrasts against matched controls) and poor absolute ones. An instruct-fitted lens repairs part of the
  late-layer absolute fidelity (L40–58) but not mid layers or prose. Under any lens, a single-condition readout is
  uninterpretable without matched controls and background correction: absolute positions moved by about 0.3 between
  instruct-lens fits while the contrasts did not. "Early detection of decisions" overreaches (erratum E32).
- *Evidence:* R1_IT_LENS_REPORT.md:52–70, 577–578; E2_behavior_REPORT.md:224–236.

### §3.2 Neuron poles (errata E40–E44)

**Erratum E40** · line 320 · caution
- *Claim:* J-VINDEX output label vs causal effect at L40–57: "0.35–0.69 (45–80% of reliability ceiling)".
- *Corrected:* J-VINDEX output labels (base-fitted J++): ≈0 at L10–30; P@10 0.35–0.67 at L40–57 (null ≤ 0.003); 0.37–0.73
  on a fresh sample, and higher at L40 with linear-regime steering (0.51 against 0.37). The 45–77% belongs to a different
  measure: over the full vocabulary the Spearman lift is 0.35–0.75, which is 45–77% of the split-half reliability ceiling of
  the causal measurement. The 0.69 is a JLens value. The plain logit lens does as well, so the validity does not come
  from the lens fit. Labels from an instruct-fitted lens were not tested.
- *Evidence:* N1_REPORT.md:28, 112–118; N1V_robustness.md:72, 84, 87.

**Erratum E41** · line 321 · caution
- *Claim:* J-VINDEX vs naive logit lens: equal at L10–30; "J++ +0.04 to +0.12 at L40–45; equal at L50–57; JLens ≥ J++".
- *Corrected:* Equal for top-10 labels at L10–30 (J++ carries a faint whole-vocabulary causal signal there, too weak for
  labels); +0.04 to +0.12 P@10 at L40–45; equal to slightly worse at L50–57 (−0.04 [−0.07, −0.01] in N1's sample, ≈0 in the
  fresh sample); JLens ≥ J++. All of this is for the base-fitted J++.
- *Evidence:* N1V_robustness.md:18, 87–91, 98–109.

**Erratum E42** · lines 323 and 369 · caution
- *Claim:* `gate_trigger_tokens` recovers "~6–8% of the reproducible signal at any layer; a direct weight·embedding readout
  wins at L10" (also bug 7).
- *Corrected:* As a top-10 token list it recovers about 6–8% of the reproducible input signal at any layer; its full ranking
  carries more (about 28% of the held-out ceiling overall, about 70% at L20–30). A direct weight·embedding readout wins at
  L10. Passing a read covector through J is a category error whichever lens is used.
- *Evidence:* N1_REPORT.md:44–50; N1V_robustness.md:168–171.

**Erratum E43** · line 326 · caution
- *Claim:* "Below L40 there is no stable single-neuron output effect to label (split-half reliability 0.00–0.07)."
- *Corrected:* At the layers sampled below L40 (L10, L20, L30) there is no context-general, magnitude-stable single-neuron
  output effect to label (split-half 0.00–0.07). At moderate steering (k = 10) the poles at L20–30 do have antipodal,
  context-specific effects; the "generic shared response" seen at 60× steering is a magnitude artifact.
- *Evidence:* N1V_robustness.md:19, 93–96, 215–216, 221–225.

**Erratum E44** · line 328 · qualified
- *Claim:* the L10–30 labels are lens prior: "random 512-neuron mixtures, and even isotropic noise, decode to equally
  'clean' clusters".
- *Corrected:* The tidy-looking J-VINDEX labels at L10–30 are a prior of the base-fitted J++ matrix: random 512-neuron
  mixtures decode to equally tight clusters. Isotropic noise also yields neat-looking clusters ("Elon, Tesla, aerospace,
  SpaceX"), but by the tightness metric they are less tight at L20–30 (0.054 / 0.040 against 0.090 / 0.087 for neurons), so
  "equally" holds for the mixtures only. The instruct model's own Jacobian lacks J++'s low-rank structure (R1 §3.2).
- *Evidence:* N1_REPORT.md:32–36, 184–191; N1V_robustness.md:197; R1_IT_LENS_REPORT.md:352–368.

### §3.3 The synthesis (errata E45–E47)

**Erratum E45** · line 342 · qualified
- *Claim:* "*J-lens as the output map* adds little over the logit lens, and only at L40–45."
- *Corrected:* The base-fitted J++ and JLens add little over the logit lens for neuron labels, and only at L40–45. An
  instruct-fitted lens does beat the logit lens on late-layer readouts (assistant-token top-1 at L50 0.45 against 0.32;
  directional r +0.12 / +0.16 over the logit lens at L40–58); whether it gives better neuron labels is untested.
- *Evidence:* N1V_robustness.md:98–109; R1_IT_LENS_REPORT.md:404–405, 463–464.

**Erratum E46** · line 351 · caution
- *Claim:* "The L30 picks are not gendered at all."
- *Corrected:* The L30 picks have only tiny effects (she−he ≤ 0.22 nats, against 13.8 for L50 n5313), measured with 60×
  steering, which is outside the linear regime at that depth. The L30 "she" pick n12173 moves she−he by +0.16 and +0.22
  (92nd and 100th percentile against random neurons), and its causal top-10 includes " feminine" and "girl".
- *Evidence:* N1A_gender.json (layer-30 poles, `c_she_minus_he`); N1V_robustness.md:49–55, 221–225.

**Erratum E47** · lines 353 and 380 · caution
- *Claim:* the synthesis earns its keep if labels are activation-grounded, restricted to L ≥ 40, gated by J++/naive
  agreement and causally spot-checked (also recommendation 5).
- *Corrected:* As stated, with two scopes. The J++/naive agreement gate (AUC 0.84 within L40–57) was validated only for
  base-fitted J++ labels and must be re-validated for any new lens. Causal spot-checks should use linear-regime steering (k
  ≈ 10, or a frozen-norm write), since 60× steering saturates the post-FFN norm at L20–50.
- *Evidence:* N1_REPORT.md:124–127; N1V_robustness.md:49–55, 111–115, 221–225.

### §3.4 Bug table (erratum E48)

**Erratum E48** · line 371 · qualified; severity under-rated
- *Claim:* bug 9, rated "Info": J++ fitted on the base bf16 model and applied to IT W4A16; evidence "Provenance block"; fix
  "Refit on IT, or report the transfer caveat in the UI".
- *Corrected:* Severity under-rated: High at least, since it touches every J++ readout the instrument makes and every J++
  or JLens reading in this report. JLens is also base-fitted. R1 gives the evidence the row lacks: J++ agrees with the
  instruct model's own average gradients at only 0.15–0.33; its assistant-token top-1 at L40–58 is 0.13–0.69 against
  0.21–0.85 for an instruct-fitted lens; and it carries a low-rank prior the instruct model lacks. Fix: label every readout
  with the lens it came from. For contrasts along stored directions, use R1's directional instruct lens at L30–58. For
  vocabulary readouts (plumes, top-k, neuron labels), the instruct full matrix exists only at layers 10, 15, …, 55, 58, and
  only 40/45/50/55/58 are usable (best at L45–58); other layers need a new fit. Keep J++ readouts flagged as base-model
  readings, and flag L10–29 for any average-Jacobian lens.
- *Evidence:* R1_IT_LENS_REPORT.md:45–48, 52–57, 189–195, 335–345, 620, 662–663.

### §3.5 Recommendations (errata E49–E50)

**Erratum E49** · line 378 · qualified
- *Claim:* recommendation 3: "Prefer L20–49 for lens comparisons; flag L50–58 pronoun readouts as unreliable."
- *Corrected:* With the base-fitted J++, use L30–49 for condition contrasts and confirm them with an instruct-fitted
  readout. At L20–29, trust only condition orderings that were validated against behaviour (r 0.88–0.99 in E1, E2 and E5);
  token-level and absolute readings there are unvalidated, since no average-Jacobian lens is stable. Flag L50–58 pronoun
  readouts unless an instruct-fitted readout agrees. With an instruct-fitted lens, L30–58 is the validated range for
  directional readouts (full matrix: only at L40/45/50/55/58).
- *Evidence:* E1_self_readout_REPORT.md:178; E2_behavior_REPORT.md:228; E5_name_REPORT.md:310–313;
  R1_IT_LENS_REPORT.md:65–68, 189–193, 335–345.

**Erratum E50** · line 381 · qualified
- *Claim:* recommendation 6: "Consider refitting J++ on the IT checkpoint".
- *Corrected:* R1 has refitted a Jacobian lens on the instruct checkpoint: a directional lens for 128 fixed directions,
  stable at L30–58 (`R1_04_gbar.pt`), and a full matrix at 11 layers, usable at five of them from L40 to L58 (`R1_03` +
  `R1_09`). The instrument does not load it yet, and its files are gitignored, so they are not in the repository. It is a
  plain same-position Jacobian, not the J++ recipe: fine-tuning and quantisation cannot be separated, and J++'s published
  R-Lens rules, cluster weighting and token filter were not tested. Drop the J++ low-rank prior, keep the L10–29 caveat for
  every average-Jacobian lens, and require J++ and the instruct readout to agree before showing L50–58 pronoun plumes.
- *Evidence:* R1_IT_LENS_REPORT.md:101–107, 607–615, 659–668; the repository's `.gitignore`:15.

### §4 Claims that did not survive verification (erratum E51)

**Erratum E51** · line 390 · caution
- *Claim:* "J-space gender is bound to the *referent* (E7)": refuted as reference, confirmed as a role default.
- *Corrected:* The verdict stands, scoped: the role default is seen in the base-fitted J++/JLens readouts and confirmed by
  lens-free probes and raw-residual referent directions, with the model's own next-token behaviour inside quotes following
  the true referent. It was not re-tested with an instruct-fitted lens.
- *Evidence:* V5_binding_dissociation_NOTES.md:54–92, 159.

## References

See [`REFERENCES.md`](../REFERENCES.md) (41 entries: model, lenses and lens methods, interpretability methods, related
work, statistics, data and software).
