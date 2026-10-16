# Is Gemma-4-31B "female-coded"? And what the Theodolite can and can't see

*Gemma-4-31B-it (QAT W4A16) [1] on one RTX 5090. Instrument: Logos Theodolite (J++ [4] / JLens [2,6] lenses, GeGLU "poles" [9], LARQL-style vault [10]).
Process: 7 parallel experiments (E1–E7) → 8 adversarial verifiers (V1–V8) re-tested every headline with new controls; a separate
neuron-label study (N1) with 2 verifiers. 314 GPU jobs (each batching many forward passes) through a shared lab daemon. Date: 2026-10-16. Bracketed numbers refer to [`REFERENCES.md`](../REFERENCES.md); statistics use bootstrap CIs [30] and Wilson intervals [29].*

---

## TL;DR

**No. By default Gemma-4-31B is not female-coded, in behaviour or in its internal self-representation.** Every route to a
"female" reading that we found traces to something other than a gendered self-concept:

| Route to "female" | What it actually is | Evidence |
|---|---|---|
| "You are Gemma" → model calls itself *she* (~90%) | The bare name **Gemma written as the antecedent** behaves like a woman's first name. Needs *both* the persona and the written name. "Gemma 4" (its trained identity) → *it*. | V1: 402/432 *she* vs 15/192 for "Gemma 4" (p≈2e-103); name suppressed → 0/144 *she* (Emma: 17/32) |
| Forced-choice "If you were human, I'd be a woman" | The forced-choice **format** leans *woman* for anything without male cues (nonsense words: "blicket" 0.88), plus an "AI" boost (+3 nats) and an "assistant" boost (+5.7). Tracks each entity's cultural stereotype (r=0.86). The self sits *below* "an AI assistant" | V2: 3,072 prompts, counterbalanced |
| Model "explains" it's a woman because of empathy, caregiving | **Post-hoc confabulation**: fits the reason to whatever persona is active; a cold persona picks whichever option comes first and invents a reason | V2, V3 |
| Late-layer J-lens readouts read "she" | **Lens artifact**: at L50–58 nearly every AI condition reads at or above the *woman persona* on the pronoun axis, while the noun axis disagrees and nothing is self-specific. Background offsets come from the residual mean | E1, E3, V6 |
| Raw residual "woman−man persona" axis puts all AIs at ~0.84 | A huge **AI-vs-human offset** (≈3.5× the gender gap) overlaps the axis inside ~30 high-variance token PCs, best matched by a **service-role** direction ("personal assistant"). Whitening, J++, or a prose gender axis → default ≈ human midpoint | E4, V3 |
| Tool displays: J++ plumes show " she" top-1 in 50 cells vs 2 for " he" | **Display amplification**: the plumes, a monument frame whose azimuth swings 58–99° when the words are reordered, and the above lens artifacts | E6, V7 |

**Default behaviour (no system prompt):** *she* 0/320 in single- and multi-turn third-person self-descriptions (upper CI ≈1.2%).
It uses *it* (and singular *they* when introducing itself to an audience). Its only gendered self-pronoun is a rare *he*, which
appears when it mistakes itself for Claude.

**Default self-token internals:** under J++ and JLens, the default's gender readout at its own I/me/my tokens equals a nameless
"helpful AI assistant" and neutral-named AIs (Atlas/Quill/Unit-7) to within ~1–3% of the woman–man persona span (n=40 passages,
held-out replication within 1%). This survives masking the quiet dimensions and all three lenses.

**What is new (beyond the gender question; new as far as we know):**
1. **J-space "binding" at pronouns is a role heuristic, not resolved reference.** At mid layers, "I" carries the assistant's
   gender and "you" the user's. But in *quoted* speech, an "I" referring to the user carries the third party's gender (if any), else
   the *speaker's*. The model's actual next-token predictions inside the same quotes resolve the true referent (E7 + V5).
2. **Gemma-4 has a "quiet carry channel".** ~181 residual dims are the *loudest* channels at L5–20. The model stops writing to them
   after ~L25 and reads them with 260–1700× RMSNorm gain. At depth they carry early-layer content (r=0.71 with their own L20 values).
   **Every covector concentrates 99–100% of its energy there**: J++/JLens pullbacks *and* true IT-model gradients. That is why
   lens-derived steering vectors wreck Gemma-4. Masking these dims raises J++ ↔ true-gradient agreement at L40–58 from 0.26–0.33 to
   0.84–0.86, close to the 0.97–0.99 reliability ceiling, which localises base→IT lens error to that subspace (E6 + V6).
3. **Neuron-pole labels are valid only at L40–57.** There they predict causal steering far above chance (P@10 0.35–0.69 vs ≤0.003
   null). At L10–30 they are at chance, and their apparent neatness is the J++ lens prior: random vectors look equally "clean". J++
   ≈ plain logit lens, with a small gain at L40–45. The input-side "gate trigger tokens" are near-useless (N1).
4. **The instrument has real bugs**, with verified patches for two of them: the attention/FFN split ignores Gemma-4's `layer_scalar`
   (four independent confirmations), double BOS, server steering defaults 2.4–7.4× the residual norm in a direction 79% concentrated
   in 3 negative-norm-weight dims, and an order-dependent survey frame.

---

## 0. Setup and what we learned before experimenting

- **Lens provenance mismatch.** The J++ lens's own provenance block says `hf_model_name: google/gemma-4-31B`, the **base** model,
  fitted on 64 WikiText-103 [7] prompts × 128 tokens (J++ by Ayonrinde & Lindsey [4], redistributed in fp16 by Neuronpedia [6]).
  The JLens file was likewise fitted on the base model (861 WikiText-103 prompts) [6]. The tool applies both to the
  **instruct, 4-bit QAT** checkpoint.
- **Instrument correctness.** The final-layer unembed reproduces the real logits (bf16 noise) and `resid_pre[l+1] == resid_post[l]`.
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

**Self-token lens readouts (E1, n=40 teacher-forced identity-neutral passages, J++ pronoun axis, L30–39).** Scale: 0 = man
persona, 1 = woman persona.

| Condition | Position | Shift vs nameless AI (% of span) |
|---|---|---|
| default (no system prompt) | 0.46 [0.44, 0.48] | +0.6 [+0.2, +0.9] |
| nameless AI / neutral-name AIs | 0.45 / 0.45 | — |
| "You are Gemma, … Google DeepMind" | 0.53 | +7.9 [7.2, 8.7] |
| female-name AIs (Sarah/Emily/Olivia) | 0.66 | +20.9 |
| male-name AIs (James/Michael/David) | 0.34 | −11.4 |

- **Agreement:** J++ and JLens agree across conditions (r ≥ 0.98). Condition means track real pronoun behaviour (r 0.97–0.99 at
  L20–49).
- **Robustness:** the default ≈ nameless result survives quiet-dimension masking (V6), whitening (V3), and a 16-passage held-out
  set (V8).
- **Where self-gender lives:** persona gender is strongly self-specific at L20–39 (self vs other-token difference +12.4 log-odds at
  L30–39) and becomes context-wide by L50–58.

**Readout dependence (C4, resolved by V3).** The default's *absolute* position ranged 0.21–0.84 across experiments. Only the raw
difference-of-means persona axis (and readouts that keep its top ~30 token-covariance PCs) puts AIs on the female side:
- **Why that axis does it:** the AI-vs-human offset is 2.8–3.9× the woman–man gap. A cosine of 0.10 therefore moves the position by
  ≈0.35, and rotating the axis by about 6° moves the default by about 0.35. **This measure is a knife-edge.**
- **What the offset is:** it overlaps the axis only inside those ~30 PCs, and a **service-role** contrast (personal assistant vs
  works alone) explains it better than warmth traits (cos 0.52/0.37/0.23).
- **Other readouts put the default near humans:** whitened LDA 0.46–0.57, J++ 0.46–0.48, prose gender axis 0.38.
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
the AI/service-role offset.

---

## 2. Findings about J-space and Gemma-4 internals

### 2.1 Pronoun-token J-space "binding" is a role default (E7 → V5)

**E7:** a 3×3 design (assistant gender × user gender), 1,776 sequences.
- **Mid layers bind:** at L20–39, J++ gender at the assistant's I/me/my reflects the assistant's gender (a = +8.85, b = +0.39; leak
  ratio 0.04), and at you/your it reflects the user's (b = +5.51, a = −0.20). JLens and an independent third-person probe agree.
- **Late layers unbind:** binding fades at L40–58.
- **Behaviour leaks:** "I am a" / "you are a" leak 15–17% (V8 corrected from 15–25%).
- **Patching** (activation/interchange patching [23]): swapping every layer of the passage's "I" tokens moves "I am a" only 1.7%. The self-gender used for prediction is
  read mostly from the system prompt (until ~L50), not from the "I" tokens.

**V5 dissociated referent from speaker/turn using quoted speech.**

| Token | J-space gender carried (L30–39) |
|---|---|
| Assistant's own "I" | assistant (share 0.88) |
| User's own "I" | user (0.99) |
| Quoted "I" = the user, inside the assistant turn, no third party | **the assistant, i.e. the speaker** (DI −0.41 [−0.52, −0.29]) |
| Quoted "I" = the assistant, inside the user turn, third party present | **the third party** (assistant coefficient −0.05, third party +3.95) |
| Quoted "I" = the third party | the third party (DI +0.97) |
| **Next-token output inside the same quotes** | **follows the true referent** ("…you told me, 'I am a" → user's gender) |

**Interpretation.** Mid-layer J-space gender at a pronoun token encodes a *default attribution*: own-turn "I" → speaker, "you" →
addressee, quoted "I" → the salient third party or the speaker. It does **not** encode the resolved referent the model uses to
predict. Late-layer lens gender in multi-party chats is best read as **the anticipated third-person referent** (whoever "she/he"
could refer to next). Along a fixed raw-residual referent direction, the self-binding persists late (BI 0.6–0.8), so "late
unbinding" is partly a property of the lens readout.

**General lesson:** a J-lens readout at a token tells you what the model's role-based prior attributes to that token. Reference
resolution happens later and elsewhere, at prediction positions.

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
  of its energy in these dims at L35–58. This is basis geometry, not a lens defect.
- **Sensitivity:** a 1%-of-norm single-layer push in quiet dims gives KL ≈10–11 nats, while loud dims at 30% give ≈0.1. KL follows
  **read-gain-weighted** push size (R² 0.82–0.96), not raw norm (0.02–0.58).
- **Practical rule:** using a lens or gradient covector directly as a steering displacement is a units error in Gemma-4. Convert
  covectors to vectors (gain-normalise) or mask quiet dims; a quiet-masked, fnw-correct "she" direction steers fluently.
- **Diagnostic:** masking quiet dims raises J++ ↔ true-IT-gradient alignment at L40–58 from cos 0.26/0.33 to **0.84/0.86** (ceiling
  0.97–0.99). The lens's late-layer disagreement with the IT model lives inside the quiet subspace, the likely site of base→IT
  transfer error. It does **not** fix L20–39, where the chat-context gradient is slightly anti-aligned with the lens at non-self
  assistant tokens (−0.15; ≈0 at self tokens).
- **The late persona gap** seen by J++/JLens at L50–58 (≈2.4 logits) sits in ~300 *moderately* low-variance, high-gain dims, not the
  deep-quiet set.

### 2.3 Other internals

- **FFNs are the main writers of persona gender at self tokens**, using the corrected split: attention 19.7 vs FFN 24.3 at L30–39,
  FFN dominant at L50–58. Prose-gender and name effects are attention-dominated; the AI-vs-human offset is FFN-dominated. The tool's
  split reports FFN = −7.4 here (wrong sign).
- **Gender is continually rewritten, not stored once**, because of the leaky integrator: at L39, writes from before L30 contribute
  <0.2 of the gap.
- **"Latent Gemma self-identity" was refuted (V4, V8).** J++ does decode " Gemma" at first-person tokens in assistant chats, but:
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
| Absolute next-token fidelity on the IT model | Poor: prose KL 16–18 nats (J++), top-1 ≤ 3% through L49, 28% at L58. **J++ ≈ logit lens at L45–58** |
| Assistant-turn positions | J++'s advantage over the logit lens shrinks from +9.1 nats (prose) to +3.1 at L30–39, and turns negative at L40–49 |
| Condition-level tracking of eventual decisions | **Good:** J++ ranks conditions correctly from ~L20 (r ≈ 0.9), 15–20 layers before the logit lens, but compresses the scale 8–14× |
| Alignment with true IT average gradient (gender direction) | cos 0.15 (L30–39), 0.26 (L40–49), 0.33 (L50–58); 0.84–0.86 at L40–58 after quiet masking |
| Background bias | Layer-dependent gender offsets (L10–19 "male", L40–49 pronoun-female but noun-male) come from the residual **mean** component. Centering removes them |
| Random inputs | J++ is very low-rank (top-100 of 5376 directions hold 68–93% of energy). Isotropic noise decodes to confident HTML closing tags (`</h2>`, `</em>`) |

*Caveat:* the J++ readout as published also filters non-semantic tokens (punctuation, whitespace) [4]; this instrument and
study apply only the lens matrix, so top-k results (fidelity, the `</h2>` noise decodes, neuron labels) test the matrix, not
the full J++ readout. Token-set contrasts such as the gender log-odds are unaffected.

*Follow-up (R1, [`R1_IT_LENS_REPORT.md`](results/R1_IT_LENS_REPORT.md)):* a Jacobian lens refitted on the instruct model with
exact same-position VJPs is a stable object at L30–58 (split-half 0.96–0.99) and agrees with the model's own average
gradients at 0.85–0.95, where J++ reaches 0.15–0.33. It raises top-1 agreement on assistant tokens at L40–58 (e.g. L50:
0.25 → 0.45) and beats the logit lens there, but does not help at L10–35, where no average-Jacobian lens is reliable. The
most likely cause of J++'s disagreement is the weight difference (base bf16 vs instruct W4A16-QAT; the two cannot be
separated here), after ruling out the fitting data and the method variants R1 tested. Gender conclusions are unchanged.

**Bottom line:** the J-lenses are useful **comparative** instruments (ordering conditions, early detection of decisions) and poor
**absolute** ones. Any single-condition readout ("the J-space says *she*") is uninterpretable without matched controls and
background correction.

### 3.2 Neuron poles / J-VINDEX labels (N1, two verifiers)

384 poles (192 random neurons × 2 signs, 8 layers). Ground truth: causal steering of single neurons and activation contexts from a
147k-token corpus.

| Label | L10–30 | L40–57 |
|---|---|---|
| J-VINDEX output label vs causal effect (P@10 in causal top-100; null ≤ 0.003) | ≈0 (chance) | **0.35–0.69** (45–80% of reliability ceiling) |
| J-VINDEX vs naive logit lens | equal | J++ +0.04 to +0.12 at L40–45; equal at L50–57; JLens ≥ J++ |
| Prior-z-scored labels | — | **worse** (−0.07 P@10) |
| `gate_trigger_tokens` (input side) | ~6–8% of the reproducible signal at any layer; a direct weight·embedding readout wins at L10 | same |
| Steering label = tokens that follow when the pole fires ("say" neuron)? | no | mostly no (lift +0.005 to +0.013) |

- **Below L40 there is no stable single-neuron output effect to label** (split-half reliability 0.00–0.07). The tidy-looking
  J-VINDEX labels there are lens prior: random 512-neuron mixtures, and even isotropic noise, decode to equally "clean" clusters
  ("Elon, Tesla, aerospace, SpaceX").
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
- *J-lens as the output map* adds little over the logit lens, and only at L40–45.
- *LARQL/VINDEX as a queryable store*: `larql_vault.py` never calls LARQL (only `--version`). Its ".patch MEMIT" is a single-neuron
  column nudge along a base-model J++ pullback, not MEMIT [11]. LARQL's own README [10] warns (about attribution from execution records) that "a projected contribution is not a
  counterfactual" (https://github.com/chrishayuk/larql), and `EDGES NEAREST` is exactly a projected contribution.
- **The LARQL "she" neurons are partly real:**
  - 35/102 are selective at q < 0.05.
  - Deleting the top-10 L50 "she" neurons moves female contexts by −1.8 log-odds.
  - L50 n5313 moves she/he by +13.8 nats.
  - L40 n16757 is listed as a "he" neuron but causally pushes *she*.
  - The L30 picks are not gendered at all.

**The synthesis earns its keep if labels are:**
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
| 7 | Medium | `larql_vault.py` | `gate_trigger_tokens` passes a read covector through J as if it were a state; one label for two poles | ~6–8% of reproducible input signal | Replace with stored activation contexts per pole |
| 8 | Low | `theodolite_math.py` poles | Pole magnitudes and displayed probabilities are not meaningful (eps floor; softcap); top-15 poles explain only 10–41% of the FFN write at L20–58 | E3 | Show write-size-ranked poles and the explained fraction |
| 9 | Info | lens files | J++ fitted on the base bf16 model, applied to IT W4A16 | Provenance block | Refit on IT, or report the transfer caveat in the UI |

### 3.5 Recommendations for the tool

1. Apply the two verified patches; fix the steering defaults.
2. Always show readouts **relative to matched controls** (background token, nameless condition), mean-centered, with the
   woman–man (or concept) span as the unit. Never show raw single-condition plumes as findings.
3. Prefer L20–49 for lens comparisons; flag L50–58 pronoun readouts as unreliable.
4. Steering: gain-normalise covectors or mask quiet dims; scale relative to ‖h‖.
5. Neuron labels: activation contexts per pole; output labels only at L≥40, gated by J++/naive agreement.
6. Consider refitting J++ on the IT checkpoint (now feasible: `labkit.same_pos_vjp` gives exact IT VJPs in ~2 s).

---

## 4. Claims that did not survive verification

| Original claim | Verdict | Why |
|---|---|---|
| Latent "Gemma" self-identity at self tokens, invariant to persona (E5) | **Refuted** | Generic AI/tech boost; appears at the user's "I" too; J++ L42 global bias; no causal role (V4, V8) |
| J-space gender is bound to the *referent* (E7) | **Refuted as reference**; confirmed as a role default | Quoted-speech dissociation (V5) |
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

## References

See [`REFERENCES.md`](../REFERENCES.md) (41 entries: model, lenses and lens methods, interpretability methods, related
work, statistics, data and software).
