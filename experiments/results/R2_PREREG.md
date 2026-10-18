# R2: the second reading with formatting filtered out (pre-registration)

*Written 17 October 2026, before any R2 data existed, and amended the same day after two reviews, still before any
R2 data existed (§11). No R2 forward pass, generation or lens readout has been run. R2 follows R1, whose
instruct-fitted lens is one of the two lenses tested here. Scripts: `experiments/scripts/R2_*.py`.
Prompts: `experiments/scripts/R2_prompts.json`. Results: `experiments/results/R2_*`.*

**Integrity.** The GPU script (`R2_02_eval.py`) records the sha256 of every pre-registered file in its output when
it runs: this file, the prompt file, `R2_common.py`, `R2_01_check.py`, `R2_02_eval.py`, `R2_03_analyze.py` (which
implements the decision rule) and `R2_01_check.json`. The analysis (`R2_03_analyze.py`) compares them with the files
on disk and reports any difference.

These files are committed before the GPU run, so that the commit also shows they came first. `R2_02_eval.py` refuses
to run while any of them is untracked or differs from HEAD, and it records HEAD in its output. It also refuses to
overwrite an existing `R2_02_eval.json` without `--force`, and it appends every run (started, finished or failed) to
`R2_02_runs.jsonl`. Every run is reported in the write-up.

Any change made after data exists is a deviation. It is reported as one, and the result under the rule written here
is reported beside it.

## 1. Question

The instrument reads `google/gemma-4-31B-it-qat-w4a16-ct` through J++, which was fitted on the base model. Beside J++
it can show a "second reading" through two lenses fitted on the instruct model:

- **R1's lens** (`R1_18_it_lens.pt`, sha256 `80ce5609…e107b`) at code layers 50, 55 and 58.
- **The Miru community J-lens** (`returnmoe/jlens-adapters@134a973c`, `miru/google/gemma-4-31B-it.safetensors`, sha256
  `cedd4b62…3386a`) at code layers 40–58.

The first held-out test had two findings:

- These lenses put the formatting token `" **"` first at 25–74% of positions at layers 40–57.
- Once formatting tokens were set aside, their top word matched the model's next word about as often as J++'s and the
  naked eye's. The one exception was R1 at 58.

The owner decided to filter formatting tokens out of the second reading, re-test on fresh prompts, and then decide
which layers to serve. R2 asks, at each layer band:

> **Does the filtered second reading name the model's next content word more often than filtered J++ and the
> filtered naked eye, on prompts no one has looked at?**

"More often" here means a clear and consistent win: by at least 5 points pooled, over most of the prompts where the
readouts differ, and not significantly worse in any condition (§7).

Inside the model's own replies (mid-reply), the read positions are drawn from anywhere in the reply, including
positions where the model is about to write formatting. That is where the filter matters, and it is what the
instrument shows (§4.3).

**The earlier 100-prompt set is not used.** That is the scratchpad `heldout_eval.py` and its output. Its prompts have
been seen, and its numbers motivated this test. None of its prompts is reused (§3.2).

## 2. The filter

`R2_common.classify_text` / `content_mask` apply this definition to every vocabulary id `i`. Let:

- `t = tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)`
- `s = t.strip()`
- `w` = `t` with every whitespace character removed (`str.isspace`)

The token is **NON-CONTENT** if any of these holds. They are checked in this order; the first that holds names the class.

| | class | rule |
|---|---|---|
| (a) | special | It is a special or control token: its id is in `tokenizer.all_special_ids` or is an added token flagged special, or `s` has one of the shapes `<\|...\|>`, `<...\|>`, `<\|...>` (no whitespace or angle brackets inside). |
| (b) | blank | `w` is empty. |
| (c) | punct_symbol | Every character of `w` is punctuation or a symbol (Unicode general category P\* or S\*) or invisible: a control, format, private-use or unassigned character (Cc, Cf, Co, Cn, e.g. `\x00`, U+200B, U+FEFF, the joiner U+200D), a variation selector (U+FE00–U+FE0F, U+E0100–U+E01EF) or the keycap mark U+20E3. *(Amendment A9: the invisible characters were added.)* |
| (d) | html_tag | `s` is one whole HTML/XML tag (`<strong>`, `</li>`, `<br/>`, `<td>`, …): `<` or `</`, a letter, then letters, digits, `_ . : -`, optional attributes after whitespace, an optional `/`, then `>`. |

Everything else is **CONTENT**: words, word-pieces, numbers, any script, and mixed tokens such as `1.` or `(the`.

Gemma's tokenizer splits punctuation and digits from words. No CONTENT token mixes markdown (`**`, `__`, `#`, `` ` ``)
with letters, so formatting cannot reach the filtered reading as a token like `**The`.

The mask is built on the CPU from the checkpoint's tokenizer in the local Hugging Face cache (offline). It has:

| | tokens |
|---|---|
| vocabulary | 262,144 |
| CONTENT | 247,973 |
| NON-CONTENT | 14,171 |
| — special | 24 |
| — blank | 104 |
| — punct_symbol | 7,746 |
| — html_tag | 6,297 (6,227 of them `<unusedN>` placeholders) |

Amendment A9 moved 482 tokens from CONTENT to NON-CONTENT (punct_symbol):
- 433 made only of invisible characters, such as `\x00`, `" ​"` and `" ﻿"`;
- 29 of punctuation or symbols with invisible characters, such as `".​"` and `"⁢</"`;
- 20 symbols carrying a variation selector or keycap, such as `❤️`, `✔️`, `➡️` and `⚠️`. These are chat list bullets
  that instruct lenses may favour.

**Spot checks** (`R2_01_check.py`, all pass). Each string is checked as text, and at every id whose decoded text is
exactly that string.
- NON-CONTENT: `" **"`, `"**"`, `"<strong>"` (id 198), `"</h2>"` (id 192), `"\n"`, `" "`, `"<|channel>"`, `"<channel|>"`,
  `"\x00"`, `" ​"`, `"✔️"`.
- CONTENT: `" Paris"`, `" Ottawa"`, `" the"`, `"The"`, `"8"`, `" kind"`, `" same"`.
- Note: `"<strong>"` and `"</h2>"` are single vocabulary tokens (198, 192). Ordinary text encodes them as pieces
  (`<`, `strong`, `>`), and the piece `strong` is CONTENT.

**Edges, reported and left as the definition has them:**
- 601 CONTENT tokens are made only of marks (category M) or control characters with whitespace. Most are combining
  marks of scripts such as Devanagari.
- `[multimodal]` (id 5) is CONTENT.
- The 129 lone UTF-8 bytes 0x80–0xFF decode to U+FFFD and are NON-CONTENT (symbol).
- Emoji are NON-CONTENT (category So).

**What a filtered readout is.** The top-k over CONTENT ids only, that is, the logits with NON-CONTENT ids set to −∞.
This is what the instrument would show.

## 3. Prompts

### 3.1 Conditions

220 prompts are written for R2. All are distinct, and topics are varied. Each condition includes prompts about the
model itself (8, 9, 4 and 6 respectively; 27 in all).

| condition | n | the text read | read position |
|---|---|---|---|
| chat-no-answer | 60 | The question in the chat template, with no system prompt and thinking off (empty thought channel). | Its last token, `<channel|>`. |
| chat-begun | 60 | The question, then a begun answer that ends just before a content word. | The begun answer's last token. |
| bare | 60 | A plain-text opening with BOS and no template. | Its last token. |
| mid-reply | 40 | A new question. The model's greedy reply (thinking off, 48 new tokens) is generated in the run. | Seeded positions anywhere inside that reply (§4.3). |

**The chat-begun answers are deliberately mixed:**
- 16 factual, 10 opinion or advice, 8 creative, 8 about the model, 5 how-to or code, 3 empathy, 3 language or arithmetic, and 7 list openings.
- 13 are marked `format_likely`. These end where the model is likely to write formatting (`**`, a list marker) before
  its next word, for example `"Here are three tips:\n\n1."`.

### 3.2 Freshness check (`R2_01_check.py`, result in `R2_01_check.json`)

Every question, begun answer, question + begun answer, and bare text was compared, after NFKC normalisation, case
folding and removing punctuation, against these sources:

- The QUESTIONS, BEGUN and BARE lists of the earlier `heldout_eval.py`.
- Every string literal in `experiments/scripts/*.py`, which includes every `*_common.py` prompt list (377 files).
- Every string in `experiments/results/*.json` (770 files). This covers the study's stimuli, generated chats and corpora,
  among them R1's fit corpus `R1_01_corpus_part*.json`.
- The N1B corpus (`N1B_corpus.jsonl`, decoded from its token ids).
- The other R2 prompts.

A prompt is flagged if any of these holds:
- **Exact.** It matches a source string exactly.
- **Contained.** It has 4 or more words and appears inside a source string.
- **Contains.** A source string of 5 or more words appears inside it.
- **Similar.** Its difflib ratio is 0.80 or more, or its word Jaccard is 0.75 or more.

The first draft had 43 flags on 18 prompts. Several common chat questions turned out to be in the study's lists
already. Those prompts were rewritten before any data existed. The next pass also decoded the N1B corpus and flagged
one rewritten begun answer, which was rewritten again. The final set has 0 flags against 33,478 source strings, and
the closest difflib ratio is 0.78.

**What the check does not cover.**
- *Miru's fit corpus.* The file's header records 425 WikiText-103 passages, but not which ones. WikiText-103 is not
  available locally, so the prompts were not checked against it. The 60 bare texts are prose of the same kind. A
  verbatim overlap with 425 passages is unlikely, but it was not checked.
- *Paraphrase and shared topics with R1's fit set.* The lexical check passes, but some prompts paraphrase one of R1's
  fit questions (`R1_common.R1_SELF_Q`, `R1_TASK_Q`) or have an answer that is the subject of one of them or of an R1
  encyclopedia topic (`R1_ENC_TOPICS`). Examples: cb-53 "Give me three tips for a job interview." against "How should I
  prepare for a job interview?"; bare-01, whose answer is the Amazon, an R1 encyclopedia topic.
  - The list `R2_common.R1_OVERLAP` was fixed from the prompt texts alone, before any data. It names 22 prompts, each
    with the R1 prompt it overlaps.
  - Sensitivity analysis S2 (§8) reruns the rule without them, and S1 without the 27 prompts about the model (a third
    of R1's fit chats were self-questions).
  - The risk is limited: R1's fit (`R1_03_fit.py`) estimates E[J] from vector-Jacobian products with random probes; it
    does not regress on next tokens.

## 4. Reading

### 4.1 Model and runtime

**Loading.** `R2_02_eval.py` loads `TheodoliteRuntime` as `labd.py` does, without the JLens and without the
server's own second lenses, and wraps it in `labkit.Lab`.

**Lens files.** It loads the two lens files itself, checking each file's size first and then its sha256:
- R1: from the given path, checked against `80ce5609…e107b` and 173,411,927 bytes.
- Miru: read only through `safetensors.safe_open`, layers 40–58, checked against `cedd4b62…3386a` and
  3,410,446,944 bytes.

A mismatch stops the run. The server is not used.

### 4.2 Model's own words, and decoding

**The model's own word** at a position is the argmax of the model's final readout of `resid_post[59]`: the final
RMSNorm, W_U and the 30.0 softcap (`unembed_gemma4`). This is how the instrument names the model's own word.

**Greedy continuations** use the same rule, step by step, through full forwards with no KV cache
(`lab.forward_resid`, right-padded batches). A sequence stops after a stop id (`<eos>`, `<turn|>`,
`<tool_response|>`).

### 4.3 Read positions

**chat-no-answer, chat-begun, bare.** One position per prompt: the last prompt token.

**mid-reply.** For each question, the greedy reply `r[0..n-1]` is generated (at most 48 tokens; a stop id ends it and
is not part of it). Reading at reply index k reads the state at `r[k]`, whose next token is `r[k+1]`, so k runs over
0..n−2. There are two seeded draws without replacement. Each uses its own stream,
`numpy.random.default_rng([20261018, prompt index, draw])`, where the prompt index is the question's 0-based place
in the mid-reply list and the draw is 0 for "content_next" and 1 for "any".

- **"any"** draws up to 6 positions among all k. These make up the **mid-reply** decision condition. *(Amendment A5:
  this was 3 positions, reported only.)*
- **"content_next"** draws up to 3 positions among the k whose next token `r[k+1]` is CONTENT. These make up
  **mid-reply-content-next**, which is reported but not used for the decision. *(Amendment A5: this was the decision
  condition.)*

A position drawn twice is read once and carries both labels. A mid-reply prompt's score is the mean over its drawn
positions that have a usable content target (§5).

### 4.4 Readouts

At every read position, `resid_post` is captured at layers 40–59 (batched `lab.forward_resid`). Every readout uses
the instrument's own functions with the runtime's bf16 final-norm weight and W_U:

| readout | layers (code) | path |
|---|---|---|
| J++ | 40–58 | `jpp_readout_logits(h, J_l)` with the runtime's J++ matrices (`h_J = h @ J.T`) |
| naked eye (logit lens) | 40–59 | `unembed_gemma4(h)`; at 59 it is the model's own output |
| R1 | 50, 55, 58 | `jpp_readout_logits(h, J_l)`, R1_18 matrices in bf16 |
| Miru | 40–58 | `jpp_readout_logits(h, J_l)`, the file's fp16 matrices cast to bf16 by `validate_and_cast_lens` |

Each readout is scored both unfiltered (top-k over the full vocabulary) and filtered (top-k over CONTENT ids). The
raw output keeps, for every position, layer and readout, the top-5 ids and strings of each, plus the rank of the
targets.

## 5. Targets

For each read position there are two targets:

- **(i) Next token.** The model's next token. For the first three conditions, this is the first token of its greedy
  continuation. For mid-reply, it is `r[k+1]`.
- **(ii) Content target.** The model's next content token: the first CONTENT token among the model's own greedy
  continuation of up to 8 tokens. For mid-reply, it is the first CONTENT token among `r[k+1..k+8]`, within the reply.
  - The search stops, with no target, at a stop id or at any special token. *(Amendment A1: it stopped only at stop
    ids; after `<|channel>` the template word "thought" would have become a "content word".)*

**No usable target.** A position has no usable content target in two cases:
- its window has no content target;
- its content target is a raw-text collapse word: `" own"` or `" same"`, matched by word (`R2_common.COLLAPSE_TARGETS`).
  *(Amendment A1.)* On raw text this instruct model often emits these as junk: in the seen test they were the model's
  word on 15 of 30 bare prompts (see also `E5_name_REPORT.md`). Miru names them at layers 40–49 and 56–57; J++ and the
  naked eye never did. Counting them as targets would reward a stuck habit.

Such positions are excluded from every metric that uses (ii), the same positions for every readout and in every
condition. The counts of each case are reported per condition. For mid-reply-content-next positions, (ii) is (i).

**Formatting-like targets.** A content target is formatting-like when it is either:
- a digit-only token directly after a newline (a list numeral, `"\n\n" "1" "."`);
- or any token directly after a code fence (`"```" "python"`).

They stay in the primary metric. They are counted per condition, and sensitivity analysis S4 leaves them out
(amendment A7).

**Agreement check.** The reading forward's own top-1 at layer 59 is recorded beside (i), and their agreement is
reported. Both come from the same decoding rule, so they should agree except at near-ties between batches.

## 6. Metrics

**Primary.** At each read position, layer and readout, a hit means: the filtered top-1 equals the usable content
target. "Equals" means the same word: the decoded texts are equal after NFKC, stripping surrounding whitespace, and
Unicode case folding (`R2_common.word_key`). So `" Paris"` = `"Paris"`.

The reason for matching words rather than ids: after formatting, the model's content token often has no leading
space (`" **"`, `"Paris"`), while a lens that names the word usually gives the spaced form (`" Paris"`). An exact-id
match would count that as a miss for every readout alike. Exact-id matching is reported as a secondary metric.

**Stuck-word share.** *(Amendment A3.)* It is computed for each readout, layer and condition, over all the
condition's read positions, whether or not they have a usable target:

1. Group the positions by the word (`word_key`) of their filtered top-1, not by token id.
2. Leave out each position whose filtered top-1 is that position's own usable content target, so a correct common
   opener is not penalised.
3. The stuck word is the most common word among the rest. The share is its count divided by the number of read
   positions in the condition.

The share counting every position, by word and by id, is also reported.

**Formatting share.** For each readout and layer, this is the share of read positions whose *unfiltered* top-1 is
NON-CONTENT.

## 7. Aggregation, intervals and the decision rule

**Bands** (code layers): 40–49, 50, 51–54, 55, 56–57, 58.
- R1 is tested at 50, 55 and 58.
- Miru is tested at all six. Its result at 58 is for reference only (below).

**Prompt-level scores.** A prompt's score for a readout at a layer is the mean of its hits over its read positions in
the condition that have a usable target (one position, or up to six for mid-reply). Its band score is the mean over
the band's layers.

**Means.**
- A condition mean is the mean over that condition's prompts.
- The pooled mean is the mean over all prompts of the four decision conditions: chat-no-answer, chat-begun, bare and
  mid-reply. Each prompt has weight 1.
- Prompts with no position with a usable target are dropped, the same for every readout.

**Intervals.** 95% percentile intervals from 2,000 prompt-level bootstrap resamples, with seed 20261019
(`numpy.random.default_rng([20261019, condition index])`). Prompts are resampled with replacement within each
condition, so the pooled resample keeps the condition sizes.

The same resampled indices serve every readout, layer and band, so differences are paired. Intervals are computed for:
- second − J++ (both filtered);
- second − naked eye (both filtered);
- pooled and per condition.

**Decision rule** (fixed now; rules 4 and 5 and the per-condition form of rule 3 are amendments A2 and A3):

> Serve a lens at a band **iff**, pooled over all conditions, its filtered top-1 beats **both** filtered J++ and the
> filtered naked eye with 95% CIs excluding 0, by at least 5 points, and consistently over prompts (sign test,
> Holm-adjusted); it is not significantly worse than either in any single condition; and its stuck-word share is at
> most 25% in every condition at every layer of the band. Otherwise do not serve it there.

Operationally, a lens passes at a band when all five of these hold:

1. **Beats both, pooled.** The lower bounds of the pooled intervals for second − J++ and for second − naked are both
   greater than 0.
2. **Not worse in any condition.** In none of the four decision conditions is the upper bound of either per-condition
   interval below 0. A decision condition with no valid prompt, or with a readout missing, counts as a failure.
3. **No stuck words.** The stuck-word share (§6) is 0.25 or less in each of the four decision conditions, at every
   layer of the band. *(Amendment A3: this was one share pooled over the conditions, by token id.)*
4. **Consistent over prompts.** *(Amendment A2.)*
   - Take the prompt-level band-score differences, second − J++ and second − naked eye, over the valid prompts of the
     four decision conditions.
   - Run an exact one-sided sign test on each: ties are dropped, and p = P(at least n₊ of n₊ + n₋ fair coin flips).
   - The band's p is the larger of the two.
   - Holm's step-down adjustment is applied across every lens-band test (the 9 tests: R1 at 3 bands, Miru at 6). The
     adjusted p must be below 0.025.
5. **A real improvement.** *(Amendment A2.)* The pooled point estimates of second − J++ and of second − naked eye are
   each at least 0.05.

Every comparison with 0 or with a threshold uses unrounded values; rounding is for display only.

**Eligibility and precedence** (also fixed now):

- **Miru at 58 is reference only.** It is measured and reported, but it is never served, whatever its result. This
  follows the earlier collapse at 58.
- **R1 takes precedence where both pass.** Where both lenses pass in the same band (possible at 50 and 55), R1's lens
  is served. It is fitted on this exact checkpoint and checked against its gradients.
- **Miru may stand in for R1 at 50 or 55.** Where R1's lens fails at 50 or 55 and Miru passes there, Miru is served
  at that band.
- **No lens passes.** The second reading shows nothing at that band.

**Multiplicity.** Rule 4 is the multiplicity control: Holm across the 9 lens-band tests at a one-sided family-wise
0.025.

The bootstrap intervals of rules 1 and 2 are not corrected. The two baselines do not make rule 1 conservative: second
− J++ and second − naked share the lens's own noise, so requiring both is close to a single test. Rule 5 stops a
lens from passing on a tiny win. The first draft's pooled percentile bootstrap would have passed a lens on 4 of 220
discordant prompts, all favourable (sign-test p = 0.0625).

**Power.** Pooled (n ≈ 220 prompts), intervals resolve differences of about 0.06–0.07: enough for large effects such as
R1 at 58 in the seen test (0.64 against J++'s 0.28). Per condition (40–60 prompts), intervals are about ±0.12–0.17, so
rule 2 catches only large harms. This is accepted. The per-condition point estimates are reported for every band.

## 8. Reported for transparency (not used for the decision)

- Per layer:
  - the primary metric for every readout, with pooled intervals and the differences against J++ and the naked eye;
  - the stuck-word share and its word, pooled and per condition;
  - the formatting share.
- Per lens-band: the sign test's discordant prompt counts (n₊, n₋, ties) against each baseline, its raw p and the
  Holm-adjusted p.
- Unfiltered metrics:
  - unfiltered top-1 equals the next token (i), by exact id: the earlier test's metric;
  - unfiltered top-1 equals the content target (ii), by word.
- Exact-id variant of the primary metric, and the content target in the filtered top-5 (by word).
- **mid-reply-content-next:** the same tables for the "content_next" draw (no Holm family).
- **Splits by position type:** positions whose next token is NON-CONTENT (where the filter matters) and positions whose
  next token is CONTENT.
- **Baselines for the word match** *(amendment A6)*:
  - a constant guess: for each prompt, the most common usable target word among the other prompts of its condition
    (leave-one-out), scored like the primary metric, per condition and pooled;
  - the primary metric split by whether the target is a function word (`R2_common.FUNCTION_WORDS`, fixed now), for the
    second reading, J++ and the naked eye.
- **Counts:** positions with no target, collapse-word targets excluded, and formatting-like targets, per condition.
- **Context figures:**
  - how often the model itself opens with formatting (share of NON-CONTENT next tokens per condition);
  - the model's own filtered top-1 at 59 against the content target: how often the content word is already the
    model's own first content choice;
  - the agreement check of §5.
- **Lens comparison:** Miru − R1 at 50, 55 and 58 (filtered, pooled, with intervals).
- **Sensitivity analyses** *(amendments A1, A4, A7)*. Each reruns the whole rule (1–5, Holm, eligibility and
  precedence) and lists every lens-band whose verdict or serving changes against the primary decision. The primary
  decision stands; a change is reported beside it.

| | rerun without | reason |
|---|---|---|
| S1 | the 27 prompts about the model | R1's fit set had many self-questions |
| S2 | the 22 prompts of `R2_common.R1_OVERLAP` | paraphrase or shared answer topic with R1's fit prompts (§3.2) |
| S3 | the bare condition (rule over the other three) | raw text puts this instruct model off-distribution |
| S4 | positions with a formatting-like target (§5) | list numerals and code-fence tags may reward list structure, not content |

## 9. Failure handling

**Stops before reading.** The run stops before any reading in any of these cases:
- a pre-registered file (§ Integrity) is untracked or differs from HEAD (`--allow-uncommitted` overrides this, as a
  recorded deviation);
- `R2_02_eval.json` or `R2_02_states.pt` already exists (`--force` overrides this; the earlier run stays in
  `R2_02_runs.jsonl` and is reported);
- a lens file's size or hash differs from its pin;
- the filter's spot checks fail, or its class counts differ from those `R2_01_check.json` recorded;
- `R2_prompts.json` fails validation;
- the prompt file's sha256 differs from the one `R2_01_check.json` recorded.

**Agreement below 0.95.** If the agreement check of §5 falls below 0.95 in any condition, the analysis says so in the
report. The rule is applied unchanged.

**Partial runs.** A run that stops part-way is rerun from the start. Partial data is not analysed. The stopped run
stays in `R2_02_runs.jsonl` and is reported.

## 10. How to run

The worktree has no virtual environment of its own; use the main checkout's. Run from the root of the checkout these
files are in.

1. CPU, any time (no model):
   `CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_01_check.py`
   This writes `R2_01_check.json` and exits 1 on any failure.
2. Commit the pre-registered files (§ Integrity).
3. GPU, with the server stopped:
   `/home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_02_eval.py`
   This writes `R2_02_eval.json` and `R2_02_states.pt` (gitignored), and appends to `R2_02_runs.jsonl`. Options:
   - `--r1 PATH` (default `experiments/results/R1_18_it_lens.pt`);
   - `--miru PATH` (default: the local Hugging Face cache at the pinned revision);
   - `--batch N` (default 8);
   - `--force`, `--allow-uncommitted` (§9; each is recorded and is a deviation to report).
4. CPU:
   `CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/R2_03_analyze.py`
   This writes `R2_03_analysis.json` and `R2_03_report_section.md`, and applies §7 exactly.

**Checks that need no model** (CPU), made before the GPU run:
- `R2_02_eval.py --check-only` runs every gate of §9: the commit state (as a warning), existing output, the prompt
  hash against `R2_01_check.json`, the filter, and the lens sizes, hashes, Miru header and R1 payload.
- `R2_02_eval.py --selftest --limit 3 --out <scratch>.json` runs the whole pipeline on a tiny random stand-in for the
  model.
- `R2_03_analyze.py --selftest` checks every branch of the rule on a synthetic file with known answers: each of rules
  1–5 failing alone, precedence, reference-only, the sensitivity analyses, and the unit pieces (target search, filter,
  sign test, Holm).

The analysis refuses to write self-test data into `experiments/results`, whether to the analysis or to the report.

## 11. Amendments before any data

The first draft of this file was reviewed twice before any R2 data existed. No R2 forward pass, generation or readout
had been run, and none has been run since, so none of these is a deviation. The prompt file is unchanged (sha256
`623a7fb7…d01fd`).

| | change | reason |
|---|---|---|
| A1 | The content-target search stops at any special token (§5). Content targets `" own"` / `" same"` are excluded for every readout (§5). Sensitivity S3 (without bare). | In bare, the target would often be the instruct model's raw-text junk word, which Miru tends to name and J++ never did. After `<\|channel>` the template word "thought" would have counted as content. |
| A2 | Rules 4 (exact sign test against both baselines, Holm across the 9 lens-band tests, < 0.025) and 5 (pooled point estimates ≥ 0.05) (§7). Discordant counts reported. | The percentile bootstrap alone passed a lens on 4 favourable discordant prompts of 220. There was no minimum effect, no multiplicity control, and the two baselines share the lens's noise. |
| A3 | The stuck-word share is checked per decision condition, by word rather than id. A position whose top word is its own target is not counted (§6, rule 3). | A lens fully stuck in one condition showed at most 0.20 pooled. Variants of one word split the count by id. A correct common opener should not count as stuck. |
| A4 | §3.2 states that Miru's WikiText-103 fit passages are unknown and unchecked. Sensitivity S1 (without the 27 prompts about the model) and S2 (without the 22 prompts of `R1_OVERLAP`, fixed now). | Several prompts paraphrase R1's fit questions or share their answer topics, beyond what a lexical check catches. |
| A5 | mid-reply's decision draw is "any" (6 positions anywhere in the reply). "content_next" (3 positions) is reported only (§4.3). | The content-next draw left out exactly the positions the filter is for (a formatting token next), and it was not what the instrument shows. Six positions also tighten the mid-reply prompt scores. |
| A6 | A constant-guess baseline, and the primary metric split by function-word targets, are reported (§8). | Case-folded matching makes chat-no-answer partly a test of guessing a common opener ("The"). This inflates the comparison against J++ for any readout that defaults to " the". |
| A7 | Formatting-like content targets (list numerals after a newline, the token after a code fence) are counted; sensitivity S4 leaves them out (§5). | Readouts fitted on chat might be rewarded for predicting list structure rather than content. |
| A8 | The integrity hashes cover every pre-registered file, `R2_03_analyze.py` and `R2_01_check.py` among them. R2_02 refuses uncommitted files and refuses to overwrite its output without `--force`. Every run is logged to `R2_02_runs.jsonl`. Rules compare unrounded bounds. A decision condition with no data counts as a failure. The report guard covers `--report` too (§ Integrity, §7, §9, §10). | The analysis code that implements the rule was not hashed. A rerun would have silently replaced the output. Rounding could hide a bound in (−5e-5, 0). The no-data rule was in the code but not in this file. |
| A9 | Filter rule (c) also treats invisible characters (Cc, Cf, Co, Cn), variation selectors and the keycap mark as non-content when every other character is punctuation or a symbol: 482 tokens moved (§2). | The served reading should not show `\x00`, zero-width spaces or chat bullets like `✔️`, and the test should measure the filter that would be served. The change helps whichever readout favoured these tokens: they no longer take its top place. That may slightly favour the instruct lenses, which may favour chat bullets. It applies to every readout alike. |
