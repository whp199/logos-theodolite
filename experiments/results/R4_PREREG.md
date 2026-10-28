# R4: R1's instruct lens at pages 56 and 59, on fresh prompts (pre-registration)

*Drafted on 27 October 2026, and amended before any data (§13):
- after a first independent read;
- for the owner's choice of 640 prompts;
- after a second independent read of the draft, the scripts and the prompts;
- on 28 October, for the owner's choice that a band is served only if R1 also passes without function-word targets
  (the S6 gate, §5).

**Registered on 28 October 2026** on the branch `r4-r1-lens` (§0), after the replication record of §4.2 passed. No
R4 prompt had then been read by the model, and no R4 forward pass, generation or readout had been run.*

*Scripts: `experiments/scripts/R4_*.py`. Prompts: `experiments/scripts/R4_prompts.json`. Results: `experiments/results/R4_*`.*

Layers are given as **code layers (pages)**, where page = code + 1. R1's lens was read at code 55 and 58, which are
pages 56 and 59.

## 0. Integrity

R4 keeps R2's and R3's scheme.

**Registration** is two commits on the branch `r4-r1-lens`, cut from `main` at `efb72fd`:
1. The first adds R4's files.
2. The CPU checks of §10 are then rerun at that commit.
3. The second, **the registration commit**, adds their records and the sha256 table at the end of this section (§0.1),
   as R3 §0 has.

The table covers:
- **R4's own files:** this file; `R4_prompts.json`; `R4_common.py`, `R4_00_replicate.py`, `R4_01_check.py`,
  `R4_02_eval.py`, `R4_03_analyze.py`, `R4_power.py`; and `R4_01_check.json`, `R4_power.json`, `R4_selftests.json`,
  `R4_03_r3_reproduction.json`, `R4_00_reread.json`, `R4_00_runs.jsonl`, `R4_00_replication.json`.
- **The files R4 imports or reads, unchanged.** Their sha256 are pinned now in `R4_common.PINNED_UNCHANGED`:
  - R1's fit lists: `R1_common.py`.
  - R2's: `R2_common.py`, `R2_03_analyze.py` (`holm`, `binom_sf_half`), `R2_prompts.json`, `R2_01_check.json`,
    `R2_02_eval.json`, `R2_03_analysis.json`.
  - R3's: `R3_common.py`, `R3_05_eval.py` (its backends, greedy decoding and draws), `R3_06_analyze.py` (the rule),
    `R3_gpu.py` (the loader and runs log), `R3_fitlib.py` and `R3_lrp.py` (imported by the backend), `R3_prompts.json`,
    `R3_sources.json`, `R3_05_eval.json`, `R3_06_analysis.json`.
  - The backend files the readouts go through: `backend/theodolite_math.py`, `backend/engine_runtime.py`,
    `backend/__init__.py`, and `backend/label_gate.py` (which `engine_runtime` imports).

  This is R3 §0's block "R2's files R3 imports or reads, unchanged". It also closes the gap of R2's amendment A8: the
  code that implements the rule is hashed.
- **The environment** (`R4_common`):
  - the model snapshot `52f3f65bc7a02d555763bc923bd1d9094898219d`, with each of its files' blob names
    (`SNAPSHOT_BLOBS`; the weights' blob is named by its sha256);
  - torch `2.14.1+cu130` and transformers `5.20.0`.
  - Recorded, and required to be the same at R4_02 as at R4_00:
    - the other libraries (numpy, compressed-tensors, accelerate, safetensors, tokenizers, huggingface-hub and
      interp-engine);
    - the GPU and its driver;
    - the sha256 of every module file the GPU process loads from `backend/`, interp_engine and R2's and R3's scripts
      (`R4_common.environment`, `loaded_code`).

**Refusal.** `R4_common.runtime_state()` checks all of these pins.
- **R4_00 and R4_02 refuse to start, and record why, while any pin differs.** No flag overrides this.
- **They also refuse while their output exists.** `--force` overrides this, and the forced run is recorded.
- **R4_02 also refuses while any registered file is untracked or differs from HEAD.** `--allow-uncommitted` overrides
  this, and the override is a recorded deviation.
- **R4_03 refuses to decide** unless:
  - the rule's own code (`R2_common`, `R2_03_analyze`, `R3_common`, `R3_06_analyze`) still has its pinned sha256;
  - `R4_common.py` (the design and `FIT_OVERLAP`) is the one R4_02's run used;
  - the replication record is the one R4_02's run checked;
  - `R4_03_analyze.py` and `R4_common.py` are committed (`--allow-uncommitted`, a recorded deviation).

Two R3 tools are avoided:
- R4_02 does not use `R3_gpu.require_committed`, which refuses any `backend/` that differs from R3's commit `d5405ee`;
  today's backend does.
- No GPU script imports `R3_06_analyze.py`, which sets `CUDA_VISIBLE_DEVICES=""` when imported. Only R4_03, R4_power
  and the CPU modes of R4_00 and R4_02 import it.

**Run logs.** R4_02 records HEAD in its output and appends every run (started, finished or failed) to
`R4_02_runs.jsonl`; R4_00 does the same in `R4_00_runs.jsonl`. Every run is reported.

**Deviations.** Any change made after any R4 GPU output exists is a deviation, a plain bug fix included. It is reported
as one, with the result under this file's rule beside it.

**Stopping for an amendment.** If, before any R4 prompt is read by the model, something here proves unworkable, the
run stops. The change is made as an amendment and the registration commit is redone. The stop is reported. The
replication record of §4.2 reads no R4 prompt.

**R1's lens file.** R1's pinned lens is `R1_18_it_lens.pt`:
- sha256 `80ce560943f39bc4183a7387512d20ad7ef779476fc1ef01b5010463a43e107b`, 173,411,927 bytes;
- the pin of R2 (`R2_common.R1_SHA256`) and R3 (`R3_common.R1_SHA256`).

Where it is:
- **The pinned file is gitignored** (`.gitignore:15`). It was found only in the `it-second-reading` worktree.
- **The main checkout's copy** has the same size and the same three matrices, but an older provenance note. Its sha256
  is `37252f5e…`, so it is not the pinned file.
- **The stable copy.** On 27 October it was copied to a stable path outside every worktree,
  `~/.local/share/theodolite/lenses/R1_18_it_lens.pt`, and its full sha256 matches the pin. `R4_common.find_r1_lens`
  tries that path first, and checks by full sha256 (`full=True`; with `full=False` the main checkout's copy passes on
  size alone). R4_00 and R4_01 require the stable copy, and R4_02 reads it.

**R3's cache.** The WikiText rows and J++'s items that the freshness check reads are another gitignored dependency. They
are in R3's worktree, `.claude/worktrees/it-jpp-refit/experiments/R3_cache`, and each is checked by sha256 against R3's
committed manifest (`R3_sources.json`) before use.

**The runtime.** R4 reads through the instrument's backend at the registration commit:
- `backend/theodolite_math.py`: `jpp_readout_logits`, `unembed_gemma4`, `validate_and_cast_lens`, `gemma4_rms_norm`;
- `backend/engine_runtime.py`: `dequantize_w4a16_full`, `patch_compressed_tensors_for_low_vram`,
  `_make_w4a16_linear_forward`.

Compared function by function with docstrings stripped, none of these has changed since R3's commit `d5405ee`. The
`theodolite_math.py` diff only adds the contrast functions, and the 4-bit forward's changes are comments. The
replication record of §4.2 checks this on the model as well.

## 1. Question

The instrument reads `google/gemma-4-31B-it-qat-w4a16-ct` through J++, which was fitted on the base model, and every
page says so. No lens fitted on the instruct model has yet passed a pre-registered test of where it may be served:
- **R2** served none;
- **R3** served neither of its two refits at any band from code 30 to 58.

R1's lens is the one lead (ROADMAP §i). R3 read it only for reference, outside the decision and its Holm family. On
R3's prompts it passed all five of R2's rules at code 55 and code 58, where on R2's prompts it had failed at both:

| page (code) | R2, 199 prompts: R1 − naked eye | R3, 283 prompts: R1 − naked eye | R1 − J++ (R2 / R3) |
|---|---|---|---|
| 56 (55) | +0.044 [−0.004, +0.092]: rules 1 and 5 fail | +0.071 [+0.035, +0.109]: all five pass | +0.085 / +0.107 |
| 59 (58) | +0.046 [+0.013, +0.084]: rule 5 fails | +0.058 [+0.024, +0.091]: all five pass | +0.075 / +0.083 |

*(From `R2_03_analysis.json` `bands.r1` and `R3_06_analysis.json` `reference_bands.r1`: pooled share of positions where
the filtered top-1 names the model's next content word. R4's own analysis code reproduces the R3 column exactly:
`R4_03_r3_reproduction.json`.)*

**What the target is, and where R1's lead comes from.** R2's filter removes only special tokens, whitespace,
punctuation and symbols, and HTML tags. So "the", "a", "of" and "is" count as the "content" target. Of R3's 678 usable
targets in the decision conditions, 304 are function words (`R2_common.FUNCTION_WORDS`): 72% in chat-no-answer, where
"the" alone is 27 of 80, and 47% in mid-reply.

R1's lead over the naked eye lies almost wholly in those targets. Under R4's own rule and code, with each set's own
bootstrap seed ("left out": the positions whose target is a function word are dropped, as S6 does):

| prompts, targets | page 56: R1 − naked eye | page 59: R1 − naked eye | rules failed (56; 59) |
|---|---|---|---|
| R3, all (the registered metric) | +0.071 [+0.035, +0.109] | +0.058 [+0.024, +0.091] | none; none |
| R3, function-word targets left out | +0.033 [−0.004, +0.072] | +0.041 [+0.002, +0.079] | 1, 5; 5 |
| R3, function-word targets only | +0.149 | +0.108 | none; none |
| R2, all (the registered metric) | +0.044 [−0.004, +0.092] | +0.046 [+0.013, +0.084] | 1, 5; 5 |
| R2, function-word targets left out | −0.004 [−0.049, +0.041] | +0.002 [−0.037, +0.041] | 1, 4, 5; 1, 4, 5 |
| R2, function-word targets only | +0.139 | +0.147 | none; none |

Over J++, the picture is milder. With function-word targets left out, R1 − J++ is +0.046 and +0.090 in R3, and +0.000
and +0.063 in R2.

R2's amendment A6 foresaw function words inflating the comparison, but only against J++. R2's rule nonetheless counts
them, and R4 keeps R2's rule. So the measure R4 tests is the model's next word under R2's filter, function words
included. It is not the next *substantive* word.

**The owner's choice (28 October 2026).** R1 must win on substantive words too, so R4 gates on S6. A band is served
only if R1 passes R2's five rules there, unchanged, **and** passes all five again under **S6**. S6 is the same rule on
the same prompts, without the positions whose target is a function word (§5).

A lead found by looking is not a finding. R4 asks the question once more, about these two layers only, on prompts no
one has looked at, with R1's lens as the only candidate:

> **At page 56 and at page 59, does R1's filtered top-1 name the model's next word more often than filtered J++ and the
> filtered naked eye, by R2's rule, both as R2 counts the next word (function words included) and without the
> function-word targets?**

Two hypotheses, one per band:
- **H1 (page 56):** R1 passes R2's five rules at code 55, and passes them again under S6.
- **H2 (page 59):** R1 passes R2's five rules at code 58, and passes them again under S6.

**What would refute.** A failure of any rule at a band, by R2's rule or under S6, refutes that band's hypothesis, and
R1 is not served there.
- R4 tests no other lens and no other layer, and adds none after data exist.
- R1's matrices exist at 11 layers. By R1's own report they are usable at five of them, but only code 55 and 58 have
  the record above.
- R3 read R1 at three layers. Code 50 (page 51) failed in both R2 and R3 (in R3 on rule 5 alone, at +0.046 over the
  naked eye), and it is not retested.

## 2. What a pass would change, and what it would not

A pass changes little, and that is the reason to say so first.

A "pass" below means a served band: R2's rule and S6 both pass there (§5).

**What a pass permits.** A pass at a band permits a later serving track, on a branch from `main`, to show R1's filtered
top-1 reading at that page. It would be a filtered second reading, offered as a request option, with J++ beside it in
the apparatus. A pass at a band supersedes R2's decision to withhold R1 at that band only; R2's and R3's failures stay
on record. The serving track must still pass the gates R3 §7.2 names:
- orientation;
- bf16 against fp32 precision;
- the stuck-word check;
- keeping the server's defaults.

Its lens card would say exactly what was tested: *the filtered top-1 (R2's filter) as the model's next word, at this
layer, on R4's prompts, better than J++ and the naked eye with function words counted and without them*. Ranks 2 and
below are not tested, since the top 5 is reported only.

**A band that passes R2's rule but fails S6 is not served.** R4's report then says plainly that R1's gain there was in
function words ("the", "of", "is"), not in substantive words. That is a finding about R1, and not a reason to serve
it.

**What a pass does not change:**
- **Every other layer.** J++ covers code 0–58, and page 60 is the model's own output. At the other 57 J++ layers J++
  stays the reading, and its caution stays.
- **The other readings through the lens.** The gathered/recalled split, the landmark bearings, the nudge and the vault
  all stay on J++ (README fault 9).
- **Neuron labels.** The label gate was validated for base-fitted J++ labels only: AUC 0.84 graded, 0.64 as the cut
  (E47). Labels through R1 would need their own validation of the gate, and R4 does not do it.
- **Pronoun sentences.** None is shown through R1 at pages 51–59 except where J++ and the instruct reading agree (E49,
  E50).
- **The early layers.** The caution at pages 11–30 stays for every averaged lens (E50).

**How much a pass would show.** A pass near the threshold, at page 59 above all, would not establish a true margin of 5
points or more (§7). It would show that R1 met R2's rule on a third, fresh set.

## 3. Prompts

### 3.1 Conditions and composition

**640 prompts** were written for R4, in R2's schema (`R2_common`'s validation, with R4's counts), with ids
`r4-<cna|cb|bare|mid>-001` to `-160`. `R4_common.validate` enforces the composition, which is R3 §4.1's doubled. The
owner chose 640 over 320 on 27 October (§7, the operating characteristic):

| condition | n | the text read | read position | prompts about the model |
|---|---|---|---|---|
| chat-no-answer | 160 | The question in the chat template, with no system prompt and thinking off (empty thought channel). | Its last token, `<channel\|>`. | 20 |
| chat-begun | 160 | The question, then a begun answer that ends just before a content word. | The begun answer's last token. | 18 |
| bare | 160 | A plain-text opening with BOS and no template. | Its last token. | 10 |
| mid-reply | 160 | A new question. The model's greedy reply (thinking off, 48 new tokens) is generated in the run. | Seeded positions anywhere inside that reply (§4.3). | 16 |

**The chat-begun answers** are R3's kinds, doubled:
- 40 factual, 22 creative, 18 advice, 18 about the model (kind `self`, exactly the 18 about-the-model prompts);
- 18 list openings, 10 opinion, 10 empathy, 8 how-to, 6 code, 6 language, 4 arithmetic.

38 are marked `format_likely`, by kind as in R3: every list, code, language and arithmetic answer, and 4 creative ones.
The prompts run from 9 to 56 tokens.

**Who wrote them, and what they saw.**
- **The writers.** Four writers wrote the prompts, one per condition, each told not to make prompts easy or hard for any
  lens. Each could read only R2's and R3's prompt files (to avoid them), `R1_common.py` (R1's fit questions, for §3.3)
  and `R4_common.py`. None opened any file under `experiments/results`. They never saw a result, an analysis or a model
  output of R2 or R3, except the matched strings that the freshness check printed back while they rewrote flagged
  prompts.
- **The bare writer** wrote 253 candidates and kept 160 for balance across genres, not by guessability or overlap.
- **The coordinator.** I merged the four sets, and in four rounds replaced 46 prompts that duplicated others in subject
  and answer (§3.2). In checking R3's file layout I saw R3's aggregate reproduction and one stored R3 reply. No
  per-prompt R1, J++ or naked-eye result of R2 or R3 was consulted while the prompts were written.
- **What the coordinator knew.** The coordinator knew R2's and R3's per-condition summaries (R1's margin is largest at
  mid-reply and smallest at chat-no-answer) and, from the second read, that R1's lead lies in function-word targets.
  It wrote every replacement prompt knowing this. The composition was fixed in advance, so that knowledge chooses no
  counts, but it could in principle bend the replacements' wording. The replacements are listed by id in §3.2, so a
  reader can rerun the decision without them.

### 3.2 Freshness check (`R4_01_check.py`, result in `R4_01_check.json`)

R3's check (`R2_common.freshness_check`, R2's rule unchanged) is rerun with R4's prompts as the subject. It compares
960 queries (each question, begun answer, question + begun answer, and bare text) with 27,261 corpus strings:
- every string literal of `experiments/scripts/*.py` and every string of `experiments/results/*.json`, in the main
  checkout and in each of the seven worktrees under `.claude/worktrees`, with R4's own files excluded and files
  deduplicated by content (401 script files, 768 results files). This holds:
  - R2's and R3's results;
  - R3's evaluation output (`R3_05_*.json`, among them the model's greedy replies to R3's mid-reply prompts);
  - R1's fit corpus;
- R2's and R3's prompt files;
- the N1B corpus (decoded);
- the 2,893 cached WikiText rows and the 1,743 J++ item strings (R3's cache, checked by sha256);
- the other R4 prompts.

**The flag rule** is R2's:
- exact;
- contained (4 or more words);
- contains (a source of 5 or more words);
- similar (difflib 0.80 or more, or word Jaccard 0.75 or more).

The check also confirms that planted copies of a WikiText row, a J++ item, an R2 prompt and an R3 prompt are flagged,
and a fresh string is not. A match inside a third-party text is recorded by source, item number and sha256, never by
text (R3 §4.2, D26).

**Result: 0 flags against the corpus, 0 among the R4 prompts** (largest similarity 0.796). The writers rewrote 13
flagged prompts in all, each with a new subject.

**The duplicate screens** go further than the rule. Each prompt should be an independent sample, and two prompts with
the same subject and the same answer count one fact twice. The writers worked apart, and a string match catches only
like wording. Four rounds of screening:
1. **Exact repeats across conditions.** Six questions were asked twice, word for word, and the rule caught them.
2. **A content-word screen** found 11 pairs with the same subject and the same expected answer, and 4 with the same
   subject. Examples: the platypus laying eggs, in bare and chat-begun; what a light-year measures, in bare and
   chat-no-answer.
3. **The second independent read** found 15 more groups, within one condition and across conditions. Examples: the
   tomb opened in 1922, twice in chat-no-answer; "tokens", in three conditions; "do you learn from our chats", in four
   prompts.
4. **An answer-level audit** of all 640, made by a fresh reader who read every prompt and wrote down its subject and
   answer, found 6 more duplicate groups (for example, "Do you get jokes?" and "Can you tell when I'm joking?") and
   11 borderline ones. One prompt was replaced in each duplicate group, and in the 5 borderline groups that leaned
   towards duplicate.

How the replacements were made:
- **Which prompt went.** One prompt of each group was replaced, keeping a chat-begun one where there was one, because
  its begun answer depends on its question.
- **Repeat flags.** Four replacements were themselves flagged against R2's, R3's or the study's prompts, or against one
  another, and were replaced again.
- **The 46 ids replaced:**
  - chat-no-answer 001, 022, 026, 030, 036, 045, 047, 048, 053, 059, 063, 070, 071, 072, 084, 086, 087, 089, 103, 107,
    108, 124, 133, 137, 139, 143, 145, 159;
  - bare 013, 029, 055, 071, 099, 139, 148, 160;
  - mid-reply 005, 012, 039, 041, 068, 083, 105, 125, 152, 157.

**What is left** are the audit's 6 borderline groups that lean towards distinct. Examples: what a website's support bot
covers, in two bare openings with different next words; a training cutoff, as a year in one prompt and as a behaviour
in another. Some prompts also share only a theme, such as time zones (a count in one prompt, an explanation in
another).

### 3.3 What the check does not cover

**R1's fit set.** R1's lens is the candidate, so overlap with its fit set is the overlap that matters.
- **The list.** `R4_common.FIT_OVERLAP` was fixed from the prompt texts alone, in the same commit as the prompts and
  before any data.
- **What it names.** 151 R4 prompts: 28 chat-no-answer, 40 chat-begun, 45 bare and 38 mid-reply. R3's list named 83
  of 320, 45 of them for R1's fit set. Each listed prompt either paraphrases one of R1's fit questions
  (`R1_common.R1_SELF_Q`, `R1_TASK_Q`), or has a subject *or expected answer* that is the subject of one of them or of
  an R1 encyclopedia topic (`R1_ENC_TOPICS`). R2 §3.2's example is bare-01, whose answer is the Amazon. For a
  next-word metric, the answer is what counts.
- **How it was made.** The writers judged their own prompts generously and quoted the R1 entry exactly. The coordinator
  judged each replacement.
- **How it is used.** Sensitivity S2 (§8) reruns the decision without these prompts.

**Two gaps** carried over from R2 and R3:
- R1's 40 /usr/share/doc chunks are in the sources only as R1 stored them, with their token ids removed (R3 §4.3).
- Miru's fit passages are not rebuilt. They are irrelevant here: Miru is not read.

R1's fit (`R1_03_fit.py`) estimates E[J] from vector-Jacobian products with random probes; it does not regress on next
tokens. So the risk of overlap is limited, as R2 §3.2 noted.

## 4. Reading

### 4.1 Model, runtime and lenses

`R4_02_eval.py` loads the served checkpoint itself, with the server stopped, as R3 §5.1 does. It uses R3_05's
`GpuBackend` (`R3_gpu.load_w4a16`, the instrument's 4-bit forward), greedy decoding and position draws, imported
unchanged.

**Gates before the model loads.** R4_02 checks:
- §0's gates;
- `R4_prompts.json` validates with R4's counts and composition, and its sha256 is the one a passing
  `R4_01_check.json` recorded;
- R2's filter has `R2_01_check.json`'s class counts;
- J++'s file and R1's file each have their pinned size and full sha256;
- `R4_00_replication.json` exists, passed, and was made at the registered runtime, on the pinned snapshot, with the
  same `R4_common.py`, `R4_02_eval.py` and `R4_00_replicate.py` (§4.2).

| readout | layers (code) | path |
|---|---|---|
| J++ (base) | 55, 58 | `jpp_readout_logits(h, J_l)`, the Neuronpedia fp16 file cast to bf16 |
| naked eye | 55, 58, 59 | `unembed_gemma4(h)`; at 59 it is the model's own output |
| R1 | 55, 58 | `jpp_readout_logits(h, J_l)`, R1_18's matrices in bf16 |

`resid_post` is captured at code 55, 58 and 59. Each readout keeps:
- R2's unfiltered and filtered top-5 ids and texts (`u`, `us`, `f`, `fs`);
- the ranks of the targets (`ur`, `fr`);
- whether the unfiltered top-1 is content (`u0c`).

The output is laid out as R3_05's (schema `R4_02/1`), so R3_06's `Table` reads it. R4 records no readout through J++'s
exclusion list, so R3's sensitivity S5 is not run. R3_06 then reports it as not run.

### 4.2 Replication record (`R4_00_replicate.py`, before registration)

**Why.** The backend has moved since R3 read through it. So before R4 is registered, a separate GPU job, approved by the
owner like any GPU run, re-reads R3's own evaluation with R4's code. It reads no R4 prompt. Its record is part of the
registration, and a failure is fixed before registering, so it never causes a deviation.

**The GPU job** writes `R4_00_reread.json` in two parts, both through R4_02's own functions.

**(a) Teacher-forced re-read.** For all 320 R3 prompts, the job feeds R3's stored inputs:
- for chat-no-answer, chat-begun and bare: each prompt as `R2_common` composes it, refused unless it rebuilds to R3's
  stored token count;
- for mid-reply: the prompt followed by R3's stored `reply_ids`.

It reads at R3's 930 stored positions, batched as R3 batched them: 8 prompts a forward, and R4_00 refuses any other
batch size. It requires the token read at every position to be the one R3 read there, so that the input is R3's. It
also requires the filtered top-1 id to equal R3's recorded one (`R3_05_eval.json`, `r[kind][layer]["f"][0]`) at
**99% or more** of the positions, for each of the seven readouts:
- J++ at 55 and 58;
- the naked eye at 55, 58 and 59;
- R1 at 55 and 58.

When R3 re-read R2's prompts through a different loader, every one of 9,861 J++, 10,380 naked-eye and 1,557 R1
filtered top-1 pairs matched (`R3_06_analysis.json`, `replication_r2_prompts`).

**(b) Regenerated targets.** It regenerates R3's greedy continuations and mid-reply replies, and places the read
positions again with R3's draw seed (20261021). It requires 95% of R3's read positions to be found again (R3 §8.6(a)),
with the same draws, next token, content target and continuation. For mid-reply, the reply must also be the same up to
the read position's next token. This checks the part (a) cannot: the model's own words, which define the targets.

**(c) Verdicts.** R3's prompts, with the teacher-forced readouts, are analysed under R4's design with R3's bootstrap
seed (20261022). R1's rules at 55 and 58 must equal R3's:
- rules 1, 2, 3 and 5 directly;
- rule 4 on the raw p, since R3 took R1's p outside a Holm family.

If every top-1 of (a) matched, every mean, interval, sign count and raw sign-test p must also equal R3's, to its
stored rounding. Otherwise each number that moved is listed.

R4_00's self-test shows how sharp (c) is. Changing R1's top-1 at 24 of the 930 positions (2.6%) at code 58 takes R1's
margin over the naked eye below 0.05, and (c) fails.

**The record**, `R4_00_replication.json`, written by `R4_00_replicate.py --compare` on the CPU, keeps what it was made
with:
- the pinned files;
- the snapshot's blobs;
- the library versions;
- the environment;
- the code the GPU process loaded;
- the sha256 of the R4 scripts that read and compared: `R4_common.py`, `R4_02_eval.py`, `R4_00_replicate.py` and
  `R4_03_analyze.py`.

It passes only if the scripts that compared are the scripts that read.

R4_02 refuses to run unless all of these hold:
- the record passed;
- it compares the re-read that R4_00's last finished run wrote (the sha256 chain from `R4_00_runs.jsonl` through
  `R4_00_reread.json` to the record);
- every item it keeps still matches;
- after the model loads, the code the process loaded is the code the replication ran.

R4_02 also records the record's sha256 in its output, and R4_03 requires it unchanged. So a change to any of those
scripts after the record, `FIT_OVERLAP` in `R4_common.py` included, calls for a new record. Every amendment must
therefore land before R4_00 runs.

**On failure,** the job reports what differs, and R4 is not registered until the cause is found. A changed backend is
fixed, or pinned to R3's tree.

**Result (28 October 2026, before registration): the record passes**, and it matches R3 exactly:
- **(a):** all seven readouts match R3's filtered top-1 at all 930 positions, 6,510 of 6,510, and the unfiltered top-1
  matches too. Every rebuilt input reads the token R3 read.
- **(b):** all 930 of R3's read positions are found again on regeneration. That means every continuation, every
  mid-reply reply up to each read position, every draw and every target.
- **(c):** R1's rules, verdicts, means, intervals, sign counts and raw p at 55 and 58 equal R3's with no difference.
  Examples: R1 − naked eye +0.0709 and +0.0578, and raw p of 1.55×10⁻¹⁰ and 3.11×10⁻⁸.

The run took 656 s on the RTX 5090 (driver 595.104.02), with the runtime and environment §0 pins. Its record is
`R4_00_replication.json`, and its re-read `R4_00_reread.json` (sha256 `aaa07cf2…`). `R4_02_eval.py --check-only` then
passes with nothing pending.

### 4.3 Decoding, read positions, targets and metrics

R2 §4.2–§6, unchanged:
- **The model's own word** at a position: the argmax of the final RMSNorm, W_U and the 30.0 softcap of `resid_post[59]`.
- **Greedy continuations:** full forwards, no KV cache, right-padded. A sequence stops after `<eos>`, `<turn|>` or
  `<tool_response|>`.
- **Read positions:** one per prompt (the last prompt token) for the first three conditions. For mid-reply, two seeded
  draws without replacement from the greedy reply, with draw indices as in R2 §4.3 and `R3_05_eval.draw_mid_positions`:
  - **draw 0** is "content_next": up to 3 positions whose next token is CONTENT, reported only (mid-reply-content-next);
  - **draw 1** is "any": up to 6 positions anywhere in the reply, the decision condition.

  Each draw has its own stream `numpy.random.default_rng([20261028, prompt index, draw])`, where the prompt index is the
  question's 0-based place in the mid-reply list. R4_02's self-test checks this against an independent
  implementation.
- **The content target:** the first CONTENT token of the model's own greedy continuation of up to 8 tokens. The search
  stops at a stop id or any special token. A raw-text collapse word (`" own"`, `" same"`) is no usable target.
- **Primary metric:** the filtered top-1 is the same word as the usable content target (`R2_common.word_key`).
- **Stuck-word share:** by word, leaving out positions whose top word is their own target.

## 5. Decision rule

R2's rule, unchanged except for the family size. It is applied by R3's code: `R4_03_analyze.py` imports
`R3_06_analyze.analyze` unchanged and passes it R4's `Design` (`R4_common.DESIGN_FIELDS`):

- **Identity and conditions:** `name="R4"`; `conditions` R2's four; `secondary_conditions` `("mid-reply-content-next",)`.
- **Readouts:** `kinds=("jpp", "naive", "r1")`, `candidates=("r1",)`, `reference=()`.
- **Baselines:** `(("jpp", "J++"), ("naive", "naked eye"))`.
- **Bands:** `(("55", (55,)), ("58", (58,)))`; `reference_only=frozenset()`.
- **Serving:** `precedence=("r1",)`, needed because `decide()` sets eligibility and serving only for lenses in it;
  `min_serve_layer=55`; `require_all_bands=True`, so Holm always runs over both tests.
- **Bootstrap:** `boot_seed=20261029` (`numpy.random.default_rng([20261029, condition index])`); `n_boot=2000`.
- **Overlap:** `overlap=dict(R4_common.FIT_OVERLAP)`, with its `overlap_why`.

R1 passes at a band when all five hold (R2 §7, R3 §7.1):

1. **Beats both, pooled.** The lower bounds of the pooled 95% intervals of R1 − J++ and R1 − naked eye are both above 0.
2. **Not worse in any condition.** In none of the four decision conditions is the upper bound of either per-condition
   interval below 0. A decision condition with no valid prompt, or a readout missing, is a failure.
3. **No stuck words.** The stuck-word share is at most 0.25 in each decision condition at every layer of the band (here,
   one layer).
4. **Consistent over prompts.** Take the exact one-sided sign test of the prompt-level band-score differences against
   each baseline, with ties dropped, and keep the larger p of the two. Holm-adjust across the **2 tests** (R1 at 55, R1
   at 58); the adjusted p must be below 0.025.
5. **A real improvement.** The pooled point estimates of R1 − J++ and R1 − naked eye are each at least 0.05.

Every comparison uses unrounded values. **The 0.05 minimum is R2's, and R4 does not change it.** R1 has twice landed
just below or just above it at page 59. Lowering it now, knowing that, would decide the test before it is run.

**The S6 gate (the owner's choice, 28 October 2026).** A band is **served** only if R1 passes all five rules there,
**and** passes all five again under S6. S6 is R3_06's whole rule, run with `R3_06_analyze.sensitivity`, on the same
reading with every position whose usable target is a function word (`R2_common.FUNCTION_WORDS`) dropped:
- A prompt left with no position leaves its condition. A decision condition left with no valid prompt fails rule 2, as
  in the main rule.
- Within S6, Holm runs over the same 2 bands. Neither rule's Holm adjustment counts the other's tests.
- Those positions are dropped, not kept with no target. Kept, a lens that named "the" correctly there would count as
  stuck on it under rule 3.
- R4_03 writes the decision as `r4_decision`, giving for each band whether R2's rule passes, whether S6 passes, and
  whether the band is served. R3_06's own `decision` (R2's rule alone) is kept beside it.

R4_03's self-test checks the rule on synthetic files with known answers:
- each of rules 2, 3, 4 and 5 failing alone at a band;
- Holm over the two bands (raw p of 0.02 and 0.02 both fail);
- a missing readout keeping the family at two;
- S5 reported as not run;
- the gate:
  - a band that passes both is served;
  - where R1's whole gain is on function-word targets, both bands pass R2's rule, fail S6, and are not served.

## 6. Eligibility

R1 is the only candidate. It is eligible at both bands, and nothing is reference-only. R4 serves nothing itself: a
served band is a recommendation to the serving track of §2.

## 7. Power

These figures are `R4_power.py`'s (CPU, in `R4_power.json`). It recomputes R1's sign counts at 55 and 58 from R2's and
R3's evaluations, and checks that they equal both analyses' records. It then resamples R2's and R3's valid prompts
together within each condition (140, 140, 82, 120), at R4's expected valid counts. Those counts are twice R3's: 160,
160, 86 and 160, which is 566. It does so 4,000 times and applies **all five rules jointly**, as R4_03 does. One
difference: the intervals of rules 1 and 2 are the normal approximation of the stratified prompt bootstrap, not 2,000
resamples inside each draw.

**The record, weighted to R4's mix:**

| | page 56: R1 − naked eye | page 56: R1 − J++ | page 59: R1 − naked eye | page 59: R1 − J++ |
|---|---|---|---|---|
| R2's prompts alone | +0.050 | +0.086 | +0.049 | +0.080 |
| R3's prompts alone | +0.071 | +0.107 | +0.058 | +0.083 |
| R2 and R3 together | **+0.062** | +0.099 | **+0.055** | +0.081 |

R1's margin over the naked eye is smallest at chat-no-answer: +0.014 at page 56, and −0.007 at page 59. It is largest
at mid-reply: +0.123 and +0.105.

**The chance of passing,** in two modes:
- **point:** the R2 and R3 record taken as the truth;
- **predictive:** the record itself resampled first, so the chance allows for its own uncertainty.

R2's rule and S6 are applied to the same drawn prompts in each draw. Under S6, a drawn prompt whose every target is a
function word has no score and drops out of that draw. In R2 and R3 together, 40, 115, 63 and 120 of the 140, 140, 82
and 120 valid prompts keep a score under S6. So about 400 of R4's 566 are expected to count in S6.

| | page 56, point | page 56, predictive | page 59, point | page 59, predictive |
|---|---|---|---|---|
| **R2's rule:** rule 1 | 1.00 | 0.96 | 1.00 | 0.97 |
| rule 2 | 0.99 | 0.94 | 0.94 | 0.81 |
| rule 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| rule 4 (Holm over 2) | 1.00 | 1.00 | 1.00 | 1.00 |
| rule 5: R1 − J++ half | 1.00 | 0.99 | 0.99 | 0.93 |
| rule 5: R1 − naked eye half | 0.82 | 0.74 | 0.66 | 0.61 |
| all five | 0.82 | 0.73 | 0.64 | 0.53 |
| **S6:** rule 1 | 0.32 | 0.35 | 0.53 | 0.53 |
| rule 2 | **0.12** | **0.20** | 0.95 | 0.80 |
| rule 3 | 1.00 | 1.00 | 1.00 | 0.99 |
| rule 4 (Holm over 2) | 0.93 | 0.82 | 0.76 | 0.68 |
| rule 5: R1 − J++ half | 0.06 | 0.16 | 0.97 | 0.89 |
| rule 5: R1 − naked eye half | **0.02** | **0.07** | **0.04** | **0.11** |
| all five | 0.006 | 0.03 | 0.04 | 0.10 |
| **served (both)** | **0.006** | **0.03** | **0.04** | **0.09** |

Across both pages, the chance that **at least one page is served** is 0.05 (point) and 0.11 (predictive); that both
are served is 0.002 or less. By R2's rule alone it would be 0.93 and 0.865.

**The record without function-word targets** (S6, weighted to R4's mix):

| | page 56: R1 − naked eye | page 56: R1 − J++ | page 59: R1 − naked eye | page 59: R1 − J++ |
|---|---|---|---|---|
| R2's prompts alone | +0.004 | +0.002 | +0.008 | +0.067 |
| R3's prompts alone | +0.033 | +0.046 | +0.041 | +0.090 |
| R2 and R3 together | **+0.021** | +0.029 | **+0.027** | +0.079 |

At page 56 without function words, R1 is behind both baselines at chat-no-answer: −0.125, on the 40 prompts that keep a
score. That is why S6's rule 2 rarely passes there.

**Which rules bind:**
- **S6 decides.** On the record, R1's margin over the naked eye without function words is about half the 0.05
  minimum at both pages, and at page 56 S6's rule 2 binds as well.
- **By R2's rule alone, rule 5 decides,** through its R1 − naked eye half.
- **Rule 2 is R2's rule's second risk at page 59.** R1 is about level with the naked eye at chat-no-answer there, so a
  per-condition interval wholly below 0 is possible.
- **Rule 1 does not bind on its own.** It failed at code 55 in R2 (lower bound −0.004) on 199 prompts. At R4's n, the
  standard error of the pooled difference is about 0.014, so any estimate of 0.05 or more has a lower bound above 0.
- **Rules 3 and 4** hold in every draw: the stuck-word shares were 0.03–0.07, and the sign tests' p of 3×10⁻⁸ or smaller
  in R3 leave Holm's factor of 2 no purchase.

**The operating characteristic (what n buys).** The unconditional chance barely moves with n, because it is dominated by
where the true margin lies. What n does change is how sharply the rule separates true margins. Here is the chance of
passing the naked-eye half of rule 5 at R4's standard error, given the true margin. The standard error at 640 prompts
is 0.0139 at page 56 and 0.0118 at page 59; at 320 it is √2 times larger:

| true margin over the naked eye | page 56, 320 prompts | page 56, 640 prompts | page 59, 320 prompts | page 59, 640 prompts |
|---|---|---|---|---|
| +0.03 | 0.15 | 0.08 | 0.12 | 0.05 |
| +0.04 | 0.31 | 0.24 | 0.27 | 0.20 |
| +0.05 | 0.50 | 0.50 | 0.50 | 0.50 |
| +0.06 | 0.69 | 0.76 | 0.73 | 0.80 |
| +0.07 | 0.85 | 0.92 | 0.88 | 0.96 |

At 640 prompts, a pass is a firmer sign of a real 5-point margin than at 320: a true +0.03 passes 5–8% of the time,
not 12–15%.

The operating characteristic for S6 has the same shape. With about 400 of the 640 prompts keeping an S6 score, the
standard error of S6's pooled R1 − naked eye is about 0.014 at both pages, so a true +0.05 still passes its rule 5
about half the time.

**What the gate means for R4.** On the record, R4 is most likely to end with neither page served. It is likely to
confirm that R1's lead is in function words. That outcome is settled, and it closes the lead.

**Selection.** R3 read R1 at three layers, and R4 retests the two that passed there. Their R3 figures are the larger of
their two records, and some of that is the luck that picked them. The pooled record keeps R2's smaller figures, so it
does not simply carry R3's draw forward. It still sits on the side of the luck, so these chances are, if anything, high.

This is accepted. A test whose likeliest outcome is a clear "no" is still the honest way to settle a lead that came from
looking.

## 8. Reported for transparency (not used for the decision)

As R2 §8 and R3 §8, for R1 at 55 and 58:
- every rule's inputs;
- per-condition intervals;
- the sign tests' discordant counts;
- the exact-id variant;
- the content target in the top 5;
- mid-reply-content-next;
- the splits by position type;
- the constant-guess baseline and the function-word split;
- the counts of targets left out;
- the formatting share;
- the reading forward's agreement with the generated next token (R2 §5).

**Sensitivity analyses** rerun the whole rule, Holm included, and list any band whose verdict changes. For S1–S4 the
primary decision stands. **S6 is the exception:** it is part of the decision, as the gate of §5.

| | rerun without | reason |
|---|---|---|
| S1 | the 64 prompts about the model | R1's fit set had many self-questions |
| S2 | the 151 prompts of `R4_common.FIT_OVERLAP` | paraphrase, shared subject or shared answer with R1's fit set (§3.3) |
| S3 | the bare condition | raw text puts this instruct model off-distribution |
| S4 | positions with a formatting-like target | list numerals and code-fence tags may reward structure, not content |
| S6 (the gate, §5) | the positions whose target is a function word (`R2_common.FUNCTION_WORDS`) | R2's filter counts function words as content, and R1's lead on R2's and R3's prompts lies almost wholly in them (§1) |

S5 (J++'s exclusion list) is not run (§4.1). The report section also prints the function-word split (function-word
targets against the rest) for R1, J++ and the naked eye, pooled and per condition. In R2's report only the comparison
with J++ was printed.

**Pooled across R2, R3 and R4** (`R4_03_analyze.pooled_record`, reported only, never a decision): R1 − naked eye and
R1 − J++ at 55 and 58 over all three prompt sets, with the prompt bootstrap stratified by condition, and each set's
own point estimate, so that the reader sees the whole record at once.

## 9. Failure handling

As R2 §9 and R3 §9:
- **Stops before reading.** R4_02 and R4_00 stop before any reading on any gate of §0, §4.1 or §4.2.
- **Agreement below 0.95.** An agreement below 0.95 between the reading forward's own top-1 at 59 and the generated next
  token (R2 §5's check, report-only) is reported, and the rule is applied unchanged.
- **Partial runs.** A run that stops part-way is rerun from the start. Its partial data is never analysed, and it stays
  in its runs log.
- **R4_03 refuses,** before writing anything:
  - an input whose schema is not R4_02's;
  - self-test data written into `experiments/results`;
  - an input whose prompt counts or prompts sha256 are not the registered ones, or whose prompts `R4_01_check.json`
    does not pass;
  - an input that is not the output of R4_02's last finished run;
  - a missing or failed replication record, or one other than the record R4_02's run checked;
  - rule code (`R2_common`, `R2_03_analyze`, `R3_common`, `R3_06_analyze`) that differs from its pin, or an
    `R4_common.py` other than the run's;
  - `R4_03_analyze.py` or `R4_common.py` uncommitted, without `--allow-uncommitted`;
  - a Holm family other than the registered 2 tests.
- **The report says** at its top whether the run was forced or allowed uncommitted files, and how many runs finished.
- **R4_00 refuses** a batch size other than R3's.
- **`R4_00_replicate.py --compare` refuses** a re-read that is not the output of R4_00's last finished run.
- **R4_01 refuses** to write its record from a cached corpus (`--corpus-cache` is for `--fragment` only).

## 10. How to run

Run from the root of the checkout, with the main checkout's environment
(`/home/eric/bin/ai/theodolite2/.venv/bin/python`, written `python` below).

1. **CPU, done before registration:**
   - `CUDA_VISIBLE_DEVICES="" python -I experiments/scripts/R4_01_check.py` writes `R4_01_check.json`. It exits 1 on
     any failure, and passes now.
   - `... R4_power.py` writes `R4_power.json`.
   - `... R4_03_analyze.py --reproduce-r3` writes `R4_03_r3_reproduction.json`. R3's record of R1 is reproduced
     exactly, with 0 mismatches.
   - The self-tests, `... R4_02_eval.py --selftest --out <scratch> --record`, `... R4_00_replicate.py --selftest
     --record` and `... R4_03_analyze.py --selftest --record`, write `R4_selftests.json`. All pass.
   - `... R4_02_eval.py --check-only` passes, waiting only for the replication record.
2. **GPU, before registration,** with the owner's approval and the server stopped:
   - `python -I experiments/scripts/R4_00_replicate.py` writes `R4_00_reread.json`;
   - then on the CPU, `... R4_00_replicate.py --compare` writes `R4_00_replication.json` (§4.2).
3. **Register:** rerun the CPU checks of step 1 at the commit, cut `r4-r1-lens`, commit the files of §0, and add their
   sha256 table to §0.
4. **GPU,** with the server stopped: `python -I experiments/scripts/R4_02_eval.py` writes `R4_02_eval.json`, appends to
   `R4_02_runs.jsonl`, and keeps the captured states in `R4_02_states.pt` (gitignored, about 60 MB), so later readings
   need no GPU.
5. **CPU:** `CUDA_VISIBLE_DEVICES="" python -I experiments/scripts/R4_03_analyze.py` writes `R4_03_analysis.json` and
   `R4_03_report_section.md`, applying §5 exactly.

**Cost.** R3's own run sets the scale (`R3_05_eval.json`): 544 s of generation for its 320 prompts (greedy, full
forwards, 48-token replies), 28.5 s of reading and 14.7 s of readouts, plus about a minute to load the model.
Generation dominates and grows with the prompts:
- **R4_02** (640 prompts) should take about 20 minutes;
- **R4_00** (320 prompts regenerated, and R3's positions read again) about 11 minutes.

## 11. The owner's choices

- **The number of prompts:** 640, chosen by the owner on 27 October.
- **What a pass licenses** (28 October), given §1's function-word finding. Three options were put:
  - (a) R2's rule alone decides, and S6 is reported only;
  - (b) R2's rule decides, and S6 must also pass for a band to be served;
  - (c) R4 is not run.

  The owner chose **(b)**, registered in §1, §2 and §5. On the record, the chance of serving anything is about 0.05
  to 0.11 (§7).
- **Still to approve:**
  - the replication run (R4_00), a GPU run of about 11 minutes, before registration;
  - the evaluation (R4_02), a GPU run of about 20 minutes, after registration.

## 12. Before registration (checklist)

- [x] Copy R1's pinned lens to the stable path of §0, and check it by size and full sha256.
- [x] Write the 640 prompts (`R4_prompts.json`) to §3.1's composition, and `R4_common.py`: counts, composition, seeds,
      bands, the design, the pins and `FIT_OVERLAP`.
- [x] `R4_01_check.py`: 0 flags against every source of §3.2, 0 among the prompts.
- [x] `R4_power.py` and `R4_power.json`, all five rules jointly.
- [x] `R4_00_replicate.py`, `R4_02_eval.py` and `R4_03_analyze.py`, with their CPU self-tests passing, and R3's record
      of R1 reproduced exactly.
- [x] A second independent read of this file and the scripts against R2_PREREG and R3_PREREG, its findings answered as
      amendments (A18–A26).
- [x] The owner's choice of §11 (what a pass licenses): the S6 gate, registered in §1, §2 and §5, applied by R4_03 and
      simulated by R4_power.
- [x] The replication record, run and passed (an owner-approved GPU run on 28 October), after every amendment had
      landed: 100% on every readout, every position found again, R1's record reproduced exactly (§4.2).
- [ ] The CPU checks rerun at the registration commit, and the registration commit on `r4-r1-lens` with §0's sha256
      table.

## 13. Amendments before any data

No R4 prompt has been read by the model, and no forward pass, generation or readout has been run, so none of these is a
deviation.

**After the first independent read** of the draft against R2_PREREG, R3_PREREG, their analysis files and R3's code:

| | change | reason |
|---|---|---|
| A1 | §0 registers the files R4 imports or reads unchanged, R2_01_check.json, the replication's inputs, the backend files, the model snapshot and library versions; R4_02 has its own commit gate and does not import R3_06. | The first draft hashed only R4's own files, leaving the rule's code unpinned (R2's A8). `R3_gpu.require_committed` refuses today's backend. R3_06 hides the GPU when imported. |
| A2 | The replication becomes a separate pre-registration record over all 320 R3 prompts: teacher-forced at 99% per readout, the naked eye at 59 included; regenerated targets found again at 95%; R3's verdicts reproduced (§4.2). | 16 prompts and a 95% bar were far looser than R3's own re-read, which matched every pair. R2 §5's check, which the draft cited, is a report-only check of a different quantity. The gate ignored the targets, named the wrong draw, and could not rebuild positions from the seed alone. Run as part of R4_02, a failure would have forced a deviation. |
| A3 | §10's reproduction of R3 uses R3's seed and compares only what R3 recorded for R1. | R4's seed and Holm family differ from R3's, and R3_06 has no sensitivity verdicts for a reference lens. |
| A4 | §5 names every `Design` field, `precedence=("r1",)` among them; R4_03 keeps R3_06's refusals and writes its own report; S5 is registered as not run. | `decide()` serves only lenses in `precedence`. `analyze` alone skips R3_06's input checks, and `report_md` is R3's. A non-R2 design otherwise attempts S5. |
| A5 | §7 relabels the pass chances by half of rule 5, reweights R2 to R3's condition counts, says rule 2 is not negligible at chat-no-answer, code 58, corrects why rule 1 does not bind, and adds the operating characteristic. | The draft's "passes rule 5" covered the naked-eye half only. R2's pooled figure carried R2's composition. Rule 1 did fail at code 55 in R2. The unconditional chance is the wrong measure for choosing n. |
| A6 | §3.1 fixes the composition to R3's exact counts, and says no per-prompt result is consulted while writing. | The draft left counts open ("about a sixth" `format_likely`, against R3's 19 of 80), and the writer knows which conditions favour R1. |
| A7 | §3.3 also flags prompts whose expected answer is an R1 fit subject, and carries R3's two stated gaps. | R2's rule flagged shared answers; for a next-word metric the answer is what counts. |
| A8 | §2 adds R3 §7.2's missing serving conditions, says a pass supersedes R2's withholding at that band only, keeps the lens's other readings on J++, names the filtered top-1 as what is tested, and corrects "58 other layers" to 57. | The serving scope was incomplete, and J++ covers code 0–58 (page 60 is the model's own output). |
| A9 | §0 says the pinned lens lives only in a worktree, copies it to a stable path, and checks it by full sha256. | Removing the worktree would delete the only pinned copy; `full=False` accepts a file on size alone. |
| A10 | §4.3 states draw 0 = content_next, draw 1 = any. | The draft listed "any" first, against R2 §4.3 and `R3_05_eval.draw_mid_positions`. |
| A11 | §0 and §10 name the branch, the exact commands, the self-test and reproduction records, a route for stopping for an amendment, and this section. | Every R2 and R3 registration has them. |
| A12 | §10's cost estimate is lowered. | (Superseded by A15.) |

**While the prompts and scripts were written:**

| | change | reason |
|---|---|---|
| A13 | 640 prompts, R3's composition doubled (§3.1, §8's S1 count); §7's figures by `R4_power.py`, all five rules jointly, in point and predictive modes, replacing the draft's normal approximation. | The owner's choice on 27 October, for the sharper operating characteristic. The approximation was registered to be superseded. |
| A14 | The freshness corpus covers every worktree under `.claude/worktrees` as well as the main checkout, deduplicated by content, and R3's prompts; the planted check includes an R3 prompt and a new fresh string. A cross-condition duplicate screen is added beyond the rule (§3.2). | R3's check read two checkouts. R3's own planted "fresh" string is now in R3_01_check.py, so it is no longer fresh. Writers working apart produced cross-condition duplicates that the rule alone did not all catch (21 in this round; see A19 for the rest). |
| A15 | §10's cost estimate is corrected: about 20 minutes for R4_02 and 11 for R4_00. | A12 scaled by readouts. In R3's run, generation took 544 of about 590 seconds, and generation grows with the prompts, not the readouts. |
| A16 | §0 lists the R3 scripts R4 imports (`R3_05_eval`, `R3_gpu`, `R3_fitlib`, `R3_lrp`), `R1_common.py`, R2's and R3's evaluation and analysis files and `R3_sources.json` among the pinned files, and pins the snapshot by its files' blob names; R3's cache is named as a gitignored dependency in R3's worktree. | R4_02 reuses R3_05's backend rather than copying it, and the freshness check and power script read those files. |
| A17 | The replication record keeps the sha256 of the R4 scripts it ran, and R4_02 requires them unchanged (§4.2). | A replication made with other code would not vouch for the code that reads R4's prompts. |

**After the second independent read** of the draft, the scripts and the prompts:

| | change | reason |
|---|---|---|
| A18 | §1 states that R2's filter counts function words as content and that R1's lead on R2's and R3's prompts lies almost wholly in function-word targets, with the figures; the question and the lens card say "the next word (R2's filter; function words count)"; S6 (without function-word targets) is registered as a sensitivity analysis; the report prints the function-word split for all three readouts; §7 gives S6's chance; §11 puts to the owner whether S6 must also pass. | 304 of R3's 678 usable targets are function words. Without them, R1 − naked eye is +0.033 and +0.041 on R3's prompts and about 0 on R2's, and R1 fails rule 5 at both pages. The draft called the target "the next content word", which it is not. |
| A19 | 30 further replacements (19 for the second read's groups, 11 for an answer-level audit of all 640), within and across conditions; §3.2 lists all 46 replaced ids and says what is left; its earlier "the pairs that remain share only a theme", which was false, is corrected. | The second read found 15 groups the string check and the word screen had missed, among them the tomb opened in 1922 in two chat-no-answer prompts, "tokens" in three conditions, and "do you learn from our chats" in four prompts. The audit then found 6 more duplicate groups and 5 borderline ones leaning that way. |
| A20 | R4_03 refuses to decide unless the rule's code has its pinned sha256, `R4_common.py` is the run's, the replication record is the run's, and its own code is committed; it records them in the analysis. | Nothing enforced the pins when the rule ran. |
| A21 | The replication record joins the registered files; R4_02 checks the sha256 chain from R4_00's runs log through the re-read to the record, and records the record's sha256 for R4_03; the record keeps the scripts that compared as well as those that read, and passes only if they are the same. | A stale or rewritten record would otherwise pass R4_02's gate. |
| A22 | R4_00 refuses a batch size other than R3's, and (a) requires every rebuilt input to read the token R3 read. | "Batched as R3 batched" was only a default, and the rebuild was checked by token count alone. |
| A23 | `backend/__init__.py` and `backend/label_gate.py` are pinned; the other libraries, the GPU and driver, and the code the GPU process loads are recorded at R4_00 and required equal at R4_02. | R3 pinned the whole backend tree; R4 pinned two files, and nothing caught a change between R4_00 and R4_02. |
| A24 | FIT_OVERLAP: the replaced prompts' entries removed and the replacements judged; entries added for close relatives of listed prompts (hiking boots, a short verse, learning Python); the saxophone's "violin family" entry dropped as a stretch. §3.3's count of R3's list corrected to 83. | Consistency of the generous rule. |
| A25 | R4_01 refuses `--corpus-cache` outside `--fragment`; the report states deviations and the number of finished runs at its top; R4_02 keeps its captured states; the self-test no longer claims batch invariance; §0's account of the override flags matches the code; §3.1 says plainly what the coordinator knew. | Second read, low findings L6–L12. |
| A26 | §0, §4.2 and §9 say that every amendment must land before R4_00 runs, since the record pins `R4_common.py` (with `FIT_OVERLAP`) and the R4 scripts. | Sequencing. |

**The owner's choice:**

| | change | reason |
|---|---|---|
| A27 | A band is served only if R1 passes R2's rule and S6 there (§1, §2, §5); R4_03 writes the decision as `r4_decision` and leads its report with it; R4_power simulates R2's rule and S6 on the same draws and reports the chance of serving (§7). | The owner's choice of 28 October 2026, option (b) of §11: the test should ask whether R1 names substantive next words better, not only "the" and "of". |
| A28 | S6 drops the positions whose target is a function word, rather than keeping them with no target. | Kept with no target, a lens that named "the" correctly there was counted as stuck on it under rule 3. In the simulation, S6's rule 3 at page 59 then passed only about 0.54 of the time; dropped, it passes 0.99 or more. The means are the same either way. |
