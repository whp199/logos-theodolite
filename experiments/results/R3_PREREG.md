# R3: J++'s recipe fitted on the instruct checkpoint (pre-registration)

*Written 21 October 2026, before any R3 data existed. No R3 GPU job has been run: not the dry run on the 12B model,
not the pilot, the router, the fit, the weights, the evaluation or the 4-bit J-lens arm. What has been run is on the
CPU only: the self-tests, the positive control against jpp_lens (§2.3), the reproduction of R2's decisions (§7.4) and
Q0, the pullbacks through lens files that already existed (§8.5). R3 follows R1 and R2. Branch `it-jpp-refit`, cut
from `main` at `d5405ee`, with R2's two experiment commits (`cea0178`, `3f5941c`) picked onto it. Scripts:
`experiments/scripts/R3_*.py`. Prompts: `experiments/scripts/R3_prompts.json`. Sources: `experiments/scripts/
R3_sources.json`. Results: `experiments/results/R3_*`.*

Layers are given as code layers (0–59, as in the scripts and in R1 and R2) and, where a reader of the pages needs
them, as page layers (1–60): code 30–58 are page layers 31–59, so the band 30–39 is pages 31–40 and code 58 is page 59.

## 0. Integrity

**Registered files.** The table below gives the sha256 of every file registered with this document, as committed with
it. Every R3 GPU job refuses to start while any of `experiments/scripts/R3_*.py`, this file, `R3_prompts.json` or
`R3_sources.json` is untracked or differs from HEAD (`R3_gpu.require_committed`); `--allow-uncommitted` overrides this,
as a recorded deviation. The dry run, the pilot, the router, the shards, the weights and Q1 run inside one frame
(`R3_gpu.gpu_job`) that records HEAD, the job's memory cap and the card's memory, and appends every run (started,
finished or failed, with the error) to `experiments/results/R3_gpu_runs.jsonl`. The evaluation records HEAD in its
output and appends every run to `R3_05_runs.jsonl`. The pilot, the router, the weights and the evaluation refuse to
overwrite their output without `--force`; the shards and Q1 resume from their own checkpoints. Every run is reported
in the write-up.

**The evaluation runtime** is the instrument's backend at commit `d5405ee964e0ac1732babd8e69fa9271de555189` (`main`
when the branch was cut; the `backend/` tree `4ae7efe0…`), which `it-jpp-refit` does not change. R3 reads through two of
its modules: `backend/theodolite_math.py` (`jpp_readout_logits`, `unembed_gemma4`, `validate_and_cast_lens`) and
`backend/engine_runtime.py` (the 4-bit dequantising linear and the low-VRAM loader), with torch 2.14.1 and
transformers 5.20.0 from the shared environment. R2's runtime was older: `it-second-reading` is 8 commits behind
`main`.

**Any change made after data exists is a deviation.** It is reported as one, and the result under the rule written
here is reported beside it. "Data" here means any output of an R3 GPU job: the dry run, the pilot, the router, the
shards, the lens files, the readings.

| file | sha256 |
|---|---|
| *R3's scripts* | |
| `experiments/scripts/R3_00_pilot.py` | `dbd355a135663522d1b9ebca7dff9633127187273cc0eddc1698ff30a5a26fa9` |
| `experiments/scripts/R3_01_check.py` | `43eb815ca431ac84a26fc5acadc5b00709cd56d64d92f3c52519adbd72023082` |
| `experiments/scripts/R3_02_router.py` | `f0696be216fbd46e1163ffba05cdf2bd3a541644bfa630fcf04166806802c01b` |
| `experiments/scripts/R3_03_fit.py` | `d6ec212dd58079b43a262bf722efd573aa571b48203b40df9755a8e986a70d2d` |
| `experiments/scripts/R3_04_weights.py` | `552f01311de81b8085c3f71bdffff77f827bc4f29dc325c241587f705dd3edb3` |
| `experiments/scripts/R3_05_eval.py` | `2d3a2c7e0124ad61012e2898ed920a389f373aafb89b3ff2e26dc02c420af764` |
| `experiments/scripts/R3_06_analyze.py` | `faad671579586983e660af2c6999620dc98ed8f28fbe97d874538271ff5a1f31` |
| `experiments/scripts/R3_Q0_pullbacks.py` | `c0f077bc7138286acb0cbd241de309ae6c34e8b6f80d6e478423f49f44fb954c` |
| `experiments/scripts/R3_Q1_jlens.py` | `76102b85ae858991cb5d50bc2abc1c1b743e4732a28fc6bfe171f9d156b1e467` |
| `experiments/scripts/R3_common.py` | `90df9a9b349b0a8fbf3159e2009f85273558a5ea849b968725d126b9419b3e27` |
| `experiments/scripts/R3_fetch.py` | `5e702f8dc746ce2fdf889d64de4e1608fd0390cd384b5f7e3ed887e6febaffba` |
| `experiments/scripts/R3_fitlib.py` | `6d1ea02986d0f47f1b6ad33522eeed36f0e3bf66e2db67fdefa22c81bff9bbc1` |
| `experiments/scripts/R3_gpu.py` | `aeb2ad44c4c573e4acee3959aec51a1a7d4852379340142a85a8d21494ee94a4` |
| `experiments/scripts/R3_lrp.py` | `18d67c6b4552ca56329fb2ce352d8eade88f3131ebabeba8af4d58386b6bac4a` |
| `experiments/scripts/R3_parity.py` | `cc7d1e2fe003e3bec8dd7030a8fc1808e1a0ae0414ed1688a0f120bdc8465763` |
| `experiments/scripts/R3_sources_check.py` | `a266fe54d1cd6f2b70badd70d2fa7cff075872fdc87cce73f566e661ed098315` |
| *R3's prompts and sources* | |
| `experiments/scripts/R3_prompts.json` | `e3cae4b1becbd1fd4ac120e69e9c06dd2baf3dc6b0b4ae067d11b019957f4329` |
| `experiments/scripts/R3_sources.json` | `39f8340a03008ecda0b528ffa5208a3b4d267b9239fb376f8c43e39780b667d6` |
| *the check the evaluation requires (§5.1)* | |
| `experiments/results/R3_01_check.json` | `9bb609af34a83a48a9f763943fc1b6b04f8daca6a7713f88484c8eb5ae20077a` |
| *R2's files R3 imports or reads, unchanged* | |
| `experiments/scripts/R2_common.py` | `a3fe4ab44edbce3eeba01efa1428c82b7abb5f8b0555d7485fb2d008d78b762d` |
| `experiments/scripts/R2_03_analyze.py` | `9293011e96b3a132a915b306b2835846cc284e073fa5644124f20048cc377ecd` |
| `experiments/scripts/R2_prompts.json` | `623a7fb73e6a082335653bb02c8e22d7e6d731ef343ff299af57709a772d01fd` |
| `experiments/results/R2_01_check.json` | `9baad05e2d0057b2f7621fb389400cd5c24e119151b292648ca81343a8181581` |
| `experiments/results/R2_02_eval.json` | `6e277f60987c88a70f85355e2e749aac8cfa18f5480a5b044befedc3626f9c70` |
| `experiments/results/R2_03_analysis.json` | `94a632527ce6d1f356c836272ef5e1c32d8bef549cb62d6a8136dae83a22a75e` |
| *the CPU records made before this registration (rewritten whenever a check is rerun)* | |
| `experiments/results/R3_parity.json` | `70ecfbac988b0ea06db4574874f3466ad0801832f309f0314e1e684f4d8ba13c` |
| `experiments/results/R3_06_r2_reproduction.json` | `1fd81d6bb74b3456afc749628b19b2f1ed3f760d8db0f51cb7eae160b7dec1b6` |
| `experiments/results/R3_Q0_pullbacks.json` | `b83f0c9cda4533661f391236eea4b1730410cc0fea1b43f55d9fcd37bed82d17` |
| `experiments/results/R3_selftests.json` | `0dc618e854bd84f25517f75089fbf3f4e3201d5aa437a9778e1973f19f71b15f` |
| *the evaluation runtime at `d5405ee` (unchanged on this branch)* | |
| `backend/theodolite_math.py` | `0bba67755da1bfaf313df46dd8453280a368dfaf72d209e4b5327c00f0c80c49` |
| `backend/engine_runtime.py` | `0a7e071ba782cdb7a70eaff961b6ad1ce371c766ccbfbee9aa61979a5672ef06` |

## 1. Questions and hypotheses

The instrument reads `google/gemma-4-31B-it-qat-w4a16-ct`, the 4-bit instruct checkpoint, through J++, a lens fitted
on the base model `google/gemma-4-31B` (its provenance, read from the shipped file: 64 WikiText prompts of 128
word-pieces, source layers 0–58, `lrp_mode` `rlens`). R1 fitted a plain same-position average Jacobian on the
instruct checkpoint; R2 tested it and the Miru community J-lens, filtered, on fresh prompts, and served neither (R1 at
code 58 failed only rule 5: +0.046 over the naked eye against 0.05 required). R1 §2.4 tested neither J++'s recipe nor
its cluster weighting or token filter, and R1 §6 and erratum E50 say that fine-tuning (base against instruct) and
quantisation (bf16 against W4A16) cannot be told apart from R1's data. **No lens fitted with J++'s own recipe on any
instruct checkpoint existed.**

R3 fits J++'s published recipe (jpp_lens at commit `d7efc0de`) on the served checkpoint, with the same selection rule
and the same code as far as §2 allows, so that the main change is the model. It holds the result to R2's rules
unchanged, on 320 new prompts. Beside that it separates, for plain J-lenses and in direction only, fine-tuning from
quantisation, through the Miru lens (bf16 instruct).

### 1.1 Hypotheses

- **H1 (serving).** J++'s recipe fitted on the W4A16 instruct model beats filtered base J++ and the filtered naked eye
  under R2's five rules (§7) at one or more of the bands 30–39, 40–49, 50, 51–54, 55, 56–57, 58. Two candidates are
  tested: **jpp-it**, the full recipe (eight neighbourhood lenses per layer, weighed by J++'s fitter), and **rl-it**,
  the pooled R-lens Jacobian (the same rows, no clusters, no weights).
- **H2 (recipe or checkpoint).** How far jpp-it agrees with base J++, in the quiet dimensions (q10) and the loud ones,
  measured against R1's yardsticks (§8.1). If the recipe, more than the checkpoint, sets the lens, jpp-it agrees with
  base J++ about as well as two methods on one model do; if the checkpoint dominates, it agrees about as badly as R1's
  instruct covectors did.
- **H3 (concentration).** Whether J++'s recipe itself concentrates a lens's energy in about 100 output directions.
  R1 measured 76–97% of the shipped Gemma J++'s energy in its top 100 output directions (R1_IT_LENS_REPORT.md:45–48,
  354). No explicit low-rank step exists in the code at `d7efc0de` (no SVD, truncation or rank penalty); whether the
  recipe yields concentration is H3. The J++ post (Appendix G) reports its Qwen maps as numerically full rank.
- **H4 (fidelity).** On R1/E3's held-out positions (R1_10's 4,808), whether jpp-it and rl-it reproduce the model's own
  final readout better than base J++: top-1 and top-10 agreement, KL and calibrated KL at code 40, 45, 50, 55, 58.
- **H5 (Jacobian filtering on instruct).** Whether the weighted experts (jpp-it) beat the pooled R-lens (rl-it) on
  the instruct model, as J++'s post found on Qwen (Appendix H, Table 4: 55.2 against 43.5 pooled).
- **H6 (fine-tuning against quantisation).** For plain J-lenses, how much of the base-to-served difference is
  fine-tuning and how much quantisation: Miru (bf16 instruct) against Neuronpedia's JLens (bf16 base) gives the
  fine-tuning axis; a plain J-lens fitted here on the 4-bit model (Q1) against Miru gives the quantisation axis.

Only H1 decides anything (§7). H2–H6 are secondary readings, reported whatever they show.

### 1.2 What would refute

- **H1** is refuted at a band when no candidate passes there. If no candidate passes at any band, refitting J++'s
  recipe on the served checkpoint did not give a second reading that beats base J++ and the naked eye by R2's
  standard; the base fit is then not shown to be what holds the second reading back. It does not show that no
  instruct lens could pass: a true effect below 0.05, or one that fails Holm's correction over 14 tests, fails too
  (§7.3). A pass at a band refutes, for that band and these prompt kinds, the claim that base J++ reads the instruct
  model as well as an instruct fit does.
- **H2.** "The recipe sets the lens" is refuted at a band if the upper end of the 95% interval (over the 98 token-pair
  directions) of jpp-it against base J++ in q10 is at or below R1's instruct covectors against base J++ (Q0: 0.27 /
  0.29 / 0.29 at code 30–39 / 40–49 / 50–58, §8.1). "The checkpoint dominates" is refuted if the lower end is at or
  above J++ against JLens (Q0: 0.77 / 0.68 / 0.66). Otherwise the reading is "both", and is reported as such.
- **H3.** "The recipe yields concentration" is refuted at a layer if jpp-it's own top-100 share is below 0.76, the
  lowest share R1 reported for the shipped file at the layers it measured; it is supported if the share is at least
  the shipped file's own share at the same layer, measured the same way (§8.2). rl-it (no weighting) is reported beside it, to say which step concentrates.
- **H4.** "jpp-it reads the instruct model more faithfully than base J++" is refuted at a layer if its top-1 agreement
  on all 4,808 positions is not above J++'s. R3_05 keeps these figures per class, not per sequence, so no interval is
  registered: like R1 §4.2's, they are point values over a fixed set of positions.
- **H5.** "Jacobian filtering helps on instruct" is refuted if jpp-it − rl-it, pooled, has a 95% interval that does
  not lie above 0 at every band where either is above the naked eye, and J++'s own held-out recall@10 (§8.3) is not
  higher for jpp-it than for rl-it at the four recall layers.
- **H6.** Per band (code 30–39 / 40–49 / 50–58, in q10), quantisation is the smaller axis if Q1 against Miru is at
  least R1's same-model, different-data yardstick (0.77 / 0.78 / 0.82, R1_04 `cos_chat_prose`); it is the larger axis
  if Q1 against Miru is at or below Miru against JLens (Q0: 0.37 / 0.39 / 0.40); between the two, both count. While Q1 runs on J++'s 64 prompts rather than Miru's own (§3.3),
  a low agreement cannot be laid on quantisation alone: the prompts changed too.

## 2. The fit recipe

### 2.1 J++'s recipe at `d7efc0de`, as R3 runs it

Read from `github.com/safety-research/jpp_lens` at `d7efc0de372e20589e899188c0cfe8434acc63ec` (17 October 2026), pinned
file by file in `R3_sources.json`, and run in its own environment (`experiments/R3_cache/env`) for the positive
control.

- **Fit prompts.** The first 64 of the 1,000 router records (§3.1), BOS forced, truncated to 128 word-pieces.
- **Kept positions.** 16 to T−2 (the first 16 and the last are dropped).
- **Rows.** For every output dimension of block 59's raw output, a one-hot cotangent set at every kept position at
  once; the gradient at a source position p is then the sum over kept p′ ≥ p of ∂h₅₉[p′]/∂hₗ[p]. Rows are exact, not
  probe estimates: ⌈5376 / B⌉ backward passes per prompt, B rows per pass.
- **The R-lens rules** (`lrp_mode` `rlens`) in the backward pass: the LN rule on every RMSNorm outside self-attention
  (the scale `(mean(x²)+ε)^(-1/2)` detached), the identity rule on GELU-tanh (`z · gate(z).detach()`), the half rule
  on the gated product (`0.5p + 0.5p.detach()`); attention, and the q/k/v norms inside it, keep plain gradients.
- **Router.** Per layer: the kept positions of all 1,000 records, unit-normalised, centred, projected on 64 principal
  components, then k-means++ with 8 clusters, seed 0, up to 100 iterations.
- **Experts.** Each of the 8 neighbourhood lenses at a layer is the position-weighted mean of its cluster's rows. An
  expert with fewer than 50 kept positions falls back to the layer's pooled Jacobian.
- **Weights.** Per layer, signed weights of unit L2 norm over the 8 experts, fitted by jpp_lens's
  `ExpertWeightingFitter`: Adam, learning rate 0.05, 200 steps from 3 starts, a sigmoid surrogate of pass@10 at
  temperature 0.1, top 3, keeping the iterate with the best exact pair-level macro pass@10 on the fitting split
  (checked every 5 steps). The fitting split is 20% of each eval's scoreable items, seed 0.
- **Items.** J++'s six readout evals (association 102, multihop 93, multilingual 107, order-ops 55, poetry 98, typo 96
  items). Order-ops is outside J++'s macro. An item is kept only if the model's greedy next word-piece matches its
  target.
- **Readout filter.** A word-piece whose decoded text has no letter or digit is left out of the readout ranking, except
  the candidate ids of the six evals' items.
- **The lens.** Jₗ = Σₑ wₑ Jₗ,ₑ, read as the instrument reads J++: `h_J = h @ J.T`.

### 2.2 Deviations

Each is a place where R3 departs from J++'s code at `d7efc0de`, from the shipped Gemma J++, or from the design R3 was
planned from.

| | deviation | why |
|---|---|---|
| D1 | The model is the served W4A16 QAT instruct checkpoint (snapshot `52f3f65`), run through a recompute-in-backward 4-bit linear (`R3_gpu.py`): the forward is the instrument's own dequantising forward; the backward dequantises again and saves only the packed weights and their scales. J++ was fitted in bf16 on the base model. | The point of R3. The 4-bit weights do not fit dequantised. |
| D2 | Source layers are code 30–58 only (pages 31–59); the backward graph starts at block 30 (J++'s `start_graph_at` = the lowest source layer); the target is block 59's raw output. The shipped J++ has layers 0–58. | RAM: 8 experts × 29 layers × 5376² × 4 bytes = 26.8 GB per shard. Nothing is served below code 30 (§7). |
| D3 | The rows are batched by `torch.autograd.grad(..., is_grads_batched=True)` or by a replicated batch of B copies, whichever the pilot picks (§10.2). The estimator is unchanged. | Speed; the 4-bit linear needs a batched backward. Both modes equal jpp_lens's `fit_shard` in the positive control (max difference 0 and 7.6e-6). |
| D4 | The router's PCA comes from a covariance streamed in fp64 over two passes of the 1,000 records, not jpp_lens's `fit_pca` (fp32 eigh on stored points). | Memory: the points of 1,000 records at 29 layers do not fit. In the positive control the components agree to \|cos\| 0.9999997 or better and every pool position gets the same expert. |
| D5 | Shard sums are written in fp32, one file per layer per shard, checkpointed and resumable; the merge reads them one layer at a time (memory-mapped). The shard files are deleted only after every merged file has been read back and its sha256 checked. | 61 GiB of RAM cannot hold three 26.8 GB sums. The streamed merge equals the in-memory one (`R3_03_fit.py --selftest`). |
| D6 | Addition: two half-lenses, A from the even shards and B from the odd ones, for jpp-it and rl-it. | A split-half ceiling for H2 and H6. |
| D7 | J++'s readout filter is used where the recipe uses it (the weight fit, J++'s recall). The band evaluation uses R2's filter for every readout, the filter the instrument would serve. Sensitivity analysis S5 swaps them. They differ: of J++'s 8,457 ids with no letter or digit, 18 are spared as item candidates and 8,439 are left out; 6,338 ids R2 counts as non-content J++ keeps, and 606 R2-content ids J++ leaves out (`R3_01_check.json`). | The test is of what would be shown. |
| D8 | Item correctness is graded on the served instruct model (bare item prompt, BOS, no chat template, one greedy step, jpp_lens's `next_token_matches_target`); association and typo, which have no question, are kept whole, as jpp_lens keeps their empty cells. | The items must be the model's own successes, and this model is the instruct one. |
| D9 | The readout is the instrument's unembed with the 30.0 softcap (top-k unchanged). | The instrument's readout. |
| D10 | Fitting is in fp32. The lens files are bf16 in the shipped layout `{"J": {l: [5376, 5376]}}`, read `h @ J.T`, with provenance `method` `jpp`, `lrp_mode` `rlens`, `hf_model_name` `google/gemma-4-31B-it-qat-w4a16-ct`. | Precision of the fit; the instrument's format. |
| D11 | Expert weights are fitted at every fitted layer (29), each on its own objective. | How the shipped file set its weights at layers other than its README's readout layers (8, 15, 22, 30, 38, 45, 52) is not recorded. |
| D12 | The code's sum over kept targets p′ ≥ p is used, not the mean over t of the post's eq. 2a. | R3 follows the code. |
| D13 | The two old-code fields of the shipped file's config (`num_s2_fitting_prompts` 10, `mask_non_semantic_fit_positions` false) are not reproduced. | R3 runs the code at `d7efc0de`, not the older code the shipped file came from. |
| D14 | The router is per layer, seed 0, E = 8, PCA 64, jpp_lens's own k-means++, up to 100 iterations, tolerance 1e-6, on the 110,804 kept positions of the 1,000 records. | J++'s recipe constants (`jpp_cli.py`), fixed here. |
| D15 | `CondenseConfig` defaults (pass_k 10, top_n 3, steps 200, learning rate 0.05, temperature 0.1, exact_every 5), fit fraction 0.2, split seed 0, minimum 50 kept positions per expert. | J++'s defaults, fixed here. |
| D16 | **The per-eval floor, fixed before grading.** An eval that keeps fewer than 10 correct items, or fewer than 2 fitting items, is dropped from the macro; the per-eval counts at each step are recorded in `R3_04_weights.json`. The primary macro leaves poetry out (multihop, multilingual, typo, association), as J++'s own Gemma lens did (koayon/jpp-lenses README: 1 of 98 poetry items passed its correctness filter); the full macro, poetry in, is fitted too (`jpp-it-full-macro`) and reported only. Fewer than 20 fitting (item, intermediate) pairs in the primary macro: the run stops for an amendment and writes no lens. | With `max(1, ⌊0.2n⌋)` fitting items per eval, one item could carry a fifth of the objective that sets the weights at every layer. |
| D17 | WikiText is read at revision `b08601e04326c79dfdd32d625aee71d232d685c3` (Miru's pin). J++'s loader pins no revision, so the shipped fit's revision is unknown. | A fixed corpus. The selection rule is J++'s. |
| D18 | R3 runs its own implementation, not jpp_lens's driver. `R3_lrp.py` reimplements the R-lens rules for Gemma 4's classes only and raises on any norm or MLP class it cannot patch; its rows equal jpp_lens's patcher bit for bit. `R3_fitlib.py` imports jpp_lens's pure functions (k-means, `normalize_center_project`, `candidate_token_ids`, the non-semantic scan, `ExpertWeightingFitter`, `combine`) and reimplements the rest (the batched-row estimator, the streamed PCA, the expert sums, the merge, the item code). | The 4-bit model and the streamed memory plan need their own loop. The positive control (§2.3) checks each piece. |
| D19 | The positive control's own conditions. PD1: jpp_lens's stages run as the library functions of `scripts/jpp_cli.py`, with a stand-in module named `pandas` (pandas is not installed) that raises on any use; no stage used it. `grade_model_correctness` and the correctness CSV need pandas and were not run, so R3's grading equals jpp_lens's by construction (both use `next_token_matches_target`), not by test. PD2: the tiny model forces smaller sizes (PCA 16, 96 router records, 32 fit prompts in 4 shards, 48 word-pieces, 8 rows per pass, a character tokenizer, made-up items in J++'s format); E = 8, skip 16, minimum 50, the rules, the split and the condense defaults are the registered ones. PD3: the weights are fitted by jpp_lens's own fitter, so the weight parity tests R3's inputs to it. | No download beyond the three approved sources is allowed; a 32-wide model cannot hold 64 components. |
| D20 | jpp_lens runs with the shared environment's versions, not its pins: torch 2.14.1 (pinned 2.13.0), transformers 5.20.0 (5.14.1), numpy 2.5.4 (2.5.1), huggingface_hub 2.2.0 (1.26.0), accelerate 1.15.0 (1.12.0); pytest matches. pandas, pyarrow and datasets are absent, so its command line and the pandas parts of `lens_evals` do not import. WikiText is read by `R3_fetch.py`'s own parquet reader; J++'s own loader, run on the cached rows, returns exactly the 1,000 records. | No download beyond the three approved sources (§3). With `--tests`, jpp_lens's own suite gives 83 passed, 0 failed; 17 test modules could not load, each only for want of pandas. |
| D21 | The evaluation loads the checkpoint itself (`R3_gpu.load_w4a16`) and calls the multimodal wrapper's `Gemma4Model` with an attention mask; R2 went through `TheodoliteRuntime` and `labkit.forward_resid`. Every readout still goes through the instrument's `jpp_readout_logits` and `unembed_gemma4` (§0). | R3 loads no runtime and no J++ on the GPU beside its own lenses. The replication on R2's prompts (§8.6) shows whether the path changes R2's numbers. |
| D22 | **Q1, the 4-bit J-lens arm** (`R3_Q1_jlens.py`, not a mode of R3_03): directional covectors for R1's 128 directions at code 30–58, not a full J-lens matrix; plain gradients; the sum over this and every later kept target (Miru's estimator); kept positions 16 to T−2; a mean over positions and a uniform mean over prompts; target block 59; the directions in batches of 32 (Miru's `dim_batch`); split halves by prompt parity. Not knowable from Miru's header, and set here: its last position is dropped (as J++ does), and its gradient mode is taken as plain only as far as "no LRP rule recorded". **Fallback:** Miru's prompts could not be rebuilt (§3.3), so Q1 runs on J++'s 64 fit prompts, and bf16 against W4A16 then also changes the prompts. | The directions are all Q0 compares; four passes per prompt instead of 5,376. |
| D23 | H4's positions are R1_10's 4,808, rebuilt on the CPU from E3's own files: 24 prose paragraphs and 24 chats (48 sequences, the longest 141 ids; the design said 40 chats). The list's sha256 is pinned (`fd596c3b…`) and was checked against `R1_10_evalfull.pt`'s own record; the states are regenerated on the GPU. J++, JLens and the naked eye must reproduce R1_10's top-1 within 0.01. | R1_10 stored summaries, not states. |
| D24 | The dry run uses the local bf16 Gemma 4 12B (`gemma4_unified`), with blocks 0–14 on the CPU (its weights do not fit under the job cap), source layers 24, 36 and 46 and target 47, a floor of 1 and 1, and 10 items per eval. `R3_lrp` accepts the 12B's norm and MLP classes only while their `forward` and `_norm` source is Gemma 4's; a changed copy raises. | A whole-pipeline run on a real Gemma 4 before touching the 31B. |
| D25 | The pilot's parity compares R3's batched rows for six output dimensions (R1's unit directions) with an independent single-direction backward from a graph rooted at block 0, in both batching modes. | The design's full rows projected on 6 directions would take a whole prompt's passes inside the 30-minute pilot. |
| D26 | J++'s item files are not vendored into `experiments/scripts/R3_jpp_items/`, as the design said; they stay in the gitignored cache with their sha256 and counts committed. The WikiText cache holds every row from 0 to the 1,000th record (4,597 rows), not only the records. | The repository does not redistribute third-party data (README.md:407–408). The rows between records let J++'s selection be proven and Miru's prompts be tried. |
| D27 | R3's integrity record is HEAD and the refusal of dirty registered files (§0); R2_02 also wrote every registered file's sha256 into its output. | The commit pins every file; §0's table gives the hashes. |
| D28 | R1's pinned lens (`80ce5609…`, 173,411,927 bytes) is read from the `it-second-reading` worktree. The main checkout's `R1_18_it_lens.pt` has the same size and the same three matrices but another sha256 (`37252f5e…`, an older provenance note) and is not used. | Only the pinned file is read. |
| D29 | R2's earlier held-out script (`heldout_eval.py`) no longer exists; its prompts are not in the freshness corpus. | It lived in another session's scratchpad. R2's prompts and every other source R2 used are in. |
| D30 | **No GPU budget is approved yet** (`R3_common.APPROVED_GPU_HOURS = None`, [NOT YET APPROVED]). The pilot writes "stop" until one is registered; the fit jobs refuse a pilot judged against another budget value. The pilot's projection is multiplied by a registered margin of 1.25. | The user decides the budget (§10.2). |
| D31 | The H2–H5 tables of `R3_06_analyze.py` are not written yet: they need the fitted lens files and R3_05's outputs. They are registered here in words (§8). Their code is to be committed before `R3_04_weights.py --run` writes any lens file; if it is not, every number it gives is labelled as analysed after the lens files existed. | The band evaluation, the sensitivity analyses and the reproduction of R2 are written and tested (§7.4). |
| D32 | Q0 was run before this registration, on lens files that already existed (§8.5). Its fine-tuning axis (Miru against JLens, 0.37 / 0.39 / 0.40 in q10) was known when H6's reading (§1.2) was written. | Q0 needs no GPU and checks the pullback code against R1. The reading thresholds come from R1's yardsticks and Q0's own rows, not from Q1, which has not run. |

### 2.3 Matched, not deviations (the positive control decided)

`R3_parity.py` (CPU) runs jpp_lens's own stages (`fit-router`, `fit-shard`, `merge-experts`, `pooled-lens`,
`fit-weights`) in jpp_lens's own environment, on a tiny Gemma 4 that keeps the real config's flags
(`hidden_size_per_layer_input` 0, `attention_k_eq_v` True, `gelu_pytorch_tanh`, the 30.0 softcap), and compares R3's
code with them. All 17 checks hold within 1e-5 × max(1, max |jpp|), most bit for bit (`R3_parity.json`): the router,
the expert sums in both batching modes and with both routers, the streamed merge with its fallback branch, the
items, the split, the exclusion list, the readout residuals, the weights and the combined lens. On the real
tokenizer, R3's encoding of all 1,000 records equals jpp_lens's and the instrument's.

Five places where R3 could have departed from J++ were set by the control to match it:

- **M1.** The weights are fitted, and the lens combined, from experts rounded to fp16, because jpp_lens's
  `fit-weights` stage reads the fp16 experts file. The pooled R-lens (rl-it) comes from the fp32 sums.
- **M2.** The weight surrogate sees softcapped logits (`HFLensModel.unembed` applies `final_logit_softcapping`).
- **M3.** The final norm is patched, as jpp_lens patches every RMSNorm outside self-attention. It is off the path of
  the block-59 target: the rules change no row through it (`R3_lrp.py --selftest`).
- **M4.** The exclusion list spares every candidate id of the six evals' items (`readout_eval_items.py:191–212`).
- **M5.** BOS is forced on raw text (`HFLensModel` `force_bos`); without it this tokenizer adds none.

## 3. Corpus

### 3.1 The fit corpus

- **Records.** J++'s selection: the first 1,000 rows of `Salesforce/wikitext`, `wikitext-103-raw-v1`, train split, at
  revision `b08601e0`, whose text is at least 600 characters once stripped, in stream order (the hub resolved that
  revision to exactly that commit). The cache holds rows 0–4,596, the 1,000th record being the last. Each
  record's row, length and sha256 and the sequence's sha256 are in `R3_sources.json`; no WikiText text is committed.
  Licence CC BY-SA 3.0 and GFDL, per the dataset card; not redistributed (NOTICE).
- **Fit prompts.** Records 0–63. With BOS and truncation to 128, 63 are 128 word-pieces long and record 29 is 125, so
  there are 7,101 kept positions. J++'s 7,104 = 64 × 111 counts Qwen's tokens.
- **Router.** All 1,000 records: 110,804 kept positions; the shortest record is 104 word-pieces.
- **Items.** J++'s six item files, inside the pinned checkout, with their sha256 and counts in `R3_sources.json`.
- **Not used:** R1's chat corpus as a second fit. The design listed it as optional and reported only; it is not part of
  R3.

So the data and the recipe are those of the shipped J++, with the deviations of §2, and only the checkpoint is meant to
change.

### 3.2 Disjointness

- **R3's prompts** (§4) have 0 flags against every source R2 used (R2's own 220 prompts among them), every cached
  WikiText row (the 1,000 records and the rows between them), every J++ item string (prompt, target, intermediates)
  and each other (§4.2).
- **R2's 220 prompts**, R3's replication set (§8.6), were checked against R3's fit sources, which R2 could not do
  (R2_PREREG.md §3.2): 0 flags over 340 query strings against 4,636 WikiText and item strings; the closest difflib
  ratio is 0.716 (`R3_01_check.json`, `r2_set_against_fit_sources`).

### 3.3 Miru's prompts

Miru's header records its selection (`concatenate_and_rechunk`, headings excluded, 200–2,000 characters, a budget of
1,000 prompts at revision `b08601e0`), a hash of the 1,000-prompt sequence (`dcb2a738…`) and of the 425 prompts it
processed (`f5797115…`), after which its convergence rule stopped it. `R3_01_check.py` tried 24 readings of that
description (paragraph packing or stream cutting, stripped or not, four separators), each hashed in 12 ways: 288
comparisons, no match. Every reading yields at most 862 prompts from the cached rows, fewer than the 1,000 the full
hash covers, and Miru Tracer's own code is not an approved source. **So Miru's prompts are not rebuilt**, and Q1 falls
back to J++'s 64 prompts (D22). Rebuilding them needs the user's approval for Miru Tracer's selection code at a pinned
commit, and probably more WikiText rows.

## 4. Prompts

### 4.1 Conditions

320 prompts are written for R3 (`R3_prompts.json`, sha256 `e3cae4b1…f4329`, R2's schema). All are distinct and their
topics are varied. Each condition has prompts about the model itself (10, 9, 5 and 8; 32 in all). Ids carry the prefix
`r3-`, so that they cannot be confused with R2's.

| condition | n | the text read | read position |
|---|---|---|---|
| chat-no-answer | 80 | The question in the chat template, with no system prompt and thinking off (empty thought channel). | Its last token, `<channel\|>`. |
| chat-begun | 80 | The question, then a begun answer that ends just before a content word. | The begun answer's last token. |
| bare | 80 | A plain-text opening with BOS and no template. | Its last token. |
| mid-reply | 80 | A new question. The model's greedy reply (thinking off, 48 new tokens) is generated in the run. | Seeded positions anywhere inside that reply (§5.3). |

The chat-begun answers are deliberately mixed: 20 factual, 14 opinion or advice, 11 creative, 9 about the model, 7
how-to or code, 5 empathy, 5 language or arithmetic and 9 list openings. 19 are marked `format_likely`.

These are 100 more prompts than R2 had (60/60/60/40).

### 4.2 Freshness check (`R3_01_check.py`, result in `R3_01_check.json`)

Every question, begun answer, question + begun answer, and bare text (480 strings) was compared, after NFKC
normalisation, case folding and removing punctuation, with `R2_common.freshness_check`, R2's rule unchanged (exact;
contained, 4 or more words; contains, a source of 5 or more words; similar, difflib 0.80 or more or word Jaccard 0.75
or more), against:

- every string literal in `experiments/scripts/*.py` and every string in `experiments/results/*.json`, of this
  checkout and the main one, R3's own files excluded (380 and 772 files); this includes R2's prompts and results and
  R1's fit corpus;
- R2's prompts (`R2_prompts.json`) and the N1B corpus (decoded from its token ids);
- every cached WikiText row (2,893 rows of three or more words: the 1,000 records and the rows between them, which hold
  the start of Miru's prompt stream);
- every J++ item string (1,743: prompts, targets and intermediates of all six evals);
- the other R3 prompts.

The corpus also flags planted copies of a WikiText record, a J++ item and an R2 prompt, and not a fresh sentence.
Matches inside third-party texts are recorded by source, item number and sha256, never by text.

The first draft had 84 flags on 26 prompts, and one pair of R3 prompts flagged each other. Several were common
questions already in the study's lists (N1B's and E3's prompts among them); one begun answer matched a J++ item word
exactly. Those 26 prompts were rewritten before any data existed, and one more was rewritten so that two prompts would
not share one answer. The final set has 0 flags against 23,724 distinct source strings (39,281 before duplicates were
removed), 0 among the R3 prompts, and the closest difflib ratio is 0.788.

### 4.3 What the check does not cover

- *Paraphrase and shared topics with a fit source.* The lexical check passes, but some prompts paraphrase one of R1's
  fit questions, or ask about something a WikiText record's article is about, or ask the same kind of question as a
  J++ eval. The list `R3_common.FIT_OVERLAP` was fixed from the prompt texts alone, before any data, in the same commit
  as the prompts. It names **83 prompts** (20 chat-no-answer, 21 chat-begun, 19 bare, 23 mid-reply), each with the
  source it overlaps: 32 a WikiText record's article (by record number), 27 J++'s items (by eval and item number), 45
  R1's fit set (some prompts more than one). The rule is generous on purpose (R3_common's comment); sensitivity
  analysis S2 reruns the rule without them, and S1 without the 32 prompts about the model.
- *Miru's fit passages.* Not rebuilt (§3.3). The cached rows hold the start of the WikiText stream Miru drew from, and
  R3's prompts have no flag against them; the rest of Miru's 425 passages is unchecked, as in R2.
- *The 40 /usr/share/doc chunks of R1's corpus* are in only as R1 stored them (token ids removed); R2 had the same gap.

## 5. Reading

### 5.1 Model, runtime and lenses

`R3_05_eval.py` loads the served checkpoint itself (D21), with the server stopped, and caps its own memory (§10.2).
Before the model it checks: the registered files are committed; there is no earlier output; `R3_prompts.json`
validates with R3's counts and its sha256 is the one a passing `R3_01_check.json` recorded; R2's filter has
`R2_01_check.json`'s class counts; J++'s exclusion list has `R3_01_check.json`'s count (8,439); every lens file has its
pinned size and sha256 (J++, JLens and Miru from R3_common, R1's lens from its pin, R3's own from
`R3_04_weights.json`); R1_10's positions rebuild with their pinned sha256; the pilot said "proceed".

| readout | layers (code) | path |
|---|---|---|
| J++ (base) | 30–58 | `jpp_readout_logits(h, J_l)`, the Neuronpedia fp16 file cast to bf16 |
| naked eye | 30–59 | `unembed_gemma4(h)`; at 59 it is the model's own output |
| jpp-it | 30–58 | `jpp_readout_logits(h, J_l)`, R3_04's bf16 file |
| rl-it | 30–58 | the same, R3_04's pooled R-lens file |
| jpp-it-full-macro | 30–58 | the same; reported only (sensitivity) |
| R1 (reference) | 50, 55, 58 | R1_18's matrices in bf16 |
| Miru (reference) | 30–58 | the file's fp16 matrices, read with `safetensors.safe_open`, cast to bf16 |

One lens layer is on the GPU at a time.

### 5.2 The model's own word, and decoding

As R2 §4.2: the model's own word at a position is the argmax of the final RMSNorm, W_U and the 30.0 softcap of
`resid_post[59]`; greedy continuations use the same rule through full forwards with no KV cache, right-padded; a
sequence stops after `<eos>`, `<turn|>` or `<tool_response|>`.

### 5.3 Read positions

As R2 §4.3, with R3's seed: one position (the last prompt token) for chat-no-answer, chat-begun and bare; for
mid-reply, two seeded draws without replacement from the greedy reply, each with its own stream
`numpy.random.default_rng([20261021, prompt index, draw])`: "any", up to 6 positions anywhere in the reply (the
decision condition), and "content_next", up to 3 positions whose next token is content (mid-reply-content-next,
reported only).

### 5.4 Readouts

At every read position `resid_post` is captured at code layers 30–59 (R2 captured 40–59). Per readout: the top-5 ids
and texts unfiltered, with R2's filter, and with J++'s exclusion list (for S5); the rank of the next word-piece
unfiltered and of the content target filtered; whether the unfiltered top-1 is content.

## 6. Targets and metrics

R2 §5–§6 unchanged: the next token and the content target (the first CONTENT token of the model's own greedy
continuation of up to 8 tokens, the search stopping at a stop id or any special token); no usable target when the
window has none or the target is a raw-text collapse word (`" own"`, `" same"`); formatting-like targets counted and
left in. **Primary metric:** the filtered top-1 is the same word as the usable content target (`R2_common.word_key`:
NFKC, surrounding whitespace stripped, case folded). **Stuck-word share** per readout, layer and condition, by word,
leaving out positions whose top word is their own target. **Formatting share:** the share of positions whose
unfiltered top-1 is non-content.

## 7. Decision rules

### 7.1 R2's rule, unchanged

Bands (code): 30–39, 40–49, 50, 51–54, 55, 56–57, 58 (pages 31–40, 41–50, 51, 52–55, 56, 57–58, 59). Prompt score,
condition means, pooled mean, and 95% percentile intervals from 2,000 prompt-level bootstrap resamples within
conditions, the same indices for every readout, layer and band, seed `numpy.random.default_rng([20261022, condition
index])`, all as R2 §7. A candidate passes at a band when all five hold:

1. **Beats both, pooled.** The lower bounds of the pooled intervals of candidate − J++ and candidate − naked eye are
   both above 0.
2. **Not worse in any condition.** In none of the four decision conditions is the upper bound of either
   per-condition interval below 0. A decision condition with no valid prompt, or a readout missing, is a failure.
3. **No stuck words.** The stuck-word share is at most 0.25 in each decision condition at every layer of the band.
4. **Consistent over prompts.** The exact one-sided sign test of the prompt-level band-score differences against each
   baseline (ties dropped), the larger p of the two, Holm-adjusted across **14 tests** (2 candidates × 7 bands), below
   0.025.
5. **A real improvement.** The pooled point estimates of candidate − J++ and candidate − naked eye are each at least
   0.05.

Every comparison uses unrounded values.

### 7.2 Eligibility, precedence and what serving would mean

- **Candidates:** jpp-it and rl-it only. R1's lens and Miru are evaluated by the same five rules for reference, outside
  the Holm family, and are never candidates.
- **Precedence:** where both pass at a band, jpp-it is the one recommended.
- **Never below code 30** (page 31): no band with a lower layer exists or is served.
- **R3 serves nothing itself.** A pass is a recommendation to a later serving track on a branch from `main`, which
  must also pass the gates a second lens needs (orientation, bf16 against fp32 precision, the stuck-word check), keep
  the server's defaults, add a filtered second reading as a request option, and withhold R1 and Miru as R2 decided.
  Its lens card would say what was tested: the next content word, not neuron labels (E45, E47), and no pronoun
  sentence unless a pronoun gate is built. Layers 11–30 keep their caution whatever R3 finds.

No page text changes while R3 is open: the pages already say that every lens reading comes through base-fitted J++.

### 7.3 Power

Effective n. In R2, 199 of the 220 prompts could be read: every chat-no-answer, chat-begun and mid-reply prompt, but
only 39 of the 60 bare ones (55 had a content target, 16 of them a collapse word), about 65%. With the same shares, R3
expects about 80 + 80 + 52 + 80 = 292 valid prompts. R2's pooled intervals resolved differences of about 0.06–0.07;
with √(199/292) ≈ 0.83, R3's should resolve about 0.05–0.06. Per condition (about 52–80 prompts) intervals should be
about ±0.10–0.15, so rule 2 still catches only large harms.

Rule 5 is a threshold on the point estimates and does not move with n: a true difference of +0.046, R1's at code 58
in R2, fails it with any number of prompts. Holm over 14 tests instead of 9 makes rule 4 stricter: the smallest
adjusted p is 14 times the smallest raw one.

### 7.4 The rule's code is R2's

`R3_06_analyze.py` re-implements R2's rule, parameterised by a design (conditions, readouts, candidates, reference
lenses, bands, eligibility, precedence, lowest servable layer, seeds). Run on `R2_02_eval.json` with R2's design, it
reproduces `R2_03_analysis.json` exactly: 236 comparisons, 0 mismatches, every verdict, rule, Holm p, interval and
sensitivity verdict (`R3_06_r2_reproduction.json`). Its self-test checks each rule failing alone, precedence, the
never-below-30 rule, the reference lenses, the sensitivity analyses, the sign test and Holm, and gives R2_03's own
synthetic file R2_03's analysis. Only pure pieces are imported from R2 (`word_key`, the collapse targets, the
formatting-like test, the sign-test tail, Holm).

## 8. Secondary readings and sensitivity analyses (not used for the decision)

### 8.1 H2: agreement with base J++

Pullbacks d·Jₗ of R1's 128 final-space directions through jpp-it, rl-it, their halves and jpp-it-full-macro
(`R3_05_pullbacks.pt`), against those through base J++, JLens and Miru (`R3_Q0_pullbacks.pt`) and R1's instruct
covectors (`R1_04_gbar.pt`). Cosines as R1_04_analyze_dir.py: over the coordinates of a mask (q10: each layer's bottom
10% of dimensions by the spread of the 4-bit instruct model's states, `V6_quiet_dims_stats.pt`; loud: the rest), per
direction and layer, averaged over a band's layers (code 30–39, 40–49, 50–58), then over the 98 token-pair directions,
with 95% intervals over them. The split-half ceiling is jpp-it-A against jpp-it-B.

Yardsticks (q10): J++ against JLens, two methods on the base model, 0.77 / 0.68 / 0.66 (R1_IT_LENS_REPORT.md:233,
reproduced by Q0 as 0.767 / 0.685 / 0.657); R1's instruct covectors against J++ 0.27 / 0.29 / 0.29 and against JLens
0.31 / 0.38 / 0.42 (Q0); the same model fitted on chat against prose 0.77 / 0.78 / 0.82 (R1_04 `cos_chat_prose`); R1's
split-half ceiling 0.96 / 0.98 / 0.99.

### 8.2 H3: concentration

For each lens and fitted layer: the share of its squared Frobenius norm in its own top 100 singular directions on the
output side, and the share inside base J++'s top-100 output subspace (R1 §3.2's method), for jpp-it, rl-it, base J++,
JLens and Miru.

### 8.3 H4 and J++'s own recall

- **H4.** On R1_10's 4,808 held-out positions: KL(final ‖ lens) at R1_10's seven logit scales (0.25, 0.5, 0.75, 1, 1.5,
  2, 3), the calibrated KL (the smallest), and top-1 and top-10 agreement with the model's own final readout, at code
  40, 45, 50, 55, 58, over all positions and per class (prose, chat), for jpp-it, rl-it, base J++, JLens, the naked
  eye, R1 and Miru (point values, as R1 §4.2 gave them). R1's figures for
  comparison (R1_IT_LENS_REPORT.md:463–465), top-1 at those layers: R1's instruct full matrix 0.21 / 0.37 / 0.45 /
  0.55 / 0.85, J++ 0.13 / 0.22 / 0.25 / 0.39 / 0.69.
- **J++'s own held-out recall@10** at code 30, 38, 45, 52 (its README's readout layers from 30), with J++'s filter, on
  the held-out (non-fitting) items of the evals the floor kept, through jpp_lens's own rank code, for jpp-it, rl-it,
  base J++, the naked eye and Miru, under the primary macro and the full one.

### 8.4 H5

jpp-it − rl-it, filtered top-1, pooled and per condition, per band, with paired intervals from the same bootstrap; and
the recall of §8.3 for both.

### 8.5 H6: fine-tuning against quantisation (Q0 and Q1)

- **Q0** (CPU, done; `R3_Q0_pullbacks.json`). The implementation check: from the pullbacks R1 stored, J++ against
  JLens equals R1_04's table in all six bands and all three masks (max difference 1.3e-8); recomputed from the two
  files, q10 at code 30–39 / 40–49 / 50–58 is 0.767 / 0.685 / 0.657 against R1's 0.77 / 0.68 / 0.66 (tolerance 0.02).
  Then the **fine-tuning axis**, Miru (bf16 instruct) against JLens (bf16 base), plain J-lenses on WikiText both:
  0.37 / 0.39 / 0.40 in q10. For reference only (different estimators): R1's 4-bit instruct covectors against Miru
  0.65 / 0.69 / 0.72.
- **Q1** (GPU, not run). The 4-bit J-lens covectors (D22) against Miru: the **quantisation axis**, with Q1's own
  split-half ceiling (Miru's is unknown). Reported in q10 and loud coordinates per band; code 58 is reported apart,
  because Miru collapses there (R2_03_report_section.md). While Q1 runs on J++'s 64 prompts, every Q1 number says so.

### 8.6 The replication set, and other reported readings

- **R2's 220 prompts**, read again by R3_05 with R2's own draw seed (20261018), so their positions are R2's. The rule
  is applied to them and reported; it decides nothing. J++, the naked eye, R1 and Miru on them should reproduce R2's
  numbers, which tests R3's reading path (D21).
- Everything R2 §8 reports, for every readout: per layer, the primary metric with pooled intervals and its
  differences, the stuck-word share and word, the formatting share; the sign test's discordant counts and raw and
  Holm p; the unfiltered metrics; the exact-id variant and the target in the filtered top 5; mid-reply-content-next;
  the splits by position type; the constant-guess baseline and the function-word split; the counts of positions
  without a target; the context figures and the agreement check.
- The half-lenses A and B enter only the pullbacks (§8.1), as the split-half ceiling; they are not read on the prompts.

### 8.7 Sensitivity analyses

Each reruns the whole rule (1–5, Holm, eligibility and precedence) and lists every lens-band whose verdict or
recommendation changes. The primary decision stands; a change is reported beside it.

| | rerun | reason |
|---|---|---|
| S1 | without the 32 prompts about the model | R1's fit set had many self-questions |
| S2 | without the 83 prompts of `R3_common.FIT_OVERLAP` | paraphrase or shared topic with a fit source (§4.3) |
| S3 | without the bare condition (the rule over the other three) | raw text puts this instruct model off-distribution |
| S4 | without positions with a formatting-like target | list numerals and code-fence tags may reward structure, not content |
| S5 | with J++'s exclusion list for every readout in place of R2's filter | D7: the recipe's own filter |

Also reported, not a sensitivity rerun of the decision: jpp-it-full-macro (poetry in the weights' macro) through the
same rule, outside the Holm family.

## 9. Failure handling

**Stops before reading.** A GPU job stops before any reading if: a registered file is untracked or differs from HEAD
(`--allow-uncommitted` overrides, a recorded deviation); its output already exists, for the pilot, the router, the
weights and the evaluation (`--force` overrides; the earlier run stays in the runs log and is reported); a pinned file's size or hash differs; the filters' counts differ from
those recorded; the prompts fail validation or their sha256 differs from a passing `R3_01_check.json`'s; the pilot's
decision is not "proceed", or was judged against another budget.

**Stops for an amendment.** The pilot fails a check (§10.2) or projects more than the approved budget, or no budget
is approved; fewer than 20 fitting pairs survive the per-eval floor (D16). Each stop writes its reason, and nothing
after it runs until an amendment is registered.

**Resumable jobs.** The router, each shard and Q1 checkpoint their state and resume after the last checkpoint. A
resumed job is not a deviation; a job restarted from the start after a change of code is, and is reported.

**Partial evaluation.** A run of R3_05 that stops part-way is rerun from the start; partial data is not analysed; the
stopped run stays in `R3_05_runs.jsonl` and is reported.

**Agreement below 0.95** between the reading forward's top-1 at 59 and the next token, in any condition: said in the
report; the rule is applied unchanged.

## 10. How to run

### 10.1 Commands

The worktree has no environment of its own: use the shared one, from the root of this checkout. CPU modes hide the
GPU themselves; `CUDA_VISIBLE_DEVICES=""` is given anyway.

1. CPU, any time (no model):
   - `R3_sources_check.py` (the cache against the manifest) and `R3_01_check.py` (pins, sources, filters, Miru's
     prompts, freshness; writes `R3_01_check.json`, exits 1 on any failure);
   - the self-tests: `R3_lrp.py`, `R3_fitlib.py`, `R3_gpu.py`, `R3_02_router.py`, `R3_03_fit.py`, `R3_04_weights.py`,
     `R3_Q1_jlens.py`, `R3_00_pilot.py`, `R3_06_analyze.py` with `--selftest`, and `R3_05_eval.py --selftest --limit
     3 --out <scratch>`; the positive control `R3_parity.py`; `R3_06_analyze.py --reproduce-r2`; `R3_Q0_pullbacks.py`;
   - `--check-only` of R3_00, R3_02, R3_03, R3_04, R3_05 and R3_Q1.
   For example: `CUDA_VISIBLE_DEVICES="" /home/eric/bin/ai/theodolite2/.venv/bin/python -I experiments/scripts/
   R3_01_check.py`.
2. Commit the registered files (§0).
3. GPU, with the `:8765` server stopped and no other instrument GPU check pending, in this order, each after the one
   before has finished:
   `R3_00_pilot.py --dry-run-12b`; `R3_00_pilot.py --pilot`; then, only if the pilot says "proceed",
   `R3_02_router.py --run`; `R3_03_fit.py --shard 0` … `--shard 7`; `R3_03_fit.py --merge` (CPU) and, after it,
   `--delete-shards`; `R3_04_weights.py --run`; `R3_05_eval.py`; `R3_Q1_jlens.py --run`.
4. CPU: `R3_06_analyze.py` (writes `R3_06_analysis.json` and `R3_06_report_section.md`, applies §7 exactly) with
   `--replication R3_05_eval_r2set.json`; `R3_Q1_jlens.py --analyze-only`.

### 10.2 The GPU budget

**The pilot comes first, and the fit needs the user's approval of a budget.** No budget is approved yet (D30); the
design recommends 24 GPU-hours.

- **The dry run** (`--dry-run-12b`) runs the whole pipeline once on the local bf16 Gemma 4 12B (D24). The pilot refuses
  to start without it.
- **The pilot** (`--pilot`, at most 30 minutes, on WikiText fit records 0 and 1; it reads no lens on an evaluation
  prompt): (a) the R-lens rules' coverage (the patched module names, counted) and the forward under the rules, whose
  top-1 word-piece must be unchanged at 99.9% of positions or more; (b) parity at cos 0.99 or more at every fitted
  layer, in both batching modes (D25); (c) memory and seconds per backward pass at B in {8, 16, 32, 64, 128}
  (`is_grads_batched`) and {1, 2, 3, 4} (replicated), each under the job cap, an out-of-memory error being a result;
  (d) the fastest configuration and the projection of GPU-hours for every R3 GPU job (shards, router, weights, Q1,
  H4's states, the evaluation), times 1.25. A failed check, a projection over the approved budget or no approved
  budget writes "stop"; only "proceed" lets R3_02–R3_05 and Q1 run.
- **The job cap.** Every GPU job caps its own allocations at 24 GiB (torch's per-process fraction) and records the
  card's free memory before and after (nvidia-smi and torch). When R3 was designed, other processes held 4,703 MiB of
  the card with no model loaded, so the headroom beside the 21–23 GB model may be only a few GB, which pushes B down.
- **Bounds.** From FLOPs: about 1.29e18 FLOP for 64 prompts × 5,376 rows × 128 tokens through blocks 30–59, which is
  3.6–14 hours at a sustained 25–100 TFLOP/s. **If the rows cannot be batched (B = 1):** 64 × 5,376 passes at R1's
  0.79 s per backward pass (R1_05_total_0.json), **about 75 hours**. For comparison, J++ reports about 7.3 H200
  GPU-hours for the Jacobians of all 63 layers of a 27B model in bf16. The rest is small beside the fit: two
  forward-only passes over the 1,000 records for the router, Q1's four batched passes per prompt, a forward per H4
  sequence, and the evaluation (R2's took about 6 minutes).

## 11. Amendments before any data

The design R3 was planned from was reviewed once before this registration; the user then settled its open decisions.
No R3 GPU job had been run, and none has been run since, so none of these is a deviation.

| | change | reason |
|---|---|---|
| A1 | A positive control: `R3_parity.py` runs jpp_lens's own stages, installed at `d7efc0de` in its own environment, beside R3's code (§2.3). | Every self-test compared R3 with its own brute force; nothing showed that R3 fits J++'s recipe. |
| A2 | The per-eval floor and the poetry-dropped primary macro (D16). | J++'s own Gemma lens dropped poetry (1 of 98 items correct); one item could carry a fifth of the macro. |
| A3 | Q1 to be fitted on Miru's own prompts, rebuilt and hash-checked; a fallback if the hashes fail (§3.3, D22); Miru called "plain gradients" only as far as "no LRP recorded". | Otherwise the quantisation comparison changes the prompts too. (The hashes did not match.) |
| A4 | No page text while R3 is open; the planned rubric, apparatus and roadmap line were reworded so that nothing claims more than is true. | "Is being made", "only the model changes", "riddles", a pronoun gate that exists nowhere, and a separation stated before any result. |
| A5 | The branch is cut from `main` with R2's two experiment commits, not from `it-second-reading`; the evaluation runtime is `main`'s at the registered commit (§0); serving is a later track (§7.2). | The R2 branch's runtime is 8 commits behind `main`, still serves R1 and Miru, and has no filtered second reading. |
| A6 | Shards written per layer, the merge streamed (D5). | 80 GB of sums at once against 61 GiB of RAM. |
| A7 | Only R2's pure pieces are imported; counts and seeds are R3's; the rule must reproduce R2's verdicts before any R3 data (§7.4). | R2's code hard-codes R2's counts, seeds, lenses and bands. |
| A8 | H4's states regenerated from R1_10's position list, its sha256 pinned (D23). | R1_10 stored only summaries. |
| A9 | Third-party texts kept out of git: only indices and hashes are committed (D26). | README.md:407–408. |
| A10 | The power statement (§7.3), the job cap and the B = 1 bound (§10.2), the 12B dry run (D24), and H3's wording (§1.1). | The first power claim was optimistic; the card's headroom is tighter than R1 found; a real Gemma 4 was on disk; finding no low-rank step in the code does not show the shipped file's concentration is not the recipe's. |
| A11 | The user's decisions: R3 may fetch, once and pinned, the 1,000 WikiText records, jpp_lens at `d7efc0de` (into its own environment) and J++'s six item files, and nothing else; fine-tuning is separated from quantisation through Miru, with no bf16 checkpoint downloaded; registration, scripts and CPU checks only in this round, no GPU run. | Recorded in R3_sources.json and in this file. |
