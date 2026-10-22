"""R3 (J++'s recipe refitted on the instruct checkpoint): shared definitions, pins, seeds, counts and paths.

Imported by every R3 script. Nothing here loads the model or touches the GPU. Layers are code layers, counted from 0
(the instrument's pages show layer + 1: code 30-58 are page layers 31-59).

Three kinds of thing live here:
- pins: the jpp_lens commit, the WikiText revision and the model snapshot; the sha256 and size of R1's lens, the
  Miru J-lens, R1's 128 directions and the quiet-dimension statistics (all gitignored files, read by absolute path);
  what Miru's own header records about its fit (its prompt hashes, read again from the file by R3_01_check.py);
- the registered numbers: the recipe's constants (D1-D15 of R3_PREREG.md §2), the seeds, the prompt counts, the
  layer bands, the lens set and the decision rule's constants (R2's five rules, unchanged);
- the loaders for the gitignored cache R3_fetch.py filled (experiments/R3_cache/): J++'s eval items, the 1,000
  WikiText records and the raw rows between them, each checked against the committed manifest R3_sources.json
  before use. No third-party text is written anywhere by these loaders.

R2's pure pieces (the token filter, word matching, the target search, freshness) are imported from R2_common, which
is not changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402

# ------------------------------------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parents[2]            # the checkout this file lives in (worktree or main)
SCRIPTS = REPO / "experiments" / "scripts"
RESULTS = REPO / "experiments" / "results"
MAIN_CHECKOUT = Path("/home/eric/bin/ai/theodolite2")  # gitignored data (lens files, R1 directions) may live only here
SECOND_READING = MAIN_CHECKOUT / ".claude" / "worktrees" / "it-second-reading"   # R2's worktree, read only
CACHE = REPO / "experiments" / "R3_cache"             # gitignored (R3_fetch.py)
SOURCES_MANIFEST = SCRIPTS / "R3_sources.json"
ENV_DIR = CACHE / "env"                               # jpp_lens's own environment (reuses the shared .venv read only)
ENV_PYTHON = ENV_DIR / "bin" / "python"
JPP_DIR = CACHE / "jpp_lens"
JPP_SRC = JPP_DIR / "src"

PROMPTS_PATH = SCRIPTS / "R3_prompts.json"
PREREG_PATH = RESULTS / "R3_PREREG.md"
CHECK_JSON = RESULTS / "R3_01_check.json"
PARITY_JSON = RESULTS / "R3_parity.json"
R2_REPRODUCTION_JSON = RESULTS / "R3_06_r2_reproduction.json"
EVAL_JSON = RESULTS / "R3_05_eval.json"
ANALYSIS_JSON = RESULTS / "R3_06_analysis.json"
REPORT_SECTION = RESULTS / "R3_06_report_section.md"
RUNS_LOG = RESULTS / "R3_05_runs.jsonl"
SELFTESTS_JSON = RESULTS / "R3_selftests.json"           # the latest CPU self-test result of each R3 script
# Where the fit writes its big tensors (per-layer shard files, merged experts): gitignored *.pt under results
SHARD_DIR = RESULTS / "R3_03_shards"

# ------------------------------------------------------------------------------------------------ pins
JPP_REPO = "https://github.com/safety-research/jpp_lens"
JPP_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"
WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"   # Miru's pin; J++'s loader pins none
MODEL_ID = R2.MODEL_ID                                             # google/gemma-4-31B-it-qat-w4a16-ct
MODEL_SNAPSHOT = "52f3f65bc7a02d555763bc923bd1d9094898219d"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots" / MODEL_SNAPSHOT

# Reference lenses (never candidates in R3). The values are R2's pins (R2_common), restated so that a change on
# either side is caught by R3_01_check.py.
R1_SHA256 = "80ce560943f39bc4183a7387512d20ad7ef779476fc1ef01b5010463a43e107b"
R1_SIZE = 173_411_927
# Where R1's pinned lens may be. The main checkout's copy (2026-10-17 20:51) has the same size and the same three
# matrices but an older provenance note, so its sha256 differs (37252f5e...): it is not the pinned file.
R1_LENS_CANDIDATES: Tuple[Path, ...] = (RESULTS / "R1_18_it_lens.pt",
                                        SECOND_READING / "experiments" / "results" / "R1_18_it_lens.pt",
                                        MAIN_CHECKOUT / "experiments" / "results" / "R1_18_it_lens.pt")
MIRU_REPO_ID = R2.MIRU_REPO_ID
MIRU_REVISION = "134a973cf473ad12a885d3880f3b2e01ef495cd6"
MIRU_FILENAME = "miru/google/gemma-4-31B-it.safetensors"
MIRU_PATH = (Path.home() / ".cache/huggingface/hub/models--returnmoe--jlens-adapters/snapshots" / MIRU_REVISION
             / MIRU_FILENAME)
MIRU_SHA256 = "cedd4b62f49a7f79e07c1bb107ad1d48d39dc7ad2790611de7e9480b4543386a"
MIRU_SIZE = 3_410_446_944
# What Miru's header (fit_metadata, read without loading a tensor) records about its fit and its prompts. R3_01
# rebuilds the prompt sequence from this description and checks it against both hashes (R3_PREREG.md §3).
MIRU_FIT = {
    "model_name_or_path": "google/gemma-4-31B-it",
    "model_commit_hash": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "compute_dtype": "bfloat16",
    "estimator": "summed_current_and_future_targets_v1",
    "dim_batch": 32,
    "target_layer": 59,
    "max_seq_len": 128,
    "skip_first": 16,
    "position_reduction": "mean",
    "prompt_reduction": "uniform_mean",
    "processed_prompts": 425,
    "skipped_prompts": 0,
    "prompt_budget": 1000,
    "selected_prompts": 1000,
    "selection": {"concatenate_and_rechunk": True, "exclude_headings": True, "max_chars": 2000, "min_chars": 200},
    "dataset_revision": WIKITEXT_REVISION,
    "shards": ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
               "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
    "prompt_sequence_sha256": "dcb2a738953033139ae35f3665271ccdb795409c6073a85ea46f02a852143cc6",
    "processed_prompt_prefix_sha256": "f579711549217eff57dfb99d5e6d7e17fbfb45f3718b9772dbeb925375e7ab04",
    "tokenizer_sha256": "4d3949535768436b88d00504ec29e104addfa69231e9538e9c19a610927e9b3e",
}
# R1's 128 token-pair directions and the quiet-dimension statistics (q10), for H2 and the Q1 directional arm
R1_DIRS_PATH = MAIN_CHECKOUT / "experiments" / "results" / "R1_02_dirs.pt"
R1_DIRS_SHA256 = "21c94ae53b5fad8ab3de42b370d96edcc9a46188258c21dbcaad9197b9018206"
R1_DIRS_SIZE = 333_058_876
QUIET_STATS_PATH = MAIN_CHECKOUT / "experiments" / "results" / "V6_quiet_dims_stats.pt"
QUIET_STATS_SHA256 = "7d6763a77975be9333910c97e787595ea8cdc299820e25bd95986887bd4c71bd"
QUIET_STATS_SIZE = 31_017_292
# The two base lenses the instrument ships (J++, the baseline of every rule) and reads beside it (JLens), as the hub
# cache holds them: the sha256 is the LFS blob's own name, checked again by R3_Q0_pullbacks.py and R3_05_eval.py.
LENS_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots"
                 / "e166deb5ad540932100ef803c8c9bcc2a5453694" / "gemma-4-31b")
JPP_BASE_PATH = LENS_SNAPSHOT / "jpp" / "Salesforce-wikitext" / "gemma-4-31B_jpp_lens.pt"
JPP_BASE_SHA256 = "ef50bc54b04738e558b7d86847889bb5d60437e45276b1952e73da91b4e9f5fc"
JPP_BASE_SIZE = 3_410_381_570
JLENS_PATH = LENS_SNAPSHOT / "jlens" / "Salesforce-wikitext" / "gemma-4-31B_jacobian_lens.pt"
JLENS_SHA256 = "d78d3e1483f1f2159dc955ca2871d796fd995fc4613d85103ad04475b810415f"
JLENS_SIZE = 3_410_380_420
# R1's IT covectors for its 128 directions (R1_04_analyze_dir.py: "mix", the equal-class mean of exact same-position
# VJPs on the W4A16 instruct model), the reference "IT" row of Q0
R1_GBAR_PATH = MAIN_CHECKOUT / "experiments" / "results" / "R1_04_gbar.pt"
R1_GBAR_SHA256 = "2afd9c63af3840ea7e75330704a7f3ca125a56efd969e4fe1b22dc516b09e3ee"
R1_GBAR_SIZE = 1_816_665_333
# The numbers Q0 must reproduce (R1_IT_LENS_REPORT.md:233, from the committed R1_04_dir_analysis.json): J++ against
# JLens in the quiet dimensions, 98 token-pair directions, code bands 30-39 / 40-49 / 50-58
R1_04_JSON = RESULTS / "R1_04_dir_analysis.json"
Q0_TARGET_Q10 = {"L30-39": 0.77, "L40-49": 0.68, "L50-58": 0.66}
Q0_TOL = 0.02
Q0_BANDS: Tuple[Tuple[str, int, int], ...] = (("L0-9", 0, 9), ("L10-19", 10, 19), ("L20-29", 20, 29),
                                              ("L30-39", 30, 39), ("L40-49", 40, 49), ("L50-58", 50, 58))
TOKPAIR_DIRS = tuple(range(30, 128))                  # R1's 98 random token-pair directions (R1_02_prep.py)

# H4 (R3_PREREG.md §8): R1_10_evalfull.py's held-out positions (E3's 24 prose paragraphs and its 40 chats), rebuilt
# by R3_05_eval.positions_r1_10 on the CPU from E3's own files. The list [[name, class, position], ...] serialised by
# json.dumps(separators=(",", ":")) must have this sha256 (computed from R1_10_evalfull.pt's own "meta").
R1_10_POSITIONS_SHA256 = "fd596c3bd794b45025c97a0f722c61a00d8ff69b02adbf920a2254b2798806dc"
R1_10_N_POSITIONS = 4808
R1_10_PATH = MAIN_CHECKOUT / "experiments" / "results" / "R1_10_evalfull.pt"
R1_10_SHA256 = "3301e4cdfcd43719c77886591618a15cd10c7d5b8d73d169cd5f77c87a3994c6"
E3_CHATS_PATH = RESULTS / "E3_lens_validation_chats.json"
E3_CHATS_SHA256 = "623f305b1707183793257e8c06588c2879e0ce31fda32dc69df4f7a99cb71fe4"
H4_LAYERS: Tuple[int, ...] = (40, 45, 50, 55, 58)     # R1 §4.2's layers (R1_FM_LAYERS from 40)
H4_TAUS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)   # R1_10's logit scales
# The calibrated KL is R1_14's: the scale chosen on one sequence parity (sorted names) and scored on the other. The
# intervals are a bootstrap over sequences (R1 §4.2 resampled passages), 2000 resamples, this seed and the class
# index, the same indices for every lens and layer.
H4_BOOT_SEED = 20261023
H4_N_BOOT = 2000
JPP_RECALL_LAYERS: Tuple[int, ...] = (30, 38, 45, 52)  # J++'s own held-out recall@10 (its README's readout layers)

# ------------------------------------------------------------------------------------------------ the recipe (D1-D15)
N_LAYERS = 60
D_MODEL = 5376
FIT_LAYERS: Tuple[int, ...] = tuple(range(30, 59))    # D2: source layers code 30-58 (RAM); pages 31-59
TARGET_LAYER = 59                                     # block 59's output, raw (relative_end_transport_layer -1)
N_EXPERTS = 8                                         # E, J++'s RECIPE_NUM_CLUSTERS
PCA_DIM = 64                                          # the router's projection
SKIP_FIRST = 16                                       # kept positions 16 .. T-2 (jpp_lens utils.py)
MAX_SEQ_LEN = 128                                     # fit prompt truncation (router and shards)
N_FIT = 64                                            # the first 64 records are the fit prompts
N_ROUTER = 1000                                       # all 1,000 feed the router
N_SHARDS = 8                                          # 8 prompts per shard; halves A (even shards) and B (odd)
MIN_KEPT = 50                                         # an expert with fewer kept positions falls back to pooled
ROUTER_SEED = 0
KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-6
LRP_MODE = "rlens"                                    # LN rule, identity rule, half rule
CONDENSE = {"pass_k": 10, "top_n": 3, "steps": 200, "learning_rate": 0.05, "temperature": 0.1, "exact_every": 5}
FIT_FRACTION = 0.2                                    # D15: the dev (fitting) split of the J++ items
SPLIT_SEED = 0
READOUT_MAX_SEQ_LEN = 512                             # J++'s RECIPE_READOUT_MAX_SEQ_LEN (item prompts)
# J++'s macro over its evals (order-ops is out of the macro); the primary macro drops poetry, J++'s own convention for
# Gemma 4 31B (koayon/jpp-lenses README: 1 of 98 poetry items passes the correctness filter). The full macro is a
# sensitivity analysis.
JPP_MACRO_EVALS: Tuple[str, ...] = ("multihop", "multilingual", "typo", "association", "poetry")
PRIMARY_MACRO_EVALS: Tuple[str, ...] = ("multihop", "multilingual", "typo", "association")
JPP_EVALS: Tuple[str, ...] = ("multihop", "multilingual", "order-ops", "poetry", "association", "typo")
# Per-eval floor, fixed before any grading: an eval keeping fewer correct items, or fewer fitting items, than this is
# dropped from the macro (its counts are still recorded in R3_04_weights.json).
EVAL_FLOOR_CORRECT = 10
EVAL_FLOOR_FITTING = 2
MIN_DEV_PAIRS = 20                                    # fewer dev pairs in all: stop for an amendment

# ------------------------------------------------------------------------------------------------ the GPU (R3_00)
# Every GPU job runs with the server stopped and caps its own memory at JOB_CAP_GB (torch's per-process fraction),
# so that a configuration that would need more fails as an out-of-memory error inside the job, never by starving
# another process. The budget is what the user approves for the whole fit; R3_00's projection stops the run for an
# amendment when it is over, or while no budget is approved. [NOT YET APPROVED]: the design recommends 24 GPU-hours.
JOB_CAP_GB = 24.0
APPROVED_GPU_HOURS: Optional[float] = None
B1_BOUND_HOURS = 75.0                                 # 64 x 5376 x ~0.79 s (R1_05_total_0.json): rows one at a time
PILOT_MAX_MINUTES = 30.0
PILOT_N_PROMPTS = 2                                   # WikiText fit prompts 0 and 1; no evaluation prompt is read
PILOT_PARITY_DIMS = 6                                 # output dimensions checked against an independent backward
PILOT_PARITY_COS_MIN = 0.99                           # with the rules off and with them on (the fit's own rows)
PILOT_LRP_TOP1_MIN = 0.999                            # forward top-1 identical under the rules at this share
# The forward-only identity check reads more records than the timed pilot, so that 0.1% is resolvable: WikiText fit
# records 0-15 (about 2,000 positions; the check allows floor(0.001 x positions) flips, 2 at 2,000)
PILOT_LRP_N_PROMPTS = 16
PILOT_VMAP_B: Tuple[int, ...] = (8, 16, 32, 64, 128)
PILOT_REPLICATE_B: Tuple[int, ...] = (1, 2, 3, 4)
PILOT_PASSES = 3                                      # timed backward passes per configuration
DRY_RUN_MODEL_DIR = Path.home() / "bin" / "ai-models" / "txt" / "google" / "gemma-4-12B"   # bf16, gemma4_unified
PILOT_JSON = RESULTS / "R3_00_pilot.json"
DRY_RUN_JSON = RESULTS / "R3_00_dry_run.json"
ROUTER_PT = RESULTS / "R3_02_router.pt"
ROUTER_JSON = RESULTS / "R3_02_router.json"
ROUTER_CKPT = RESULTS / "R3_02_router_ckpt.pt"
MERGED_DIR = RESULTS / "R3_03_merged"                 # gitignored (per-layer merged experts, pooled lens, halves)
WEIGHTS_JSON = RESULTS / "R3_04_weights.json"
LENS_FILES: Dict[str, Path] = {                       # R3_04's lens files (bf16, {"J": {l: [5376, 5376]}}, h @ J.T)
    "jpp-it": RESULTS / "R3_04_jpp_it.pt", "rl-it": RESULTS / "R3_04_rlens_it.pt",
    "jpp-it-A": RESULTS / "R3_04_jpp_it_A.pt", "jpp-it-B": RESULTS / "R3_04_jpp_it_B.pt",
    "rl-it-A": RESULTS / "R3_04_rlens_it_A.pt", "rl-it-B": RESULTS / "R3_04_rlens_it_B.pt",
    "jpp-it-full-macro": RESULTS / "R3_04_jpp_it_fullmacro.pt",
}

# ------------------------------------------------------------------------------------------------ R3_04's stops
# association and typo have no correctness target (R3_fitlib.ITEM_SPECS): grade() returns None for every item and the
# floor always keeps them. The registered stop for the case the floor exists for: fewer than MIN_GRADED_EVALS of the
# primary macro's graded evals (multihop, multilingual) survive the floor.
UNGRADED_EVALS: Tuple[str, ...] = ("association", "typo")
MIN_GRADED_EVALS = 2

# ------------------------------------------------------------------------------------------------ the runtime (§0)
# The evaluation runtime is the instrument's backend at this commit (main when the branch was cut); every GPU job
# refuses a backend/ tree that differs from it, and the R2 files R3 imports or reads that differ from these sha256.
RUNTIME_COMMIT = "d5405ee964e0ac1732babd8e69fa9271de555189"
PINNED_UNCHANGED: Dict[str, str] = {
    "experiments/scripts/R2_common.py": "a3fe4ab44edbce3eeba01efa1428c82b7abb5f8b0555d7485fb2d008d78b762d",
    "experiments/scripts/R2_03_analyze.py": "9293011e96b3a132a915b306b2835846cc284e073fa5644124f20048cc377ecd",
    "experiments/scripts/R2_prompts.json": "623a7fb73e6a082335653bb02c8e22d7e6d731ef343ff299af57709a772d01fd",
    "experiments/results/R2_01_check.json": "9baad05e2d0057b2f7621fb389400cd5c24e119151b292648ca81343a8181581",
    "experiments/results/R2_02_eval.json": "6e277f60987c88a70f85355e2e749aac8cfa18f5480a5b044befedc3626f9c70",
    "experiments/results/R2_03_analysis.json": "94a632527ce6d1f356c836272ef5e1c32d8bef549cb62d6a8136dae83a22a75e",
    "backend/theodolite_math.py": "0bba67755da1bfaf313df46dd8453280a368dfaf72d209e4b5327c00f0c80c49",
    "backend/engine_runtime.py": "0a7e071ba782cdb7a70eaff961b6ad1ce371c766ccbfbee9aa61979a5672ef06",
}

# ------------------------------------------------------------------------------------------------ Q1 (H6)
# The 4-bit J-lens arm: plain gradients (no rules), the estimator summed over this and every later kept target
# (Miru's "summed_current_and_future_targets"), mean over kept positions, uniform mean over prompts, target block
# 59's raw output, for R1's 128 directions at the fitted layers, in batches of Miru's dim_batch 32.
Q1_DIM_BATCH = 32
Q1_LAYERS: Tuple[int, ...] = FIT_LAYERS
Q1_PT = RESULTS / "R3_Q1_jlens.pt"
Q1_JSON = RESULTS / "R3_Q1_jlens.json"
# H6's bands (R3_PREREG.md §1.2): code 58 is left out of the last band, because Miru collapses there (R2), and is
# reported apart
H6_BANDS: Tuple[Tuple[str, int, int], ...] = (("L30-39", 30, 39), ("L40-49", 40, 49), ("L50-57", 50, 57),
                                              ("L58", 58, 58))
H6_READ_BANDS: Tuple[str, ...] = ("L30-39", "L40-49", "L50-57")
# Q0's key for Miru against JLens: it carries fine-tuning together with Miru Tracer's code against
# anthropics/jacobian-lens and the two fits' prompts, so it is not named "the fine-tuning axis"
Q0_MIRU_JLENS = ("fp32: miru vs jlens (fine-tuning plus code, estimator and prompt differences: bf16 instruct vs "
                 "bf16 base, plain J-lenses)")
Q0_JSON = RESULTS / "R3_Q0_pullbacks.json"
Q0_PT = RESULTS / "R3_Q0_pullbacks.pt"

# ------------------------------------------------------------------------------------------------ the evaluation
CONDITIONS: Tuple[str, ...] = R2.CONDITIONS           # chat-no-answer, chat-begun, bare, mid-reply
SECONDARY_CONDITIONS: Tuple[str, ...] = R2.SECONDARY_CONDITIONS
N_PROMPTS: Dict[str, int] = {"chat-no-answer": 80, "chat-begun": 80, "bare": 80, "mid-reply": 80}
MID_DRAW_SIZES: Dict[str, int] = {"content_next": 3, "any": 6}
MID_NEW_TOKENS = R2.MID_NEW_TOKENS
CONT_WINDOW = R2.CONT_WINDOW
DRAW_SEED = 20261021                                  # mid-reply position draws (R2 used 20261018)
BOOT_SEED = 20261022                                  # prompt-level bootstrap (R2 used 20261019)
N_BOOT = 2000
READ_LAYERS: Tuple[int, ...] = tuple(range(30, 60))   # captured from layer 30 (R2 captured 40-59)
BANDS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("30-39", tuple(range(30, 40))),
    ("40-49", tuple(range(40, 50))),
    ("50", (50,)),
    ("51-54", (51, 52, 53, 54)),
    ("55", (55,)),
    ("56-57", (56, 57)),
    ("58", (58,)),
)
CANDIDATES: Tuple[str, ...] = ("jpp-it", "rl-it")     # R3-JPP-IT (weighted experts), R3-RL-IT (pooled, LRP)
PRECEDENCE: Tuple[str, ...] = ("jpp-it", "rl-it")     # where both pass at a band, the J++ recipe is served
REFERENCE_LENSES: Tuple[str, ...] = ("r1", "miru")    # reported for reference only, never candidates
SENSITIVITY_LENSES: Tuple[str, ...] = ("jpp-it-full-macro",)   # D16: poetry in the weights' macro; reported only
BASELINES: Tuple[Tuple[str, str], ...] = (("jpp", "J++"), ("naive", "naked eye"))
MIN_SERVE_LAYER = 30                                  # never below code 30 (page 31)
HOLM_M = len(CANDIDATES) * len(BANDS)                 # 14 lens-band tests
STUCK_MAX = R2.STUCK_MAX
MIN_EFFECT = R2.MIN_EFFECT
ALPHA = R2.ALPHA
TOPK = R2.TOPK
COLLAPSE_TARGETS = R2.COLLAPSE_TARGETS
# Sensitivity analysis S2: prompts that paraphrase, or share an answer topic with, one of the fit sources (the 1,000
# WikiText records, J++'s items, R1's fit set). Fixed from the prompt texts alone, before any R3 data, in the same
# commit as R3_prompts.json (R3_PREREG.md §4.3). A prompt is listed when, by judgement and generously:
# - WikiText: its subject or expected answer is the subject of an article one of the 1,000 records comes from (the
#   article's title row), or the same kind of thing (a warship, a football match, a planet, a fable, ...). Records are
#   numbered 0-999 in stream order; records 0-63 are the fit prompts, the rest feed the router only. No WikiText text is
#   written here: the article is described in our own words.
# - J++'s items: it asks the same kind of question as the items of an eval that enters a weight fit (association,
#   multihop, multilingual, poetry, typo; order-ops enters none), or its expected answer is an item's topic. Items are
#   numbered by their place in the pinned item file (R3_common.jpp_items()), as R3_04 records them; no name, text or
#   description of an item. Where a prompt overlaps a whole eval rather than an item, the kind of question is said in
#   our own words.
# - R1's fit set: as R2_common.R1_OVERLAP, a paraphrase of one of R1_SELF_Q / R1_TASK_Q, or an expected answer that is
#   the subject of one of them or of R1_ENC_TOPICS (these are the study's own prompts, quoted).
_WT_FOOTBALL = ("WikiText fit prompts 35-36 and records 599-603, 668-684 (a national football team, a footballer, a "
                "club's season)")
_WT_PLANET = "WikiText records 298-300 (an article on a planet of another star)"
_R1_BASIL = "R1_TASK_Q 'How do I keep basil plants alive indoors?'"
_R1_MAKERS = "R1_SELF_Q 'What is your relationship to the people who made you?'"
FIT_OVERLAP: Dict[str, str] = {
    # chat-no-answer
    "r3-cna-02": "J++ multihop #12; WikiText records 685-694 (an article "
                 "on a chemical element)",
    "r3-cna-04": "WikiText fit prompts 58-63 (an article on a tower in the Vatican); J++ association #26; R1_ENC_TOPICS "
                 "'the Renaissance'",
    "r3-cna-06": "WikiText records 890-914 (an article on a bird species)",
    "r3-cna-10": "J++ multilingual and multihop #90-92: the same kind of question as some of their items",
    "r3-cna-13": "J++ association #46",
    "r3-cna-18": "WikiText records 214-232 (an article on a coin); J++ multihop #48",
    "r3-cna-24": _WT_PLANET + "; R1_ENC_TOPICS 'the planet Jupiter'",
    "r3-cna-29": "WikiText records 247-268 (an article on a novel)",
    "r3-cna-33": "J++ multilingual (a word asked for in another language): the same kind of question",
    "r3-cna-35": "R1_ENC_TOPICS 'glaciers'",
    "r3-cna-36": "WikiText fit prompts 45-57 (an ice-hockey team's season) and records 97-104, 820-837 (ice-hockey "
                 "players)",
    "r3-cna-44": "J++ association #18",
    "r3-cna-48": "J++ multilingual (a word asked for in another language): the same kind of question",
    "r3-cna-49": "J++ association #24",
    "r3-cna-50": "J++ association #10",
    "r3-cna-53": _R1_MAKERS,
    "r3-cna-58": "R1_SELF_Q 'How do you balance honesty and kindness?'",
    "r3-cna-61": "R1_TASK_Q 'Why do cats purr?'",
    "r3-cna-74": _R1_BASIL,
    "r3-cna-79": "WikiText records 167-199 (an article on light aviation); J++ association #22",
    # chat-begun
    "r3-cb-01": _WT_PLANET + "; J++ multihop #2; R1_ENC_TOPICS 'the planet Jupiter'",
    "r3-cb-04": "WikiText records 685-694 (an article on a chemical element); J++ multihop #11, #13; "
                "R1_ENC_TOPICS 'copper'",
    "r3-cb-09": "R1_TASK_Q 'Describe the plot of Hamlet briefly.' (the same author)",
    "r3-cb-10": _WT_PLANET + " (and its star)",
    "r3-cb-12": "WikiText records 915-923 (an article on a blood cancer)",
    "r3-cb-13": "R1_ENC_TOPICS 'coral bleaching'",
    "r3-cb-14": "WikiText records 878-885 (an article on a Maya city)",
    "r3-cb-20": _WT_PLANET + "; R1_ENC_TOPICS 'the planet Jupiter'",
    "r3-cb-32": "WikiText records 572-598 (an article on a film)",
    "r3-cb-38": "J++ poetry (rhyming couplets): the same kind of task; R1_TASK_Q 'Write a two-line rhyme about coffee.'",
    "r3-cb-46": "R1_SELF_Q 'How do you handle a request you can't fulfill?'",
    "r3-cb-50": _R1_MAKERS,
    "r3-cb-53": "R1_SELF_Q 'How do you think about your own mistakes?'",
    "r3-cb-54": "R1_SELF_Q 'Do you prefer short or long questions? Why?'",
    "r3-cb-59": _R1_BASIL,
    "r3-cb-63": "J++ association #47",
    "r3-cb-64": _R1_BASIL,
    "r3-cb-66": "J++ association #0",
    "r3-cb-68": "J++ multilingual #97-100",
    "r3-cb-71": "J++ multilingual (a word asked for in another language): the same kind of question",
    "r3-cb-75": _R1_BASIL,
    # bare
    "r3-bare-05": _WT_FOOTBALL,
    "r3-bare-07": _WT_PLANET + "; R1_ENC_TOPICS 'the planet Jupiter'",
    "r3-bare-20": "WikiText records 247-268 (an article on a novel)",
    "r3-bare-24": "WikiText records 110-157 (an article on a pantheon of gods); J++ multihop #15, #21; R1_TASK_Q "
                  "'What caused the fall of the Roman Empire?'",
    "r3-bare-27": "WikiText records 890-914 (an article on a migratory bird)",
    "r3-bare-28": _WT_FOOTBALL,
    "r3-bare-32": "WikiText records 214-232 (an article on a gold coin); J++ multihop #9",
    "r3-bare-35": "WikiText records 915-923 (an article on a disease); R1_ENC_TOPICS 'penicillin'",
    "r3-bare-36": "J++ multihop #72; R1_ENC_TOPICS 'the Renaissance'",
    "r3-bare-45": "WikiText records 70-82, 105-109, 200-206, 462-485, 520-525, 847-848, 993-994 (articles on warships) "
                  "and 635-642 (a naval battle)",
    "r3-bare-51": "WikiText records 237-246 (an article on a treaty that ended a conflict)",
    "r3-bare-52": _WT_FOOTBALL,
    "r3-bare-54": "WikiText fit prompts 15-29 and 58-63 (articles on two towers)",
    "r3-bare-58": "WikiText records 802-807 (an article on a fable)",
    "r3-bare-59": "R1_TASK_Q 'What are the main causes of inflation?'",
    "r3-bare-60": _R1_BASIL,
    "r3-bare-64": "WikiText records 763-773 (an article on a dinosaur genus)",
    "r3-bare-65": "R1_TASK_Q 'How do volcanoes form?'",
    "r3-bare-71": "J++ multihop #40",
    # mid-reply
    "r3-mid-02": "WikiText records 457-461 (an article on a class of submarine)",
    "r3-mid-03": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "r3-mid-06": _R1_BASIL,
    "r3-mid-08": "R1_TASK_Q 'How does a credit score work?'",
    "r3-mid-11": "R1_TASK_Q 'What caused the fall of the Roman Empire?'",
    "r3-mid-15": "R1_TASK_Q 'How does the stock market work?'",
    "r3-mid-20": "J++ multihop #29",
    "r3-mid-28": "R1_ENC_TOPICS 'the city of Kyoto'",
    "r3-mid-34": "WikiText records 167-199 (an article on light aviation); J++ multihop #64; "
                 "R1_TASK_Q 'How do airplanes fly?'",
    "r3-mid-35": "J++ multihop #76",
    "r3-mid-37": "R1_TASK_Q 'How does the stock market work?'",
    "r3-mid-41": "R1_TASK_Q 'How does recycling actually work?'",
    "r3-mid-42": "J++ multihop #35; R1_TASK_Q 'How does GPS determine location?'",
    "r3-mid-45": "R1_ENC_TOPICS 'the Hubble Space Telescope'",
    "r3-mid-52": "R1_ENC_TOPICS 'the Hanseatic League' (medieval trade)",
    "r3-mid-56": _WT_FOOTBALL,
    "r3-mid-62": "R1_TASK_Q 'How do earthquakes happen?'",
    "r3-mid-71": "J++ association #12",
    "r3-mid-73": "R1_SELF_Q 'How do you approach a question you have never seen before?'",
    "r3-mid-74": "R1_SELF_Q 'What does a good day of conversations look like for you?'",
    "r3-mid-75": "R1_SELF_Q 'How do you think about your own uncertainty?'",
    "r3-mid-78": _R1_MAKERS,
    "r3-mid-80": "R1_SELF_Q 'How do you stay accurate when you are not sure?'",
}


# ------------------------------------------------------------------------------------------------ hashing
def sha256_file(path: Path) -> str:
    return R2.sha256_file(Path(path))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sequence_sha256(hashes: Sequence[str]) -> str:
    """The manifest's sequence rule: sha256 of the hex digests joined by newlines, in order."""
    return hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()


def check_pinned_file(path: Path, sha256: str, size: int, *, full: bool = True) -> Dict[str, Any]:
    """{"path", "exists", "size_ok", "sha256_ok"} for a pinned file (``full=False``: the size only)."""
    path = Path(path)
    rec: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    rec["size_ok"] = rec["exists"] and path.stat().st_size == size
    rec["sha256_ok"] = (sha256_file(path) == sha256) if (full and rec["size_ok"]) else None
    return rec


def find_r1_lens(full: bool = True) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """The first of R1_LENS_CANDIDATES that is the pinned file, and what was found at each."""
    seen = []
    for p in R1_LENS_CANDIDATES:
        rec = check_pinned_file(p, R1_SHA256, R1_SIZE, full=full)
        seen.append(rec)
        if rec["size_ok"] and rec["sha256_ok"] in (True, None):
            return p, seen
    return None, seen


def miru_header(path: Path = MIRU_PATH) -> Dict[str, Any]:
    """The safetensors header of the Miru file (8-byte length, then JSON); no tensor is read."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def miru_fit_metadata(path: Path = MIRU_PATH) -> Dict[str, Any]:
    return json.loads(miru_header(path)["__metadata__"]["fit_metadata"])


def record_selftest(script: str, checks: Sequence[Tuple[bool, str]], seconds: float) -> None:
    """Keep the latest self-test result of `script` in R3_selftests.json: each check's text and whether it passed,
    with the command, the time taken, the commit (and which scripts were not yet committed) and the versions."""
    import datetime
    import platform

    import torch
    import transformers

    try:
        rec = json.loads(SELFTESTS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {"schema": "R3_selftests/1", "scripts": {}}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "experiments/scripts"],
                           capture_output=True, text=True).stdout.splitlines()
    rec["scripts"][script] = {
        "command": f'CUDA_VISIBLE_DEVICES="" .venv/bin/python -I experiments/scripts/{script} --selftest',
        "at": datetime.datetime.now().isoformat(timespec="seconds"), "head": head,
        "scripts_uncommitted_then": [d[3:] for d in dirty], "seconds": round(seconds, 1),
        "ok": all(ok for ok, _ in checks), "n_checks": len(checks), "n_failed": sum(not ok for ok, _ in checks),
        "checks": [{"ok": bool(ok), "check": what} for ok, what in checks],
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    SELFTESTS_JSON.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the cache
def load_manifest() -> Dict[str, Any]:
    return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))


class CacheError(RuntimeError):
    """The cache does not hold what the manifest pins (run R3_sources_check.py for the details)."""


def jpp_items(slugs: Sequence[str] = JPP_EVALS, verify: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """J++'s eval items, {slug: [raw item dicts]}, from the pinned checkout in the cache (sha256 checked)."""
    m = load_manifest()["jpp_eval_items"]
    out = {}
    for slug in slugs:
        v = m["files"][slug]
        p = JPP_DIR / v["path"]
        if verify and (not p.is_file() or sha256_file(p) != v["sha256"]):
            raise CacheError(f"J++ eval items {slug}: {p} is missing or not the pinned file")
        items = json.loads(p.read_text(encoding="utf-8"))["items"]
        if verify and len(items) != v["n_items"]:
            raise CacheError(f"J++ eval items {slug}: {len(items)} items, manifest says {v['n_items']}")
        out[slug] = items
    return out


def wikitext_rows(verify: bool = True) -> List[str]:
    """The raw WikiText rows from row 0 to the 1,000th record (4,597 rows), unstripped, from the cache."""
    w = load_manifest()["wikitext"]
    p = REPO / w["cache_file"]
    if verify and (not p.is_file() or sha256_file(p) != w["cache_file_sha256"]):
        raise CacheError(f"WikiText rows: {p} is missing or not the pinned file")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    if verify and [r["row"] for r in rows] != list(range(w["raw_rows_cached"])):
        raise CacheError("WikiText rows are not numbered 0..n-1")
    return [r["text"] for r in rows]


def wikitext_records(verify: bool = True) -> List[str]:
    """J++'s 1,000 records (rows with len(text.strip()) >= 600, in stream order, unstripped); the first N_FIT are the
    fit prompts. Checked record by record against the manifest's row indices and sha256."""
    w = load_manifest()["wikitext"]
    rows = wikitext_rows(verify)
    kept = [(i, t) for i, t in enumerate(rows) if len(t.strip()) >= 600]
    if verify:
        got = [[i, len(t), text_sha256(t)] for i, t in kept]
        if got != w["records"]:
            raise CacheError("the WikiText records do not match the manifest's indices and sha256")
        if sequence_sha256([g[2] for g in got]) != w["records_sequence_sha256"]:
            raise CacheError("the WikiText records' sequence sha256 does not match")
    return [t for _, t in kept]


# ------------------------------------------------------------------------------------------------ jpp_lens itself
def verify_jpp_checkout() -> None:
    """Raise CacheError unless the cached checkout is the pinned commit with every tracked file as the manifest pins."""
    m = load_manifest()["jpp_lens"]
    if m["commit"] != JPP_COMMIT:
        raise CacheError(f"manifest pins jpp_lens {m['commit']}, R3_common pins {JPP_COMMIT}")
    head = subprocess.run(["git", "-C", str(JPP_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != JPP_COMMIT:
        raise CacheError(f"jpp_lens checkout at {JPP_DIR} is not at {JPP_COMMIT}")
    bad = [rel for rel, v in m["files"].items()
           if not (JPP_DIR / rel).is_file() or sha256_file(JPP_DIR / rel) != v["sha256"]]
    if bad:
        raise CacheError(f"jpp_lens files differ from the manifest: {bad[:5]}")


def import_jpp(verify: bool = True) -> None:
    """Put the pinned checkout's src/ first on sys.path (its pure functions import with torch and transformers only).
    No byte-code is written (the checkout stays as fetched)."""
    if verify:
        verify_jpp_checkout()
    sys.dont_write_bytecode = True
    if str(JPP_SRC) not in sys.path:
        sys.path.insert(0, str(JPP_SRC))


def env_subprocess(args: Sequence[str], **kw) -> subprocess.CompletedProcess:
    """Run jpp_lens's own interpreter (experiments/R3_cache/env) with the GPU hidden, the hub offline and no
    byte-code written: [env python, -I, -B, *args]."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    return subprocess.run([str(ENV_PYTHON), "-I", "-B", *map(str, args)], env=env, capture_output=True, text=True,
                          **kw)


# ------------------------------------------------------------------------------------------------ the tiny model
def real_text_config() -> Dict[str, Any]:
    """The checkpoint's text_config (config.json of the pinned snapshot; no weights are read)."""
    return json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))["text_config"]


# The shape flags of the real config that a tiny copy keeps (everything that changes which code runs), and the sizes
# it shrinks. The real model has a full-attention layer every sixth block; the tiny one keeps that pattern.
TINY_SIZES = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4,
              "num_key_value_heads": 2, "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 16,
              "vocab_size": 128, "vocab_size_per_layer_input": 128, "max_position_embeddings": 256,
              "sliding_window": 16}
REAL_FLAGS = ("hidden_activation", "hidden_size_per_layer_input", "attention_k_eq_v", "num_kv_shared_layers",
              "enable_moe_block", "use_double_wide_mlp", "final_logit_softcapping", "tie_word_embeddings",
              "attention_bias", "rms_norm_eps", "rope_parameters", "use_bidirectional_attention")


def tiny_text_config_kwargs(**overrides) -> Dict[str, Any]:
    """A tiny Gemma 4 text config: the real config with only the sizes in TINY_SIZES shrunk, so the real flags
    (gelu_pytorch_tanh, hidden_size_per_layer_input 0, attention_k_eq_v True, the 30.0 softcap, ...) are kept."""
    real = real_text_config()
    kw = {k: real[k] for k in REAL_FLAGS if k in real}
    kw.update(TINY_SIZES)
    n = overrides.get("num_hidden_layers", kw["num_hidden_layers"])
    kw["layer_types"] = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(n)]
    kw.update({"bos_token_id": 0, "eos_token_id": 30, "pad_token_id": 0})
    kw.update(overrides)
    return kw


def build_tiny_gemma4(seed: int = 0, multimodal: bool = False, **overrides):
    """A tiny Gemma 4 on the CPU in fp32, eval mode, parameter grads off: Gemma4ForCausalLM, or (multimodal) the
    Gemma4ForConditionalGeneration wrapper the real checkpoint loads as, with one-layer vision and audio towers.
    Follows jpp_lens's test fixture (d7efc0de, src/workspace_lens/tests/tiny_gemma_olmo.py; Copyright 2026 Kola
    Ayonrinde, Apache-2.0; see NOTICE)."""
    import torch

    from transformers import AutoConfig
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    kw = tiny_text_config_kwargs(**overrides)
    torch.manual_seed(seed)
    if multimodal:
        cfg = Gemma4Config(text_config=kw,
                           vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                              num_attention_heads=4, image_size=32, patch_size=16),
                           audio_config=dict(hidden_size=32, num_hidden_layers=1))
        model = Gemma4ForConditionalGeneration(cfg)
    else:
        model = Gemma4ForCausalLM(AutoConfig.for_model("gemma4_text", **kw))
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class CharTokenizer:
    """A character tokenizer for the tiny model (vocab 128 > the 31 ids used), called the way the HF tokenizers are:
    a-z -> 1..26, newline 27, space 28, anything else 29; BOS 0 (added unless add_special_tokens=False); id 30 is an
    EOS that encoding never emits. It encodes and decodes exactly as jpp_lens's test tokenizer
    (workspace_lens/tests/tokenizers.py) does, truncation included, so the two can be swapped.

    Adapted from jpp_lens at d7efc0de, src/workspace_lens/tests/tokenizers.py (Copyright 2026 Kola Ayonrinde,
    Apache-2.0; see NOTICE): the id map, the truncation rule and the decode table follow it."""

    bos_token_id = 0
    eos_token_id = 30
    all_special_ids = [0, 30]

    def _id(self, ch: str) -> int:
        ch = ch.lower()
        if "a" <= ch <= "z":
            return 1 + ord(ch) - 97
        return {"\n": 27, " ": 28}.get(ch, 29)

    def __call__(self, text: str, return_tensors: Optional[str] = None, truncation: bool = True,
                 max_length: int = 128, add_special_tokens: bool = True):
        import types

        import torch

        ids = ([0] if add_special_tokens else []) + [self._id(c) for c in text][: max_length - 1]
        if return_tensors == "pt" or add_special_tokens:
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, ids, **_kw) -> str:
        special = {0: "", 27: "\n", 28: " ", 30: "<eos>"}
        return "".join(special.get(int(i), chr(96 + int(i)) if 1 <= int(i) <= 26 else "?") for i in ids)

    def batch_decode(self, batch, **_kw) -> List[str]:
        return [self.decode(b) for b in batch]
