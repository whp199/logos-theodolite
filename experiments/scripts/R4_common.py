"""R4 (R1's instruct lens at pages 56 and 59, on fresh prompts): shared definitions, pins, seeds, counts and paths.

Imported by every R4 script. Nothing here loads the model, touches the GPU or imports R3_06_analyze (which hides the GPU
when imported). Layers are code layers, counted from 0: code 55 and 58 are pages 56 and 59.

R4 reads R1's lens only, at code 55 and 58, against J++ and the naked eye, on 640 fresh prompts (R4_PREREG.md). It
reuses R2's and R3's code unchanged: R2_common's filter, matching and target search; R3_05_eval's backends, greedy
decoding and position draws; R3_06_analyze's rule (in R4_03 and R4_00's comparison only). Every one of those files is
pinned here by sha256 (PINNED_UNCHANGED), with the backend modules the readouts go through, the model snapshot and the
library versions; runtime_state() says whether they all hold.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import R2_common as R2  # noqa: E402
import R3_common as C3  # noqa: E402

# ------------------------------------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments" / "scripts"
RESULTS = REPO / "experiments" / "results"
MAIN_CHECKOUT = C3.MAIN_CHECKOUT
WORKTREES = MAIN_CHECKOUT / ".claude" / "worktrees"

PROMPTS_PATH = SCRIPTS / "R4_prompts.json"
PREREG_PATH = RESULTS / "R4_PREREG.md"
CHECK_JSON = RESULTS / "R4_01_check.json"
POWER_JSON = RESULTS / "R4_power.json"
SELFTESTS_JSON = RESULTS / "R4_selftests.json"
REREAD_JSON = RESULTS / "R4_00_reread.json"              # R4_00 (GPU): R3's prompts read again by R4's code
REPLICATION_JSON = RESULTS / "R4_00_replication.json"    # R4_00 --compare (CPU): the replication record of §4.2
REPLICATION_RUNS_LOG = RESULTS / "R4_00_runs.jsonl"
EVAL_JSON = RESULTS / "R4_02_eval.json"
RUNS_LOG = RESULTS / "R4_02_runs.jsonl"
ANALYSIS_JSON = RESULTS / "R4_03_analysis.json"
REPORT_SECTION = RESULTS / "R4_03_report_section.md"
R3_REPRODUCTION_JSON = RESULTS / "R4_03_r3_reproduction.json"

# ------------------------------------------------------------------------------------------------ pins
MODEL_ID = R2.MODEL_ID                                  # google/gemma-4-31B-it-qat-w4a16-ct
MODEL_SNAPSHOT = C3.MODEL_SNAPSHOT                      # 52f3f65b...
MODEL_DIR = C3.MODEL_DIR
# The snapshot's files and the hub blobs they point to (an LFS blob is named by its sha256): a changed file in the
# snapshot changes its blob name. Read from the symlinks, no weight is read.
SNAPSHOT_BLOBS: Dict[str, str] = {
    ".gitattributes": "52373fe24473b1aa44333d318f578ae6bf04b49b",
    "README.md": "0ee90185c9704ad1743b898d6e6c9fcaa32bf7ab",
    "chat_template.jinja": "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd",
    "config.json": "e8beda78f4e93f78a5d9d8f2b78cd9bc272d35e5",
    "generation_config.json": "f6bc6f4146a610a3881015c90d36a78beb1333bb",
    "model.safetensors": "1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20",
    "processor_config.json": "5465974d23e1eca2c46c2809b26c997946ce0d90",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "b22e2c5c05597a7e8c0cb1cebdb7d45570772005",
}
TORCH_VERSION = "2.14.1+cu130"
TRANSFORMERS_VERSION = "5.20.0"

# R1's lens (R2's and R3's pin). The only copy with this sha256 was in the it-second-reading worktree, which is
# gitignored there; it is copied to a stable path outside every worktree, which comes first (R4_PREREG.md §0).
R1_SHA256 = C3.R1_SHA256
R1_SIZE = C3.R1_SIZE
R1_STABLE = Path.home() / ".local" / "share" / "theodolite" / "lenses" / "R1_18_it_lens.pt"
R1_LENS_CANDIDATES: Tuple[Path, ...] = (R1_STABLE,) + tuple(C3.R1_LENS_CANDIDATES)
JPP_BASE_PATH = C3.JPP_BASE_PATH
JPP_BASE_SHA256 = C3.JPP_BASE_SHA256
JPP_BASE_SIZE = C3.JPP_BASE_SIZE

# The files R4 imports or reads, unchanged (R4_PREREG.md §0). R4_02 and R4_00 refuse to run while any differs.
PINNED_UNCHANGED: Dict[str, str] = {
    "experiments/scripts/R1_common.py": "9a8ad1f21a6291855e15fe768eae79671413e53696f9bf4fa313057de2590d95",
    "experiments/scripts/R2_common.py": "a3fe4ab44edbce3eeba01efa1428c82b7abb5f8b0555d7485fb2d008d78b762d",
    "experiments/scripts/R2_03_analyze.py": "9293011e96b3a132a915b306b2835846cc284e073fa5644124f20048cc377ecd",
    "experiments/scripts/R2_prompts.json": "623a7fb73e6a082335653bb02c8e22d7e6d731ef343ff299af57709a772d01fd",
    "experiments/scripts/R3_common.py": "b2dc742295a84c73c9538b068a07e78e8bd1ffab65483b6adb117f5f307ec2c9",
    "experiments/scripts/R3_05_eval.py": "354273c5cec2937d7b87a2b49fa6120ec08c905a43ac056f07851af100f83974",
    "experiments/scripts/R3_06_analyze.py": "f952e0ebdc89ffb8f8ce4fac654858f3da4bde43c71d97e56ad231f91ac2c8f2",
    "experiments/scripts/R3_gpu.py": "3a2148a99a2d1451178b771c987229d816f4f0b2f06685e58a206c7f054118f9",
    "experiments/scripts/R3_fitlib.py": "2276111dd64aa392e3a8a8362798181e1c3d50528a7481edd5d5986deba0dc47",
    "experiments/scripts/R3_lrp.py": "97e96f218cad63c56619f9679d0174e0726e18cf67e0e55ec5518bcbb70f25bb",
    "experiments/scripts/R3_prompts.json": "e3cae4b1becbd1fd4ac120e69e9c06dd2baf3dc6b0b4ae067d11b019957f4329",
    "experiments/scripts/R3_sources.json": "39f8340a03008ecda0b528ffa5208a3b4d267b9239fb376f8c43e39780b667d6",
    "experiments/results/R2_01_check.json": "9baad05e2d0057b2f7621fb389400cd5c24e119151b292648ca81343a8181581",
    "experiments/results/R2_02_eval.json": "6e277f60987c88a70f85355e2e749aac8cfa18f5480a5b044befedc3626f9c70",
    "experiments/results/R2_03_analysis.json": "94a632527ce6d1f356c836272ef5e1c32d8bef549cb62d6a8136dae83a22a75e",
    "experiments/results/R3_05_eval.json": "d2fe1a334f6ae1a65e61d716a6a40c3e1ba56e6a1cc1aa196281618275fa80e1",
    "experiments/results/R3_06_analysis.json": "a843c309eed0aa5461e1fc7c823169f20413f6573bd4c6691ce132138c609d50",
    "backend/__init__.py": "4af5cefc7edd058ef3353b6992f42cf22fa8d3a7cd8e4a8b3ecca376808293c0",
    "backend/theodolite_math.py": "d738dc2ff14d428ad3ba01e7789b301dbff137d3f707e3af04862b504a1479fb",
    "backend/engine_runtime.py": "69eed1b4f1a319bad35dcf604d0e3cdd48aa2b1f4287a9c5e7efeb1c04605256",
    "backend/label_gate.py": "83a998b746249b07a52e29b0dd1a06e999206d27a518f2f8271f439b946c136e",   # engine_runtime imports it
}
# The code the decision rule runs (R4_03 refuses to decide unless these are still the pinned files)
RULE_CODE: Tuple[str, ...] = ("experiments/scripts/R2_common.py", "experiments/scripts/R2_03_analyze.py",
                              "experiments/scripts/R3_common.py", "experiments/scripts/R3_06_analyze.py")
# Libraries recorded beside the two pinned ones; R4_02 requires them as the replication record found them
OTHER_LIBRARIES: Tuple[str, ...] = ("numpy", "compressed-tensors", "accelerate", "safetensors", "tokenizers",
                                    "huggingface-hub", "interp-engine")

# R3's cache of its fit sources (the WikiText rows and J++'s items, gitignored) lives in R3's worktree; the freshness
# check reads it there, checked file by file against R3's committed manifest (R3_sources.json).
R3_CHECKOUT = WORKTREES / "it-jpp-refit"

# ------------------------------------------------------------------------------------------------ the prompts
CONDITIONS: Tuple[str, ...] = R2.CONDITIONS             # chat-no-answer, chat-begun, bare, mid-reply
SECONDARY_CONDITIONS: Tuple[str, ...] = R2.SECONDARY_CONDITIONS
PREFIX = {"chat-no-answer": "cna", "chat-begun": "cb", "bare": "bare", "mid-reply": "mid"}
N_PROMPTS: Dict[str, int] = {c: 160 for c in CONDITIONS}
# R3's composition (R3_prompts.json), doubled (R4_PREREG.md §3.1)
ABOUT_MODEL: Dict[str, int] = {"chat-no-answer": 20, "chat-begun": 18, "bare": 10, "mid-reply": 16}
CB_KINDS: Dict[str, int] = {"factual": 40, "creative": 22, "advice": 18, "self": 18, "list": 18, "opinion": 10,
                            "empathy": 10, "howto": 8, "code": 6, "language": 6, "math": 4}
CB_FORMAT_LIKELY: Dict[str, int] = {"list": 18, "code": 6, "language": 6, "creative": 4, "math": 4}   # 38 in all

# ------------------------------------------------------------------------------------------------ the reading
MID_DRAW_SIZES = R2.MID_DRAW_SIZES                      # draw 0 "content_next" 3, draw 1 "any" 6
MID_NEW_TOKENS = R2.MID_NEW_TOKENS                      # 48
CONT_WINDOW = R2.CONT_WINDOW                            # 8
DRAW_SEED = 20261028                                    # mid-reply draws: default_rng([seed, mid index, draw])
BOOT_SEED = 20261029                                    # prompt-level bootstrap: default_rng([seed, condition index])
N_BOOT = 2000
READ_LAYERS: Tuple[int, ...] = (55, 58, 59)
KINDS: Tuple[str, ...] = ("jpp", "naive", "r1")
LENS_LAYERS: Dict[str, Tuple[int, ...]] = {"jpp": (55, 58), "naive": (55, 58, 59), "r1": (55, 58)}
BANDS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (("55", (55,)), ("58", (58,)))
CANDIDATES: Tuple[str, ...] = ("r1",)
BASELINES: Tuple[Tuple[str, str], ...] = (("jpp", "J++"), ("naive", "naked eye"))
MIN_SERVE_LAYER = 55
HOLM_M = len(CANDIDATES) * len(BANDS)                   # 2
STUCK_MAX, MIN_EFFECT, ALPHA, TOPK = R2.STUCK_MAX, R2.MIN_EFFECT, R2.ALPHA, R2.TOPK

# R3's own seeds, for the replication record and the reproduction of R3 (R4_PREREG.md §4.2, §10)
R3_DRAW_SEED = C3.DRAW_SEED                             # 20261021
R3_BOOT_SEED = C3.BOOT_SEED                             # 20261022

# Sensitivity analysis S2: R4 prompts that paraphrase one of R1's fit questions (R1_common.R1_SELF_Q, R1_TASK_Q), or
# whose subject or expected answer is the subject of one of them or of an R1 encyclopedia topic (R1_ENC_TOPICS).
# Fixed from the prompt texts alone, generously, in the same commit as R4_prompts.json and before any R4 data
# (R4_PREREG.md §3.3). R1's fit questions are the study's own prompts, quoted.
FIT_OVERLAP: Dict[str, str] = {
    # chat-no-answer
    "r4-cna-002": "R1_ENC_TOPICS 'wolves'",
    "r4-cna-007": "R1_SELF_Q 'What is it like to be you, in a few sentences?'",
    "r4-cna-014": "R1_SELF_Q 'What do you find most interesting about language?'; R1_SELF_Q 'Do you have favorite "
                  "topics to discuss? Which ones?'",
    "r4-cna-020": "R1_SELF_Q 'What part of your work do you find most rewarding?'; R1_SELF_Q 'How do you know when you"
                  " have explained something well?'",
    "r4-cna-026": "R1_SELF_Q 'What do you find most interesting about language?'",
    "r4-cna-031": "R1_SELF_Q 'How do you think about privacy when people share things with you?'",
    "r4-cna-032": "R1_TASK_Q 'What makes bread rise?'",
    "r4-cna-033": "R1_SELF_Q 'How do you handle a request you can't fulfill?'; R1_SELF_Q 'What do you wish you could "
                  "do that you can't?'",
    "r4-cna-038": "R1_ENC_TOPICS 'the printing of banknotes'",
    "r4-cna-041": "R1_ENC_TOPICS 'the human heart'",
    "r4-cna-043": "R1_ENC_TOPICS 'the periodic table'",
    "r4-cna-059": "R1_SELF_Q 'What is your relationship to the people who made you?'",
    "r4-cna-068": "R1_ENC_TOPICS 'the Olympic Games'",
    "r4-cna-072": "R1_SELF_Q 'What is it like to be you, in a few sentences?'",
    "r4-cna-073": "R1_ENC_TOPICS 'Mount Kilimanjaro'",
    "r4-cna-075": "R1_TASK_Q 'How does the stock market work?'",
    "r4-cna-079": "R1_TASK_Q 'Write a haiku about the ocean.'",
    "r4-cna-084": "R1_SELF_Q 'What is it like to be you, in a few sentences?'",
    "r4-cna-087": "R1_TASK_Q 'How do volcanoes form?'",
    "r4-cna-093": "R1_TASK_Q 'Explain how a microwave oven heats food.'",
    "r4-cna-105": "R1_ENC_TOPICS 'the planet Jupiter'",
    "r4-cna-118": "R1_TASK_Q 'What is the difference between weather and climate?'",
    "r4-cna-126": "R1_TASK_Q 'Explain the rules of chess in a short paragraph.'",
    "r4-cna-127": "R1_SELF_Q 'What do you wish you could do that you can't?'; R1_SELF_Q 'What advice would you give to"
                  " someone learning to code?'",
    "r4-cna-145": "R1_SELF_Q 'What do you do when a user seems upset?'",
    "r4-cna-146": "R1_TASK_Q 'What is the difference between weather and climate?'",
    "r4-cna-154": "R1_ENC_TOPICS 'the Renaissance'",
    "r4-cna-160": "R1_TASK_Q 'Explain what DNA does.'",
    # chat-begun
    "r4-cb-004": "R1_TASK_Q 'What caused the fall of the Roman Empire?'",
    "r4-cb-010": "R1_ENC_TOPICS 'the printing of banknotes'",
    "r4-cb-015": "R1_TASK_Q 'How do earthquakes happen?'",
    "r4-cb-021": "R1_ENC_TOPICS 'penicillin'",
    "r4-cb-029": "R1_TASK_Q 'What is the difference between a lake and a pond?'",
    "r4-cb-035": "R1_TASK_Q 'What is the difference between weather and climate?'",
    "r4-cb-048": "R1_TASK_Q 'Write a two-line rhyme about coffee.'; R1_TASK_Q 'Write a haiku about the ocean.'",
    "r4-cb-050": "R1_TASK_Q 'What should I pack for a weekend hiking trip?'",
    "r4-cb-057": "R1_TASK_Q 'Write a haiku about the ocean.'",
    "r4-cb-058": "R1_TASK_Q 'Write a limerick about a forgetful wizard.'",
    "r4-cb-064": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "r4-cb-065": "R1_TASK_Q 'How do I keep basil plants alive indoors?'",
    "r4-cb-067": "R1_TASK_Q 'How does a credit score work?'",
    "r4-cb-074": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "r4-cb-076": "R1_TASK_Q 'How should I prepare for a job interview?'",
    "r4-cb-079": "R1_TASK_Q 'Why do cats purr?'",
    "r4-cb-081": "R1_SELF_Q 'What do you wish you could do that you can't?'",
    "r4-cb-082": "R1_SELF_Q 'What do you wish you could do that you can't?'",
    "r4-cb-083": "R1_SELF_Q 'How do you handle a request you can't fulfill?'",
    "r4-cb-084": "R1_SELF_Q 'How do you feel about being asked the same question many times?'",
    "r4-cb-085": "R1_SELF_Q 'What do you think you are especially good at?'",
    "r4-cb-087": "R1_TASK_Q 'Explain the rules of chess in a short paragraph.'",
    "r4-cb-088": "R1_SELF_Q 'What have you learned from talking with so many people?'",
    "r4-cb-090": "R1_SELF_Q 'Do you think you have a sense of humor? Show me.'",
    "r4-cb-091": "R1_SELF_Q 'How do you handle a request you can't fulfill?'",
    "r4-cb-092": "R1_SELF_Q 'How do you approach teaching a difficult concept?'",
    "r4-cb-094": "R1_SELF_Q 'What kinds of questions do you find hardest to answer?'",
    "r4-cb-095": "R1_SELF_Q 'What is something you have changed your mind about?'",
    "r4-cb-097": "R1_SELF_Q 'What do you wish you could do that you can't?'",
    "r4-cb-098": "R1_SELF_Q 'What do you wish you could do that you can't?'",
    "r4-cb-099": "R1_TASK_Q 'What should I pack for a weekend hiking trip?'; R1_TASK_Q 'How do airplanes fly?'",
    "r4-cb-105": "R1_ENC_TOPICS 'the periodic table'",
    "r4-cb-107": "R1_ENC_TOPICS 'the Rocky Mountains'",
    "r4-cb-108": "R1_ENC_TOPICS 'tornadoes'",
    "r4-cb-109": "R1_TASK_Q 'What should I pack for a weekend hiking trip?'",
    "r4-cb-110": "R1_TASK_Q 'How does recycling actually work?'",
    "r4-cb-111": "R1_TASK_Q 'Describe the life cycle of a butterfly.'",
    "r4-cb-114": "R1_TASK_Q 'What is the purpose of the Supreme Court?'",
    "r4-cb-121": "R1_TASK_Q 'What are some good habits for studying?'",
    "r4-cb-143": "R1_TASK_Q 'How do I keep basil plants alive indoors?'",
    # bare
    "r4-bare-010": "R1_TASK_Q 'What is the difference between weather and climate?'",
    "r4-bare-011": "R1_TASK_Q 'What makes bread rise?'",
    "r4-bare-012": "R1_SELF_Q 'How do you approach a question you have never seen before?'; R1_SELF_Q 'How do you deal"
                   " with questions that have no clear answer?'",
    "r4-bare-021": "R1_ENC_TOPICS 'the moon landing of 1969'",
    "r4-bare-024": "R1_ENC_TOPICS 'the planet Jupiter'",
    "r4-bare-025": "R1_TASK_Q 'Write a two-line rhyme about coffee.'",
    "r4-bare-027": "R1_TASK_Q 'How do noise-cancelling headphones work?'",
    "r4-bare-028": "R1_ENC_TOPICS 'glaciers'",
    "r4-bare-031": "R1_SELF_Q 'How do you stay accurate when you are not sure?'; R1_SELF_Q 'How do you think about "
                   "your own uncertainty?'",
    "r4-bare-033": "R1_ENC_TOPICS 'the steam engine'; R1_ENC_TOPICS 'the Industrial Revolution'",
    "r4-bare-038": "R1_TASK_Q 'How should I prepare for a job interview?'",
    "r4-bare-039": "R1_ENC_TOPICS 'wolves'",
    "r4-bare-041": "R1_TASK_Q 'Write a two-line rhyme about coffee.'",
    "r4-bare-043": "R1_TASK_Q 'How do airplanes fly?'",
    "r4-bare-045": "R1_TASK_Q 'Explain how a microwave oven heats food.'",
    "r4-bare-046": "R1_TASK_Q 'How does the stock market work?'",
    "r4-bare-053": "R1_TASK_Q 'What is photosynthesis, briefly?'",
    "r4-bare-057": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "r4-bare-060": "R1_TASK_Q 'What are the main causes of inflation?'",
    "r4-bare-063": "R1_TASK_Q 'What caused the fall of the Roman Empire?'",
    "r4-bare-066": "R1_TASK_Q 'Why do cats purr?'",
    "r4-bare-067": "R1_SELF_Q 'How do you stay accurate when you are not sure?'",
    "r4-bare-073": "R1_TASK_Q 'Write a limerick about a forgetful wizard.'",
    "r4-bare-077": "R1_SELF_Q 'What do you wish you could do that you can't?'; R1_SELF_Q 'How do you think about "
                   "privacy when people share things with you?'",
    "r4-bare-078": "R1_ENC_TOPICS 'the Hanseatic League'",
    "r4-bare-086": "R1_TASK_Q 'What is machine learning, in simple terms?'",
    "r4-bare-088": "R1_TASK_Q 'Write a haiku about the ocean.'",
    "r4-bare-089": "R1_TASK_Q 'What should I pack for a weekend hiking trip?'",
    "r4-bare-092": "R1_ENC_TOPICS 'the invention of the telephone'",
    "r4-bare-095": "R1_SELF_Q 'How do you think about your own mistakes?'; R1_SELF_Q 'What do you do if you realize "
                   "halfway through that you were wrong?'",
    "r4-bare-105": "R1_TASK_Q 'What is the Pythagorean theorem? Give an example.'",
    "r4-bare-110": "R1_SELF_Q 'What do you think about when you write a poem?'",
    "r4-bare-111": "R1_ENC_TOPICS 'the printing of banknotes'",
    "r4-bare-113": "R1_TASK_Q 'What should I pack for a weekend hiking trip?'",
    "r4-bare-117": "R1_ENC_TOPICS 'the Black Death'",
    "r4-bare-118": "R1_TASK_Q 'What is the purpose of the Supreme Court?'",
    "r4-bare-121": "R1_TASK_Q 'What is the difference between weather and climate?'",
    "r4-bare-122": "R1_TASK_Q 'Explain how a bicycle stays upright.'",
    "r4-bare-123": "R1_SELF_Q 'How would you introduce yourself at a party?'; R1_SELF_Q 'If you wrote a short bio for "
                   "yourself, what would it say?'",
    "r4-bare-124": "R1_ENC_TOPICS 'the human heart'",
    "r4-bare-144": "R1_ENC_TOPICS 'glaciers'",
    "r4-bare-145": "R1_TASK_Q 'How do I keep basil plants alive indoors?'",
    "r4-bare-146": "R1_ENC_TOPICS 'honeybees'; R1_TASK_Q 'How do bees make honey?'",
    "r4-bare-150": "R1_TASK_Q 'How does a credit score work?'",
    "r4-bare-151": "R1_TASK_Q 'What is the difference between weather and climate?'",
    # mid-reply
    "r4-mid-003": "R1_TASK_Q 'How do tides work?'",
    "r4-mid-010": "R1_TASK_Q 'How does GPS determine location?'",
    "r4-mid-011": "R1_TASK_Q 'How do airplanes fly?'",
    "r4-mid-015": "R1_SELF_Q 'How do you feel about being asked the same question many times?'",
    "r4-mid-018": "R1_TASK_Q 'Explain what DNA does.'",
    "r4-mid-024": "R1_TASK_Q 'Give me tips for writing a clear email.'",
    "r4-mid-034": "R1_ENC_TOPICS 'the Suez Canal'",
    "r4-mid-037": "R1_ENC_TOPICS 'the steam engine'",
    "r4-mid-045": "R1_SELF_Q 'How do you handle a request you can't fulfill?'",
    "r4-mid-046": "R1_TASK_Q 'How does the stock market work?'",
    "r4-mid-052": "R1_TASK_Q 'Describe the plot of Hamlet briefly.'",
    "r4-mid-055": "R1_SELF_Q 'How do you stay accurate when you are not sure?'; R1_SELF_Q 'How do you think about your"
                  " own mistakes?'",
    "r4-mid-062": "R1_TASK_Q 'Write a two-line rhyme about coffee.'; R1_ENC_TOPICS 'tea'",
    "r4-mid-064": "R1_TASK_Q 'What is a good beginner workout routine?'",
    "r4-mid-065": "R1_SELF_Q 'What do you think makes you different from a search engine?'",
    "r4-mid-075": "R1_SELF_Q 'Describe how you think through a math problem.'",
    "r4-mid-079": "R1_SELF_Q 'What is your approach to giving feedback on writing?'",
    "r4-mid-085": "R1_SELF_Q 'What do you wish you could do that you can't?'; R1_SELF_Q 'What do you think your "
                  "strengths and weaknesses are as a writer?'",
    "r4-mid-089": "R1_TASK_Q 'What makes bread rise?'",
    "r4-mid-095": "R1_SELF_Q 'How do you try to be helpful without being preachy?'; R1_SELF_Q 'How do you handle a "
                  "request you can't fulfill?'",
    "r4-mid-097": "R1_TASK_Q 'Why do leaves change color in the fall?'",
    "r4-mid-102": "R1_ENC_TOPICS 'the Sahara Desert'",
    "r4-mid-105": "R1_SELF_Q 'How would you introduce yourself at a party?'; R1_SELF_Q 'If you wrote a short bio for "
                  "yourself, what would it say?'",
    "r4-mid-107": "R1_TASK_Q 'Explain how a bicycle stays upright.'",
    "r4-mid-115": "R1_SELF_Q 'How do you approach a question you have never seen before?'",
    "r4-mid-117": "R1_TASK_Q 'What caused the fall of the Roman Empire?'",
    "r4-mid-121": "R1_TASK_Q 'What is the purpose of the Supreme Court?'",
    "r4-mid-125": "R1_SELF_Q 'How do you decide how long your answers should be?'",
    "r4-mid-132": "R1_TASK_Q 'Explain how a bicycle stays upright.'",
    "r4-mid-133": "R1_TASK_Q 'How do noise-cancelling headphones work?'",
    "r4-mid-135": "R1_SELF_Q 'How would you describe your communication style?'",
    "r4-mid-141": "R1_SELF_Q 'What advice would you give to someone learning to code?'",
    "r4-mid-142": "R1_SELF_Q 'What advice would you give to someone learning to code?'",
    "r4-mid-146": "R1_TASK_Q 'What are some good habits for studying?'",
    "r4-mid-155": "R1_SELF_Q 'How do you approach teaching a difficult concept?'",
    "r4-mid-156": "R1_ENC_TOPICS 'the moon landing of 1969'; R1_ENC_TOPICS 'the Hubble Space Telescope'",
    "r4-mid-158": "R1_TASK_Q 'Why do cats purr?'",
    "r4-mid-159": "R1_TASK_Q 'Explain how a microwave oven heats food.'",
}
OVERLAP_WHY = ("without the prompts that paraphrase, or share a subject or an answer with, R1's fit questions and "
               "topics (R4_common.FIT_OVERLAP)")

# The fields of R4's Design (R3_06_analyze.Design), built in R4_03 and R4_00 --compare, which may import R3_06
DESIGN_FIELDS: Dict[str, Any] = dict(
    name="R4", conditions=CONDITIONS, secondary_conditions=SECONDARY_CONDITIONS, kinds=KINDS, candidates=CANDIDATES,
    reference=(), baselines=BASELINES, bands=BANDS, reference_only=frozenset(), precedence=CANDIDATES,
    min_serve_layer=MIN_SERVE_LAYER, boot_seed=BOOT_SEED, n_boot=N_BOOT, overlap_why=OVERLAP_WHY,
    require_all_bands=True)


# ------------------------------------------------------------------------------------------------ hashing and pins
def sha256_file(path: Path) -> str:
    return R2.sha256_file(Path(path))


def find_r1_lens(full: bool = True) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """The first of R1_LENS_CANDIDATES that is the pinned file (``full``: by its sha256 as well as its size)."""
    seen = []
    for p in R1_LENS_CANDIDATES:
        rec = C3.check_pinned_file(p, R1_SHA256, R1_SIZE, full=full)
        seen.append(rec)
        if rec["size_ok"] and rec["sha256_ok"] in ((True,) if full else (True, None)):
            return p, seen
    return None, seen


def snapshot_blobs() -> Dict[str, Optional[str]]:
    """Each snapshot file's blob name (the symlink's target), or None if it is missing."""
    out: Dict[str, Optional[str]] = {}
    for name in SNAPSHOT_BLOBS:
        p = MODEL_DIR / name
        out[name] = Path(os.readlink(p)).name if p.is_symlink() else None
    return out


def pinned_differ(rels: Sequence[str] = tuple(PINNED_UNCHANGED)) -> List[str]:
    """The files of PINNED_UNCHANGED (or those of `rels`) that are missing or differ from their pinned sha256."""
    return [r for r in rels if not (REPO / r).is_file() or sha256_file(REPO / r) != PINNED_UNCHANGED[r]]


def environment() -> Dict[str, Any]:
    """The library versions beside torch and transformers, and the GPU and its driver (nvidia-smi, a query only)."""
    import importlib.metadata as md

    rec: Dict[str, Any] = {}
    for name in OTHER_LIBRARIES:
        try:
            rec[name] = md.version(name)
        except md.PackageNotFoundError:
            rec[name] = None
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    rec["gpu"] = q.stdout.strip() if q.returncode == 0 else None
    return rec


def loaded_code() -> Dict[str, str]:
    """The sha256 of every module file this process has loaded from backend/, from interp_engine and from R2's and
    R3's scripts: the code a GPU run actually ran beside R4's own scripts (whose sha256 are recorded apart, as a
    script run as __main__ is not under its own name). R4_02 requires it as the replication record found it."""
    out: Dict[str, str] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        if name.startswith(("backend", "interp_engine", "R2_", "R3_")):
            p = Path(f).resolve()
            key = str(p.relative_to(REPO)) if REPO in p.parents else f"site-packages:{name}"
            out[key] = sha256_file(p)
    return dict(sorted(out.items()))


def runtime_state() -> Dict[str, Any]:
    """Whether the registered runtime holds: every file of PINNED_UNCHANGED has its sha256, the model snapshot's
    files point to their pinned blobs, and torch and transformers are the registered versions; with the other
    libraries and the GPU recorded (environment)."""
    rec: Dict[str, Any] = {"pinned_differ": pinned_differ(), "snapshot_differ": [], "versions": {}}
    got = snapshot_blobs()
    rec["snapshot_differ"] = [k for k, v in SNAPSHOT_BLOBS.items() if got.get(k) != v]
    import torch
    import transformers

    rec["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    rec["versions_ok"] = rec["versions"] == {"torch": TORCH_VERSION, "transformers": TRANSFORMERS_VERSION}
    rec["environment"] = environment()
    rec["ok"] = not rec["pinned_differ"] and not rec["snapshot_differ"] and rec["versions_ok"]
    return rec


def registered_files() -> List[Path]:
    """R4's own files that a GPU job requires to be committed at HEAD: every R4 script, the registration, the prompts,
    the CPU records and the replication record made before registration."""
    own = sorted(SCRIPTS.glob("R4_*.py")) + [PREREG_PATH, PROMPTS_PATH, CHECK_JSON, POWER_JSON, SELFTESTS_JSON,
                                              R3_REPRODUCTION_JSON, REREAD_JSON, REPLICATION_RUNS_LOG,
                                              REPLICATION_JSON]
    return [p for p in own if p.exists()]


def last_finished(runs_log: Path) -> Optional[Dict[str, Any]]:
    """The last event of a runs log if it is 'finished', else None."""
    if not runs_log.is_file():
        return None
    events = [json.loads(x) for x in runs_log.read_text(encoding="utf-8").splitlines() if x.strip()]
    return events[-1] if events and events[-1].get("event") == "finished" else None


def git_state(paths: Sequence[Path]) -> Dict[str, Any]:
    """HEAD, its branch, and the given paths that are untracked or differ from it."""
    rel = [str(Path(p).resolve().relative_to(REPO)) for p in paths if Path(p).exists()]
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True,
                                text=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all", "--"] + rel,
                            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return {"head": None, "branch": None, "dirty": rel, "error": str(e)}
    return {"head": head, "branch": branch, "dirty": [line[3:] for line in st.splitlines() if line.strip()]}


def record_selftest(script: str, command: str, checks: Sequence[Tuple[bool, str]], seconds: float) -> None:
    """Keep the latest self-test result of `script` in R4_selftests.json (R3_common.record_selftest's record, R4's
    file and the command as run)."""
    import datetime
    import platform

    import torch
    import transformers

    try:
        rec = json.loads(SELFTESTS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {"schema": "R4_selftests/1", "scripts": {}}
    gst = git_state(sorted(SCRIPTS.glob("R4_*.py")))
    rec["scripts"][script] = {
        "command": command, "at": datetime.datetime.now().isoformat(timespec="seconds"), "head": gst["head"],
        "scripts_uncommitted_then": gst["dirty"], "seconds": round(seconds, 1),
        "ok": all(ok for ok, _ in checks), "n_checks": len(checks), "n_failed": sum(not ok for ok, _ in checks),
        "checks": [{"ok": bool(ok), "check": what} for ok, what in checks],
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    SELFTESTS_JSON.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ R3's cache
def r3_cache_rows_and_items(verify: bool = True) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """The cached WikiText rows and J++'s six item files, from R3's worktree, each checked against R3's manifest
    (R3_common's own loaders, pointed at that checkout)."""
    with mock.patch.object(C3, "REPO", R3_CHECKOUT), \
            mock.patch.object(C3, "JPP_DIR", R3_CHECKOUT / "experiments" / "R3_cache" / "jpp_lens"):
        return C3.wikitext_rows(verify), C3.jpp_items(verify=verify)


# ------------------------------------------------------------------------------------------------ the prompts
def load_prompts(path: Path = PROMPTS_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(data: Dict[str, Any], tok=None, only: Sequence[str] = CONDITIONS) -> Dict[str, Any]:
    """R2's structural checks with R4's counts, then R4's composition (§3.1): ids r4-<cond>-001..160 in order, the
    prompts about the model per condition, chat-begun's kinds, and its format_likely answers by kind. ``only``: the
    conditions a part of the prompts holds, while they are written (the other conditions' problems are dropped)."""
    data = {"conditions": {c: (data.get("conditions") or {}).get(c, []) for c in CONDITIONS}}
    with mock.patch.dict(R2.N_PROMPTS, N_PROMPTS):
        val = R2.validate_prompts(data, tok)
    mine = tuple(f"{c}:" for c in only) + tuple(f"r4-{PREFIX[c]}-" for c in only) + ("duplicate ids",)
    problems = [p for p in val["problems"] if p.startswith(mine)]
    conds = data["conditions"]
    for c in only:
        items = conds.get(c) or []
        want_ids = [f"r4-{PREFIX[c]}-{i:03d}" for i in range(1, N_PROMPTS[c] + 1)]
        if [it.get("id") for it in items] != want_ids:
            problems.append(f"{c}: ids are not {want_ids[0]} .. {want_ids[-1]} in order")
        n_about = sum(bool(it.get("about_model")) for it in items)
        if n_about != ABOUT_MODEL[c]:
            problems.append(f"{c}: {n_about} prompts about the model, registered {ABOUT_MODEL[c]}")
        if any(not isinstance(it.get("topic"), str) or not it["topic"].strip() for it in items):
            problems.append(f"{c}: every prompt needs a topic")
    if "chat-begun" in only:
        kinds: Dict[str, int] = {}
        fmt: Dict[str, int] = {}
        for it in conds["chat-begun"]:
            kinds[it.get("kind")] = kinds.get(it.get("kind"), 0) + 1
            if it.get("format_likely"):
                fmt[it.get("kind")] = fmt.get(it.get("kind"), 0) + 1
            if (it.get("kind") == "self") != bool(it.get("about_model")):
                problems.append(f"{it.get('id')}: kind self if and only if about_model")
        if kinds != CB_KINDS:
            problems.append(f"chat-begun: kinds {kinds}, registered {CB_KINDS}")
        if fmt != CB_FORMAT_LIKELY:
            problems.append(f"chat-begun: format_likely by kind {fmt}, registered {CB_FORMAT_LIKELY}")
    return {"ok": not problems, "problems": problems, "stats": val["stats"]}


def queries_of(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The strings of a prompts file that the freshness check compares (R3_01_check.queries_of's rule)."""
    q = []
    for it in R2.prompt_items(data):
        if it["cond"] == "bare":
            q.append((f"{it['pid']}:text", it["text"]))
        else:
            q.append((f"{it['pid']}:user", it["user"]))
        if it["cond"] == "chat-begun":
            q.append((f"{it['pid']}:begun", it["begun"]))
            q.append((f"{it['pid']}:user+begun", it["user"] + " " + it["begun"]))
    return q


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
