#!/usr/bin/env bash
# The first whole break_rel layer (L45-break_rel), read on 30 October 2026 with the trial's own reader, unchanged
# (atlas_data/census/trials/prereg, its SHA256SUMS checked 10 of 10 OK first), on the CPU. The ranking was first made
# again with the collapse rule (the run ranked itself with the code it began with, before the rule was written).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L45-break_rel.jsonl
HERE=experiments/results/census_L45_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
.venv/bin/python -I scripts/census_tools.py rank 45 --show 60
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L45_verdict.json" > "$HERE/L45_verdict.txt" || true
# the trial's 1,022 fresh neurons against this run's 2,048 fresh controls
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --fresh "$P/trial_L45_fresh.txt" --out "$HERE/L45_trial1022_verdict.json" > "$HERE/L45_trial1022_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L45_pm1_verdict.json" > "$HERE/L45_pm1_verdict.txt" || true
.venv/bin/python -I "$HERE/compare.py"
# (the copies kept here leave out each reading's named_words, to keep the record small)
