#!/usr/bin/env bash
# The second whole break_rel layer (L40-break_rel, the home 5090, 30-31 October 2026), read on 31 October with the
# trial's own reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked first), on the CPU, as layer
# 45's was (../census_L45_whole/read.sh). No gates were fixed for layer 40 before it ran: it is read against layer
# 45's, after the fact. The run ranked itself with the collapse rule, so its ranking is read as it stands.
# Peak memory 6.1 GB, 19 minutes beside a GPU job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L40-break_rel.jsonl
HERE=experiments/results/census_L40_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L40_verdict.json" > "$HERE/L40_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L40_pm1_verdict.json" > "$HERE/L40_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
