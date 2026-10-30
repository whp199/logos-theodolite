#!/usr/bin/env bash
# Layer 45 read against controls mixed from its own write directions (census.CONTROL_KINDS "columns"), 30 October 2026,
# on the CPU, with the trial's own reader unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked first).
# The controls were taken alone on the GPU:
#   census_run.py --layer 45 --neurons none --control-kind columns --seed 11 --random-controls 2048 --batch 64
#                 --max-new 48 --unpacker compiled --out-dir atlas_data/census/nulls --run L45-break_rel-columns-controls
# (begun 09:32 CDT; a GPU watchdog stopped it at 10:34 with 41,664 of 49,158 answers written, during a kernel build; the
# same command, rerun at 10:39, wrote the rest by 10:50)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
N=atlas_data/census/nulls
HERE=experiments/results/census_L45_columns
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the layer's 21,504 neurons (L45-break_rel, as written) against the 2,048 columns controls
.venv/bin/python -I scripts/census_tools.py join 45 "$N/L45-break_rel-columns-controls.jsonl" --out "$N/L45-break_rel-columns.jsonl"
.venv/bin/python -I scripts/census_tools.py rank "$N/L45-break_rel-columns.jsonl" --show 60 > "$HERE/rank.txt"
.venv/bin/python -I "$P/trial_read.py" "$N/L45-break_rel-columns.jsonl" --gates L45 --cuts 48 --out "$HERE/L45_columns_verdict.json" > "$HERE/L45_columns_verdict.txt" || true
# the +-1 answers alone (its G5b compares with a units file ranking all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$N/L45-break_rel-columns.jsonl" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L45_columns_pm1_verdict.json" > "$HERE/L45_columns_pm1_verdict.txt" || true
# the columns controls standing as neurons against the layer run's isotropic controls
.venv/bin/python -I "$HERE/mixes_as_neurons.py"
.venv/bin/python -I "$P/trial_read.py" "$N/L45-mixes-as-neurons.jsonl" --gates L45 --cuts 48 --out "$HERE/mixes_verdict.json" > "$HERE/mixes_verdict.txt" || true
# the copies kept here leave out each reading's named_words, to keep the record small
.venv/bin/python -I -c 'import json, sys
for p in sys.argv[1:]:
    v = json.load(open(p))
    for r in v["readings"].values():
        r.pop("named_words", None)
    v["note"] = "named_words left out of this copy (the reader writes them; read.sh rebuilds them)"
    json.dump(v, open(p, "w"), indent=1)' "$HERE"/L45_columns_verdict.json "$HERE"/L45_columns_pm1_verdict.json "$HERE"/mixes_verdict.json
.venv/bin/python -I "$HERE/compare.py"
# each null's controls dose by dose: what sets each one's largest
.venv/bin/python -I "$HERE/control_tails.py"
