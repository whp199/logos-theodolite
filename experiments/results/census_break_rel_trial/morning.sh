#!/usr/bin/env bash
# Read the break_rel trial. First approve or amend /home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_read.py's GATES; write any amendment, with its time and
# reason, into /home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg before this runs (sha256sum -c below then names the file that changed).
(cd "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg" && sha256sum -c SHA256SUMS) || echo "the pre-registered files above differ from SHA256SUMS: record why in /home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="/home/eric/bin/ai/theodolite2/.claude/worktrees/census-break-rel"
/home/eric/bin/ai/theodolite2/.venv/bin/python -I "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_read.py" "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L45-break_rel-trial.jsonl" --gates L45 --fresh "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_L45_fresh.txt" \
    --retest-ref "/home/eric/bin/ai/theodolite2/atlas_data/census/runs/L45-calib_rel.jsonl" --reasked "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_L45_reasked.txt" \
    --out "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L45_verdict.json" > "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L45_verdict.txt" 2>&1 || true
/home/eric/bin/ai/theodolite2/.venv/bin/python -I "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_read.py" "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L57-break_rel-probe.jsonl" --gates L57 --out "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L57_verdict.json" > "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L57_verdict.txt" 2>&1 || true
cat "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L45_verdict.txt" "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/L57_verdict.txt"
