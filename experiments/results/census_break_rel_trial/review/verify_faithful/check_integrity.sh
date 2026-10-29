#!/usr/bin/env bash
# Read-only integrity checks: prereg hashes, run hashes against the branch's runs.txt, the branch copy against the
# trial folder, timestamps (UTC), census.py at e94eea6 vs 372baec vs the worktree (loose git objects, no git command),
# the step 0 pick re-run, and the calibration layers' read gain beside their log-prob AUC.
T=/home/eric/bin/ai/theodolite2/atlas_data/census/trials
B=/home/eric/bin/ai/theodolite2/.claude/worktrees/census-break-rel/experiments/results/census_break_rel_trial
V=$(dirname "$(readlink -f "$0")")
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I -B"
echo "== sha256sum -c"; (cd $T/prereg && sha256sum -c SHA256SUMS)
echo "== run hashes"; (cd $T && sha256sum L45-break_rel-trial.jsonl L57-break_rel-probe.jsonl); cat $B/runs.txt
echo "== branch copy vs trial folder"
for f in $(cd $B && find . -type f | sort); do cmp -s "$B/$f" "$T/$f" && echo "same $f" || echo "DIFF $f"; done
echo "== timestamps (UTC)"; cat $T/prereg/prereg_at
TZ=UTC stat -c '%y %n' $T/floor_L45_break_rel.json $T/step0_pick.json $T/*.t0 $T/*.t1 $T/*.units.jsonl $T/*_verdict.json $T/prereg/__pycache__/*
for f in $T/*.t0 $T/*.t1; do echo "$(basename $f) $(TZ=UTC date -d @$(cat $f) +%FT%TZ)"; done
echo "== worktree HEAD"; cat /home/eric/bin/ai/theodolite2/.git/worktrees/census-break-rel/HEAD /home/eric/bin/ai/theodolite2/.git/refs/heads/census-break-rel
cd /tmp
echo "== census.py by commit"; $PY $V/git_blob.py | head -3; $PY $V/git_tree_diff.py
echo "== step 0 pick re-run"; $PY $T/prereg/step0_pick.py $T/floor_L45_break_rel.json | cmp - $T/step0_pick.json && echo "same as step0_pick.json"
echo "== calibration: read gain vs log-prob AUC (48 pieces)"
$PY -c "
import json
c=json.load(open('$T/prereg/calib_expect.json'))['calib_rel cut 48']
for i,e in enumerate(c['extra']):
    m=json.load(open('/home/eric/bin/ai/theodolite2/atlas_data/census/runs/L%d-calib_rel.meta.json'%e['L']))
    print(e['L'], 'read_gain', m['read_gain']['ratio'], 'cov_lp', round(c['per']['cov_lp'][i][0],3))
"
