#!/usr/bin/env bash
# Read-only: what the worktree the morning read imported (THEODOLITE_TRIAL_REPO) holds against the commit the runs
# ran from (e94eea6). --no-optional-locks: git writes nothing (no index refresh). Output to this folder.
R=/home/eric/bin/ai/theodolite2; W=$R/.claude/worktrees/census-break-rel
G=$(sed 's/gitdir: //' $W/.git)
echo "worktree HEAD ref: $(cat $G/HEAD) -> $(cat $R/.git/refs/heads/census-break-rel 2>/dev/null)"
echo "reflog:"; cut -d' ' -f1,2,5- $G/logs/HEAD | cut -c1-200
echo "--- files changed under backend/ and scripts/ since e94eea6 (working tree):"
git --no-optional-locks -C $W diff --stat e94eea6 -- backend scripts
echo "--- backend/census.py, e94eea6 -> working tree:"
git --no-optional-locks -C $W diff e94eea6 -- backend/census.py
echo "--- names the reader uses that the diff touches:"
grep -noE "DOSES_FROM|DOSES_CALIBRATED|DESIGNS\b" /home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/trial_read.py /home/eric/bin/ai/theodolite2/atlas_data/census/trials/prereg/readlib.py || echo "none"
