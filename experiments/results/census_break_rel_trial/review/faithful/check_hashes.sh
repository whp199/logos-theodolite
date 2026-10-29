#!/usr/bin/env bash
# Read-only: the pre-registration's hashes, the runs' sha256 against the branch copy's runs.txt, the branch copy
# against the trial folder byte for byte, the scratchpad originals against the prereg copies, and the timeline.
T=/home/eric/bin/ai/theodolite2/atlas_data/census/trials
B=/home/eric/bin/ai/theodolite2/.claude/worktrees/census-break-rel/experiments/results/census_break_rel_trial
S=/tmp/claude-1000/-home-eric-bin-ai-theodolite2/55e2a1fc-688e-4ae1-82e6-1ef32bab9bd9/scratchpad/design/synth
echo "--- sha256sum -c SHA256SUMS"; (cd $T/prereg && sha256sum -c SHA256SUMS); echo "exit $?"
echo "--- runs"; (cd $T && sha256sum L45-break_rel-trial.jsonl L57-break_rel-probe.jsonl); cat $B/runs.txt
echo "--- branch copy vs trial folder"
for f in floor_L45_break_rel.json L45-break_rel-trial.meta.json L45_verdict.json L45_verdict.txt L57-break_rel-probe.meta.json L57_verdict.json L57_verdict.txt morning.sh step0_pick.json; do cmp -s $T/$f $B/$f && echo "same $f" || echo "DIFFERS $f"; done
for f in $(cd $T/prereg && ls | grep -v __pycache__); do cmp -s $T/prereg/$f $B/prereg/$f && echo "same prereg/$f" || echo "DIFFERS prereg/$f"; done
echo "--- scratchpad originals vs prereg copies"
for f in tonight.sh step0_pick.py trial_read.py readlib.py calib_expect.json trial_L45_neurons.txt trial_L45_fresh.txt trial_L45_reasked.txt probe_L57_neurons.txt; do cmp -s $T/prereg/$f $S/$f && echo "same $f" || echo "DIFFERS $f"; done
echo "--- not in SHA256SUMS:"; ls -la --time-style=+%FT%T%z $T/prereg/__pycache__
echo "--- timeline (UTC)"; echo "prereg_at $(cat $T/prereg/prereg_at)"
for r in L45-break_rel-trial L57-break_rel-probe; do echo "$r t0 $(date -u -d @$(cat $T/$r.t0) +%FT%TZ) t1 $(date -u -d @$(cat $T/$r.t1) +%FT%TZ)"; done
for f in floor_L45_break_rel.json L45-break_rel-trial.units.jsonl L57-break_rel-probe.units.jsonl morning.sh L45_verdict.json L57_verdict.json; do echo "$f mtime $(date -u -r $T/$f +%FT%TZ)"; done
