#!/usr/bin/env bash
# break_rel trial on the home RTX 5090, ~2-2.3 GPU-hours. Run from a clean checkout of the break_rel commit (the patch
# committed on a branch, check_census.py passing); R is that checkout. Stops on the first failure, and run again it
# continues where it stopped (a finished run is kept as it is). Nothing here commits, pushes or merges, and nothing here
# reads a verdict: the reader is pre-registered (copied and hashed into $D/prereg before step 0) and run in the morning,
# after the owner has approved or amended its GATES, by $D/morning.sh.
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2}
S=/tmp/claude-1000/-home-eric-bin-ai-theodolite2/55e2a1fc-688e-4ae1-82e6-1ef32bab9bd9/scratchpad/design/synth
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data   # the stats index's p50, as the calibration read it
D=$THEODOLITE_ATLAS_DIR/census/trials                                    # not census/runs: no page reads a trial as a layer's census
P=$D/prereg
# the one venv (a worktree may have none): census_run.py and verify_atlas_writer.py put their own checkout ($R) first on
# sys.path, so $R's code is the code run
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
grep -q "break_rel" "$R/backend/census.py" || { echo "break_rel is not in $R/backend/census.py: apply break_rel.v2.patch and commit it first"; exit 2; }
git -C "$R" rev-parse --verify -q HEAD > /dev/null || { echo "$R is not a git checkout: run from a checkout of the break_rel commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || {
    echo "$R has uncommitted changes: run from a clean checkout of the break_rel commit"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
mkdir -p "$D"

# the pre-registration: the reader, its gates, step 0's rule and the neuron lists, copied beside the runs and hashed
# before anything is measured (kept as first written when the script is run again)
FILES="tonight.sh step0_pick.py trial_read.py readlib.py calib_expect.json trial_L45_neurons.txt trial_L45_fresh.txt trial_L45_reasked.txt probe_L57_neurons.txt"
if [ ! -f "$P/SHA256SUMS" ]; then
    mkdir -p "$P"
    for f in $FILES; do cp "$S/$f" "$P/"; done
    git -C "$R" rev-parse HEAD > "$P/repo_commit"
    (cd "$P" && sha256sum $FILES repo_commit > SHA256SUMS && date -u +%FT%TZ > prereg_at)
    echo "pre-registered in $P (SHA256SUMS)"
fi
(cd "$P" && grep -E ' (step0_pick\.py|trial_L45_neurons\.txt|probe_L57_neurons\.txt)$' SHA256SUMS | sha256sum -c --quiet) || {
    echo "$P's step 0 rule or neuron lists differ from their SHA256SUMS"; exit 2; }

# step 0: the writer's floor on the six questions and the three spares, batches 64 and 96 (looks at no neuron; ~10 min)
[ -f "$D/floor_L45_break_rel.json" ] || $PY "$R/scripts/verify_atlas_writer.py" --layer 45 --max-new 48 --batches 64,96 \
    --unpacker compiled --questions who,orange,free,bicycle,walk,chair,kettle,sleep,bridge --out "$D/floor_L45_break_rel.json"
PICK=$($PY "$P/step0_pick.py" "$D/floor_L45_break_rel.json") || { echo "step 0 stopped: $PICK"; exit 3; }
echo "$PICK" > "$D/step0_pick.json"
QS=$(echo "$PICK" | $PY -c 'import json,sys; print(",".join(json.load(sys.stdin)["questions"]))')
B=$(echo "$PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["batch"])')
DQ=$($PY -c 'import sys; sys.path.insert(0, sys.argv[1]); from backend import census as cz; print(",".join(cz.DESIGN_QUESTIONS["break_rel"]))' "$R")
if [ "$QS" != "$DQ" ]; then
    echo "step 0 picked $QS; $R's break_rel asks $DQ: set census.DESIGN_QUESTIONS/DESIGN_HALVES['break_rel'] (and check_census's q6)"
    echo "to the pick, rerun check_census, commit, and run this again (the floor is kept)"; exit 4
fi
echo "questions $QS, batch $B"

done_run() {  # every answer of the run's job list is on disk
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$1.jsonl" 2>/dev/null
}
run() {  # layer, neuron file, controls, run name
    if done_run "$4"; then echo "$4: every answer already written, kept as it is"; return 0; fi
    [ -e "$D/$4.jsonl" ] || date -u +%s > "$D/$4.t0"   # the first start (a continued run keeps it)
    $PY "$R/scripts/census_run.py" --layer "$1" --design break_rel --neurons "$(cat "$2")" --random-controls "$3" --seed 3 \
        --batch "$B" --max-new 48 --unpacker compiled --out-dir "$D" --run "$4" 2>&1 | tee -i -a "$D/$4.log"
    done_run "$4" || { echo "$4 stopped before its last answer: run this again to continue it row for row"; exit 6; }
    date -u +%s > "$D/$4.t1"
}
run 45 "$P/trial_L45_neurons.txt" 1024 L45-break_rel-trial      # 1,086 neurons + 1,024 controls: ~1.0 h at 96, 1.3 at 64
run 57 "$P/probe_L57_neurons.txt" 512 L57-break_rel-probe       # 507 neurons + 512 controls: ~0.5 h at 96, 0.6 at 64

# the morning's reading, CPU only, from the pre-registered copies (the first line says whether they are as hashed)
cat > "$D/morning.sh" <<EOF
#!/usr/bin/env bash
# Read the break_rel trial. First approve or amend $P/trial_read.py's GATES; write any amendment, with its time and
# reason, into $P before this runs (sha256sum -c below then names the file that changed).
(cd "$P" && sha256sum -c SHA256SUMS) || echo "the pre-registered files above differ from SHA256SUMS: record why in $P"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
$PY "$P/trial_read.py" "$D/L45-break_rel-trial.jsonl" --gates L45 --fresh "$P/trial_L45_fresh.txt" \\
    --retest-ref "$THEODOLITE_ATLAS_DIR/census/runs/L45-calib_rel.jsonl" --reasked "$P/trial_L45_reasked.txt" \\
    --out "$D/L45_verdict.json" > "$D/L45_verdict.txt" 2>&1 || true
$PY "$P/trial_read.py" "$D/L57-break_rel-probe.jsonl" --gates L57 --out "$D/L57_verdict.json" > "$D/L57_verdict.txt" 2>&1 || true
cat "$D/L45_verdict.txt" "$D/L57_verdict.txt"
EOF
chmod +x "$D/morning.sh"
echo "both runs written in $D; not read tonight. In the morning: approve or amend $P/trial_read.py's GATES, then $D/morning.sh"
