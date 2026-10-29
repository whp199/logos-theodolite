#!/bin/bash
# Is a census run the same bit for bit across processes on one card: an unbroken run, the same run in two shards merged,
# and two canaries taken in separate processes.
cd /home/eric/bin/ai/theodolite2
D=/tmp/claude-1000/-home-eric-bin-ai-theodolite2/55e2a1fc-688e-4ae1-82e6-1ef32bab9bd9/scratchpad/det
mkdir -p $D/census $D/sharded
PY=".venv/bin/python -I"
ARGS="--layer 45 --design calib_rel --neurons random:16 --random-controls 8 --batch 64 --max-new 64 --unpacker compiled --out-dir $D/census --no-rank"
$PY scripts/census_run.py $ARGS --run DET > $D/unbroken.log 2>&1; echo "unbroken rc=$?"
$PY scripts/census_run.py ${ARGS/census --no-rank/sharded --no-rank} --run DET --shard 1/2 > $D/s1.log 2>&1; echo "shard1 rc=$?"
$PY scripts/census_run.py ${ARGS/census --no-rank/sharded --no-rank} --run DET --shard 2/2 > $D/s2.log 2>&1; echo "shard2 rc=$?"
ls $D/census $D/sharded
$PY scripts/census_run.py $ARGS --run CANA --canary-only > $D/ca.log 2>&1; echo "canary a rc=$?"
$PY scripts/census_run.py $ARGS --run CANB --canary-only > $D/cb.log 2>&1; echo "canary b rc=$?"
ls $D/census $D/sharded
echo DONE
