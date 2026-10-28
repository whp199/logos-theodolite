#!/usr/bin/env bash
# The one-hour pilot on a rented card, before any budget is asked for (the plan's budget rule: a pilot on the same card
# class projects the job, x 1.25, against the approval). Run it on the pod after setup.sh, with nothing else on the card:
#
#     scripts/pod/pilot.sh                                           # layer 45, the defaults below
#     scripts/pod/pilot.sh --home-canary ~/L45-main_rel.canary.jsonl --approved-hours 12 --jobs whole_layer:2,steer_lens:1
#     scripts/pod/pilot.sh --dry-run                                 # the whole flow on the CPU stand-in, a temporary root
#
# Into pilot/<UTC time>/ under the volume root, one log (pilot.log, each step's own output also in out/<step>.txt) and
# one summary (pilot.json, pod_tool.py summary):
#  1. the self-test: the card, the versions, the unpacker's test, the runtime class (selftest.json);
#  2. the decode step and peak memory of census-shaped batches for each unpacker (--unpackers) at each batch
#     (--batches), every row run to --max-new, a batch that does not fit recorded and the larger skipped (steps.json);
#     the fastest that fit is the pilot's own (unless --unpacker and --batch are given): the batch fixes a run's
#     determinism class, so it is measured here, not assumed in a job's command (critic D5);
#  3. the writer's floor (scripts/verify_atlas_writer.py: with no push, how far answers written in census batches of 32
#     and 64, as /steer's as-it-is row and by interp-engine's loop agree with the same answer written alone; about 18
#     minutes with the reference unpacker on a 5090, less with a faster one) (atlas_writer_floor.json);
#  4. a calib_rel run (--calib-neurons and --calib-controls, at the pilot's batch, unpacker and --max-new) timed (its wall
#     time, the model's load with it), and its dose window (census_tools.py curves) (census/, calib.json, curves.json);
#  5. one steer-lens layer (atlas_build.py steer --layers --steer-layer) timed as the build logs it, into the pilot's own
#     index folder (the stats index linked in when built, for its push unit) (index/, steer.json);
#  6. a canary batch (census_run.py --canary-only): with --home-canary, the batch the home card wrote, taken with the
#     same settings (its .canary.json beside it) and compared field by field (census_tools.py canary); without, the
#     whole-layer job's own plan batch 0, to compare later (canary/, canary.json);
#  7. the zstd ratio of the calib_rel run's file (zstd.json);
#  8. the summary: each job's projected GPU-hours on the fastest batch that fit, x 1.25, and with --approved-hours and
#     --jobs, "proceed" or "stop".
# A step that fails is said and the pilot goes on; it exits 1 if any did. Every Python runs with every variable naming a
# token, key or secret left out of its environment. Copy the floor into experiments/results/ at home (one file per card
# class: atlas_writer_floor_<card>.json) for the census card to quote it.
#
# --dry-run runs every step on the CPU stand-in in a temporary root (or --root): layer 3, batches 2,4,8, --max-new 8,
# 4 neurons and 2 controls, unless given; its numbers are the stand-in's, and say nothing of a card.
set -euo pipefail
PROG=pilot.sh
# shellcheck source=scripts/pod/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    help_text "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --root DIR            the volume root (default $THEODOLITE_POD_ROOT, else /workspace)
  --layer L             the layer every step pushes, counting from 0 (default 45; 3 on a dry run)
  --batches LIST        batch sizes timed (default 32,96,128,256,512; 2,4,8 on a dry run)
  --unpackers LIST      unpackers timed (default reference,lut,compiled; resident needs ~60 GB more)
  --max-new N           word-pieces per answer, as the jobs will take them (default 64; 8 on a dry run)
  --unpacker MODE       the unpacker for steps 3-6 (default: the fastest that fit in step 2)
  --batch B             the batch for steps 4 and 6 (default: the fastest that fit in step 2)
  --calib-neurons SPEC  default random:8 (random:4 on a dry run)
  --calib-controls C    default 8 (2 on a dry run)
  --canary-controls C   the whole-layer job's controls, for its canary when no home one is given (default 256; 4 dry)
  --steer-layer L       the steer-lens layer timed (default --layer)
  --home-canary FILE    a canary the home card wrote (census_run.py --canary-only), its .canary.json beside it
  --approved-hours H    the GPU-hours approved, to write "proceed" or "stop"
  --jobs LIST           what the approval is for, e.g. whole_layer:2,depth_curve:1,steer_lens:1
  --skip LIST           steps left out: steps,floor,calib,steer,canary,zstd
  --dry-run, --stand-in on the CPU stand-in, in a temporary root unless --root
  -h, --help            this
EOF
}

ROOT="" DRY=0 LAYER="" BATCHES="" UNPS=reference,lut,compiled MAXNEW="" UNP="" BATCH="" CN="" CC="" KC="" SL=""
HOME_CANARY="" APPROVED="" JOBS="" SKIP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root) [ $# -ge 2 ] || die "--root needs a folder"; ROOT=$2; shift 2 ;;
        --layer) [ $# -ge 2 ] || die "--layer needs a layer"; LAYER=$2; shift 2 ;;
        --batches) [ $# -ge 2 ] || die "--batches needs sizes"; BATCHES=$2; shift 2 ;;
        --unpackers) [ $# -ge 2 ] || die "--unpackers needs names"; UNPS=$2; shift 2 ;;
        --max-new) [ $# -ge 2 ] || die "--max-new needs a count"; MAXNEW=$2; shift 2 ;;
        --unpacker) [ $# -ge 2 ] || die "--unpacker needs a mode"; UNP=$2; shift 2 ;;
        --batch) [ $# -ge 2 ] || die "--batch needs a size"; BATCH=$2; shift 2 ;;
        --calib-neurons) [ $# -ge 2 ] || die "--calib-neurons needs a spec"; CN=$2; shift 2 ;;
        --calib-controls) [ $# -ge 2 ] || die "--calib-controls needs a count"; CC=$2; shift 2 ;;
        --canary-controls) [ $# -ge 2 ] || die "--canary-controls needs a count"; KC=$2; shift 2 ;;
        --steer-layer) [ $# -ge 2 ] || die "--steer-layer needs a layer"; SL=$2; shift 2 ;;
        --home-canary) [ $# -ge 2 ] || die "--home-canary needs a file"; HOME_CANARY=$2; shift 2 ;;
        --approved-hours) [ $# -ge 2 ] || die "--approved-hours needs hours"; APPROVED=$2; shift 2 ;;
        --jobs) [ $# -ge 2 ] || die "--jobs needs a list"; JOBS=$2; shift 2 ;;
        --skip) [ $# -ge 2 ] || die "--skip needs a list"; SKIP=$2; shift 2 ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
if [ "$DRY" = 1 ]; then
    LAYER=${LAYER:-3} BATCHES=${BATCHES:-2,4,8} MAXNEW=${MAXNEW:-8} CN=${CN:-random:4} CC=${CC:-2} KC=${KC:-4}
else
    LAYER=${LAYER:-45} BATCHES=${BATCHES:-32,96,128,256,512} MAXNEW=${MAXNEW:-64} CN=${CN:-random:8} CC=${CC:-8} KC=${KC:-256}
fi
SL=${SL:-$LAYER}
if [ -n "$HOME_CANARY" ]; then
    [ -f "$HOME_CANARY" ] || die "there is no canary $HOME_CANARY"
    HOME_CANARY=$(cd "$(dirname "$HOME_CANARY")" && pwd)/$(basename "$HOME_CANARY")
fi
for x in "$LAYER" "$SL" "$MAXNEW" "$CC" "$KC"; do [[ "$x" =~ ^[0-9]+$ ]] || die "a layer or count is a whole number, not $x"; done

resolve_root
load_env
pod_python
P=$ROOT/pilot/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$P"
exec > >(tee -a "$P/pilot.log") 2> >(tee -a "$P/pilot.log" >&2)
cd "$REPO"
say "$(stamp): the pilot, into $P (layer $LAYER, batches $BATCHES, unpackers $UNPS, --max-new $MAXNEW)"
FAILED=()
skipped() { case ",$SKIP," in *",$1,"*) say "== $1: skipped (--skip)"; return 0 ;; esac; return 1; }
# a step: said, timed (with its process's load), its output kept in out/<step>.txt too, and gone on from whatever its end
mkdir -p "$P/out"
step() {
    local name=$1 t0 rc
    shift
    say "== $name"
    t0=$(now_s)
    set +e
    scrubbed "$@" 2>&1 | tee "$P/out/$name.txt"
    rc=${PIPESTATUS[0]}
    set -e
    LAST_S=$(elapsed "$t0" "$(now_s)")
    if [ "$rc" = 0 ]; then say "== $name: done in ${LAST_S}s"; else say "== $name: FAILED (rc $rc) after ${LAST_S}s"; FAILED+=("$name"); fi
    return 0
}

# 1. the self-test
step selftest "$PY" -I "$POD_DIR/pod_tool.py" selftest "${STAND[@]}" --unpacker "${UNP:-lut}" --out "$P/selftest.json"

# 2. decode steps and peak memory, per unpacker and batch
if ! skipped steps; then
    step steps "$PY" -I "$POD_DIR/pod_tool.py" steps "${STAND[@]}" --layer "$LAYER" --batches "$BATCHES" --unpackers "$UNPS" \
        --max-new "$MAXNEW" --out "$P/steps.json"
fi
if [ -z "$UNP" ] || [ -z "$BATCH" ]; then
    read -r best_u best_b < <("$PY" -I -c '
import json, sys
try:
    b = json.load(open(sys.argv[1])).get("best") or {}
except Exception:
    b = {}
print(b.get("unpacker") or "reference", b.get("batch") or 0)' "$P/steps.json")
    UNP=${UNP:-$best_u}
    if [ -z "$BATCH" ]; then
        BATCH=$best_b
        [ "$BATCH" != 0 ] || { BATCH=32; say "no batch measured to fit: steps 4 and 6 take --batch 32"; }
    fi
fi
say "the pilot's unpacker $UNP and batch $BATCH (steps 4 and 6)"

# 3. the writer's floor
if ! skipped floor; then
    step floor "$PY" -I scripts/verify_atlas_writer.py "${STAND[@]}" --layer "$LAYER" --max-new "$MAXNEW" --unpacker "$UNP" \
        --out "$P/atlas_writer_floor.json"
fi

# 4. a calib_rel run, timed, and its dose window
CALIB=$P/census/pilot-L$LAYER-calib_rel.jsonl
if ! skipped calib; then
    step calib "$PY" -I scripts/census_run.py "${STAND[@]}" --layer "$LAYER" --design calib_rel --neurons "$CN" \
        --random-controls "$CC" --batch "$BATCH" --max-new "$MAXNEW" --unpacker "$UNP" --run "pilot-L$LAYER-calib_rel" \
        --out-dir "$P/census"
    answers=0
    if [ -f "$CALIB" ]; then answers=$(wc -l <"$CALIB"); fi
    printf '{"run": "%s", "wall_s": %s, "answers": %s, "batch": %s, "unpacker": "%s", "max_new": %s, "neurons": "%s", "controls": %s}\n' \
        "$(basename "$CALIB")" "$LAST_S" "$answers" "$BATCH" "$UNP" "$MAXNEW" "$CN" "$CC" >"$P/calib.json"
    if [ -f "$CALIB" ]; then
        step curves "$PY" -I scripts/census_tools.py "${STAND[@]}" curves "$CALIB" --json "$P/curves.json"
    fi
fi

# 5. one steer-lens layer, timed, in the pilot's own index folder
if ! skipped steer; then
    mkdir -p "$P/index"
    if [ "$DRY" = 1 ]; then ACT=$THEODOLITE_ATLAS_DIR/stand-in/index/act_scale.npz; else ACT=$THEODOLITE_ATLAS_DIR/index/act_scale.npz; fi
    if [ -f "$ACT" ]; then ln -sf "$ACT" "$P/index/act_scale.npz"; fi
    step steer "$PY" -I scripts/atlas_build.py steer "${STAND[@]}" --layers "$SL" --out "$P/index" --unpacker "$UNP"
    # the layer's own time, as the build logs it (the step's wall time adds the model's load)
    layer_s=$(sed -n 's/^steer lens: layer [0-9]* done (\([0-9.]*\) s)$/\1/p' "$P/out/steer.txt" | tail -1)
    unit="the study's 36 texts"
    if [ -f "$ACT" ]; then unit="the stats index"; fi
    printf '{"layer": %s, "seconds": %s, "wall_s": %s, "unpacker": "%s", "push_unit": "%s"}\n' "$SL" "${layer_s:-null}" "$LAST_S" \
        "$UNP" "$unit" >"$P/steer.json"
fi

# 6. a canary: the home card's batch written here, or the whole-layer job's own
if ! skipped canary; then
    if [ -n "$HOME_CANARY" ]; then
        mapfile -t CARGS < <("$PY" -I "$POD_DIR/pod_tool.py" canary-args "$HOME_CANARY")
        [ ${#CARGS[@]} -gt 0 ] || FAILED+=(canary-args)
    else
        CARGS=(--layer "$LAYER" --design main_rel --neurons all --random-controls "$KC" --batch "$BATCH" --max-new "$MAXNEW"
               --run "pilot-L$LAYER-main_rel")
    fi
    if [ ${#CARGS[@]} -gt 0 ]; then
        step canary "$PY" -I scripts/census_run.py "${STAND[@]}" "${CARGS[@]}" --unpacker "$UNP" --canary-only --out-dir "$P/canary"
        OURS=""
        for f in "$P"/canary/*.canary.jsonl; do
            if [ -e "$f" ]; then OURS=$f; break; fi
        done
        if [ -n "$HOME_CANARY" ] && [ -n "$OURS" ]; then
            step canary-compare "$PY" -I scripts/census_tools.py canary "$OURS" "$HOME_CANARY" --json "$P/canary.json"
        elif [ -n "$OURS" ]; then
            printf '{"same": null, "ours": "%s", "says": "no home canary given: %s is this card'"'"'s plan batch 0, to set against another card'"'"'s (census_tools.py canary)"}\n' \
                "$(basename "$OURS")" "$(basename "$OURS")" >"$P/canary.json"
        fi
    fi
fi

# 7. the zstd ratio of the run file
if ! skipped zstd; then
    if [ ! -f "$CALIB" ]; then
        printf '{"ratio": null, "says": "no calib_rel run file to compress"}\n' >"$P/zstd.json"
    elif command -v zstd >/dev/null; then
        raw=$(stat -c %s "$CALIB")
        z=$(zstd -q -c -3 "$CALIB" | wc -c)
        printf '{"file": "%s", "raw_bytes": %s, "zstd_bytes": %s, "level": 3, "ratio": %s, "says": "%s"}\n' "$(basename "$CALIB")" \
            "$raw" "$z" "$(awk -v r="$raw" -v z="$z" 'BEGIN { printf "%.2f", r / z }')" \
            "$(awk -v r="$raw" -v z="$z" 'BEGIN { printf "%.2fx at level 3 (%d bytes to %d)", r / z, r, z }')" >"$P/zstd.json"
        say "zstd: $(sed -n 's/.*"says": "\(.*\)".*/\1/p' "$P/zstd.json")"
    else
        printf '{"ratio": null, "says": "zstd is not installed here: the ratio is not measured"}\n' >"$P/zstd.json"
        say "zstd is not installed here: the ratio is not measured"
    fi
fi

# 8. the summary
SARGS=()
[ -z "$APPROVED" ] || SARGS=(--approved-hours "$APPROVED" --jobs "$JOBS")
step summary "$PY" -I "$POD_DIR/pod_tool.py" summary "$P" "${SARGS[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    say "$(stamp): the pilot ended with ${#FAILED[@]} step(s) failed: ${FAILED[*]} (pilot.log)"
    exit 1
fi
say "$(stamp): the pilot ended, every step done: $P/pilot.json"
