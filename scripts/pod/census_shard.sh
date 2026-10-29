#!/usr/bin/env bash
# One shard of one layer's census (scripts/census_run.py --shard i/n), detached on a pod (after neuron-explorer's
# sweep/run_layer.sh): it runs under setsid nohup, so it outlives the SSH session that started it, into a log, with
# sentinels for its start and end, under `timeout` at a multiple of its projected time, and, only when asked, stops the
# pod at the end.
#
#     scripts/pod/census_shard.sh --layer 45 --shard 1/4 --projected-hours 2.6 -- --neurons all --batch 128 --max-new 64 --unpacker lut
#     scripts/pod/census_shard.sh --layers 20,30,40 --design main_rel --projected-hours 1.5 --stop-pod -- --neurons random:384 --random-controls 64
#     scripts/pod/census_shard.sh --dry-run --root /tmp/pod --layer 3 --shard 0/2 --projected-hours 0.1 --wait -- --neurons 0-7 --max-new 8
#
# Arguments after -- go to census_run.py as they are (--neurons, --batch, --max-new, --random-controls, --seed,
# --unpacker...); give every shard of a run the same ones (the merge refuses shards taken otherwise) and the --batch the
# pilot measured on this card class (a run's bits depend on it). The job is named after its run (--run, default
# L<layer>-<design>) and shard: logs/<job>.log holds census_run.py's output between CENSUS_START and CENSUS_DONE,
# CENSUS_FAILED or CENSUS_TIMEOUT lines; logs/<job>.started is written at its start and logs/<job>.done, .failed or
# .timeout at its end (sync.sh --until-done waits for them; pod_tool.py listing names jobs not ended); logs/<job>.job.sh
# is the script it ran, with no secret in it. One GPU job holds the card at a time (logs/gpu.lock, which pilot.sh and
# setup.sh's self-test take too): a second is refused while one runs. The timeout sends census_run.py SIGINT at
# --projected-hours x --margin (1.25, the budget rule's), so it stops between steps and keeps every whole batch; the
# same command continues it. Run it again after a stop, a timeout or a lost pod to finish the shard: rows written are
# skipped, a torn batch written again whole.
#
# --stop-pod (off by default) runs `runpodctl stop pod "$RUNPOD_POD_ID"` at the end, whatever the job's end (an idle pod is
# the likeliest waste); --stop-cmd CMD runs CMD instead (e.g. "runpodctl remove pod \$RUNPOD_POD_ID" to terminate it: a
# network volume stays, but a pod's own volume disk goes with the pod, so pull first, sync.sh --final). The stop runs
# with the pod's own environment, where the provider puts its key; census_run.py runs with every variable naming a
# token, key or secret left out of its environment. --dry-run runs the job on the CPU stand-in (census_run.py
# --stand-in) in a temporary root (or --root), and only names the default stop command.

# shellcheck disable=SC2016  # the job script's lines are single-quoted on purpose: the job expands them, not this
set -euo pipefail
PROG=census_shard.sh
# shellcheck source=scripts/pod/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    help_text "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --layer L | --layers SPEC  the layer (counting from 0), or several in one process ("20,30,40-45"), each its own run
  --shard i/n                shard i of n of the run's planned chunks (leave it out for the whole run)
  --design NAME              main_rel (the pods' default, the design pilot.sh projects; census_run.py's own is break_rel),
                             break_rel, calib_rel or calib_ceil
  --run NAME                 the run's name (default L<layer>-<design>; with --layers, L{layer}-<design>)
  --projected-hours H        the pilot's projection for this job, in hours; the job is stopped at H x --margin
  --margin M                 default 1.25
  --no-timeout               run with no time limit (said in the log)
  --stop-pod                 stop the pod at the end (runpodctl stop pod "$RUNPOD_POD_ID")
  --stop-cmd CMD             the command that stops it (implies --stop-pod)
  --wait                     wait for the job to end, then print its log's tail (exit: the job's own)
  --root DIR                 the volume root (default $THEODOLITE_POD_ROOT, else /workspace)
  --dry-run, --stand-in      on the CPU stand-in, in a temporary root unless --root
  -h, --help                 this
EOF
}

ROOT="" DRY=0 LAYER="" LAYERS="" SHARD="" DESIGN=main_rel RUN="" HOURS="" MARGIN=1.25 NOTIME=0 STOP=0 STOPCMD="" WAIT=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --layer) [ $# -ge 2 ] || die "--layer needs a layer"; LAYER=$2; shift 2 ;;
        --layers) [ $# -ge 2 ] || die "--layers needs layers"; LAYERS=$2; shift 2 ;;
        --shard) [ $# -ge 2 ] || die "--shard needs i/n"; SHARD=$2; shift 2 ;;
        --design) [ $# -ge 2 ] || die "--design needs a name"; DESIGN=$2; shift 2 ;;
        --run) [ $# -ge 2 ] || die "--run needs a name"; RUN=$2; shift 2 ;;
        --projected-hours) [ $# -ge 2 ] || die "--projected-hours needs hours"; HOURS=$2; shift 2 ;;
        --margin) [ $# -ge 2 ] || die "--margin needs a number"; MARGIN=$2; shift 2 ;;
        --no-timeout) NOTIME=1; shift ;;
        --stop-pod) STOP=1; shift ;;
        --stop-cmd) [ $# -ge 2 ] || die "--stop-cmd needs a command"; STOPCMD=$2; STOP=1; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --root) [ $# -ge 2 ] || die "--root needs a folder"; ROOT=$2; shift 2 ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) die "unknown option $1 (census_run.py's own go after --; see --help)" ;;
    esac
done
[ -n "$LAYER" ] || [ -n "$LAYERS" ] || die "name the layer (--layer) or the layers (--layers)"
[ -z "$LAYER" ] || [ -z "$LAYERS" ] || die "give --layer or --layers, not both"
[ -z "$LAYER" ] || [[ "$LAYER" =~ ^[0-9]+$ ]] || die "--layer counts from 0, not $LAYER"
if [ -n "$SHARD" ]; then
    if ! [[ "$SHARD" =~ ^([0-9]+)/([0-9]+)$ ]] || [ "${BASH_REMATCH[1]}" -ge "${BASH_REMATCH[2]}" ]; then
        die "--shard is i/n, shard i of n counting from 0 (0/4 to 3/4), not $SHARD"
    fi
fi
awk -v m="$MARGIN" 'BEGIN { exit !(m + 0 >= 1) }' || die "--margin is 1 or more, not $MARGIN"
if [ "$NOTIME" = 0 ]; then
    [ -n "$HOURS" ] || die "give the pilot's projection (--projected-hours H; the job stops at H x $MARGIN), or --no-timeout"
    awk -v h="$HOURS" 'BEGIN { exit !(h + 0 > 0) }' || die "--projected-hours is a number of hours above 0, not $HOURS"
    SECS=$(awk -v h="$HOURS" -v m="$MARGIN" 'BEGIN { s = h * 3600 * m; printf "%d", (s == int(s)) ? s : int(s) + 1 }')
fi
for x in "${EXTRA[@]}"; do
    case "$x" in --shard|--layer|--layers|--run|--design|--stand-in|--out-dir) die "give $x to census_shard.sh itself, not after --" ;; esac
done

resolve_root
load_env
pod_python
if [ -n "$LAYERS" ]; then
    RUN=${RUN:-L{layer\}-$DESIGN}
    case "$RUN" in *"{layer}"*) ;; *) die "with --layers each layer is its own run: name them by {layer} in --run" ;; esac
    JOB=${RUN//\{layer\}/$(printf '%s' "$LAYERS" | tr ',' '_')}
    WHERE=(--layers "$LAYERS")
else
    RUN=${RUN:-L$LAYER-$DESIGN}
    JOB=$RUN
    WHERE=(--layer "$LAYER")
fi
if [ -n "$SHARD" ]; then
    JOB="$JOB.s${SHARD%/*}of${SHARD#*/}"
    WHERE+=(--shard "$SHARD")
fi
LOGS=$ROOT/logs
LOG=$LOGS/$JOB.log
JOBF=$LOGS/$JOB.job.sh
CMD=("$PY" -I -u scripts/census_run.py "${WHERE[@]}" --design "$DESIGN" --run "$RUN" "${STAND[@]}" "${EXTRA[@]}")

hold_card  # refused here while another job holds the card; let go at once, the job taking it for itself
exec 9>&-

if [ "$STOP" = 1 ]; then
    if [ -n "$STOPCMD" ]; then
        STOPLINE=$STOPCMD
    elif [ "$DRY" = 1 ]; then
        STOPLINE="echo 'dry run: the pod would be stopped here: runpodctl stop pod \"\$RUNPOD_POD_ID\"'"
    else
        # shellcheck disable=SC2016
        STOPLINE='if [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"; else echo "no RUNPOD_POD_ID here: stop the pod from the console"; false; fi'
    fi
fi

rm -f "$LOGS/$JOB.done" "$LOGS/$JOB.failed" "$LOGS/$JOB.timeout"
{
    printf '#!/usr/bin/env bash\n'
    printf '# %s: written by scripts/pod/census_shard.sh on %s and run detached (setsid nohup). No secret is in it.\n' "$JOB" "$(stamp)"
    printf 'set -uo pipefail\n'
    printf 'PROG=%q\n' "$JOB"
    printf '. %q\n' "$ROOT/env.sh"
    printf '. %q\n' "$POD_DIR/common.sh"
    [ "$DRY" = 1 ] && printf 'export CUDA_VISIBLE_DEVICES=\n'
    printf 'cd %q || exit 1\n' "$REPO"
    printf 'LOG=%q\n' "$LOG"
    printf 'exec 9>%q\n' "$LOGS/gpu.lock"
    printf 'if ! flock -n 9; then echo "CENSUS_REFUSED $(date -u +%%FT%%TZ): another GPU job holds the card" >>"$LOG"; '
    printf 'date -u +%%FT%%TZ >%q; exit 1; fi\n' "$LOGS/$JOB.failed"
    # its start, with its session and the signals it ignores (setsid makes it a session's leader, nohup ignores SIGHUP)
    printf 'printf "%%s pid %%s session %%s sigign %%s\\n" "$(date -u +%%FT%%TZ)" "$$" "$(cut -d" " -f6 /proc/$$/stat 2>/dev/null)" '
    printf '"$(sed -n "s/^SigIgn:[[:space:]]*//p" /proc/$$/status 2>/dev/null)" >%q\n' "$LOGS/$JOB.started"
    printf 'echo "CENSUS_START $(date -u +%%FT%%TZ)" >>"$LOG"\n'
    if [ "$NOTIME" = 1 ]; then
        printf 'echo "no time limit (--no-timeout)" >>"$LOG"\n'
        printf 'scrubbed %s >>"$LOG" 2>&1\n' "$(printf '%q ' "${CMD[@]}")"
    else
        printf 'echo "time limit %ss (%s h projected x %s)" >>"$LOG"\n' "$SECS" "$HOURS" "$MARGIN"
        printf 'scrubbed timeout --signal=INT --kill-after=300 %s %s >>"$LOG" 2>&1\n' "$SECS" "$(printf '%q ' "${CMD[@]}")"
    fi
    printf 'rc=$?\n'
    printf 'case $rc in 0) end=done; word=CENSUS_DONE ;; 124) end=timeout; word=CENSUS_TIMEOUT ;; *) end=failed; word=CENSUS_FAILED ;; esac\n'
    printf 'echo "$word rc=$rc $(date -u +%%FT%%TZ)" >>"$LOG"\n'
    printf 'printf "%%s rc=%%s\\n" "$(date -u +%%FT%%TZ)" "$rc" >%q."$end"\n' "$LOGS/$JOB"
    printf 'exec 9>&-\n'
    if [ "$STOP" = 1 ]; then
        printf 'echo "stopping the pod $(date -u +%%FT%%TZ)" >>"$LOG"\n'
        printf '{ %s ; } >>"$LOG" 2>&1 || echo "the stop failed: stop the pod from the console" >>"$LOG"\n' "$STOPLINE"
    fi
    printf 'exit $rc\n'
} >"$JOBF"
chmod +x "$JOBF"
printf 'CENSUS_JOB %s %s\n' "$(stamp)" "${CMD[*]}" >>"$LOG"

setsid nohup bash "$JOBF" </dev/null >/dev/null 2>&1 &
PID=$!
say "$JOB detached (pid $PID): its log logs/$JOB.log, its end logs/$JOB.done, .failed or .timeout, under $ROOT"
if [ "$NOTIME" = 1 ]; then
    say "no time limit"
else
    say "stopped at ${SECS}s ($HOURS h projected x $MARGIN), between steps; the same command continues it"
fi
if [ "$STOP" = 1 ]; then
    say "the pod is stopped at the end: $STOPLINE"
fi
if [ "$WAIT" = 1 ]; then
    while :; do
        for end in done failed timeout; do
            if [ -e "$LOGS/$JOB.$end" ]; then
                tail -n 5 "$LOG" | sed 's/^/  /'
                say "$JOB: $end"
                if [ "$end" = done ]; then exit 0; fi
                exit 1
            fi
        done
        sleep 1
    done
fi
exit 0
