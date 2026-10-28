#!/usr/bin/env bash
# Pull a pod's results to this machine, every N minutes, and once more at the end, checked (after neuron-explorer's
# lesson, research_log.md:280: its pods' captures were deleted before they were pulled, and had to be rebuilt). Run it
# on the owner's machine, against the pod's SSH:
#
#     scripts/pod/sync.sh --from root@203.0.113.7 --port 22022 --to ~/pods/h200-a --every 10        # until stopped
#     scripts/pod/sync.sh --from root@203.0.113.7 --port 22022 --to ~/pods/h200-a --until-done      # then the final pull
#     scripts/pod/sync.sh --from root@203.0.113.7 --port 22022 --to ~/pods/h200-a --final           # the final pull alone
#     scripts/pod/sync.sh --dry-run --root /tmp/pod --to /tmp/home --final                          # a local "pod"
#
# What it pulls, from under the pod's volume root, into the same places under --to: the census runs, their metas, the
# shards' canaries and rankings (atlas_data/census/runs/), the jobs' logs and sentinels (logs/), the pilots (pilot/)
# and the steer lens's layers (atlas_data/index/steer_lens.d/). Each round is two rsync passes over SSH: the rows
# (*.jsonl, *.log), which only grow, with --append-verify (a part-written last line is harmless: the census's readers
# leave it out and census_run.py cuts it), then everything else whole (the metas are rewritten, not appended). The
# final pull copies everything plainly (a file the pod has cut short, a torn batch written again, is then replaced
# too), lists the pod's files (rsync --list-only into <to>/remote.list), and checks the copy against that listing and
# against what each census meta names (pod_tool.py listing: every file at its size, a run's rows, a shard's canary, a
# joined run's shards; jobs not yet ended named): it exits 1, in plain words, when anything is missing. Delete a pod or
# its volume only after a final pull that says so. --until-done pulls every --every minutes until every job on the pod
# has ended (logs/<job>.started each with its .done, .failed or .timeout), then pulls the last time.
#
# No secret is read: SSH uses the key named (--key), or your agent. --dry-run takes the "pod" from a local folder
# (--root, a dry run's volume root: the stand-in's census folder under it), into --to (default: a temporary folder).
set -euo pipefail
PROG=sync.sh
# shellcheck source=scripts/pod/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    help_text "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --from USER@HOST     the pod's SSH login (RunPod: "SSH over exposed TCP", root@<ip>)
  --port P             its SSH port (default 22)
  --key FILE           the SSH key (default: your agent's, or ~/.ssh's)
  --remote-root DIR    the pod's volume root (default /workspace)
  --to DIR             where the copy goes on this machine
  --every MIN          minutes between pulls (default 10; may be a fraction)
  --rounds N           stop after N pulls (default: until stopped)
  --once               one pull, then stop
  --until-done         pull until every job on the pod has ended, then the final pull
  --final              the final pull and its check alone
  --dry-run            from a local folder (--root) instead of a pod
  --root DIR           with --dry-run: that folder
  -h, --help           this
EOF
}

FROM="" PORT=22 KEY="" RROOT=/workspace TO="" EVERY=10 ROUNDS=0 FINAL=0 UNTIL=0 DRY=0 ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from) [ $# -ge 2 ] || die "--from needs user@host"; FROM=$2; shift 2 ;;
        --port) [ $# -ge 2 ] || die "--port needs a port"; PORT=$2; shift 2 ;;
        --key) [ $# -ge 2 ] || die "--key needs a file"; KEY=$2; shift 2 ;;
        --remote-root) [ $# -ge 2 ] || die "--remote-root needs a folder"; RROOT=$2; shift 2 ;;
        --to) [ $# -ge 2 ] || die "--to needs a folder"; TO=$2; shift 2 ;;
        --every) [ $# -ge 2 ] || die "--every needs minutes"; EVERY=$2; shift 2 ;;
        --rounds) [ $# -ge 2 ] || die "--rounds needs a count"; ROUNDS=$2; shift 2 ;;
        --once) ROUNDS=1; shift ;;
        --until-done) UNTIL=1; shift ;;
        --final) FINAL=1; shift ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        --root) [ $# -ge 2 ] || die "--root needs a folder"; ROOT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
awk -v m="$EVERY" 'BEGIN { exit !(m + 0 > 0) }' || die "--every is minutes above 0, not $EVERY"
[[ "$ROUNDS" =~ ^[0-9]+$ ]] || die "--rounds is a count, not $ROUNDS"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "--port is a number, not $PORT"
command -v rsync >/dev/null || die "rsync is not installed here (and the pod needs it too: setup.sh installs it)"

if [ "$DRY" = 1 ]; then
    [ -n "$ROOT" ] || die "a dry run pulls from a local folder: give it as --root (a dry run's volume root)"
    [ -d "$ROOT" ] || die "there is no folder $ROOT"
    SRC_ROOT=$(cd "$ROOT" && pwd)
    RUNS=atlas_data/stand-in/census/runs
    STEER=atlas_data/stand-in/index/steer_lens.d
    RSH=()
    if [ -z "$TO" ]; then
        TO=$(mktemp -d "${TMPDIR:-/tmp}/theodolite-pulled.XXXXXX")
        say "dry run: the copy goes to $TO"
    fi
else
    [ -n "$FROM" ] || die "name the pod's SSH login (--from root@<ip>), or give --dry-run"
    [ -n "$TO" ] || die "name where the copy goes (--to)"
    SRC_ROOT="$FROM:$RROOT"
    RUNS=atlas_data/census/runs
    STEER=atlas_data/index/steer_lens.d
    ssh_cmd="ssh -p $PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new${KEY:+ -i $(printf '%q' "$KEY")}"
    RSH=(-e "$ssh_cmd")
fi
mkdir -p "$TO"
TO=$(cd "$TO" && pwd)
SYNCLOG=$TO/sync.log
# the sources, each under the root's "/./" so the copy keeps the pod's layout below it
SRCS=()
for d in "$RUNS" logs pilot "$STEER"; do SRCS+=("$SRC_ROOT/./$d/"); done

note() { say "$*" >&2; printf '%s %s\n' "$(stamp)" "$*" >>"$SYNCLOG"; }

pull() {  # rsync, a vanished or absent file (rc 23/24, a job writing meanwhile) said but not fatal
    local rc=0
    rsync "${RSH[@]}" "$@" || rc=$?
    case $rc in
        0) return 0 ;;
        23|24) note "rsync: some files were absent or vanished while it read (rc $rc): the next pull takes them"; return 0 ;;
        *) note "rsync failed (rc $rc)"; return "$rc" ;;
    esac
}

round() {
    pull -a --relative --append-verify --include='*/' --include='*.jsonl' --include='*.log' --exclude='*' "${SRCS[@]}" "$TO/" \
        || return 1
    pull -a --relative --exclude='*.jsonl' --exclude='*.log' "${SRCS[@]}" "$TO/" || return 1
    local files
    files=$(find "$TO" -type f ! -name sync.log ! -name remote.list ! -name listing.json | wc -l)
    note "pulled: $files files under $TO"
}

running_jobs() {  # jobs started on the pod and not ended, by the copy of its logs
    local s n=0
    for s in "$TO"/logs/*.started; do
        [ -e "$s" ] || continue
        local j=${s%.started}
        [ -e "$j.done" ] || [ -e "$j.failed" ] || [ -e "$j.timeout" ] || n=$((n + 1))
    done
    printf '%s' "$n"
}

final() {
    note "the final pull: everything, whole"
    pull -a --relative "${SRCS[@]}" "$TO/" || return 1
    pull -r --relative --list-only --no-human-readable "${SRCS[@]}" >"$TO/remote.list" || return 1
    # the listing needs no package beyond Python's own
    if [ -n "${POD_PYTHON:-}" ]; then PY=$POD_PYTHON; elif [ -x "$REPO/.venv/bin/python" ]; then PY=$REPO/.venv/bin/python; else PY=python3; fi
    if "$PY" -I "$POD_DIR/pod_tool.py" listing --local "$TO" --remote-list "$TO/remote.list" --json "$TO/listing.json" \
        | tee -a "$SYNCLOG"; then
        note "the copy holds everything the pod has, and every file its metas name: $TO"
        return 0
    fi
    note "the copy is NOT whole (above, and $TO/listing.json): pull again before the pod or its volume goes"
    return 1
}

if [ "$FINAL" = 1 ]; then
    final
    exit $?
fi
k=0
while :; do
    k=$((k + 1))
    round || note "round $k failed; trying again in $EVERY min"
    if [ "$UNTIL" = 1 ] && [ -n "$(ls "$TO"/logs/*.started 2>/dev/null)" ] && [ "$(running_jobs)" = 0 ]; then
        note "every job on the pod has ended"
        final
        exit $?
    fi
    if [ "$ROUNDS" -gt 0 ] && [ "$k" -ge "$ROUNDS" ]; then
        note "$k pull(s) done; the final pull is sync.sh --final"
        exit 0
    fi
    sleep "$(awk -v m="$EVERY" 'BEGIN { printf "%.3f", m * 60 }')"
done
