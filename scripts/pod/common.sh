# shellcheck shell=bash disable=SC2034  # STAND and PY are set here for the scripts that source this file
# What the pod scripts share (each sources this file): the checkout, the volume root and its env file, the Python, the
# stand-in's dry run, a child's environment kept clear of secrets, and plain-words exits. Nothing here reads, prints or
# writes a secret's value.

POD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$POD_DIR/../.." && pwd)
DEFAULT_ROOT=/workspace
: "${PROG:=pod}"

say() { printf '%s: %s\n' "$PROG" "$*"; }
die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
now_s() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }

# the comment block after a script's shebang, as its --help prints it
help_text() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$1"; }

# Run a command with every variable that names a token, key, secret, password or credential left out of its
# environment, but those given as --keep NAME first: a census, a pilot or a server started from these scripts inherits
# none (the scientist's CLIs get the server's whole environment less the endpoint keys, backend/scientist_chat.py
# child_env). The values never pass through argv or a file.
scrubbed() {
    local keep=" " a=(env) v
    while [ "${1:-}" = --keep ]; do keep="$keep$2 "; shift 2; done
    for v in $(compgen -e); do
        case "$keep" in *" $v "*) continue ;; esac
        case "$v" in
            *TOKEN*|*SECRET*|*API_KEY*|*ACCESS_KEY*|*CREDENTIALS*|*PASSWORD*|*PASSWD*) a+=(-u "$v") ;;
        esac
    done
    "${a[@]}" "$@"
}

# the volume root: --root, else (a dry run) a fresh temporary one, else $THEODOLITE_POD_ROOT, else /workspace
resolve_root() {
    if [ -n "${ROOT:-}" ]; then
        mkdir -p "$ROOT"
        ROOT=$(cd "$ROOT" && pwd)
    elif [ "${DRY:-0}" = 1 ]; then
        ROOT=$(mktemp -d "${TMPDIR:-/tmp}/theodolite-pod.XXXXXX")
        say "dry run: the volume root is $ROOT (a temporary folder, left for you to read)"
    else
        ROOT=${THEODOLITE_POD_ROOT:-$DEFAULT_ROOT}
    fi
}

# The env file every pod script, and every shell on the pod, sources: the caches, the venv's Python, the atlas's data
# and the field notes all under the volume root, so a stopped pod restarts ready and nothing is left on the container's
# disk; the code offline (the downloads set it back for themselves alone).
write_env() {
    local f=$ROOT/env.sh
    {
        printf '# Written by scripts/pod/setup.sh on %s: everything under %s, so a stopped pod restarts ready.\n' "$(stamp)" "$ROOT"
        printf '# No secret belongs here: a token is handed to the one command that needs it (the download), never written.\n'
        printf 'export THEODOLITE_POD_ROOT=%q\n' "$ROOT"
        printf 'export HF_HOME=%q HF_HUB_OFFLINE=1\n' "$ROOT/hf"
        printf 'export UV_CACHE_DIR=%q UV_PYTHON_INSTALL_DIR=%q UV_INSTALL_DIR=%q\n' "$ROOT/.uv-cache" "$ROOT/.uv-python" "$ROOT/bin"
        printf 'export THEODOLITE_ATLAS_DIR=%q\n' "$ROOT/atlas_data"
        printf 'export XDG_DATA_HOME=%q\n' "$ROOT/share"
        # shellcheck disable=SC2016  # $PATH is for env.sh to expand when sourced
        printf 'case ":$PATH:" in *:%q:*) ;; *) export PATH=%q:"$PATH" ;; esac\n' "$ROOT/bin" "$ROOT/bin"
    } >"$f.tmp"
    mv "$f.tmp" "$f"
}

# the folders every job writes to, made at setup so a sync finds each from the first
make_folders() {
    mkdir -p "$ROOT/logs" "$ROOT/pilot" "$ROOT/hf" "$ROOT/share" "$ROOT/bin" \
        "$ROOT/atlas_data/census/runs" "$ROOT/atlas_data/index/steer_lens.d" \
        "$ROOT/atlas_data/stand-in/census/runs" "$ROOT/atlas_data/stand-in/index/steer_lens.d"
}

# source the env file (a dry run writes one if setup.sh has not), and the stand-in's flags when dry
load_env() {
    if [ ! -f "$ROOT/env.sh" ]; then
        [ "${DRY:-0}" = 1 ] || die "there is no $ROOT/env.sh: run scripts/pod/setup.sh on this pod first (or give --root)"
        write_env
        make_folders
    fi
    # shellcheck disable=SC1091
    . "$ROOT/env.sh"
    mkdir -p "$ROOT/logs"
    STAND=()
    if [ "${DRY:-0}" = 1 ]; then
        STAND=(--stand-in)
        export CUDA_VISIBLE_DEVICES=
    fi
}

# the venv's Python: $POD_PYTHON, else the checkout's .venv (setup.sh makes it)
pod_python() {
    if [ -n "${POD_PYTHON:-}" ]; then
        [ -x "$POD_PYTHON" ] || die "POD_PYTHON=$POD_PYTHON is not a Python"
        PY=$POD_PYTHON
    elif [ -x "$REPO/.venv/bin/python" ]; then
        PY=$REPO/.venv/bin/python
    else
        die "there is no venv at $REPO/.venv: run scripts/pod/setup.sh first (or set POD_PYTHON)"
    fi
}

# One GPU job holds the card at a time: the card's lock (logs/gpu.lock, flock on fd 9) taken and held until this shell
# ends (its children hold it with it), or the script refused in plain words while another job holds it. A census
# shard's job, the pilot and setup.sh's self-test each take it, so none is measured, or runs out of memory, on a card
# another shares.
hold_card() {
    exec 9>"$ROOT/logs/gpu.lock"
    flock -n 9 || die "another GPU job holds this pod's card (logs/gpu.lock: a census shard, logs/*.started without an end, a pilot or" \
        "setup.sh's self-test): one at a time"
}

# where census runs land: the stand-in's own folder on a dry run, as census_run.py --stand-in puts them
census_runs_dir() {
    if [ "${DRY:-0}" = 1 ]; then
        printf '%s' "$THEODOLITE_ATLAS_DIR/stand-in/census/runs"
    else
        printf '%s' "${THEODOLITE_CENSUS_DIR:-$THEODOLITE_ATLAS_DIR/census/runs}"
    fi
}
