#!/usr/bin/env bash
# Prepare a rented card for the atlas's hard jobs (after neuron-explorer's sweep/setup_pod.sh): everything under one
# volume root (default /workspace, a RunPod network volume's mount), so that a stopped pod restarts ready and a pod
# deleted loses nothing but its container. Run it from the checkout, itself on the volume:
#
#     git clone <repo> /workspace/logos-theodolite && cd /workspace/logos-theodolite && git checkout <commit>
#     scripts/pod/setup.sh --constraints scripts/pod/constraints/home-2026-10-28.txt
#     HF_TOKEN=$RUNPOD_SECRET_hf scripts/pod/setup.sh ...   # a token only if the download without one is refused
#     scripts/pod/setup.sh --dry-run                        # the whole flow on the CPU stand-in, in a temporary root
#
# In order: <root>/env.sh, sourced by every pod script (and from ~/.bashrc), putting HF_HOME, uv's caches and Python,
# THEODOLITE_ATLAS_DIR (indexes, census runs, the scientist's folder) and XDG_DATA_HOME (field notes, saved prompts)
# on the volume, with HF_HUB_OFFLINE=1; rsync, tmux and zstd, and uv, where missing; the checkpoint and the two lenses
# downloaded in the background while the venv is made, pinned to the snapshots the study used (README §v), each tried
# without a token first; the venv (uv, Python 3.12) from the constraints file when one is given (scripts/
# export_constraints.sh on the home machine: the pins are part of a run's runtime class), torch fetched for the backend
# the file names, then checked to hold every pinned version; the stats index's sha256 (the census reads its push unit
# there, and shards must share it); and last a self-test line: the card, the versions, the unpacker's test against the
# reference on every 4-bit linear, and the runtime class (the self-test holds the card's lock, logs/gpu.lock, as a
# census shard and the pilot do, and is refused while another job holds it).
#
# Secrets: whether the checkpoint is gated is not settled here (the plan for rented cards, §6), so the download goes
# first with no token at all. HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is read from this script's environment only, then
# taken out of it at once: it reaches the one retry of the download that needs it, and no other process; it is never
# printed or written (not to env.sh, not to a log, not to $HF_HOME/token). Every Python this starts runs with every
# variable naming a token, key, secret or credential left out of its environment (common.sh, scrubbed).
#
# --dry-run runs the same flow in a temporary root (or --root) on the CPU stand-in: nothing is installed or fetched
# (the venv is the one running, POD_PYTHON or the checkout's, checked against --constraints offline), the download is
# run only by a command named in POD_HF_CMD (a check's fake), and the self-test loads scripts/atlas_stand_in.py.
set -euo pipefail
PROG=setup.sh
# shellcheck source=scripts/pod/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_REPO=google/gemma-4-31B-it-qat-w4a16-ct
MODEL_REV=52f3f65bc7a02d555763bc923bd1d9094898219d
LENS_REPO=neuronpedia/jacobian-lens
LENS_REV=e166deb5ad540932100ef803c8c9bcc2a5453694
LENS_FILES=(gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt
            gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt)

usage() {
    help_text "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --root DIR              the volume root (default $THEODOLITE_POD_ROOT, else /workspace; a dry run: a temporary one)
  --constraints FILE      install the venv from these pins (scripts/export_constraints.sh); its `# torch-backend:` line
                          picks torch's build
  --torch-backend NAME    uv's --torch-backend (default: the constraints file's, else auto)
  --python-version V      the venv's Python (default 3.12)
  --act-scale-sha256 HEX  refuse a stats index (atlas_data/index/act_scale.npz) with another sha256
  --unpacker MODE         the unpacker the self-test installs and tests (default lut; reference, lut, compiled, resident)
  --no-download           leave the checkpoint and lenses as they are
  --no-selftest           stop before loading the model
  --dry-run, --stand-in   the whole flow on the CPU stand-in in a temporary root (see above)
  -h, --help              this
EOF
}

ROOT="" DRY=0 CONSTR="" TB="" PYV=3.12 ACT_SHA="" UNP=lut DOWNLOAD=1 SELFTEST=1
while [ $# -gt 0 ]; do
    case "$1" in
        --root) [ $# -ge 2 ] || die "--root needs a folder"; ROOT=$2; shift 2 ;;
        --constraints) [ $# -ge 2 ] || die "--constraints needs a file"; CONSTR=$2; shift 2 ;;
        --torch-backend) [ $# -ge 2 ] || die "--torch-backend needs a name"; TB=$2; shift 2 ;;
        --python-version) [ $# -ge 2 ] || die "--python-version needs a version"; PYV=$2; shift 2 ;;
        --act-scale-sha256) [ $# -ge 2 ] || die "--act-scale-sha256 needs a digest"; ACT_SHA=$2; shift 2 ;;
        --unpacker) [ $# -ge 2 ] || die "--unpacker needs a mode"; UNP=$2; shift 2 ;;
        --no-download) DOWNLOAD=0; shift ;;
        --no-selftest) SELFTEST=0; shift ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
case "$UNP" in reference|lut|compiled|resident) ;; *) die "--unpacker is reference, lut, compiled or resident, not $UNP" ;; esac
if [ -n "$CONSTR" ]; then
    [ -f "$CONSTR" ] || die "there is no constraints file $CONSTR"
    CONSTR=$(cd "$(dirname "$CONSTR")" && pwd)/$(basename "$CONSTR")
fi

# the token, kept in this shell alone (not exported) for the download's retry, and out of the environment from here on
_hf_token=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

resolve_root
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup.log") 2> >(tee -a "$ROOT/logs/setup.log" >&2)
say "$(stamp): setting up under $ROOT, the checkout at $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
[ -n "$_hf_token" ] && say "a Hugging Face token is in the environment: kept for the download alone, used only if the download without one is refused"
case "$REPO/" in
    "$ROOT"/*) ;;
    *) [ "$DRY" = 1 ] || say "warning: the checkout ($REPO) is not under $ROOT: its venv is lost with the container; clone it onto the volume" ;;
esac

# 1. the env file
write_env
make_folders
# shellcheck disable=SC1091
. "$ROOT/env.sh"
if [ "$DRY" = 0 ]; then
    grep -qsF "$ROOT/env.sh" "$HOME/.bashrc" || printf '. %q\n' "$ROOT/env.sh" >>"$HOME/.bashrc"
fi
say "1. $ROOT/env.sh: HF_HOME, uv's caches, THEODOLITE_ATLAS_DIR and XDG_DATA_HOME under the root; HF_HUB_OFFLINE=1"

# 2. the tools
missing=()
for t in rsync tmux zstd; do command -v "$t" >/dev/null || missing+=("$t"); done
if [ ${#missing[@]} -gt 0 ]; then
    if [ "$DRY" = 1 ]; then
        say "2. not installed here: ${missing[*]} (a pod gets them from apt; the dry run goes on without)"
    else
        say "2. installing ${missing[*]}"
        (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}") >>"$ROOT/logs/apt.log" 2>&1 \
            || say "warning: apt-get could not install ${missing[*]} (see logs/apt.log)"
    fi
else
    say "2. rsync, tmux and zstd are here"
fi
UV=$(command -v uv || true)
if [ -z "$UV" ] && [ -x "$ROOT/bin/uv" ]; then UV=$ROOT/bin/uv; fi
if [ -z "$UV" ]; then
    [ "$DRY" = 0 ] || die "uv is not installed here, and a dry run installs nothing"
    say "   installing uv into $ROOT/bin"
    curl -LsSf https://astral.sh/uv/install.sh | scrubbed env UV_INSTALL_DIR="$ROOT/bin" INSTALLER_NO_MODIFY_PATH=1 sh >>"$ROOT/logs/uv-install.log" 2>&1
    UV=$ROOT/bin/uv
fi
say "   uv: $("$UV" --version 2>/dev/null)"

# 3. the downloads, in the background beside the venv
DL_LOG=$ROOT/logs/download.log
hub_pin=""
[ -n "$CONSTR" ] && hub_pin=$(sed -n 's/^huggingface-hub==//p' "$CONSTR" | head -1)
if [ "$DRY" = 1 ]; then
    if [ -n "${POD_HF_CMD:-}" ]; then
        read -r -a HF_CMD <<<"$POD_HF_CMD"
    else
        HF_CMD=()
    fi
else
    HF_CMD=("$UV" tool run --from "huggingface_hub${hub_pin:+==$hub_pin}" hf)
fi

fetch() {  # repo revision [files...]: without a token first, then, if one was given and that was refused, with it
    local repo=$1 rev=$2
    shift 2
    printf '%s: %s at %s, without a token\n' "$(stamp)" "$repo" "$rev" >>"$DL_LOG"
    # no token at all: none in the environment, and none read from a file (huggingface_hub's own switch)
    if scrubbed env HF_HUB_OFFLINE=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 "${HF_CMD[@]}" download "$repo" --revision "$rev" "$@" \
        >>"$DL_LOG" 2>&1; then
        return 0
    fi
    if [ -z "$_hf_token" ]; then
        printf '%s: %s refused without a token, and none was given (HF_TOKEN in setup.sh'"'"'s environment)\n' "$(stamp)" "$repo" >>"$DL_LOG"
        return 1
    fi
    printf '%s: %s refused without a token; again with the one given (not shown)\n' "$(stamp)" "$repo" >>"$DL_LOG"
    (
        export HF_TOKEN="$_hf_token"
        scrubbed --keep HF_TOKEN env HF_HUB_OFFLINE=0 "${HF_CMD[@]}" download "$repo" --revision "$rev" "$@"
    ) >>"$DL_LOG" 2>&1
}

download() {
    rm -f "$ROOT/logs/download.done" "$ROOT/logs/download.failed"
    if fetch "$MODEL_REPO" "$MODEL_REV" && fetch "$LENS_REPO" "$LENS_REV" "${LENS_FILES[@]}"; then
        stamp >"$ROOT/logs/download.done"
    else
        stamp >"$ROOT/logs/download.failed"
        return 1
    fi
}

DL=""
if [ "$DOWNLOAD" = 0 ]; then
    say "3. downloads left as they are (--no-download)"
elif [ ${#HF_CMD[@]} -eq 0 ]; then
    say "3. dry run: no download (it would fetch $MODEL_REPO at $MODEL_REV and $LENS_REPO at $LENS_REV, ${#LENS_FILES[@]} files, into $HF_HOME)"
else
    say "3. downloading $MODEL_REPO at ${MODEL_REV:0:8} and $LENS_REPO at ${LENS_REV:0:8} into $HF_HOME, in the background (logs/download.log)"
    download &
    DL=$!
fi

# 4. the venv, from the constraints
check_pins() {  # does the venv hold every pinned version? uv, offline, changing nothing
    local out
    if out=$("$UV" pip install --dry-run --offline -c "$CONSTR" -r "$REPO/pyproject.toml" --python "$PY" 2>&1); then
        if printf '%s' "$out" | grep -q 'Would make no changes'; then
            say "   the venv holds every version $CONSTR pins"
            return 0
        fi
    fi
    printf '%s\n' "$out" | grep -E '^ *[-+~]|Would|error' | head -20 >&2 || true
    return 1
}
if [ -n "$CONSTR" ] && [ -z "$TB" ]; then
    TB=$(sed -n 's/^# torch-backend: *//p' "$CONSTR" | head -1)
fi
if [ "$DRY" = 1 ]; then
    pod_python
    say "4. dry run: the venv is the running one ($PY), nothing installed"
    if [ -n "$CONSTR" ]; then
        check_pins || die "the venv does not hold the versions $CONSTR pins (uv's words above)"
    fi
else
    PY=$REPO/.venv/bin/python
    [ -x "$PY" ] || scrubbed "$UV" venv --python "$PYV" "$REPO/.venv" >>"$ROOT/logs/venv.log" 2>&1
    say "4. installing into $REPO/.venv" "${CONSTR:+from $CONSTR}" "(torch: ${TB:-auto})"
    [ -n "$CONSTR" ] || say "   warning: no constraints file: the newest versions pyproject.toml allows, a runtime class of their own (export the home venv's with scripts/export_constraints.sh)"
    pins=()
    [ -z "$CONSTR" ] || pins=(-c "$CONSTR")
    scrubbed env HF_HUB_OFFLINE=0 "$UV" pip install --python "$PY" -r "$REPO/pyproject.toml" "${pins[@]}" \
        --torch-backend "${TB:-auto}" >>"$ROOT/logs/venv.log" 2>&1 || die "the install failed (logs/venv.log)"
    if [ -n "$CONSTR" ]; then
        check_pins || die "the venv does not hold the versions $CONSTR pins (logs/venv.log)"
    fi
fi

# 5. the stats index, which the census's push unit is read from (shards must share it to merge)
if [ "$DRY" = 1 ]; then
    ACT=$THEODOLITE_ATLAS_DIR/stand-in/index/act_scale.npz
else
    ACT=$THEODOLITE_ATLAS_DIR/index/act_scale.npz
fi
if [ -f "$ACT" ]; then
    sha=$(sha256sum "$ACT" | cut -d' ' -f1)
    say "5. the stats index: $ACT, sha256 $sha"
    [ -z "$ACT_SHA" ] || [ "$ACT_SHA" = "$sha" ] || die "the stats index's sha256 is $sha, not $ACT_SHA: copy the home machine's again"
else
    [ -z "$ACT_SHA" ] || die "--act-scale-sha256 was given but there is no $ACT: copy the home machine's atlas_data/index up first"
    say "5. no stats index at $ACT: census_run.py reads its push unit over the study's 36 texts instead, a p50 of each card's own;"
    say "   copy the home machine's atlas_data/index/act_scale.npz up (rsync or the volume's S3 API) before a sharded census"
fi

# 6. the downloads' end
if [ -n "$DL" ]; then
    if wait "$DL"; then
        say "6. downloaded (logs/download.log)"
    else
        die "the download failed (logs/download.log): if it was refused, give a read-only token in HF_TOKEN for this script alone"
    fi
fi

# 7. the self-test: the card, the versions, the unpacker's test and the runtime class
if [ "$SELFTEST" = 0 ]; then
    say "7. no self-test (--no-selftest); ready but untested"
    exit 0
fi
STAND=()
[ "$DRY" = 1 ] && STAND=(--stand-in) && export CUDA_VISIBLE_DEVICES=
hold_card
cd "$REPO"
say "7. the self-test (the model loads; on a network volume this takes longer than the 16 s measured from a local disk)"
scrubbed "$PY" -I "$POD_DIR/pod_tool.py" selftest "${STAND[@]}" --unpacker "$UNP" --out "$ROOT/logs/selftest.json"
