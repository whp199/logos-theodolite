#!/usr/bin/env bash
# The census volunteer's one command (VOLUNTEERS.md): take one layer of the census on a card of your own, with the
# kit's fixed settings (volunteers/kit.env), and pack it to send. Run it from a checkout at the kit's tag:
#
#     scripts/volunteer/census.sh check            # can this machine take a layer? the card, its memory, the driver, the disk
#     scripts/volunteer/census.sh reserve 47       # a claim on layer 47, and the line to post
#     scripts/volunteer/census.sh setup            # the venv, the checkpoint, the lenses and the stats index (about 35 GB)
#     scripts/volunteer/census.sh canary 47        # one batch and the writer's floor, set against the home card's
#     scripts/volunteer/census.sh run 47           # the whole layer, detached (about 13 hours on a 5090); again to continue
#     scripts/volunteer/census.sh status 47        # how far, how fast, when: the line to post
#     scripts/volunteer/census.sh pack 47          # the run checked whole and packed: the file to send, the line to post
#     scripts/volunteer/census.sh release 47       # give the layer back
#     scripts/volunteer/census.sh --dry-run --root /tmp/v ...   # the same on the CPU stand-in (its layers 2 and 3)
#
# Everything lives under one folder (--root, default $THEODOLITE_POD_ROOT, else ~/theodolite-census): the env file,
# the downloads, the venv's caches, the runs, the logs, and claims/ with each claim's secret. Keep the secret: the
# bundle you send carries it, and it is how the bundle is known to come from the claim. Layers count from 0
# (page = layer + 1), and the kit takes only volunteers/kit.env's KIT_LAYERS.
#
# What goes home is the pack's bundle alone: the answers, the run's meta (the card's name, torch, CUDA and the library
# versions, the commit), the canary, the writer's floor and a manifest with the claim. The logs stay here (they name
# your folders). This script uploads nothing: you choose where the bundle goes. The census runs with every variable
# naming a token, key or secret left out of its environment (scripts/pod/common.sh, scrubbed).
set -euo pipefail
PROG=census.sh
# shellcheck source=scripts/pod/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/../pod/common.sh"
VOL_DIR=$REPO/scripts/volunteer

usage() {
    help_text "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options (before or after the command):
  --root DIR       the folder everything lives under (default $THEODOLITE_POD_ROOT, else ~/theodolite-census)
  --name NAME      reserve: the name the board shows (letters, digits, . _ -; default anon)
  --wait           run: wait for the run to end, then print its log's tail
  --dry-run        on the CPU stand-in (its layers 2 and 3, a few neurons); give --root to keep one folder across steps
  -h, --help       this
EOF
}

ROOT="" DRY=0 NAME=anon WAIT=0 CMD="" LAYER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root) [ $# -ge 2 ] || die "--root needs a folder"; ROOT=$2; shift 2 ;;
        --name) [ $# -ge 2 ] || die "--name needs a name"; NAME=$2; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option $1 (see --help)" ;;
        *)
            if [ -z "$CMD" ]; then CMD=$1
            elif [ -z "$LAYER" ]; then LAYER=$1
            else die "one command and one layer, not also $1 (see --help)"
            fi
            shift ;;
    esac
done
[ -n "$CMD" ] || { usage; exit 2; }
case "$NAME" in *[!A-Za-z0-9._-]*|"") die "--name is letters, digits, '.', '_' and '-' only" ;; esac
[ ${#NAME} -le 24 ] || die "--name is 24 characters at most"

# the kit's settings, the stand-in's in their place on a dry run
# shellcheck source=volunteers/kit.env
. "$REPO/volunteers/kit.env"
if [ "$DRY" = 1 ]; then
    KIT_LAYERS=$KIT_DRY_LAYERS KIT_NEURONS=$KIT_DRY_NEURONS KIT_CONTROLS=$KIT_DRY_CONTROLS
    KIT_BATCH=$KIT_DRY_BATCH KIT_MAX_NEW=$KIT_DRY_MAX_NEW
fi
RUN_ARGS=(--design "$KIT_DESIGN" --neurons "$KIT_NEURONS" --random-controls "$KIT_CONTROLS" --seed "$KIT_SEED"
          --batch "$KIT_BATCH" --max-new "$KIT_MAX_NEW" --unpacker "$KIT_UNPACKER")
DRYF=()
[ "$DRY" = 1 ] && DRYF=(--dry-run)

if [ -z "$ROOT" ] && [ "$DRY" = 0 ]; then
    ROOT=${THEODOLITE_POD_ROOT:-$HOME/theodolite-census}
fi
resolve_root
CLAIMS=$ROOT/claims

need_layer() {
    [ -n "$LAYER" ] || die "$CMD needs a layer: census.sh $CMD <layer> (one of: $KIT_LAYERS)"
    LAYER=${LAYER#L}
    case " $KIT_LAYERS " in
        *" $LAYER "*) ;;
        *) die "the kit takes layers $KIT_LAYERS (counting from 0), not $LAYER: break_rel's doses are calibrated at 40-50 alone, and 45 is taken" ;;
    esac
    RUN=L$LAYER-$KIT_DESIGN
    CLAIM=$CLAIMS/L$LAYER.claim
}

claim_field() {  # a field of the layer's claim file ("" when there is none)
    [ -f "$CLAIM" ] || { printf ''; return; }
    sed -n "s/^$1=//p" "$CLAIM" | head -1
}

card_name() {  # the card the census will run on: CUDA_VISIBLE_DEVICES's first, else 0
    if [ "$DRY" = 1 ]; then printf 'CPU stand-in'; return; fi
    local i=${CUDA_VISIBLE_DEVICES:-0}
    i=${i%%,*}
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits -i "$i" 2>/dev/null \
        | head -1 | awk -F', *' '{ gsub(/"/, "", $1); printf "%s %d GB", $1, ($2 + 512) / 1024 }'
}

canary_dir() {  # where the canary, its check and the writer's floor are kept: beside the runs' folder
    local d
    d=$(census_runs_dir)/../canary
    mkdir -p "$d"
    (cd "$d" && pwd)
}

kit_tag_here() {  # is the checkout at the kit's tag, with nothing tracked changed?
    git -C "$REPO" tag --points-at HEAD 2>/dev/null | grep -qx "$KIT_TAG" || return 1
    [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]
}

cmd_check() {
    local bad=0 warn=0
    no() { say "no: $*"; bad=1; }
    hm() { say "warning: $*"; warn=1; }
    ok() { say "ok: $*"; }
    [ "$(uname -s)" = Linux ] && ok "Linux" || hm "this is $(uname -s): the kit is written for Linux (WSL2 untried)"
    if [ "$DRY" = 1 ]; then
        ok "a dry run: the CPU stand-in, no card needed"
    elif ! command -v nvidia-smi >/dev/null; then
        no "no nvidia-smi: the census needs an NVIDIA card and its driver"
    else
        local i=${CUDA_VISIBLE_DEVICES:-0} name mem cuda
        i=${i%%,*}
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$i" 2>/dev/null | head -1)
        mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i" 2>/dev/null | head -1 | tr -d ' ')
        cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
        [ -n "$name" ] || no "nvidia-smi names no card $i"
        if [ -n "$mem" ] && [ "$mem" -ge "$KIT_MIN_VRAM_MIB" ]; then
            ok "card $i: $name, $mem MiB"
        else
            no "card $i: ${name:-?}, ${mem:-?} MiB: the kit needs one card of at least $KIT_MIN_VRAM_MIB MiB (the model is about 22 GB, a batch of 64 about 5 GB more; it is not split across cards)"
        fi
        if [ -n "$cuda" ] && [ "$(printf '%s\n%s\n' "$KIT_MIN_CUDA" "$cuda" | sort -V | head -1)" = "$KIT_MIN_CUDA" ]; then
            ok "the driver runs CUDA $cuda"
        else
            no "the driver runs CUDA ${cuda:-?}: the pinned torch (2.14.1+cu130) needs a driver for CUDA $KIT_MIN_CUDA or newer (580 or later)"
        fi
        local apps
        apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$i" 2>/dev/null | grep -c . || true)
        [ "$apps" = 0 ] || hm "$apps process(es) hold memory on card $i: close them before the canary and the run"
    fi
    local have
    have=$(df -Pk "$ROOT" 2>/dev/null | awk 'NR == 2 { printf "%d", $4 / 1048576 }')
    if [ "$DRY" = 1 ]; then
        ok "a dry run: ${have:-?} GB free under $ROOT is plenty for the stand-in"
    elif [ -n "$have" ] && [ "$have" -ge "$KIT_MIN_DISK_GB" ]; then
        ok "$have GB free under $ROOT"
    elif [ -f "$ROOT/logs/download.done" ]; then
        ok "$have GB free under $ROOT, the downloads done"
    else
        no "${have:-?} GB free under $ROOT: the kit needs about $KIT_MIN_DISK_GB GB (checkpoint 22, lenses 6.4, venv, a run 2.6 and its bundle)"
    fi
    local ram
    ram=$(awk '/^MemTotal:/ { printf "%d", $2 / 1048576 }' /proc/meminfo 2>/dev/null || echo 0)
    [ "${ram:-0}" -ge 30 ] && ok "$ram GB of memory" || hm "${ram:-?} GB of memory: 32 GB or more is safer for loading the checkpoint"
    local t miss=()
    for t in git curl flock sha256sum tar zstd; do command -v "$t" >/dev/null || miss+=("$t"); done
    if [ ${#miss[@]} -gt 0 ]; then
        no "not installed: ${miss[*]} (e.g. sudo apt install ${miss[*]})"
    else
        ok "git, curl, flock, sha256sum, tar and zstd are here"
    fi
    command -v cc >/dev/null || command -v gcc >/dev/null \
        || hm "no C compiler (cc, gcc): Triton may need one to build its launcher for the compiled unpacker (e.g. sudo apt install build-essential)"
    if [ "$DRY" = 1 ]; then
        :
    elif kit_tag_here; then
        ok "the checkout is at $KIT_TAG, nothing changed"
    else
        no "the checkout is not at $KIT_TAG with nothing changed: git fetch --tags && git checkout $KIT_TAG (a run's commit is checked when it comes home)"
    fi
    if [ "$bad" = 1 ]; then
        say "this machine cannot take a layer as it is (the lines marked no)"
        return 1
    fi
    if [ "$warn" = 1 ]; then
        say "this machine can take a layer (read the warnings first): census.sh reserve <layer>, then setup"
    else
        say "this machine can take a layer: census.sh reserve <layer>, then setup"
    fi
}

cmd_reserve() {
    need_layer
    mkdir -p "$CLAIMS"
    chmod 700 "$CLAIMS"
    if [ -f "$CLAIM" ] && [ -z "$(claim_field released)" ]; then
        say "you hold a claim on layer $LAYER already ($CLAIM); its line again:"
    else
        local secret code
        secret=$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')
        code=$(printf '%s' "$secret" | sha256sum | cut -c1-8)
        ( umask 077
          { printf 'secret=%s\ncode=%s\nlayer=%s\nname=%s\ncard=%s\nkit=%s\nat=%s\n' "$secret" "$code" "$LAYER" "$NAME" "$(card_name)" "$KIT" "$(stamp)"
          } >"$CLAIM.tmp" && mv "$CLAIM.tmp" "$CLAIM" )
        say "a claim on layer $LAYER, its secret kept in $CLAIM (keep it: the bundle you send carries it). Post this line"
        say "in the thread or on the board ($KIT_BOARD_URL), and wait to see it on the board before the run:"
    fi
    printf '\nCENSUS RESERVE L%s claim=%s card="%s" name=%s kit=%s\n\n' "$LAYER" "$(claim_field code)" "$(claim_field card)" \
        "$(claim_field name)" "$KIT"
}

cmd_setup() {
    mkdir -p "$ROOT/atlas_data/index"
    local act=$ROOT/atlas_data/index/act_scale.npz
    local -a scripts
    if [ "$DRY" = 1 ]; then
        say "dry run: no stats index fetched (the stand-in reads its push unit over its own texts)"
        scripts=("$POD_DIR/setup.sh" --dry-run --root "$ROOT" --no-download --no-bashrc --unpacker "$KIT_UNPACKER")
    else
        if [ -f "$act" ] && [ "$(sha256sum "$act" | cut -d' ' -f1)" = "$KIT_ACT_SCALE_SHA256" ]; then
            say "the stats index is here: $act"
        else
            say "fetching the stats index (31 MB; the push's unit is read there, so every layer is pushed alike) from $KIT_ACT_SCALE_URL"
            curl -fL --retry 3 -o "$act.part" "$KIT_ACT_SCALE_URL" || die "could not fetch $KIT_ACT_SCALE_URL"
            [ "$(sha256sum "$act.part" | cut -d' ' -f1)" = "$KIT_ACT_SCALE_SHA256" ] || {
                rm -f "$act.part"; die "the stats index fetched is not the kit's (sha256 differs): ask on the board"; }
            mv "$act.part" "$act"
        fi
        scripts=("$POD_DIR/setup.sh" --root "$ROOT" --constraints "$REPO/$KIT_CONSTRAINTS" --act-scale-sha256 "$KIT_ACT_SCALE_SHA256"
                 --unpacker "$KIT_UNPACKER" --no-bashrc)
    fi
    "${scripts[@]}"
    say "set up under $ROOT; next: census.sh canary <your layer>"
}

cmd_canary() {
    need_layer
    load_env
    pod_python
    [ -f "$CLAIM" ] || say "warning: no claim on layer $LAYER here (census.sh reserve $LAYER): the canary is taken, but post a claim before the run"
    hold_card
    local out
    out=$(canary_dir)
    cd "$REPO"
    say "1. the canary: plan batch 0 of layer $LAYER's run, written alone (the model loads first)"
    scrubbed "$PY" -I scripts/census_run.py "${STAND[@]}" --layer "$LAYER" "${RUN_ARGS[@]}" --canary-only --out-dir "$out"
    say "2. the writer's floor: with no push, how far an answer agrees with itself across batch shapes, on this card"
    scrubbed "$PY" -I scripts/verify_atlas_writer.py "${STAND[@]}" --layer "$LAYER" --design "$KIT_DESIGN" --batches "$KIT_BATCH" \
        --max-new "$KIT_MAX_NEW" --seed "$KIT_SEED" --unpacker "$KIT_UNPACKER" --out "$out/L$LAYER.writer_floor.json"
    say "3. the canary set against the home card's"
    scrubbed "$PY" -I "$VOL_DIR/volunteer_tool.py" canary-check "${DRYF[@]}" --layer "$LAYER" --canary "$out/$RUN.canary.jsonl" \
        --out "$out/L$LAYER.canary_check.json"
    say "next: census.sh run $LAYER"
}

cmd_run() {
    need_layer
    load_env
    pod_python
    local out
    out=$(canary_dir)
    [ -f "$out/L$LAYER.canary_check.json" ] || die "take the canary first: census.sh canary $LAYER"
    if [ "$DRY" = 0 ] && ! kit_tag_here; then
        die "the checkout is not at $KIT_TAG with nothing changed (git status; git checkout $KIT_TAG): a run of other code is not taken home"
    fi
    [ -f "$CLAIM" ] || say "warning: no claim on layer $LAYER here: post one (census.sh reserve $LAYER) so no one else takes it"
    local w=()
    [ "$WAIT" = 1 ] && w=(--wait)
    "$POD_DIR/census_shard.sh" "${DRYF[@]}" --root "$ROOT" --layer "$LAYER" --design "$KIT_DESIGN" --no-timeout "${w[@]}" -- \
        "${RUN_ARGS[@]:2}"
    local bits
    bits=$(sed -n 's/.*"bits": *"\([a-z]*\)".*/\1/p' "$out/L$LAYER.canary_check.json" | head -1)
    say "the run is under way (or done, with --wait). It stops with the card's job ended or the machine down; the same"
    say "command continues it, answer for answer. Watch it: census.sh status $LAYER. Post this line:"
    printf '\nCENSUS START L%s claim=%s bits=%s card="%s"\n\n' "$LAYER" "$(claim_field code)" "${bits:-?}" "$(card_name)"
}

cmd_status() {
    need_layer
    load_env
    pod_python
    cd "$REPO"
    scrubbed "$PY" -I "$VOL_DIR/volunteer_tool.py" status "${DRYF[@]}" --layer "$LAYER" --run "$(census_runs_dir)/$RUN.jsonl" \
        --logs "$ROOT/logs" --claim "$CLAIM"
}

cmd_pack() {
    need_layer
    load_env
    pod_python
    command -v zstd >/dev/null || die "zstd is not installed (e.g. sudo apt install zstd)"
    [ -f "$CLAIM" ] || die "no claim on layer $LAYER here ($CLAIM): the bundle carries the claim's secret"
    cd "$REPO"
    scrubbed "$PY" -I "$VOL_DIR/volunteer_tool.py" pack "${DRYF[@]}" --layer "$LAYER" --run "$(census_runs_dir)/$RUN.jsonl" \
        --canary-dir "$(canary_dir)" --claim "$CLAIM" --out "$ROOT/outbox"
}

cmd_release() {
    need_layer
    [ -f "$CLAIM" ] || die "no claim on layer $LAYER here"
    [ -z "$(claim_field released)" ] || say "you released layer $LAYER already; its line again:"
    [ -n "$(claim_field released)" ] || printf 'released=%s\n' "$(stamp)" >>"$CLAIM"
    say "layer $LAYER given back. Post this line (it shows the secret, which ends the claim):"
    printf '\nCENSUS RELEASE L%s claim=%s secret=%s\n\n' "$LAYER" "$(claim_field code)" "$(claim_field secret)"
}

case "$CMD" in
    check) cmd_check ;;
    reserve) cmd_reserve ;;
    setup) cmd_setup ;;
    canary) cmd_canary ;;
    run) cmd_run ;;
    status) cmd_status ;;
    pack) cmd_pack ;;
    release) cmd_release ;;
    *) die "no command $CMD: check, reserve, setup, canary, run, status, pack or release (see --help)" ;;
esac
