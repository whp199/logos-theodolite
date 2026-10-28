#!/usr/bin/env bash
# The home venv's exact versions (uv pip freeze) written to a dated constraints file, so that a rented card's venv can be
# pinned to the home card's runtime class (critic M2): pyproject.toml pins floors only and there is no lock file, so
# the versions a census run records (census.runtime_record: torch and its CUDA, transformers, interp-engine,
# compressed-tensors) are whatever was installed last. scripts/pod/setup.sh --constraints <file> installs from it.
#
#     scripts/export_constraints.sh                        # .venv -> scripts/pod/constraints/home-<date>.txt
#     scripts/export_constraints.sh --python ~/other/.venv/bin/python --out constraints.txt
#     scripts/export_constraints.sh --dry-run              # into a temporary folder, the checkout left as it is
#
# The file is a pip/uv constraints file: one name==version a line, after a header of comments naming the date, the
# commit, the Python, torch's CUDA and the torch backend uv needs to fetch that build (`# torch-backend: cu130`, which
# setup.sh reads). A package installed from a local path or URL cannot be fetched on a pod: it is written commented
# out, and said. Nothing secret is read or written.
set -euo pipefail
PROG=export_constraints.sh
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --python PATH   the interpreter whose environment is frozen (default: <checkout>/.venv/bin/python)
  --out FILE      where to write (default: scripts/pod/constraints/home-<YYYY-MM-DD>.txt in the checkout)
  --dry-run       write into a temporary folder instead, and say where
  -h, --help      this
EOF
}

die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }

PY="$REPO/.venv/bin/python"
OUT=""
DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --python) [ $# -ge 2 ] || die "--python needs a path"; PY=$2; shift 2 ;;
        --out) [ $# -ge 2 ] || die "--out needs a file"; OUT=$2; shift 2 ;;
        --dry-run|--stand-in) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

[ -x "$PY" ] || die "there is no Python at $PY: give the venv's with --python"
UV=$(command -v uv || true)
if [ -z "$UV" ]; then
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$c" ]; then UV=$c; break; fi
    done
fi
[ -n "$UV" ] || die "uv is needed to read the environment (https://github.com/astral-sh/uv)"

day=$(date -u +%Y-%m-%d)
if [ "$DRY" = 1 ]; then
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/theodolite-constraints.XXXXXX")
    OUT=${OUT:+$tmp/$(basename "$OUT")}
    OUT=${OUT:-$tmp/home-$day.txt}
fi
OUT=${OUT:-$REPO/scripts/pod/constraints/home-$day.txt}
mkdir -p "$(dirname "$OUT")"

err=$(mktemp)
trap 'rm -f "$err"' EXIT
if ! frozen=$("$UV" pip freeze --exclude-editable --python "$PY" 2>"$err"); then
    cat "$err" >&2
    die "uv pip freeze failed on $PY"
fi
# torch's own build: its version carries the CUDA it was built for (2.14.1+cu130), which uv picks by --torch-backend
pyver=unknown torchver=none torchcuda=none
read -r pyver torchver torchcuda < <("$PY" -I -c '
import sys
try:
    import torch
    t, c = torch.__version__, torch.version.cuda or "none"
except Exception:
    t, c = "none", "none"
print("%d.%d.%d" % sys.version_info[:3], t, c)')
case "$torchcuda" in
    none) backend=cpu ;;
    *) backend=cu$(printf '%s' "$torchcuda" | tr -d .)
       case "$backend" in cu??) backend=${backend}0 ;; esac ;;
esac
[ "$torchver" = none ] && backend=""
commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
state=$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null | head -1)

local_pins=0
{
    printf '# The exact versions of the environment of %s, for scripts/pod/setup.sh --constraints (uv pip install -c).\n' "$PY"
    printf '# written: %s by scripts/export_constraints.sh at %s%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" \
        "$([ -n "$state" ] && printf ' (with uncommitted changes)')"
    printf '# machine: %s %s\n' "$(uname -m)" "$(uname -s)"
    printf '# python: %s\n' "$pyver"
    printf '# torch: %s, CUDA %s\n' "$torchver" "$torchcuda"
    [ -n "$backend" ] && printf '# torch-backend: %s\n' "$backend"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
            *" @ "*|-e\ *|file:*) printf '# not portable (installed from a path or URL): %s\n' "$line"; local_pins=$((local_pins + 1)) ;;
            *) printf '%s\n' "$line" ;;
        esac
    done <<<"$frozen"
} >"$OUT.tmp"
mv "$OUT.tmp" "$OUT"

pins=$(grep -c '^[A-Za-z0-9]' "$OUT" || true)
printf '%s: %s pins from %s (python %s, torch %s, CUDA %s) -> %s\n' "$PROG" "$pins" "$PY" "$pyver" "$torchver" "$torchcuda" "$OUT"
[ -n "$backend" ] && printf '%s: a pod installs torch with uv'"'"'s --torch-backend %s (setup.sh reads it from the file)\n' "$PROG" "$backend"
[ "$local_pins" -gt 0 ] && printf '%s: %s installed from a path or URL, written commented out: install them on the pod by hand\n' "$PROG" "$local_pins"
[ "$DRY" = 1 ] && printf '%s: dry run: written into %s, the checkout untouched\n' "$PROG" "$(dirname "$OUT")"
exit 0
