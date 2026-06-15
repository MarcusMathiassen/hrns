#!/bin/sh
# hrns — single-line install:  curl -fsSL https://raw.githubusercontent.com/MarcusMathiassen/hrns/main/install.sh | sh

set -eu

REPO="MarcusMathiassen/hrns"
BRANCH="${HRNS_BRANCH:-main}"
DEST="${HRNS_HOME:-$HOME/.hrns}"

# ---- check Python ----------------------------------------------------------
PY=$(command -v python3 || command -v python || echo "")
if [ -z "$PY" ]; then
    echo "ERROR: Python not found. Install Python 3.9+ first." >&2
    exit 1
fi

PY_VER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -lt 9 ]; then
    echo "ERROR: Python 3.9+ required (found $PY_VER)." >&2
    exit 1
fi

echo "  python $PY_VER found"

# ---- download hrns ---------------------------------------------------------
echo "  downloading hrns ($BRANCH)..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

URL="https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMP/hrns.tar.gz"
elif command -v wget >/dev/null 2>&1; then
    wget -q "$URL" -O "$TMP/hrns.tar.gz"
else
    echo "ERROR: need curl or wget" >&2
    exit 1
fi

tar xzf "$TMP/hrns.tar.gz" -C "$TMP"
SRC="$TMP/hrns-$BRANCH"

# ---- install ---------------------------------------------------------------
echo "  installing..."
cd "$SRC"
if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
    "$PY" -m pip install .
else
    "$PY" -m pip install --user .
fi

# ---- done ------------------------------------------------------------------
echo ""
echo "  hrns installed."
echo ""
echo "  Run:  hrns"
echo ""
echo "  First:  hrns  ->  /connect  (set your DeepSeek API key)"
echo ""
