#!/bin/bash
# =======================================================================
# fetch results + logs from trillium back to local mac.
#
# only pulls plot/json/npz files and slurm .out/.err logs — never any
# large files (model weights, hf cache) that would bloat the local repo.
# single rsync per phase, no error masking (per docs/trillium.md §3.3).
#
# usage:
#   bash slurm/fetch_results.sh                    # all results, no logs
#   bash slurm/fetch_results.sh <tag>              # one tag's results
#   bash slurm/fetch_results.sh <tag> --logs       # tag's results + logs
#   bash slurm/fetch_results.sh --logs             # all results + logs
# =======================================================================
set -euo pipefail

REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/swiglu"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TAG=""
FETCH_LOGS=false
for arg in "$@"; do
    case "$arg" in
        --logs) FETCH_LOGS=true ;;
        *)      TAG="$arg" ;;
    esac
done

mkdir -p "${LOCAL_DIR}/results" "${LOCAL_DIR}/logs"

# ── results ─────────────────────────────────────────────────────────
echo "[fetch] results"
if [ -n "$TAG" ]; then
    SRC="${REMOTE}:${REMOTE_DIR}/results/${TAG}/"
    DST="${LOCAL_DIR}/results/${TAG}/"
    mkdir -p "$DST"
else
    SRC="${REMOTE}:${REMOTE_DIR}/results/"
    DST="${LOCAL_DIR}/results/"
fi

BEFORE=$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')
rsync -avz \
    --include='*/' \
    --include='*.json' \
    --include='*.png' \
    --include='*.npz' \
    --include='*.csv' \
    --include='*.txt' \
    --exclude='*' \
    "$SRC" "$DST"
AFTER=$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "  result files: ${BEFORE} -> ${AFTER} ($((AFTER - BEFORE)) new)"

# ── logs ────────────────────────────────────────────────────────────
if [ "$FETCH_LOGS" = true ]; then
    echo "[fetch] logs"
    BEFORE=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ -n "$TAG" ]; then
        rsync -avz \
            --include="*${TAG}*.out" \
            --include="*${TAG}*.err" \
            --exclude='*' \
            "${REMOTE}:${REMOTE_DIR}/logs/" \
            "${LOCAL_DIR}/logs/"
    else
        rsync -avz \
            --include='*.out' \
            --include='*.err' \
            --exclude='*' \
            "${REMOTE}:${REMOTE_DIR}/logs/" \
            "${LOCAL_DIR}/logs/"
    fi
    AFTER=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  log files: ${BEFORE} -> ${AFTER} ($((AFTER - BEFORE)) new)"
fi

echo "[fetch] done"
