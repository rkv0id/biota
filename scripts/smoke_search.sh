#!/usr/bin/env bash
# Shared scaffold for biota search smoke tests across device/transport variants.
#
# Mirror of scripts/smoke_ecosystem.sh. Builds a wheel, installs it into a
# hermetic venv, runs `biota search` with the appropriate flags for the
# requested device/transport combo, and verifies an archive was produced.
#
# Three transports x three devices = up to nine variants (cluster + mps not
# supported because Linux nodes don't have MPS). Each variant is a 5-line
# justfile recipe that exports env vars and invokes this script.
#
# Required env vars:
#   SMOKE_LABEL         tag used in log lines, e.g. "smoke-local-cuda-search"
#   SMOKE_DEVICE        cpu | mps | cuda
#   SMOKE_TRANSPORT     noray | local | cluster
#
# Required for cluster transport:
#   SMOKE_RAY_ADDRESS   passed to --ray-address (e.g. 10.10.12.1:6379)
#
# Optional:
#   SMOKE_BATCH_SIZE    defaults to 4 (cpu) or 16 (mps/cuda)
#   SMOKE_WORKERS       defaults to 4 (local) or 3 (cluster); ignored for noray
#   SMOKE_BUDGET        defaults to 20 (noray/local) or 30 (cluster)
#
# All venv lives under /tmp and is cleaned up on exit.

set -euo pipefail

: "${SMOKE_LABEL:?SMOKE_LABEL required}"
: "${SMOKE_DEVICE:?SMOKE_DEVICE required (cpu|mps|cuda)}"
: "${SMOKE_TRANSPORT:?SMOKE_TRANSPORT required (noray|local|cluster)}"

SMOKE_VENV=/tmp/biota-smoke-venv

log() { echo "[$SMOKE_LABEL] $*"; }

cleanup() {
    log "cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
}
trap cleanup EXIT

log "cleaning previous smoke venv (if any)..."
rm -rf "$SMOKE_VENV"

log "building wheel..."
rm -f dist/biota-*.whl
uv build --wheel >/dev/null
WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
log "wheel: $WHEEL"

uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"

# Resolve defaults from device and transport.
if [ -z "${SMOKE_BATCH_SIZE:-}" ]; then
    if [ "$SMOKE_DEVICE" = "cpu" ]; then
        SMOKE_BATCH_SIZE=4
    else
        SMOKE_BATCH_SIZE=16
    fi
fi

if [ -z "${SMOKE_BUDGET:-}" ]; then
    if [ "$SMOKE_TRANSPORT" = "cluster" ]; then
        SMOKE_BUDGET=30
    else
        SMOKE_BUDGET=20
    fi
fi
# Random-phase ~25% of budget, matching pre-extraction recipes.
SMOKE_RANDOM_PHASE=$((SMOKE_BUDGET / 4))

# Build transport-specific args.
TRANSPORT_ARGS=()
case "$SMOKE_TRANSPORT" in
    noray)
        # No --local-ray, no --ray-address. Sequential batch dispatch.
        ;;
    local)
        TRANSPORT_ARGS+=(--local-ray)
        TRANSPORT_ARGS+=(--workers "${SMOKE_WORKERS:-4}")
        ;;
    cluster)
        : "${SMOKE_RAY_ADDRESS:?SMOKE_RAY_ADDRESS required for cluster transport}"
        TRANSPORT_ARGS+=(--ray-address "$SMOKE_RAY_ADDRESS")
        TRANSPORT_ARGS+=(--workers "${SMOKE_WORKERS:-3}")
        ;;
    *)
        echo "[$SMOKE_LABEL] unknown SMOKE_TRANSPORT=$SMOKE_TRANSPORT" >&2
        exit 2
        ;;
esac

# MPS needs the fallback flag for ops without MPS implementations.
ENV_ARGS=()
if [ "$SMOKE_DEVICE" = "mps" ]; then
    ENV_ARGS+=(env PYTORCH_ENABLE_MPS_FALLBACK=1)
fi

log "running search (device=$SMOKE_DEVICE, transport=$SMOKE_TRANSPORT, batch=$SMOKE_BATCH_SIZE, budget=$SMOKE_BUDGET)..."
# cd /tmp so we are NOT in the source tree - guards against any tool
# inadvertently re-adding src/ to sys.path
cd /tmp && ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "$SMOKE_VENV/bin/biota" search \
    --preset dev \
    --budget "$SMOKE_BUDGET" --random-phase "$SMOKE_RANDOM_PHASE" \
    --device "$SMOKE_DEVICE" --batch-size "$SMOKE_BATCH_SIZE" \
    --grid 32 --steps 110 \
    ${TRANSPORT_ARGS[@]+"${TRANSPORT_ARGS[@]}"}

log "done."
