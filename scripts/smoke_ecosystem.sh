#!/usr/bin/env bash
# Shared scaffold for biota ecosystem smoke tests across device/topology variants.
#
# Why this exists: a CPU local-Ray smoke, an MPS local-Ray smoke, a CUDA
# local-Ray smoke, a CPU cluster smoke, and a CUDA cluster smoke all share
# the same outer structure (build wheel, hermetic venv, seed archive, write
# yaml, run, verify, clean up). Without this script, each justfile recipe
# would be ~80 lines of duplicated shell. With it, each recipe is a 5-line
# env-var-setting wrapper.
#
# Required env vars:
#   SMOKE_LABEL         tag used in log lines, e.g. "smoke-ray-cuda-ecosystem"
#   SMOKE_DEVICE        cpu | mps | cuda
#   SMOKE_RAY_MODE      local | cluster
#
# Required for cluster mode:
#   SMOKE_RAY_ADDRESS   passed to --ray-address (e.g. 10.10.12.1:6379)
#
# Optional:
#   SMOKE_GPU_FRACTION  if unset, --gpu-fraction is omitted (CLI derives from device)
#   SMOKE_WORKERS       defaults to 2
#
# All venv/data lives under /tmp and is cleaned up on exit.

set -euo pipefail

: "${SMOKE_LABEL:?SMOKE_LABEL required}"
: "${SMOKE_DEVICE:?SMOKE_DEVICE required (cpu|mps|cuda)}"
: "${SMOKE_RAY_MODE:?SMOKE_RAY_MODE required (local|cluster)}"

SMOKE_VENV=/tmp/biota-smoke-venv
SMOKE_DATA=/tmp/biota-smoke-eco
SMOKE_WORKERS="${SMOKE_WORKERS:-2}"

log() { echo "[$SMOKE_LABEL] $*"; }

cleanup() {
    log "cleaning up smoke venv and data..."
    rm -rf "$SMOKE_VENV" "$SMOKE_DATA"
}
trap cleanup EXIT

log "cleaning previous smoke venv and data (if any)..."
rm -rf "$SMOKE_VENV" "$SMOKE_DATA"

log "building wheel..."
rm -f dist/biota-*.whl
uv build --wheel >/dev/null
WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
log "wheel: $WHEEL"

uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"

mkdir -p "$SMOKE_DATA/archive/seed-run"
log "seeding archive deterministically..."
"$SMOKE_VENV/bin/python" - <<'PYEOF'
import pickle
import numpy as np
from biota.search.archive import Archive
from biota.search.result import RolloutResult

k = 3
creature = RolloutResult(
    params={
        "R": 8.0,
        "r": [0.5] * k,
        "m": [0.15] * k,
        "s": [0.015] * k,
        "h": [0.5] * k,
        "a": [[0.5, 0.5, 0.5]] * k,
        "b": [[0.5, 0.5, 0.5]] * k,
        "w": [[0.5, 0.5, 0.5]] * k,
    },
    seed=0,
    descriptors=(0.3, 0.5, 0.6),
    quality=0.8,
    rejection_reason=None,
    thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
    parent_cell=None,
    created_at=0.0,
    compute_seconds=1.0,
)
arc = Archive()
arc._cells[(5, 8, 3)] = creature
with open("/tmp/biota-smoke-eco/archive/seed-run/archive.pkl", "wb") as f:
    pickle.dump(arc, f)
print("seeded cell (5, 8, 3) in seed-run")
PYEOF

cat > "$SMOKE_DATA/experiments.yaml" <<'YAMLEOF'
experiments:
  - name: alpha
    grid: 64
    steps: 10
    snapshot_every: 5
    border: torus
    output_format: gif
    spawn: {min_dist: 20, patch: 8, seed: 0}
    sources:
      - {run: seed-run, cell: [5, 8, 3], n: 2}
  - name: beta
    grid: 64
    steps: 10
    snapshot_every: 5
    border: wall
    output_format: gif
    spawn: {min_dist: 20, patch: 8, seed: 1}
    sources:
      - {run: seed-run, cell: [5, 8, 3], n: 2}
YAMLEOF

# Build the CLI invocation. Ray flag depends on mode; gpu-fraction passed
# only when explicitly set (let the CLI derive its default from device).
RAY_ARGS=()
if [ "$SMOKE_RAY_MODE" = "local" ]; then
    RAY_ARGS+=(--local-ray)
elif [ "$SMOKE_RAY_MODE" = "cluster" ]; then
    : "${SMOKE_RAY_ADDRESS:?SMOKE_RAY_ADDRESS required for cluster mode}"
    RAY_ARGS+=(--ray-address "$SMOKE_RAY_ADDRESS")
else
    echo "[$SMOKE_LABEL] unknown SMOKE_RAY_MODE=$SMOKE_RAY_MODE" >&2
    exit 2
fi

GPU_ARGS=()
if [ -n "${SMOKE_GPU_FRACTION:-}" ]; then
    GPU_ARGS+=(--gpu-fraction "$SMOKE_GPU_FRACTION")
fi

# MPS needs the fallback flag for ops without MPS implementations.
ENV_ARGS=()
if [ "$SMOKE_DEVICE" = "mps" ]; then
    ENV_ARGS+=(env PYTORCH_ENABLE_MPS_FALLBACK=1)
fi

log "running 2-experiment ecosystem (device=$SMOKE_DEVICE, ray=$SMOKE_RAY_MODE, workers=$SMOKE_WORKERS, gpu_fraction=${SMOKE_GPU_FRACTION:-derived})..."
cd "$SMOKE_DATA" && "${ENV_ARGS[@]}" "$SMOKE_VENV/bin/biota" ecosystem \
    --config experiments.yaml \
    --archive-dir archive \
    --output-dir ecosystem \
    --device "$SMOKE_DEVICE" \
    --workers "$SMOKE_WORKERS" \
    "${RAY_ARGS[@]}" \
    "${GPU_ARGS[@]}"

test -d "$SMOKE_DATA/ecosystem"
log "runs produced:"
ls -1 "$SMOKE_DATA/ecosystem"
test "$(ls -1 "$SMOKE_DATA/ecosystem" | wc -l)" -eq 2
log "done."
