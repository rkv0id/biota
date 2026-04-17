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
#   SMOKE_LABEL         tag used in log lines, e.g. "smoke-local-cuda-ecosystem"
#   SMOKE_DEVICE        cpu | mps | cuda
#   SMOKE_TRANSPORT     noray | local | cluster
#
# Required for cluster transport:
#   SMOKE_RAY_ADDRESS   passed to --ray-address (e.g. 10.10.12.1:6379)
#
# Optional:
#   SMOKE_GPU_FRACTION  if unset, --gpu-fraction is omitted (CLI derives from device)
#   SMOKE_WORKERS       defaults to 2; ignored for noray transport
#
# All venv/data lives under /tmp and is cleaned up on exit.

set -euo pipefail

: "${SMOKE_LABEL:?SMOKE_LABEL required}"
: "${SMOKE_DEVICE:?SMOKE_DEVICE required (cpu|mps|cuda)}"
: "${SMOKE_TRANSPORT:?SMOKE_TRANSPORT required (noray|local|cluster)}"

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
mkdir -p "$SMOKE_DATA/archive/seed-run-b"
log "seeding archive deterministically..."
"$SMOKE_VENV/bin/python" - <<'PYEOF'
import pickle
import numpy as np
from biota.search.archive import Archive
from biota.search.result import RolloutResult

k = 3
def make_creature(seed):
    rng = __import__('numpy').random.default_rng(seed)
    return RolloutResult(
        params={
            "R": float(rng.uniform(6.0, 12.0)),
            "r": list(rng.uniform(0.3, 0.8, k).astype(float)),
            "m": list(rng.uniform(0.1, 0.4, k).astype(float)),
            "s": list(rng.uniform(0.01, 0.08, k).astype(float)),
            "h": list(rng.uniform(0.2, 0.8, k).astype(float)),
            "a": [list(rng.uniform(0.1, 0.9, 3).astype(float)) for _ in range(k)],
            "b": [list(rng.uniform(0.1, 0.9, 3).astype(float)) for _ in range(k)],
            "w": [list(rng.uniform(0.05, 0.3, 3).astype(float)) for _ in range(k)],
        },
        seed=int(seed),
        descriptors=(0.3, 0.5, 0.6),
        quality=0.8,
        rejection_reason=None,
        thumbnail=np.zeros((16, 32, 32), dtype=np.uint8),
        parent_cell=None,
        created_at=0.0,
        compute_seconds=1.0,
    )

for run_id, seed, coords in [("seed-run", 1, (5, 8, 3)), ("seed-run-b", 2, (3, 4, 5))]:
    arc = Archive()
    arc._cells[coords] = make_creature(seed)
    path = f"/tmp/biota-smoke-eco/archive/{run_id}/archive.pkl"
    with open(path, "wb") as f:
        pickle.dump(arc, f)
    print(f"seeded cell {coords} in {run_id}")
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
  - name: hetero
    grid: 64
    steps: 20
    snapshot_every: 10
    border: torus
    output_format: gif
    spawn: {min_dist: 20, patch: 8, seed: 0}
    sources:
      - {run: seed-run,   cell: [5, 8, 3], n: 1}
      - {run: seed-run-b, cell: [3, 4, 5], n: 1}
YAMLEOF

# Build the CLI invocation. Transport selects Ray flag (or none); --workers
# only meaningful with Ray; --gpu-fraction passed only when explicitly set
# (let the CLI derive its default from device).
TRANSPORT_ARGS=()
case "$SMOKE_TRANSPORT" in
    noray)
        # Sequential dispatch. No --local-ray, no --ray-address, no --workers.
        ;;
    local)
        TRANSPORT_ARGS+=(--local-ray --workers "$SMOKE_WORKERS")
        ;;
    cluster)
        : "${SMOKE_RAY_ADDRESS:?SMOKE_RAY_ADDRESS required for cluster transport}"
        TRANSPORT_ARGS+=(--ray-address "$SMOKE_RAY_ADDRESS" --workers "$SMOKE_WORKERS")
        ;;
    *)
        echo "[$SMOKE_LABEL] unknown SMOKE_TRANSPORT=$SMOKE_TRANSPORT" >&2
        exit 2
        ;;
esac

GPU_ARGS=()
if [ -n "${SMOKE_GPU_FRACTION:-}" ]; then
    GPU_ARGS+=(--gpu-fraction "$SMOKE_GPU_FRACTION")
fi

# MPS needs the fallback flag for ops without MPS implementations.
ENV_ARGS=()
if [ "$SMOKE_DEVICE" = "mps" ]; then
    ENV_ARGS+=(env PYTORCH_ENABLE_MPS_FALLBACK=1)
fi

log "running 2-experiment ecosystem (device=$SMOKE_DEVICE, transport=$SMOKE_TRANSPORT, gpu_fraction=${SMOKE_GPU_FRACTION:-derived})..."
cd "$SMOKE_DATA" && ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "$SMOKE_VENV/bin/biota" ecosystem \
    --config experiments.yaml \
    --archive-dir archive \
    --output-dir ecosystem \
    --device "$SMOKE_DEVICE" \
    ${TRANSPORT_ARGS[@]+"${TRANSPORT_ARGS[@]}"} \
    ${GPU_ARGS[@]+"${GPU_ARGS[@]}"}

# All transports now write outputs driver-local.
test -d "$SMOKE_DATA/ecosystem"
log "runs produced:"
ls -1 "$SMOKE_DATA/ecosystem"
test "$(ls -1 "$SMOKE_DATA/ecosystem" | wc -l)" -eq 3

# Verify summary.json content for each run.
log "verifying summary.json fields..."
"$SMOKE_VENV/bin/python" - <<'PYEOF'
import json, sys
from pathlib import Path

eco = Path("/tmp/biota-smoke-eco/ecosystem")
runs = sorted(eco.iterdir())
assert len(runs) == 3, f"expected 3 runs, got {len(runs)}"

for run_dir in runs:
    summary = json.loads((run_dir / "summary.json").read_text())
    m = summary["measures"]
    name = summary["name"]

    # All runs must have the v3.0.0 keys present.
    assert "interaction_coefficients" in m, f"{name}: missing interaction_coefficients"
    assert "outcome_label" in m, f"{name}: missing outcome_label"

    if summary["mode"] == "homogeneous":
        assert m["interaction_coefficients"] == [], f"{name}: homo run should have empty coefficients"
        assert m["outcome_label"] == "", f"{name}: homo run should have empty outcome_label"
    else:
        # Heterogeneous: coefficient matrix must be S x S (S=2 here).
        ic = m["interaction_coefficients"]
        assert len(ic) == 2, f"{name}: expected 2x2 coefficient matrix, got {len(ic)} rows"
        assert all(len(row) == 2 for row in ic), f"{name}: coefficient rows wrong length"
        assert m["outcome_label"] in {"merger", "coexistence", "exclusion", "fragmentation"}, \
            f"{name}: unexpected outcome_label {m['outcome_label']!r}"
        print(f"  {name}: outcome={m['outcome_label']}, coefficients={ic}")

print("all summary.json checks passed")
PYEOF

log "done."
