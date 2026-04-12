default:
    @just --list

sync:
    uv sync

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

typecheck:
    uv run pyright

check: lint format-check typecheck test

# Real-Ray smoke test, hermetic. Builds a wheel from the current source,
# installs it into a throwaway venv at /tmp/biota-smoke-venv, runs a small
# Ray-backed search through the wheel-installed binary, and tears the venv
# down. Hermetic on purpose: every run starts from a clean wheel install,
# so a stale venv can't paper over a packaging regression.
#
# Tests the full local-Ray code path: @ray.remote decoration, ObjectRef
# serialization, ray.wait, ray.get, and the cross-worker round trip.
# The default test suite runs in no-Ray mode and cannot exercise any of that.
#
# Equivalent GPU test: just smoke-ray-mps (MPS) or just smoke-ray-cuda (CUDA).
smoke-ray:
    #!/usr/bin/env bash
    set -euo pipefail
    SMOKE_VENV=/tmp/biota-smoke-venv
    echo "[smoke-ray] cleaning previous smoke venv (if any)..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray] building wheel..."
    rm -f dist/biota-*.whl
    uv build --wheel >/dev/null
    WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
    echo "[smoke-ray] wheel: $WHEEL"
    echo "[smoke-ray] creating smoke venv at $SMOKE_VENV..."
    uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
    echo "[smoke-ray] installing wheel into smoke venv..."
    uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"
    echo "[smoke-ray] running CPU local-Ray smoke search..."
    # cd /tmp so we are NOT in the source tree - guards against any tool
    # inadvertently re-adding src/ to sys.path
    cd /tmp && "$SMOKE_VENV/bin/biota" search \
        --preset dev --budget 20 --random-phase 5 \
        --batch-size 4 --workers 4 \
        --local-ray \
        --grid 32 --steps 110
    echo "[smoke-ray] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray] done."

# MPS variant of the smoke test. Same hermetic wheel setup; passes --device mps
# and a meaningful batch size to exercise the GPU code path on Apple Silicon.
smoke-ray-mps:
    #!/usr/bin/env bash
    set -euo pipefail
    SMOKE_VENV=/tmp/biota-smoke-venv
    echo "[smoke-ray-mps] cleaning previous smoke venv (if any)..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray-mps] building wheel..."
    rm -f dist/biota-*.whl
    uv build --wheel >/dev/null
    WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
    echo "[smoke-ray-mps] wheel: $WHEEL"
    uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
    uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"
    echo "[smoke-ray-mps] running MPS no-Ray smoke search (batch-size 16)..."
    cd /tmp && PYTORCH_ENABLE_MPS_FALLBACK=1 "$SMOKE_VENV/bin/biota" search \
        --preset dev --budget 20 --random-phase 5 \
        --device mps --batch-size 16 \
        --grid 32 --steps 110
    echo "[smoke-ray-mps] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray-mps] done."

# CUDA variant of the smoke test. Intended for the cluster nodes.
# Runs without Ray (no-Ray path exercises the batch engine directly).
# Pass RAY_ADDRESS to also test the cluster attach path.
smoke-ray-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    SMOKE_VENV=/tmp/biota-smoke-venv
    echo "[smoke-ray-cuda] cleaning previous smoke venv (if any)..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray-cuda] building wheel..."
    rm -f dist/biota-*.whl
    uv build --wheel >/dev/null
    WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
    echo "[smoke-ray-cuda] wheel: $WHEEL"
    uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
    uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"
    echo "[smoke-ray-cuda] running CUDA no-Ray smoke search (batch-size 16)..."
    cd /tmp && "$SMOKE_VENV/bin/biota" search \
        --preset dev --budget 20 --random-phase 5 \
        --device cuda --batch-size 16 \
        --grid 32 --steps 110
    echo "[smoke-ray-cuda] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray-cuda] done."

# CPU-only cluster smoke test. Runs without GPU so it works on any node
# regardless of whether the OCuLink GPU is powered on. Exercises the full
# cluster attach code path: Ray bringup, task dispatch, ObjectRef round trip.
# Run this first on the miniverse nodes to confirm the wheel, Ray wiring,
# and batch dispatch all work before switching the GPU on.
#
# Requires an already-running Ray cluster. Pass the head address:
#   just smoke-cluster 10.10.12.1:6379
smoke-cluster HEAD_ADDR:
    #!/usr/bin/env bash
    set -euo pipefail
    SMOKE_VENV=/tmp/biota-smoke-venv
    echo "[smoke-cluster] cleaning previous smoke venv (if any)..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-cluster] building wheel..."
    rm -f dist/biota-*.whl
    uv build --wheel >/dev/null
    WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
    echo "[smoke-cluster] wheel: $WHEEL"
    uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
    uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"
    echo "[smoke-cluster] running CPU cluster smoke search against {{HEAD_ADDR}}..."
    cd /tmp && "$SMOKE_VENV/bin/biota" search \
        --ray-address {{HEAD_ADDR}} \
        --preset dev --budget 30 --random-phase 10 \
        --device cpu --batch-size 2 --workers 3 \
        --grid 32 --steps 110
    echo "[smoke-cluster] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-cluster] done."

# GPU cluster smoke test. Same as smoke-cluster but with CUDA device and a
# real batch size. Run this after smoke-cluster confirms the no-GPU path works.
# Requires GPUs powered on and declared in the Ray cluster (--num-gpus=1 in
# ray start on each node).
#
#   just smoke-cluster-cuda 10.10.12.1:6379
smoke-cluster-cuda HEAD_ADDR:
    #!/usr/bin/env bash
    set -euo pipefail
    SMOKE_VENV=/tmp/biota-smoke-venv
    echo "[smoke-cluster-cuda] cleaning previous smoke venv (if any)..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-cluster-cuda] building wheel..."
    rm -f dist/biota-*.whl
    uv build --wheel >/dev/null
    WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
    echo "[smoke-cluster-cuda] wheel: $WHEEL"
    uv venv --python 3.12 "$SMOKE_VENV" >/dev/null
    uv pip install --python "$SMOKE_VENV/bin/python" --quiet "$WHEEL"
    echo "[smoke-cluster-cuda] running CUDA cluster smoke search against {{HEAD_ADDR}}..."
    cd /tmp && "$SMOKE_VENV/bin/biota" search \
        --ray-address {{HEAD_ADDR}} \
        --preset dev --budget 30 --random-phase 10 \
        --device cuda --batch-size 16 --workers 3 \
        --grid 32 --steps 110
    echo "[smoke-cluster-cuda] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-cluster-cuda] done."

# Before/after benchmark against the v0.3.0 baseline (341s for 500 rollouts
# standard preset on the cluster). Run post-v0.4.0 to record the actual
# speedup. Times the full biota search invocation and prints wall clock.
# Requires GPUs on and an already-running Ray cluster.
#
#   just benchmark 10.10.12.1:6379
benchmark HEAD_ADDR:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "[benchmark] standard preset, 500 rollouts, cuda B=64, workers=3"
    echo "[benchmark] cluster: {{HEAD_ADDR}}"
    echo "[benchmark] v0.3.0 baseline: ~341s"
    echo "[benchmark] starting..."
    START=$(date +%s)
    biota search \
        --ray-address {{HEAD_ADDR}} \
        --preset standard --budget 500 \
        --device cuda --batch-size 64 --workers 3
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "[benchmark] wall clock: ${ELAPSED}s (vs 341s baseline)"

# Build a wheel from the current source and install it into ~/.biota-runtime/
# on the local node. Run this on every cluster node before bringing up Ray.
# Idempotent - re-running picks up any source changes via --force-reinstall.
# Pass --clean to nuke the runtime venv first.
cluster-install *ARGS:
    ./scripts/cluster_install.sh {{ARGS}}

clean:
    rm -rf .pytest_cache .ruff_cache dist build
    find . -type d -name __pycache__ -exec rm -rf {} +
