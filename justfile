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

# === smoke tests ===
#
# Three transports x three devices x two commands. Each leaf recipe is a
# 5-line wrapper around scripts/smoke_search.sh or scripts/smoke_ecosystem.sh,
# both hermetic (build wheel, install in throwaway venv, run binary, verify,
# clean up). Naming: smoke-<transport>-<device>-<command>, plus a
# smoke-<transport>-<device> umbrella that runs both commands.
#
# Transports:
#   noray     sequential, no Ray involvement (catches packaging regressions
#             and exercises the device-specific code paths in pytest's blind
#             spot since CI runs CPU-only)
#   local     fresh single-host Ray instance (catches @ray.remote decoration,
#             ObjectRef serialization, ray.wait/ray.get round trip)
#   cluster   attach to an already-running Ray cluster (catches cross-node
#             EcosystemConfig serialization, cluster bringup wiring)
#
# Devices:
#   cpu       always available
#   mps       Apple Silicon dev machines
#   cuda      cluster nodes with GPUs powered on
#
# cluster + mps is omitted because Linux nodes don't have MPS.
#
# Top-level smoke-ray runs the local-machine CPU-only sanity (no GPU
# required, no cluster attach required). Run individual cells for GPU or
# cluster coverage.

# --- noray (sequential, no Ray) ---

smoke-noray-cpu-search:
    SMOKE_LABEL=smoke-noray-cpu-search \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_search.sh

smoke-noray-cpu-signal-search:
    SMOKE_LABEL=smoke-noray-cpu-signal-search \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=noray \
    SMOKE_SIGNAL_FIELD=1 \
    ./scripts/smoke_search.sh

smoke-noray-cpu-ecosystem:
    SMOKE_LABEL=smoke-noray-cpu-ecosystem \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_ecosystem.sh

smoke-noray-cpu: smoke-noray-cpu-search smoke-noray-cpu-signal-search smoke-noray-cpu-ecosystem

smoke-noray-mps-search:
    SMOKE_LABEL=smoke-noray-mps-search \
    SMOKE_DEVICE=mps \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_search.sh

smoke-noray-mps-ecosystem:
    SMOKE_LABEL=smoke-noray-mps-ecosystem \
    SMOKE_DEVICE=mps \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_ecosystem.sh

smoke-noray-mps: smoke-noray-mps-search smoke-noray-mps-ecosystem

smoke-noray-cuda-search:
    SMOKE_LABEL=smoke-noray-cuda-search \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_search.sh

smoke-noray-cuda-ecosystem:
    SMOKE_LABEL=smoke-noray-cuda-ecosystem \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=noray \
    ./scripts/smoke_ecosystem.sh

smoke-noray-cuda: smoke-noray-cuda-search smoke-noray-cuda-ecosystem

# --- local Ray (single-host, fresh ray.init) ---

smoke-local-cpu-search:
    SMOKE_LABEL=smoke-local-cpu-search \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_search.sh

smoke-local-cpu-ecosystem:
    SMOKE_LABEL=smoke-local-cpu-ecosystem \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_ecosystem.sh

smoke-local-cpu: smoke-local-cpu-search smoke-local-cpu-ecosystem

smoke-local-mps-search:
    SMOKE_LABEL=smoke-local-mps-search \
    SMOKE_DEVICE=mps \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_search.sh

smoke-local-mps-ecosystem:
    SMOKE_LABEL=smoke-local-mps-ecosystem \
    SMOKE_DEVICE=mps \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_ecosystem.sh

smoke-local-mps: smoke-local-mps-search smoke-local-mps-ecosystem

smoke-local-cuda-search:
    SMOKE_LABEL=smoke-local-cuda-search \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_search.sh

smoke-local-cuda-ecosystem:
    SMOKE_LABEL=smoke-local-cuda-ecosystem \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=local \
    ./scripts/smoke_ecosystem.sh

smoke-local-cuda: smoke-local-cuda-search smoke-local-cuda-ecosystem

# --- cluster Ray (attach to already-running cluster) ---
#
# Each takes HEAD_ADDR positional, e.g. just smoke-cluster-cpu 10.10.12.1:6379

smoke-cluster-cpu-search HEAD_ADDR:
    SMOKE_LABEL=smoke-cluster-cpu-search \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=cluster \
    SMOKE_RAY_ADDRESS={{HEAD_ADDR}} \
    ./scripts/smoke_search.sh

smoke-cluster-cpu-ecosystem HEAD_ADDR:
    SMOKE_LABEL=smoke-cluster-cpu-ecosystem \
    SMOKE_DEVICE=cpu \
    SMOKE_TRANSPORT=cluster \
    SMOKE_RAY_ADDRESS={{HEAD_ADDR}} \
    ./scripts/smoke_ecosystem.sh

smoke-cluster-cpu HEAD_ADDR: (smoke-cluster-cpu-search HEAD_ADDR) (smoke-cluster-cpu-ecosystem HEAD_ADDR)

smoke-cluster-cuda-search HEAD_ADDR:
    SMOKE_LABEL=smoke-cluster-cuda-search \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=cluster \
    SMOKE_RAY_ADDRESS={{HEAD_ADDR}} \
    ./scripts/smoke_search.sh

smoke-cluster-cuda-ecosystem HEAD_ADDR:
    SMOKE_LABEL=smoke-cluster-cuda-ecosystem \
    SMOKE_DEVICE=cuda \
    SMOKE_TRANSPORT=cluster \
    SMOKE_RAY_ADDRESS={{HEAD_ADDR}} \
    ./scripts/smoke_ecosystem.sh

smoke-cluster-cuda HEAD_ADDR: (smoke-cluster-cuda-search HEAD_ADDR) (smoke-cluster-cuda-ecosystem HEAD_ADDR)

# Top-level umbrella: CPU-only sanity verifiable on any machine. Runs noray
# and local-Ray on CPU for both search and ecosystem. For GPU or cluster
# coverage, run the specific cells (e.g. smoke-local-cuda or
# smoke-cluster-cuda HEAD_ADDR) explicitly.
smoke-ray: smoke-noray-cpu smoke-local-cpu

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
