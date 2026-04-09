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
# Use this before tagging a release or after any change to ray_compat.py,
# cli.py's Ray wiring, pyproject.toml dependencies, or anything that affects
# how biota imports or installs. The default test suite runs in no-Ray mode
# and cannot exercise Ray's @ray.remote decoration, ObjectRef serialization,
# ray.wait, ray.get, the cross-worker round trip, or the runtime-env code
# path - this recipe does.
#
# Why a wheel install instead of `uv run` from the source tree: see
# DECISIONS.md 2026-04-09 "A1 fix". TL;DR: `uv run` puts src/ on sys.path
# which makes Ray ship+rebuild biota on every worker (~7s overhead per run).
# A wheel install in a normal venv defeats that detection entirely.
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
    echo "[smoke-ray] running search through wheel-installed biota..."
    # cd /tmp so we are NOT in the source tree - guards against any tool
    # inadvertently re-adding src/ to sys.path
    cd /tmp && "$SMOKE_VENV/bin/biota" search \
        --preset dev --budget 20 --random-phase 5 \
        --max-concurrent 4 --num-workers 4 \
        --local-ray \
        --grid 32 --steps 110
    echo "[smoke-ray] cleaning up smoke venv..."
    rm -rf "$SMOKE_VENV"
    echo "[smoke-ray] done."

# Build a wheel from the current source and install it into ~/.biota-runtime/
# on the local node. Run this on every cluster node before bringing up Ray
# (and on the laptop if you want to run biota outside the source tree).
# Idempotent - re-running picks up any source changes via --force-reinstall.
# Pass --clean to nuke the runtime venv first.
cluster-install *ARGS:
    ./scripts/cluster_install.sh {{ARGS}}

clean:
    rm -rf .pytest_cache .ruff_cache dist build
    find . -type d -name __pycache__ -exec rm -rf {} +
