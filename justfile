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

# Real-Ray smoke test. Runs an actual Ray-backed search with --local-ray
# to exercise code paths the default (no-ray) test suite cannot reach:
# ray.remote decoration, ObjectRef serialization, ray.wait, ray.get, and
# the full cross-worker round trip. Run this before tagging a release or
# after any change to ray_compat.py or cli.py's Ray wiring. Takes ~30s
# due to Ray's working_dir auto-packaging overhead (see DECISIONS.md).
smoke-ray:
    uv run biota search \
        --preset dev --budget 5 --random-phase 5 \
        --max-concurrent 2 --num-workers 2 \
        --local-ray \
        --grid 32 --steps 110

clean:
    rm -rf .pytest_cache .ruff_cache dist build
    find . -type d -name __pycache__ -exec rm -rf {} +
