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

clean:
    rm -rf .pytest_cache .ruff_cache dist build
    find . -type d -name __pycache__ -exec rm -rf {} +
