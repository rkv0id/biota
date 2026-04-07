"""Tests for the biota CLI.

Uses Typer's CliRunner to invoke commands in-process without spawning
subprocesses. All tests use --no-ray and tiny configs to keep runtime
low.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biota.cli import app
from biota.ray_compat import shutdown

runner = CliRunner()

# Rich-rendered help output wraps each `--` prefix in its own ANSI color
# sequence, splitting flag names like "--preset" into "-" + reset + "-preset"
# at the byte level. CI runs in color mode (TTY detected), local runs
# typically don't. Strip ANSI codes before substring assertions so the test
# behaves the same in both environments.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def clean_runtime() -> Iterator[None]:
    shutdown()
    yield
    shutdown()


# === doctor ===


def test_doctor_exits_zero() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_doctor_output_mentions_key_components() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    output = result.stdout
    assert "biota" in output
    assert "python" in output
    assert "torch" in output
    assert "biota.search" in output
    assert "biota.ray_compat" in output


def test_doctor_reports_torch_device_availability() -> None:
    result = runner.invoke(app, ["doctor"])
    assert "cpu:" in result.stdout
    assert "mps:" in result.stdout
    assert "cuda:" in result.stdout


# === search --help ===


def test_search_help_lists_all_flags() -> None:
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.stdout)
    for flag in (
        "--preset",
        "--budget",
        "--random-phase",
        "--max-concurrent",
        "--no-ray",
        "--num-workers",
        "--device",
        "--base-seed",
        "--checkpoint-every",
        "--runs-root",
        "--grid",
        "--steps",
    ):
        assert flag in output, f"missing flag {flag} in search --help output"


def test_search_unknown_preset_errors_cleanly() -> None:
    result = runner.invoke(app, ["search", "--preset", "nonexistent", "--no-ray", "--budget", "1"])
    assert result.exit_code != 0


# === search end-to-end (small, --no-ray) ===


def test_search_runs_to_completion(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "--preset",
            "dev",
            "--budget",
            "5",
            "--random-phase",
            "5",
            "--no-ray",
            "--grid",
            "32",
            "--steps",
            "110",
            "--runs-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    # Final stdout line is the scriptable archive_size summary
    assert "archive_size=" in result.stdout


def test_search_creates_run_dir(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "search",
            "--preset",
            "dev",
            "--budget",
            "5",
            "--random-phase",
            "5",
            "--no-ray",
            "--grid",
            "32",
            "--steps",
            "110",
            "--runs-root",
            str(tmp_path),
        ],
    )
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "archive.pkl").exists()
    assert (run_dir / "events.jsonl").exists()


def test_search_grid_steps_override_applies(tmp_path: Path) -> None:
    """--grid and --steps should override the preset's simulation config."""
    import json

    runner.invoke(
        app,
        [
            "search",
            "--preset",
            "dev",  # dev = 96x96/200
            "--budget",
            "3",
            "--random-phase",
            "3",
            "--no-ray",
            "--grid",
            "32",
            "--steps",
            "110",
            "--runs-root",
            str(tmp_path),
        ],
    )
    run_dir = next(tmp_path.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    assert config["rollout"]["sim"]["grid"] == 32
    assert config["rollout"]["steps"] == 110
