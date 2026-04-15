"""Tests for the biota CLI.

Uses Typer's CliRunner to invoke commands in-process without spawning
subprocesses. Tests that run actual searches use the default no-ray mode
and tiny configs to keep runtime low.
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
    assert "biota.ecosystem" in output


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
        "--batch-size",
        "--workers",
        "--local-ray",
        "--ray-address",
        "--device",
        "--base-seed",
        "--checkpoint-every",
        "--output-dir",
        "--grid",
        "--steps",
        "--border",
    ):
        assert flag in output, f"missing flag {flag} in search --help output"


def test_search_unknown_preset_errors_cleanly() -> None:
    result = runner.invoke(app, ["search", "--preset", "nonexistent", "--budget", "1"])
    assert result.exit_code != 0


def test_search_local_ray_and_ray_address_mutually_exclusive() -> None:
    """Passing both --local-ray and --ray-address should error cleanly."""
    result = runner.invoke(
        app,
        [
            "search",
            "--local-ray",
            "--ray-address",
            "10.10.12.1",
            "--budget",
            "1",
        ],
    )
    assert result.exit_code != 0
    # Typer's BadParameter writes to the click error stream, which shows up
    # in result.output (not result.stdout).
    output = _strip_ansi(result.output)
    assert "mutually exclusive" in output.lower()


# === search end-to-end (small, default no-ray) ===


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
            "--grid",
            "32",
            "--steps",
            "110",
            "--output-dir",
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
            "--grid",
            "32",
            "--steps",
            "110",
            "--output-dir",
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
            "--grid",
            "32",
            "--steps",
            "110",
            "--output-dir",
            str(tmp_path),
        ],
    )
    run_dir = next(tmp_path.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    assert config["rollout"]["sim"]["grid_h"] == 32
    assert config["rollout"]["sim"]["grid_w"] == 32
    assert config["rollout"]["steps"] == 110


# === ecosystem --help ===


def test_ecosystem_help_lists_infra_flags_only() -> None:
    result = runner.invoke(app, ["ecosystem", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.stdout)
    # Infrastructure flags that should be present.
    for flag in (
        "--config",
        "--archive-dir",
        "--output-dir",
        "--device",
        "--local-ray",
        "--ray-address",
        "--workers",
        "--gpu-fraction",
    ):
        assert flag in output, f"missing flag {flag} in ecosystem --help output"
    # Old inline-flag interface is gone.
    for flag in ("--run", "--cell", "--n", "--snapshot-every", "--patch", "--min-dist"):
        assert flag not in output, f"old inline flag {flag} unexpectedly still present"


# === ecosystem error cases ===


def test_ecosystem_missing_config_errors_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["ecosystem", "--config", str(tmp_path / "no-such-file.yaml")],
    )
    assert result.exit_code != 0
    assert "config file not found" in _strip_ansi(result.output).lower()


def test_ecosystem_bad_yaml_references_parser_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiments:\n  - name: [unclosed\n")
    result = runner.invoke(app, ["ecosystem", "--config", str(bad)])
    assert result.exit_code != 0
    assert "yaml" in _strip_ansi(result.output).lower()


def test_ecosystem_missing_name_surfaces_error(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "experiments:\n"
        "  - grid: 64\n"
        "    steps: 5\n"
        "    snapshot_every: 5\n"
        "    border: wall\n"
        "    output_format: gif\n"
        "    spawn: {min_dist: 20, patch: 8, seed: 0}\n"
        "    sources: [{run: r, cell: [0, 0, 0], n: 1}]\n"
    )
    result = runner.invoke(app, ["ecosystem", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "name" in _strip_ansi(result.output).lower()


def _minimal_config(path: Path) -> Path:
    """Write a minimally-valid ecosystem config to disk and return its path.

    Used by tests that exercise CLI flag validation; the config has to parse
    cleanly so the validation runs after parsing reaches the body.
    """
    cfg = path / "ecosystem.yaml"
    cfg.write_text(
        "experiments:\n"
        "  - name: x\n"
        "    grid: 64\n"
        "    steps: 5\n"
        "    snapshot_every: 5\n"
        "    border: wall\n"
        "    output_format: gif\n"
        "    spawn: {min_dist: 20, patch: 8, seed: 0}\n"
        "    sources: [{run: r, cell: [0, 0, 0], n: 1}]\n"
    )
    return cfg


def test_ecosystem_rejects_cuda_with_zero_gpu_fraction(tmp_path: Path) -> None:
    """The contradictory combo: cuda but no GPU reservation. Must refuse."""
    cfg = _minimal_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "ecosystem",
            "--config",
            str(cfg),
            "--device",
            "cuda",
            "--local-ray",
            "--gpu-fraction",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "contradictory" in _strip_ansi(result.output).lower()


def test_ecosystem_rejects_negative_gpu_fraction(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "ecosystem",
            "--config",
            str(cfg),
            "--local-ray",
            "--gpu-fraction",
            "-0.5",
        ],
    )
    assert result.exit_code != 0
    assert ">= 0" in _strip_ansi(result.output)


def test_ecosystem_rejects_local_ray_and_ray_address_together(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "ecosystem",
            "--config",
            str(cfg),
            "--local-ray",
            "--ray-address",
            "localhost:6379",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in _strip_ansi(result.output).lower()


def test_ecosystem_warns_on_gpu_fraction_with_cpu_device(tmp_path: Path) -> None:
    """Non-zero gpu-fraction with --device cpu should print a warning but
    not fail; user might intentionally want this for cluster scheduling
    reasons we cannot predict.

    We pair the warning trigger with --workers 0 so that the workers
    validation rejects after the warning prints, exiting fast without
    actually starting Ray. The warning fires before any dispatch.
    """
    cfg = _minimal_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "ecosystem",
            "--config",
            str(cfg),
            "--device",
            "cpu",
            "--local-ray",
            "--gpu-fraction",
            "0.5",
            "--workers",
            "0",  # forces an early exit after the warning fires
        ],
    )
    output = _strip_ansi(result.output).lower()
    # Warning must appear regardless of whether Ray init succeeded.
    assert "reserves gpus" in output, f"missing GPU-waste warning in: {result.output!r}"
    # And the workers=0 validation should be the actual exit cause.
    assert result.exit_code != 0
    assert "workers" in output


def test_ecosystem_end_to_end(tmp_path: Path) -> None:
    """Full config-driven ecosystem run from a search-produced archive."""
    import pickle

    import numpy as np

    from biota.search.archive import Archive
    from biota.search.result import RolloutResult

    # Build a minimal archive with one occupied cell
    archive_dir = tmp_path / "archive" / "test-run"
    archive_dir.mkdir(parents=True)
    archive = Archive(descriptor_names=("velocity", "gyradius", "spectral_entropy"))
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
    archive._cells[(5, 8, 3)] = creature  # type: ignore[reportPrivateUsage]
    with open(archive_dir / "archive.pkl", "wb") as f:
        pickle.dump(archive, f)

    # Write a valid ecosystem config pointing at that archive
    ecosystem_dir = tmp_path / "ecosystem"
    cfg_path = tmp_path / "experiments.yaml"
    cfg_path.write_text(
        "experiments:\n"
        "  - name: smoke-run\n"
        "    grid: 64\n"
        "    steps: 5\n"
        "    snapshot_every: 5\n"
        "    border: wall\n"
        "    output_format: frames\n"
        "    spawn:\n"
        "      min_dist: 20\n"
        "      patch: 8\n"
        "      seed: 0\n"
        "    sources:\n"
        "      - run: test-run\n"
        "        cell: [5, 8, 3]\n"
        "        n: 2\n"
    )

    result = runner.invoke(
        app,
        [
            "ecosystem",
            "--config",
            str(cfg_path),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--output-dir",
            str(ecosystem_dir),
        ],
    )
    assert result.exit_code == 0, f"ecosystem CLI failed: {result.output}"
    assert "output=" in result.stdout
    # Verify output directory was created with expected files
    eco_runs = list(ecosystem_dir.iterdir())
    assert len(eco_runs) == 1
    run_dir = eco_runs[0]
    assert run_dir.name.endswith("-smoke-run"), (
        f"expected run dir to end with experiment name, got {run_dir.name}"
    )
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "trajectory.npy").exists()
