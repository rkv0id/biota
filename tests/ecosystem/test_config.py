"""Tests for the ecosystem YAML config parser.

Every assertion here targets a real failure mode: malformed fields, wrong
types, missing required keys, duplicate names. The parser errors must
reference the experiment name so users can find the offending entry in a
multi-experiment config.
"""

import tempfile
import textwrap
from pathlib import Path

import pytest

from biota.ecosystem.config import (
    ConfigError,
    CreatureSource,
    EcosystemConfig,
    SpawnConfig,
    load_config_file,
)


def _write(text: str) -> Path:
    """Write a YAML string to a temp file and return its Path.

    delete=False because the caller reads the file after this function
    returns; the caller cleans up via Path.unlink(missing_ok=True).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        return Path(tmp.name)


def _load(
    text: str, archive_dir: str = "archive", device: str = "cpu"
) -> tuple[EcosystemConfig, ...]:
    path = _write(text)
    try:
        return load_config_file(path, default_archive_dir=archive_dir, default_device=device)
    finally:
        path.unlink(missing_ok=True)


# === happy path ===


def test_parses_minimal_single_experiment() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: solo
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn:
              min_dist: 20
              patch: 8
              seed: 0
            sources:
              - run: test-run
                cell: [1, 2, 3]
                n: 4
    """)
    experiments = _load(yaml_text)
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.name == "solo"
    assert exp.grid_h == 64 and exp.grid_w == 64
    assert exp.steps == 10
    assert exp.snapshot_every == 5
    assert exp.border == "wall"
    assert exp.output_format == "gif"
    assert exp.spawn == SpawnConfig(min_dist=20, patch=8, seed=0)
    assert exp.sources == (
        CreatureSource(archive_dir="archive", run_id="test-run", coords=(1, 2, 3), n=4),
    )
    assert exp.device == "cpu"
    assert exp.is_heterogeneous is False


def test_parses_rectangular_grid_list() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: rect
            grid: [192, 512]
            steps: 10
            snapshot_every: 5
            border: torus
            output_format: gif
            spawn: {min_dist: 20, patch: 8, seed: 0}
            sources:
              - {run: r, cell: [0, 0, 0], n: 1}
    """)
    exp = _load(yaml_text)[0]
    assert exp.grid_h == 192 and exp.grid_w == 512


def test_heterogeneous_flag() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: het
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 8, seed: 0}
            sources:
              - {run: a, cell: [0, 0, 0], n: 2}
              - {run: b, cell: [1, 1, 1], n: 2}
    """)
    exp = _load(yaml_text)[0]
    assert exp.is_heterogeneous is True
    assert len(exp.sources) == 2


def test_source_archive_dir_override() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: mixed
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 8, seed: 0}
            sources:
              - {run: a, cell: [0, 0, 0], n: 2}
              - archive_dir: other-archive
                run: b
                cell: [1, 1, 1]
                n: 2
    """)
    exp = _load(yaml_text, archive_dir="default-archive")[0]
    assert exp.sources[0].archive_dir == "default-archive"
    assert exp.sources[1].archive_dir == "other-archive"


def test_source_patch_override_parsed() -> None:
    """Per-source patch override is parsed when present, None when absent."""
    yaml_text = textwrap.dedent("""
        experiments:
          - name: mixed-patches
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 32, seed: 0}
            sources:
              - {run: a, cell: [0, 0, 0], n: 2}
              - {run: b, cell: [1, 1, 1], n: 2, patch: 80}
    """)
    exp = _load(yaml_text)[0]
    # Source 0 has no override; the parser leaves patch as None and the runner
    # falls back to spawn.patch at execution time.
    assert exp.sources[0].patch is None
    # Source 1 explicitly overrides.
    assert exp.sources[1].patch == 80
    # Sanity: spawn-level default is preserved.
    assert exp.spawn.patch == 32


def test_source_patch_must_be_positive() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: bad-patch
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 32, seed: 0}
            sources:
              - {run: a, cell: [0, 0, 0], n: 2, patch: 0}
    """)
    with pytest.raises(ConfigError, match=r"'sources\[0\]\.patch' must be positive"):
        _load(yaml_text)


def test_source_patch_must_be_integer() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: bad-patch-type
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 32, seed: 0}
            sources:
              - {run: a, cell: [0, 0, 0], n: 2, patch: 64.5}
    """)
    with pytest.raises(ConfigError, match=r"'sources\[0\]\.patch' must be an integer"):
        _load(yaml_text)


def test_multiple_experiments_sequential() -> None:
    yaml_text = textwrap.dedent("""
        experiments:
          - name: first
            grid: 64
            steps: 10
            snapshot_every: 5
            border: wall
            output_format: gif
            spawn: {min_dist: 20, patch: 8, seed: 0}
            sources: [{run: r, cell: [0, 0, 0], n: 1}]
          - name: second
            grid: 128
            steps: 20
            snapshot_every: 10
            border: torus
            output_format: frames
            spawn: {min_dist: 30, patch: 10, seed: 42}
            sources: [{run: s, cell: [1, 1, 1], n: 2}]
    """)
    experiments = _load(yaml_text)
    assert [e.name for e in experiments] == ["first", "second"]
    assert experiments[0].grid_w == 64
    assert experiments[1].grid_w == 128


# === failure cases ===


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config_file(
            Path("/tmp/does-not-exist-biota.yaml"),
            default_archive_dir="archive",
            default_device="cpu",
        )


def test_invalid_yaml_raises() -> None:
    with pytest.raises(ConfigError, match="YAML parse error"):
        _load("experiments:\n  - name: [unclosed list\n")


def test_top_level_not_mapping() -> None:
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        _load("- just a list\n- no experiments key\n")


def test_missing_experiments_key() -> None:
    with pytest.raises(ConfigError, match="experiments"):
        _load("something_else: true\n")


def test_empty_experiments_list() -> None:
    with pytest.raises(ConfigError, match="need at least one"):
        _load("experiments: []\n")


def test_missing_name_refers_to_index() -> None:
    with pytest.raises(ConfigError, match=r"experiments\[0\].*'name'"):
        _load(
            textwrap.dedent("""
            experiments:
              - grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_empty_name_rejected() -> None:
    with pytest.raises(ConfigError, match="missing or empty 'name'"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: ""
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_missing_required_key_references_experiment_name() -> None:
    # The name 'needs-steps' should appear in the error.
    with pytest.raises(ConfigError, match=r"'needs-steps'.*'steps'"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: needs-steps
                grid: 64
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_snapshot_every_larger_than_steps() -> None:
    with pytest.raises(ConfigError, match=r"snapshot_every.*cannot exceed steps"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: bad
                grid: 64
                steps: 5
                snapshot_every: 10
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_bad_border_value() -> None:
    with pytest.raises(ConfigError, match="'border' must be one of"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: bad-border
                grid: 64
                steps: 10
                snapshot_every: 5
                border: reflective
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_bad_output_format() -> None:
    with pytest.raises(ConfigError, match="'output_format' must be one of"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: bad-out
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: mp4
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_bad_grid_wrong_list_length() -> None:
    with pytest.raises(ConfigError, match="'grid' as a list must have exactly 2 entries"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: bad-grid
                grid: [64, 64, 64]
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_bad_grid_negative() -> None:
    with pytest.raises(ConfigError, match=r"'grid'.*positive"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: neg
                grid: -1
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0], n: 1}]
        """)
        )


def test_zero_sources_rejected() -> None:
    with pytest.raises(ConfigError, match="'sources' is empty"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: empty-sources
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: []
        """)
        )


def test_source_missing_n() -> None:
    with pytest.raises(ConfigError, match=r"sources\[0\]\.n is required"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: nope
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0, 0]}]
        """)
        )


def test_source_cell_wrong_length() -> None:
    with pytest.raises(ConfigError, match="cell must have exactly 3"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: nope
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, 0], n: 1}]
        """)
        )


def test_source_cell_negative_rejected() -> None:
    with pytest.raises(ConfigError, match="cell entries must be non-negative"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: nope
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: r, cell: [0, -1, 0], n: 1}]
        """)
        )


def test_duplicate_experiment_name_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate experiment name"):
        _load(
            textwrap.dedent("""
            experiments:
              - name: twin
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: a, cell: [0, 0, 0], n: 1}]
              - name: twin
                grid: 64
                steps: 10
                snapshot_every: 5
                border: wall
                output_format: gif
                spawn: {min_dist: 20, patch: 8, seed: 0}
                sources: [{run: b, cell: [1, 1, 1], n: 1}]
        """)
        )


def test_bad_device_rejected() -> None:
    with pytest.raises(ConfigError, match="device 'tpu'"):
        _load(
            textwrap.dedent("""
                experiments:
                  - name: x
                    grid: 64
                    steps: 10
                    snapshot_every: 5
                    border: wall
                    output_format: gif
                    spawn: {min_dist: 20, patch: 8, seed: 0}
                    sources: [{run: r, cell: [0, 0, 0], n: 1}]
            """),
            device="tpu",
        )
