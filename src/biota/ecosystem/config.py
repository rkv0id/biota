"""Declarative config for ecosystem experiments.

Ecosystem runs are driven by a YAML file that defines one or more experiments.
The CLI only carries infrastructure flags (device, local-ray, I/O dirs). Every
experiment-level knob lives in the config.

This module provides the data model and the YAML parser. It does not touch the
filesystem beyond reading the config file itself: verification that archive
runs exist, that cells are occupied, and that params are valid is the runner's
concern since the runner loads the archives anyway.

The parser errors loudly at the first problem it finds, pointing at the
offending experiment by its declared name. There are no silent defaults; every
required field must be set explicitly. This is deliberate, to keep ecosystem
configs fully self-describing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

_VALID_BORDERS = ("wall", "torus")
_VALID_OUTPUT_FORMATS = ("gif", "frames")
_VALID_DEVICES = ("cpu", "mps", "cuda")


class ConfigError(ValueError):
    """Raised when an ecosystem config file is malformed.

    Carries the offending experiment name (when known) so callers can surface
    a precise message. Raised from load_config_file.
    """


@dataclass(frozen=True)
class CreatureSource:
    """One creature type placed into an ecosystem experiment.

    Attributes:
        archive_dir: Directory containing the source archive run. Optional in
            the YAML, inherits the global default when unset.
        run_id: Archive run directory name under archive_dir.
        coords: (y, x, z) archive cell coordinate.
        n: Number of copies of this creature to spawn in the ecosystem.
        patch: Optional per-source override for initial patch side length. When
            None, falls back to the experiment's spawn.patch. Useful when one
            species in a heterogeneous run has a substantially different
            natural scale than the others (e.g. a small fast glider mixed with
            a large dense colony).
    """

    archive_dir: str
    run_id: str
    coords: tuple[int, int, int]
    n: int
    patch: int | None = None


@dataclass(frozen=True)
class SpawnConfig:
    """Per-experiment spawn parameters shared across all sources.

    n (creature count) has moved to CreatureSource in v2.2.0 so different
    sources can place different numbers of copies.
    """

    min_dist: int
    patch: int
    seed: int


@dataclass(frozen=True)
class EcosystemConfig:
    """Complete specification of one ecosystem experiment.

    One source in `sources` triggers the existing homogeneous code path
    (global scalar params, unchanged from v2.0.0). Two or more sources triggers
    the heterogeneous code path with species-indexed parameter localization.
    """

    name: str
    sources: tuple[CreatureSource, ...]
    grid_h: int
    grid_w: int
    steps: int
    snapshot_every: int
    spawn: SpawnConfig
    device: str
    border: str
    output_format: str

    @property
    def is_heterogeneous(self) -> bool:
        return len(self.sources) >= 2


def load_config_file(
    path: Path,
    default_archive_dir: str,
    default_device: str,
) -> tuple[EcosystemConfig, ...]:
    """Parse and validate an ecosystem config YAML.

    default_archive_dir is applied to any source that doesn't specify one.
    default_device is applied to every experiment (experiments do not carry
    their own device; infrastructure belongs on the CLI).

    Raises ConfigError on any malformed field, including the experiment name
    when available.
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    raw_dict = cast(dict[str, Any], raw)

    experiments_raw = raw_dict.get("experiments")
    if experiments_raw is None:
        raise ConfigError(f"{path}: missing required top-level key 'experiments'")
    if not isinstance(experiments_raw, list):
        raise ConfigError(
            f"{path}: 'experiments' must be a list, got {type(experiments_raw).__name__}"
        )
    experiments_list = cast(list[Any], experiments_raw)
    if not experiments_list:
        raise ConfigError(f"{path}: 'experiments' list is empty; need at least one")

    parsed: list[EcosystemConfig] = []
    names_seen: set[str] = set()
    for i, raw_exp in enumerate(experiments_list):
        if not isinstance(raw_exp, dict):
            raise ConfigError(f"{path}: experiments[{i}] must be a mapping")
        exp_dict = cast(dict[str, Any], raw_exp)
        cfg = _parse_experiment(exp_dict, i, default_archive_dir, default_device)
        if cfg.name in names_seen:
            raise ConfigError(
                f"duplicate experiment name {cfg.name!r}; names must be unique within a config"
            )
        names_seen.add(cfg.name)
        parsed.append(cfg)

    return tuple(parsed)


def _parse_experiment(
    raw: dict[str, Any],
    index: int,
    default_archive_dir: str,
    default_device: str,
) -> EcosystemConfig:
    """Validate and build one EcosystemConfig from a YAML mapping."""
    # Name first: every subsequent error message references it.
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(
            f"experiments[{index}]: missing or empty 'name' "
            f"(names are required, no auto-generation)"
        )

    def require(key: str) -> Any:  # pyright: ignore[reportAny]
        if key not in raw:
            raise ConfigError(f"experiment {name!r}: missing required key {key!r}")
        return raw[key]  # pyright: ignore[reportAny]

    grid_h, grid_w = _parse_grid(require("grid"), name)
    steps = _parse_positive_int(require("steps"), "steps", name)
    snapshot_every = _parse_positive_int(require("snapshot_every"), "snapshot_every", name)
    if snapshot_every > steps:
        raise ConfigError(
            f"experiment {name!r}: snapshot_every ({snapshot_every}) cannot exceed steps ({steps})"
        )

    border = require("border")
    if border not in _VALID_BORDERS:
        raise ConfigError(
            f"experiment {name!r}: 'border' must be one of {_VALID_BORDERS}, got {border!r}"
        )

    output_format = require("output_format")
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ConfigError(
            f"experiment {name!r}: 'output_format' must be one of "
            f"{_VALID_OUTPUT_FORMATS}, got {output_format!r}"
        )

    spawn = _parse_spawn(require("spawn"), name)
    sources = _parse_sources(require("sources"), name, default_archive_dir)

    if default_device not in _VALID_DEVICES:
        raise ConfigError(
            f"device {default_device!r} not in {_VALID_DEVICES} (set via --device on the CLI)"
        )

    return EcosystemConfig(
        name=name,
        sources=sources,
        grid_h=grid_h,
        grid_w=grid_w,
        steps=steps,
        snapshot_every=snapshot_every,
        spawn=spawn,
        device=default_device,
        border=border,
        output_format=output_format,
    )


def _parse_grid(raw: Any, name: str) -> tuple[int, int]:  # pyright: ignore[reportAny]
    """Accept either an int (square grid) or a [H, W] list."""
    if isinstance(raw, int):
        if raw <= 0:
            raise ConfigError(f"experiment {name!r}: 'grid' must be positive, got {raw}")
        return raw, raw
    if isinstance(raw, list):
        raw_list = cast(list[Any], raw)
        if len(raw_list) != 2:
            raise ConfigError(
                f"experiment {name!r}: 'grid' as a list must have exactly 2 entries "
                f"[H, W], got {len(raw_list)}"
            )
        h, w = raw_list
        if not isinstance(h, int) or not isinstance(w, int):
            raise ConfigError(
                f"experiment {name!r}: 'grid' list entries must be integers, got {raw_list}"
            )
        if h <= 0 or w <= 0:
            raise ConfigError(
                f"experiment {name!r}: 'grid' entries must be positive, got {raw_list}"
            )
        return h, w
    raise ConfigError(
        f"experiment {name!r}: 'grid' must be an int or [H, W] list, got {type(raw).__name__}"
    )


def _parse_positive_int(raw: Any, field: str, name: str) -> int:  # pyright: ignore[reportAny]
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ConfigError(
            f"experiment {name!r}: {field!r} must be an integer, got {type(raw).__name__}"
        )
    if raw <= 0:
        raise ConfigError(f"experiment {name!r}: {field!r} must be positive, got {raw}")
    return raw


def _parse_spawn(raw: Any, name: str) -> SpawnConfig:  # pyright: ignore[reportAny]
    if not isinstance(raw, dict):
        raise ConfigError(
            f"experiment {name!r}: 'spawn' must be a mapping, got {type(raw).__name__}"
        )
    spawn_dict = cast(dict[str, Any], raw)
    for key in ("min_dist", "patch", "seed"):
        if key not in spawn_dict:
            raise ConfigError(f"experiment {name!r}: spawn.{key} is required")

    min_dist = _parse_positive_int(spawn_dict["min_dist"], "spawn.min_dist", name)
    patch = _parse_positive_int(spawn_dict["patch"], "spawn.patch", name)
    seed = spawn_dict["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"experiment {name!r}: spawn.seed must be an integer, got {type(seed).__name__}"
        )
    return SpawnConfig(min_dist=min_dist, patch=patch, seed=seed)


def _parse_sources(
    raw: Any,  # pyright: ignore[reportAny]
    name: str,
    default_archive_dir: str,
) -> tuple[CreatureSource, ...]:
    if not isinstance(raw, list):
        raise ConfigError(
            f"experiment {name!r}: 'sources' must be a list, got {type(raw).__name__}"
        )
    raw_list = cast(list[Any], raw)
    if not raw_list:
        raise ConfigError(f"experiment {name!r}: 'sources' is empty; need at least one")

    out: list[CreatureSource] = []
    for i, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"experiment {name!r}: sources[{i}] must be a mapping, got {type(entry).__name__}"
            )
        entry_dict = cast(dict[str, Any], entry)
        out.append(_parse_one_source(entry_dict, i, name, default_archive_dir))
    return tuple(out)


def _parse_one_source(
    raw: dict[str, Any],
    index: int,
    name: str,
    default_archive_dir: str,
) -> CreatureSource:
    for key in ("run", "cell", "n"):
        if key not in raw:
            raise ConfigError(f"experiment {name!r}: sources[{index}].{key} is required")

    run_id = raw["run"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ConfigError(f"experiment {name!r}: sources[{index}].run must be a non-empty string")

    archive_dir = raw.get("archive_dir", default_archive_dir)
    if not isinstance(archive_dir, str) or not archive_dir.strip():
        raise ConfigError(
            f"experiment {name!r}: sources[{index}].archive_dir must be a non-empty string"
        )

    coords = _parse_coords(raw["cell"], index, name)
    n = _parse_positive_int(raw["n"], f"sources[{index}].n", name)

    # Optional per-source patch override.
    patch_raw = raw.get("patch")
    patch: int | None
    if patch_raw is None:
        patch = None
    else:
        patch = _parse_positive_int(patch_raw, f"sources[{index}].patch", name)

    return CreatureSource(archive_dir=archive_dir, run_id=run_id, coords=coords, n=n, patch=patch)


def _parse_coords(raw: Any, index: int, name: str) -> tuple[int, int, int]:  # pyright: ignore[reportAny]
    if not isinstance(raw, list):
        raise ConfigError(
            f"experiment {name!r}: sources[{index}].cell must be a list of 3 integers, "
            f"got {type(raw).__name__}"
        )
    raw_list = cast(list[Any], raw)
    if len(raw_list) != 3:
        raise ConfigError(
            f"experiment {name!r}: sources[{index}].cell must have exactly 3 entries, "
            f"got {len(raw_list)}"
        )
    for v in raw_list:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ConfigError(
                f"experiment {name!r}: sources[{index}].cell entries must be integers, "
                f"got {raw_list}"
            )
        if v < 0:
            raise ConfigError(
                f"experiment {name!r}: sources[{index}].cell entries must be non-negative, "
                f"got {raw_list}"
            )
    return raw_list[0], raw_list[1], raw_list[2]
