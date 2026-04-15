"""Ecosystem simulation: spawn archive creatures on a shared grid.

Public API:
    EcosystemConfig, CreatureSource, SpawnConfig - experiment configuration
    EcosystemResult, EcosystemMeasures           - post-run outputs
    load_config_file, ConfigError                - YAML parser
    run_ecosystem                                - execute a single experiment
"""

from biota.ecosystem.config import (
    ConfigError,
    CreatureSource,
    EcosystemConfig,
    SpawnConfig,
    load_config_file,
)
from biota.ecosystem.result import EcosystemMeasures, EcosystemResult
from biota.ecosystem.run import run_ecosystem

__all__ = [
    "ConfigError",
    "CreatureSource",
    "EcosystemConfig",
    "EcosystemMeasures",
    "EcosystemResult",
    "SpawnConfig",
    "load_config_file",
    "run_ecosystem",
]
