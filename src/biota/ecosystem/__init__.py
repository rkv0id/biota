"""Ecosystem simulation: spawn archive creatures on a shared grid.

The ecosystem module implements v2.0.0 of the biota experimental loop:
take one or more creatures from the MAP-Elites archive and observe what
happens when multiple copies share a large Flow-Lenia grid with global
parameters.

Public API:
    run_ecosystem(config, archive, on_event) -> EcosystemResult
"""

from biota.ecosystem.result import EcosystemConfig, EcosystemResult
from biota.ecosystem.run import run_ecosystem

__all__ = ["EcosystemConfig", "EcosystemResult", "run_ecosystem"]
