"""
infrastructure/brightway/montecarlo/_pedigree_stats:
Value Object para estadísticas del PedigreeSampler.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PedigreeSamplerStats:
    """Estadísticas descriptivas del estado del sampler de pedigrí."""

    built: bool
    n_entries: int
    n_processes: int
    n_methods: int
    n_samples: int
