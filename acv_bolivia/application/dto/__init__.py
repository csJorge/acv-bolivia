"""
application.dto: DTOs de la Capa de Aplicación.

Diferencia con los DTOs de infraestructura:
    - infrastructure/brightway/dto.py: DTOs crudos de cálculo (DeterministicScoreDTO,
      HotspotDTO, etc.) que representan datos intermedios.
    - application/dto/: DTOs de alto nivel que representan el resultado FINAL
      de un caso de uso, listos para ser presentados.

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# DTOs principales (uno por caso de uso)
# ==============================================================================
from .build_inventory import BuildInventoryResult
from .run_lca import RunLCAResult

# ==============================================================================
# DTOs auxiliares (sub-estructuras de los DTOs principales)
# ==============================================================================
from .run_montecarlo import MonteCarloProjectStats, RunMonteCarloResult
from .run_sensitivity import RunSensitivityResult

__all__ = [
    # DTOs principales
    "BuildInventoryResult",
    # DTOs auxiliares
    "MonteCarloProjectStats",
    "RunLCAResult",
    "RunMonteCarloResult",
    "RunSensitivityResult",
]
