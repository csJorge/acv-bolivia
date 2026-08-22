"""
application.use_cases: Casos de Uso de la Capa de Aplicación.

Cada caso de uso representa una operación de negocio completa del framework ACV Bolivia.
Los casos de uso:
    - Dependen de abstracciones (Protocols de application.contracts).
    - No conocen implementaciones concretas de infraestructura.
    - Retornan DTOs (application.dto) como resultado.
    - Son ensamblados por los compositores en infrastructure/composition/.

Casos de uso disponibles:
    - BuildInventoryUseCase: Construye el inventario ACV desde Excel hacia Brightway2.
    - RunLCAUseCase: Ejecuta el cálculo LCIA determinístico.
    - RunMonteCarloUseCase: Orquesta simulaciones Monte Carlo (BW, FG, PIV).
    - RunSensitivityUseCase: Ejecuta análisis de sensibilidad paramétrica.

Autor: Jorge Luis Corrales Suarez
"""

from .build_inventory import BuildInventoryUseCase
from .run_lca import RunLCAUseCase
from .run_montecarlo import MonteCarloRequest, RunMonteCarloUseCase
from .run_sensitivity import (
    RunSensitivityUseCase,
    SensitivityRequest,
    default_sensitivity_analyzers,
    resolve_component_samples,
)

__all__ = [
    # Casos de uso principales
    "BuildInventoryUseCase",
    # Request DTOs (agrupación de parámetros)
    "MonteCarloRequest",
    "RunLCAUseCase",
    "RunMonteCarloUseCase",
    "RunSensitivityUseCase",
    "SensitivityRequest",
    # Helpers públicos
    "default_sensitivity_analyzers",
    "resolve_component_samples",
]
