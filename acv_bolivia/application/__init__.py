"""
application: Capa de Aplicación del Sistema ACV Bolivia.

Arquitectura (Clean Architecture):
    application/
    ├── contracts.py      - Protocolos que la infraestructura debe implementar.
    ├── dto/              - DTOs de resultado de cada caso de uso.
    └── use_cases/        - Casos de uso que orquestan la lógica de negocio.

Flujo de dependencias:
    interfaces/ → application/ → core/
                      ↑
              infrastructure/ (implementa los protocols de application/contracts.py)

Uso desde la capa de interfaces (adaptadores):
    >>> from application.use_cases import RunLCAUseCase
    >>> from infrastructure.composition import create_run_lca_use_case
    >>> use_case = create_run_lca_use_case(config, connector, ...)
    >>> result = use_case.run(patron_metodo="ReCiPe 2016")

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Contratos (Protocols de la capa de aplicación)
# ==============================================================================
# ==============================================================================
# DTOs (Resultados de casos de uso)
# ==============================================================================
# ==============================================================================
# Casos de uso
# ==============================================================================
from . import contracts, dto, use_cases
from .services import calculate_mc_stats
from .contracts import (
    BrightwayActivityBuilder,
    BrightwayConnectionManager,
    ForegroundSimulationStrategy,
    InventoryDataAuditor,
    # Inventario
    InventoryLoader,
    LCAInfrastructureProvider,
    LCIAComputingStrategy,
    MCResultsReader,
    MCStatsCalculationStrategy,
    # LCIA
    MethodFilteringStrategy,
    # Monte Carlo
    MonteCarloSimulationStrategy,
    PIVSimulationStrategy,
    # Normalización y persistencia
    ResultsNormalizationStrategy,
    ResultsPersistenceStrategy,
    # Procesamiento
    SampleProcessingStrategy,
    # Sensibilidad
    SensitivityEngineStrategy,
)
from .dto import (
    # DTOs principales
    BuildInventoryResult,
    # DTOs auxiliares
    MonteCarloProjectStats,
    RunLCAResult,
    RunMonteCarloResult,
    RunSensitivityResult,
)
from .use_cases import (
    BuildInventoryUseCase,
    # Request DTOs
    MonteCarloRequest,
    RunLCAUseCase,
    RunMonteCarloUseCase,
    RunSensitivityUseCase,
    SensitivityRequest,
    # Helpers
    default_sensitivity_analyzers,
    resolve_component_samples,
)

__all__ = [
    "BrightwayActivityBuilder",
    "BrightwayConnectionManager",
    # DTOs principales
    "BuildInventoryResult",
    "calculate_mc_stats",
    # Casos de uso
    "BuildInventoryUseCase",
    "ForegroundSimulationStrategy",
    "InventoryDataAuditor",
    # Contratos
    "InventoryLoader",
    "LCAInfrastructureProvider",
    "LCIAComputingStrategy",
    "MCResultsReader",
    "MCStatsCalculationStrategy",
    "MethodFilteringStrategy",
    # DTOs auxiliares
    "MonteCarloProjectStats",
    # Request DTOs
    "MonteCarloRequest",
    "MonteCarloSimulationStrategy",
    "PIVSimulationStrategy",
    "ResultsNormalizationStrategy",
    "ResultsPersistenceStrategy",
    "RunLCAResult",
    "RunLCAUseCase",
    "RunMonteCarloResult",
    "RunMonteCarloUseCase",
    "RunSensitivityResult",
    "RunSensitivityUseCase",
    "SampleProcessingStrategy",
    "SensitivityEngineStrategy",
    "SensitivityRequest",
    # Submódulos
    "contracts",
    # Helpers
    "default_sensitivity_analyzers",
    "dto",
    "resolve_component_samples",
    "use_cases",
]
