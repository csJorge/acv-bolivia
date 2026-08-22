"""
infrastructure.brightway - Adaptadores e Infraestructura Brightway2 del
Sistema ACV Bolivia.

Este paquete expone los componentes de persistencia, conectores de entorno y motores
numéricos encargados de interactuar con el backend de cálculo ambiental de Brightway2.

Todos los adaptadores implementan o alimentan los contratos abstractos
(Protocols) del Core,
permitiendo un desacoplamiento total de la lógica matemática respecto a las
estructuras físicas de las bases de datos de inventario y de fondo (Ecoinvent).

Componentes Expuestos:
    - LCACalculator: Motor de cálculo analítico determinístico de impactos y hotspots.
    - BrightwayConnector: Conector dinámico encargado de inicializar el
    entorno y sys.path.
    - ActivityRepository: Repositorio SQLite de persistencia de nodos e intercambios.
    - BrightwayLCAProvider: Adaptador que implementa LCAInfrastructureProvider
    para sensibilidad.
    - MonteCarloRunner: Motor de simulación estocástica secuencial de fondo y frente.
    - ForegroundMCRunner: Simulador estático rápido del frente enfocado en
    análisis ML (SHAP/PRCC).
    - PIVMonteCarloRunner: Simulador lineal instantáneo con inyección de
    varianza por pedigrí.
    - SampleProcessor: Procesador de muestras con reglas físicas (DEP, MIX).
    - MethodFilter: Filtro analítico de categorías jerárquicas ambientales
    (midpoint/endpoint).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

# ==============================================================================
# Componentes Analíticos Determinísticos y de Persistencia
# ==============================================================================
from .bw_lca_calculator import LCACalculator
from .bw_context import BrightwayConnector
from .bw_activity_repository import ActivityRepository
from .adapters import BrightwayLCAProvider

# ==============================================================================
# DTOs de Resultado (Retornados por los Calculadores y Runners)
# ==============================================================================
from .dto import (
    LCACalculationResult,
    DeterministicScoreDTO,
    HotspotDTO,
    MonteCarloSimulationResult,
    ForegroundSimulationResult,
    PIVSimulationResult,
)

# ==============================================================================
# Motores de Simulación Estocástica (Importación Directa desde montecarlo/)
# ==============================================================================
from .montecarlo import (
    MonteCarloRunner,
    ForegroundMCRunner,
    PIVMonteCarloRunner,
)

# ==============================================================================
# Procesamiento de Muestras (Strategy Pattern)
# ==============================================================================
from .montecarlo import (
    SampleProcessor,
    create_sample_processor,
    DependencyRule,
    MixRule,
)

# ==============================================================================
# Muestreo Estadístico y Utilidades Matriciales
# ==============================================================================
from .montecarlo import (
    sample_vectorized,
    ComponentToMatrixMapper,
    build_c_stack,
    build_data_positions,
    patch_matrix,
)

# ==============================================================================
# Pedigrí y Solucionadores
# ==============================================================================
from .montecarlo import (
    PedigreeSampler,
    get_solver,
)

# ==============================================================================
# Filtros y Validaciones
# ==============================================================================
from .montecarlo import MethodFilter
from .validators import validate_ecoinvent_db

# ==============================================================================
# Protocolos (Para Inversión de Dependencias)
# ==============================================================================
from .protocols import (
    BrightwayConnectionManager,
    ActivityBuilder,
    MonteCarloSimulator,
)

# ==============================================================================
# Constantes
# ==============================================================================
from .constants import (
    BIOSPHERE_DB_NAME,
    DEFAULT_LOCATION,
    DEFAULT_UNIT,
    PRODUCTION_EXCHANGE_TYPE,
    TECHNOSPHERE_EXCHANGE_TYPE,
    BIOSPHERE_EXCHANGE_TYPE,
    DEFAULT_IMPACT_UNIT,
    EXCHANGE_COMPONENT_TAG,
    MATCHING_AMOUNT_EPSILON,
    MATCHING_TOLERANCE,
    KG_TO_TONNES_FACTOR,
)

__all__ = [
    # Componentes Analíticos Determinísticos y de Persistencia
    "LCACalculator",
    "BrightwayConnector",
    "ActivityRepository",
    "BrightwayLCAProvider",
    # DTOs de Resultado
    "LCACalculationResult",
    "DeterministicScoreDTO",
    "HotspotDTO",
    "MonteCarloSimulationResult",
    "ForegroundSimulationResult",
    "PIVSimulationResult",
    # Motores de Simulación Estocástica
    "MonteCarloRunner",
    "ForegroundMCRunner",
    "PIVMonteCarloRunner",
    # Procesamiento de Muestras
    "SampleProcessor",
    "create_sample_processor",
    "DependencyRule",
    "MixRule",
    # Muestreo Estadístico y Utilidades Matriciales
    "sample_vectorized",
    "ComponentToMatrixMapper",
    "build_c_stack",
    "build_data_positions",
    "patch_matrix",
    # Pedigrí y Solucionadores
    "PedigreeSampler",
    "get_solver",
    # Filtros y Validaciones
    "MethodFilter",
    "validate_ecoinvent_db",
    # Protocolos
    "BrightwayConnectionManager",
    "ActivityBuilder",
    "MonteCarloSimulator",
    # Constantes
    "BIOSPHERE_DB_NAME",
    "DEFAULT_LOCATION",
    "DEFAULT_UNIT",
    "PRODUCTION_EXCHANGE_TYPE",
    "TECHNOSPHERE_EXCHANGE_TYPE",
    "BIOSPHERE_EXCHANGE_TYPE",
    "DEFAULT_IMPACT_UNIT",
    "EXCHANGE_COMPONENT_TAG",
    "MATCHING_AMOUNT_EPSILON",
    "MATCHING_TOLERANCE",
    "KG_TO_TONNES_FACTOR",
]
