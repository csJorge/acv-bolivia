"""
infrastructure - Adaptadores Externos del Sistema ACV Bolivia.

Esta capa conecta el dominio puro con las herramientas externas:
    - input/: Carga de datos desde Excel (ExcelInventoryLoader, parsers, validadores).
    - brightway/: Interacción con Brightway2 (conectores, repositorios, motores MC).
    - persistence/: Persistencia de caché en disco (ResultsFileRepository).
    - composition/: Raíz de composición (ensamblaje de casos de uso con dependencias).

Arquitectura en Capas (Clean Architecture):
    El dominio (core/) es independiente de esta capa. Los casos de uso (application/)
    reciben abstracciones (Protocol) y esta capa provee las implementaciones concretas.

Uso desde la Capa de Aplicación (Recomendado):
    >>> from infrastructure.composition import create_build_inventory_use_case
    >>> use_case = create_build_inventory_use_case(config)
    >>> result = use_case.run(force_rebuild=False)

Uso Directo (Solo para Casos Especiales):
    >>> from infrastructure import ExcelInventoryLoader, BrightwayConnector
    >>> loader = ExcelInventoryLoader(path="inventario.xlsx")
    >>> inventory = loader.load()

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Subpaquetes (Acceso Directo)
# ==============================================================================
from . import brightway, composition, input, persistence

# ==============================================================================
# Componentes Analíticos Determinísticos (brightway/)
# ==============================================================================
# ==============================================================================
# DTOs de Resultado (brightway/)
# ==============================================================================
# ==============================================================================
# Motores de Simulación Estocástica (brightway/montecarlo/)
# ==============================================================================
# ==============================================================================
# Procesamiento de Muestras (brightway/montecarlo/)
# ==============================================================================
# ==============================================================================
# Muestreo Estadístico y Utilidades (brightway/montecarlo/)
# ==============================================================================
# ==============================================================================
# Pedigrí y Solucionadores (brightway/montecarlo/)
# ==============================================================================
# ==============================================================================
# Protocolos (brightway/)
# ==============================================================================
# ==============================================================================
# Validaciones (brightway/)
# ==============================================================================
# ==============================================================================
# Constantes (brightway/)
# ==============================================================================
from .brightway import (
    BIOSPHERE_DB_NAME,
    BIOSPHERE_EXCHANGE_TYPE,
    DEFAULT_IMPACT_UNIT,
    DEFAULT_LOCATION,
    DEFAULT_UNIT,
    EXCHANGE_COMPONENT_TAG,
    KG_TO_TONNES_FACTOR,
    MATCHING_TOLERANCE,
    PRODUCTION_EXCHANGE_TYPE,
    TECHNOSPHERE_EXCHANGE_TYPE,
    ActivityBuilder,
    ActivityRepository,
    BrightwayConnectionManager,
    BrightwayConnector,
    BrightwayLCAProvider,
    ComponentToMatrixMapper,
    DependencyRule,
    DeterministicScoreDTO,
    ForegroundMCRunner,
    ForegroundSimulationResult,
    HotspotDTO,
    LCACalculationResult,
    LCACalculator,
    MethodFilter,
    MixRule,
    MonteCarloRunner,
    MonteCarloSimulationResult,
    MonteCarloSimulator,
    PedigreeSampler,
    PIVMonteCarloRunner,
    PIVSimulationResult,
    SampleProcessor,
    build_c_stack,
    build_data_positions,
    create_sample_processor,
    get_solver,
    patch_matrix,
    sample_vectorized,
    validate_ecoinvent_db,
)

# ==============================================================================
# Composición (composition/)
# ==============================================================================
from .composition import (
    create_build_inventory_use_case,
)

# ==============================================================================
# Carga de Datos (input/)
# ==============================================================================
from .input import (
    RESERVED_EXCEL_COLUMNS,
    ExcelInventoryLoader,
    ExcelRawData,
    InventoryLoadResult,
    build_generation_dict,
    parse_uncertainty_from_excel_row,
    validate_inventory_mapping,
)

# ==============================================================================
# Persistencia de Caché (persistence/)
# ==============================================================================
from .persistence import (
    CacheStorage,
    PickleGzipStorage,
    ResultsFileRepository,
)

__all__ = [
    # Constantes
    "BIOSPHERE_DB_NAME",
    "BIOSPHERE_EXCHANGE_TYPE",
    "DEFAULT_IMPACT_UNIT",
    "DEFAULT_LOCATION",
    "DEFAULT_UNIT",
    "EXCHANGE_COMPONENT_TAG",
    "KG_TO_TONNES_FACTOR",
    "MATCHING_TOLERANCE",
    "PRODUCTION_EXCHANGE_TYPE",
    "RESERVED_EXCEL_COLUMNS",
    "TECHNOSPHERE_EXCHANGE_TYPE",
    "ActivityBuilder",
    "ActivityRepository",
    # Protocolos
    "BrightwayConnectionManager",
    "BrightwayConnector",
    "BrightwayLCAProvider",
    "CacheStorage",
    "ComponentToMatrixMapper",
    "DependencyRule",
    "DeterministicScoreDTO",
    # Carga de Datos
    "ExcelInventoryLoader",
    "ExcelRawData",
    "ForegroundMCRunner",
    "ForegroundSimulationResult",
    "HotspotDTO",
    "InventoryLoadResult",
    # DTOs de Resultado
    "LCACalculationResult",
    # Componentes Analíticos Determinísticos
    "LCACalculator",
    "MethodFilter",
    "MixRule",
    # Motores de Simulación Estocástica
    "MonteCarloRunner",
    "MonteCarloSimulationResult",
    "MonteCarloSimulator",
    "PIVMonteCarloRunner",
    "PIVSimulationResult",
    # Pedigrí y Solucionadores
    "PedigreeSampler",
    "PickleGzipStorage",
    # Persistencia de Caché
    "ResultsFileRepository",
    # Procesamiento de Muestras
    "SampleProcessor",
    "brightway",
    "build_c_stack",
    "build_data_positions",
    "build_generation_dict",
    "composition",
    # Composición
    "create_build_inventory_use_case",
    "create_sample_processor",
    "get_solver",
    # Subpaquetes
    "input",
    "parse_uncertainty_from_excel_row",
    "patch_matrix",
    "persistence",
    # Muestreo Estadístico y Utilidades
    "sample_vectorized",
    # Validaciones
    "validate_ecoinvent_db",
    "validate_inventory_mapping",
]
