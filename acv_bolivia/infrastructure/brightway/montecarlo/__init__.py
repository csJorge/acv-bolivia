"""
infrastructure.brightway.montecarlo: Motor de Simulación Estocástica de Montecarlo.

Este submódulo expone los componentes numéricos, algebraicos y
estadísticos encargados de
procesar los análisis de incertidumbre paramétrica y epistémica en el
inventario (frente)
y en la base de datos de fondo (Ecoinvent).

Estrategias de Simulación Disponibles:
    1. MonteCarloRunner (Completo): Perturba simultáneamente el inventario
     y el fondo (Ecoinvent)
       de forma secuencial lineal, aislando la variabilidad integral del sistema.
    2. ForegroundMCRunner (Estático): Varía exclusivamente los parámetros del
     Excel, manteniendo
       Ecoinvent nominal. Genera las muestras sincronizadas necesarias
       para SHAP, regresiones y PRCC.
    3. PIVMonteCarloRunner (Lineal): Aproximación lineal instantánea mediante
     producto escalar O(N x k)
       basado en h_vectors analíticos y matrices de perturbación por pedigrí de fondo.

Optimizaciones Científicas Implementadas:
    - sample_vectorized(): Rutinas de muestreo pseudoaleatorio delegadas al
      C nativo de NumPy (50-100x más rápido).
    - C_stack matricial: Consolidación esparza CSR para evaluar múltiples
      métodos en un único producto de vectores.
    - patch_matrix(): Parchado de offsets in-place en el array de datos de
      Scipy CSR a coste lineal O(k).
    - Selector Dinámico de Solucionadores: Enrutamiento óptimo entre Intel
      MKL PARDISO y SciPy SuperLU.

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Motores de Simulación (API Pública Principal)
# ==============================================================================
# ==============================================================================
# DTOs de Resultado (Retornados por los Runners)
# ==============================================================================
from ....infrastructure.brightway.dto import (
    ForegroundSimulationResult,
    MonteCarloSimulationResult,
    PIVSimulationResult,
)
from ._bw_runner import MonteCarloRunner
from ._component_mapper import ComponentToMatrixMapper
from ._distribution_strategies import (
    DeterministicStrategy,
    LognormalStrategy,
    NormalStrategy,
    SamplingStrategy,
    TriangularStrategy,
    UniformStrategy,
    WeibullStrategy,
)

# ==============================================================================
# Muestreo Estadístico
# ==============================================================================
from ._distributions import sample_vectorized
from ._fg_matrix_patcher import ForegroundMatrixPatcher
from ._fg_runner import ForegroundMCRunner

# ==============================================================================
# Utilidades Matriciales
# ==============================================================================
from ._matrix_utils import (
    build_c_stack,
    build_data_positions,
    patch_matrix,
)

# ==============================================================================
# Pedigrí y Solucionadores
# ==============================================================================
from ._pedigree_sampler import PedigreeSampler
from ._pedigree_stats import PedigreeSamplerStats
from ._piv_runner import PIVMonteCarloRunner
from ._piv_vector_calculator import PivVectorCalculator

# ==============================================================================
# Presenter de Diagnóstico
# ==============================================================================
from ._processor_presenter import ProcessorPresenter

# ==============================================================================
# Procesamiento de Muestras (Strategy Pattern)
# ==============================================================================
from ._sample_processor import SampleProcessor, create_sample_processor
from ._sampling_rules import (
    DependencyRule,
    MixRule,
    SamplingRule,
)

# ==============================================================================
# Componentes Internos de los Runners (Para extensión y testing)
# ==============================================================================
from ._simulation_loop import SimulationLoop
from ._solver import get_solver
from .method_filter import MethodFilter

__all__ = [
    "ComponentToMatrixMapper",
    "DependencyRule",
    "DeterministicStrategy",
    "ForegroundMCRunner",
    "ForegroundMatrixPatcher",
    "ForegroundSimulationResult",
    "LognormalStrategy",
    "MethodFilter",
    "MixRule",
    # Motores de Simulación (API Principal)
    "MonteCarloRunner",
    # DTOs de Resultado
    "MonteCarloSimulationResult",
    "NormalStrategy",
    "PIVMonteCarloRunner",
    "PIVSimulationResult",
    # Pedigrí y Solucionadores
    "PedigreeSampler",
    "PedigreeSamplerStats",
    "PivVectorCalculator",
    # Presenter
    "ProcessorPresenter",
    # Procesamiento de Muestras
    "SampleProcessor",
    "SamplingRule",
    "SamplingStrategy",
    # Componentes Internos (Para extensión)
    "SimulationLoop",
    "TriangularStrategy",
    "UniformStrategy",
    "WeibullStrategy",
    # Utilidades Matriciales
    "build_c_stack",
    "build_data_positions",
    "create_sample_processor",
    "get_solver",
    "patch_matrix",
    # Muestreo Estadístico
    "sample_vectorized",
]
