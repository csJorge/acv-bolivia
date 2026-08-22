"""
analysis: Pipelines de analisis matematico y estadistico del framework ACV Bolivia.

Subpaquetes:
    - sensitivity/: analisis de sensibilidad parametrica (motor, plotter, exporter,
      y seis metodos: Delta LCA, Morris, Sobol, SHAP, Correlation, Regression).
    - convergence/: diagnostico de convergencia de simulaciones Monte Carlo y de la
      confiabilidad de los indices de sensibilidad.
    - statistical_tests/: pruebas estadisticas sobre distribuciones (KS, Cohen's d).

Arquitectura:
    analysis/ depende de core.domain (protocolos y entidades).
    application/ depende de analysis/ (via inyeccion en compositores).
    infrastructure/ provee adaptadores que satisfacen los protocolos del dominio.

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Subpaquetes
# ==============================================================================

from .sensitivity import (
    CorrelationAnalyzer,
    DeltaLCAAnalyzer,
    MorrisAnalyzer,
    RegressionAnalyzer,
    SensitivityEngine,
    SensitivityExporter,
    SensitivityPlotter,
    SHAPAnalyzer,
    SobolAnalyzer,
)
from .statistical_tests.ks_tests import KSResult, PairwiseKSResult, run_pairwise_ks

__all__ = [
    "CorrelationAnalyzer",
    "DeltaLCAAnalyzer",
    "KSResult",
    "MorrisAnalyzer",
    "PairwiseKSResult",
    "RegressionAnalyzer",
    "SHAPAnalyzer",
    # Componentes de sensibilidad (re-exportados)
    "SensitivityEngine",
    "SensitivityExporter",
    "SensitivityPlotter",
    "SobolAnalyzer",
    "run_pairwise_ks",
    # Subpaquetes
    "sensitivity",
]
