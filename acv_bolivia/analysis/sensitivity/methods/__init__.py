"""
analysis.sensitivity.methods: Metodos de analisis de sensibilidad.

Analizadores disponibles:
    - DeltaLCAAnalyzer: perturbacion directa (+/-Delta%), para diagramas de Tornado.
    - MorrisAnalyzer: efectos elementales (mu*, sigma), screening global.
    - SobolAnalyzer: indices de varianza (S1, ST), analisis global con interacciones.
    - SHAPAnalyzer: valores de Shapley via ML (XGBoost/RandomForest/Ridge).
    - CorrelationAnalyzer: Pearson, Spearman y PRCC sobre muestras MC.
    - RegressionAnalyzer: SRC y SRRC via OLS vectorizado.

Autor: Jorge Luis Corrales Suarez
"""

# Analizadores (implementan SensitivityAnalyzer)
from .correlation import CorrelationAnalyzer, CorrelationResult

# Resultados detallados por metodo (para plotter/exporter via report.get_raw)
from .delta_lca import DeltaLCAAnalyzer, DeltaLCAResult
from .morris import MorrisAnalyzer, MorrisResult
from .regression import RegressionAnalyzer, RegressionResult
from .shap import SHAPAnalyzer, SHAPResult
from .sobol import SobolAnalyzer, SobolResult

__all__ = [
    "CorrelationAnalyzer",
    "CorrelationResult",
    # Analizadores
    "DeltaLCAAnalyzer",
    # Resultados detallados
    "DeltaLCAResult",
    "MorrisAnalyzer",
    "MorrisResult",
    "RegressionAnalyzer",
    "RegressionResult",
    "SHAPAnalyzer",
    "SHAPResult",
    "SobolAnalyzer",
    "SobolResult",
]
