"""
analysis.sensitivity - Motor, plotter y exporter de analisis de sensibilidad.

Expone los tres componentes principales del analisis de sensibilidad:
    - SensitivityEngine: orquesta los analizadores por proyecto/metodo.
    - SensitivityPlotter: genera visualizaciones estandar (Tornado, Morris, etc.).
    - SensitivityExporter: exporta reportes a Excel con hojas por metodo.

El SensitivityReport (entidad de dominio con rank aggregation) vive en
core.domain.models, no aqui, por ser logica de dominio pura.

Uso tipico desde un caso de uso:
    >>> engine = SensitivityEngine(lca_provider, analyzers, exclude_methods)
    >>> report = engine.run(project_name, method_id, component_samples, lca_scores)
    >>> SensitivityPlotter(output_dir).plot_all(report)
    >>> SensitivityExporter(output_dir).export(report)

Autor: Jorge Luis Corrales Suarez
"""

# Motor de ejecucion
# Re-exportar analizadores para conveniencia (acceso directo sin sub-import)
from .methods import (
    CorrelationAnalyzer,
    DeltaLCAAnalyzer,
    MorrisAnalyzer,
    RegressionAnalyzer,
    SHAPAnalyzer,
    SobolAnalyzer,
)
from .sensitivity_engine import SensitivityEngine
from .sensitivity_exporter import SensitivityExporter

# Plotter y exporter
from .sensitivity_plotter import SensitivityPlotter

__all__ = [
    "CorrelationAnalyzer",
    # Analizadores (re-exportados)
    "DeltaLCAAnalyzer",
    "MorrisAnalyzer",
    "RegressionAnalyzer",
    "SHAPAnalyzer",
    # Componentes principales
    "SensitivityEngine",
    "SensitivityExporter",
    "SensitivityPlotter",
    "SobolAnalyzer",
]
