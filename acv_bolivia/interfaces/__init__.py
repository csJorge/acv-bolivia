"""
interfaces: Capa de interfaz del framework ACV.

Este paquete contiene los componentes que interactúan con el mundo exterior:
exportadores, visualizadores y el motor de orquestación de alto nivel.

Módulos:
    - exporter: LCAExporter para generar reportes Excel de LCA/MC.
    - plotter: LCAPlotter para gráficos de resultados LCA/MC.
    - piv_plotter: PIVPlotter para gráficos de resultados PIV y SHAP.
    - acv_engine: ACVEngine, el motor de orquestación con control granular
      sobre las 4 fases (build, run_lca, run_montecarlo, run_sensitivity).

"""

from .acv_engine import ACVEngine
from .exporter import LCAExporter
from .piv_plotter import PIVPlotter
from .plotter import LCAPlotter

__all__ = [
    "ACVEngine",
    "LCAExporter",
    "LCAPlotter",
    "PIVPlotter",
]
