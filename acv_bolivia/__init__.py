"""
acv_bolivia - Framework de Análisis de Ciclo de Vida para proyectos
de energía renovable en Bolivia.

Integra Brightway2 + Ecoinvent 3.12 con simulación Monte Carlo,
tests estadísticos Kolmogorov-Smirnov y análisis de sensibilidad
multi-método (Delta LCA, Morris, Sobol, Correlation/PRCC, Regression/SRRC,
SHAP).

Uso rápido (Google Colab / Jupyter):

    >>> from acv_bolivia import ACVEngine
    >>> eng = ACVEngine.from_json("config/settings.json")
    >>> eng.build().run_lca().run_montecarlo()
    >>> eng.export("Reporte_ACV_Bolivia_2025")

Acceso a capas inferiores:

    >>> from acv_bolivia.core.domain import Project, Exchange
    >>> from acv_bolivia.infrastructure import BrightwayConnector

Autor: Jorge Luis Corrales Suarez
"""

__version__ = "1.0.0"
__author__ = "Jorge Luis Corrales Suarez"
__email__ = "jcorralessuarez@gmail.com"
__license__ = "MIT"

from .config.app_config import AppConfig
from .core.domain.models import Exchange, LCAResult, Project
from .interfaces.acv_engine import ACVEngine

__all__ = [
    "ACVEngine",
    "AppConfig",
    "Exchange",
    "LCAResult",
    "Project",
]
