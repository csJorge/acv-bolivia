"""
infrastructure.brightway.dto: Objetos de Transferencia de Datos (DTOs)
de Infraestructura.

Define las estructuras de datos que retornan los componentes de
infraestructura (calculadores y runners) hacia la capa de aplicación.

Los DTOs no permiten reasignar sus atributos, pero sus listas, diccionarios y
arrays internos siguen siendo mutables.

Estos DTOs son crudos y específicos de Brightway2. La capa de aplicación
es responsable de transformarlos en entidades de dominio puras si es necesario.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from ...core.domain.contracts import MethodId


@dataclass(frozen=True)
class DeterministicScoreDTO:
    """Score determinístico de un proyecto y método de impacto.

    Attributes
    ----------
    project_id : str
        Identificador del proyecto evaluado.
    method_id : MethodId
        Identificador completo del método de impacto.
    method_label : str
        Etiqueta legible del método.
    score : float
        Impacto determinístico calculado.
    unit : str
        Unidad del impacto.
    """

    project_id: str
    method_id: MethodId
    method_label: str
    score: float
    unit: str


@dataclass(frozen=True)
class HotspotDTO:
    """Contribución de un exchange al impacto de un proyecto y método.

    Attributes
    ----------
    project_id : str
        Identificador del proyecto evaluado.
    method_id : MethodId
        Identificador completo del método de impacto.
    method_label : str
        Etiqueta legible del método.
    component_id : str
        Identificador del componente del inventario.
    background_process_name : str
        Nombre del proceso de fondo asociado.
    impact : float
        Impacto atribuido al exchange.
    unit : str
        Unidad del impacto.
    percentage : float
        Porcentaje del impacto total atribuido al exchange.
    """

    project_id: str
    method_id: MethodId
    method_label: str
    component_id: str
    background_process_name: str
    impact: float
    unit: str
    percentage: float


@dataclass(frozen=True)
class LCACalculationResult:
    """Resultados de un cálculo LCIA determinístico.

    Attributes
    ----------
    scores : list of DeterministicScoreDTO
        Scores calculados por proyecto y método.
    hotspots : list of HotspotDTO
        Contribuciones de exchanges identificadas como hotspots.
    elapsed_seconds : float
        Tiempo total del cálculo, en segundos.
    """

    scores: list[DeterministicScoreDTO] = field(default_factory=list)
    hotspots: list[HotspotDTO] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class MonteCarloSimulationResult:
    """Resultados de una simulación Monte Carlo completa con Brightway2.

    Attributes
    ----------
    scores : NDArray[Any]
        Scores con forma ``(n_methods, n_projects, n_iterations)``.
    project_ids : list of str
        Proyectos en el orden de la segunda dimensión de ``scores``.
    method_ids : list of MethodId
        Métodos en el orden de la primera dimensión de ``scores``.
    method_labels : list of str
        Etiquetas legibles alineadas con ``method_ids``.
    elapsed_seconds : float
        Tiempo total de la simulación, en segundos.
    iterations_completed : int
        Número de iteraciones completadas.
    solver_name : str
        Nombre del solucionador utilizado.
    """

    scores: NDArray[Any]  # shape: (n_methods, n_projects, n_iterations)
    project_ids: list[str] = field(default_factory=list)
    method_ids: list[MethodId] = field(default_factory=list)
    method_labels: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    iterations_completed: int = 0
    solver_name: str = ""


@dataclass(frozen=True)
class ForegroundSimulationResult:
    """Resultados de una simulación Monte Carlo del foreground.

    Attributes
    ----------
    project_id : str
        Identificador del proyecto simulado.
    method_scores : dict[MethodId, NDArray[Any]]
        Scores por método, normalmente con forma ``(n_iterations,)``.
    component_samples : dict[str, NDArray[Any]]
        Muestras generadas por componente, con una entrada por componente.
    iterations_completed : int
        Número de iteraciones completadas.
    """

    project_id: str
    method_scores: dict[MethodId, NDArray[Any]] = field(default_factory=dict)
    component_samples: dict[str, NDArray[Any]] = field(default_factory=dict)
    iterations_completed: int = 0


@dataclass(frozen=True)
class PIVSimulationResult:
    """Resultados de una simulación Monte Carlo mediante PIV.

    Attributes
    ----------
    project_id : str
        Identificador del proyecto simulado.
    method_scores : dict[MethodId, NDArray[Any]]
        Scores por método, normalmente con forma ``(n_iterations,)``.
    component_samples : dict[str, NDArray[Any]]
        Muestras generadas por componente.
    piv_contributions : dict[MethodId, dict[str, NDArray[Any]]]
        Contribuciones por método y componente, normalmente con una muestra
        por iteración.
    iterations_completed : int
        Número de iteraciones completadas.
    """

    project_id: str
    method_scores: dict[MethodId, NDArray[Any]] = field(default_factory=dict)
    component_samples: dict[str, NDArray[Any]] = field(default_factory=dict)
    piv_contributions: dict[MethodId, dict[str, NDArray[Any]]] = field(
        default_factory=dict
    )
    iterations_completed: int = 0
