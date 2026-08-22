"""
core.domain.contracts - Interfaces abstractas y contenedores de datos del dominio.

Refactorizado para cumplir estrictamente con:
- Dependency Inversion Principle (DIP) mediante Protocols.
- Interface Segregation Principle (ISP) unificando repositorios.
- Domain-Driven Design (DDD) con Value Objects y Repositorios claros.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from numpy.typing import NDArray

# Tipos de dominio personalizados para evitar "Primitive Obsession"
MethodId = tuple[str, ...]  # Estándar Brightway2 para métodos de impacto
ParameterDict = dict[str, float]


# ==============================================================================
# 1. VALUE OBJECTS Y DTOs DEL DOMINIO
# ==============================================================================


@dataclass(frozen=True)
class ComponentSensitivityScore:
    """Value Object inmutable. Representa la sensibilidad de un componente."""

    component: str
    score: float
    metric_name: str

    @property
    def is_significant(self) -> bool:
        """Regla de dominio: un score es significativo si su valor absoluto
        supera un umbral."""
        return abs(self.score) > 1e-6  # Ejemplo de lógica de dominio dentro del VO


@dataclass
class AnalyzerResult:
    """
    Entidad agregada de resultado de análisis.
    Mantiene la excelente separación entre la vista genérica (scores)
    y el detalle técnico (raw_results).
    """

    method_name: str
    scores: list[ComponentSensitivityScore] = field(default_factory=list)
    raw_results: list[Any] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.error_message is None


@dataclass(frozen=True)
class MappingDiagnostic:
    """Value Object que encapsula el diagnóstico de mapeo matricial."""

    unmatched_components: list[str]
    shared_cells_groups: list[list[str]]
    mapped_cells_count: int
    nominal_verified_score: float

    @property
    def has_critical_mapping_issues(self) -> bool:
        """Regla de dominio: es crítico si hay componentes no mapeados o colisiones."""
        return len(self.unmatched_components) > 0 or len(self.shared_cells_groups) > 0


# ==============================================================================
# 2. CONTRATOS DE INFRAESTRUCTURA (Inversión de Dependencias)
# ==============================================================================


class LcaEvaluator(Protocol):
    """
    Contrato para el evaluador de LCA.
    """

    def evaluate(self, parameters: ParameterDict) -> float:
        """
        Evalúa el impacto dado un diccionario de parámetros.
        Debe manejar internamente las excepciones de Brightway y devolver
        un float (o lanzar una excepción de dominio controlada).
        """
        ...


class LCAInfrastructureProvider(Protocol):
    """
    Contrato que debe implementar el adaptador de Infraestructura
    (ej. Brightway2Adapter).
    """

    def get_nominal_parameters(self, project_name: str) -> ParameterDict: ...

    def create_evaluator(
        self,
        project_name: str,
        method_id: MethodId,
        functional_unit_amount: float = 1.0,
    ) -> LcaEvaluator:
        """
        Factory method que devuelve un evaluador listo para ser usado por
        los algoritmos de sensibilidad (Morris, Sobol, etc.).
        """
        ...

    def create_piv_evaluator(
        self, nominal_params: ParameterDict
    ) -> LcaEvaluator | None:
        """
        Devuelve un evaluador lineal optimizado (PIV) si la infraestructura lo soporta.
        """
        ...

    def get_latest_mapping_diagnostic(self) -> MappingDiagnostic | None: ...


class SensitivityAnalyzer(Protocol):
    """
    Contrato polimórfico para algoritmos matemáticos.
    """

    @property
    def method_name(self) -> str: ...

    @property
    def requires_variance(self) -> bool:
        """
        True si necesita muestras MC previas (ej. SHAP, Correlación).
        False si evalúa la función en vivo (ej. Morris, Sobol, Delta).
        """
        ...

    def execute(
        self,
        nominal_params: ParameterDict,
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult: ...


# ==============================================================================
# 3. CONTRATOS DE REPOSITORIO (Patrón Repositorio de DDD)
# ==============================================================================


class LcaResultsRepository(Protocol):
    """
    Único punto de entrada para persistir y recuperar resultados de LCA y MC.
    Aplica el Principio de Segregación de Interfaces (ISP): el dominio no necesita
    saber si el resultado vino de PIV, Foreground o MC estándar. Solo le importa
    guardar y leer "Resultados de Simulación".
    """

    # --- Comandos (Write) ---
    def save_deterministic_result(
        self, project_name: str, method_id: MethodId, score: float
    ) -> None: ...

    def save_hotspots(
        self, project_name: str, method_id: MethodId, hotspots: list[Any]
    ) -> None: ...

    def save_simulation_result(
        self,
        project_name: str,
        method_id: MethodId,
        scores: NDArray,
        component_samples: dict[str, NDArray] | None = None,
        simulation_type: str = "standard",  # "standard", "foreground", "piv"
    ) -> None:
        """
        Unifica la persistencia de MC, Foreground y PIV.
        La implementación en infraestructura decidirá en qué tabla/archivo guardarlo.
        """
        ...

    # --- Queries (Read) ---
    def get_simulation_scores(
        self, project_name: str | None = None, method_id: MethodId | None = None
    ) -> list[NDArray]:
        """Obtiene los arreglos de scores de las simulaciones."""
        ...

    def get_component_samples(self, project_name: str) -> dict[str, NDArray] | None:
        """Obtiene las muestras de los componentes para un proyecto."""
        ...
