"""
core.domain.models: Entidades y Value Objects de Dominio del Sistema ACV Bolivia.

Modela el análisis de ciclo de vida (LCA), sin dependencia de Brightway2,
Pandas o Scipy.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core.domain.contracts import AnalyzerResult, MappingDiagnostic, MethodId
from ...core.domain.uncertainty import UncertaintyParams

# ==============================================================================
# 1. VALUE OBJECTS (Objetos de Valor Inmutables)
# ==============================================================================


@dataclass(frozen=True)
class Quantity:
    """Value Object que encapsula una cantidad física y su unidad. Evita la
    obsesión por primitivos."""

    amount: float
    unit: str

    def __post_init__(self):
        if self.amount < 0.0:
            raise ValueError(
                "La cantidad física no puede ser negativa en este contexto "
                "de inventario."
            )


@dataclass(frozen=True)
class Exchange:
    """
    Value Object que representa un flujo físico de entrada o salida en una actividad.
    Es inmutable (frozen=True) para garantizar la integridad referencial
    dentro del Proyecto.
    """

    component_id: (
        str  # Identificador del componente en el modelo de dominio (ej. 'torre_acero')
    )
    quantity: Quantity  # Objeto de valor inmutable
    exchange_type: str  # 'technosphere', 'biosphere', 'production'
    uncertainty: UncertaintyParams | None = None
    background_process_name: str | None = (
        None  # Nombre del proceso en la base de datos de fondo (ej. Ecoinvent)
    )

    @property
    def is_uncertain(self) -> bool:
        return self.uncertainty is not None


# ==============================================================================
# 2. ENTIDADES Y AGREGADOS (Aggregate Roots)
# ==============================================================================


@dataclass
class Project:
    """
    Aggregate Root que representa un proyecto o instalación energética.
    Es responsable de mantener la consistencia de sus intercambios (exchanges).
    """

    id: str
    name: str
    generation_kwh: float
    exchanges: list[Exchange] = field(default_factory=list)

    def add_exchange(self, exchange: Exchange) -> None:
        """Agrega un intercambio validando las reglas de negocio del dominio."""
        if exchange.quantity.amount <= 0.0:
            raise ValueError(
                f"El intercambio para el componente '{exchange.component_id}' "
                "debe tener una cantidad > 0."
            )
        self.exchanges.append(exchange)

    def get_technosphere_exchanges(self) -> list[Exchange]:
        """Método de dominio que filtra y devuelve solo los flujos de la tecnosfera."""
        return [e for e in self.exchanges if e.exchange_type == "technosphere"]

    @property
    def has_uncertainty(self) -> bool:
        """Regla de dominio: el proyecto requiere análisis estocástico si al
        menos un intercambio es incierto."""
        return any(e.is_uncertain for e in self.exchanges)


# ==============================================================================
# 3. VALUE OBJECTS DE RESULTADO (Resultados de Casos de Uso)
# ==============================================================================


@dataclass(frozen=True)
class LCAResult:
    """Value Object inmutable con el score de impacto determinístico."""

    project_id: str
    method_id: MethodId  # Consistente con el contrato (Tuple[str, ...])
    method_label: (
        str  # String legible para presentación (ej. "ReCiPe 2016 - Climate Change")
    )
    score: float
    score_per_kwh: float | None

    @property
    def is_normalized(self) -> bool:
        return self.score_per_kwh is not None and self.score_per_kwh > 0.0


@dataclass(frozen=True)
class HotspotResult:
    """Value Object inmutable que representa la contribución de un insumo
    al impacto total."""

    project_id: str
    method_id: MethodId
    component_id: str  # Vinculación al dominio (no al Excel)
    background_process_name: str
    impact: float
    impact_per_kwh: float | None
    unit: str

    @property
    def is_dominant(self) -> bool:
        """Regla de dominio: un hotspot se considera dominante si supera el 5%
        del impacto total (ejemplo)."""
        # Esta lógica podría requerir pasar el impacto total, o evaluarse en la
        # capa de aplicación.
        return abs(self.impact) > 1e-6


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Value Object que representa la distribución empírica de una simulación estocástica.
    Usa Sequence para permitir tanto listas nativas como np.ndarray sin
    acoplar el dominio a Numpy.
    """

    project_id: str
    method_id: MethodId
    scores: Sequence[float]
    simulation_type: str = "standard"  # "standard", "foreground", "piv"

    @property
    def n_iterations(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        """Cálculo de dominio básico sobre la distribución."""
        if self.n_iterations == 0:
            return 0.0
        # Usamos sum/len para no depender de numpy en el dominio puro,
        # aunque la infraestructura probablemente haya pasado un np.ndarray.
        return sum(self.scores) / self.n_iterations


# ==============================================================================
# 4. ENTIDADES DE ANÁLISIS DE SENSIBILIDAD
# ==============================================================================


@dataclass
class SensitivityReport:
    """Entidad de dominio: consolidador de resultados de análisis de sensibilidad.

    Implementa el algoritmo de agregación de rangos (Rank Aggregation) por media
    de rangos para generar un consenso unificado e inmune a sesgos de escala.

    Attributes
    ----------
    project_id : str
        Identificador del proyecto evaluado.
    method_id : MethodId
        Tupla del método de impacto evaluado.
    results : Dict[str, AnalyzerResult]
        Resultados por método de sensibilidad {method_name: result}.
    errors : List[str]
        Errores capturados durante la ejecución.
    methods_run : List[str]
        Nombres de métodos ejecutados exitosamente.
    skipped_methods : List[str]
        Nombres de métodos omitidos por configuración.
    diagnostic : Optional[MappingDiagnostic]
        Diagnóstico de integridad del mapeo matricial.
    """

    project_id: str
    method_id: MethodId
    results: dict[str, AnalyzerResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    methods_run: list[str] = field(default_factory=list)
    skipped_methods: list[str] = field(default_factory=list)
    diagnostic: MappingDiagnostic | None = None

    def add_result(self, result: AnalyzerResult) -> None:
        """Registra el resultado de un analizador o captura su error interno."""
        if result.error_message:
            self.errors.append(f"{result.method_name}: {result.error_message}")
        else:
            self.results[result.method_name] = result
            self.methods_run.append(result.method_name)

    def get_raw(self, method_name: str) -> list[Any]:
        """Retorna los resultados crudos de un método, o lista vacía si no existe."""
        res = self.results.get(method_name)
        return res.raw_results if res is not None else []

    def top_components(self, n: int = 10) -> list[str]:
        """Consolida rankings individuales mediante agregación ponderada por
        posición."""
        score_sum: dict[str, float] = {}
        count: dict[str, int] = {}

        for res in self.results.values():
            sorted_scores = sorted(res.scores, key=lambda x: abs(x.score), reverse=True)
            total_elements = len(sorted_scores)
            for rank, item in enumerate(sorted_scores):
                weight = total_elements - rank
                score_sum[item.component] = score_sum.get(item.component, 0.0) + weight
                count[item.component] = count.get(item.component, 0) + 1

        if not score_sum:
            return []

        avg = {
            component: score_sum[component] / count[component]
            for component in score_sum
        }

        return sorted(
            avg,
            key=lambda component: avg[component],
            reverse=True,
        )[:n]

    @property
    def has_errors(self) -> bool:
        """Regla de dominio: el reporte tiene errores si alguno fue capturado."""
        return len(self.errors) > 0

    @property
    def methods_executed_count(self) -> int:
        """Número de métodos ejecutados exitosamente."""
        return len(self.methods_run)
