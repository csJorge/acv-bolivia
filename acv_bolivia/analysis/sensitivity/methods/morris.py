"""
analysis.sensitivity.methods.morris - Método de Morris (Elementary Effects).

Define MorrisResult (contenedor inmutable de resultados detallados por
componente) y MorrisAnalyzer (implementación del protocolo SensitivityAnalyzer).

run_morris() contiene la lógica matemática real (muestreo de trayectorias vía
SALib, evaluación LCA puntual, análisis de efectos elementales).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import NDArray

from ....analysis.sensitivity.methods._shared import (
    build_salib_problem,
    evaluate_lca_batch,
)
from ....core.domain.contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
)
from ....core.services.sensitivity_bounds import bounds_from_samples

# Constantes de control analítico e integridad estadística
RELIABILITY_CI_RATIO_THRESHOLD: float = 0.5
_NEGLIGIBLE_INDEX_THRESHOLD: float = 0.01


@dataclass(frozen=True)
class MorrisResult:
    """Resultado detallado del método de Morris para un componente específico.

    Conserva la granularidad matemática requerida para el análisis en el plano
    mu*/sigma y las validaciones de convergencia de trayectorias.

    Nota: El contexto (proyecto/método) lo proporciona el SensitivityReport.
    """

    component: str
    mu: float = 0.0
    mu_star: float = 0.0
    sigma: float = 0.0
    n_trajectories: int = 0
    mu_star_conf: float = 0.0  # Amplitud del intervalo de confianza bootstrap de SALib

    @property
    def is_nonlinear(self) -> bool:
        """True si el componente exhibe comportamiento no-lineal o interacciones
        cruzadas."""
        return self.mu_star > 0 and (self.sigma / self.mu_star) > 0.5

    @property
    def abs_primary(self) -> float:
        """Métrica primaria de ordenamiento utilizada por el ecosistema de reportes."""
        return self.mu_star

    @property
    def is_reliable(self) -> bool:
        """False si el índice mu_star no muestra evidencia suficiente de
        convergencia."""
        if self.mu_star <= _NEGLIGIBLE_INDEX_THRESHOLD:
            return True
        return (self.mu_star_conf / self.mu_star) <= RELIABILITY_CI_RATIO_THRESHOLD

    @property
    def reliability_note(self) -> str:
        """Explicación en texto plano si el índice no ha convergido; cadena
        vacía si sí."""
        if self.is_reliable:
            return ""
        ratio = (self.mu_star_conf / self.mu_star) if self.mu_star > 0 else float("inf")
        return (
            f"Intervalo de confianza amplio: mu_star_conf/mu_star={ratio:.2f} "
            f"(umbral={RELIABILITY_CI_RATIO_THRESHOLD}). "
            f"Se recomienda aumentar n_trajectories (actual={self.n_trajectories})."
        )

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea el resultado detallado a las estructuras genéricas del framework."""
        return [
            ComponentSensitivityScore(
                component=self.component, score=self.mu_star, metric_name="mu_star"
            ),
            ComponentSensitivityScore(
                component=self.component, score=self.sigma, metric_name="sigma"
            ),
        ]


def run_morris(
    param_bounds: dict[str, tuple[float, float]],
    evaluator: LcaEvaluator,
    n_trajectories: int = 20,
    n_levels: int = 4,
) -> list[MorrisResult]:
    """Ejecuta el método de Morris y retorna los resultados por componente.

    Parameters
    ----------
    param_bounds : dict[str, tuple[float, float]]
        Rango de exploración por componente, ``{componente: (min, max)}``.
    evaluator : LcaEvaluator
        Evaluador del modelo LCA en vivo.
    n_trajectories : int, default 20
        Número de trayectorias aleatorias (N).
    n_levels : int, default 4
        Número de niveles de la grilla espacial (p).

    Returns
    -------
    list[MorrisResult]
        Un MorrisResult por componente, con mu, mu_star, sigma y mu_star_conf.
    """
    try:
        from SALib.analyze import morris as morris_analyze
        from SALib.sample import morris as morris_sample
    except ImportError:
        raise ImportError(
            "SALib es requerido para ejecutar el método de Morris. "
            "Por favor, instala la dependencia externa: pip install SALib"
        )

    if not param_bounds:
        raise ValueError(
            "No se pudieron determinar los rangos de exploración para Morris."
        )

    components = list(param_bounds.keys())
    problem = build_salib_problem(param_bounds)

    X = morris_sample.sample(
        problem,
        N=n_trajectories,
        num_levels=n_levels,
        optimal_trajectories=None,
    )

    Y = evaluate_lca_batch(X, evaluator, components)

    Si = morris_analyze.analyze(
        problem,
        X,
        Y,
        conf_level=0.95,
        print_to_console=False,
        num_levels=n_levels,
    )

    return [
        MorrisResult(
            component=comp,
            mu=float(Si["mu"][i]),
            mu_star=float(Si["mu_star"][i]),
            sigma=float(Si["sigma"][i]),
            n_trajectories=n_trajectories,
            mu_star_conf=float(Si["mu_star_conf"][i]),
        )
        for i, comp in enumerate(components)
    ]


class MorrisAnalyzer:
    """Analizador de Morris (Elementary Effects) para el motor genérico de sensibilidad.

    Implementa el protocolo SensitivityAnalyzer delegando el cálculo matemático
    a run_morris() y adaptando su salida al formato genérico ComponentSensitivityScore.
    """

    def __init__(self, n_trajectories: int = 20, n_levels: int = 4) -> None:
        self._n_trajectories = n_trajectories
        self._n_levels = n_levels

    @property
    def method_name(self) -> str:
        return "morris"

    @property
    def requires_variance(self) -> bool:
        """False: evalúa el modelo en vivo por trayectorias, sin muestras MC previas."""
        return False

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult:
        """Ejecuta las trayectorias de Morris y consolida el resultado genérico."""

        param_bounds = bounds_from_samples(component_samples, nominal_params)

        results = run_morris(
            param_bounds=param_bounds,
            evaluator=evaluator,
            n_trajectories=self._n_trajectories,
            n_levels=self._n_levels,
        )

        unified_scores: list[ComponentSensitivityScore] = []
        for r in results:
            unified_scores.extend(r.to_component_scores())

        return AnalyzerResult(
            method_name=self.method_name,
            scores=unified_scores,
            raw_results=results,
        )
