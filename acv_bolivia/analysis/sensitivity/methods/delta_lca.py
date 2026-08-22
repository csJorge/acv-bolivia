"""
analysis.sensitivity.methods.delta_lca - Perturbacion directa (Delta LCA / Tornado).

Calcula la elasticidad parametrica puntual mediante variaciones controladas de +/-Delta%
sobre los valores nominales del inventario, determinando el impacto diferencial
adecuado para la construccion de diagramas de Tornado.

Nota sobre rendimiento: este metodo requiere 2k+1 evaluaciones LCA (k componentes).
No es vectorizable porque cada evaluacion implica una refactorizacion matricial
completa. La optimizacion real se obtiene cuando el SensitivityEngine inyecta un
LcaEvaluator PIV (aproximacion lineal con h-vectors), reduciendo el costo a
productos escalares instantaneos.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....core.domain.contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
)


@dataclass(frozen=True)
class DeltaLCAResult:
    """Resultados detallados de perturbacion local y elasticidad para un componente.

    Conserva los scores perturbados y las amplitudes de oscilacion para dar soporte
    directo a la renderizacion del Tornado Chart en la capa de interfaces.

    Nota: el contexto (proyecto/método) lo proporciona el SensitivityReport.
    """

    component: str
    nominal_value: float = 0.0
    delta_fraction: float = 0.1
    score_nominal: float = 0.0
    score_plus: float = 0.0
    score_minus: float = 0.0
    sensitivity_index: float = 0.0

    @property
    def delta_score_plus(self) -> float:
        """Cambio absoluto en el impacto al incrementar el parametro."""
        return self.score_plus - self.score_nominal

    @property
    def delta_score_minus(self) -> float:
        """Cambio absoluto en el impacto al disminuir el parametro."""
        return self.score_minus - self.score_nominal

    @property
    def delta_score_rel_plus(self) -> float:
        """Cambio porcentual relativo (%) al incrementar el parametro."""
        if self.score_nominal == 0:
            return 0.0
        return 100.0 * self.delta_score_plus / abs(self.score_nominal)

    @property
    def delta_score_rel_minus(self) -> float:
        """Cambio porcentual relativo (%) al disminuir el parametro."""
        if self.score_nominal == 0:
            return 0.0
        return 100.0 * self.delta_score_minus / abs(self.score_nominal)

    @property
    def swing(self) -> float:
        """Amplitud de oscilacion total del score (eje de visualizacion Tornado)."""
        return self.score_plus - self.score_minus

    @property
    def abs_primary(self) -> float:
        """Magnitud absoluta de la elasticidad utilizada para el ordenamiento."""
        return abs(self.sensitivity_index)

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea el indice de elasticidad local a la estructura generica del
        framework."""
        return [
            ComponentSensitivityScore(
                component=self.component,
                score=self.sensitivity_index,
                metric_name=f"elasticity_delta_{self.delta_fraction}",
            )
        ]


class DeltaLCAAnalyzer:
    """Analizador determinista basado en perturbacion unidireccional directa.

    Implementa el protocolo SensitivityAnalyzer. Evalua de forma aislada el
    gradiente local de cambio sin requerir dependencias externas.
    """

    def __init__(
        self,
        deltas: Sequence[float] = (0.1, 0.2),
        components_subset: list[str] | None = None,
    ) -> None:
        """Configura los rangos de perturbacion diferencial y los filtros de variables.

        Parameters
        ----------
        deltas : Sequence[float]
            Fracciones de variacion a aplicar (ej. 0.1 = +/-10%).
        components_subset : Optional[List[str]]
            Subconjunto restrictivo opcional de componentes a evaluar.
        """
        self._deltas: list[float] = [float(d) for d in deltas]
        self._components_subset = components_subset

    @property
    def method_name(self) -> str:
        return "delta_lca"

    @property
    def requires_variance(self) -> bool:
        """Evalua evaluation_fn en vivo por perturbacion; no necesita muestras
        MC previas."""
        return False

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, np.ndarray] | None = None,
    ) -> AnalyzerResult:
        """Ejecuta el analisis secuencial por perturbacion local incremental.

        Reutiliza el evaluador inyectado. Si el engine inyecto un PIV evaluator,
        las 2k+1 evaluaciones son productos escalares instantaneos. Si inyecto un
        evaluador matricial completo, cada evaluacion implica refactorizacion.
        """
        components = self._components_subset or list(nominal_params.keys())
        score_nominal = evaluator.evaluate(nominal_params)

        raw_results: list[DeltaLCAResult] = []

        for comp in components:
            nominal_val = float(nominal_params.get(comp, 0.0))
            if nominal_val == 0:
                continue

            for delta in self._deltas:
                params_plus = dict(nominal_params)
                params_plus[comp] = nominal_val * (1.0 + delta)
                score_plus = evaluator.evaluate(params_plus)

                params_minus = dict(nominal_params)
                params_minus[comp] = nominal_val * (1.0 - delta)
                score_minus = evaluator.evaluate(params_minus)

                # Elasticidad puntual: (DeltaY/Y) / (DeltaX/X)
                delta_y = (score_plus - score_minus) / 2.0
                delta_x = nominal_val * delta

                if score_nominal != 0 and delta_x != 0:
                    si = (delta_y / score_nominal) / (delta_x / nominal_val)
                else:
                    si = 0.0

                raw_results.append(
                    DeltaLCAResult(
                        component=comp,
                        nominal_value=nominal_val,
                        delta_fraction=delta,
                        score_nominal=score_nominal,
                        score_plus=score_plus,
                        score_minus=score_minus,
                        sensitivity_index=si,
                    )
                )

        raw_results.sort(key=lambda item: item.abs_primary, reverse=True)

        unified_scores: list[ComponentSensitivityScore] = []
        for item in raw_results:
            unified_scores.extend(item.to_component_scores())

        return AnalyzerResult(
            method_name=self.method_name,
            scores=unified_scores,
            raw_results=raw_results,
        )
