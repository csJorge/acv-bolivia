"""
analysis.sensitivity.methods.regression - Regresión lineal estandarizada (SRC/SRRC).

Calcula coeficientes de regresión lineal estandarizados (SRC) y de rangos (SRRC)
mediante mínimos cuadrados ordinarios (OLS) vectorizados, evaluando la validez del
modelo lineal a través del coeficiente de determinación R^2.

Implementación completamente vectorizada con NumPy/SciPy (lstsq via LAPACK,
rankdata vectorizado, broadcasting). Costo O(n*k^2) para n muestras y k componentes.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from ....core.domain.contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
)


@dataclass(frozen=True)
class RegressionResult:
    """Resultados detallados de regresión paramétrica estandarizada para un componente.

    Conserva los coeficientes SRC, SRRC y el R^2 del modelo global para dar soporte
    a verificaciones analíticas y capas de visualización avanzada.

    Nota: el contexto (proyecto/método) lo proporciona el SensitivityReport.
    """

    component: str
    src: float | None = None
    srrc: float | None = None
    r2_model: float | None = None
    n: int = 0
    is_reliable: bool = True
    reliability_note: str = ""

    @property
    def primary_index(self) -> float | None:
        """Índice jerárquico principal para el ordenamiento: SRRC > SRC."""
        return self.srrc if self.srrc is not None else self.src

    @property
    def abs_primary(self) -> float:
        """Valor absoluto de la métrica principal para ranking."""
        v = self.primary_index
        return abs(v) if v is not None else 0.0

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea el coeficiente estandarizado principal a la estructura generica."""
        v = self.primary_index
        if v is None:
            return []
        return [
            ComponentSensitivityScore(
                component=self.component, score=v, metric_name="regression_primary"
            )
        ]


# ==============================================================================
# Nucleo de algebra lineal vectorizada (OLS)
# ==============================================================================


def _ols_vectorized(X: NDArray, y: NDArray) -> tuple[NDArray, float]:
    """Resuelve la ecuación normal de mínimos cuadrados mediante descomposición LAPACK.

    Parameters
    ----------
    X : NDArray
        Matriz de diseño estandarizada (n_samples, n_components).
    y : NDArray
        Vector objetivo estandarizado (n_samples,).

    Returns
    -------
    tuple[NDArray, float]
        Tupla (beta, r2): vector de coeficientes estandarizados y R^2 del modelo.
    """
    n, k = X.shape
    if k == 0 or n <= k:
        return np.zeros(k), 0.0

    # linalg.lstsq usa SVD por debajo, manejando multicolinealidad de forma segura
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot > 0 and residuals.size > 0:
        r2 = 1.0 - (float(residuals[0]) / ss_tot)
    else:
        # Cálculo manual de respaldo si la matriz presenta déficit de rango
        y_pred = X @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return beta, max(0.0, min(1.0, r2))


# ==============================================================================
# API publica: implementacion del analizador
# ==============================================================================


class RegressionAnalyzer:
    """Analizador estadístico basado en regresión lineal multivariada estandarizada.

    Implementa el protocolo SensitivityAnalyzer. Evalua el impacto atributivo
    procesando las matrices originales (SRC) y de rangos (SRRC).
    """

    def __init__(self, compute_src: bool = True, compute_srrc: bool = True) -> None:
        """Configura los estimadores de regresión a activar en la corrida.

        Parameters
        ----------
        compute_src : bool
            Si True, calcula coeficientes sobre datos estandarizados.
        compute_srrc : bool
            Si True, calcula coeficientes sobre rangos estandarizados.
        """
        self._compute_src = compute_src
        self._compute_srrc = compute_srrc

    @property
    def method_name(self) -> str:
        return "regression"

    @property
    def requires_variance(self) -> bool:
        """Regresión de muestras vs. scores: sin variabilidad la matriz es
        degenerada."""
        return True

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult:
        """Ejecuta las regresiones multivariadas sobre las muestras existentes
        de Monte Carlo.

        Detecta componentes sin variación analítica (std=0) y fija su coeficiente
        en 0.0 por convención para prevenir división por cero. is_reliable queda
        en False para esos componentes, ya que ese 0.0 no representa una ausencia
        real de efecto, sino la imposibilidad de estimarlo con esa muestra.
        """
        if component_samples is None or lca_scores.size == 0:
            raise ValueError(
                "El analisis de regresion requiere matrices validas de muestras "
                "y de scores."
            )

        components = list(component_samples.keys())
        n_samples = len(lca_scores)

        # 1. Construccion de matrices de diseno consolidadas en NumPy
        X_raw = np.column_stack([component_samples[c] for c in components])
        y_raw = np.asarray(lca_scores, dtype=np.float64)

        # 2. Guard de seguridad contra componentes invariantes
        #    (desviacion estandar nula)
        std_X = np.std(X_raw, axis=0, ddof=1)
        zero_var_mask = std_X == 0
        std_X[zero_var_mask] = 1.0  # Aislar y neutralizar el denominador

        src_coefs = np.zeros(len(components))
        srrc_coefs = np.zeros(len(components))
        r2_src = 0.0
        r2_srrc = 0.0

        # 3. Regresion estandarizada OLS (SRC)
        if self._compute_src:
            X_std = (X_raw - np.mean(X_raw, axis=0)) / std_X
            y_std = (y_raw - np.mean(y_raw)) / (np.std(y_raw, ddof=1) or 1.0)
            src_coefs, r2_src = _ols_vectorized(X_std, y_std)

        # 4. Regresion de rangos estandarizada OLS (SRRC)
        if self._compute_srrc:
            X_rank = np.apply_along_axis(rankdata, 0, X_raw)
            y_rank = rankdata(y_raw)

            std_X_rank = np.std(X_rank, axis=0, ddof=1)
            std_X_rank[std_X_rank == 0] = 1.0  # Proteccion sobre la matriz transmutada

            X_rank_std = (X_rank - np.mean(X_rank, axis=0)) / std_X_rank
            y_rank_std = (y_rank - np.mean(y_rank)) / (np.std(y_rank, ddof=1) or 1.0)
            srrc_coefs, r2_srrc = _ols_vectorized(X_rank_std, y_rank_std)

        r2_model = r2_srrc if self._compute_srrc else r2_src

        # 5. Ensamblaje e inyeccion inmutable de resultados
        raw_results: list[RegressionResult] = []

        for i, comp in enumerate(components):
            is_zero_var = bool(zero_var_mask[i])
            r = RegressionResult(
                component=comp,
                r2_model=r2_model,
                n=n_samples,
                is_reliable=not is_zero_var,
                reliability_note=(
                    "Componente sin variabilidad en la muestra Montecarlo (std=0); "
                    "el coeficiente se fija en 0.0 por convencion y no representa "
                    "ausencia real de efecto, sino imposibilidad de estimarlo."
                    if is_zero_var
                    else ""
                ),
            )

            if self._compute_src:
                r = replace(r, src=float(src_coefs[i]))
            if self._compute_srrc:
                r = replace(r, srrc=float(srrc_coefs[i]))

            raw_results.append(r)

        raw_results.sort(key=lambda item: item.abs_primary, reverse=True)

        unified_scores: list[ComponentSensitivityScore] = []
        for item in raw_results:
            unified_scores.extend(item.to_component_scores())

        return AnalyzerResult(
            method_name=self.method_name,
            scores=unified_scores,
            raw_results=raw_results,
        )
