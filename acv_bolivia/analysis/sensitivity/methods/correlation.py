"""
analysis.sensitivity.methods.correlation: Correlaciones lineales y de rangos.

Analizador estadístico nativo (Pearson, Spearman y PRCC) a costo cero sobre
simulaciones de Montecarlo ya calculadas.

Implementación vectorizada con SciPy/NumPy para rendimiento óptimo con
N >= 10,000 iteraciones. PRCC calculado vía proyección ortogonal de rangos
(Gram-Schmidt vectorizado).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, rankdata, spearmanr

from ....core.domain.contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
)


@dataclass(frozen=True)
class CorrelationResult:
    """Coeficientes de correlación detallados para un componente específico.

    Nota: El contexto (proyecto/método) lo proporciona el SensitivityReport.
    """

    component: str
    n: int = 0
    pearson_r: float | None = None
    p_value: float | None = None
    spearman_rho: float | None = None
    prcc: float | None = None
    x_values: NDArray | None = None
    y_values: NDArray | None = None

    @property
    def abs_primary(self) -> float:
        """Métrica primaria de ordenamiento: PRCC > Spearman > Pearson."""
        for value in (self.prcc, self.spearman_rho, self.pearson_r):
            if value is not None:
                return abs(value)
        return 0.0

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea los coeficientes calculados a las estructuras genéricas del
        framework."""
        scores: list[ComponentSensitivityScore] = []
        if self.pearson_r is not None:
            scores.append(
                ComponentSensitivityScore(
                    component=self.component,
                    score=self.pearson_r,
                    metric_name="pearson_r",
                )
            )
        if self.spearman_rho is not None:
            scores.append(
                ComponentSensitivityScore(
                    component=self.component,
                    score=self.spearman_rho,
                    metric_name="spearman_rho",
                )
            )
        if self.prcc is not None:
            scores.append(
                ComponentSensitivityScore(
                    component=self.component, score=self.prcc, metric_name="prcc"
                )
            )
        return scores


# ==============================================================================
# Primitivas estadísticas vectorizadas (SciPy/NumPy)
# ==============================================================================


def _pearson_vectorized(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float]:
    """Coeficiente de Pearson r y p-valor bilateral. Vectorizado vía SciPy.

    Cortocircuito nulo: si x o y no tienen variabilidad, retorna (0.0, 1.0).
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0, 1.0

    r, p = pearsonr(x, y)
    # pearsonr puede retornar NaN en casos degenerados
    if np.isnan(r) or np.isnan(p):
        return 0.0, 1.0
    return float(r), float(p)


def _spearman_vectorized(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float]:
    """Coeficiente de Spearman rho y p-valor. Vectorizado vía SciPy."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0, 1.0

    rho, p = spearmanr(x, y)
    if np.isnan(rho) or np.isnan(p):
        return 0.0, 1.0
    return float(rho), float(p)


# ==============================================================================
# PRCC vectorizado (Gram-Schmidt sobre rangos con NumPy)
# ==============================================================================


def _partial_rank_correlation_vectorized(
    x_matrix: NDArray,  # shape: (n_samples, k_components)
    y: NDArray,  # shape: (n_samples,)
    target_idx: int,
) -> float:
    """Calcula el PRCC de la columna ``target_idx`` controlando las demás.

    Implementación vectorizada con NumPy: proyección Gram-Schmidt sobre rangos
    usando ``np.dot`` y operaciones de broadcasting.

    Parameters
    ----------
    x_matrix : np.ndarray
        Matriz de muestras con forma (n_samples, k_components).
    y : np.ndarray
        Vector de scores con forma (n_samples,).
    target_idx : int
        Índice de la variable objetivo en ``x_matrix``.

    Returns
    -------
    float
        Coeficiente PRCC en [-1, 1].
    """
    k = x_matrix.shape[1]

    if k < 2:
        rho, _ = _spearman_vectorized(x_matrix[:, 0], y)
        return rho

    # Rangos con manejo de empates (average rank) vía scipy.rankdata
    rank_y = rankdata(y).astype(np.float64)
    rank_x = np.column_stack(
        [rankdata(x_matrix[:, j]).astype(np.float64) for j in range(k)]
    )

    # Separar variable objetivo y variables de control
    target_x = rank_x[:, target_idx]
    others_x = np.delete(rank_x, target_idx, axis=1)  # shape: (n, k-1)

    # Proyección ortogonal vía mínimos cuadrados (OLS vectorizado)
    # Residuos de y sobre others_x: resid_y = y - others_x @ beta_y
    # Residuos de target_x sobre others_x: resid_x = target_x - others_x @ beta_x
    # beta = (X^T X)^{-1} X^T y  →  residuos = y - X @ beta

    resid_y = _ols_residuals_vectorized(rank_y, others_x)
    resid_x = _ols_residuals_vectorized(target_x, others_x)

    r, _ = _pearson_vectorized(resid_x, resid_y)
    return r


def _ols_residuals_vectorized(
    dependent: np.ndarray,  # shape: (n,)
    predictors: np.ndarray,  # shape: (n, k)
) -> np.ndarray:
    """Calcula residuos OLS vía mínimos cuadrados vectorizados (NumPy).

    Usa np.linalg.lstsq para estabilidad numérica y velocidad.
    """
    # Residuos = dependiente - predictores @ coeficientes
    # lstsq resuelve: predictors @ beta = dependent
    # → beta = pinv(predictors) @ dependent
    beta, _, _, _ = np.linalg.lstsq(predictors, dependent, rcond=None)
    predicted = predictors @ beta
    return cast(np.ndarray, dependent - predicted)


# ==============================================================================
# API pública: implementación del analizador
# ==============================================================================


class CorrelationAnalyzer:
    """Analizador estadístico de coeficientes de correlación lineal y de rangos."""

    def __init__(
        self,
        compute_pearson: bool = True,
        compute_spearman: bool = True,
        compute_prcc: bool = True,
    ) -> None:
        self._compute_pearson = compute_pearson
        self._compute_spearman = compute_spearman
        self._compute_prcc = compute_prcc

    @property
    def method_name(self) -> str:
        return "correlation"

    @property
    def requires_variance(self) -> bool:
        return True

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult:
        """Calcula las correlaciones entre las muestras del inventario y los
        scores LCA."""
        if component_samples is None or lca_scores.size == 0:
            raise ValueError(
                "El análisis de correlación requiere una matriz válida de muestras "
                "y sus respectivos scores."
            )

        n_samples = len(lca_scores)
        components = list(component_samples.keys())
        k = len(components)

        # Construir matriz X (n_samples, k) en memoria contigua
        # Esto es clave para el rendimiento: una sola asignación en lugar de k listas
        x_matrix = np.column_stack(
            [np.asarray(component_samples[c], dtype=np.float64) for c in components]
        )
        y_array = np.asarray(lca_scores, dtype=np.float64)

        raw_results: list[CorrelationResult] = []

        for i, comp in enumerate(components):
            x_vec = x_matrix[:, i]
            if len(x_vec) != n_samples:
                continue

            r = CorrelationResult(
                component=comp,
                n=n_samples,
                x_values=x_vec.copy(),  # copia para inmutabilidad
                y_values=y_array.copy(),
            )

            if self._compute_pearson:
                pearson_val, p_val = _pearson_vectorized(x_vec, y_array)
                r = replace(r, pearson_r=pearson_val, p_value=p_val)

            if self._compute_spearman:
                spearman_val, p_spearman = _spearman_vectorized(x_vec, y_array)
                current_p = r.p_value if r.p_value is not None else p_spearman
                r = replace(r, spearman_rho=spearman_val, p_value=current_p)

            if self._compute_prcc and k >= 2:
                prcc_val = _partial_rank_correlation_vectorized(x_matrix, y_array, i)
                r = replace(r, prcc=prcc_val)

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
