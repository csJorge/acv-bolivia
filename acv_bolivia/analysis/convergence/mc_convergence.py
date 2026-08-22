"""
analysis.convergence.mc_convergence: Diagnósticos de convergencia de Monte Carlo.

Este módulo evalúa si el número de iteraciones (N) utilizado en una simulación
de Monte Carlo es suficiente, o si los estadísticos (media, IC95, CV) aún
presentan variabilidad significativa debido al azar del muestreo.

Es un módulo de solo lectura: ninguna función ejecuta cálculos LCA nuevos,
modifica un runner existente o altera valores almacenados. Solo consume
secuencias de scores y devuelve diagnósticos estructurados.

Diagnósticos implementados:
    1. running_statistics: Estabilidad de la media y el CV acumulados.
    2. monte_carlo_standard_error: Precisión numérica de la estimación de la media.
    3. split_half_test: Comparación estadística entre la primera y segunda mitad.
    4. seed_comparison: Consistencia entre múltiples corridas con distintas semillas.
    5. percentile_convergence: Estabilidad de los percentiles de las colas
       (p2.5, p97.5).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...analysis.statistical_tests.ks_tests import KSResult, run_ks_test


def _checkpoints(n: int, n_checkpoints: int, min_checkpoint: int) -> list[int]:
    """Genera índices de corte crecientes y únicos entre min_checkpoint y n.

    Utiliza un espaciado logarítmico para proporcionar mayor resolución al
    inicio de la simulación (donde las estadísticas cambian rápidamente) y
    menor resolución al final (donde deberían estabilizarse).

    Parameters
    ----------
    n : int
        Número total de iteraciones disponibles.
    n_checkpoints : int
        Número aproximado de puntos de corte a generar.
    min_checkpoint : int
        Índice mínimo para el primer punto de corte.

    Returns
    -------
    list[int]
        Lista ordenada de índices de corte únicos.
    """
    if n <= min_checkpoint:
        return [n]

    raw = np.unique(
        np.round(
            np.logspace(math.log10(min_checkpoint), math.log10(n), num=n_checkpoints)
        ).astype(int)
    )
    raw = raw[(raw >= min_checkpoint) & (raw <= n)]

    if raw.size == 0 or raw[-1] != n:
        raw = np.append(raw, n)

    return sorted({int(x) for x in raw})


def _cv(arr: np.ndarray) -> float | None:
    """Calcula el coeficiente de variación (desviación estándar / |media|).

    Parameters
    ----------
    arr : np.ndarray
        Array unidimensional de valores numéricos.

    Returns
    -------
    Optional[float]
        Coeficiente de variación,
        None si la media es cero o el array tiene menos de 2 elementos.
    """
    if arr.size < 2:
        return None

    mean = float(np.mean(arr))
    if mean == 0:
        return None

    std = float(np.std(arr, ddof=1))
    return std / abs(mean)


def _relative_diff(a: float, b: float) -> float:
    """Calcula la diferencia relativa |a - b| / |b| con protección contra
    división por cero.

    Parameters
    ----------
    a : float
        Valor estimado.
    b : float
        Valor de referencia.

    Returns
    -------
    float
        Diferencia relativa, o infinito si b es cero y a no lo es.
    """
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b)


def _find_stabilization_point(
    checkpoints: list[int],
    track: list[float],
    reference: float,
    tolerance: float,
) -> int | None:
    """Identifica el punto a partir del cual una serie se mantiene dentro de
    una tolerancia.

    Escanea hacia atrás desde el final de la serie para encontrar el primer
    punto que cumple la condición de estabilidad de forma permanente.

    Parameters
    ----------
    checkpoints : list[int]
        Lista de índices evaluados.
    track : list[float]
        Valores de la métrica en cada checkpoint.
    reference : float
        Valor de referencia (generalmente el valor final de la serie).
    tolerance : float
        Tolerancia relativa máxima permitida.

    Returns
    -------
    int | None
        El índice del checkpoint donde se estabilizó, o None si nunca se
        cumple la condición.
    """
    stabilized_idx = None
    for i in range(len(checkpoints) - 1, -1, -1):
        if _relative_diff(track[i], reference) <= tolerance:
            stabilized_idx = i
        else:
            break

    if stabilized_idx is None:
        return None
    return checkpoints[stabilized_idx]


@dataclass
class RunningStatsResult:
    """Resultado de la evolución de la media y el CV acumulados.

    Attributes
    ----------
    iterations : list[int]
        Puntos de corte evaluados.
    running_mean : list[float]
        Media acumulada en cada punto de corte.
    running_cv : list[Optional[float]]
        Coeficiente de variación acumulado en cada punto de corte.
    final_mean : float
        Media calculada con todas las iteraciones.
    final_cv : Optional[float]
        Coeficiente de variación calculado con todas las iteraciones.
    stabilized_at : Optional[int]
        Iteración a partir de la cual la media se estabilizó dentro de la
        tolerancia definida.
    tolerance : float
        Tolerancia relativa utilizada para el criterio de estabilización.
    n_total : int
        Número total de iteraciones válidas.
    """

    iterations: list[int]
    running_mean: list[float]
    running_cv: list[float | None]
    final_mean: float
    final_cv: float | None
    stabilized_at: int | None
    tolerance: float
    n_total: int

    @property
    def converged(self) -> bool:
        """Indica si la media se estabilizó antes del 90% de las iteraciones totales."""
        if self.stabilized_at is None:
            return False
        return self.stabilized_at <= 0.9 * self.n_total

    def summary(self) -> str:
        pct = (100 * self.stabilized_at / self.n_total) if self.stabilized_at else None
        cv_txt = f"{self.final_cv:.4f}" if self.final_cv is not None else "N/D"

        return (
            f"Media final: {self.final_mean:.6g} | CV final: {cv_txt}\n"
            f"Estabilizada desde la iteración {self.stabilized_at}"
            + (f" ({pct:.0f}% del total de {self.n_total})" if pct else "")
            + f"\nConvergencia {'OK' if self.converged else 'CUESTIONABLE'} "
            f"(criterio: estabilizar antes del 90% de las iteraciones)."
        )


def running_statistics(
    scores: Sequence[float],
    n_checkpoints: int = 50,
    tolerance: float = 0.02,
    min_checkpoint: int = 10,
) -> RunningStatsResult:
    """Calcula la media y el CV acumulados a lo largo de la simulación.

    Parameters
    ----------
    scores : Sequence[float]
        Secuencia de scores obtenidos de una simulación de Monte Carlo.
    n_checkpoints : int, optional
        Número de puntos de corte a calcular (espaciado logarítmico).
        Por defecto es 50.
    tolerance : float, optional
        Tolerancia relativa para declarar estabilización. Por defecto es 0.02 (2%).
    min_checkpoint : int, optional
        Número mínimo de iteraciones para comenzar a evaluar la convergencia.
        Por defecto es 10.

    Returns
    -------
    RunningStatsResult
        Objeto con las curvas de convergencia y el punto de estabilización.

    Raises
    ------
    ValueError
        Si el número de iteraciones válidas es menor que `min_checkpoint`.
    """
    arr = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    n = arr.size

    if n < min_checkpoint:
        raise ValueError(
            f"Se necesitan al menos {min_checkpoint} iteraciones válidas "
            f"para evaluar convergencia; se recibieron {n}."
        )

    checkpoints = _checkpoints(n, n_checkpoints, min_checkpoint)
    running_mean: list[float] = []
    running_cv: list[float | None] = []

    for c in checkpoints:
        sub = arr[:c]
        running_mean.append(float(np.mean(sub)))
        running_cv.append(_cv(sub))

    final_mean = running_mean[-1]
    final_cv = running_cv[-1]

    stabilized_at = _find_stabilization_point(
        checkpoints, running_mean, final_mean, tolerance
    )

    return RunningStatsResult(
        iterations=checkpoints,
        running_mean=running_mean,
        running_cv=running_cv,
        final_mean=final_mean,
        final_cv=final_cv,
        stabilized_at=stabilized_at,
        tolerance=tolerance,
        n_total=n,
    )


@dataclass
class MCSEResult:
    """Resultado del cálculo del Error Estándar de Monte Carlo (MCSE).

    Attributes
    ----------
    n : int
        Número de iteraciones utilizadas.
    mean : float
        Media muestral.
    std : float
        Desviación estándar muestral (con ddof=1).
    mcse : float
        Error estándar de Monte Carlo (std / sqrt(n)).
    relative_mcse : Optional[float]
        MCSE relativo (mcse / |mean|). None si la media es cero.
    n_required_for : dict[float, int]
        Diccionario que mapea precisiones objetivo al número de iteraciones
        necesarias para alcanzarlas.
    """

    n: int
    mean: float
    std: float
    mcse: float
    relative_mcse: float | None
    n_required_for: dict[float, int]

    def summary(self) -> str:
        rel_txt = (
            f"{self.relative_mcse * 100:.3f}%"
            if self.relative_mcse is not None
            else "N/D"
        )
        lines = [
            f"N={self.n} | media={self.mean:.6g} | std={self.std:.6g}",
            f"MCSE={self.mcse:.6g} | MCSE relativo={rel_txt}",
        ]

        if self.n_required_for:
            lines.append("N necesario para cada precisión objetivo:")
            for prec, n_req in sorted(self.n_required_for.items()):
                flag = (
                    "(ya alcanzado)"
                    if n_req <= self.n
                    else f"(faltan {n_req - self.n})"
                )
                lines.append(f"  +/- {prec * 100:.1f}%: N={n_req} {flag}")

        return "\n".join(lines)


def monte_carlo_standard_error(
    scores: Sequence[float],
    target_precisions: Sequence[float] = (0.01, 0.02, 0.05),
) -> MCSEResult:
    """Calcula el error estándar de Monte Carlo (MCSE) de la media.

    Parameters
    ----------
    scores : Sequence[float]
        Secuencia de scores de una simulación de Monte Carlo.
    target_precisions : Sequence[float] | None
        Secuencia de precisiones relativas objetivo a evaluar.
        Por defecto es (0.01, 0.02, 0.05).

    Returns
    -------
    MCSEResult
        Objeto con el MCSE actual y el número de iteraciones necesarias
        para cada precisión objetivo.

    Raises
    ------
    ValueError
        Si hay menos de 2 iteraciones válidas.
    """
    arr = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    n = arr.size

    if n < 2:
        raise ValueError("Se necesitan al menos 2 iteraciones válidas.")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    mcse = std / math.sqrt(n)
    relative_mcse = (mcse / abs(mean)) if mean != 0 else None

    n_required: dict[float, int] = {}
    if mean != 0 and std > 0:
        for prec in target_precisions:
            n_req = (std / (prec * abs(mean))) ** 2
            n_required[prec] = max(1, math.ceil(n_req))

    return MCSEResult(
        n=n,
        mean=mean,
        std=std,
        mcse=mcse,
        relative_mcse=relative_mcse,
        n_required_for=n_required,
    )


@dataclass
class SplitHalfResult:
    """Resultado de la comparación estadística entre dos mitades de una simulación.

    Attributes
    ----------
    ks_result : KSResult
        Resultado completo del test de Kolmogorov-Smirnov.
    split_mode : str
        Modo de división utilizado ("interleaved" o "sequential").
    n_half_a : int
        Número de elementos en la primera mitad.
    n_half_b : int
        Número de elementos en la segunda mitad.
    """

    ks_result: KSResult
    split_mode: str
    n_half_a: int
    n_half_b: int

    @property
    def converged(self) -> bool:
        """Indica si el tamaño del efecto es trivial (d de Cohen < 0.2)."""
        return self.ks_result.cohens_d < 0.2

    @property
    def significant_but_trivial(self) -> bool:
        """Indica si el p-valor es significativo pero el efecto es trivial."""
        return self.ks_result.significant and self.ks_result.cohens_d < 0.2

    def summary(self) -> str:
        veredicto = (
            "CONVERGIÓ"
            if self.converged
            else "NO CONVERGIÓ (efecto no trivial, d >= 0.2)"
        )
        nota = (
            "\nNota: p < alpha pero d de Cohen indica efecto trivial - "
            "con este N, el test KS es sensible a diferencias sin relevancia "
            "práctica; prevalece el tamaño del efecto."
            if self.significant_but_trivial
            else ""
        )

        return (
            f"División: {self.split_mode} (n_a={self.n_half_a}, n_b={self.n_half_b})\n"
            f"D={self.ks_result.d_statistic:.4f} | p={self.ks_result.p_value:.4f} | "
            f"d de Cohen={self.ks_result.cohens_d:.4f} | "
            f"solapamiento={self.ks_result.overlap_index:.4f}\n"
            f"Veredicto: {veredicto}{nota}"
        )


def split_half_test(
    scores: Sequence[float],
    method_name: str,
    project_name: str = "simulacion",
    split_mode: str = "interleaved",
    alpha: float = 0.05,
) -> SplitHalfResult:
    """Compara dos mitades de la misma simulación utilizando el test de KS.

    Parameters
    ----------
    scores : Sequence[float]
        Secuencia de scores de una simulación de Monte Carlo.
    method_name : str
        Nombre del método de impacto evaluado.
    project_name : str, optional
        Nombre del proyecto o simulación. Por defecto es "simulacion".
    split_mode : str, optional
        Modo de división: "interleaved" (pares vs impares) o "sequential"
        (primera mitad vs segunda mitad). Por defecto es "interleaved".
    alpha : float, optional
        Nivel de significancia para el test de KS. Por defecto es 0.05.

    Returns
    -------
    SplitHalfResult
        Objeto con el resultado del test y el veredicto de convergencia.

    Raises
    ------
    ValueError
        Si hay menos de 10 iteraciones válidas o si `split_mode` no es válido.
    """
    arr = [s for s in scores if math.isfinite(s)]
    n = len(arr)

    if n < 10:
        raise ValueError(
            "Se necesitan al menos 10 iteraciones válidas para dividir en mitades."
        )

    if split_mode == "interleaved":
        half_a = arr[0::2]
        half_b = arr[1::2]
        label_a, label_b = "iteraciones pares", "iteraciones impares"
    elif split_mode == "sequential":
        mid = n // 2
        half_a = arr[:mid]
        half_b = arr[mid:]
        label_a, label_b = "primera mitad", "segunda mitad"
    else:
        raise ValueError('split_mode debe ser "interleaved" o "sequential"')

    ks = run_ks_test(
        half_a,
        half_b,
        project_a=f"{project_name} ({label_a})",
        project_b=f"{project_name} ({label_b})",
        method_name=method_name,
        alpha=alpha,
    )

    return SplitHalfResult(
        ks_result=ks,
        split_mode=split_mode,
        n_half_a=len(half_a),
        n_half_b=len(half_b),
    )


@dataclass
class SeedComparisonResult:
    """Resultado de la comparación entre múltiples corridas con distintas semillas.

    Attributes
    ----------
    labels : list[str]
        Etiquetas identificativas de cada corrida.
    pairwise_results : list[KSResult]
        Lista de resultados del test de KS para cada par de corridas comparado.
    all_consistent : bool
        True si ningún par mostró un tamaño de efecto no trivial (d >= 0.2).
    max_cohens_d : float
        Mayor tamaño de efecto observado entre cualquier par de corridas.
    """

    labels: list[str]
    pairwise_results: list[KSResult]
    all_consistent: bool
    max_cohens_d: float

    def summary(self) -> str:
        lines = [f"Corridas comparadas: {', '.join(self.labels)}"]

        for ks in self.pairwise_results:
            trivial = ks.cohens_d < 0.2
            flag = "OK" if trivial else "DIVERGE"
            nota = (
                " (p<alpha pero efecto trivial: prevalece d)"
                if (ks.significant and trivial)
                else ""
            )
            lines.append(
                f"  {ks.project_a} vs {ks.project_b}: "
                f"p={ks.p_value:.4f}, d={ks.cohens_d:.4f} [{flag}]{nota}"
            )

        veredicto = (
            "CONSISTENTE ENTRE SEMILLAS"
            if self.all_consistent
            else "INCONSISTENTE ENTRE SEMILLAS"
        )
        lines.append(
            f"Veredicto: {veredicto} (peor d de Cohen = {self.max_cohens_d:.4f})"
        )

        return "\n".join(lines)


def seed_comparison(
    score_lists: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    method_name: str = "",
    alpha: float = 0.05,
) -> SeedComparisonResult:
    """Compara dos o más corridas independientes de la misma simulación.

    Parameters
    ----------
    score_lists : Sequence[Sequence[float]]
        Lista de secuencias de scores, una por cada corrida (mínimo 2).
    labels : Optional[Sequence[str]], optional
        Etiquetas para cada corrida. Si es None, se generan automáticamente.
    method_name : str, optional
        Nombre del método de impacto evaluado.
    alpha : float, optional
        Nivel de significancia para cada test de KS. Por defecto es 0.05.

    Returns
    -------
    SeedComparisonResult
        Objeto con todas las comparaciones por pares y el veredicto de consistencia.

    Raises
    ------
    ValueError
        Si hay menos de 2 corridas o si la longitud de `labels` no coincide
        con la de `score_lists`.
    """
    if len(score_lists) < 2:
        raise ValueError("Se necesitan al menos 2 corridas para comparar.")

    if labels is None:
        labels = [f"corrida_{i+1}" for i in range(len(score_lists))]

    labels_list = list(labels)
    if len(labels_list) != len(score_lists):
        raise ValueError("labels debe tener el mismo largo que score_lists.")

    pairwise: list[KSResult] = []
    for i in range(len(score_lists)):
        for j in range(i + 1, len(score_lists)):
            ks = run_ks_test(
                list(score_lists[i]),
                list(score_lists[j]),
                project_a=labels_list[i],
                project_b=labels_list[j],
                method_name=method_name,
                alpha=alpha,
            )
            pairwise.append(ks)

    all_consistent = all(ks.cohens_d < 0.2 for ks in pairwise)
    max_d = max((ks.cohens_d for ks in pairwise), default=0.0)

    return SeedComparisonResult(
        labels=labels_list,
        pairwise_results=pairwise,
        all_consistent=all_consistent,
        max_cohens_d=max_d,
    )


@dataclass
class PercentileConvergenceResult:
    """Resultado de la evolución de percentiles específicos a lo largo de la simulación.

    Attributes
    ----------
    iterations : list[int]
        Puntos de corte evaluados.
    percentile_tracks : dict[float, list[float]]
        Diccionario que mapea cada percentil a su lista de valores en cada checkpoint.
    final_values : dict[float, float]
        Diccionario con el valor final de cada percentil.
    stabilized_at : dict[float, Optional[int]]
        Diccionario con la iteración de estabilización de cada percentil.
    tolerance : float
        Tolerancia relativa utilizada para el criterio de estabilización.
    n_total : int
        Número total de iteraciones válidas.
    """

    iterations: list[int]
    percentile_tracks: dict[float, list[float]]
    final_values: dict[float, float]
    stabilized_at: dict[float, int | None]
    tolerance: float
    n_total: int

    @property
    def converged(self) -> bool:
        """Indica si todos los percentiles se estabilizaron antes del 90% de N."""
        if not self.stabilized_at or any(
            v is None for v in self.stabilized_at.values()
        ):
            return False
        return all(
            v is not None and v <= 0.9 * self.n_total
            for v in self.stabilized_at.values()
        )

    def summary(self) -> str:
        lines = []
        for p in sorted(self.final_values):
            stab = self.stabilized_at.get(p)
            pct = f"{100 * stab / self.n_total:.0f}%" if stab else "N/D"
            lines.append(
                f"  p{p}: valor final={self.final_values[p]:.6g}, "
                f"estabilizado desde iter. {stab} ({pct} de N)"
            )

        veredicto = "OK" if self.converged else "CUESTIONABLE"
        return (
            "Convergencia de percentiles:\n"
            + "\n".join(lines)
            + f"\nVeredicto: {veredicto}"
        )


def percentile_convergence(
    scores: Sequence[float],
    percentiles: Sequence[float] = (2.5, 97.5),
    n_checkpoints: int = 50,
    tolerance: float = 0.05,
    min_checkpoint: int = 30,
) -> PercentileConvergenceResult:
    """Calcula la evolución de percentiles específicos a lo largo de la simulación.

    Parameters
    ----------
    scores : Sequence[float]
        Secuencia de scores de una simulación de Monte Carlo.
    percentiles : Sequence[float], optional
        Percentiles a rastrear (0-100). Por defecto es (2.5, 97.5).
    n_checkpoints : int, optional
        Resolución de la curva (espaciado logarítmico). Por defecto es 50.
    tolerance : float, optional
        Tolerancia relativa para declarar estabilización. Por defecto es 0.05 (5%).
    min_checkpoint : int, optional
        Mínimo de iteraciones antes de evaluar convergencia. Por defecto es 30.

    Returns
    -------
    PercentileConvergenceResult
        Objeto con las curvas y puntos de estabilización de cada percentil.

    Raises
    ------
    ValueError
        Si el número de iteraciones válidas es menor que `min_checkpoint`.
    """
    arr = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    n = arr.size

    if n < min_checkpoint:
        raise ValueError(
            f"Se necesitan al menos {min_checkpoint} iteraciones válidas "
            f"para evaluar convergencia de percentiles; se recibieron {n}."
        )

    checkpoints = _checkpoints(n, n_checkpoints, min_checkpoint)
    tracks: dict[float, list[float]] = {p: [] for p in percentiles}

    for c in checkpoints:
        sub = arr[:c]
        for p in percentiles:
            tracks[p].append(float(np.percentile(sub, p, method="linear")))

    final_values = {p: tracks[p][-1] for p in percentiles}
    stabilized_at = {
        p: _find_stabilization_point(checkpoints, tracks[p], final_values[p], tolerance)
        for p in percentiles
    }

    return PercentileConvergenceResult(
        iterations=checkpoints,
        percentile_tracks=tracks,
        final_values=final_values,
        stabilized_at=stabilized_at,
        tolerance=tolerance,
        n_total=n,
    )


@dataclass
class ConvergenceReport:
    """Reporte agregado de los cinco diagnósticos de convergencia.

    Attributes
    ----------
    project_name : str
        Nombre del proyecto analizado.
    method_name : str
        Nombre del método de impacto analizado.
    n_iterations : int
        Número de iteraciones de la simulación evaluada.
    running_stats : RunningStatsResult
        Resultado del diagnóstico de media y CV acumulados.
    mcse : MCSEResult
        Resultado del diagnóstico de error estándar de Monte Carlo.
    split_half : SplitHalfResult
        Resultado del diagnóstico de comparación de mitades.
    percentiles : PercentileConvergenceResult
        Resultado del diagnóstico de convergencia de percentiles.
    seed_comp : Optional[SeedComparisonResult]
        Resultado del diagnóstico de comparación entre semillas, o None si
        no se proveyeron corridas adicionales.
    """

    project_name: str
    method_name: str
    n_iterations: int
    running_stats: RunningStatsResult
    mcse: MCSEResult
    split_half: SplitHalfResult
    percentiles: PercentileConvergenceResult
    seed_comp: SeedComparisonResult | None = None

    @property
    def is_converged(self) -> bool:
        """Veredicto agregado: requiere que todos los chequeos disponibles
        indiquen convergencia."""
        checks = [
            self.running_stats.converged,
            self.split_half.converged,
            self.percentiles.converged,
        ]
        if self.seed_comp is not None:
            checks.append(self.seed_comp.all_consistent)
        return all(checks)

    def summary(self) -> str:
        sep = "-" * 70
        parts = [
            sep,
            f"DIAGNOSTICO DE CONVERGENCIA - {self.project_name} / {self.method_name}",
            f"N = {self.n_iterations} iteraciones",
            sep,
            "[1] Media y CV acumulados:",
            self.running_stats.summary(),
            "",
            "[2] Error estandar de Monte Carlo:",
            self.mcse.summary(),
            "",
            "[3] Comparacion de mitades:",
            self.split_half.summary(),
            "",
            "[4] Convergencia de percentiles:",
            self.percentiles.summary(),
        ]

        if self.seed_comp is not None:
            parts.extend(
                ["", "[5] Comparacion entre semillas:", self.seed_comp.summary()]
            )
        else:
            parts.extend(
                [
                    "",
                    (
                        "[5] Comparacion entre semillas: NO EVALUADA "
                        "(no se proveyeron corridas adicionales)"
                    ),
                ]
            )

        veredicto = "CONVERGIO" if self.is_converged else "REVISAR - ver detalle arriba"
        parts.extend(
            [
                sep,
                f"VEREDICTO GLOBAL: {veredicto}",
                sep,
            ]
        )

        return "\n".join(parts)


def run_convergence_diagnostics(
    scores: Sequence[float],
    project_name: str,
    method_name: str,
    seed_runs: Sequence[Sequence[float]] | None = None,
    seed_labels: Sequence[str] | None = None,
    mean_tolerance: float = 0.02,
    percentile_tolerance: float = 0.05,
    mcse_target_precisions: Sequence[float] = (0.01, 0.02, 0.05),
) -> ConvergenceReport:
    """Ejecuta los cinco diagnósticos de convergencia sobre una simulación
    de Monte Carlo.

    Parameters
    ----------
    scores : Sequence[float]
        Lista de scores de la simulación de Monte Carlo principal.
    project_name : str
        Nombre del proyecto para identificar el reporte.
    method_name : str
        Nombre del método de impacto evaluado.
    seed_runs : Optional[Sequence[Sequence[float]]], optional
        Lista de listas de scores de corridas adicionales con otra semilla.
        Si es None, este diagnóstico se omite.
    seed_labels : Optional[Sequence[str]], optional
        Etiquetas para las corridas adicionales.
    mean_tolerance : float, optional
        Tolerancia para el diagnóstico de media acumulada. Por defecto es 0.02.
    percentile_tolerance : float, optional
        Tolerancia para el diagnóstico de percentiles. Por defecto es 0.05.
    mcse_target_precisions : Sequence[float], optional
        Precisiones objetivo para el cálculo de iteraciones requeridas del MCSE.
        Por defecto es (0.01, 0.02, 0.05).

    Returns
    -------
    ConvergenceReport
        Reporte con los cinco diagnósticos y un veredicto agregado.
    """
    running = running_statistics(scores, tolerance=mean_tolerance)
    mcse = monte_carlo_standard_error(scores, target_precisions=mcse_target_precisions)
    split = split_half_test(scores, method_name=method_name, project_name=project_name)
    pctile = percentile_convergence(scores, tolerance=percentile_tolerance)

    seed_result = None
    if seed_runs:
        all_runs = [list(scores)] + [list(s) for s in seed_runs]
        labels = ["principal"] + list(
            seed_labels or [f"seed_{i+1}" for i in range(len(seed_runs))]
        )
        seed_result = seed_comparison(all_runs, labels=labels, method_name=method_name)

    return ConvergenceReport(
        project_name=project_name,
        method_name=method_name,
        n_iterations=len([s for s in scores if math.isfinite(s)]),
        running_stats=running,
        mcse=mcse,
        split_half=split,
        percentiles=pctile,
        seed_comp=seed_result,
    )
