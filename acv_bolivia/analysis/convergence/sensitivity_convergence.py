"""
analysis.convergence.sensitivity_convergence: Confiabilidad del análisis
de sensibilidad.

Responde tres preguntas distintas de las que resuelve mc_convergence.py
(que evalúa la simulación Monte Carlo en sí):

  1. ¿Los índices de Sobol/Morris que ya calculé son fiables con el N
     de muestras/trayectorias que usé? - usa SobolResult.is_reliable y
     MorrisResult.is_reliable, agregados aquí en un resumen por lote.
  2. ¿El ranking de importancia de componentes es estable si aumento el
     número de muestras/trayectorias, o cambia de forma sustancial?
  3. ¿La aproximación lineal PIV (PIVMonteCarloRunner) da resultados
     compatibles con el modo Monte Carlo completo para el mismo caso,
     o introduce un sesgo relevante?

Módulo de solo lectura: no ejecuta cálculos LCA nuevos por sí mismo. Las
funciones (1) y (3) operan sobre resultados ya calculados; la función (2)
también opera sobre resultados ya calculados en distintas corridas (el
usuario decide cuántas corridas a qué N generar, ver ejemplos en cada función).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import spearmanr

from ...analysis.statistical_tests.ks_tests import KSResult, run_ks_test

# ---------------------------------------------------------------------------
# 1. Resumen de confiabilidad de un lote de resultados Sobol/Morris
# ---------------------------------------------------------------------------


@dataclass
class ReliabilitySummary:
    """Resumen de confiabilidad de un lote de resultados de Sobol o Morris.

    Atributos:
        method:              "sobol" o "morris" (qué tipo de resultado se resumió).
        total:                Cantidad de resultados evaluados (uno por componente).
        n_unreliable:         Cuántos tienen is_reliable == False.
        unreliable_components: Nombres de los componentes no confiables,
                              con su reliability_note.
    """

    method: str
    total: int
    n_unreliable: int
    unreliable_components: list[tuple[str, str]]  # (componente, nota)

    @property
    def all_reliable(self) -> bool:
        return self.n_unreliable == 0

    def summary(self) -> str:
        if self.all_reliable:
            return (
                f"[{self.method.upper()}] {self.total}/{self.total} índices "
                f"confiables. No se requiere aumentar el muestreo."
            )
        lines = [
            (
                f"[{self.method.upper()}] {self.n_unreliable}/{self.total} "
                f"índice(s) NO confiable(s):"
            )
        ]
        for comp, note in self.unreliable_components:
            lines.append(f"  - {comp}: {note}")
        return "\n".join(lines)


def summarize_sobol_reliability(results: Sequence) -> ReliabilitySummary:
    """Agrega ``.is_reliable`` / ``.reliability_note`` de una lista de SobolResult.

    Parameters
    ----------
    results : Sequence[SobolResult]
        Salida de ``run_sobol()``.

    Returns
    -------
    ReliabilitySummary
        Conteo y detalle de componentes no confiables.

    Examples
    --------
    >>> results = run_sobol(bounds, lca_fn, "climate change", "El Dorado")
    >>> summary = summarize_sobol_reliability(results)
    >>> print(summary.summary())
    >>> if not summary.all_reliable:
    ...     print("Aumentar n_samples y repetir para los componentes listados.")
    """
    unreliable = [
        (r.component, r.reliability_note) for r in results if not r.is_reliable
    ]
    return ReliabilitySummary(
        method="sobol",
        total=len(results),
        n_unreliable=len(unreliable),
        unreliable_components=unreliable,
    )


def summarize_morris_reliability(results: Sequence) -> ReliabilitySummary:
    """Agrega ``.is_reliable`` / ``.reliability_note`` de una lista de MorrisResult.

    Requirements
    ------------
    Cada MorrisResult debe exponer los atributos ``is_reliable`` y
    ``reliability_note``.

    Examples
    --------
    >>> results = run_morris(bounds, lca_fn, "climate change", "El Dorado")
    >>> summary = summarize_morris_reliability(results)
    >>> print(summary.summary())
    """
    unreliable = [
        (r.component, r.reliability_note) for r in results if not r.is_reliable
    ]
    return ReliabilitySummary(
        method="morris",
        total=len(results),
        n_unreliable=len(unreliable),
        unreliable_components=unreliable,
    )


# ---------------------------------------------------------------------------
# 2. Estabilidad del ranking frente al tamaño de muestra
# ---------------------------------------------------------------------------


@dataclass
class RankingComparison:
    """Comparación de dos rankings consecutivos (p.ej. N=20 vs N=40)."""

    label_a: str
    label_b: str
    spearman_rho: float
    spearman_p: float
    top_k: int
    top_k_overlap: float  # fracción [0,1] de componentes en común en el top-k
    common_n: int  # cuántos componentes se pudieron comparar

    @property
    def stable(self) -> bool:
        """True si el orden es consistente (rho >= 0.8) y el top-k casi
        no cambió (overlap >= (k-1)/k, es decir, como máximo 1 componente
        distinto entre los dos top-k)."""
        min_overlap = (self.top_k - 1) / self.top_k if self.top_k > 0 else 1.0
        return self.spearman_rho >= 0.8 and self.top_k_overlap >= min_overlap


@dataclass
class RankingStabilityResult:
    """Resultado de comparar el ranking de importancia de componentes a
    través de corridas con distinto tamaño de muestra (N creciente).

    Atributos:
        labels:       Etiquetas de cada corrida, en el orden dado
                      (normalmente de N más chico a N más grande).
        comparisons:   Una RankingComparison por cada par consecutivo de
                      corridas (labels[i] vs labels[i+1]).
        all_stable:    True si TODAS las comparaciones consecutivas son
                      estables.
    """

    labels: list[str]
    comparisons: list[RankingComparison]
    all_stable: bool

    def summary(self) -> str:
        lines = [f"Corridas comparadas (orden dado): {' -> '.join(self.labels)}"]
        for c in self.comparisons:
            flag = "ESTABLE" if c.stable else "CAMBIA"
            lines.append(
                f"  {c.label_a} -> {c.label_b}: "
                f"Spearman rho={c.spearman_rho:.3f} (p={c.spearman_p:.4f}), "
                f"top-{c.top_k} en común={c.top_k_overlap * 100:.0f}% "
                f"(n_comparables={c.common_n}) [{flag}]"
            )
        veredicto = (
            "RANKING ESTABLE"
            if self.all_stable
            else "RANKING TODAVÍA CAMBIA - considere aumentar N"
        )
        lines.append(f"Veredicto: {veredicto}")
        return "\n".join(lines)


def ranking_stability(
    rankings: Sequence[Sequence[str]],
    labels: Sequence[str] | None = None,
    top_k: int = 5,
) -> RankingStabilityResult:
    """Evalúa si el ranking de importancia de componentes se estabiliza a
    medida que crece el tamaño de muestra (N de Sobol, n_trajectories de
    Morris, o cualquier lista ordenada de nombres de componente).

    Dos métricas independientes por par consecutivo:
      1. Correlación de rangos de Spearman entre los DOS rankings
         completos (sobre los componentes en común): sensible a cambios
         de orden en cualquier parte de la lista.
      2. Solapamiento del top-k: cuántos de los k componentes más
         importantes en una corrida siguen entre los k más importantes en
         la siguiente.

    Parameters
    ----------
    rankings : Sequence[Sequence[str]]
        Cada elemento es una lista de nombres de componente ordenada de más
        a menos importante (p.ej. ``[r.component for r in run_morris(...)]``).
        Se comparan pares consecutivos, por lo que deben entregarse en orden
        de N creciente.
    labels : Sequence[str] | None
        Etiquetas de cada corrida (p.ej. ["N=20", "N=40", "N=80"]). Si no se
        dan, se numeran automáticamente.
    top_k : int, default 5
        Tamaño del top a comparar.

    Returns
    -------
    RankingStabilityResult
        Comparaciones consecutivas y veredicto de estabilidad.

    Examples
    --------
    >>> r20 = run_morris(bounds, lca_fn, "climate change",
    ...                 "El Dorado", n_trajectories=20)
        >>> r40 = run_morris(bounds, lca_fn, "climate change",
        ...                 "El Dorado", n_trajectories=40)
        >>> r80 = run_morris(bounds, lca_fn, "climate change",
        ...                 "El Dorado", n_trajectories=80)
        >>> result = ranking_stability(
        ...     [[r.component for r in r20],
        ...      [r.component for r in r40],
        ...      [r.component for r in r80]],
        ...     labels=["N=20", "N=40", "N=80"],
        ... )
        >>> print(result.summary())
        >>> result.all_stable
        True
    """
    if len(rankings) < 2:
        raise ValueError("Se necesitan al menos 2 rankings para evaluar estabilidad.")
    if labels is None:
        labels = [f"corrida_{i + 1}" for i in range(len(rankings))]
    labels = list(labels)
    if len(labels) != len(rankings):
        raise ValueError("labels debe tener el mismo largo que rankings.")

    comparisons: list[RankingComparison] = []
    for i in range(len(rankings) - 1):
        a, b = list(rankings[i]), list(rankings[i + 1])
        common = [c for c in a if c in set(b)]
        if len(common) < 2:
            raise ValueError(
                f"Los rankings '{labels[i]}' y '{labels[i + 1]}' no comparten "
                f"suficientes componentes en común para comparar (¿son del "
                f"mismo proyecto/método?)."
            )
        rank_a = {c: pos for pos, c in enumerate(a)}
        rank_b = {c: pos for pos, c in enumerate(b)}
        ranks_a_common = [rank_a[c] for c in common]
        ranks_b_common = [rank_b[c] for c in common]

        rho, p = spearmanr(ranks_a_common, ranks_b_common)

        k = min(top_k, len(a), len(b))
        top_a = set(a[:k])
        top_b = set(b[:k])
        overlap = len(top_a & top_b) / k if k > 0 else 1.0

        comparisons.append(
            RankingComparison(
                label_a=labels[i],
                label_b=labels[i + 1],
                spearman_rho=float(rho),
                spearman_p=float(p),
                top_k=k,
                top_k_overlap=overlap,
                common_n=len(common),
            )
        )

    return RankingStabilityResult(
        labels=labels,
        comparisons=comparisons,
        all_stable=all(c.stable for c in comparisons),
    )


# ---------------------------------------------------------------------------
# 3. Validación de la aproximación lineal PIV contra el modo completo
# ---------------------------------------------------------------------------


@dataclass
class PIVValidationResult:
    """Resultado de comparar el modo PIV (aproximación lineal) contra el
    modo Monte Carlo completo, para el mismo (proyecto, método).

    Atributos:
        ks_result:  El KSResult completo de la comparación.
        mean_bias:  (media_PIV - media_completo) / media_completo - sesgo
                   relativo introducido por la linealización, con signo
                   (positivo = PIV sobreestima).
    """

    ks_result: KSResult
    mean_bias: float

    @property
    def piv_is_valid(self) -> bool:
        """True si PIV y el modo completo son estadísticamente compatibles.

        Usa el mismo criterio de tamaño de efecto que en mc_convergence.py:
        d de Cohen < 0.2. Para la justificación de por qué se usa el efecto
        y no el p-valor, ver SplitHalfResult.converged.
        """
        return self.ks_result.cohens_d < 0.2

    def summary(self) -> str:
        veredicto = (
            "PIV VÁLIDO para este caso"
            if self.piv_is_valid
            else "PIV DIVERGE del modo completo: el supuesto de linealidad no se cumple"
        )
        return (
            f"PIV vs. Monte Carlo completo - {self.ks_result.method_name}\n"
            f"Sesgo relativo de la media: {self.mean_bias * 100:+.2f}%\n"
            f"d de Cohen={self.ks_result.cohens_d:.4f} | "
            f"solapamiento={self.ks_result.overlap_index:.4f} | "
            f"p={self.ks_result.p_value:.4f}\n"
            f"Veredicto: {veredicto}"
        )


def validate_piv_against_full(
    piv_scores: Sequence[float],
    full_scores: Sequence[float],
    method_name: str,
    project_name: str = "",
    alpha: float = 0.05,
) -> PIVValidationResult:
    """Compara la distribución de scores del modo PIV (aproximación lineal del
    motor Monte Carlo, ``PIVMonteCarloRunner``) contra la del modo Monte Carlo
    completo (``MonteCarloRunner``), para el mismo proyecto y método.

    Parameters
    ----------
    piv_scores : Sequence[float]
        Scores obtenidos con ``PIVMonteCarloRunner`` (modo PIV).
    full_scores : Sequence[float]
        Scores del modo completo para el MISMO proyecto y método.
    method_name : str
        Nombre del método de impacto (para el reporte).
    project_name : str, default ''
        Nombre del proyecto (para el reporte).
    alpha : float, default 0.05
        Nivel de significancia del test KS.

    Returns
    -------
    PIVValidationResult
        Veredicto y sesgo relativo de la media.

    Examples
    --------
    >>> piv  = manager.get_mc_results("El Dorado", "climate change")  # modo PIV
        >>> full = ...  # scores del modo completo para el mismo caso
        >>> result = validate_piv_against_full(piv[0].scores, full[0].scores,
        ...                                     "climate change", "El Dorado")
        >>> print(result.summary())
        >>> result.piv_is_valid
        True
    """
    ks = run_ks_test(
        list(piv_scores),
        list(full_scores),
        project_a=f"{project_name} (PIV)".strip(),
        project_b=f"{project_name} (completo)".strip(),
        method_name=method_name,
        alpha=alpha,
    )
    mean_bias = (ks.mean_a - ks.mean_b) / ks.mean_b if ks.mean_b != 0 else float("nan")
    return PIVValidationResult(ks_result=ks, mean_bias=mean_bias)
