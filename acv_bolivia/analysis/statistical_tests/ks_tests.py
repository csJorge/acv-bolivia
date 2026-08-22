"""
analysis.statistical_tests.ks_tests: Tests de Kolmogorov-Smirnov para
distribuciones de Monte Carlo.

Implementa el test KS de dos muestras (scipy.stats.ks_2samp) complementado con:
    1. Corrección de Bonferroni para controlar la Family-Wise Error Rate en
     comparaciones múltiples.
    2. Cohen's d para cuantificar el tamaño del efecto (independiente del
     tamaño de muestra).
    3. Overlap Index (OVL) para medir el solapamiento real de las distribuciones.

Este módulo es no paramétrico, lo que lo hace ideal para scores de LCA que raramente
siguen una distribución normal.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

from ...application.contracts import MCResultsReader


@dataclass
class KSResult:
    """Resultado del test KS de dos muestras para un par de proyectos.

    Attributes
    ----------
    project_a : str
        Nombre del primer proyecto.
    project_b : str
        Nombre del segundo proyecto.
    method_name : str
        Método de impacto evaluado.
    d_statistic : float
        Estadístico D = sup|F_n - G_m|. Rango [0, 1].
    p_value : float
        P-valor del test. p < alpha indica rechazo de la hipótesis nula.
    significant : bool
        True si p < alpha (distribuciones estadísticamente distintas).
    cohens_d : float
        Tamaño del efecto normalizado.
    overlap_index : float
        Solapamiento de distribuciones en el rango [0, 1].
    n_a : int
        Tamaño de la muestra A.
    n_b : int
        Tamaño de la muestra B.
    mean_a : float
        Media de la distribución A.
    mean_b : float
        Media de la distribución B.
    std_a : float
        Desviación estándar de la distribución A.
    std_b : float
        Desviación estándar de la distribución B.
    alpha : float, optional
        Nivel de significancia utilizado. Por defecto 0.05.
    interpretation : str
        Texto legible resumido para informes o tesis.
    """

    project_a: str
    project_b: str
    method_name: str
    d_statistic: float
    p_value: float
    significant: bool
    cohens_d: float
    overlap_index: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    alpha: float = 0.05
    interpretation: str = ""

    def __post_init__(self) -> None:
        if not self.interpretation:
            sig_txt = "significativa" if self.significant else "no significativa"
            size_txt = self._effect_size_label()
            self.interpretation = (
                f"{self.project_a} vs {self.project_b} | {self.method_name}: "
                f"D={self.d_statistic:.4f}, p={self.p_value:.4f} ({sig_txt}), "
                f"d={self.cohens_d:.3f} ({size_txt}), OVL={self.overlap_index:.3f}"
            )

    def _effect_size_label(self) -> str:
        d = abs(self.cohens_d)
        if d < 0.2:
            return "efecto pequeno"
        elif d < 0.5:
            return "efecto mediano"
        elif d < 0.8:
            return "efecto grande"
        else:
            return "efecto muy grande"


@dataclass
class PairwiseKSResult:
    """Resultados agregados de todos los pares para todos los métodos.

    Attributes
    ----------
    results : list[KSResult]
        Lista de resultados KS por cada par de proyectos y método.
    alpha_raw : float
        Nivel de significancia base sin corrección.
    alpha_bonferroni : float
        Nivel de significancia ajustado por corrección de Bonferroni.
    n_tests : int
        Número total de tests ejecutados.
    n_significant : int
        Número de tests que resultaron estadísticamente significativos.
    """

    results: list[KSResult]
    alpha_raw: float
    alpha_bonferroni: float
    n_tests: int
    n_significant: int

    @property
    def significant_results(self) -> list[KSResult]:
        """Retorna solo los resultados que fueron estadísticamente significativos."""
        return [r for r in self.results if r.significant]

    def summary(self) -> str:
        lines = [
            (
                f"Pairwise KS: {self.n_tests} tests | "
                f"alpha_bonferroni={self.alpha_bonferroni:.4f} | "
                f"Significativos: {self.n_significant}/{self.n_tests}"
            ),
        ]
        for r in self.significant_results:
            lines.append(f"  * {r.interpretation}")
        return "\n".join(lines)

    def as_matrix(self, method_name: str) -> dict | None:
        """Genera una matriz de p-valores y estadísticos D para un método específico.

        Parameters
        ----------
        method_name : str
            Nombre del método de impacto a filtrar.

        Returns
        -------
        Optional[dict]
            Diccionario con las matrices 'p_values', 'd_statistics' y la lista
            'projects',
            o None si no hay resultados para ese método.
        """
        method_results = [r for r in self.results if r.method_name == method_name]
        if not method_results:
            return None

        projects = sorted(
            {r.project_a for r in method_results}
            | {r.project_b for r in method_results}
        )
        idx = {p: i for i, p in enumerate(projects)}
        n = len(projects)

        p_mat = np.ones((n, n))
        d_mat = np.zeros((n, n))

        for r in method_results:
            i, j = idx[r.project_a], idx[r.project_b]
            p_mat[i, j] = p_mat[j, i] = r.p_value
            d_mat[i, j] = d_mat[j, i] = r.d_statistic

        return {
            "projects": projects,
            "p_values": p_mat,
            "d_statistics": d_mat,
            "alpha": self.alpha_bonferroni,
        }


def run_ks_test(
    scores_a: list[float],
    scores_b: list[float],
    project_a: str,
    project_b: str,
    method_name: str,
    alpha: float = 0.05,
) -> KSResult:
    """Ejecuta el test KS de dos muestras entre dos distribuciones de Monte Carlo.

    Parameters
    ----------
    scores_a : list[float]
        Scores de la simulación del proyecto A.
    scores_b : list[float]
        Scores de la simulación del proyecto B.
    project_a : str
        Nombre identificador del proyecto A.
    project_b : str
        Nombre identificador del proyecto B.
    method_name : str
        Nombre del método de impacto LCA evaluado.
    alpha : float, optional
        Nivel de significancia. Por defecto 0.05.

    Returns
    -------
    KSResult
        Objeto con el estadístico D, p-valor, tamaño del efecto e índice de
        solapamiento.
    """
    arr_a = np.asarray([s for s in scores_a if np.isfinite(s)])
    arr_b = np.asarray([s for s in scores_b if np.isfinite(s)])

    if len(arr_a) < 5 or len(arr_b) < 5:
        return KSResult(
            project_a=project_a,
            project_b=project_b,
            method_name=method_name,
            d_statistic=float("nan"),
            p_value=float("nan"),
            significant=False,
            cohens_d=float("nan"),
            overlap_index=float("nan"),
            n_a=len(arr_a),
            n_b=len(arr_b),
            mean_a=float(np.mean(arr_a)) if len(arr_a) else float("nan"),
            mean_b=float(np.mean(arr_b)) if len(arr_b) else float("nan"),
            std_a=float(np.std(arr_a, ddof=1)) if len(arr_a) > 1 else float("nan"),
            std_b=float(np.std(arr_b, ddof=1)) if len(arr_b) > 1 else float("nan"),
            alpha=alpha,
            interpretation="Muestra insuficiente (n < 5).",
        )

    stat, pval = scipy_stats.ks_2samp(arr_a, arr_b)

    mean_a, std_a = float(np.mean(arr_a)), float(np.std(arr_a, ddof=1))
    mean_b, std_b = float(np.mean(arr_b)), float(np.std(arr_b, ddof=1))

    pooled_std = float(np.sqrt((std_a**2 + std_b**2) / 2))
    cohens_d = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    ovl = _overlap_index(arr_a, arr_b)

    return KSResult(
        project_a=project_a,
        project_b=project_b,
        method_name=method_name,
        d_statistic=float(stat),
        p_value=float(pval),
        significant=float(pval) < alpha,
        cohens_d=cohens_d,
        overlap_index=ovl,
        n_a=len(arr_a),
        n_b=len(arr_b),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        alpha=alpha,
    )


def run_pairwise_ks(
    manager: MCResultsReader,
    method_names: list[str] | None = None,
    alpha: float = 0.05,
    generation_dict: dict[str, float] | None = None,
    use_per_kwh: bool = True,
) -> PairwiseKSResult:
    """Ejecuta el test KS para todos los pares de proyectos en los métodos
    especificados.

    Aplica corrección de Bonferroni sobre el número de tests efectivamente ejecutables,
    evitando una corrección excesiva cuando faltan datos para algunas combinaciones.

    Parameters
    ----------
    manager : MCResultsReader
        Interfaz para leer los resultados de Monte Carlo.
    method_names : Optional[list[str]], optional
        Lista de métodos a evaluar. Si es None, evalúa todos los disponibles.
    alpha : float, optional
        Nivel de significancia base. Por defecto 0.05.
    generation_dict : Optional[dict[str, float]], optional
        Diccionario {proyecto: kWh} para normalizar los scores.
    use_per_kwh : bool, optional
        Si True y hay generation_dict, normaliza los scores antes de comparar.

    Returns
    -------
    PairwiseKSResult
        Objeto con todos los tests realizados y las matrices de resultados.
    """
    all_mc = manager.get_mc_results()
    if not all_mc:
        return PairwiseKSResult(
            results=[],
            alpha_raw=alpha,
            alpha_bonferroni=alpha,
            n_tests=0,
            n_significant=0,
        )

    by_proj_method: dict[tuple, list[float]] = defaultdict(list)
    for mc in all_mc:
        key = (mc.project_name, mc.method_name)
        scores = mc.scores
        if use_per_kwh and generation_dict:
            gen = generation_dict.get(mc.project_name, 0)
            if gen and gen > 0:
                scores = [s / gen for s in scores]
        by_proj_method[key].extend(scores)

    projects = sorted({k[0] for k in by_proj_method})
    methods = sorted({k[1] for k in by_proj_method})

    if method_names:
        methods = [m for m in methods if m in method_names]

    runnable_tests = [
        (method, proj_a, proj_b)
        for method in methods
        for proj_a, proj_b in combinations(projects, 2)
        if by_proj_method.get((proj_a, method)) and by_proj_method.get((proj_b, method))
    ]

    n_tests = len(runnable_tests)
    alpha_bonferroni = alpha / n_tests if n_tests > 0 else alpha

    results = [
        run_ks_test(
            scores_a=by_proj_method[(proj_a, method)],
            scores_b=by_proj_method[(proj_b, method)],
            project_a=proj_a,
            project_b=proj_b,
            method_name=method,
            alpha=alpha_bonferroni,
        )
        for method, proj_a, proj_b in runnable_tests
    ]

    n_sig = sum(1 for r in results if r.significant)

    return PairwiseKSResult(
        results=results,
        alpha_raw=alpha,
        alpha_bonferroni=alpha_bonferroni,
        n_tests=n_tests,
        n_significant=n_sig,
    )


def run_overlap_index(
    scores_a: list[float],
    scores_b: list[float],
) -> float:
    """Calcula el índice de solapamiento (OVL) entre dos distribuciones.

    Parameters
    ----------
    scores_a : list[float]
        Scores de la primera distribución.
    scores_b : list[float]
        Scores de la segunda distribución.

    Returns
    -------
    float
        Índice de solapamiento en el rango [0, 1].
    """
    return _overlap_index(
        np.asarray([s for s in scores_a if np.isfinite(s)]),
        np.asarray([s for s in scores_b if np.isfinite(s)]),
    )


def _overlap_index(arr_a: np.ndarray, arr_b: np.ndarray, bins: int = 200) -> float:
    """Calcula el OVL numérico mediante aproximación por histograma conjunto.

    Parameters
    ----------
    arr_a : np.ndarray
        Primera muestra de datos.
    arr_b : np.ndarray
        Segunda muestra de datos.
    bins : int, optional
        Número de intervalos para el histograma. Por defecto 200.

    Returns
    -------
    float
        Valor del índice de solapamiento.
    """
    if len(arr_a) == 0 or len(arr_b) == 0:
        return float("nan")

    lo = min(float(arr_a.min()), float(arr_b.min()))
    hi = max(float(arr_a.max()), float(arr_b.max()))

    if lo >= hi:
        return 1.0

    edges = np.linspace(lo, hi, bins + 1)
    hist_a = np.histogram(arr_a, bins=edges, density=True)[0]
    hist_b = np.histogram(arr_b, bins=edges, density=True)[0]

    bin_width = edges[1] - edges[0]
    ovl = float(np.sum(np.minimum(hist_a, hist_b)) * bin_width)

    return max(0.0, min(1.0, ovl))
