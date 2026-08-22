"""
analysis.sensitivity.methods.sobol - Indices de Sobol (analisis de varianza global).

Define SobolResult (contenedor inmutable de resultados detallados por
componente) y SobolAnalyzer (implementacion del protocolo SensitivityAnalyzer).

run_sobol() contiene la logica matematica real (matriz de Saltelli, evaluacion
LCA puntual, analisis de varianza) y retorna List[SobolResult] con
s1_conf/st_conf/is_reliable intactos.

Optimizacion algoritmica: SobolAnalyzer aplica un screening previo por
top_k_screening para reducir el numero de componentes evaluados, disminuyendo
el costo de N*(2k+2) a N*(2*top_k+2) evaluaciones LCA sin perder informacion
relevante.

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

# Constantes de control analitico e integridad estadistica
RELIABILITY_CI_RATIO_THRESHOLD: float = 0.5
_NEGLIGIBLE_INDEX_THRESHOLD: float = 0.01


@dataclass(frozen=True)
class SobolResult:
    """Indices de Sobol detallados para un componente especifico.

    Conserva la granularidad matematica requerida para graficos y diagnosticos
    de convergencia, cumpliendo con los requisitos de inmutabilidad del dominio.

    Nota: el contexto (proyecto/método) lo proporciona el SensitivityReport.
    """

    component: str
    s1: float = 0.0
    s1_conf: float = 0.0
    st: float = 0.0
    st_conf: float = 0.0
    n_samples: int = 0

    @property
    def interaction(self) -> float:
        """ST - S1: magnitud de las interacciones con otros parametros."""
        return max(0.0, self.st - self.s1)

    @property
    def abs_primary(self) -> float:
        """Metrica primaria de ordenamiento utilizada por el ecosistema de reportes."""
        return self.st

    @property
    def is_reliable(self) -> bool:
        """False si el indice de Sobol no muestra evidencia suficiente de convergencia.

        Evalua estimadores negativos (artefacto de sub-convergencia de Saltelli)
        y la amplitud del intervalo de confianza bootstrap frente al umbral.
        """
        if self.st < 0 or self.s1 < 0:
            return False
        if (
            self.st > _NEGLIGIBLE_INDEX_THRESHOLD
            and (self.st_conf / self.st) > RELIABILITY_CI_RATIO_THRESHOLD
        ):
            return False
        return not (
            self.s1 > _NEGLIGIBLE_INDEX_THRESHOLD
            and self.s1_conf / self.s1 > RELIABILITY_CI_RATIO_THRESHOLD
        )

    @property
    def reliability_note(self) -> str:
        """Explicacion en texto plano si el indice no ha convergido; cadena
        vacia si sÍ."""
        if self.is_reliable:
            return ""
        if self.st < 0 or self.s1 < 0:
            return (
                f"S1/ST negativo (s1={self.s1:.4g}, st={self.st:.4g}): "
                f"senal de sub-convergencia con n_samples={self.n_samples}. "
                f"Se recomienda aumentar n_samples."
            )
        ratio = (self.st_conf / self.st) if self.st > 0 else float("inf")
        return (
            f"Intervalo de confianza amplio: st_conf/st={ratio:.2f} "
            f"(umbral={RELIABILITY_CI_RATIO_THRESHOLD}). "
            f"Se recomienda aumentar n_samples (actual={self.n_samples})."
        )

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea el resultado detallado a las estructuras genericas del framework.

        Exporta tanto el indice de primer orden (S1) como el total (ST) para
        que el motor de consenso procese ambas metricas si lo requiere.
        """
        return [
            ComponentSensitivityScore(
                component=self.component, score=self.st, metric_name="ST"
            ),
            ComponentSensitivityScore(
                component=self.component, score=self.s1, metric_name="S1"
            ),
        ]


def run_sobol(
    param_bounds: dict[str, tuple[float, float]],
    evaluator: LcaEvaluator,
    n_samples: int = 512,
    calc_second_order: bool = False,
) -> list[SobolResult]:
    """Ejecuta el analisis de varianza de Sobol y retorna los resultados por componente.

    Parameters
    ----------
    param_bounds : Dict[str, Tuple[float, float]]
        {componente: (min, max)} - rango de exploracion de cada parametro.
    evaluator : LcaEvaluator
        Evaluador LCA inyectado (matricial o PIV).
    n_samples : int
        N base de muestras de Saltelli. Debe ser potencia de 2.
        Total de evaluaciones resultantes = N * (2k + 2).
    calc_second_order : bool
        Si es True, calcula interacciones entre pares (S2).

    Returns
    -------
    List[SobolResult]
        Un SobolResult por componente, con S1, ST, sus intervalos de
        confianza e is_reliable ya calculados.

    Raises
    ------
    ImportError
        Si SALib no esta instalado.
    ValueError
        Si param_bounds esta vacio.
    """
    try:
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        raise ImportError(
            "SALib es requerido para ejecutar las simulaciones de Sobol. "
            "Instala la dependencia externa ejecutando: pip install SALib"
        )

    if not param_bounds:
        raise ValueError("No se pudieron determinar los rangos (bounds) para Sobol.")

    components = list(param_bounds.keys())
    problem = build_salib_problem(param_bounds)

    X = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)

    Y = evaluate_lca_batch(X, evaluator, components)

    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=calc_second_order,
        conf_level=0.95,
        print_to_console=False,
    )

    return [
        SobolResult(
            component=comp,
            s1=float(Si["S1"][i]),
            s1_conf=float(Si["S1_conf"][i]),
            st=float(Si["ST"][i]),
            st_conf=float(Si["ST_conf"][i]),
            n_samples=n_samples,
        )
        for i, comp in enumerate(components)
    ]


class SobolAnalyzer:
    """Analizador de Indices de Sobol para el motor generico de sensibilidad.

    Implementa el protocolo SensitivityAnalyzer delegando el calculo matematico
    a run_sobol() y adaptando su salida al formato generico ComponentSensitivityScore.
    Gestiona ademas el filtrado por screening de parametros dominantes antes de
    invocar el muestreo de Saltelli (evita evaluar componentes irrelevantes).
    """

    def __init__(
        self,
        n_samples: int = 512,
        top_k_screening: int = 8,
        calc_second_order: bool = False,
    ) -> None:
        """Configura los hiperparametros estadisticos del algoritmo de Sobol.

        Parameters
        ----------
        n_samples : int
            N base de muestras de Saltelli. Debe ser potencia de 2.
            Total de evaluaciones resultantes = N * (2k + 2).
        top_k_screening : int
            Cantidad maxima de parametros influyentes a aislar del ranking
            de consenso previo antes de evaluar.
        calc_second_order : bool
            Si es True, calcula interacciones entre pares (S2).
        """
        self._n_samples = n_samples
        self._top_k = top_k_screening
        self._calc_second_order = calc_second_order

    @property
    def method_name(self) -> str:
        return "sobol"

    @property
    def requires_variance(self) -> bool:
        """Evalua evaluation_fn en vivo por muestreo de Saltelli; no necesita
        muestras MC previas."""
        return False

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult:
        """Aisla los componentes dominantes por screening y ejecuta run_sobol()
        sobre ellos.

        Interroga al callback del reporte para saber que componentes lideran el
        consenso actual; si no hay analisis previos, cae al fallback de los
        primeros top_k_screening componentes nominales.
        """
        top_comps = top_components_provider(self._top_k)
        if not top_comps:
            top_comps = list(nominal_params.keys())[: self._top_k]

        cs_top = {
            c: component_samples[c]
            for c in top_comps
            if component_samples and c in component_samples
        }
        nominal_top = {c: nominal_params[c] for c in top_comps if c in nominal_params}
        param_bounds = bounds_from_samples(cs_top, nominal_top)

        results = run_sobol(
            param_bounds=param_bounds,
            evaluator=evaluator,
            n_samples=self._n_samples,
            calc_second_order=self._calc_second_order,
        )

        unified_scores: list[ComponentSensitivityScore] = []
        for r in results:
            unified_scores.extend(r.to_component_scores())

        return AnalyzerResult(
            method_name=self.method_name,
            scores=unified_scores,
            raw_results=results,
        )
