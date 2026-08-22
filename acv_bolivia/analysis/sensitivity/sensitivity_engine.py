"""
analysis.sensitivity.sensitivity_engine: Motor de análisis de sensibilidad unificado.

Este módulo orquesta los métodos de sensibilidad disponibles en el framework,
calcula las simulaciones de Montecarlo requeridas a través de la infraestructura
inyectada y consolida un SensitivityReport con ranking de consenso por proyecto.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.domain.contracts import (
    AnalyzerResult,
    LcaEvaluator,
    LCAInfrastructureProvider,
    MethodId,
    SensitivityAnalyzer,
)
from ...core.domain.models import SensitivityReport

logger = logging.getLogger(__name__)


class SensitivityEngine:
    """Motor central de ejecución de análisis de sensibilidad.

    Orquesta la ejecución de métodos polimórficos de forma segura, abstrayendo
    los flujos de control, los tiempos de ejecución y el manejo de excepciones.

    """

    def __init__(
        self,
        lca_provider: LCAInfrastructureProvider,
        analyzers: Sequence[SensitivityAnalyzer],
        exclude_methods: set[str] | None = None,
    ) -> None:
        """Inyecta las dependencias e interfaces abstractas del dominio.

        Parameters
        ----------
        lca_provider : LCAInfrastructureProvider
            Adaptador concreto de cálculo de ciclo de vida.
        analyzers : Sequence[SensitivityAnalyzer]
            Colección de algoritmos matemáticos (plugins) a ejecutar.
        exclude_methods : Optional[Set[str]]
            Conjunto opcional de nombres de métodos a omitir.
        """
        self._lca_provider = lca_provider
        self._analyzers = analyzers
        self._exclude = exclude_methods or set()

    def run(
        self,
        project_name: str,
        method_tuple: MethodId,
        component_samples_raw: dict[str, list[float]] | None = None,
        lca_scores_raw: list[float] | None = None,
    ) -> SensitivityReport:
        """Orquesta el ciclo completo de análisis para un proyecto y método ambiental.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto a evaluar.
        method_tuple : MethodId
            Tupla completa del método de impacto BW2.
        component_samples_raw : dict[str, list[float]] | None
            Muestras Montecarlo por componente. None desactiva
            correlation/regression/shap (requieren variabilidad).
        lca_scores_raw : list[float] | None
            Scores ya calculados de una simulación Montecarlo previa.
            El motor nunca calcula ni decide cómo obtener scores por su cuenta.

        Returns
        -------
        SensitivityReport
            Reporte consolidado con rankings de consenso.
        """
        method_name = method_tuple[1] if len(method_tuple) > 1 else str(method_tuple)
        report = SensitivityReport(project_id=project_name, method_id=method_tuple)

        # 1. Normalización de tipos a arreglos puros de NumPy
        component_samples: dict[str, NDArray] | None = None
        if component_samples_raw is not None:
            component_samples = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in component_samples_raw.items()
            }

        # 2. Construcción de evaluadores (puntual y PIV lineal)
        nominal_params = self._lca_provider.get_nominal_parameters(project_name)
        evaluator = self._lca_provider.create_evaluator(project_name, method_tuple)
        piv_evaluator = self._lca_provider.create_piv_evaluator(nominal_params)
        active_evaluator = piv_evaluator if piv_evaluator is not None else evaluator

        # 3. Diagnóstico de integridad del mapeo matricial
        report.diagnostic = self._lca_provider.get_latest_mapping_diagnostic()

        # 4. Scores ya resueltos por el llamador
        lca_scores = (
            np.asarray(lca_scores_raw, dtype=np.float64)
            if lca_scores_raw is not None
            else np.array([], dtype=np.float64)
        )

        # 5. Despacho polimórfico de los analizadores
        for analyzer in self._analyzers:
            if analyzer.method_name in self._exclude:
                report.skipped_methods.append(analyzer.method_name)
                continue

            result = self._run_safe(
                analyzer=analyzer,
                nominal_params=nominal_params,
                evaluator=active_evaluator,
                lca_scores=lca_scores,
                report=report,
                component_samples=component_samples,
            )
            report.add_result(result)

        logger.info(
            "[%s | %s] Análisis completado: %d métodos ejecutados, %d errores.",
            project_name,
            method_name,
            report.methods_executed_count,
            len(report.errors),
        )

        return report

    def _run_safe(
        self,
        analyzer: SensitivityAnalyzer,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: np.ndarray,
        report: SensitivityReport,
        component_samples: dict[str, np.ndarray] | None,
    ) -> AnalyzerResult:
        """Aísla la ejecución de un algoritmo específico protegiendo el bucle
        principal."""
        t0 = time.time()

        try:
            result = analyzer.execute(
                nominal_params=nominal_params,
                evaluator=evaluator,  # LcaEvaluator en lugar de Callable
                lca_scores=lca_scores,
                top_components_provider=report.top_components,
                component_samples=component_samples,
            )
            result.execution_time_seconds = time.time() - t0
            return result

        except Exception as e:
            logger.exception(
                "Falla en analizador '%s'",
                analyzer.method_name,
            )
            return AnalyzerResult(
                method_name=analyzer.method_name,
                execution_time_seconds=time.time() - t0,
                error_message=(
                    f"Falla matemática o de dependencia: " f"{type(e).__name__}: {e}"
                ),
            )
