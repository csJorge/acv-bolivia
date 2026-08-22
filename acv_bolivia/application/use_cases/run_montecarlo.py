"""
application.use_cases.run_montecarlo - Caso de uso: simulación Monte Carlo.

Orquesta los modos de simulación Monte Carlo (BW, Foreground, PIV) y acumula
los resultados. No conoce implementaciones concretas de infraestructura,
dependiendo exclusivamente de abstracciones (Protocols) mediante Inversión
de Dependencias.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...application.contracts import (
    ForegroundSimulationStrategy,
    MCStatsCalculationStrategy,
    MethodFilteringStrategy,
    MonteCarloSimulationStrategy,
    PIVSimulationStrategy,
    ResultsPersistenceStrategy,
    SampleProcessorFactory,
)
from ...application.dto.build_inventory import BuildInventoryResult
from ...application.dto.run_montecarlo import (
    MonteCarloProjectStats,
    RunMonteCarloResult,
)
from ...config.app_config import AppConfig
from ...core.domain.contracts import MethodId

if TYPE_CHECKING:
    from ...infrastructure.brightway.dto import (
        ForegroundSimulationResult,
        MonteCarloSimulationResult,
        PIVSimulationResult,
    )


logger = logging.getLogger(__name__)


@dataclass
class MonteCarloRequest:
    """
    Agrupación de parámetros de configuración para la simulación Monte Carlo.

    Este patrón (Parameter Object) evita la explosión de parámetros en el
    método run() y facilita la extensión futura sin romper la firma del método.

    Attributes
    ----------
    run_bw_mc : bool
        Si es True, ejecuta Monte Carlo completo (foreground + background).
    run_foreground_mc : bool
        Si es True, ejecuta Monte Carlo solo del inventario (foreground).
    run_piv : bool
        Si es True, ejecuta la aproximación lineal PIV.
    iterations : int
        Número de iteraciones para el modo BW MC.
    fg_iterations : int
        Número de iteraciones para los modos Foreground MC y PIV.
    patron_metodo : str
        Patrón de texto para filtrar métodos de impacto.
    nivel_metodo : str
        Nivel del método de impacto (ej. 'midpoint (H)').
    methods : Optional[List[MethodId]]
        Lista explícita de métodos a evaluar. Si es None, se filtran por patrón.
    functional_unit : float
        Cantidad de la unidad funcional para el cálculo.
    mc_config : dict[str, dict[str, Any]] | None
        Configuración de reglas de dependencias y mezclas por proyecto.
    dependency_config : dict[str, dict[str, Any]] | None
        Configuración global de dependencias físicas (fallback).
    mix_config : dict[float, list[str]] | None
        Configuración global de mezclas (fallback).
    ecoinvent_db_name : str | None
        Nombre de la base de datos de Ecoinvent (requerido para PIV).
    fg_seed : int | None
        Semilla para la generación de números aleatorios en FG/PIV.
    verbose_processor : bool
        Si es True, habilita logs detallados del procesador de muestras.
    include_pedigree : bool
        Si es True, incluye variabilidad del background por pedigrí en PIV.
    h_pedigree_n : int
        Número de muestras unitarias por proceso de fondo para pedigrí.
    correlate_pedigree : bool
        Si es True, preserva correlación física en el muestreo de pedigrí.
    enforce_physical_constraints : bool
        Si es True, aplica truncamiento para garantizar que los flujos mantengan
        su signo físico (flujos positivos ≥ 0, flujos negativos ≤ 0).
        Por defecto True.
    save_cache : bool
        Si es True, guarda los resultados en disco.
    cache_filename : Optional[str]
        Nombre del archivo de caché. Si es None, se usa un nombre por defecto.
    """

    run_bw_mc: bool = True
    run_foreground_mc: bool = False
    run_piv: bool = False

    iterations: int = 1000
    fg_iterations: int = 500

    patron_metodo: str = "ReCiPe 2016"
    nivel_metodo: str = "midpoint (H)"
    methods: list[MethodId] | None = None
    functional_unit: float = 1.0

    mc_config: dict[str, dict[str, Any]] | None = None
    dependency_config: dict[str, dict[str, Any]] | None = None
    mix_config: dict[float, list[str]] | None = None

    ecoinvent_db_name: str | None = None
    fg_seed: int | None = 42

    verbose_processor: bool = False
    include_pedigree: bool = False
    h_pedigree_n: int = 1000
    correlate_pedigree: bool = False
    enforce_physical_constraints: bool = True

    save_cache: bool = True
    cache_filename: str | None = None


class RunMonteCarloUseCase:
    """
    Orquesta los modos de simulación Monte Carlo.

    Depende exclusivamente de abstracciones (Protocols) definidas en
    application.contracts, cumpliendo con el Principio de Inversión de
    Dependencias (DIP).
    """

    def __init__(
        self,
        config: AppConfig,
        method_filter: MethodFilteringStrategy,
        bw_mc_strategy: MonteCarloSimulationStrategy,
        fg_mc_strategy: ForegroundSimulationStrategy,
        piv_strategy: PIVSimulationStrategy,
        processor_factory: SampleProcessorFactory,
        stats_calculator: MCStatsCalculationStrategy,
        persistence: ResultsPersistenceStrategy[RunMonteCarloResult] | None = None,
    ) -> None:
        """
        Inicializa el caso de uso inyectando sus dependencias.

        Parameters
        ----------
        config : AppConfig
            Configuración de la aplicación.
        method_filter : MethodFilteringStrategy
            Estrategia para filtrar métodos de impacto.
        bw_mc_strategy : MonteCarloSimulationStrategy
            Estrategia de ejecución para Monte Carlo completo (BW).
        fg_mc_strategy : ForegroundSimulationStrategy
            Estrategia de ejecución para Monte Carlo del primer plano.
        piv_strategy : PIVSimulationStrategy
            Estrategia de ejecución para la aproximación lineal PIV.
        processor_factory : Callable
            Fábrica para crear procesadores de muestras por proyecto.
        stats_calculator : MCStatsCalculationStrategy
            Estrategia para calcular estadísticas descriptivas de los scores.
        persistence : Optional[ResultsPersistenceStrategy]
            Estrategia de persistencia en disco. Si es None, no se guarda caché.
        """
        self.config = config
        self.method_filter = method_filter
        self.bw_mc_strategy = bw_mc_strategy
        self.fg_mc_strategy = fg_mc_strategy
        self.piv_strategy = piv_strategy
        self.processor_factory = processor_factory
        self.stats_calculator = stats_calculator
        self.persistence = persistence

    def run(
        self,
        build_result: BuildInventoryResult,
        request: MonteCarloRequest,
    ) -> RunMonteCarloResult:
        """
        Ejecuta los modos de simulación Monte Carlo solicitados.

        Parameters
        ----------
        build_result : BuildInventoryResult
            Resultado del caso de uso de construcción de inventario.
        request : MonteCarloRequest
            Parámetros de configuración para la ejecución.

        Returns
        -------
        RunMonteCarloResult
            DTO con los resultados acumulados, estadísticas y metadatos.

        Raises
        ------
        ValueError
            Si la configuración o los resultados de entrada son inválidos.
        """
        if not build_result.success:
            return RunMonteCarloResult(
                success=False,
                error_message=f"BuildInventory falló: {build_result.error_message}",
            )

        start_time = time.time()
        modes_run: list[str] = []

        # Estructuras de datos fuertemente tipadas y anidadas por proyecto
        collected_scores: dict[MethodId, dict[str, np.ndarray]] = defaultdict(dict)
        all_component_samples: dict[str, dict[str, np.ndarray]] = {}
        all_piv_contributions: dict[str, dict[MethodId, dict[str, np.ndarray]]] = {}
        total_iterations_completed: int = 0

        # 1. Determinar métodos
        methods = request.methods or self.method_filter.filter(
            patron=request.patron_metodo, nivel=request.nivel_metodo
        )
        if not methods:
            return RunMonteCarloResult(
                success=False,
                error_message="No se encontraron métodos de impacto.",
            )

        # 2. Preparar procesadores de muestras (FG y PIV)
        per_project_processors = self.processor_factory(
            mc_config=request.mc_config or {},
            projects=build_result.projects,
            dependency_config=request.dependency_config,
            mix_config=request.mix_config,
            verbose=request.verbose_processor,
            enforce_physical_constraints=request.enforce_physical_constraints,
        )

        # Inyectar procesadores en las estrategias si el adaptador lo soporta
        if hasattr(self.fg_mc_strategy, "_sample_processor"):
            self.fg_mc_strategy._sample_processor = per_project_processors
        if hasattr(self.piv_strategy, "_sample_processor"):
            self.piv_strategy._sample_processor = per_project_processors

        # 3. Ejecutar BW MC (Completo)
        if request.run_bw_mc:
            logger.info(
                "Ejecutando BW Monte Carlo (%d iteraciones)...", request.iterations
            )
            bw_result: MonteCarloSimulationResult = self.bw_mc_strategy.run(
                iterations=request.iterations,
                functional_unit=request.functional_unit,
            )

            if bw_result.iterations_completed > 0 and bw_result.scores.size > 0:
                for m_idx, method_id in enumerate(bw_result.method_ids):
                    for p_idx, project_name in enumerate(bw_result.project_ids):
                        # scores shape: (n_methods, n_projects, n_iterations)
                        collected_scores[method_id][project_name] = bw_result.scores[
                            m_idx, p_idx, :
                        ]

                total_iterations_completed = max(
                    total_iterations_completed, bw_result.iterations_completed
                )
                modes_run.append("bw_mc")
            else:
                logger.warning(
                    "BW MC no generó resultados (posiblemente falta de actividades/"
                    "métodos en la BD)."
                )

        # 4. Ejecutar Foreground MC
        if request.run_foreground_mc:
            logger.info(
                "Ejecutando Foreground MC (%d iteraciones)...", request.fg_iterations
            )
            fg_results: list[ForegroundSimulationResult] = self.fg_mc_strategy.run(
                iterations=request.fg_iterations
            )

            for res in fg_results:
                if res.iterations_completed > 0:
                    for method_id, scores_array in res.method_scores.items():
                        collected_scores[method_id][res.project_id] = scores_array

                    # Asignación anidada directa para evitar sobrescritura de claves
                    all_component_samples[res.project_id] = res.component_samples
                    total_iterations_completed = max(
                        total_iterations_completed, res.iterations_completed
                    )

            if fg_results:
                modes_run.append("foreground_mc")

        # 5. Ejecutar PIV MC (Mutuamente excluyente con FG en esta lógica)
        elif request.run_piv:
            ei_db = request.ecoinvent_db_name or self.config.get(
                "ecoinvent_source_db_name"
            )
            if not ei_db:
                logger.warning("ecoinvent_db_name no especificado. Omitiendo PIV.")
            else:
                logger.info(
                    "Ejecutando PIV MC (%d iteraciones)...", request.fg_iterations
                )
                piv_results: list[PIVSimulationResult] = self.piv_strategy.run(
                    iterations=request.fg_iterations
                )

                for piv_res in piv_results:
                    if piv_res.iterations_completed > 0:
                        for method_id, scores_array in piv_res.method_scores.items():
                            collected_scores[method_id][
                                piv_res.project_id
                            ] = scores_array

                        # Asignación anidada directa
                        all_component_samples[piv_res.project_id] = (
                            piv_res.component_samples
                        )

                        # Asignación directa de contribuciones PIV
                        # (ya vienen estructuradas por el runner)
                        if piv_res.piv_contributions:
                            all_piv_contributions[piv_res.project_id] = (
                                piv_res.piv_contributions
                            )

                        total_iterations_completed = max(
                            total_iterations_completed, piv_res.iterations_completed
                        )

                if piv_results:
                    modes_run.append("piv")

        # 6. Verificar si se recolectaron datos
        if not collected_scores:
            return RunMonteCarloResult(
                success=False,
                error_message="Ningún modo de Monte Carlo generó resultados. "
                "Revise el inventario y los métodos.",
            )

        # 7. Calcular estadísticas
        stats: list[MonteCarloProjectStats] = []
        if modes_run:
            logger.info("Calculando estadísticas MC...")
            stats = self.stats_calculator.calculate(
                scores=dict(collected_scores),
                generation_dict=build_result.generation_dict,
            )

        # 8. Persistir si se solicita (Incluyendo PIV contributions)
        cache_path = None
        if request.save_cache and self.persistence is not None:
            result_to_cache = RunMonteCarloResult(
                modes_run=modes_run,
                scores=dict(collected_scores),
                stats=stats,
                component_samples=all_component_samples,
                piv_contributions=all_piv_contributions,
                iterations_completed=total_iterations_completed,
            )

            saved = self.persistence.save(
                result_to_cache, filename=request.cache_filename
            )

            cache_path = saved

        elapsed = time.time() - start_time
        logger.info("Simulación MC concluida en %.1fs. Modos: %s", elapsed, modes_run)

        return RunMonteCarloResult(
            success=True,
            modes_run=modes_run,
            iterations_completed=total_iterations_completed,
            stats=stats,
            component_samples=all_component_samples,
            scores=dict(collected_scores),
            piv_contributions=all_piv_contributions,
            elapsed_seconds=elapsed,
            cache_path=cache_path,
        )
