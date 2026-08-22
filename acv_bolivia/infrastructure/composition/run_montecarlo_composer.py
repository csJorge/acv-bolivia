"""
infrastructure.composition.run_montecarlo_composer: Compositor del caso
de uso RunMonteCarloUseCase.

Este módulo es el punto de ensamblaje donde se inyectan las implementaciones
concretas de infraestructura en el caso de uso RunMonteCarloUseCase, satisfaciendo
los protocolos definidos en application/contracts.py.

El compositor orquesta los tres modos de simulación Monte Carlo:
    - BW MC completo (MonteCarloRunner): foreground + Ecoinvent perturbado.
    - Foreground MC (ForegroundMCRunner): solo inventario Excel.
    - PIV MC (PIVMonteCarloRunner): aproximación lineal con h-vectors.

Además, inyecta la fábrica de procesadores de muestras (para reglas DEP/MIX)
y el calculador de estadísticas descriptivas.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from ...application.contracts import (
    ForegroundSimulationStrategy,
    MCStatsCalculationStrategy,
    MethodFilteringStrategy,
    MonteCarloSimulationStrategy,
    PIVSimulationStrategy,
    ResultsPersistenceStrategy,
    SampleProcessingStrategy,
)
from ...application.dto.run_montecarlo import (
    MonteCarloProjectStats,
    RunMonteCarloResult,
)
from ...application.use_cases.run_montecarlo import RunMonteCarloUseCase
from ...config.app_config import AppConfig
from ...core.domain.contracts import MethodId
from ...core.domain.models import Project
from ...infrastructure.brightway import (
    BrightwayConnector,
    MethodFilter,
)
from ...infrastructure.brightway.dto import (
    ForegroundSimulationResult,
    MonteCarloSimulationResult,
    PIVSimulationResult,
)
from ...infrastructure.brightway.montecarlo import (
    ForegroundMCRunner,
    MonteCarloRunner,
    PIVMonteCarloRunner,
    create_sample_processor,
)
from ...infrastructure.persistence import ResultsFileRepository

logger = logging.getLogger(__name__)


# ==============================================================================
# Adaptadores de Infraestructura a Protocolos de Aplicación
# ==============================================================================


class _MethodFilterAdapter(MethodFilteringStrategy):
    """Adapta MethodFilter al protocolo MethodFilteringStrategy."""

    def __init__(self, bd_module: Any) -> None:
        self._filter = MethodFilter(bd_module=bd_module)

    def filter(
        self,
        patron: str,
        nivel: str,
        exclude_lt: bool = True,
    ) -> list[MethodId]:
        """Filtra métodos de impacto según patrón y nivel."""
        return self._filter.filter(patron=patron, nivel=nivel, exclude_lt=exclude_lt)


class _MonteCarloRunnerAdapter(MonteCarloSimulationStrategy):
    """Adapta MonteCarloRunner al protocolo MonteCarloSimulationStrategy."""

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        methods: list[MethodId],
        functional_unit: float = 1.0,
    ) -> None:
        self._bc = bc_module
        self._bd = bd_module
        self._local_db_name = local_db_name
        self._methods = methods
        self._functional_unit = functional_unit
        self._runner: MonteCarloRunner | None = None

    def run(
        self,
        iterations: int,
        functional_unit: float = 1.0,
    ) -> MonteCarloSimulationResult:
        """Ejecuta la simulación Monte Carlo completa (BW MC)."""
        self._runner = MonteCarloRunner(
            bc_module=self._bc,
            bd_module=self._bd,
            local_db_name=self._local_db_name,
            methods=self._methods,
            functional_unit=functional_unit,
        )
        return self._runner.run(iterations=iterations)

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        if self._runner is not None:
            self._runner.cleanup()
            self._runner = None


class _ForegroundMCRunnerAdapter(ForegroundSimulationStrategy):
    """Adapta ForegroundMCRunner al protocolo ForegroundSimulationStrategy."""

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        methods: list[MethodId],
        projects: list[Project],
        sample_processor: dict[str, SampleProcessingStrategy],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
        functional_unit: float = 1.0,
        seed: int | None = 42,
    ) -> None:
        """Inicializa el adaptador con todos los parámetros necesarios.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado.
        bd_module : Any
            Módulo bw2data inyectado.
        local_db_name : str
            Nombre de la base de datos local.
        methods : List[MethodId]
            Lista de métodos de impacto.
        projects : List[Project]
            Lista de proyectos del dominio.
        sample_processor : Dict[str, SampleProcessingStrategy]
            Procesadores de muestras por proyecto.
        technical_maps : Dict[str, Dict[str, str]]
            Mapeo {project_name: {componente: proceso_ecoinvent}}.
        location_maps : Dict[str, Dict[str, str]]
            Mapeo {project_name: {componente: ubicación_ecoinvent}}.
        functional_unit : float, optional
            Unidad funcional. Por defecto 1.0.
        seed : Optional[int], optional
            Semilla pseudoaleatoria. Por defecto 42.
        """
        self._bc = bc_module
        self._bd = bd_module
        self._local_db_name = local_db_name
        self._methods = methods
        self._projects = projects
        self._sample_processor = sample_processor
        self._technical_maps = technical_maps
        self._location_maps = location_maps
        self._functional_unit = functional_unit
        self._seed = seed
        self._runner: ForegroundMCRunner | None = None

    def run(self, iterations: int) -> list[ForegroundSimulationResult]:
        """Ejecuta la simulación Foreground MC."""
        self._runner = ForegroundMCRunner(
            bc_module=self._bc,
            bd_module=self._bd,
            local_db_name=self._local_db_name,
            methods=self._methods,
            projects=self._projects,
            sample_processor=self._sample_processor,
            technical_maps=self._technical_maps,
            location_maps=self._location_maps,
            functional_unit=self._functional_unit,
            seed=self._seed,
        )
        return self._runner.run(iterations=iterations)

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        if self._runner is not None:
            self._runner.cleanup()
            self._runner = None


class _PIVMonteCarloRunnerAdapter(PIVSimulationStrategy):
    """Adapta PIVMonteCarloRunner al protocolo PIVSimulationStrategy."""

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        ecoinvent_db_name: str,
        methods: list[MethodId],
        projects: list[Project],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
        sample_processor: dict[str, SampleProcessingStrategy],
        code_maps: dict[str, dict[str, str]] | None = None,
        unit_maps: dict[str, dict[str, str]] | None = None,
        functional_unit: float = 1.0,
        seed: int | None = 42,
        include_pedigree: bool = False,
        h_pedigree_n: int = 1000,
        correlate_pedigree: bool = False,
    ) -> None:
        self._bc = bc_module
        self._bd = bd_module
        self._local_db_name = local_db_name
        self._ecoinvent_db_name = ecoinvent_db_name
        self._methods = methods
        self._projects = projects
        self._technical_maps = technical_maps
        self._location_maps = location_maps
        self._code_maps = code_maps or {}
        self._unit_maps = unit_maps or {}
        self._sample_processor = sample_processor
        self._functional_unit = functional_unit
        self._seed = seed
        self._include_pedigree = include_pedigree
        self._h_pedigree_n = h_pedigree_n
        self._correlate_pedigree = correlate_pedigree
        self._runner: PIVMonteCarloRunner | None = None

    def run(self, iterations: int) -> list[PIVSimulationResult]:
        """Ejecuta la simulación PIV MC."""
        self._runner = PIVMonteCarloRunner(
            bc_module=self._bc,
            bd_module=self._bd,
            local_db_name=self._local_db_name,
            ecoinvent_db_name=self._ecoinvent_db_name,
            methods=self._methods,
            projects=self._projects,
            technical_maps=self._technical_maps,
            location_maps=self._location_maps,
            code_maps=self._code_maps,
            unit_maps=self._unit_maps,
            sample_processor=self._sample_processor,
            functional_unit=self._functional_unit,
            seed=self._seed,
            include_pedigree=self._include_pedigree,
            h_pedigree_n=self._h_pedigree_n,
            correlate_pedigree=self._correlate_pedigree,
        )
        return self._runner.run(iterations=iterations)

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        if self._runner is not None:
            self._runner.cleanup()
            self._runner = None


class _StatsCalculatorAdapter(MCStatsCalculationStrategy):
    """Adapta calculate_mc_stats al protocolo MCStatsCalculationStrategy."""

    def calculate(
        self,
        scores: dict[MethodId, dict[str, NDArray[Any]]],
        generation_dict: dict[str, float],
    ) -> list[MonteCarloProjectStats]:
        """Calcula estadísticas descriptivas para todas las series de scores MC."""
        from ...application.services.stats import calculate_mc_stats

        return calculate_mc_stats(scores=scores, generation_dict=generation_dict)


class _ProcessorFactoryAdapter:
    """Adapta create_sample_processor a una fábrica de procesadores por proyecto."""

    def __call__(
        self,
        mc_config: dict[str, dict[str, Any]],
        projects: list[Project],
        dependency_config: dict[str, dict[str, Any]] | None = None,
        mix_config: dict[float, list[str]] | None = None,
        verbose: bool = False,
        enforce_physical_constraints: bool = True,
    ) -> dict[str, SampleProcessingStrategy]:
        """Crea un processor por proyecto mezclando reglas GLOBAL + específicas.

        La prioridad de fusión por proyecto es:
        mc_config[proyecto] > mc_config["GLOBAL"] > dependency_config/mix_config.
        """
        global_cfg = mc_config.get("GLOBAL", {})
        global_dep = global_cfg.get("dependencies", {})
        global_mix = global_cfg.get("mixes", {})

        processors: dict[str, SampleProcessingStrategy] = {}
        for project in projects:
            proj_cfg = mc_config.get(project.name, {})

            merged_dep = {
                **(dependency_config or {}),
                **global_dep,
                **proj_cfg.get("dependencies", {}),
            }
            merged_mix = {
                **(mix_config or {}),
                **global_mix,
                **proj_cfg.get("mixes", {}),
            }

            # Extraer valores nominales del proyecto
            nominal_values = {
                exc.component_id: float(exc.quantity.amount)
                for exc in project.exchanges
                if exc.exchange_type == "technosphere"
            }

            processors[project.name] = create_sample_processor(
                dependency_config=merged_dep or None,
                mix_config=merged_mix or None,
                nominal_values=nominal_values,
                enforce_physical_constraints=enforce_physical_constraints,
                verbose=verbose,
            )
        return processors


class _PersistenceAdapter(ResultsPersistenceStrategy[RunMonteCarloResult]):
    """Adapta ResultsFileRepository al protocolo ResultsPersistenceStrategy."""

    def __init__(self, output_dir: str | Path) -> None:
        self._repo: ResultsFileRepository[RunMonteCarloResult] = ResultsFileRepository(
            output_dir
        )

    def save(
        self, data: RunMonteCarloResult, filename: str | None = None
    ) -> Path | None:
        """Serializa y guarda datos en disco."""
        return self._repo._storage.save(data, filename or "montecarlo_results")

    def load(self, filename: str | None = None) -> RunMonteCarloResult | None:
        """Carga datos desde disco."""
        try:
            return self._repo._storage.load(filename or "montecarlo_results")
        except FileNotFoundError:
            return None

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si el archivo existe."""
        return self._repo._storage.exists(filename or "montecarlo_results")


# ==============================================================================
# Función Fábrica Principal
# ==============================================================================


def create_run_montecarlo_use_case(
    config: AppConfig,
    connector: BrightwayConnector,
    local_db_name: str,
    ecoinvent_db_name: str,
    methods: list[MethodId],
    projects: list[Project],
    technical_maps: dict[str, dict[str, str]],
    location_maps: dict[str, dict[str, str]],
    output_dir: str | Path,
    code_maps: dict[str, dict[str, str]] | None = None,
    unit_maps: dict[str, dict[str, str]] | None = None,
    functional_unit: float = 1.0,
    fg_seed: int | None = 42,
    include_pedigree: bool = False,
    h_pedigree_n: int = 1000,
    correlate_pedigree: bool = False,
    enforce_physical_constraints: bool = True,
) -> RunMonteCarloUseCase:
    """Factory method que compone todas las dependencias del caso de uso
    RunMonteCarloUseCase.

    Parameters
    ----------
    config : AppConfig
        Configuración de la aplicación.
    connector : BrightwayConnector
        Conector ya conectado a Brightway2.
    local_db_name : str
        Nombre de la base de datos local.
    ecoinvent_db_name : str
        Nombre de la base de datos de Ecoinvent (para PIV).
    methods : List[MethodId]
        Lista de métodos de impacto a evaluar.
    projects : List[Project]
        Lista de proyectos del dominio.
    technical_maps : Dict[str, Dict[str, str]]
        Mapeo {project_name: {component: proceso_ei}}.
    location_maps : Dict[str, Dict[str, str]]
        Mapeo {project_name: {component: ubicación_ei}}.
    code_maps : Dict[str, Dict[str, str]], optional
        Mapeo {project_name: {component: código_ei}}.
    unit_maps : Dict[str, Dict[str, str]], optional
        Mapeo {project_name: {component: unidad_ei}}.
    output_dir : str | Path
        Directorio de salida para caché.
    functional_unit : float, optional
        Unidad funcional. Por defecto 1.0.
    fg_seed : Optional[int], optional
        Semilla pseudoaleatoria para FG/PIV. Por defecto 42.
    include_pedigree : bool, optional
        Si True, incluye variabilidad del background por pedigrí en PIV.
    h_pedigree_n : int, optional
        Número de muestras unitarias por proceso de fondo. Por defecto 1000.
    correlate_pedigree : bool, optional
        Si True, preserva correlación física en el muestreo de pedigrí.
    enforce_physical_constraints : bool, optional
        Si True, aplica truncamiento para garantizar que los flujos mantengan
        su signo físico. Por defecto True.

    Returns
    -------
    RunMonteCarloUseCase
        Caso de uso listo para ejecutar con todas las dependencias inyectadas.

    Example
    -------
    >>> from config.app_config import AppConfig
    >>> from infrastructure.brightway import BrightwayConnector
    >>> from infrastructure.composition import create_run_montecarlo_use_case
    >>>
    >>> config = AppConfig()
    >>> connector = BrightwayConnector(project_name="mi_proyecto")
    >>> connector.connect()
    >>>
    >>> use_case = create_run_montecarlo_use_case(
    ...     config=config,
    ...     connector=connector,
    ...     local_db_name="mi_proyecto",
    ...     ecoinvent_db_name="ecoinvent-3.9-cutoff",
    ...     methods=[("ReCiPe 2016", "midpoint (H)", "climate change")],
    ...     projects=[project1, project2],
    ...     technical_maps={"project1": {...}, "project2": {...}},
    ...     location_maps={"project1": {...}, "project2": {...}},
    ...     output_dir="./resultados",
    ... )
    >>>
    >>> from application.use_cases.run_montecarlo import MonteCarloRequest
    >>> request = MonteCarloRequest(
    ...     run_bw_mc=True,
    ...     run_foreground_mc=True,
    ...     iterations=1000,
    ...     fg_iterations=500,
    ... )
    >>> result = use_case.run(build_result, request)
    """
    bd = connector.get_data_module()
    bc = connector.get_calc_module()

    # Crear adaptadores para cada estrategia de simulación
    bw_mc_adapter = _MonteCarloRunnerAdapter(
        bc_module=bc,
        bd_module=bd,
        local_db_name=local_db_name,
        methods=methods,
        functional_unit=functional_unit,
    )

    fg_mc_adapter = _ForegroundMCRunnerAdapter(
        bc_module=bc,
        bd_module=bd,
        local_db_name=local_db_name,
        methods=methods,
        projects=projects,
        sample_processor={},  # Se inyectará dinámicamente desde el caso de uso
        technical_maps=technical_maps,
        location_maps=location_maps,
        functional_unit=functional_unit,
        seed=fg_seed,
    )

    piv_adapter = _PIVMonteCarloRunnerAdapter(
        bc_module=bc,
        bd_module=bd,
        local_db_name=local_db_name,
        ecoinvent_db_name=ecoinvent_db_name,
        methods=methods,
        projects=projects,
        technical_maps=technical_maps,
        location_maps=location_maps,
        code_maps=code_maps,
        unit_maps=unit_maps,
        sample_processor={},  # Se inyectará dinámicamente desde el caso de uso
        functional_unit=functional_unit,
        seed=fg_seed,
        include_pedigree=include_pedigree,
        h_pedigree_n=h_pedigree_n,
        correlate_pedigree=correlate_pedigree,
    )

    return RunMonteCarloUseCase(
        config=config,
        method_filter=_MethodFilterAdapter(bd),
        bw_mc_strategy=bw_mc_adapter,
        fg_mc_strategy=fg_mc_adapter,
        piv_strategy=piv_adapter,
        processor_factory=_ProcessorFactoryAdapter(),
        stats_calculator=_StatsCalculatorAdapter(),
        persistence=_PersistenceAdapter(output_dir),
    )
