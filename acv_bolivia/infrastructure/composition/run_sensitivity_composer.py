"""
infrastructure.composition.run_sensitivity_composer: Compositor del caso
de uso RunSensitivityUseCase.

Este módulo es el punto de ensamblaje donde se inyectan las implementaciones
concretas de infraestructura en el caso de uso RunSensitivityUseCase, satisfaciendo
los protocolos definidos en application/contracts.py.

El compositor orquesta:
    - El motor de sensibilidad (SensitivityEngine)
    - Los seis analizadores (Delta LCA, Morris, Sobol, SHAP, Correlation, Regression)
    - El proveedor LCA (BrightwayLCAProvider)
    - El procesador de muestras (para reglas DEP/MIX)
    - El generador de muestras sintéticas (fallback cuando no hay MC previo)

La generación de gráficos y la exportación a Excel NO forman parte del caso de
uso: se invocan de forma explícita desde la capa de interfaces
(ver ACVEngine.plot_sensitivity() y ACVEngine.export_sensitivity()).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...analysis.sensitivity import SensitivityEngine
from ...analysis.sensitivity.methods import (
    CorrelationAnalyzer,
    DeltaLCAAnalyzer,
    MorrisAnalyzer,
    RegressionAnalyzer,
    SHAPAnalyzer,
    SobolAnalyzer,
)
from ...application.contracts import (
    LCAInfrastructureProvider,
    SampleProcessingStrategy,
    SensitivityEngineStrategy,
)
from ...application.dto.run_montecarlo import RunMonteCarloResult
from ...application.use_cases.run_sensitivity import (
    DEFAULT_ANALYZER_SETTINGS,
    RunSensitivityUseCase,
)
from ...config.app_config import AppConfig
from ...core.domain.contracts import MethodId, SensitivityAnalyzer
from ...core.domain.models import Project
from ...infrastructure.brightway import (
    BrightwayConnector,
    BrightwayLCAProvider,
)
from ...infrastructure.brightway.montecarlo import (
    create_sample_processor,
    sample_vectorized,
)
from ...infrastructure.persistence import ResultsFileRepository

logger = logging.getLogger(__name__)


# ==============================================================================
# Adaptadores de Infraestructura a Protocolos de Aplicación
# ==============================================================================


class _LCAProviderAdapter:
    """Adapta BrightwayLCAProvider al protocolo LCAInfrastructureProvider."""

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        project: Project,
        technical_map: dict[str, str],
        location_map: dict[str, str],
        sample_processor: SampleProcessingStrategy | None = None,
    ) -> None:
        self._provider = BrightwayLCAProvider(
            bc_module=bc_module,
            bd_module=bd_module,
            local_db_name=local_db_name,
            project=project,
            technical_map=technical_map,
            location_map=location_map,
            sample_processor=sample_processor,
        )

    def get_nominal_parameters(self, project_name: str) -> dict[str, float]:
        """Retorna parámetros nominales del proyecto."""
        return self._provider.get_nominal_parameters(project_name)

    def create_evaluator(
        self,
        project_name: str,
        method_id: MethodId,
        functional_unit_amount: float = 1.0,
    ) -> Any:
        """Crea un evaluador LCA."""
        return self._provider.create_evaluator(
            project_name=project_name,
            method_id=method_id,
            functional_unit_amount=functional_unit_amount,
        )

    def create_piv_evaluator(self, nominal_params: dict[str, float]) -> Any | None:
        """Crea un evaluador PIV si es posible."""
        return self._provider.create_piv_evaluator(nominal_params)

    def get_latest_mapping_diagnostic(self) -> Any | None:
        """Retorna diagnóstico del último mapeo."""
        return self._provider.get_latest_mapping_diagnostic()

    def extract_piv_h_vectors(
        self, lca_result: Any, project_name: str, method_name: str
    ) -> dict[str, float]:
        """Extrae vectores h para PIV."""
        return self._provider.extract_piv_h_vectors(
            lca_result, project_name, method_name
        )


class _SensitivityEngineAdapter:
    """Adapta SensitivityEngine al protocolo SensitivityEngineStrategy."""

    def __init__(
        self,
        lca_provider: LCAInfrastructureProvider,
        analyzers: Sequence[SensitivityAnalyzer],
        exclude_methods: set[str] | None = None,
    ) -> None:
        self._engine = SensitivityEngine(
            lca_provider=lca_provider,
            analyzers=analyzers,
            exclude_methods=exclude_methods,
        )

    def run(
        self,
        project_name: str,
        method_tuple: MethodId,
        component_samples_raw: dict[str, list[float]] | None = None,
        lca_scores_raw: Any | None = None,
    ) -> Any:
        """Ejecuta el análisis de sensibilidad."""
        return self._engine.run(
            project_name=project_name,
            method_tuple=method_tuple,
            component_samples_raw=component_samples_raw,
            lca_scores_raw=lca_scores_raw,
        )


class _MCResultsReaderAdapter:
    """Adapta RunMonteCarloResult al protocolo MCResultsReader."""

    def __init__(self, mc_result: RunMonteCarloResult) -> None:
        self._mc_result = mc_result

    def get_mc_results(
        self,
        project_name: str | None = None,
        method_name: Any | None = None,
    ) -> list[Any]:
        """Retorna resultados MC filtrados por proyecto y/o método."""
        # Convertir el DTO de aplicación al formato esperado por el engine
        results = []
        for method_id, project_scores in self._mc_result.scores.items():
            for proj_id, scores in project_scores.items():
                if project_name is not None and proj_id != project_name:
                    continue
                if method_name is not None and method_id != method_name:
                    continue
                # Crear un objeto compatible con el formato legacy
                results.append(
                    {
                        "project_name": proj_id,
                        "method_name": method_id,
                        "scores": scores,
                    }
                )
        return results

    def get_component_samples(self, project_name: str) -> dict[str, Any] | None:
        """Retorna muestras de componentes de un proyecto."""
        return self._mc_result.component_samples.get(project_name)


class _PersistenceAdapter:
    """Adapta ResultsFileRepository al protocolo ResultsPersistenceStrategy."""

    def __init__(self, output_dir: str | Path) -> None:
        self._repo: ResultsFileRepository[Any] = ResultsFileRepository(output_dir)

    def save(self, data: Any, filename: str | None = None) -> Path | None:
        """Serializa y guarda datos en disco."""
        return self._repo._storage.save(data, filename or "sensitivity_results")

    def load(self, filename: str | None = None) -> Any | None:
        """Carga datos desde disco."""
        try:
            return self._repo._storage.load(filename or "sensitivity_results")
        except FileNotFoundError:
            return None

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si el archivo existe."""
        return self._repo._storage.exists(filename or "sensitivity_results")


class _SyntheticSamplerAdapter:
    """Adapta sample_vectorized a un generador de muestras sintéticas."""

    def __call__(
        self,
        project: Project,
        n_samples: int,
    ) -> dict[str, NDArray[Any]]:
        """Genera muestras sintéticas desde la incertidumbre del dominio."""
        rng = np.random.default_rng(42)
        samples: dict[str, NDArray[Any]] = {}
        for exc in project.exchanges:
            if exc.exchange_type != "technosphere":
                continue
            samples[exc.component_id] = sample_vectorized(
                exc.uncertainty,
                float(exc.quantity.amount),
                rng,
                n_samples,
            )
        return samples


class _AnalyzerFactoryAdapter:
    """Fábrica que crea la lista estándar de analizadores de sensibilidad."""

    def __call__(
        self, settings: dict[str, dict[str, Any]] | None = None
    ) -> list[SensitivityAnalyzer]:
        """Retorna la lista estándar de analizadores con los ajustes aplicados.

        Los ajustes recibidos ya vienen resueltos (usuario + settings.json);
        aquí se fusionan sobre los valores por defecto del framework
        (DEFAULT_ANALYZER_SETTINGS), de modo que un parámetro ausente conserva
        su default.
        """
        merged = {
            method: {**defaults, **(settings.get(method, {}) if settings else {})}
            for method, defaults in DEFAULT_ANALYZER_SETTINGS.items()
        }
        return [
            DeltaLCAAnalyzer(**merged["delta_lca"]),
            MorrisAnalyzer(**merged["morris"]),
            SobolAnalyzer(**merged["sobol"]),
            SHAPAnalyzer(**merged["shap"]),
            CorrelationAnalyzer(**merged["correlation"]),
            RegressionAnalyzer(**merged["regression"]),
        ]


# ==============================================================================
# Función Fábrica Principal
# ==============================================================================


def create_run_sensitivity_use_case(
    config: AppConfig,
    connector: BrightwayConnector,
    local_db_name: str,
    projects: list[Project],
    technical_maps: dict[str, dict[str, str]],
    location_maps: dict[str, dict[str, str]],
    mc_result: RunMonteCarloResult | None = None,
    output_dir: str | Path | None = None,
) -> RunSensitivityUseCase:
    """Factory method que compone todas las dependencias del caso de uso
    RunSensitivityUseCase.

    Parameters
    ----------
    config : AppConfig
        Configuración de la aplicación.
    connector : BrightwayConnector
        Conector ya conectado a Brightway2.
    local_db_name : str
        Nombre de la base de datos local.
    projects : List[Project]
        Lista de proyectos del dominio.
    technical_maps : Dict[str, Dict[str, str]]
        Mapeo {project_name: {component: proceso_ei}}.
    location_maps : Dict[str, Dict[str, str]]
        Mapeo {project_name: {component: ubicación_ei}}.
    mc_result : Optional[RunMonteCarloResult]
        Resultado de Monte Carlo previo (si existe).
    output_dir : Optional[str | Path]
        Directorio de salida para la persistencia de caché. Si None, no se
        inyecta persistencia (no se guarda caché).

    Returns
    -------
    RunSensitivityUseCase
        Caso de uso listo para ejecutar con todas las dependencias inyectadas.

    Example
    -------
    >>> from config.app_config import AppConfig
    >>> from infrastructure.brightway import BrightwayConnector
    >>> from infrastructure.composition import create_run_sensitivity_use_case
    >>>
    >>> config = AppConfig()
    >>> connector = BrightwayConnector(project_name="mi_proyecto")
    >>> connector.connect()
    >>>
    >>> use_case = create_run_sensitivity_use_case(
    ...     config=config,
    ...     connector=connector,
    ...     local_db_name="mi_proyecto",
    ...     projects=[project1, project2],
    ...     technical_maps={"project1": {...}, "project2": {...}},
    ...     location_maps={"project1": {...}, "project2": {...}},
    ...     mc_result=mc_result,
    ... )
    >>>
    >>> from application.use_cases.run_sensitivity import SensitivityRequest
    >>> request = SensitivityRequest(
    ...     analyzers=None,
    ...     exclude_methods={"morris", "sobol"},
    ... )
    >>> result = use_case.run(build_result, lca_result, request)
    """
    bd = connector.get_data_module()
    bc = connector.get_calc_module()

    # Crear adaptador para MC results reader (si hay resultado previo)
    mc_reader = _MCResultsReaderAdapter(mc_result) if mc_result else None

    # Crear fábricas
    analyzer_factory = _AnalyzerFactoryAdapter()
    synthetic_sampler = _SyntheticSamplerAdapter()

    # Crear fábrica de procesadores DEP/MIX por proyecto
    def processor_factory(
        project: Project,
        dependency_config: dict[str, Any] | None,
        mix_config: dict[float, list[str]] | None,
        enforce_physical_constraints: bool,
    ) -> SampleProcessingStrategy | None:
        """Crea el procesador físico de muestras para un proyecto específico.

        Las reglas se aplican sobre las muestras (sintéticas, externas o
        provenientes de un MC previo) antes de alimentar a los analizadores
        por muestras, y también se inyectan en el proveedor LCA para que los
        métodos vivos (Sobol, Morris, Delta) evalúen el modelo con las
        dependencias entre componentes activas.
        """
        nominal_values = {
            exc.component_id: float(exc.quantity.amount)
            for exc in project.exchanges
            if exc.exchange_type == "technosphere"
        }
        has_rules = bool(dependency_config or mix_config) or (
            enforce_physical_constraints and nominal_values
        )
        if not has_rules:
            return None
        return create_sample_processor(
            dependency_config=dependency_config or None,
            mix_config=mix_config or None,
            nominal_values=nominal_values,
            enforce_physical_constraints=enforce_physical_constraints,
        )

    # Crear fábrica de LCA providers (uno por proyecto)
    def lca_provider_factory(
        project: Project,
        sample_processor: SampleProcessingStrategy | None = None,
    ) -> LCAInfrastructureProvider:
        """Crea un LCA provider para un proyecto específico."""
        return _LCAProviderAdapter(
            bc_module=bc,
            bd_module=bd,
            local_db_name=local_db_name,
            project=project,
            technical_map=technical_maps.get(project.name, {}),
            location_map=location_maps.get(project.name, {}),
            sample_processor=sample_processor,
        )

    # Crear fábrica de engines (uno por proyecto)
    def engine_factory(
        lca_provider: LCAInfrastructureProvider,
        analyzers: Sequence[SensitivityAnalyzer],
        exclude_methods: set[str],
    ) -> SensitivityEngineStrategy:
        """Crea un engine de sensibilidad para un proyecto específico."""
        return _SensitivityEngineAdapter(
            lca_provider=lca_provider,
            analyzers=analyzers,
            exclude_methods=exclude_methods,
        )

    return RunSensitivityUseCase(
        config=config,
        lca_provider_factory=lca_provider_factory,
        engine_factory=engine_factory,
        mc_reader=mc_reader,
        synthetic_sampler=synthetic_sampler,
        processor_factory=processor_factory,
        analyzer_factory=analyzer_factory,
        persistence=(
            _PersistenceAdapter(output_dir) if output_dir is not None else None
        ),
    )
