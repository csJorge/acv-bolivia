"""
application.use_cases.run_sensitivity - Caso de uso: análisis de sensibilidad.

Coordina la ejecución de los análisis de sensibilidad paramétrica sobre los
proyectos y métodos ambientales configurados. No conoce implementaciones
concretas de infraestructura ni de análisis.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...application.contracts import (
    LCAInfrastructureProvider,
    MCResultsReader,
    ResultsPersistenceStrategy,
    SampleProcessingStrategy,
    SensitivityEngineStrategy,
)
from ...application.dto.build_inventory import BuildInventoryResult
from ...application.dto.run_lca import RunLCAResult
from ...application.dto.run_montecarlo import RunMonteCarloResult
from ...application.dto.run_sensitivity import RunSensitivityResult
from ...config.app_config import AppConfig
from ...core.domain.contracts import SensitivityAnalyzer
from ...core.domain.models import Project

logger = logging.getLogger(__name__)

# Iteraciones mínimas de Montecarlo para confiar en los analizadores que
# requieren varianza (correlation/regression/shap).
_MIN_ITERATIONS_FOR_STATISTICAL_METHODS = 30


# ==============================================================================
# Request DTO (Patrón Parameter Object)
# ==============================================================================


@dataclass
class SensitivityRequest:
    """Agrupación de parámetros de configuración para el análisis de sensibilidad.

    Esto evita la 'explosión de parámetros' en el método run() y facilita
    la extensión futura sin romper la firma del método (OCP).
    """

    analyzers: Sequence[SensitivityAnalyzer] | None = None
    exclude_methods: set[str] | None = None
    method_indices: list[int] | None = None
    project_indices: list[int] | None = None

    component_samples: dict[str, dict[str, NDArray]] | None = None
    n_synthetic_samples: int = 500
    analyzer_settings: dict[str, dict[str, Any]] | None = None

    dependency_config: dict[str, Any] | None = None
    mix_config: dict[float, list[str]] | None = None
    mc_config: dict[str, dict[str, Any]] | None = None
    enforce_physical_constraints: bool = True

    save_cache: bool = True
    cache_filename: str | None = None


# ==============================================================================
# Helper: Fábrica de analizadores estándar
# ==============================================================================

DEFAULT_ANALYZER_SETTINGS: dict[str, dict[str, Any]] = {
    # Parámetros por defecto de cada analizador (nivel "fallback").
    "delta_lca": {"deltas": (0.1, 0.2), "components_subset": None},
    "morris": {"n_trajectories": 20, "n_levels": 4},
    "sobol": {"n_samples": 512, "top_k_screening": 8, "calc_second_order": False},
    "shap": {"explainer_type": "tree", "n_background": 100},
    "correlation": {
        "compute_pearson": True,
        "compute_spearman": True,
        "compute_prcc": True,
    },
    "regression": {"compute_src": True, "compute_srrc": True},
}

# Aliases planos legacy de la sección "sensibilidad" de settings.json.
# Claves: settings.sensibilidad.<alias> -> (metodo, parametro).
_ANALYZER_FLAT_ALIASES: dict[str, tuple[str, str]] = {
    "delta_values": ("delta_lca", "deltas"),
    "morris_trajectories": ("morris", "n_trajectories"),
    "sobol_n_samples": ("sobol", "n_samples"),
    "sobol_top_k": ("sobol", "top_k_screening"),
    "shap_explainer": ("shap", "explainer_type"),
}


def config_analyzer_settings(config: AppConfig, method_name: str) -> dict[str, Any]:
    """Configuración de un analizador desde settings.json.

    Se leen, por orden de prioridad, los bloques anidados
    ``sensibilidad.analyzers.<metodo>`` y los alias planos legacy
    (``delta_values``, ``morris_trajectories``, etc.). Devuelve solo los
    parámetros presentes en la configuración.

    Parameters
    ----------
    config : AppConfig
        Configuración de la aplicación.
    method_name : str
        Nombre del método de sensibilidad (clave de DEFAULT_ANALYZER_SETTINGS).

    Returns
    -------
    Dict[str, Any]
        Parámetros configurados para el analizador (vacío si no hay ninguno).
    """
    section = config.get("sensibilidad", {}) or {}
    settings: dict[str, Any] = {}
    for flat_key, (target, param) in _ANALYZER_FLAT_ALIASES.items():
        if target == method_name and section.get(flat_key) is not None:
            settings[param] = section[flat_key]
    nested = section.get("analyzers", {}) or {}
    settings.update(nested.get(method_name, {}) or {})
    return settings


def default_sensitivity_analyzers(
    analyzer_factory: Callable[
        [dict[str, dict[str, Any]] | None], list[SensitivityAnalyzer]
    ],
    settings: dict[str, dict[str, Any]] | None = None,
) -> list[SensitivityAnalyzer]:
    """Construye el conjunto estándar de analizadores de sensibilidad.

    Esta función es una conveniencia que delega la instanciación a una fábrica
    inyectada, manteniendo el caso de uso desacoplado de implementaciones concretas.

    Parameters
    ----------
    analyzer_factory : Callable
        Función que retorna la lista de analizadores instanciados y acepta los
        ajustes por método, con firma
        ``Callable[[Optional[Dict[str, Dict[str, Any]]], List[SensitivityAnalyzer]]``:
        ``{method_name: {param: value}}`` o None.
    settings : Optional[Dict[str, Dict[str, Any]]]
        Ajustes resueltos por analizador. Si es None, la fábrica usa sus valores
        por defecto.

    Returns
    -------
    List[SensitivityAnalyzer]
        Lista de analizadores listos para ejecutar.
    """
    return analyzer_factory(settings)


# ==============================================================================
# Helper: Resolución de muestras de componentes
# ==============================================================================


def resolve_component_samples(
    project: Project,
    component_names: set[str],
    external_samples: dict[str, dict[str, NDArray]] | None,
    manager: MCResultsReader | None,
    n_samples_to_use: int,
    synthetic_sampler: Callable[[Project, int], dict[str, NDArray]] | None,
) -> dict[str, NDArray] | None:
    """Resuelve las muestras a analizar, en orden de prioridad decreciente.

    1. Muestras externas provistas explícitamente por el llamador.
    2. Muestras reales de una simulación FG/PIV previa en el manager.
    3. Muestras sintéticas generadas desde la incertidumbre del dominio.

    Parameters
    ----------
    project : Project
        Proyecto del dominio.
    component_names : Set[str]
        Nombres de componentes a filtrar.
    external_samples : Optional[Dict[str, Dict[str, NDArray]]]
        Muestras externas {project_id: {component_id: array}}.
    manager : Optional[MCResultsReader]
        Lector de resultados MC.
    n_samples_to_use : int
        Número de muestras a usar.
    synthetic_sampler : Callable[[Project, int], Dict[str, NDArray]]
        Función que genera muestras sintéticas desde el dominio.

    Returns
    -------
    Optional[Dict[str, NDArray]]
        Muestras de componentes o None si no se pudieron resolver.
    """
    # Prioridad 1: Muestras externas
    if external_samples and project.name in external_samples:
        logger.info("[%s] Usando component_samples externos.", project.name)
        return external_samples[project.name]

    # Prioridad 2: Muestras de MC previo
    if manager is not None:
        proj_raw = manager.get_component_samples(project.name)
        if proj_raw:
            matched = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in proj_raw.items()
                if k in component_names
            }
            if matched and len(next(iter(matched.values()))) == n_samples_to_use:
                return matched

    # Prioridad 3: Muestras sintéticas
    if synthetic_sampler is None:
        raise ValueError(
            f"No hay muestras MC previas para '{project.name}' y no se "
            "configuró synthetic_sampler. RunSensitivityUseCase necesita "
            "al menos una fuente de datos para este proyecto."
        )
    logger.info(
        "[%s] Generadas %d muestras sintéticas desde el dominio.",
        project.name,
        n_samples_to_use,
    )
    return synthetic_sampler(project, n_samples_to_use)


# ==============================================================================
# Caso de Uso (DIP Compliant)
# ==============================================================================


class RunSensitivityUseCase:
    """Ejecuta el análisis de sensibilidad completo para los proyectos y
    métodos configurados.

    No conoce implementaciones concretas de infraestructura ni de análisis.
    """

    def __init__(
        self,
        config: AppConfig,
        lca_provider_factory: Callable[
            [Project, SampleProcessingStrategy | None], LCAInfrastructureProvider
        ],
        engine_factory: Callable[
            [LCAInfrastructureProvider, Sequence[SensitivityAnalyzer], set[str]],
            SensitivityEngineStrategy,
        ],
        mc_reader: MCResultsReader | None = None,
        synthetic_sampler: Callable[[Project, int], dict[str, NDArray]] | None = None,
        analyzer_factory: (
            Callable[[dict[str, dict[str, Any]] | None], list[SensitivityAnalyzer]]
            | None
        ) = None,
        persistence: ResultsPersistenceStrategy[RunSensitivityResult] | None = None,
        processor_factory: (
            Callable[
                [Project, dict[str, Any] | None, dict[float, list[str]] | None, bool],
                SampleProcessingStrategy | None,
            ]
            | None
        ) = None,
    ) -> None:
        """Inyección de dependencias mediante constructor.

        Parameters
        ----------
        config : AppConfig
            Configuración de la aplicación.
        lca_provider_factory : Callable
            Fábrica que crea LCAInfrastructureProvider para un proyecto.
        engine_factory : Callable
            Fábrica que crea SensitivityEngineStrategy.
        mc_reader : Optional[MCResultsReader]
            Lector de resultados MC. Si None, se usan muestras sintéticas.
        synthetic_sampler : Optional[Callable]
            Función que genera muestras sintéticas desde el dominio.
        analyzer_factory : Optional[Callable]
            Fábrica que crea la lista de analizadores estándar.
        persistence : Optional[ResultsPersistenceStrategy]
            Estrategia de persistencia en disco. Si es None, no se guarda caché.
        processor_factory : Optional[Callable]
            Fábrica que crea el procesador de muestras DEP/MIX por proyecto
            (reglas de dependencia, mezcla y restricciones físicas). Recibe
            (project, dependency_config, mix_config, enforce_physical_constraints)
            y devuelve un SampleProcessingStrategy o None. Si es None, no se
            aplican reglas sobre las muestras.
        """
        self.config = config
        self.lca_provider_factory = lca_provider_factory
        self.engine_factory = engine_factory
        self.mc_reader = mc_reader
        self.synthetic_sampler = synthetic_sampler
        self.analyzer_factory = analyzer_factory
        self.persistence = persistence
        self.processor_factory = processor_factory

    def run(
        self,
        build_result: BuildInventoryResult,
        lca_result: RunLCAResult,
        request: SensitivityRequest,
        mc_result: RunMonteCarloResult | None = None,
    ) -> RunSensitivityResult:
        """Ejecuta el análisis de sensibilidad para cada combinación (proyecto, método).

        Parameters
        ----------
        build_result : BuildInventoryResult
            Resultado de BuildInventoryUseCase.
        lca_result : RunLCAResult
            Resultado de RunLCAUseCase (provee los métodos evaluados).
        request : SensitivityRequest
            Parámetros de configuración del análisis.
        mc_result : RunMonteCarloResult | None
            Resultado de RunMonteCarloUseCase, si ya se corrió alguna simulación.

        Returns
        -------
        RunSensitivityResult
            Resultado con reportes por combinación evaluada.
        """
        if not build_result.success:
            return RunSensitivityResult(
                success=False,
                error_message=f"BuildInventory falló: {build_result.error_message}",
            )

        start_time = time.time()

        # 1. Preparar analizadores
        # Resolución por parámetro: entrada de usuario > settings.json > fallback.
        if request.analyzers is not None:
            analyzers = list(request.analyzers)
        elif self.analyzer_factory is not None:
            config_settings = {
                name: config_analyzer_settings(self.config, name)
                for name in DEFAULT_ANALYZER_SETTINGS
            }
            user_settings = request.analyzer_settings or {}
            resolved = {
                name: {
                    **config_settings[name],
                    **user_settings.get(name, {}),
                }
                for name in DEFAULT_ANALYZER_SETTINGS
            }
            analyzers = default_sensitivity_analyzers(self.analyzer_factory, resolved)
        else:
            analyzers = []
        available_methods = {a.method_name for a in analyzers}
        methods_requiring_variance = {
            a.method_name for a in analyzers if a.requires_variance
        }

        # Excluir métodos: entrada de usuario > settings.json
        # (sensibilidad.exclude_methods).
        exclude_input = (
            request.exclude_methods
            if request.exclude_methods is not None
            else self.config.get("sensibilidad.exclude_methods", None)
        )
        # Convertir exclude_methods a set para la operación de diferencia
        exclude_set = set(exclude_input) if exclude_input else set()
        invalid = exclude_set - available_methods
        if invalid:
            logger.warning(
                "Métodos de exclusión no reconocidos: %s. Se ignorarán.", invalid
            )
        exclude = exclude_set & available_methods

        # 2. Determinar proyectos y métodos
        all_methods = lca_result.methods
        all_projects = build_result.projects
        methods = (
            [all_methods[i] for i in request.method_indices]
            if request.method_indices
            else all_methods
        )
        projects = (
            [all_projects[i] for i in request.project_indices]
            if request.project_indices
            else all_projects
        )

        # 3. Determinar iteraciones MC disponibles
        mc_iterations = 0

        # Fuente principal: DTO de aplicación (RunMonteCarloResult)
        if mc_result is not None and mc_result.success:
            mc_iterations = mc_result.iterations_completed

            # Fallback: si iterations_completed es 0, intentar inferir de los samples
            if mc_iterations == 0 and mc_result.component_samples:
                first_project_samples = next(
                    iter(mc_result.component_samples.values()), {}
                )
                if first_project_samples:
                    first_component_array = next(iter(first_project_samples.values()))
                    if hasattr(first_component_array, "__len__"):
                        mc_iterations = len(first_component_array)

        # Fallback legacy: usar mc_reader si está disponible y no hay iteraciones
        if mc_iterations == 0 and self.mc_reader is not None:
            try:
                mc_series = self.mc_reader.get_mc_results()
                if mc_series and len(mc_series) > 0:
                    first = mc_series[0]
                    # Manejar tanto dict como objeto con atributo .scores
                    if isinstance(first, dict):
                        scores = first.get("scores", [])
                        mc_iterations = len(scores) if hasattr(scores, "__len__") else 0
                    elif hasattr(first, "scores"):
                        mc_iterations = len(first.scores)
            except Exception:
                logger.exception(
                    "No se pudo leer mc_reader para inferir iteraciones: %s"
                )

        n_samples_to_use = (
            mc_iterations if mc_iterations > 0 else request.n_synthetic_samples
        )

        # 4. Ejecutar análisis por combinación (proyecto, método)
        reports = []
        total = len(projects) * len(methods)
        done = 0

        for project in projects:
            component_names = {
                exc.component_id
                for exc in project.exchanges
                if exc.exchange_type == "technosphere"
            }

            global_cfg = (request.mc_config or {}).get("GLOBAL", {})
            proj_cfg = (request.mc_config or {}).get(project.name, {})

            merged_dep = {
                **(request.dependency_config or {}),
                **global_cfg.get("dependencies", {}),
                **proj_cfg.get("dependencies", {}),
            }
            merged_mix = {
                **(request.mix_config or {}),
                **global_cfg.get("mixes", {}),
                **proj_cfg.get("mixes", {}),
            }

            sample_processor = (
                self.processor_factory(
                    project,
                    merged_dep or None,
                    merged_mix or None,
                    request.enforce_physical_constraints,
                )
                if self.processor_factory is not None
                else None
            )

            proj_samples = resolve_component_samples(
                project=project,
                component_names=component_names,
                external_samples=request.component_samples,
                manager=self.mc_reader,
                n_samples_to_use=n_samples_to_use,
                synthetic_sampler=self.synthetic_sampler,
            )

            if sample_processor is not None and proj_samples:
                proj_samples = sample_processor(proj_samples)

            has_variance = proj_samples is not None and any(
                np.std(v) > 1e-12 for v in proj_samples.values()
            )
            effective_exclude = set(exclude)

            if not has_variance:
                effective_exclude |= methods_requiring_variance
                logger.info(
                    "[%s] Sin variabilidad en muestras → omitiendo %s",
                    project.name,
                    sorted(methods_requiring_variance),
                )
            elif mc_iterations == 0:
                effective_exclude |= methods_requiring_variance
                logger.info(
                    "[%s] Sin scores MC → omitiendo %s",
                    project.name,
                    sorted(methods_requiring_variance),
                )
            elif mc_iterations < _MIN_ITERATIONS_FOR_STATISTICAL_METHODS:
                logger.warning(
                    "[%s] AVISO: solo %d iteraciones MC - correlación/regresión/SHAP "
                    "poco confiables",
                    project.name,
                    mc_iterations,
                )

            # Crear LCA provider para este proyecto
            lca_provider = self.lca_provider_factory(project, sample_processor)

            # Crear engine para este proyecto
            engine = self.engine_factory(lca_provider, analyzers, effective_exclude)

            for method in methods:
                done += 1
                method_name = method[1]
                logger.info("[%d/%d] %s | %s", done, total, project.name, method_name)

                try:
                    component_samples_raw = (
                        {k: v.tolist() for k, v in proj_samples.items()}
                        if has_variance and proj_samples
                        else None
                    )

                    # Obtener scores MC para este proyecto+método
                    lca_scores_raw = None
                    if self.mc_reader:
                        matching_mc = self.mc_reader.get_mc_results(
                            project_name=project.name, method_name=method
                        )
                        if matching_mc:
                            lca_scores_raw = matching_mc[0]["scores"]

                    # Ejecutar análisis
                    report = engine.run(
                        project_name=project.name,
                        method_tuple=method,
                        component_samples_raw=component_samples_raw,
                        lca_scores_raw=lca_scores_raw,
                    )
                    reports.append(report)

                except Exception as e:
                    logger.exception(
                        "Falla en la combinación %s | %s: %s",
                        project.name,
                        method,
                        type(e).__name__,
                    )
                    continue

        elapsed = time.time() - start_time
        logger.info(
            "Sensibilidad completada: %d reportes generados en %.1fs",
            len(reports),
            elapsed,
        )

        result = RunSensitivityResult(
            reports=reports,
            elapsed_seconds=elapsed,
            success=len(reports) > 0,
        )

        if request.save_cache and self.persistence is not None and result.success:
            saved = self.persistence.save(result, filename=request.cache_filename)
            result.cache_path = saved

        return result
