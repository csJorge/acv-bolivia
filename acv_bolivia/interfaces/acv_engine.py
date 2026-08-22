"""
interfaces.acv_engine: Motor de orquestación del framework ACV Bolivia.

Principios de diseño:
    - Cada método mapea 1:1 contra el use case / Request DTO correspondiente
      de application/.
    - Los plotters se exponen como objetos de control granular (self.plotter,
      self.piv_plotter(project_id)).
    - El estado (build_result, lca_result, mc_result, sensitivity_result) se
      mantiene internamente entre fases para no forzar a threadear objetos
      a mano entre celdas, pero queda expuesto como propiedades de solo
      lectura para inspección ad-hoc.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from ..analysis.convergence.mc_convergence import (
    ConvergenceReport,
    run_convergence_diagnostics,
)
from ..analysis.sensitivity.sensitivity_exporter import SensitivityExporter
from ..analysis.sensitivity.sensitivity_plotter import SensitivityPlotter
from ..application.dto.build_inventory import BuildInventoryResult
from ..application.dto.run_lca import RunLCAResult
from ..application.dto.run_montecarlo import RunMonteCarloResult
from ..application.dto.run_sensitivity import RunSensitivityResult
from ..application.use_cases.run_montecarlo import MonteCarloRequest
from ..application.use_cases.run_sensitivity import SensitivityRequest
from ..config.app_config import AppConfig
from ..core.domain.contracts import MethodId, SensitivityAnalyzer
from ..infrastructure.brightway.bw_context import BrightwayConnector
from ..infrastructure.composition.build_inventory_composer import (
    create_build_inventory_use_case,
)
from ..infrastructure.composition.run_lca_composer import create_run_lca_use_case
from ..infrastructure.composition.run_montecarlo_composer import (
    create_run_montecarlo_use_case,
)
from ..infrastructure.composition.run_sensitivity_composer import (
    create_run_sensitivity_use_case,
)
from ..infrastructure.persistence import (
    list_cached_files,
    load_latest_cache,
)
from ..interfaces.exporter import LCAExporter
from ..interfaces.piv_plotter import PIVPlotter
from ..interfaces.plotter import LCAPlotter

logger = logging.getLogger(__name__)

# Subcarpeta de salida y nombre de caché por defecto por fase.
_CACHE_PHASE_DIR = {
    "lca": "lca",
    "montecarlo": "montecarlo",
    "sensibilidad": "sensibilidad",
}
_DEFAULT_CACHE_FILENAME = {
    "lca": "lca_results",
    "montecarlo": "montecarlo_results",
    "sensibilidad": "sensitivity_results",
}


class ACVEngine:
    """Motor de orquestación del pipeline ACV, con control granular total.

    Mantiene el estado entre fases (build → lca → montecarlo → sensitivity)
    y expone cada fase con el 100% de los parámetros que su use case
    subyacente soporta.

    Examples
    --------
    Flujo típico:

    >>> eng = ACVEngine.from_json("config/settings.json")
    >>> eng.build()
    >>> eng.run_lca()
    >>> eng.run_montecarlo(run_bw_mc=True, run_foreground_mc=True, fg_iterations=500)
    >>> eng.run_sensitivity()
    >>> eng.export("Reporte_ACV_Bolivia")

    Acceso a resultados para análisis ad-hoc:

    >>> eng.mc_result.stats
    >>> eng.lca_result.hotspots

    Graficar con control granular:

    >>> for proyecto in eng.build_result.project_names:
    ...     eng.plotter.graficar_hotspot_apilados(proyecto, top_n=10)
    """

    def __init__(self, config: AppConfig) -> None:
        """Inicializa el motor con una configuración ya cargada.

        No conecta a Brightway2 todavía: la conexión es diferida hasta el
        primer método que la necesite (build(), run_lca(), etc.), vía
        _connector (propiedad lazy).

        Parameters
        ----------
        config : AppConfig
            Configuración de la aplicación (rutas, nombres de BD, etc.)
        """
        self.config = config

        self._build_result: BuildInventoryResult | None = None
        self._lca_result: RunLCAResult | None = None
        self._mc_result: RunMonteCarloResult | None = None
        self._sensitivity_result: RunSensitivityResult | None = None

        self._connector: BrightwayConnector | None = None

    @classmethod
    def from_json(cls, config_path: str) -> ACVEngine:
        """Crea un motor cargando la configuración desde un archivo JSON.

        Parameters
        ----------
        config_path : str
            Ruta al archivo settings.json.

        Returns
        -------
        ACVEngine
            Listo para usar (sin conectar a Brightway2 todavía).
        """
        config = AppConfig.load_from_json(config_path)
        return cls(config)

    # ------------------------------------------------------------------
    # Conexión a Brightway2 (diferida, compartida entre fases 2-4)
    # ------------------------------------------------------------------

    def _get_connector(self) -> BrightwayConnector:
        """Retorna el conector a Brightway2, conectando en el primer uso.

        Notes
        -----
        `create_build_inventory_use_case()` maneja su propio conector interno,
        por lo que este método garantiza la reutilización del conector para
        las fases subsiguientes (LCIA, Monte Carlo, Sensibilidad).
        """
        if self._connector is None:
            self._connector = BrightwayConnector(
                project_name=self.config.get("proyecto"),
                brightway_dir=self.config.get("rutas.bw2"),
                env_path=self.config.get("rutas.entorno"),
            )
            self._connector.connect()
        return self._connector

    # ------------------------------------------------------------------
    # Caché de resultados (lectura)
    # ------------------------------------------------------------------

    def _resolve_cache_filename(self, phase: str, cache_filename: str | None) -> str:
        """Resuelve el nombre de caché por defecto para una fase."""
        return cache_filename or _DEFAULT_CACHE_FILENAME[phase]

    def _load_from_cache(self, phase: str, cache_filename: str | None) -> Any | None:
        """Carga el caché más reciente de una fase, o None si no existe.

        Busca entre las carpetas fechadas de la fase (lca/, montecarlo/,
        sensibilidad/) bajo config.BASE_OUTPUT_PATH.
        """
        subdir = _CACHE_PHASE_DIR[phase]
        name = self._resolve_cache_filename(phase, cache_filename)
        return load_latest_cache(
            self.config.BASE_OUTPUT_PATH,
            subdir,
            name,
        )

    # ------------------------------------------------------------------
    # Fase 1: Inventario
    # ------------------------------------------------------------------

    def build(self, force_rebuild: bool = False) -> ACVEngine:
        """Fase 1: carga el Excel de inventario y construye la BD en Brightway2.

        Mapea 1:1 contra BuildInventoryUseCase.run() - el único parámetro
        que soporta ese método es force_rebuild.

        Parameters
        ----------
        force_rebuild : bool, default False
            Si True, borra completamente la BD local de Brightway2 y la
            reconstruye desde cero. Si False, reutiliza la BD existente si ya fue
            construida en una sesión anterior.

        Returns
        -------
        ACVEngine
            self, para permitir encadenamiento de métodos.

        Raises
        ------
        RuntimeError
            Si la construcción del inventario falla.
        """
        use_case = create_build_inventory_use_case(self.config)
        self._build_result = use_case.run(force_rebuild=force_rebuild)
        if not self._build_result.success:
            raise RuntimeError(f"Error en build(): {self._build_result.error_message}")
        return self

    # ------------------------------------------------------------------
    # Fase 2: LCIA determinístico
    # ------------------------------------------------------------------

    def run_lca(
        self,
        patron_metodo: str | None = None,
        nivel_metodo: str | None = None,
        top_n_hotspot: int | None = None,
        functional_unit: float | None = None,
        generation_dict: dict[str, float] | None = None,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_filename: str | None = None,
    ) -> ACVEngine:
        """Fase 2: calcula LCIA determinístico para todos los proyectos del inventario.

        Parameters
        ----------
        patron_metodo : str | None, default None
            Subcadena para buscar métodos de impacto en Brightway2.
            Si None (o vacío), usa config['lca.patron_metodo']
            (valor predeterminado 'ReCiPe 2016').
        nivel_metodo : str | None, default None
            Nivel jerárquico del método (ej. 'midpoint (H)', 'midpoint (E)',
            'endpoint (H)'). Si None (o vacío), usa config['lca.nivel_metodo']
            (valor predeterminado 'midpoint (H)').
        top_n_hotspot : int | None, default None
            Cantidad de procesos primarios a incluir en el análisis de hotspots.
            Si None (o 0), usa config['lca.top_n_hotspot']
            (valor predeterminado 50).
        functional_unit : float | None, default None
            Escala numérica de la unidad funcional.
            Si None (o 0), usa config['lca.functional_unit']
            (valor predeterminado 1.0).
        generation_dict : dict[str, float] | None
            Mapeo {project_name: kwh_generados} para normalización por kWh.
            Si None, toma los valores definidos en el inventario.
        use_cache : bool, default True
            Si True, intenta cargar la caché más reciente de esta fase
            (el archivo `cache_filename` o 'lca_results') y devuelve el
            resultado guardado sin recalcular. Si no hay caché, calcula.
        save_cache : bool, default True
            Si calcula y True, guarda en disco la caché serializada
            (bajo el nombre `cache_filename` si se indicó).
        cache_filename : str | None
            Nombre personalizado de la caché a leer/guardar.

        Returns
        -------
        ACVEngine
            self, para encadenamiento.

        Raises
        ------
        RuntimeError
            Si no se ejecutó build() previamente o si el caso de uso reporta fallas.
        """
        cache_name = self._resolve_cache_filename("lca", cache_filename)

        if use_cache:
            cached = self._load_from_cache("lca", cache_filename)
            if cached is not None:
                self._lca_result = cached
                logger.info(
                    "run_lca(): resultado cargado desde caché '%s'.", cache_name
                )
                return self

        self._assert_built()
        assert self._build_result is not None

        connector = self._get_connector()
        process_to_component = {
            v: k for k, v in self._build_result.technical_map.items()
        }
        output_dir = self.config.get_dated_output_folder("lca")

        use_case = create_run_lca_use_case(
            config=self.config,
            connector=connector,
            local_db_name=self._build_result.local_db_name,
            process_to_component=process_to_component,
            output_dir=output_dir,
        )
        self._lca_result = use_case.run(
            patron_metodo=patron_metodo
            or self.config.get("lca.patron_metodo", "ReCiPe 2016"),
            nivel_metodo=nivel_metodo
            or self.config.get("lca.nivel_metodo", "midpoint (H)"),
            top_n_hotspot=top_n_hotspot or self.config.get("lca.top_n_hotspot", 50),
            functional_unit=functional_unit
            or self.config.get("lca.functional_unit", 1.0),
            generation_dict=generation_dict or self._build_result.generation_dict,
            save_cache=save_cache,
            cache_filename=cache_filename,
        )
        if not self._lca_result.success:
            raise RuntimeError(f"Error en run_lca(): {self._lca_result.error_message}")
        return self

    # ------------------------------------------------------------------
    # Fase 3: Monte Carlo
    # ------------------------------------------------------------------

    def run_montecarlo(
        self,
        run_bw_mc: bool = True,
        run_foreground_mc: bool = False,
        run_piv: bool = False,
        iterations: int = 1000,
        fg_iterations: int = 500,
        ecoinvent_db_name: str | None = None,
        functional_unit: float = 1.0,
        fg_seed: int | None = 42,
        include_pedigree: bool = False,
        h_pedigree_n: int = 1000,
        correlate_pedigree: bool = False,
        enforce_physical_constraints: bool = True,
        mc_config: dict[str, dict[str, Any]] | None = None,
        dependency_config: dict[str, dict[str, Any]] | None = None,
        mix_config: dict[float, list[str]] | None = None,
        verbose_processor: bool = False,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_filename: str | None = None,
    ) -> ACVEngine:
        """Fase 3: simulación Monte Carlo:  BW completo, Foreground y/o PIV.
        .
                Los tres modos son independientes y combinables: run_bw_mc perturba
                foreground + background completo de Ecoinvent vía el motor nativo
                de Brightway2 (bc.MonteCarloLCA); run_foreground_mc perturba solo los
                parámetros del inventario
                (background fijo); run_piv es la aproximación lineal escalar
                (h-vectors x muestras).

                Parameters
                ----------
                run_bw_mc : bool, default True
                    Activar Monte Carlo completo de Brightway2.
                run_foreground_mc : bool, default False
                    Activar Monte Carlo solo foreground (Excel).
                run_piv : bool, default False
                    Activar la aproximación lineal PIV.
                iterations : int, default 1000
                    Iteraciones para BW MC.
                fg_iterations : int, default 500
                    Iteraciones para Foreground MC y/o PIV.
                ecoinvent_db_name : str, optional
                    Nombre de la BD de Ecoinvent en Brightway2. Requerido si
                    run_piv=True. Si None, usa el de config.
                functional_unit : float, default 1.0
                    Unidad funcional para el cálculo.
                fg_seed : int, optional, default 42
                    Semilla de aleatoriedad para FG MC / PIV.
                    None = no reproducible entre corridas.
                include_pedigree : bool, default False
                    Si True, incluye variabilidad adicional del background por
                    matriz de pedigrí en el muestreo PIV.
                h_pedigree_n : int, default 1000
                    Iteraciones para calcular la incertidumbre de pedigrí (solo si
                    include_pedigree=True).
                correlate_pedigree : bool, default False
                    Si True, preserva correlación física entre componentes al
                    muestrear con pedigrí.
                enforce_physical_constraints : bool, default True
                    Si True, trunca los flujos muestreados para que mantengan su
                    signo físico (positivos truncados a >= 0, negativos a <= 0).
                mc_config : dict[str, dict[str, Any]] | None
                    Configuración de reglas de dependencia/mezcla por proyecto.
                    Si None, usa build_result.mc_config.
                dependency_config : dict[str, dict[str, Any]] | None
                    Reglas de dependencia física globales (fallback si un proyecto
                    no tiene reglas propias en mc_config).
                    Formato: {componente_dependiente:
                    {"base_comps": [...], "factor": float}}.
                mix_config : dict[float, list[str]] | None
                    Restricciones de suma globales (fallback).
                    Formato: {valor_objetivo: [lista_de_componentes]}.
                verbose_processor : bool, default False
                    Si True, logs detallados del procesador de muestras
                    (SampleProcessor) durante la ejecución.
                use_cache : bool, default True
                    Si True, intenta cargar la caché más reciente de esta fase
                    (el archivo `cache_filename` o 'montecarlo_results') y devuelve
                    el resultado guardado sin simular. Si no hay caché, simula.
                save_cache : bool, default True
                    Si simula y True, guarda los resultados en disco.
                cache_filename : str | None
                    Nombre del archivo de caché a leer/guardar (ej. 'smc_bw',
                    'smc_piv'). None = por defecto.

                Returns
                -------
                ACVEngine
                    self, para encadenamiento.

                Raises
                ------
                RuntimeError
                    Si run_lca() no se llamó antes.

        """
        cache_name = self._resolve_cache_filename("montecarlo", cache_filename)

        if use_cache:
            cached = self._load_from_cache("montecarlo", cache_filename)
            if cached is not None:
                self._mc_result = cached
                logger.info(
                    "run_montecarlo(): resultado cargado desde caché '%s'.",
                    cache_name,
                )
                return self

        self._assert_lca_run()
        assert self._build_result is not None
        assert self._lca_result is not None

        connector = self._get_connector()
        build_result = self._build_result

        technical_maps = {
            name: build_result.technical_map for name in build_result.project_names
        }
        location_maps = {
            name: build_result.location_map for name in build_result.project_names
        }
        code_maps = {name: build_result.code_map for name in build_result.project_names}
        unit_maps = {name: build_result.unit_map for name in build_result.project_names}
        output_dir = self.config.get_dated_output_folder("montecarlo")

        ei_db_name = ecoinvent_db_name or self.config.get("ecoinvent_source_db_name")
        use_case = create_run_montecarlo_use_case(
            config=self.config,
            connector=connector,
            local_db_name=build_result.local_db_name,
            ecoinvent_db_name=ei_db_name,
            methods=self._lca_result.methods,
            projects=build_result.projects,
            technical_maps=technical_maps,
            location_maps=location_maps,
            code_maps=code_maps,
            unit_maps=unit_maps,
            output_dir=output_dir,
            functional_unit=functional_unit,
            fg_seed=fg_seed,
            include_pedigree=include_pedigree,
            h_pedigree_n=h_pedigree_n,
            correlate_pedigree=correlate_pedigree,
            enforce_physical_constraints=enforce_physical_constraints,
        )
        request = MonteCarloRequest(
            run_bw_mc=run_bw_mc,
            run_foreground_mc=run_foreground_mc,
            run_piv=run_piv,
            iterations=iterations,
            fg_iterations=fg_iterations,
            patron_metodo="",
            nivel_metodo="",
            methods=self._lca_result.methods,
            functional_unit=functional_unit,
            mc_config=mc_config if mc_config is not None else build_result.mc_config,
            dependency_config=dependency_config,
            mix_config=mix_config,
            ecoinvent_db_name=ei_db_name,
            fg_seed=fg_seed,
            verbose_processor=verbose_processor,
            include_pedigree=include_pedigree,
            h_pedigree_n=h_pedigree_n,
            correlate_pedigree=correlate_pedigree,
            enforce_physical_constraints=enforce_physical_constraints,
            save_cache=save_cache,
            cache_filename=cache_filename,
        )
        self._mc_result = use_case.run(build_result, request)
        return self

    # ------------------------------------------------------------------
    # Fase 4: Sensibilidad
    # ------------------------------------------------------------------

    def run_sensitivity(
        self,
        analyzers: Sequence[SensitivityAnalyzer] | None = None,
        exclude_methods: set[str] | None = None,
        method_indices: list[int] | None = None,
        project_indices: list[int] | None = None,
        component_samples: dict[str, dict[str, NDArray[Any]]] | None = None,
        n_synthetic_samples: int = 500,
        analyzer_settings: dict[str, dict[str, Any]] | None = None,
        dependency_config: dict[str, dict[str, Any]] | None = None,
        mix_config: dict[float, list[str]] | None = None,
        mc_config: dict[str, dict[str, Any]] | None = None,
        enforce_physical_constraints: bool = True,
        use_cache: bool = True,
        save_cache: bool = True,
        cache_filename: str | None = None,
    ) -> ACVEngine:
        """Fase 4: análisis de sensibilidad multi-método.

        Los analizadores por muestra (correlation/PRCC, regression/SRRC,
        SHAP) requieren component_samples y scores MC previos (ver Groen et
        al., 2014); delta_lca, morris y sobol evalúan el modelo en vivo y no
        los requieren. Si component_samples no se pasa explícitamente, se usa
        mc_result.component_samples (poblado si run_montecarlo(
        run_foreground_mc=True) se ejecutó antes).

        Este método solo calcula los reportes; los gráficos y la exportación
        a Excel se generan explícitamente con plot_sensitivity() y
        export_sensitivity().

        Parameters
        ----------
        analyzers : Sequence[SensitivityAnalyzer] | None
            Lista explícita de analizadores de sensibilidad a ejecutar.
            Si None, usa el conjunto estándar (Delta LCA, Morris, Sobol,
            Correlation/PRCC, Regression, SHAP).
        exclude_methods : set(str) | None
            Nombres de analizadores de sensibilidad a excluir del análisis
            (p. ej. {'morris', 'sobol'}). Analizadores disponibles:
            'delta_lca', 'morris', 'sobol', 'correlation', 'regression',
            'shap'. No son métodos de impacto: estos últimos se controlan
            con method_indices.
        method_indices : list[int] | None
            Índices de métodos a incluir (de lca_result.methods).
            None = todos.
        project_indices : list[int] | None
            Índices de proyectos a incluir (de build_result.projects).
            None = todos.
        component_samples : dict[str, dict[str, NDArray[Any]]] | None
            Muestras sincronizadas {project_id: {component_id: array}} para
            los analizadores por muestra (correlation/regression/SHAP).
            Si None, usa mc_result.component_samples.
        n_synthetic_samples : int, default 500
            Si no hay component_samples reales disponibles, número de
            muestras sintéticas a generar como fallback (afecta solo a los
            métodos que las requieren).
        analyzer_settings : dict[str, dict[str, Any]] | None
            Ajustes por analizador, p. ej. {"sobol": {"n_samples": 256,
            "top_k_screening": 6}, "morris": {"n_trajectories": 30}}. Cada
            solver tiene sus propios parámetros (ver
            DEFAULT_ANALYZER_SETTINGS). La prioridad por parámetro es la
            misma que el método de impacto del LCA: entrada explícita
            (analyzer_settings) > configuración (sección `sensibilidad` del
            settings.json, claves `analyzers.<metodo>` y alias planos legacy)
            > valor por defecto del analizador. Si analyzers no es None, se
            usan esos analizadores tal cual y este argumento se ignora.
        dependency_config : dict[str, dict[str, Any]] | None
            Reglas de dependencia física globales (fallback de baja
            prioridad), mismo formato que en run_montecarlo(). Se aplican a
            las muestras (sintéticas, externas o de un MC previo) antes de
            los analizadores por muestra y al proveedor LCA de los métodos
            vivos (delta_lca, morris, sobol), de modo que las variables
            derivadas responden a sus bases.
        mix_config : dict[float, list[str]] | None
            Restricciones de suma (fallback de baja prioridad).
        mc_config : dict[str, dict[str, Any]] | None
            Reglas de dependencia/mezcla por proyecto, mismo formato que en
            run_montecarlo(): {"GLOBAL": {...}, "<proyecto>": {...}} con
            claves "dependencies"/"mixes". Si None, usa build_result.mc_config
            (hoja Config_MC del Excel). La prioridad de fusión por proyecto es:
            mc_config[proyecto] > mc_config["GLOBAL"] > dependency_config/mix_config.
        enforce_physical_constraints : bool, default True
            Si True, trunca las muestras de los flujos para preservar el
            signo físico (nominal >= 0 no puede volverse negativo y viceversa).
        use_cache : bool, default True
            Si True, intenta cargar la caché más reciente de esta fase
            (el archivo `cache_filename` o 'sensitivity_results') y devuelve
            los reportes guardados sin volver a analizar. Si no hay caché,
            ejecuta el análisis.
        save_cache : bool, default True
            Si analiza y True, guarda los reportes en disco.
        cache_filename : str | None
            Nombre del archivo de caché a leer/guardar. None = por defecto.
        Returns
        -------
        ACVEngine
            self, para encadenamiento.

        Raises
        ------
        RuntimeError
            Si run_lca() no se llamó antes.
        """
        cache_name = self._resolve_cache_filename("sensibilidad", cache_filename)

        if use_cache:
            cached = self._load_from_cache("sensibilidad", cache_filename)
            if cached is not None:
                self._sensitivity_result = cached
                logger.info(
                    "run_sensitivity(): reportes cargados desde caché '%s'.",
                    cache_name,
                )
                return self

        self._assert_lca_run()
        assert self._build_result is not None
        assert self._lca_result is not None

        connector = self._get_connector()
        build_result = self._build_result

        technical_maps = {
            name: build_result.technical_map for name in build_result.project_names
        }
        location_maps = {
            name: build_result.location_map for name in build_result.project_names
        }

        use_case = create_run_sensitivity_use_case(
            config=self.config,
            connector=connector,
            local_db_name=build_result.local_db_name,
            projects=build_result.projects,
            technical_maps=technical_maps,
            location_maps=location_maps,
            mc_result=self._mc_result,
            output_dir=self.config.get_dated_output_folder("sensibilidad"),
        )
        request = SensitivityRequest(
            analyzers=analyzers,
            exclude_methods=exclude_methods,
            method_indices=method_indices,
            project_indices=project_indices,
            component_samples=component_samples
            or (self._mc_result.component_samples if self._mc_result else None),
            n_synthetic_samples=n_synthetic_samples,
            analyzer_settings=analyzer_settings,
            dependency_config=dependency_config,
            mix_config=mix_config,
            mc_config=mc_config if mc_config is not None else build_result.mc_config,
            enforce_physical_constraints=enforce_physical_constraints,
            save_cache=save_cache,
            cache_filename=cache_filename,
        )
        self._sensitivity_result = use_case.run(
            build_result, self._lca_result, request, mc_result=self._mc_result
        )
        if not self._sensitivity_result.success:
            raise RuntimeError(
                f"Error en run_sensitivity(): {self._sensitivity_result.error_message}"
            )
        return self

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def export(self, nombre_archivo: str = "Reporte_ACV_Bolivia") -> Path:
        """Exporta los resultados LCIA/MC a un archivo Excel.

        Parameters
        ----------
        nombre_archivo : str, default 'Reporte_ACV_Bolivia'
            Nombre base del archivo sin extensión.

        Returns
        -------
        Path
            Ruta del archivo .xlsx generado.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado run_lca() previamente.
        """
        self._assert_lca_run()
        assert self._lca_result is not None
        output_dir = self.config.get_dated_output_folder("reportes")
        exporter = LCAExporter(
            lca_result=self._lca_result,
            mc_result=self._mc_result,
            output_dir=output_dir,
        )
        return exporter.export(nombre_archivo=nombre_archivo)

    def plot_sensitivity(
        self,
        project_id: str,
        method_id: MethodId,
        output_dir: str | Path | None = None,
        close_figs: bool = True,
    ) -> list[Path]:
        """Genera los gráficos de un reporte de sensibilidad ya calculado.

        La generación de gráficos es explícita y separada del cálculo
        (run_sensitivity() no dibuja nada). Permite replotear una combinación
        sin volver a ejecutar el análisis.

        Parameters
        ----------
        project_id : str
            Nombre del proyecto.
        method_id : MethodId
            Tupla identificadora del método de impacto, tal como aparece en
            lca_result.methods.
        output_dir : str | Path | None
            Directorio de salida de los PNG. Si None, usa la carpeta fechada
            'graficos' de la configuración.
        close_figs : bool, default True
            Si True, cierra cada figura tras guardarla (comportamiento por
            defecto, evita fugas de memoria). Si False, deja las figuras
            abiertas para que se muestren en el notebook (Jupyter) u otro
            entorno interactivo.

        Returns
        -------
        list[Path]
            Rutas de los archivos PNG generados.

        Raises
        ------
        RuntimeError
            Si no se ejecutó run_sensitivity() previamente.
        KeyError
            Si el par (project_id, method_id) no existe en los resultados.
        """
        if self._sensitivity_result is None:
            raise RuntimeError("Primero llama a engine.run_sensitivity().")
        report = self._sensitivity_result.get_report(project_id, method_id)
        if report is None:
            combinaciones = ", ".join(
                f"({r.project_id}, {r.method_id})"
                for r in self._sensitivity_result.reports
            )
            raise KeyError(
                f"No existe SensitivityReport para ({project_id!r}, {method_id!r}). "
                f"Combinaciones disponibles: {combinaciones}"
            )
        target_dir = (
            Path(output_dir)
            if output_dir
            else self.config.get_dated_output_folder("graficos")
        )
        plotter = SensitivityPlotter(output_dir=target_dir)
        return plotter.plot_all(report, close_figs=close_figs)

    def export_sensitivity(
        self,
        project_id: str,
        method_id: MethodId,
        nombre: str = "Sensibilidad_ACV",
    ) -> Path:
        """Re-exporta un reporte de sensibilidad ya calculado.

        Parameters
        ----------
        project_id : str
            Nombre del proyecto.
        method_id : MethodId
            Tupla identificadora del método de impacto, tomada de
            ``lca_result.methods``.
        nombre : str, default 'Sensibilidad_ACV'
            Nombre base del archivo de salida.

        Returns
        -------
        Path
            Ruta del archivo .xlsx generado.

        Raises
        ------
        RuntimeError
            Si no se ejecuto run_sensitivity() previamente.
        KeyError
            Si el par (project_id, method_id) no existe en los resultados.
        """
        if self._sensitivity_result is None:
            raise RuntimeError("Primero llama a engine.run_sensitivity().")
        report = self._sensitivity_result.get_report(project_id, method_id)
        if report is None:
            combinaciones = ", ".join(
                f"({r.project_id}, {r.method_id})"
                for r in self._sensitivity_result.reports
            )
            raise KeyError(
                f"No existe SensitivityReport para ({project_id!r}, {method_id!r}). "
                f"Combinaciones disponibles: {combinaciones}"
            )
        output_dir = self.config.get_dated_output_folder("sensibilidad")
        exporter = SensitivityExporter(output_dir=output_dir)
        return exporter.export(report, nombre=nombre)

    # ------------------------------------------------------------------
    # Pruebas de convergencia Monte Carlo
    # ------------------------------------------------------------------

    def run_convergence_diagnostics(
        self,
        project_name: str,
        method_name: str,
        seed_runs: Sequence[Sequence[float]] | None = None,
        seed_labels: Sequence[str] | None = None,
        mean_tolerance: float = 0.02,
        percentile_tolerance: float = 0.05,
        mcse_target_precisions: Sequence[float] = (0.01, 0.02, 0.05),
    ) -> ConvergenceReport:
        """Diagnóstico de convergencia para una combinación proyecto/método.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto a analizar.
        method_name : str
            Subcadena o nombre del método de impacto.
        seed_runs : Sequence[Sequence[float]] | None
            Corridas MC adicionales con semillas distintas, para comparación.
        seed_labels : Sequence[str] | None
            Etiquetas descriptivas para las corridas de semillas.
        mean_tolerance : float, default 0.02
            Tolerancia relativa de convergencia de la media.
        percentile_tolerance : float, default 0.05
            Tolerancia relativa de convergencia para percentiles.
        mcse_target_precisions : Sequence[float], default (0.01, 0.02, 0.05)
            Precisiones objetivo para el error estándar de Monte Carlo (MCSE).

        Returns
        -------
        ConvergenceReport
            Reporte con los resultados de estabilidad y diagnóstico.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado run_montecarlo() previamente.
        KeyError
            Si el método o proyecto especificado no existe en mc_result.
        """
        if self._mc_result is None:
            raise RuntimeError("Primero llama a run_montecarlo().")

        assert self._mc_result is not None

        method_id = next(
            (m for m in self._mc_result.scores if method_name.lower() in m[1].lower()),
            None,
        )
        if method_id is None:
            raise KeyError(f"Método '{method_name}' no encontrado en mc_result.scores.")

        scores = list(self._mc_result.scores[method_id][project_name])
        return run_convergence_diagnostics(
            scores=scores,
            project_name=project_name,
            method_name=method_id[1],
            seed_runs=seed_runs,
            seed_labels=seed_labels,
            mean_tolerance=mean_tolerance,
            percentile_tolerance=percentile_tolerance,
            mcse_target_precisions=mcse_target_precisions,
        )

    # ------------------------------------------------------------------
    # Plotters - control granular
    # ------------------------------------------------------------------

    @property
    def plotter(self) -> LCAPlotter:
        """LCAPlotter instanciado con el estado actual de los resultados.

        Returns
        -------
        LCAPlotter
            Con los 5 métodos individuales: graficar_comparativa,
            graficar_hotspot_apilados, graficar_mc_distribucion,
            graficar_mc_boxplots, graficar_cv_ranking.

        Raises
        ------
        RuntimeError
            Si no se ejecuto run_lca() previamente.

        Examples
        --------
        >>> metodo = ("ReCiPe 2016", "climate change", "kg CO2 eq")
        >>> eng.plotter.graficar_comparativa(usar_kwh=True)
        >>> eng.plotter.graficar_hotspot_apilados("El Dorado", top_n=10)
        >>> eng.plotter.graficar_mc_distribucion("El Dorado", metodo)
        >>> eng.plotter.graficar_mc_boxplots(metodo)
        >>> eng.plotter.graficar_cv_ranking()
        """
        self._assert_lca_run()
        assert self._lca_result is not None
        output_dir = self.config.get_dated_output_folder("graficos")
        return LCAPlotter(
            lca_result=self._lca_result,
            output_dir=output_dir,
            mc_result=self._mc_result,
        )

    def piv_plotter(self, project_id: str) -> PIVPlotter:
        """PIVPlotter para las contribuciones PIV de un proyecto específico.

        Parameters
        ----------
        project_id : str
            Nombre exacto del proyecto (ver build_result.project_names).

        Returns
        -------
        PIVPlotter
            Instancia configurada para la visualización del proyecto.

        Raises
        ------
        RuntimeError
            Si run_montecarlo(run_piv=True) no se llamó antes.
        KeyError
            Si project_id no tiene contribuciones PIV calculadas.

        Examples
        --------
        >>> metodo = ("ReCiPe 2016", "climate change", "kg CO2 eq")
        >>> eng.piv_plotter("El Dorado").piv_hotspot_distributions(
        ...     "El Dorado", metodo
        ... )
        >>> report = eng.sensitivity_result.get_report("El Dorado", metodo)
        >>> eng.piv_plotter("El Dorado").shap_vs_piv_scatter(
        ...     report, "El Dorado", metodo
        ... )
        """
        if self._mc_result is None or "piv" not in self._mc_result.modes_run:
            raise RuntimeError("Primero llama a run_montecarlo(run_piv=True).")
        output_dir = self.config.get_dated_output_folder("graficos")
        return PIVPlotter(
            piv_contributions=self._mc_result.piv_contributions[project_id],
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Acceso a resultados (solo lectura)
    # ------------------------------------------------------------------

    @property
    def build_result(self) -> BuildInventoryResult | None:
        """BuildInventoryResult: Resultado de build().
        Resultado de la Fase 1 (build) o None si no se ha ejecutado.
        """
        return self._build_result

    @property
    def lca_result(self) -> RunLCAResult | None:
        """RunLCAResult: Resultado de run_lca().
        Resultado de la Fase 2 (run_lca) o None si no se ha ejecutado.
        """
        return self._lca_result

    @property
    def mc_result(self) -> RunMonteCarloResult | None:
        """RunMonteCarloResult: Resultado de run_montecarlo().
        Resultado de la Fase 3 (run_montecarlo) o None si no se ha ejecutado.
        """
        return self._mc_result

    @property
    def sensitivity_result(self) -> RunSensitivityResult | None:
        """RunSensitivityResult: resultado de run_sensitivity().

        None si la fase aún no se ha ejecutado.
        """
        return self._sensitivity_result

    def list_caches(self, phase: str) -> list[str]:
        """Lista los nombres de caché disponibles para una fase.

        Útil para recuperar análisis nombrados en sesiones anteriores
        (ej. 'smc_bw', 'smc_piv') y pasarlos a run_* con cache_filename=.

        Parameters
        ----------
        phase : str
            Fase: 'lca', 'montecarlo' o 'sensibilidad'.

        Returns
        -------
        list[str]
            Nombres de caché disponibles (sin extensión), del más reciente
            al más antiguo.

        Raises
        ------
        ValueError
            Si la fase no es válida.
        """
        if phase not in _CACHE_PHASE_DIR:
            raise ValueError(
                f"Fase desconocida: '{phase}'. Válidas: {sorted(_CACHE_PHASE_DIR)}"
            )
        subdir = _CACHE_PHASE_DIR[phase]
        seen: set[str] = set()
        names: list[str] = []
        for path in list_cached_files(self.config.BASE_OUTPUT_PATH, subdir):
            stem = path.name.removesuffix(".pkl.gz")
            if stem not in seen:
                seen.add(stem)
                names.append(stem)
        return names

    # ------------------------------------------------------------------
    # Guardas de orden de ejecución
    # ------------------------------------------------------------------

    def _assert_built(self) -> None:
        if self._build_result is None:
            raise RuntimeError("Primero llama a engine.build().")

    def _assert_lca_run(self) -> None:
        self._assert_built()
        if self._lca_result is None:
            raise RuntimeError("Primero llama a engine.run_lca().")
