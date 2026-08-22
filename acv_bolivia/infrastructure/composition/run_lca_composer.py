"""
infrastructure.composition.run_lca_composer: Compositor del caso de uso RunLCAUseCase.

Este módulo es el punto de ensamblaje (Composition Root) donde se inyectan las
implementaciones concretas de infraestructura en el caso de uso RunLCAUseCase,
satisfaciendo los protocolos definidos en application.contracts.

El compositor conoce los detalles técnicos (Brightway2, persistencia en disco),
pero el caso de uso permanece desacoplado gracias a la Inversión de Dependencias (DIP).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...application.contracts import (
    LCIAComputingStrategy,
    MethodFilteringStrategy,
    ResultsNormalizationStrategy,
    ResultsPersistenceStrategy,
)
from ...application.dto.run_lca import RunLCAResult
from ...application.use_cases.run_lca import RunLCAUseCase
from ...config.app_config import AppConfig
from ...core.domain.contracts import MethodId
from ...core.domain.models import HotspotResult, LCAResult
from ...core.services.normalization import NormalizationReport, normalize_by_generation
from ...infrastructure.brightway import BrightwayConnector, LCACalculator, MethodFilter
from ...infrastructure.brightway.dto import LCACalculationResult
from ...infrastructure.persistence import ResultsFileRepository

logger = logging.getLogger(__name__)


# ==============================================================================
# Adaptadores de Infraestructura a Protocolos de Aplicación
# ==============================================================================


class _MethodFilterAdapter(MethodFilteringStrategy):
    """Adapta MethodFilter al protocolo MethodFilteringStrategy.

    Este adaptador envuelve el MethodFilter de infraestructura para que cumpla
    con el protocolo de aplicación, permitiendo la inyección de dependencias
    sin acoplar el caso de uso a la implementación concreta.
    """

    def __init__(self, bd_module: Any) -> None:
        """Inicializa el adaptador con el módulo de datos de Brightway2.

        Parameters
        ----------
        bd_module : Any
            Módulo bw2data inyectado desde el conector.
        """
        self._filter = MethodFilter(bd_module=bd_module)

    def filter(
        self,
        patron: str,
        nivel: str,
        exclude_lt: bool = True,
    ) -> list[MethodId]:
        """Filtra métodos de impacto según patrón y nivel.

        Parameters
        ----------
        patron : str
            Patrón de búsqueda (ej. 'ReCiPe 2016').
        nivel : str
            Nivel del método (ej. 'midpoint (H)').
        exclude_lt : bool, optional
            Si es True, excluye métodos de largo plazo. Por defecto True.

        Returns
        -------
        List[MethodId]
            Lista de tuplas de métodos de impacto que coinciden.
        """
        return self._filter.filter(patron=patron, nivel=nivel, exclude_lt=exclude_lt)


class _LCACalculatorAdapter(LCIAComputingStrategy):
    """Adapta LCACalculator al protocolo LCIAComputingStrategy.

    Este adaptador envuelve el LCACalculator de infraestructura para que cumpla
    con el protocolo de aplicación, permitiendo calcular el impacto LCIA sin
    acoplar el caso de uso a Brightway2.
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        process_to_component: dict[Any, str],
    ) -> None:
        """Inicializa el adaptador con los módulos y el mapeo de procesos.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado.
        bd_module : Any
            Módulo bw2data inyectado.
        local_db_name : str
            Nombre de la base de datos local del inventario.
        process_to_component : Dict[Any, str]
            Mapeo de claves de proceso de Ecoinvent a IDs de componentes del dominio.
        """
        self._bc = bc_module
        self._bd = bd_module
        self._local_db_name = local_db_name
        self._process_to_component = process_to_component

    def compute(
        self,
        methods: list[MethodId],
        top_n_hotspot: int,
        functional_unit: float,
    ) -> LCACalculationResult:
        """Calcula el impacto LCIA y retorna el DTO de infraestructura.

        Parameters
        ----------
        methods : List[MethodId]
            Lista de métodos de impacto a evaluar.
        top_n_hotspot : int
            Número de hotspots a extraer por proyecto y método.
        functional_unit : float
            Cantidad de la unidad funcional para el cálculo.

        Returns
        -------
        LCACalculationResult
            DTO con los scores determinísticos y hotspots calculados.
        """
        calculator = LCACalculator(
            bc_module=self._bc,
            bd_module=self._bd,
            local_db_name=self._local_db_name,
            methods=methods,
            process_to_component=self._process_to_component,
        )
        return calculator.run(
            top_n_hotspot=top_n_hotspot,
            functional_unit=functional_unit,
        )


class _NormalizationAdapter(ResultsNormalizationStrategy):
    """Adapta normalize_by_generation al protocolo ResultsNormalizationStrategy.

    Este adaptador opera directamente sobre listas de entidades del dominio,
    eliminando por completo la dependencia del legacy LCAResultsManager.
    La función normalize_by_generation ya está refactorizada para recibir
    listas de LCAResult y HotspotResult directamente.
    """

    def normalize(
        self,
        lca_results: list[LCAResult],
        hotspots: list[HotspotResult],
        generation_dict: dict[str, float],
    ) -> NormalizationReport:
        """Normaliza los scores por la generación eléctrica del proyecto.

        Parameters
        ----------
        lca_results : List[LCAResult]
            Lista de resultados determinísticos a normalizar.
        hotspots : List[HotspotResult]
            Lista de hotspots a normalizar.
        generation_dict : Dict[str, float]
            Diccionario {project_id: kwh_generados}.

        Returns
        -------
        NormalizationReport
            Reporte con el detalle de la operación de normalización.
        """
        return normalize_by_generation(
            lca_results=lca_results,
            hotspots=hotspots,
            generation_dict=generation_dict,
        )


class _PersistenceAdapter(ResultsPersistenceStrategy[RunLCAResult]):
    """Adapta ResultsFileRepository al protocolo ResultsPersistenceStrategy.

    Este adaptador envuelve el ResultsFileRepository de infraestructura para
    que cumpla con el protocolo de aplicación, permitiendo persistir resultados
    sin acoplar el caso de uso al sistema de archivos.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Inicializa el adaptador con el directorio de salida.

        Parameters
        ----------
        output_dir : str | Path
            Directorio base para guardar los archivos de caché.
        """
        self._repo: ResultsFileRepository[RunLCAResult] = ResultsFileRepository(
            output_dir
        )

    def save(self, data: RunLCAResult, filename: str | None = None) -> Path | None:
        """Serializa y guarda datos en disco.

        Parameters
        ----------
        data : Any
            Datos a persistir (ej. diccionario con resultados del caso de uso).
        filename : Optional[str], optional
            Nombre del archivo. Si es None, se usa un nombre por defecto.

        Returns
        -------
        Optional[Path]
            Ruta del archivo guardado, o None si falló.
        """
        return self._repo.save(data, filename or "lca_results")

    def load(self, filename: str | None = None) -> Any | None:
        """Carga datos desde disco.

        Parameters
        ----------
        filename : Optional[str], optional
            Nombre del archivo. Si es None, se usa el nombre por defecto.

        Returns
        -------
        Optional[Any]
            Datos cargados, o None si el archivo no existe.
        """
        try:
            return self._repo._storage.load(filename or "lca_results")
        except FileNotFoundError:
            return None

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si el archivo de caché existe.

        Parameters
        ----------
        filename : Optional[str], optional
            Nombre del archivo. Si es None, se usa el nombre por defecto.

        Returns
        -------
        bool
            True si el archivo existe, False en caso contrario.
        """
        return self._repo._storage.exists(filename or "lca_results")


# ==============================================================================
# Función Fábrica Principal
# ==============================================================================


def create_run_lca_use_case(
    config: AppConfig,
    connector: BrightwayConnector,
    local_db_name: str,
    process_to_component: dict[Any, str],
    output_dir: str | Path,
) -> RunLCAUseCase:
    """Factory method que compone todas las dependencias de RunLCAUseCase.

    Parameters
    ----------
    config : AppConfig
        Configuración global de la aplicación.
    connector : BrightwayConnector
        Conector ya inicializado y conectado a Brightway2.
    local_db_name : str
        Nombre de la base de datos local del inventario.
    process_to_component : Dict[Any, str]
        Mapeo de procesos de Ecoinvent a componentes del dominio.
    output_dir : str | Path
        Directorio de salida para la persistencia de caché.

    Returns
    -------
    RunLCAUseCase
        Instancia del caso de uso lista para ejecutar con todas sus
        dependencias inyectadas.

    Example
    -------
    >>> config = AppConfig()
    >>> connector = BrightwayConnector(project_name="mi_proyecto")
    >>> connector.connect()
    >>> use_case = create_run_lca_use_case(
    ...     config=config,
    ...     connector=connector,
    ...     local_db_name="mi_proyecto",
    ...     process_to_component={"process_key": "component_id"},
    ...     output_dir="./resultados",
    ... )
    >>> result = use_case.run(patron_metodo="ReCiPe 2016", nivel_metodo="midpoint (H)")
    """
    bd = connector.get_data_module()
    bc = connector.get_calc_module()

    return RunLCAUseCase(
        method_filter=_MethodFilterAdapter(bd),
        lca_calculator=_LCACalculatorAdapter(
            bc, bd, local_db_name, process_to_component
        ),
        normalizer=_NormalizationAdapter(),
        persistence=_PersistenceAdapter(output_dir),
    )
