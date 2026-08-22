"""
application.contracts: Protocolos de la Capa de Aplicación.

Define los contratos (Protocols) que la capa de aplicación requiere de la
infraestructura y de módulos auxiliares (analysis/).

Los compositores en infrastructure/composition/ son responsables de inyectar
las implementaciones concretas que satisfacen estos protocolos.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from numpy.typing import NDArray

from ..core.domain.contracts import MethodId  # , SensitivityAnalyzer
from ..core.domain.models import HotspotResult, LCAResult, Project
from ..core.domain.validators import ValidationReport
from ..core.services.normalization import NormalizationReport

if TYPE_CHECKING:
    from ..infrastructure.brightway.dto import (
        ForegroundSimulationResult,
        LCACalculationResult,
        MonteCarloSimulationResult,
        PIVSimulationResult,
    )


# ==============================================================================
# Contratos de Carga de Inventario
# ==============================================================================

T = TypeVar("T")


class InventoryLoader(Protocol):
    """Carga el inventario desde cualquier fuente (Excel, JSON, DB, etc.)."""

    def load(self, force_reload: bool = False) -> Any:
        """Carga el inventario y retorna un InventoryLoadResult.

        Parameters
        ----------
        force_reload : bool
            Si True, ignora caché y recarga desde la fuente.

        Returns
        -------
        Any
            InventoryLoadResult con proyectos, mapeos y generación.
        """
        ...


class InventoryDataAuditor(Protocol):
    """Audita la calidad de datos de entrada."""

    def audit(
        self,
        project_rows: list[dict],
        mc_rows: list[dict],
        mc_mode: str,
    ) -> ValidationReport:
        """Audita filas crudas del inventario.

        Parameters
        ----------
        project_rows : List[dict]
            Filas crudas de la hoja Proyectos.
        mc_rows : List[dict]
            Filas crudas de la hoja MC.
        mc_mode : str
            Modo de simulación ('full', 'foreground', 'piv').

        Returns
        -------
        ValidationReport
            Reporte con errores y advertencias.
        """
        ...


class BrightwayConnectionManager(Protocol):
    """Gestiona la conexión a Brightway2."""

    def connect(self) -> None:
        """Establece la conexión y activa el proyecto."""
        ...

    def disconnect(self) -> None:
        """Libera recursos y limpia el estado."""
        ...

    def get_local_db_name(self) -> str:
        """Retorna el nombre de la base de datos local.

        Returns
        -------
        str
            Nombre de la BD local.
        """
        ...

    def get_data_module(self) -> Any:
        """Retorna el módulo bw2data.

        Returns
        -------
        Any
            Referencia al módulo bw2data.
        """
        ...

    def get_calc_module(self) -> Any:
        """Retorna el módulo bw2calc.

        Returns
        -------
        Any
            Referencia al módulo bw2calc.
        """
        ...


class BrightwayActivityBuilder(Protocol):
    """Construye actividades en Brightway2."""

    def build(
        self,
        projects: list[Project],
        location_map: dict[str, str],
        technical_map: dict[str, str],
        code_map: dict[str, str] | None = None,
        unit_map: dict[str, str] | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """Compila y sincroniza la base de datos local.

        Parameters
        ----------
        projects : List[Project]
            Proyectos del dominio.
        location_map : Dict[str, str]
            Mapeo {componente: ubicación_ei}.
        technical_map : Dict[str, str]
            Mapeo {componente: proceso_ei}.
        code_map : Dict[str, str], optional
            Mapeo {componente: código_ei}.
        unit_map : Dict[str, str], optional
            Mapeo {componente: unidad_ei}.
        force_rebuild : bool
            Si True, borra la BD existente y reconstruye.
        """
        ...


# ==============================================================================
# Contratos de Cálculo LCIA
# ==============================================================================


class MethodFilteringStrategy(Protocol):
    """Filtra métodos de impacto según patrón y nivel."""

    def filter(
        self,
        patron: str,
        nivel: str,
        exclude_lt: bool = True,
    ) -> list[MethodId]:
        """Retorna métodos que coinciden con los criterios.

        Parameters
        ----------
        patron : str
            Patrón de búsqueda (ej. 'ReCiPe 2016').
        nivel : str
            Nivel del método (ej. 'midpoint (H)').
        exclude_lt : bool
            Si True, excluye métodos de largo plazo.

        Returns
        -------
        List[MethodId]
            Lista de métodos que coinciden.
        """
        ...


class LCIAComputingStrategy(Protocol):
    """Calcula LCIA determinístico."""

    def compute(
        self,
        methods: list[MethodId],
        top_n_hotspot: int,
        functional_unit: float,
    ) -> LCACalculationResult:
        """Calcula LCIA y retorna DTOs.

        Parameters
        ----------
        methods : List[MethodId]
            Métodos de impacto a calcular.
        top_n_hotspot : int
            Número de hotspots por proyecto/método.
        functional_unit : float
            Unidad funcional.

        Returns
        -------
        LCACalculationResult
            DTOs con scores y hotspots.
        """
        ...


# ==============================================================================
# Contratos de Simulación Monte Carlo
# ==============================================================================


class MonteCarloSimulationStrategy(Protocol):
    """Ejecuta simulación Monte Carlo completa."""

    def run(
        self,
        iterations: int,
        functional_unit: float = 1.0,
    ) -> MonteCarloSimulationResult:
        """Ejecuta la simulación y retorna DTOs.

        Parameters
        ----------
        iterations : int
            Número de iteraciones.
        functional_unit : float
            Unidad funcional.

        Returns
        -------
        MonteCarloSimulationResult
            DTOs con scores simulados.
        """
        ...

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        ...


class ForegroundSimulationStrategy(Protocol):
    """Ejecuta simulación Foreground MC."""

    def run(self, iterations: int) -> list[ForegroundSimulationResult]:
        """Ejecuta la simulación y retorna DTOs.

        Parameters
        ----------
        iterations : int
            Número de iteraciones.

        Returns
        -------
        List[ForegroundSimulationResult]
            DTOs con scores simulados (uno por proyecto).
        """
        ...

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        ...


class PIVSimulationStrategy(Protocol):
    """Ejecuta simulación PIV MC."""

    def run(self, iterations: int) -> list[PIVSimulationResult]:
        """Ejecuta la simulación y retorna DTOs.

        Parameters
        ----------
        iterations : int
            Número de iteraciones.

        Returns
        -------
        List[PIVSimulationResult]
            DTOs con scores simulados (uno por proyecto).
        """
        ...

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        ...


# ==============================================================================
# Contratos de Procesamiento de Muestras
# ==============================================================================


class SampleProcessingStrategy(Protocol):
    """Aplica reglas físicas sobre muestras vectorizadas."""

    def __call__(
        self,
        samples: dict[str, Any],
    ) -> dict[str, Any]:
        """Procesa las muestras y retorna las procesadas.

        Parameters
        ----------
        samples : Dict[str, Any]
            Muestras brutas {componente: array}.

        Returns
        -------
        Dict[str, Any]
            Muestras procesadas con reglas aplicadas.
        """
        ...


class SampleProcessorFactory(Protocol):
    """Fábrica de procesadores de muestras (FG/PIV) por proyecto.

    Crea un SampleProcessingStrategy por proyecto mezclando reglas globales y
    específicas de configuración de Monte Carlo.
    """

    def __call__(
        self,
        mc_config: dict[str, dict[str, Any]],
        projects: list[Project],
        dependency_config: dict[str, dict[str, Any]] | None = None,
        mix_config: dict[float, list[str]] | None = None,
        verbose: bool = False,
        enforce_physical_constraints: bool = True,
    ) -> dict[str, SampleProcessingStrategy]:
        """Crea los procesadores de muestras para cada proyecto.

        Parameters
        ----------
        mc_config : dict[str, dict[str, Any]]
            Configuración por proyecto (con clave "GLOBAL" para reglas globales).
        projects : list[Project]
            Proyectos del dominio.
        dependency_config : dict[str, dict[str, Any]] | None
            Reglas de dependencia globales de baja prioridad (fallback si
            mc_config no define reglas para un proyecto).
        mix_config : dict[float, list[str]] | None
            Restricciones de suma globales de baja prioridad.
        verbose : bool, optional
            Si True, activa logs verbosos.
        enforce_physical_constraints : bool, optional
            Si True, aplica restricciones físicas a las muestras.

        Returns
        -------
        dict[str, SampleProcessingStrategy]
            {project_name: procesador de muestras}.
        """
        ...


# ==============================================================================
# Contratos de Normalización y Persistencia
# ==============================================================================


class ResultsNormalizationStrategy(Protocol):
    """Normaliza resultados por generación eléctrica."""

    def normalize(
        self,
        lca_results: list[LCAResult],
        hotspots: list[HotspotResult],
        generation_dict: dict[str, float],
    ) -> NormalizationReport:
        """Normaliza y retorna reporte.

        Parameters
        ----------
        lca_results : List[LCAResult]
            Resultados LCIA a normalizar.
        hotspots : List[HotspotResult]
            Hotspots a normalizar.
        generation_dict : Dict[str, float]
            {project_name: kwh_generados}.

        Returns
        -------
        NormalizationReport
            Reporte de normalización.
        """
        ...


class ResultsPersistenceStrategy(Protocol[T]):
    """Persiste resultados en disco."""

    def save(self, data: T, filename: str | None = None) -> Path | None:
        """Serializa y guarda datos.

        Parameters
        ----------
        data : Any
            Datos a persistir.
        filename : str | None
            Nombre del archivo. Si None, usa default.

        Returns
        -------
        Optional[Path]
            Ruta del archivo guardado, o None si falló.
        """
        ...

    def load(self, filename: str | None = None) -> T | None:
        """Carga datos desde disco.

        Parameters
        ----------
        filename : str | None
            Nombre del archivo. Si None, usa default.

        Returns
        -------
        Any | None
            Datos cargados, o None si no existe.
        """
        ...

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si el archivo existe.

        Parameters
        ----------
        filename : str | None
            Nombre del archivo. Si None, usa default.

        Returns
        -------
        bool
            True si existe.
        """
        ...


# ==============================================================================
# Contratos de Lectura de Resultados MC
# ==============================================================================


class MCResultsReader(Protocol):
    """Lee resultados Monte Carlo persistidos."""

    def get_mc_results(
        self,
        project_name: str | None = None,
        method_name: Any | None = None,
    ) -> list[Any]:
        """Retorna resultados MC filtrados por proyecto y/o método.

        Parameters
        ----------
        project_name : str | None
            Nombre del proyecto.
        method_name : Any | None
            Método de impacto (str o tupla).

        Returns
        -------
        List[Any]
            Lista de resultados MC.
        """
        ...

    def get_component_samples(self, project_name: str) -> dict[str, Any] | None:
        """Retorna muestras de componentes de un proyecto.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto.

        Returns
        -------
        dict[str, Any] | None
            Muestras {componente: array} o None.
        """
        ...


# ==============================================================================
# Contratos de Estadísticas MC
# ==============================================================================


class MCStatsCalculationStrategy(Protocol):
    """Calcula estadísticas descriptivas para resultados Monte Carlo."""

    def calculate(
        self,
        scores: dict[MethodId, dict[str, NDArray]],
        generation_dict: dict[str, float],
    ) -> list[Any]:
        """Calcula y retorna estadísticas descriptivas (ej. MCStats o
        MonteCarloProjectStats)."""
        ...


# ==============================================================================
# Contratos de Análisis de Sensibilidad
# ==============================================================================


class SensitivityEngineStrategy(Protocol):
    """Motor de análisis de sensibilidad.

    Coordina la ejecución de múltiples analizadores sobre un proyecto/método.
    """

    def run(
        self,
        project_name: str,
        method_tuple: MethodId,
        component_samples_raw: dict[str, list[float]] | None = None,
        lca_scores_raw: Any | None = None,
    ) -> Any:
        """Ejecuta el análisis de sensibilidad.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto.
        method_tuple : MethodId
            Método de impacto.
        component_samples_raw : dict[str, list[float]]
            Muestras de componentes {componente: lista_valores}.
        lca_scores_raw : Any | Non
            Scores MC ya calculados.

        Returns
        -------
        Any
            Reporte de sensibilidad (SensitivityReport de analysis/).
        """
        ...


class LCAInfrastructureProvider(Protocol):
    """Proveedor de infraestructura LCA para análisis de sensibilidad.

    Implementa el contrato de core.domain.contracts.LCAInfrastructureProvider.
    Se incluye aquí para conveniencia de los compositores.
    """

    def get_nominal_parameters(self, project_name: str) -> dict[str, float]:
        """Retorna parámetros nominales del proyecto."""
        ...

    def create_evaluator(
        self,
        project_name: str,
        method_id: MethodId,
        functional_unit_amount: float = 1.0,
    ) -> Any:
        """Crea un evaluador LCA."""
        ...

    def create_piv_evaluator(self, nominal_params: dict[str, float]) -> Any | None:
        """Crea un evaluador PIV si es posible."""
        ...

    def get_latest_mapping_diagnostic(self) -> Any | None:
        """Retorna diagnóstico del último mapeo."""
        ...

    def extract_piv_h_vectors(
        self, lca_result: Any, project_name: str, method_name: str
    ) -> dict[str, float]:
        """Extrae vectores h para PIV."""
        ...
