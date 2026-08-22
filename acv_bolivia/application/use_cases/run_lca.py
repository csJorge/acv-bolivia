"""
application.use_cases.run_lca - Caso de uso: cálculo LCIA determinístico.

Ejecuta el cálculo LCIA para todos los proyectos del inventario y todos
los métodos que coincidan con el patrón especificado. Calcula además
los hotspots (top N insumos dominantes) por proyecto y método.

No conoce implementaciones concretas de infraestructura. Todas las
dependencias se inyectan mediante Protocols (DIP).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ...application.contracts import (
    LCIAComputingStrategy,
    MethodFilteringStrategy,
    ResultsNormalizationStrategy,
    ResultsPersistenceStrategy,
)
from ...application.dto.run_lca import RunLCAResult
from ...core.domain.models import HotspotResult, LCAResult

if TYPE_CHECKING:
    from ...infrastructure.brightway.dto import LCACalculationResult


logger = logging.getLogger(__name__)


class RunLCAUseCase:
    """Ejecuta el cálculo LCIA determinístico completo.

    Orquesta:
    - Filtrado de métodos (MethodFilteringStrategy)
    - Cálculo LCIA (LCIAComputingStrategy)
    - Conversión de DTOs a entidades de dominio
    - Normalización por generación (ResultsNormalizationStrategy)
    - Persistencia opcional (ResultsPersistenceStrategy)

    No conoce implementaciones concretas.
    """

    def __init__(
        self,
        method_filter: MethodFilteringStrategy,
        lca_calculator: LCIAComputingStrategy,
        normalizer: ResultsNormalizationStrategy,
        persistence: ResultsPersistenceStrategy[RunLCAResult] | None = None,
    ) -> None:
        """Inyección de dependencias mediante constructor.

        Parameters
        ----------
        method_filter : MethodFilteringStrategy
            Estrategia de filtrado de métodos.
        lca_calculator : LCIAComputingStrategy
            Estrategia de cálculo LCIA.
        normalizer : ResultsNormalizationStrategy
            Estrategia de normalización por generación.
        persistence : Optional[ResultsPersistenceStrategy]
            Estrategia de persistencia. Si es None, no se guarda caché.
        """
        self.method_filter = method_filter
        self.lca_calculator = lca_calculator
        self.normalizer = normalizer
        self.persistence = persistence

    def run(
        self,
        patron_metodo: str = "ReCiPe 2016",
        nivel_metodo: str = "midpoint (H)",
        top_n_hotspot: int = 50,
        functional_unit: float = 1.0,
        generation_dict: dict[str, float] | None = None,
        save_cache: bool = False,
        cache_filename: str | None = None,
    ) -> RunLCAResult:
        """Ejecuta el cálculo LCIA completo.

        Parameters
        ----------
        patron_metodo : str
            Patrón para filtrar métodos (ej. 'ReCiPe 2016').
        nivel_metodo : str
            Nivel del método (ej. 'midpoint (H)').
        top_n_hotspot : int
            Número de insumos top a registrar por actividad.
        functional_unit : float
            Cantidad de unidad funcional.
        generation_dict : Optional[Dict[str, float]]
            {project_name: kwh_generados}. Requerido para normalización.
        save_cache : bool
            Si True y hay persistence, guarda resultados en disco.
        cache_filename : Optional[str]
            Nombre del archivo. Si None, usa el default.

        Returns
        -------
        RunLCAResult
            Resultado con entidades de dominio normalizadas.
        """
        start_time = time.time()

        # 1. Filtrar métodos
        logger.info(
            "Filtrando métodos: patron='%s', nivel='%s'", patron_metodo, nivel_metodo
        )
        methods = self.method_filter.filter(patron=patron_metodo, nivel=nivel_metodo)

        if not methods:
            logger.warning(
                "No se encontraron métodos para '%s' / '%s'.",
                patron_metodo,
                nivel_metodo,
            )
            return RunLCAResult(
                success=False,
                error_message=(
                    f"No se encontraron métodos para '{patron_metodo}' / "
                    f"'{nivel_metodo}'."
                ),
            )

        # 2. Calcular LCIA
        logger.info(
            "Calculando LCIA: %d métodos, top_n=%d", len(methods), top_n_hotspot
        )
        calc_result: LCACalculationResult = self.lca_calculator.compute(
            methods=methods,
            top_n_hotspot=top_n_hotspot,
            functional_unit=functional_unit,
        )

        gen_dict = generation_dict or {}

        # 3. Convertir DTOs de infraestructura a entidades de dominio
        lca_results = self._to_domain_lca_results(calc_result, gen_dict)
        hotspots = self._to_domain_hotspots(calc_result, gen_dict)

        # 4. Normalizar por generación
        norm_report = None
        if gen_dict:
            norm_report = self.normalizer.normalize(
                lca_results=lca_results,
                hotspots=hotspots,
                generation_dict=gen_dict,
            )
            logger.info("Normalización completada: %s", norm_report)

        # 5. Persistir si se solicita
        cache_path = None
        if save_cache and self.persistence is not None:
            result_to_cache = RunLCAResult(
                lca_results=lca_results,
                hotspots=hotspots,
                methods=methods,
                norm_report=norm_report,
            )
            saved = self.persistence.save(result_to_cache, filename=cache_filename)
            cache_path = saved

        elapsed = time.time() - start_time
        logger.info("Cálculo LCIA concluido en %.1fs", elapsed)

        return RunLCAResult(
            lca_results=lca_results,
            hotspots=hotspots,
            methods=methods,
            norm_report=norm_report,
            cache_path=cache_path,
            elapsed_seconds=elapsed,
            success=True,
        )

    # ------------------------------------------------------------------
    # Métodos privados de mapeo
    # ------------------------------------------------------------------

    def _to_domain_lca_results(
        self,
        calc_result: LCACalculationResult,
        generation_dict: dict[str, float],
    ) -> list[LCAResult]:
        """Convierte DeterministicScoreDTO a LCAResult del dominio."""
        results: list[LCAResult] = []
        for dto in calc_result.scores:
            gen = generation_dict.get(dto.project_id, 1.0)
            score_per_kwh = dto.score / gen if gen > 0 else None
            results.append(
                LCAResult(
                    project_id=dto.project_id,
                    method_id=dto.method_id,
                    method_label=dto.method_label,
                    score=dto.score,
                    score_per_kwh=score_per_kwh,
                )
            )
        return results

    def _to_domain_hotspots(
        self,
        calc_result: LCACalculationResult,
        generation_dict: dict[str, float],
    ) -> list[HotspotResult]:
        """Convierte HotspotDTO a HotspotResult del dominio."""
        results: list[HotspotResult] = []
        for dto in calc_result.hotspots:
            gen = generation_dict.get(dto.project_id, 1.0)
            impact_per_kwh = dto.impact / gen if gen > 0 else None
            results.append(
                HotspotResult(
                    project_id=dto.project_id,
                    method_id=dto.method_id,
                    component_id=dto.component_id,
                    background_process_name=dto.background_process_name,
                    impact=dto.impact,
                    impact_per_kwh=impact_per_kwh,
                    unit=dto.unit,
                )
            )
        return results
