"""
core.services.normalization: Normalización de scores por generación eléctrica.

Convierte scores de impacto absolutos a scores por unidad funcional (1 kWh).
Opera sobre las listas de objetos LCAResult y HotspotResult, manejando de
forma segura dataclasses inmutables (frozen=True) mediante dataclasses.replace.

Fórmula:
    score_per_kwh = score_absoluto / generacion_kwh(proyecto)

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import dataclasses
import logging
import math

from ...core.domain.models import HotspotResult, LCAResult

logger = logging.getLogger(__name__)


class NormalizationReport:
    """Resultado de una operación de normalización."""

    def __init__(self) -> None:
        self.normalized: int = 0
        self.skipped: int = 0
        self.errors_count: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.errors_count += 1
        logger.error(msg)

    def __str__(self) -> str:
        return (
            f"Normalización completada: {self.normalized} resultados normalizados, "
            f"{self.skipped} omitidos, {self.errors_count} errores."
        )

    def log_detail(self) -> None:
        """Registra el reporte completo con advertencias y errores."""
        logger.info(str(self))
        for w in self.warnings:
            logger.warning("  %s", w)
        for e in self.errors:
            logger.error("  %s", e)


def normalize_by_generation(
    lca_results: list[LCAResult],
    hotspots: list[HotspotResult],
    generation_dict: dict[str, float],
    overwrite: bool = True,
) -> NormalizationReport:
    """Normaliza scores por energía generada.

    Parameters
    ----------
    lca_results : List[LCAResult]
        Lista de resultados determinísticos a normalizar.
    hotspots : List[HotspotResult]
        Lista de hotspots a normalizar.
    generation_dict : Dict[str, float]
        {project_id: kwh_generados}.
    overwrite : bool
        Si False, salta proyectos que ya tienen score_per_kwh calculado.

    Returns
    -------
    NormalizationReport
        Detalle de qué se normalizó y qué falló.
    """
    report = NormalizationReport()

    def resolve_generation(project_id: str) -> float:
        gen = generation_dict.get(project_id)
        if gen is None:
            report.add_warning(
                f"'{project_id}' no está en generation_dict. "
                f"Se usará 1.0 como fallback (sin normalización real)."
            )
            return 1.0

        if not math.isfinite(gen) or gen <= 0.0:
            report.add_error(
                f"'{project_id}': generación debe ser finita y > 0. "
                f"Se asignará 0.0 para evitar una normalización inválida."
            )
            return 0.0

        return gen

    # --- LCAResults ---
    for i, result in enumerate(lca_results):
        if not overwrite and result.score_per_kwh is not None:
            report.skipped += 1
            continue

        gen = resolve_generation(result.project_id)

        if gen == 0:
            # Usar dataclasses.replace para dataclasses frozen
            lca_results[i] = dataclasses.replace(result, score_per_kwh=0.0)
            continue

        new_score_per_kwh = result.score / gen
        # Usar dataclasses.replace para dataclasses frozen
        lca_results[i] = dataclasses.replace(result, score_per_kwh=new_score_per_kwh)
        report.normalized += 1

    # --- HotspotResults ---
    for i, hotspot in enumerate(hotspots):
        if not overwrite and hotspot.impact_per_kwh is not None:
            report.skipped += 1
            continue

        gen = resolve_generation(hotspot.project_id)

        if gen == 0:
            hotspots[i] = dataclasses.replace(hotspot, impact_per_kwh=0.0)
            continue

        new_impact_per_kwh = hotspot.impact / gen
        # Usar dataclasses.replace para dataclasses frozen
        hotspots[i] = dataclasses.replace(hotspot, impact_per_kwh=new_impact_per_kwh)
        report.normalized += 1

    return report
