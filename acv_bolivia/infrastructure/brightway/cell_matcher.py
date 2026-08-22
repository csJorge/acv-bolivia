"""Matching exacto de celdas entre el dominio y la matriz de Brightway2.

Responsabilidad única: resolver la correspondencia entre componentes
del dominio y celdas (row, col) de la matriz de tecnosfera, usando
metadata inyectada (component_id) para matching exacto, con fallback
por (proceso, monto) exacto si la metadata no está disponible.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ...core.domain.models import Project
from ...infrastructure.brightway.constants import (
    MATCHING_AMOUNT_EPSILON,
    TECHNOSPHERE_EXCHANGE,
)

logger = logging.getLogger(__name__)


class CellMatchResult:
    """Resultado del matching de celdas."""

    def __init__(
        self,
        matrix_cell_to_comps: dict[tuple[int, int], list[str]],
        matched_components: set[str],
        unmatched_components: list[str],
        shared_cells_groups: list[list[str]],
        mapped_cells_count: int,
    ) -> None:
        """Inicializa un resultado inmutable en cuanto a sus colecciones públicas.

        Parameters
        ----------
        matrix_cell_to_comps : dict[tuple[int, int], list[str]]
            Componentes agrupados por celda física de la matriz.
        matched_components : set[str]
            Componentes encontrados con éxito.
        unmatched_components : list[str]
            Componentes que no pudieron asociarse a una celda.
        shared_cells_groups : list[list[str]]
            Grupos de componentes que comparten una celda.
        mapped_cells_count : int
            Número de celdas físicas mapeadas.
        """
        self.matrix_cell_to_comps = matrix_cell_to_comps
        self.matched_components = matched_components
        self.unmatched_components = unmatched_components
        self.shared_cells_groups = shared_cells_groups
        self.mapped_cells_count = mapped_cells_count


class CellMatcher:
    """Resuelve la correspondencia exacta entre componentes del dominio y celdas BW2.

    Estrategia de matching (en orden de prioridad):
    1. Metadata inyectada: exc.get("component") → component_id (exacto, O(1))
    2. Fallback por (proceso, monto) exacto: sin tolerancia (exacto, O(n*m))
    """

    def match(
        self,
        project: Project,
        bw_activity: Any,
        act_dict: dict[Any, int],
        output_idx: int,
        technical_map: dict[str, str],
        location_map: dict[str, str] | None = None,
    ) -> CellMatchResult:
        """Ejecuta el matching por metadata o por proceso y monto exactos.

        Parameters
        ----------
        project : Project
            Proyecto del dominio cuyos exchanges se deben mapear.
        bw_activity : Any
            Actividad Brightway2 que contiene los exchanges.
        act_dict : dict[Any, int]
            Mapeo de claves de entrada a filas de la matriz.
        output_idx : int
            Columna de la actividad consumidora.
        technical_map : dict[str, str]
            Mapeo de componentes a nombres de procesos Ecoinvent.
        location_map : dict[str, str] or None
            Ubicación esperada por componente. Si no se proporciona una
            ubicación, el fallback no filtra por ella.

        Returns
        -------
        CellMatchResult
            Resultado del matching y diagnóstico de componentes no encontrados.

        Raises
        ------
        ValueError
            Si ``output_idx`` es negativo.
        """
        if output_idx < 0:
            raise ValueError("output_idx debe ser un índice no negativo.")
        matrix_cell_to_comps: dict[tuple[int, int], list[str]] = defaultdict(list)
        matched_components: set[str] = set()
        bw_exchanges = list(bw_activity.technosphere())

        # Construir índice de exchanges BW2 por metadata "component" (O(1) lookup)
        bw_by_component_tag: dict[str, list[Any]] = defaultdict(list)
        for exc_bw in bw_exchanges:
            comp_tag = exc_bw.get("component")
            if comp_tag:
                bw_by_component_tag[str(comp_tag).strip().lower()].append(exc_bw)

        # Matching por cada componente del dominio
        for exc_dom in project.exchanges:
            if exc_dom.exchange_type != TECHNOSPHERE_EXCHANGE:
                continue

            comp = exc_dom.component_id
            dom_amt = float(exc_dom.quantity.amount)
            if dom_amt == 0:
                continue

            # Estrategia 1: Matching por metadata inyectada (exacto, O(1))
            tagged_exchanges = bw_by_component_tag.get(comp.strip().lower(), [])
            tagged_matches = [
                exc_bw
                for exc_bw in tagged_exchanges
                if abs(float(exc_bw["amount"]) - dom_amt) < MATCHING_AMOUNT_EPSILON
            ]
            if len(tagged_matches) == 1:
                exc_bw = tagged_matches[0]
                row_idx = act_dict.get(exc_bw.input.key)
                if row_idx is not None:
                    matrix_cell_to_comps[(int(row_idx), output_idx)].append(comp)
                    matched_components.add(comp)
                    continue
            if tagged_exchanges:
                logger.warning(
                    "Metadata inconsistente para '%s': no hay un exchange "
                    "etiquetado con el monto esperado.",
                    comp,
                )
                continue

            # Estrategia 2: Fallback por (proceso, monto) exacto (sin tolerancia)
            expected_ei_name = str(technical_map.get(comp, "")).strip().lower()
            if not expected_ei_name:
                continue
            expected_location = (
                str((location_map or {}).get(comp, "")).strip()
                if location_map and comp in location_map
                else None
            )

            fallback_matches = []
            for exc_bw in bw_exchanges:
                bw_name = exc_bw.input.get("name", "").lower()
                if bw_name != expected_ei_name:
                    continue
                if (
                    expected_location is not None
                    and str(exc_bw.input.get("location", "")).strip()
                    != expected_location
                ):
                    continue

                if abs(float(exc_bw["amount"]) - dom_amt) < MATCHING_AMOUNT_EPSILON:
                    row_idx = act_dict.get(exc_bw.input.key)
                    if row_idx is not None:
                        fallback_matches.append((int(row_idx), exc_bw))

            if len(fallback_matches) == 1:
                row_idx, _ = fallback_matches[0]
                matrix_cell_to_comps[(row_idx, output_idx)].append(comp)
                matched_components.add(comp)
            elif len(fallback_matches) > 1:
                logger.warning(
                    "Match ambiguo para '%s': %d exchanges con proceso y monto "
                    "iguales.",
                    comp,
                    len(fallback_matches),
                )

        # Consolidar métricas de diagnóstico
        all_domain_components = {
            e.component_id
            for e in project.exchanges
            if e.exchange_type == TECHNOSPHERE_EXCHANGE and float(e.quantity.amount) > 0
        }
        unmatched = sorted(all_domain_components - matched_components)
        shared = [v for k, v in matrix_cell_to_comps.items() if len(v) > 1]

        if unmatched:
            logger.warning("Componentes no mapeados: %s", unmatched)

        return CellMatchResult(
            matrix_cell_to_comps=dict(matrix_cell_to_comps),
            matched_components=matched_components,
            unmatched_components=unmatched,
            shared_cells_groups=shared,
            mapped_cells_count=len(matrix_cell_to_comps),
        )
