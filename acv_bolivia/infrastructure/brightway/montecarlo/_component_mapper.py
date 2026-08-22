"""
infrastructure/brightway/montecarlo/_component_mapper:
Mapeo de componentes del dominio a posiciones en la matriz de tecnosfera.

Responsabilidad única: resolver la correspondencia entre componentes del dominio
y posiciones (arr_i) en tech_params, aplicando desambiguación por metadata inyectada
o por (proceso, monto) exacto.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ....core.domain.models import Project
from ....infrastructure.brightway.constants import DEFAULT_LOCATIONS

logger = logging.getLogger(__name__)


class ComponentToMatrixMapper:
    """Mapea componentes del dominio a posiciones en la matriz de tecnosfera."""

    def map(
        self,
        lca: Any,
        bw_activity: Any,
        project: Project,
        act_dict: dict[Any, int],
        technical_map: dict[str, str],
        location_map: dict[str, str],
    ) -> tuple[list[tuple[int, str]], dict[int, float]]:
        """Mapea componentes a posiciones (arr_i) en tech_params.

        Parameters
        ----------
        lca : Any
            Instancia LCA de BW2 ya factorizada.
        bw_activity : Any
            Actividad BW2 del proyecto.
        project : Project
            Objeto Project del dominio.
        act_dict : Dict[Any, int]
            activity_dict de la instancia LCA.
        technical_map : Dict[str, str]
            Mapeo {componente: proceso_ei}.
        location_map : Dict[str, str]
            Mapeo {componente: ubicación_ei}.

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[int, float]]
            (relevant_rows, baseline) donde relevant_rows es una lista de
            (arr_i, component_name) y baseline es {arr_i: amount_nominal}.
        """
        output_idx = int(act_dict.get(bw_activity.key, -1))

        # Índice de exchanges BW: (nombre_lower, loc) -> lista de candidatos
        bw_exc_index = self._build_bw_exchange_index(bw_activity, act_dict)

        # Mapear componentes del dominio a posiciones en tech_params
        comp_to_row_idx = self._map_domain_to_bw(
            project, bw_exc_index, technical_map, location_map
        )

        # Resolver posiciones en tech_params
        relevant_rows, baseline = self._resolve_tp_positions(
            lca, project, comp_to_row_idx, output_idx
        )

        return relevant_rows, baseline

    def _build_bw_exchange_index(
        self, bw_activity: Any, act_dict: dict[Any, int]
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Construye índice de exchanges BW2 por (nombre, ubicación)."""
        bw_exc_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for exc in bw_activity.technosphere():
            in_key = exc.input.key
            row_idx = act_dict.get(in_key)
            if row_idx is None:
                continue
            key = (
                exc.input.get("name", "").strip().lower(),
                exc.input.get("location", "GLO"),
            )
            bw_exc_index.setdefault(key, []).append(
                {
                    "row_idx": int(row_idx),
                    "amount": float(exc["amount"]),
                    "in_key": in_key,
                    "list_pos": len(bw_exc_index.get(key, [])),
                }
            )
        return bw_exc_index

    def _map_domain_to_bw(
        self,
        project: Project,
        bw_exc_index: dict[tuple[str, str], list[dict[str, Any]]],
        technical_map: dict[str, str],
        location_map: dict[str, str],
    ) -> dict[str, int]:
        """Mapea componentes del dominio a row_idx de BW2."""
        domain_exchanges = [
            e for e in project.exchanges if e.exchange_type == "technosphere"
        ]
        comp_to_row_idx: dict[str, int] = {}
        used_positions: set[tuple[tuple[str, str], int]] = set()

        for exc_dom in domain_exchanges:
            comp = exc_dom.component_id
            dom_amt = float(exc_dom.quantity.amount)

            # Buscar candidato por (proceso, ubicación) con fallbacks
            candidates = self._find_candidates(
                comp, dom_amt, bw_exc_index, technical_map, location_map
            )

            # Seleccionar mejor candidato no usado
            best = self._select_best_candidate(candidates, dom_amt, used_positions)
            if best is not None:
                comp_to_row_idx[comp] = best["row_idx"]
                used_positions.add((best["lookup_key"], best["list_pos"]))

        return comp_to_row_idx

    def _find_candidates(
        self,
        comp: str,
        dom_amt: float,
        bw_exc_index: dict[tuple[str, str], list[dict[str, Any]]],
        technical_map: dict[str, str],
        location_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Busca candidatos por (proceso, ubicación) con fallbacks."""
        ei_name = str(technical_map.get(comp, "")).strip().lower()
        if not ei_name:
            return []

        # Intentar con ubicación específica y fallbacks
        for loc in [location_map.get(comp, "GLO")] + DEFAULT_LOCATIONS:
            key = (ei_name, loc.strip())
            if key in bw_exc_index:
                candidates = bw_exc_index[key]
                for c in candidates:
                    c["lookup_key"] = key
                return candidates

        return []

    def _select_best_candidate(
        self,
        candidates: list[dict[str, Any]],
        dom_amt: float,
        used_positions: set[tuple[tuple[str, str], int]],
    ) -> dict[str, Any] | None:
        """Selecciona el mejor candidato no usado por cercanía de monto."""
        best, best_rel = None, float("inf")
        for c in candidates:
            pid = (c["lookup_key"], c["list_pos"])
            if pid in used_positions:
                continue
            rel = (
                abs(c["amount"] - dom_amt) / dom_amt
                if dom_amt > 0
                else abs(c["amount"])
            )
            if rel < best_rel:
                best_rel, best = rel, c
        return best

    def _resolve_tp_positions(
        self,
        lca: Any,
        project: Project,
        comp_to_row_idx: dict[str, int],
        output_idx: int,
    ) -> tuple[list[tuple[int, str]], dict[int, float]]:
        """Resuelve posiciones en tech_params para cada componente."""
        tp = lca.tech_params
        tp_index: dict[tuple[int, int], list[int]] = {}
        for i, r in enumerate(tp):
            tp_index.setdefault((int(r["row"]), int(r["col"])), []).append(i)

        domain_exchanges = [
            e for e in project.exchanges if e.exchange_type == "technosphere"
        ]
        relevant_rows: list[tuple[int, str]] = []
        used_tp_positions: set[int] = set()
        comps_by_rc: dict[tuple[int, int], list[tuple[str, float]]] = defaultdict(list)

        for comp, row_idx in comp_to_row_idx.items():
            dom_amt = float(
                next(
                    e.quantity.amount
                    for e in domain_exchanges
                    if e.component_id == comp
                )
            )
            comps_by_rc[(row_idx, output_idx)].append((comp, dom_amt))

        for (row_idx, col_idx), comp_amt_list in comps_by_rc.items():
            tp_positions = tp_index.get((row_idx, col_idx), [])
            if not tp_positions:
                continue
            for comp, dom_amt in sorted(
                comp_amt_list, key=lambda x: x[1], reverse=True
            ):
                best_pos, best_rel = None, float("inf")
                for pos in tp_positions:
                    if pos in used_tp_positions:
                        continue
                    rel = (
                        abs(float(tp[pos]["amount"]) - dom_amt) / dom_amt
                        if dom_amt > 0
                        else abs(float(tp[pos]["amount"]))
                    )
                    if rel < best_rel:
                        best_rel, best_pos = rel, pos
                if best_pos is not None:
                    relevant_rows.append((best_pos, comp))
                    used_tp_positions.add(best_pos)

        baseline = {arr_i: float(tp[arr_i]["amount"]) for arr_i, _ in relevant_rows}
        return relevant_rows, baseline
