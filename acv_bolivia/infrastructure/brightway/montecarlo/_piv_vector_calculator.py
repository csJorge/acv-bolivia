"""
infrastructure/brightway/montecarlo/_piv_vector_calculator:
Cálculo de vectores h para el método PIV.

Responsabilidad única: calcular h(i) = LCA determinístico unitario
por componente y método ambiental, indexando Ecoinvent en caché.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

from ....core.domain.contracts import MethodId
from ....core.domain.models import Project
from ...brightway.constants import DEFAULT_LOCATION
from ...brightway.ei_name_resolver import EcoinventNameResolver

logger = logging.getLogger(__name__)


class PivVectorCalculator:
    """Calcula los vectores h analíticos para el método PIV."""

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        ecoinvent_db_name: str,
    ) -> None:
        self.bc = bc_module
        self.bd = bd_module
        self.ecoinvent_db_name = ecoinvent_db_name

    def calculate(
        self,
        project: Project,
        methods: list[MethodId],
        technical_map: dict[str, str],
        location_map: dict[str, str],
        code_map: dict[str, str] | None = None,
        unit_map: dict[str, str] | None = None,
    ) -> dict[MethodId, dict[str, float]]:
        """Calcula h(i) por componente y método.

        Parameters
        ----------
        project : Project
            Entidad del dominio con los exchanges.
        methods : List[MethodId]
            Lista de métodos de impacto.
        technical_map : Dict[str, str]
            Mapeo {componente: proceso_ei}.
        location_map : Dict[str, str]
            Mapeo {componente: ubicación_ei}.
        code_map : Dict[str, str], optional
            Mapeo {componente: código_ei}.
        unit_map : Dict[str, str], optional
            Mapeo {componente: unidad_ei}.

        Returns
        -------
        Dict[MethodId, Dict[str, float]]
            {method_id: {component_id: h_value}}.
        """
        ei_db = self.bd.Database(self.ecoinvent_db_name)
        ei_resolver = EcoinventNameResolver(ei_db)
        ei_resolver.build_index(technical_map, location_map)

        for warning in ei_resolver.warnings:
            logger.warning(warning)

        code_map = code_map or {}
        unit_map = unit_map or {}

        ei_cache = {
            comp: ei_resolver.resolve(
                comp,
                technical_map[comp],
                str(location_map.get(comp, DEFAULT_LOCATION)).strip(),
                unit=unit_map.get(comp),
                code=code_map.get(comp),
            )
            for comp in technical_map
        }

        self._audit_integrity(project, technical_map, ei_cache)

        h_vectors: dict[MethodId, dict[str, float]] = {}
        for method in methods:
            h_vectors[method] = {}
            for exc in project.exchanges:
                if exc.exchange_type != "technosphere":
                    continue
                comp = exc.component_id
                ei_act = ei_cache.get(comp)
                if ei_act is None:
                    continue

                try:
                    lca_h = self.bc.LCA({ei_act: 1.0}, method)
                    lca_h.lci()
                    lca_h.lcia()
                    h_vectors[method][comp] = float(lca_h.score)
                except Exception as e:
                    logger.exception(
                        "Falló el cálculo del vector unitario h(%s): %s",
                        comp,
                        type(e).__name__,
                    )

        logger.info(
            "Completado. %d/%d coeficientes lineales consolidados.",
            len(next(iter(h_vectors.values()), {})),
            len(technical_map),
        )
        return h_vectors

    def _audit_integrity(
        self,
        project: Project,
        technical_map: dict[str, str],
        ei_cache: dict[str, Any],
    ) -> None:
        """Audita la integridad del mapeo técnico."""
        all_domain_components = sorted(
            {
                e.component_id
                for e in project.exchanges
                if e.exchange_type == "technosphere"
            }
        )
        missing_tech_map = [c for c in all_domain_components if c not in technical_map]
        missing_ei_match = [
            c for c in all_domain_components if c in technical_map and c not in ei_cache
        ]

        if missing_tech_map:
            logger.warning(
                "%d componente(s) sin entrada en tech_map. Excluidos: %s",
                len(missing_tech_map),
                missing_tech_map,
            )
        if missing_ei_match:
            logger.warning(
                "%d componente(s) mapeados pero sin coincidencia en Ecoinvent. "
                "Excluidos: %s",
                len(missing_ei_match),
                missing_ei_match,
            )
