"""
infrastructure.brightway.adapters: Adaptador concreto para Brightway2.

Implementa el contrato LCAInfrastructureProvider del dominio, orquestando
los componentes internos (CellMatcher, evaluadores, PivVectorExtractor)
para aislar las capas de análisis matemático del ecosistema Brightway2.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any, cast

from ...core.domain.contracts import (
    LcaEvaluator,
    MappingDiagnostic,
    MethodId,
    ParameterDict,
)
from ...core.domain.models import Project
from ...infrastructure.brightway.cell_matcher import CellMatcher
from ...infrastructure.brightway.evaluators import MatrixLcaEvaluator, PivLcaEvaluator
from ...infrastructure.brightway.piv_extractor import PivVectorExtractor

logger = logging.getLogger(__name__)


class BrightwayLCAProvider:
    """Implementación concreta de LCAInfrastructureProvider para Brightway2.

    Orquesta:
    - CellMatcher: matching de componentes a celdas de matriz
    - MatrixLcaEvaluator: evaluación completa
    - PivLcaEvaluator: evaluación lineal aproximada (PIV)
    - PivVectorExtractor: extracción de h-vectors
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        project: Project,
        technical_map: dict[str, str],
        piv_h_vectors: dict[str, float] | None = None,
        sample_processor: Any | None = None,
        location_map: dict[str, str] | None = None,
    ) -> None:
        self.bc = bc_module
        self.bd = bd_module
        self.local_db = bd_module.Database(local_db_name)
        self.project = project
        self.technical_map = technical_map  # Inyectado, no accedido como _technical_map
        self.location_map = location_map or {}

        self.piv_h_vectors = piv_h_vectors or {}
        self.sample_processor = sample_processor

        # Componentes internos (SRP)
        self._cell_matcher = CellMatcher()
        self._piv_extractor = PivVectorExtractor()

        self._latest_diagnostic: MappingDiagnostic | None = None

    # ------------------------------------------------------------------
    # API pública (implementa LCAInfrastructureProvider)
    # ------------------------------------------------------------------

    def get_nominal_parameters(self, project_name: str) -> ParameterDict:
        """Extrae los parámetros nominales del dominio."""
        self._ensure_project_name(project_name)
        return {
            exc.component_id: float(exc.quantity.amount)
            for exc in self.project.exchanges
            if exc.exchange_type == "technosphere"
        }

    def create_evaluator(
        self,
        project_name: str,
        method_id: MethodId,
        functional_unit_amount: float = 1.0,
    ) -> LcaEvaluator:
        """Construye un evaluador LCA con matching de celdas."""
        self._ensure_project_name(project_name)
        bw_activity = self._resolve_activity(project_name)
        act_dict = self._get_act_dict(bw_activity, method_id, functional_unit_amount)
        output_idx = self._get_output_idx(act_dict, bw_activity, project_name)

        match_result = self._cell_matcher.match(
            project=self.project,
            bw_activity=bw_activity,
            act_dict=act_dict,
            output_idx=output_idx,
            technical_map=self.technical_map,
            location_map=self.location_map,
        )

        nominal_params = self.get_nominal_parameters(project_name)

        evaluator = MatrixLcaEvaluator(
            bc_module=self.bc,
            bw_activity=bw_activity,
            method_tuple=method_id,
            functional_unit=functional_unit_amount,
            matrix_cell_to_comps=match_result.matrix_cell_to_comps,
            nominal_amounts=nominal_params,
            sample_processor=self.sample_processor,
        )

        # Registrar diagnóstico
        nominal_score = evaluator.evaluate(nominal_params)
        self._latest_diagnostic = MappingDiagnostic(
            unmatched_components=match_result.unmatched_components,
            shared_cells_groups=match_result.shared_cells_groups,
            mapped_cells_count=match_result.mapped_cells_count,
            nominal_verified_score=nominal_score,
        )

        return evaluator

    def create_piv_evaluator(
        self,
        nominal_params: ParameterDict,
    ) -> LcaEvaluator | None:
        """Construye un evaluador PIV si hay h-vectors disponibles."""
        if not self.piv_h_vectors:
            return None

        return PivLcaEvaluator(
            h_vectors=self.piv_h_vectors,
            nominal_params=nominal_params,
            sample_processor=self.sample_processor,
        )

    def get_latest_mapping_diagnostic(self) -> MappingDiagnostic | None:
        """Retorna el diagnóstico de la última ejecución de matching."""
        return self._latest_diagnostic

    def extract_piv_h_vectors(
        self,
        lca_result: Any,
        project_name: str,
        method_name: str,
    ) -> dict[str, float]:
        """Delega la extracción de h-vectors al extractor especializado."""
        return self._piv_extractor.extract(lca_result, project_name, method_name)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _resolve_activity(self, project_name: str) -> Any:
        """Resuelve la actividad raíz en la BD de Brightway."""
        matches = [act for act in self.local_db if act["name"] == project_name]
        if len(matches) > 1:
            raise RuntimeError(
                f"Hay {len(matches)} actividades con el nombre "
                f"'{project_name}' en la BD; el nombre debe ser único."
            )
        if matches:
            return matches[0]
        raise RuntimeError(f"Actividad '{project_name}' no encontrada en la BD.")

    def _ensure_project_name(self, project_name: str) -> None:
        """Evita usar el estado de un proyecto con otro nombre solicitado."""
        if project_name != self.project.name:
            raise ValueError(
                f"El proveedor está configurado para el proyecto "
                f"'{self.project.name}', no para '{project_name}'."
            )

    def _get_act_dict(
        self, bw_activity: Any, method_id: MethodId, functional_unit: float
    ) -> dict[Any, int]:
        """Obtiene el diccionario de actividades del objeto LCA."""
        lca = self.bc.LCA({bw_activity: functional_unit}, method_id)
        lca.lci()
        lca.lcia()
        dicts = getattr(lca, "dicts", None)
        activity_dict = getattr(dicts, "activity", None)
        if activity_dict is not None:
            return cast(dict[Any, int], activity_dict)
        activity_dict = getattr(lca, "activity_dict", None)
        if activity_dict is not None:
            return cast(dict[Any, int], activity_dict)
        raise RuntimeError(
            "El objeto LCA no expone 'dicts.activity' ni 'activity_dict'."
        )

    def _get_output_idx(
        self, act_dict: dict[Any, int], bw_activity: Any, project_name: str
    ) -> int:
        """Obtiene el índice de salida de la actividad."""
        output_idx = act_dict.get(bw_activity.key)
        if output_idx is None:
            raise RuntimeError(
                f"Llave de actividad '{project_name}' no indexada en el modelo."
            )
        return int(output_idx)
