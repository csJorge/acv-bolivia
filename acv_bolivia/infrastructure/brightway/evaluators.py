"""
infrastructure/brightway/evaluators: Evaluadores concretos que implementan
el Protocol LcaEvaluator del dominio.

Cada evaluador encapsula una estrategia de evaluación LCA:
- MatrixLcaEvaluator: evaluación completa con re-factorización matricial
- PivLcaEvaluator: evaluación lineal aproximada con h-vectors
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import SparseEfficiencyWarning

from ...core.domain.contracts import ParameterDict
from ...infrastructure.brightway.montecarlo._sample_processor import SampleProcessor

logger = logging.getLogger(__name__)

SampleProcessorCallable: TypeAlias = Callable[
    [dict[str, NDArray[Any]]], dict[str, NDArray[Any]]
]


def _apply_sample_processor(
    processor: SampleProcessorCallable,
    params: ParameterDict,
) -> ParameterDict:
    wrapped = {
        key: np.array([float(value)], dtype=np.float64) for key, value in params.items()
    }
    processed = processor(wrapped)
    result: ParameterDict = {}
    for key, values in processed.items():
        if values.size == 0:
            raise ValueError(
                f"El sample_processor devolvió un array vacío para '{key}'."
            )
        value = float(values[0])
        if not math.isfinite(value):
            raise ValueError(
                f"El sample_processor devolvió un valor no finito para '{key}'."
            )
        result[key] = value
    return result


class MatrixLcaEvaluator:
    """Evaluador LCA que modifica la matriz de tecnosfera y re-factoriza.

    Implementa el Protocol LcaEvaluator del dominio.
    """

    def __init__(
        self,
        bc_module: Any,
        bw_activity: Any,
        method_tuple: tuple[str, ...],
        functional_unit: float,
        matrix_cell_to_comps: dict[tuple[int, int], list[str]],
        nominal_amounts: dict[str, float],
        sample_processor: SampleProcessor | SampleProcessorCallable | None = None,
    ) -> None:
        self.bc = bc_module
        self.bw_activity = bw_activity
        self.method_tuple = method_tuple
        self.functional_unit = functional_unit
        self.matrix_cell_to_comps = matrix_cell_to_comps
        self.nominal_amounts = nominal_amounts
        self.sample_processor = sample_processor

        # Inicializar LCA y capturar matriz original
        self.lca = bc_module.LCA({bw_activity: functional_unit}, method_tuple)
        self.lca.lci()
        self.lca.lcia()
        self.original_tech_matrix = self.lca.technosphere_matrix.copy()

    def evaluate(self, parameters: ParameterDict) -> float:
        """Evalúa el impacto para una combinación de parámetros.

        Parameters
        ----------
        parameters : dict[str, float]
            Valores de los componentes del inventario.

        Returns
        -------
        float
            Score LCIA calculado por Brightway2.

        Raises
        ------
        RuntimeError
            Si el evaluador ya fue limpiado.
        ValueError
            Si un parámetro no es finito.
        """
        if self.original_tech_matrix is None:
            raise RuntimeError("El evaluador matricial ya fue limpiado.")
        _validate_parameters(parameters)
        if self.sample_processor is not None:
            parameters = self._apply_processor(parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
            new_matrix = self.original_tech_matrix.copy()

            for (r_idx, c_idx), comps in self.matrix_cell_to_comps.items():
                orig_val = self.original_tech_matrix[r_idx, c_idx]
                nom_sum = sum(self.nominal_amounts.get(c, 0.0) for c in comps)
                if nom_sum == 0:
                    continue
                current_sum = sum(
                    float(parameters.get(c, self.nominal_amounts.get(c, 0.0)))
                    for c in comps
                )
                new_matrix[r_idx, c_idx] = orig_val * (current_sum / nom_sum)

            self.lca.technosphere_matrix = new_matrix
            self.lca.redo_lci({self.bw_activity: self.functional_unit})
            self.lca.redo_lcia()
            return float(self.lca.score)

    def _apply_processor(self, params: ParameterDict) -> ParameterDict:
        """Aplica el procesador y convierte su salida a escalares finitos."""
        processor = self.sample_processor
        if processor is None:
            return params
        return _apply_sample_processor(processor, params)

    def cleanup(self) -> None:
        """Libera la matriz original de la memoria.

        Notes
        -----
        Después de este método, ``evaluate()`` deja de estar disponible.
        """
        self.original_tech_matrix = None


class PivLcaEvaluator:
    """Evaluador LCA lineal aproximado basado en h-vectors (PIV).

    Implementa el Protocol LcaEvaluator del dominio.
    """

    def __init__(
        self,
        h_vectors: dict[str, float],
        nominal_params: ParameterDict,
        sample_processor: SampleProcessor | SampleProcessorCallable | None = None,
    ) -> None:
        self.h_vectors = {c: float(v) for c, v in h_vectors.items()}
        self.nominal_params = nominal_params
        self.sample_processor = sample_processor

    def evaluate(self, parameters: ParameterDict) -> float:
        """Evalúa el impacto mediante el producto punto con h-vectors.

        Parameters
        ----------
        parameters : dict[str, float]
            Valores de los componentes del inventario.

        Returns
        -------
        float
            Impacto aproximado calculado por PIV.

        Raises
        ------
        KeyError
            Si falta un componente requerido por ``h_vectors``.
        ValueError
            Si un parámetro no es finito.
        """
        _validate_parameters(parameters)
        if self.sample_processor is not None:
            parameters = self._apply_processor(parameters)

        total = 0.0
        for component, h_value in self.h_vectors.items():
            if component in parameters:
                value = parameters[component]
            elif component in self.nominal_params:
                value = self.nominal_params[component]
            else:
                raise KeyError(f"Falta el parámetro requerido '{component}'.")
            total += float(value) * h_value
        return total

    def _apply_processor(self, params: ParameterDict) -> ParameterDict:
        """Aplica el procesador y convierte su salida a escalares finitos."""
        processor = self.sample_processor
        if processor is None:
            return params
        return _apply_sample_processor(processor, params)


def _validate_parameters(parameters: ParameterDict) -> None:
    for key, value in parameters.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"El parámetro '{key}' debe ser finito.")
