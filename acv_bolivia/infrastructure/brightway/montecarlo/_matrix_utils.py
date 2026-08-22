"""
infrastructure.brightway.montecarlo._matrix_utils - Utilidades de matrices dispersas.

Provee las funciones necesarias para localizar e inyectar valores en la matriz
tecnológica A de Brightway2 de forma eficiente durante el loop de Montecarlo.

Implementa un parchado in-place O(k) directo sobre el array de datos de Scipy CSR
(k = componentes con incertidumbre, típicamente 5-15), evitando el costo
O(nnz ≈ 400k) de reconstruir la matriz completa en cada iteración estocástica.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp

from ....core.domain.contracts import MethodId

logger = logging.getLogger(__name__)


def build_c_stack(lca: Any, methods: list[MethodId], bd_module: Any) -> sp.csr_matrix:
    """Construye la matriz C apilada (n_methods × n_bio) para todos los métodos.

    Parameters
    ----------
    lca : Any
        Instancia LCA ya factorizada.
    methods : list[MethodId]
        lista de métodos de impacto.
    bd_module : Any
        Módulo bw2data.

    Returns
    -------
    sp.csr_matrix
        Matriz C apilada en formato CSR.
    """
    n_bio = lca.biosphere_matrix.shape[0]
    bio_dict = (
        lca.dicts.biosphere
        if hasattr(lca, "dicts") and hasattr(lca.dicts, "biosphere")
        else getattr(lca, "biosphere_dict", {})
    )

    data, rows, cols = [], [], []
    for m_idx, method in enumerate(methods):
        for flow_key, cf in bd_module.Method(method).load():
            idx = bio_dict.get(flow_key)
            if idx is not None:
                data.append(float(cf))
                rows.append(m_idx)
                cols.append(int(idx))

    return sp.csr_matrix((data, (rows, cols)), shape=(len(methods), n_bio))


def build_data_positions(
    lca: Any, relevant_rows: list[tuple[int, str]]
) -> list[tuple[int, int, float]]:
    """Pre-calcula posiciones en CSR .data para las celdas que varían en MC.

    Debe llamarse UNA SOLA VEZ después de lca.lci(), antes del loop.

    Parameters
    ----------
    lca : Any
        Instancia LCA ya factorizada.
    relevant_rows : list[tuple[int, str]]
        lista de (arr_i, component_name).

    Returns
    -------
    list[tuple[int, int, float]]
        lista de (data_pos, flip_sign, baseline) para cada celda.
    """
    tp = lca.tech_params
    mat = lca.technosphere_matrix

    positions: list[tuple[int, int, float]] = []
    for arr_i, _comp in relevant_rows:
        row = int(tp[arr_i]["row"])
        col = int(tp[arr_i]["col"])

        # Encontrar posición física en CSR .data
        row_start = int(mat.indptr[row])
        row_end = int(mat.indptr[row + 1])
        col_slice = mat.indices[row_start:row_end]
        hit = np.where(col_slice == col)[0]

        if len(hit) == 0:
            logger.warning(
                "Celda (row=%d, col=%d) no encontrada en CSR. Posición -1.", row, col
            )
            positions.append((-1, 1, float(tp[arr_i]["amount"])))
            continue

        data_pos = row_start + int(hit[0])
        flip_sign = -1.0 if ("flip" in tp.dtype.names and tp[arr_i]["flip"]) else 1.0
        baseline = float(tp[arr_i]["amount"]) * flip_sign
        positions.append((data_pos, int(flip_sign), baseline))

    return positions


def patch_matrix(
    lca: Any,
    relevant_rows: list[tuple[int, str]],
    data_positions: list[tuple[int, int, float]],
    new_values: dict[str, float],
    dom_nominals: dict[str, float],
) -> None:
    """Actualiza in-place las celdas de technosphere_matrix que cambian.

    No reconstruye la matriz: modifica directamente la vista en memoria
    .data[data_pos]. Aplica escalado proporcional conservando el signo
    e integridad de la celda CSR original.

    Parameters
    ----------
    lca : Any
        Instancia LCA con technosphere_matrix.
    relevant_rows : list[tuple[int, str]]
        lista de (arr_i, component_name).
    data_positions : list[tuple[int, int, float]]
        lista de (data_pos, flip_sign, baseline).
    new_values : dict[str, float]
        {component_name: valor_muestreado}.
    dom_nominals : dict[str, float]
        {component_name: valor_nominal}.
    """
    mat = lca.technosphere_matrix.data  # Vista directa al buffer de C

    # 1. Calcular la contribución individual de cada componente, SIN escribir
    #    todavía - así ninguna escritura puede pisar a otra antes de sumarlas.
    contribuciones: dict[int, float] = {}
    for (arr_i, comp), (data_pos, flip_sign, baseline) in zip(
        relevant_rows, data_positions
    ):
        if data_pos < 0:
            continue  # Celda no indexada

        sample = new_values.get(comp)
        if sample is None:
            valor = baseline  # Reset al baseline
        else:
            dom_nom = dom_nominals.get(comp, 0.0)
            if dom_nom != 0.0:
                valor = baseline * (float(sample) / dom_nom)
            else:
                valor = float(sample) * flip_sign

        # 2. Acumular por data_pos: si dos componentes comparten celda, sus
        #    contribuciones se SUMAN acá en vez de que la última pise a la
        #    primera. Para una celda con un solo componente, esto es
        #    exactamente equivalente a asignar ese único valor.
        contribuciones[data_pos] = contribuciones.get(data_pos, 0.0) + valor

    # 3. Escribir cada celda una sola vez, con el total ya acumulado.
    for data_pos, total in contribuciones.items():
        mat[data_pos] = total
