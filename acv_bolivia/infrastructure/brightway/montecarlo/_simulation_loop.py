"""
infrastructure/brightway/montecarlo/_simulation_loop:
Bucle de simulación numérica estocástica.

Responsabilidad única: ejecutar el ciclo iterativo de perturbación
y resolución, reutilizando memoria mediante descomposiciones LAPACK.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ....infrastructure.brightway.montecarlo._solver import _SOLVER, _SOLVER_NAME

logger = logging.getLogger(__name__)


class SimulationLoop:
    """Ejecuta el bucle de simulación Monte Carlo con reutilización de memoria."""

    def __init__(
        self,
        mc_object: Any,
        act_keys: list[tuple],
        c_matrix: sp.csr_matrix,
        methods: list[tuple],
    ) -> None:
        self.mc = mc_object
        self.act_keys = act_keys
        self.c_matrix = c_matrix
        self.methods = methods

        self.n_proc = mc_object.technosphere_matrix.shape[0]
        self.n_methods = len(methods)
        self.n_act = len(act_keys)

    def run(
        self, iterations: int, functional_unit: float
    ) -> tuple[NDArray[Any], float]:
        """Ejecuta el bucle y retorna (scores, elapsed_seconds).

        Returns
        -------
        tuple
            (final_scores: NDArray[Any], elapsed: float)
            final_scores tiene shape (n_methods, n_act, iterations).
        """
        rhs_buf = np.empty((self.n_proc, self.n_methods), dtype=np.float64)
        final_scores = np.zeros(
            (self.n_methods, self.n_act, iterations), dtype=np.float64
        )

        act_indices: list[int | None] = []
        prev_nnz = -1

        logger.info(
            "Iniciando ciclo secuencial: %d iteraciones × %d procesos × %d "
            "dimensiones.",
            iterations,
            self.n_act,
            self.n_methods,
        )
        start_time = time.time()

        for it in range(iterations):
            next(self.mc)  # Perturbación de fondo y frente

            if it == 0:
                d = getattr(self.mc, "dicts", None)
                mc_act_dict = (
                    d.activity
                    if d and hasattr(d, "activity")
                    else self.mc.activity_dict
                )
                act_indices = [mc_act_dict.get(k) for k in self.act_keys]

            # Reutilización de memoria: si la estructura no varía, sobreescribimos
            cur_nnz = self.mc.technosphere_matrix.nnz
            if cur_nnz != prev_nnz:
                A_T_csc = self.mc.technosphere_matrix.T.tocsc()
                prev_nnz = cur_nnz
            else:
                A_T_csc.data[:] = self.mc.technosphere_matrix.T.tocsc().data

            # Multiplicación y proyección ortogonal
            np.copyto(
                rhs_buf,
                (self.c_matrix @ self.mc.biosphere_matrix).toarray().T,
            )

            try:
                Y = _SOLVER(A_T_csc, rhs_buf)
                for a_idx, mat_idx in enumerate(act_indices):
                    if mat_idx is not None:
                        final_scores[:, a_idx, it] = Y[mat_idx, :] * functional_unit
            except Exception as e:
                logger.exception(
                    "Iteración %d falló: %s. Asignando NaN.", it, type(e).__name__
                )
                final_scores[:, :, it] = np.nan

        elapsed = time.time() - start_time
        logger.info(
            "Simulación concluida en %.1fs (%.0f iteraciones/s).",
            elapsed,
            iterations / elapsed,
        )

        return final_scores, elapsed

    @property
    def solver_name(self) -> str:
        return _SOLVER_NAME
