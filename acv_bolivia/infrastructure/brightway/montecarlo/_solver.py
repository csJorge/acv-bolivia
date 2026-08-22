"""
infrastructure.brightway.montecarlo._solver: Selector Dinámico de Solucionador Disperso.

Implementa el enrutamiento y la selección automática del motor de álgebra lineal
esparcido de mayor rendimiento disponible en el entorno de ejecución.

Optimiza la resolución de sistemas de ecuaciones de inventario lineales (A^T · Y = B)
priorizando bibliotecas nativas de C/Fortran de subprocesos múltiples:
    1. Pypardiso (Intel MKL PARDISO): Solucionador directo multihilo de alto
       rendimiento.
       Aceleración de 3-5x en matrices de tecnosfera de gran escala.
    2. SciPy SuperLU: Fallback determinista monohilo (SciPy Sparse).
       Siempre disponible en la distribución básica estándar.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import scipy.sparse as sp
import scipy.sparse.linalg
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SparseSolverFunction = Callable[
    [sp.csc_matrix | sp.csr_matrix, NDArray[Any]], NDArray[Any]
]


def _build_solver() -> tuple[SparseSolverFunction, str]:
    """
    Inspecciona las dependencias del entorno de ejecución e instancia el
    backend correspondiente.

    Aísla las importaciones tardías condicionales evitando fallas de
    Importación en cascada.

    Returns
    -------
    Tuple[SparseSolverFunction, str]
        La función ejecutable del solucionador matricial y su nombre descriptivo.
    """
    try:
        import pypardiso

        def _solve_pardiso(
            A: sp.csc_matrix | sp.csr_matrix, b: NDArray[Any]
        ) -> NDArray[Any]:
            """Solucionador acelerado por hardware mediante hilos nativos Intel
            MKL PARDISO."""
            return cast(NDArray[Any], pypardiso.spsolve(A, b))

        logger.info(
            "Motor de resolución: Pypardiso (MKL Acelerado por Hardware) detectado "
            "y activo."
        )
        return _solve_pardiso, "Pypardiso (MKL Acelerado)"

    except ImportError:
        logger.warning(
            "Pypardiso no encontrado. Usando fallback: SciPy SuperLU (Monohilo). "
            "Para acelerar la simulación, instala pypardiso: pip install pypardiso"
        )

        def _solve_superlu(
            A: sp.csc_matrix | sp.csr_matrix, b: NDArray[Any]
        ) -> NDArray[Any]:
            """Solucionador de respaldo determinista monohilo basado en SciPy
            SuperLU."""
            return cast(NDArray[Any], scipy.sparse.linalg.spsolve(A, b))

        return _solve_superlu, "SciPy SuperLU (Fallback Monohilo)"


# Instanciación única al importar el módulo (Patrón Singleton implícito).
# Esto garantiza que la inspección del entorno solo se hace una vez por proceso.
_SOLVER, _SOLVER_NAME = _build_solver()


def get_solver() -> tuple[SparseSolverFunction, str]:
    """
    Retorna las instancias configuradas del solucionador activo para
    propósitos de auditoría.

    Garantiza que los corredores numéricos masivos del sistema puedan extraer
    la rutina matemática de forma agnóstica sin acoplarse a librerías propietarias.

    Returns
    -------
    Tuple[SparseSolverFunction, str]
        Función de cómputo matricial y la cadena del backend.
    """
    return _SOLVER, _SOLVER_NAME
