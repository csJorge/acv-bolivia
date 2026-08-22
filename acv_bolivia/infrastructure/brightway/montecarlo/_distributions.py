"""
infrastructure.brightway.montecarlo._distributions: Muestreo Estadístico Vectorizado.

Implementa las rutinas de generación pseudoaleatoria numéricas para las distribuciones
de incertidumbre del ciclo de vida (LCA). Delega el muestreo específico a estrategias
individuales (Strategy Pattern), permitiendo agregar nuevas distribuciones sin
modificar este código (OCP).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ....core.domain.uncertainty import UncertaintyParams
from ....infrastructure.brightway.montecarlo._distribution_strategies import (
    DeterministicStrategy,
    LognormalStrategy,
    NormalStrategy,
    SamplingStrategy,
    TriangularStrategy,
    UniformStrategy,
    WeibullStrategy,
)

logger = logging.getLogger(__name__)


# Registro de estrategias por type_id (OCP: agregar nuevas aquí sin modificar el código)
_STRATEGIES: dict[int, SamplingStrategy] = {
    0: DeterministicStrategy(),
    1: DeterministicStrategy(),
    2: LognormalStrategy(),
    3: NormalStrategy(),
    4: UniformStrategy(),
    5: TriangularStrategy(),
    8: WeibullStrategy(),
}


def _get_strategy(utype: int) -> SamplingStrategy:
    """Obtiene la estrategia de muestreo para el tipo de distribución."""
    return _STRATEGIES.get(utype, DeterministicStrategy())


def _extract_params(
    unc: UncertaintyParams | dict[str, Any] | None, nominal: float
) -> tuple[int, dict[str, Any]]:
    """Extrae el type_id y los parámetros estadísticos.

    Parameters
    ----------
    unc : UncertaintyParams | Dict[str, Any] | None
        Instancia del dominio, diccionario legacy o None.
    nominal : float
        Valor base del intercambio.

    Returns
    -------
    tuple[int, Dict[str, Any]]
        (type_id, params_dict).
    """
    if isinstance(unc, UncertaintyParams):
        params = unc.get_statistical_properties(nominal)
        utype = params.get("type_id", 0)
    elif isinstance(unc, dict):
        # Soporte legacy para diccionarios de Brightway2
        utype = int(
            unc.get(
                "type_id", unc.get("uncertainty type", unc.get("uncertainty_type", 0))
            )
            or 0
        )
        params = unc
    else:
        utype = 0
        params = {"amount": nominal}

    return utype, params


def _sample_distribution(
    unc: UncertaintyParams | dict[str, Any] | None,
    nominal: float,
    rng: np.random.Generator,
) -> float:
    """Genera una única muestra escalar desde la distribución especificada.

    Parameters
    ----------
    unc : UncertaintyParams | Dict[str, Any] | None
        Instancia del dominio, diccionario legacy o None.
    nominal : float
        Valor base del intercambio.
    rng : np.random.Generator
        Generador pseudoaleatorio aislado.

    Returns
    -------
    float
        Valor flotante escalar simulado.
    """
    utype, params = _extract_params(unc, nominal)
    strategy = _get_strategy(utype)

    try:
        return strategy.sample_scalar(params, nominal, rng)
    except Exception as e:
        logger.exception(
            "Error muestreando distribución type_id=%s: %s. Retornando nominal.",
            utype,
            type(e).__name__,
        )
        return float(nominal)


def sample_vectorized(
    unc: UncertaintyParams | dict[str, Any] | None,
    nominal: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Genera un arreglo masivo de N muestras vectorizadas.

    Delega el muestreo a la estrategia correspondiente, optimizando el rendimiento
    en un factor de 50-100x frente a muestras secuenciales escalares.

    Parameters
    ----------
    unc : UncertaintyParams | Dict[str, Any] | None
        Instancia del dominio, diccionario legacy o None.
    nominal : float
        Valor base del intercambio.
    rng : np.random.Generator
        Generador pseudoaleatorio aislado.
    n : int
        Número de muestras a generar.

    Returns
    -------
    np.ndarray
        Arreglo unidimensional de NumPy con dimensiones (n,).
    """
    utype, params = _extract_params(unc, nominal)
    strategy = _get_strategy(utype)

    try:
        return strategy.sample_vectorized(params, nominal, rng, n)
    except Exception as e:
        logger.exception(
            "Error muestreando distribución type_id=%s: %s. Retornando nominal.",
            utype,
            type(e).__name__,
        )
        return np.full(n, nominal, dtype=np.float64)
