"""
infrastructure.input.parsers: Parsers específicos del formato Excel.

Traducen filas crudas del Excel a objetos del dominio.
Este módulo conoce la estructura del Excel (nombres de columnas),
por lo que pertenece a infraestructura, no al dominio.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core.domain.uncertainty import DistributionType, UncertaintyParams

# Mapeo de nombres de distribución en el Excel a tipos del dominio.
# Incluye alias comunes para tolerancia de entrada.
_DISTRIBUTION_MAP: dict[str, DistributionType] = {
    "normal": DistributionType.NORMAL,
    "lognormal": DistributionType.LOGNORMAL,
    "triangular": DistributionType.TRIANGULAR,
    "uniforme": DistributionType.UNIFORM,
    "uniform": DistributionType.UNIFORM,
    "weibull": DistributionType.WEIBULL,
}


def parse_uncertainty_from_excel_row(
    row: Mapping[str, Any],
) -> UncertaintyParams | None:
    """Convierte una fila cruda del Excel (hoja MC) a UncertaintyParams.

    Conoce las columnas del Excel: ``'distribucion'``, ``'parametro_1'``,
    ``'parametro_2'``.

    Parameters
    ----------
    row : Mapping[str, Any]
        Diccionario de una fila de la hoja MC.

    Returns
    -------
    UncertaintyParams | None
        Instancia de UncertaintyParams, o None si la distribución es
        determinística, no reconocida, o si las celdas numéricas están
        corruptas.
    """
    dist_str = str(row.get("distribucion", "")).lower().strip()
    target_dist = _DISTRIBUTION_MAP.get(dist_str)

    if not target_dist or not target_dist.is_stochastic:
        return None

    try:
        p1 = float(row["parametro_1"]) if row.get("parametro_1") is not None else None
        p2 = float(row["parametro_2"]) if row.get("parametro_2") is not None else None
        return UncertaintyParams(distribution=target_dist, p1=p1, p2=p2)
    except (ValueError, TypeError, KeyError):
        return None
