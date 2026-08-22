"""
infrastructure.input.helpers: Funciones auxiliares para el formato Excel.

Conocen la estructura del Excel (nombres de columnas) y proporcionan
utilidades para transformar datos crudos del Excel a estructuras útiles.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def build_generation_dict(
    project_records: Sequence[Mapping[str, Any]],
    name_col: str = "Nombre_Parque",
    gen_col: str = "Generacion_kWh",
) -> dict[str, float]:
    """Construye el diccionario de generación desde una lista de registros del Excel.

    Parameters
    ----------
    project_records : Sequence[Mapping[str, Any]]
        Lista de dicts, uno por fila del Excel.
    name_col : str, default 'Nombre_Parque'
        Columna con el nombre del proyecto.
    gen_col : str, default 'Generacion_kWh'
        Columna con la generación en kWh.

    Returns
    -------
    dict[str, float]
        ``{project_name: kwh_generados}``. Una celda vacía, no numérica o
        NaN se registra como 0.0.
    """
    result: dict[str, float] = {}
    for record in project_records:
        name = str(record.get(name_col, "")).strip()
        if not name:
            continue

        gen = record.get(gen_col, 0)
        try:
            gen_val = float(gen)
        except (ValueError, TypeError):
            gen_val = 0.0

        result[name] = 0.0 if math.isnan(gen_val) else gen_val

    return result
