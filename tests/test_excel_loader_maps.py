"""Tests del reconeixement de les columnes `Codigo Ecoinvent` i `Unidad`.

Verifica que `ExcelInventoryLoader` extrae los mapeos opcionales de código y
unidad desde la hoja `Mapeo`, manteniendo la compatibilidad con hojas que no
los declaran (los mapas quedan vacíos).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import pandas as pd
import pytest

from acv_bolivia.infrastructure.input.excel_loader import ExcelInventoryLoader


def _write_inventario(
    path,
    *,
    with_code: bool,
    with_unit: bool,
) -> None:
    """Construye un Excel mínimo con las hojas requeridas."""
    mapping: dict[str, object] = {
        "Componente": ["fundacion", "torre"],
        "Proceso Ecoinvent": ["concrete, Portland", "steel, low-alloyed"],
        "Ubicacion": ["RoW", "GLO"],
    }
    if with_code:
        mapping["Codigo Ecoinvent"] = ["code_m3", "code_steel"]
    if with_unit:
        mapping["Unidad"] = ["cubic meter", "kilogram"]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(mapping).to_excel(writer, sheet_name="Mapeo", index=False)
        pd.DataFrame(
            {
                "ID_Proyecto": ["p1"],
                "Nombre_Parque": ["El Dorado"],
                "Generacion_kWh": [1000.0],
                "fundacion": [650.0],
                "torre": [1200.0],
            }
        ).to_excel(writer, sheet_name="Proyectos", index=False)


@pytest.mark.parametrize(
    "with_code,with_unit",
    [(True, True), (True, False), (False, True)],
)
def test_carga_mapa_code_y_unidad(tmp_path, with_code, with_unit):
    path = tmp_path / "inv.xlsx"
    _write_inventario(path, with_code=with_code, with_unit=with_unit)

    loader = ExcelInventoryLoader(path)
    result = loader.load(force_reload=True)

    if with_code:
        assert result.code_map == {
            "fundacion": "code_m3",
            "torre": "code_steel",
        }
    else:
        assert result.code_map == {}

    if with_unit:
        assert result.unit_map == {
            "fundacion": "cubic meter",
            "torre": "kilogram",
        }
    else:
        assert result.unit_map == {}


def test_sin_columnas_opcionales_los_mapas_quedan_vacios(tmp_path):
    path = tmp_path / "inv.xlsx"
    _write_inventario(path, with_code=False, with_unit=False)

    loader = ExcelInventoryLoader(path)
    result = loader.load(force_reload=True)

    assert result.code_map == {}
    assert result.unit_map == {}
