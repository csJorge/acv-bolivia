"""
infrastructure.input.validators: Validaciones estructurales del formato Excel.

Validan la coherencia entre las columnas del Excel y los mapeos técnicos.
Conocen la estructura del Excel (columnas reservadas), por lo que
pertenecen a infraestructura, no al dominio.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from ...core.domain.validators import ValidationReport

# Columnas reservadas del Excel (metadatos, no son componentes)
RESERVED_EXCEL_COLUMNS: frozenset[str] = frozenset(
    {
        "id_proyecto",
        "nombre_parque",
        "generacion_kwh",
    }
)


def validate_inventory_mapping(
    inventory_columns: set[str],
    mapping_keys: set[str],
    reserved: frozenset[str] | None = None,
) -> ValidationReport:
    """Valida la simetría entre columnas del Excel y claves del mapeo técnico.

    Normaliza ambos conjuntos a minúsculas para neutralizar falsos negativos
    por sensibilidad a mayúsculas o espacios en blanco residuales.

    Parameters
    ----------
    inventory_columns : Iterable[str]
        Columnas del DataFrame de proyectos.
    mapping_keys : Iterable[str]
        Claves del diccionario de mapeo técnico.
    reserved : frozenset[str] | None
        Columnas a ignorar (default: RESERVED_EXCEL_COLUMNS).

    Returns
    -------
    ValidationReport
        Advertencia si sobran columnas sin mapeo; error bloqueante si faltan
        columnas para claves del mapeo técnico.
    """
    if reserved is None:
        reserved = RESERVED_EXCEL_COLUMNS

    errors: list[str] = []
    warnings: list[str] = []

    clean_cols = {str(c).strip().lower() for c in inventory_columns} - reserved
    clean_keys = {str(k).strip().lower() for k in mapping_keys}

    unmapped_cols = clean_cols - clean_keys
    if unmapped_cols:
        warnings.append(
            f"Columnas en Excel sin correspondencia técnica (serán omitidas): "
            f"{unmapped_cols}"
        )

    surplus_keys = clean_keys - clean_cols
    if surplus_keys:
        errors.append(
            f"Claves de correspondencia ausentes en las columnas del Excel: "
            f"{surplus_keys}"
        )

    return ValidationReport.create(errors=errors, warnings=warnings)
