"""
infrastructure.input.auditors: Auditorías de calidad de datos del inventario Excel.

Esta capa conoce la estructura de las hojas y columnas del Excel.
Implementa el protocolo InventoryDataAuditor de application/contracts.py
mediante la clase InventoryDataAuditorAdapter.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.domain.uncertainty import DistributionType
from ...core.domain.validators import ValidationReport

logger = logging.getLogger(__name__)

# Columnas reservadas del Excel que no son componentes
_RESERVED_COLUMNS = {
    "proyecto",
    "project",
    "nombre",
    "name",
    "generacion",
    "generation",
    "kwh",
    "componente",
    "component",
    "proceso",
    "process",
    "monto",
    "amount",
    "cantidad",
    "quantity",
    "unidad",
    "unit",
    "ubicacion",
    "location",
    "tipo",
    "type",
    "distribucion",
    "distribution",
    "media",
    "mean",
    "desviacion",
    "std",
    "scale",
    "minimo",
    "min",
    "maximo",
    "max",
    "moda",
    "mode",
    "forma",
    "shape",
}

# Nombres válidos de distribuciones estocásticas
_VALID_DISTRIBUTION_NAMES = {d.value for d in DistributionType if d.is_stochastic}


# ==============================================================================
# Funciones de auditoría específicas del formato Excel
# ==============================================================================


def audit_component_amounts_excel(
    project_rows: list[dict[str, Any]],
) -> ValidationReport:
    """Audita filas crudas del Excel buscando montos no positivos o faltantes.

    Parameters
    ----------
    project_rows : List[Dict[str, Any]]
        Filas crudas de la hoja de proyectos del Excel.

    Returns
    -------
    ValidationReport
        Reporte con errores y advertencias encontrados.
    """
    errors: list[str] = []

    for row_idx, row in enumerate(
        project_rows, start=2
    ):  # start=2 porque fila 1 es header
        project_name = row.get("proyecto", row.get("project", f"Fila_{row_idx}"))
        component = row.get("componente", row.get("component", ""))
        amount = row.get("monto", row.get("amount", None))

        # Verificar que el monto existe
        if amount is None or str(amount).strip() == "":
            errors.append(
                f"[{project_name}] Componente '{component}': monto vacío o "
                f"faltante (fila {row_idx})."
            )
            continue

        # Verificar que el monto es numérico
        try:
            amount_val = float(str(amount).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            errors.append(
                f"[{project_name}] Componente '{component}': monto no numérico "
                f"'{amount}' (fila {row_idx})."
            )
            continue

        # Verificar que el monto es positivo
        if amount_val <= 0:
            errors.append(
                f"[{project_name}] Componente '{component}': monto no positivo "
                f"({amount_val}) (fila {row_idx})."
            )

    return ValidationReport.create(errors=errors)


def audit_distribution_names_excel(
    mc_rows: list[dict[str, Any]],
) -> ValidationReport:
    """Audita filas crudas del Excel buscando nombres de distribución inválidos.

    Parameters
    ----------
    mc_rows : List[Dict[str, Any]]
        Filas crudas de la hoja MC del Excel.

    Returns
    -------
    ValidationReport
        Reporte con errores y advertencias encontrados.
    """
    warnings: list[str] = []

    for row_idx, row in enumerate(mc_rows, start=2):
        component = row.get("componente", row.get("component", f"Fila_{row_idx}"))
        dist_name = (
            str(row.get("distribucion", row.get("distribution", ""))).strip().lower()
        )

        # Distribución vacía = determinística (no es error, es válido)
        if not dist_name or dist_name in (
            "",
            "deterministic",
            "deterministica",
            "fijo",
            "fixed",
        ):
            continue

        # Verificar que el nombre es una distribución estocástica válida
        if dist_name not in _VALID_DISTRIBUTION_NAMES:
            warnings.append(
                f"Componente '{component}': distribución '{dist_name}' no reconocida. "
                f"Válidas: {sorted(_VALID_DISTRIBUTION_NAMES)}. "
                "Se tratará como determinística."
            )

    return ValidationReport.create(warnings=warnings)


def audit_technical_mapping_excel(
    project_rows: list[dict[str, Any]],
) -> ValidationReport:
    """Audita que cada componente tenga un proceso Ecoinvent mapeado.

    Parameters
    ----------
    project_rows : List[Dict[str, Any]]
        Filas crudas de la hoja de proyectos del Excel.

    Returns
    -------
    ValidationReport
        Reporte con errores y advertencias encontrados.
    """
    errors: list[str] = []

    for row_idx, row in enumerate(project_rows, start=2):
        project_name = row.get("proyecto", row.get("project", f"Fila_{row_idx}"))
        component = row.get("componente", row.get("component", ""))
        process = row.get("proceso", row.get("process", ""))

        if not component or str(component).strip() == "":
            continue

        if not process or str(process).strip() == "":
            errors.append(
                f"[{project_name}] Componente '{component}': sin proceso Ecoinvent "
                f"mapeado (fila {row_idx})."
            )

    return ValidationReport.create(errors=errors)


def audit_generation_values_excel(
    project_rows: list[dict[str, Any]],
) -> ValidationReport:
    """Audita que los valores de generación eléctrica sean positivos.

    Parameters
    ----------
    project_rows : List[Dict[str, Any]]
        Filas crudas de la hoja de proyectos del Excel.

    Returns
    -------
    ValidationReport
        Reporte con errores y advertencias encontrados.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_projects: dict[str, float] = {}

    for row_idx, row in enumerate(project_rows, start=2):
        project_name = str(row.get("proyecto", row.get("project", ""))).strip()
        generation = row.get("generacion", row.get("generation", row.get("kwh", None)))

        if not project_name:
            continue

        if generation is None or str(generation).strip() == "":
            if project_name not in seen_projects:
                warnings.append(
                    f"Proyecto '{project_name}': generación no especificada. "
                    f"Se usará 1.0 como fallback (sin normalización real)."
                )
                seen_projects[project_name] = 1.0
            continue

        try:
            gen_val = float(str(generation).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            errors.append(
                f"Proyecto '{project_name}': generación no numérica "
                f"'{generation}' (fila {row_idx})."
            )
            continue

        if gen_val <= 0:
            errors.append(
                f"Proyecto '{project_name}': generación no positiva "
                f"({gen_val}) (fila {row_idx})."
            )
        elif project_name not in seen_projects:
            seen_projects[project_name] = gen_val

    return ValidationReport.create(errors=errors, warnings=warnings)


# ==============================================================================
# Adaptador: implementa el protocolo InventoryDataAuditor
# ==============================================================================


class InventoryDataAuditorAdapter:
    """Adaptador que implementa el protocolo InventoryDataAuditor.

    Orquesta las funciones de auditoría específicas del formato Excel
    y consolida sus reportes en un único ValidationReport.

    Satisface el protocolo:
        def audit(self, project_rows, mc_rows, mc_mode) -> ValidationReport
    """

    def audit(
        self,
        project_rows: list[dict[str, Any]],
        mc_rows: list[dict[str, Any]],
        mc_mode: str,
    ) -> ValidationReport:
        """Ejecuta todas las auditorías y consolida el reporte.

        Parameters
        ----------
        project_rows : List[Dict[str, Any]]
            Filas crudas de la hoja Proyectos.
        mc_rows : List[Dict[str, Any]]
            Filas crudas de la hoja MC.
        mc_mode : str
            Modo de simulación ('full', 'foreground', 'piv').

        Returns
        -------
        ValidationReport
            Reporte consolidado con todos los errores y advertencias.
        """
        consolidated = ValidationReport(
            is_valid=True
        )  # Inicializa asumiendo que todo esta bien.

        # 1. Auditar montos de componentes
        amounts_report = audit_component_amounts_excel(project_rows)
        consolidated.merge(amounts_report)

        # 2. Auditar mapeo técnico (proceso Ecoinvent)
        mapping_report = audit_technical_mapping_excel(project_rows)
        consolidated.merge(mapping_report)

        # 3. Auditar valores de generación
        generation_report = audit_generation_values_excel(project_rows)
        consolidated.merge(generation_report)

        # 4. Auditar distribuciones (solo si hay filas MC)
        if mc_rows:
            dist_report = audit_distribution_names_excel(mc_rows)
            consolidated.merge(dist_report)

        # Log resumen
        if consolidated.errors:
            logger.error(
                "Auditoría de inventario: %d errores, %d advertencias.",
                len(consolidated.errors),
                len(consolidated.warnings),
            )
        elif consolidated.warnings:
            logger.warning(
                "Auditoría de inventario: 0 errores, %d advertencias.",
                len(consolidated.warnings),
            )
        else:
            logger.info("Auditoría de inventario: sin errores ni advertencias.")

        return consolidated
