"""
core.domain.validators: Validadores de reglas de negocio del dominio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...core.domain.models import Project

# ==============================================================================
# 1. Resultado de validación (Value Object inmutable)
# ==============================================================================


@dataclass(frozen=True)
class ValidationReport:
    """Value Object inmutable con el resultado de una validación de dominio."""

    is_valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls, errors: list[str] | None = None, warnings: list[str] | None = None
    ) -> ValidationReport:
        """Factory method para construir el reporte de forma inmutable."""
        return cls(
            is_valid=len(errors or []) == 0,
            errors=tuple(errors or []),
            warnings=tuple(warnings or []),
        )

    def __str__(self) -> str:
        lines: list[str] = []
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [AVISO]   {w}")
        status = "OK" if self.is_valid else "FALLIDA"
        return f"Validación {status}:\n" + (
            "\n".join(lines) if lines else "  Sin observaciones."
        )

    def merge(self, other: ValidationReport) -> ValidationReport:
        """Fusiona otro reporte en este (acumula errores y advertencias)."""
        new_errors = list(self.errors) + list(other.errors)
        new_warnings = list(self.warnings) + list(other.warnings)
        return ValidationReport.create(errors=new_errors, warnings=new_warnings)


# ==============================================================================
# 2. Validaciones de dominio puro
# ==============================================================================


def validate_project_consistency(project: Project) -> ValidationReport:
    """Valida las reglas de negocio de un proyecto del dominio."""
    errors: list[str] = []
    warnings: list[str] = []

    if not math.isfinite(project.generation_kwh) or project.generation_kwh <= 0.0:
        errors.append(f"Proyecto '{project.name}': la generación_kwh debe ser > 0.")

    tech_exchanges = project.get_technosphere_exchanges()
    if not tech_exchanges:
        warnings.append(
            f"Proyecto '{project.name}': no tiene intercambios de tecnosfera."
        )

    return ValidationReport.create(errors=errors, warnings=warnings)


def validate_sensitivity_components(
    components: list[str], valid_components: set[str]
) -> ValidationReport:
    """Valida que los componentes de sensibilidad existan en el dominio."""
    errors: list[str] = []
    for comp in components:
        if comp not in valid_components:
            errors.append(
                f"Componente '{comp}' no existe en el inventario del dominio."
            )
    return ValidationReport.create(errors=errors)


def validate_method_id(method_id: object) -> ValidationReport:
    """Valida que un MethodId sea una tupla no vacía de textos no vacíos."""
    errors: list[str] = []
    if (
        not isinstance(method_id, tuple)
        or not method_id
        or not all(isinstance(part, str) and part.strip() for part in method_id)
    ):
        errors.append(
            "MethodId inválido: debe ser una tupla no vacía de textos no vacíos. "
            f"Recibido: {method_id}"
        )
    return ValidationReport.create(errors=errors)
