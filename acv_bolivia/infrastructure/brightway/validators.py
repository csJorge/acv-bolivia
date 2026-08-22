"""infrastructure/brightway/validators: Validaciones específicas de Brightway2."""

from __future__ import annotations

from ...core.domain.validators import ValidationReport


def validate_ecoinvent_db(
    available_databases: set[str],
    expected_db_name: str,
) -> ValidationReport:
    """Valida la presencia física de la base de datos de fondo de Ecoinvent
    en Brightway2."""
    errors = []
    warnings = []

    if expected_db_name not in available_databases:
        errors.append(
            f"Base de datos de fondo '{expected_db_name}' no localizada en Brightway2. "
            f"Bases registradas en el entorno: {available_databases}"
        )
    else:
        warnings.append(
            f"Base de datos de fondo '{expected_db_name}' localizada correctamente."
        )

    return ValidationReport.create(errors=errors, warnings=warnings)
