"""
application.dto.build_inventory - DTO del caso de uso BuildInventoryUseCase.

Representa el resultado de la construcción del inventario ACV desde Excel
hacia Brightway2, incluyendo proyectos, mapeos, configuración de Monte Carlo
y diagnóstico de calidad.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.domain.models import Project
from ...core.domain.validators import ValidationReport


@dataclass
class BuildInventoryResult:
    """Resultado del caso de uso BuildInventoryUseCase.

    Contiene los proyectos cargados del Excel, los mapeos técnicos y de
    ubicación, la configuración de Monte Carlo (reglas DEP/MIX) y el
    diagnóstico de calidad de datos.

    Attributes
    ----------
    projects : List[Project]
        Proyectos del dominio con sus exchanges.
    technical_map : Dict[str, str]
        Mapeo {componente: proceso_ei}.
    location_map : Dict[str, str]
        Mapeo {componente: ubicación_ei}.
    generation_dict : Dict[str, float]
        Mapeo {project_name: kwh_generados}.
    local_db_name : str
        Nombre de la base de datos local en Brightway2.
    mc_config : Dict[str, Dict[str, Any]]
        Configuración de reglas de Monte Carlo por proyecto (DEP/MIX).
    data_quality : ValidationReport
        Diagnóstico de calidad de datos de entrada.
    success : bool
        True si la construcción finalizó sin errores.
    error_message : Optional[str]
        Mensaje de error si success=False.
    """

    projects: list[Project] = field(default_factory=list)
    technical_map: dict[str, str] = field(default_factory=dict)
    location_map: dict[str, str] = field(default_factory=dict)
    code_map: dict[str, str] = field(default_factory=dict)
    unit_map: dict[str, str] = field(default_factory=dict)
    generation_dict: dict[str, float] = field(default_factory=dict)
    local_db_name: str = ""
    mc_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_quality: ValidationReport | None = None
    success: bool = True
    error_message: str | None = None

    @property
    def project_names(self) -> list[str]:
        """Retorna los nombres de todos los proyectos cargados."""
        return [p.name for p in self.projects]

    @property
    def n_projects(self) -> int:
        """Retorna el número de proyectos cargados."""
        return len(self.projects)

    @property
    def n_components(self) -> int:
        """Retorna el número de componentes únicos en el mapeo técnico."""
        return len(self.technical_map)
