"""
application.dto.run_lca - DTO del caso de uso RunLCAUseCase.

Representa el resultado del cálculo LCIA determinístico completo,
incluyendo scores, hotspots y normalización por generación.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...core.domain.contracts import MethodId
from ...core.domain.models import HotspotResult, LCAResult
from ...core.services.normalization import NormalizationReport


@dataclass
class RunLCAResult:
    """Resultado del caso de uso RunLCAUseCase.

    Contiene las entidades de dominio ya normalizadas, listas para ser
    consumidas por presentadores o exportadores.

    Attributes
    ----------
    lca_results : List[LCAResult]
        Resultados LCIA por proyecto/método (con score_per_kwh normalizado).
    hotspots : List[HotspotResult]
        Hotspots por proyecto/método (con impact_per_kwh normalizado).
    methods : List[MethodId]
        Métodos de impacto evaluados.
    norm_report : Optional[NormalizationReport]
        Reporte de normalización por generación.
    cache_path : Optional[Path]
        Ruta del archivo de caché si se guardó.
    elapsed_seconds : float
        Tiempo total de ejecución en segundos.
    success : bool
        True si el cálculo finalizó sin errores.
    error_message : Optional[str]
        Mensaje de error si success=False.
    """

    lca_results: list[LCAResult] = field(default_factory=list)
    hotspots: list[HotspotResult] = field(default_factory=list)
    methods: list[MethodId] = field(default_factory=list)
    norm_report: NormalizationReport | None = None
    cache_path: Path | None = None
    elapsed_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None

    @property
    def n_methods(self) -> int:
        """Retorna el número de métodos evaluados."""
        return len(self.methods)

    @property
    def n_projects(self) -> int:
        """Retorna el número de proyectos únicos evaluados."""
        return len({r.project_id for r in self.lca_results})

    def get_results_by_project(self, project_name: str) -> list[LCAResult]:
        """Retorna los resultados LCIA de un proyecto específico."""
        return [r for r in self.lca_results if r.project_id == project_name]

    def get_hotspots_by_project(self, project_name: str) -> list[HotspotResult]:
        """Retorna los hotspots de un proyecto específico."""
        return [h for h in self.hotspots if h.project_id == project_name]
