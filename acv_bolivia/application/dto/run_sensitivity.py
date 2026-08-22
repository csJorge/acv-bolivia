"""
application.dto.run_sensitivity: DTO del caso de uso RunSensitivityUseCase.

Representa el resultado del análisis de sensibilidad: un SensitivityReport
completo (core.domain.models) por cada combinación (proyecto, método).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...core.domain.contracts import MethodId
from ...core.domain.models import SensitivityReport


@dataclass
class RunSensitivityResult:
    """Resultado del caso de uso RunSensitivityUseCase.

    Contiene un SensitivityReport completo (core.domain.models) por cada
    combinación (proyecto, método) analizada - el mismo objeto que sirve de
    entrada a ACVEngine.plot_sensitivity() y ACVEngine.export_sensitivity(),
    no una copia resumida. Permite graficar o exportar sin volver a correr
    el análisis completo.

    Attributes
    ----------
    reports : List[SensitivityReport]
        Un SensitivityReport por combinación (proyecto, método) analizada.
        Cada uno trae sus propios .project_id, .method_id, .methods_run,
        .errors y .top_components().
    methods_executed : List[str]
        Nombres de los métodos de sensibilidad ejecutados.
    elapsed_seconds : float
        Tiempo total de ejecución en segundos.
    success : bool
        True si el análisis finalizó sin errores.
    error_message : Optional[str]
        Mensaje de error si success=False.
    cache_path : Optional[Path]
        Ruta del archivo de caché si se guardó.
    """

    reports: list[SensitivityReport] = field(default_factory=list)
    methods_executed: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    cache_path: Path | None = None

    @property
    def n_projects(self) -> int:
        """Retorna el número de proyectos analizados."""
        return len({r.project_id for r in self.reports})

    @property
    def n_methods_analyzed(self) -> int:
        """Retorna el número de métodos de impacto analizados."""
        return len({r.method_id for r in self.reports})

    def get_reports_for_project(self, project_id: str) -> list[SensitivityReport]:
        """Retorna todos los reportes de un proyecto específico."""
        return [r for r in self.reports if r.project_id == project_id]

    def get_report(
        self, project_id: str, method_id: MethodId
    ) -> SensitivityReport | None:
        """Retorna el SensitivityReport de una combinación (proyecto, método).

        Parameters
        ----------
        project_id : str
            Nombre del proyecto.
        method_id : MethodId
            Método de impacto.

        Returns
        -------
        Optional[SensitivityReport]
            El reporte completo, o None si esa combinación no se analizó
            (o falló durante el análisis).
        """
        return next(
            (
                r
                for r in self.reports
                if r.project_id == project_id and r.method_id == method_id
            ),
            None,
        )
