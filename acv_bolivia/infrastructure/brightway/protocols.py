"""
infrastructure/brightway/protocols
Protocolos específicos de la infraestructura Brightway2.
"""

from __future__ import annotations

from typing import Any, Protocol

from ...core.domain.models import MonteCarloResult, Project


class ActivityBuilder(Protocol):
    """Protocolo para construir actividades en Brightway2."""

    def build(
        self,
        projects: list[Project],
        location_map: dict[str, str],
        technical_map: dict[str, str],
        force_rebuild: bool = False,
    ) -> None:
        """Compila y sincroniza la base de datos local."""
        ...

    def validate_ecoinvent(self) -> bool:
        """Verifica la disponibilidad de la base de datos de fondo."""
        ...


class BrightwayConnectionManager(Protocol):
    """Protocolo para gestionar la conexión a Brightway2."""

    def connect(self) -> None:
        """Establece la conexión y activa el proyecto."""
        ...

    def disconnect(self) -> None:
        """Libera recursos y limpia el estado."""
        ...

    def get_local_db_name(self) -> str:
        """Retorna el nombre de la base de datos local del proyecto."""
        ...

    def get_data_module(self) -> Any:
        """Retorna el módulo bw2data activo."""
        ...

    def available_databases(self) -> set[str]:
        """Retorna las bases de datos registradas."""
        ...


class MonteCarloSimulator(Protocol):
    """Protocolo para simuladores Monte Carlo."""

    def run(
        self,
        iterations: int,
        functional_unit: float = 1.0,
    ) -> MonteCarloResult:
        """Ejecuta la simulación y retorna los resultados."""
        ...

    def cleanup(self) -> None:
        """Libera recursos de memoria."""
        ...
