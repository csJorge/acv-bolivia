"""
infrastructure.brightway.bw_context: Conector y Gestor del Entorno Brightway2.

Fachada de bajo nivel que configura variables de entorno, inyecta dinámicamente
directorios de datos en sys.path y orquesta las importaciones diferidas de los
módulos científicos de Brightway2 (bw2calc, bw2data, bw2io).

Garantiza el aislamiento del contexto de ejecución en entornos interactivos
(Google Colab) o CLI.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from ...infrastructure.brightway.constants import (
    BRIGHTWAY2_DIR_ENV_VAR,
    SITE_PACKAGES_DIR,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Componentes internos (SRP)
# ==============================================================================


class _EnvironmentConfigurator:
    """Configura variables de entorno e inyecta rutas en sys.path."""

    def configure(
        self,
        brightway_dir: str | None,
        env_path: str | None,
    ) -> None:
        if brightway_dir:
            os.environ[BRIGHTWAY2_DIR_ENV_VAR] = brightway_dir
            logger.info("%s configurado en: %s", BRIGHTWAY2_DIR_ENV_VAR, brightway_dir)
        elif brightway_dir is not None:
            logger.warning("La ruta de Brightway2 está vacía y será ignorada.")

        if env_path is None:
            return
        if not os.path.isdir(env_path):
            logger.warning("El entorno configurado no existe: %s", env_path)
            return

        site_packages = self._resolve_site_packages(env_path)
        if site_packages is None:
            logger.warning(
                "No se encontró site-packages en el entorno configurado: %s", env_path
            )
            return
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
            logger.info("sys.path expandido dinámicamente: %s", site_packages)

    @staticmethod
    def _resolve_site_packages(env_path: str) -> str | None:
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = (
            os.path.join(env_path, "lib", python_version, SITE_PACKAGES_DIR),
            os.path.join(env_path, "Lib", SITE_PACKAGES_DIR),
        )
        return next((path for path in candidates if os.path.isdir(path)), None)


class _BrightwayImporter:
    """Realiza las importaciones diferidas de los módulos científicos."""

    def import_modules(self) -> tuple[Any, Any, Any, Any]:
        try:
            import brightway2 as bw
            import bw2calc as bc
            import bw2data as bd
            import bw2io as bi

            return bw, bd, bi, bc
        except ImportError as e:
            raise ImportError(
                f"No se pudieron importar los módulos de Brightway2. "
                f"Verifique que el paquete esté instalado en el entorno activo. "
                f"Error original: {e}"
            ) from e


class _ProjectManager:
    """Gestiona la creación y activación de proyectos en Brightway2."""

    def __init__(self, bd_module: Any) -> None:
        self.bd = bd_module

    def ensure_project(self, project_name: str) -> None:
        if project_name not in self.bd.projects:
            self.bd.projects.create_project(project_name)
            logger.info("Proyecto '%s' creado en el backend.", project_name)

        self.bd.projects.set_current(project_name)
        logger.info("Contexto activo conmutado a: '%s'", project_name)


# ==============================================================================
# Conector principal (Orquestador)
# ==============================================================================


class BrightwayConnector:
    """Gestiona la sesión y el estado de conexión con Brightway2.

    Implementa el protocolo BrightwayConnectionManager para cumplir con DIP.
    Aplica importaciones diferidas defensivas y habilita la inyección de
    entornos virtuales aislados.
    """

    def __init__(
        self,
        project_name: str,
        brightway_dir: str | None = None,
        env_path: str | None = None,
        local_db_name: str | None = None,
    ) -> None:
        if not project_name or not project_name.strip():
            raise ValueError("project_name debe ser un nombre no vacío.")

        self.project_name = project_name
        self.brightway_dir = brightway_dir
        self.env_path = env_path
        self._local_db_name = local_db_name or project_name

        # Componentes internos (SRP)
        self._env_configurator = _EnvironmentConfigurator()
        self._importer = _BrightwayImporter()
        self._project_manager: _ProjectManager | None = None

        # Estado diferido
        self.bw: Any | None = None
        self.bd: Any | None = None
        self.bi: Any | None = None
        self.bc: Any | None = None

        self._connected = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establece la conexión física con el backend e inicializa el proyecto."""
        if self._connected:
            return

        try:
            self._env_configurator.configure(self.brightway_dir, self.env_path)
            bw, bd, bi, bc = self._importer.import_modules()
            project_manager = _ProjectManager(bd)
            project_manager.ensure_project(self.project_name)
        except Exception:
            self.bw = self.bd = self.bi = None
            self.bc = None
            self._project_manager = None
            self._connected = False
            raise

        self.bw, self.bd, self.bi, self.bc = bw, bd, bi, bc
        self._project_manager = project_manager
        self._connected = True

    def disconnect(self) -> None:
        """Libera recursos y limpia el estado de conexión."""
        self.bw = self.bd = self.bi = self.bc = None
        self._project_manager = None
        self._connected = False
        logger.info("Conexión a Brightway2 cerrada.")

    def get_local_db_name(self) -> str:
        """Retorna el nombre de la base de datos local del proyecto."""
        return self._local_db_name

    def get_data_module(self) -> Any:
        self._assert_connected()
        return self.bd

    def get_calc_module(self) -> Any:
        self._assert_connected()
        return self.bc

    def get_io_module(self) -> Any:
        self._assert_connected()
        return self.bi

    def get_all_modules(self) -> tuple[Any, Any, Any, Any]:
        self._assert_connected()
        return self.bw, self.bd, self.bi, self.bc

    def available_databases(self) -> set[str]:
        self._assert_connected()
        assert self.bd is not None
        return set(self.bd.databases)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Guards internos
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "BrightwayConnector no está conectado. "
                "Debe invocar .connect() antes de acceder a los módulos."
            )
