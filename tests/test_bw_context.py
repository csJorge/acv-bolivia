"""Pruebas del conector sin instalar Brightway ni sus dependencias."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from acv_bolivia.infrastructure.brightway.bw_context import (
    BrightwayConnector,
    _EnvironmentConfigurator,
)


class FakeProjectManager:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.current: str | None = None

    def __contains__(self, _name: str) -> bool:
        return False

    def create_project(self, name: str) -> None:
        self.created.append(name)

    def set_current(self, name: str) -> None:
        self.current = name


class FakeImporter:
    def __init__(self, modules=None, error: Exception | None = None) -> None:
        self.modules = modules or ("bw", SimpleNamespace(databases={}), "bi", "bc")
        self.error = error

    def import_modules(self):
        if self.error is not None:
            raise self.error
        return self.modules


def test_resolve_site_packages_reconoce_windows(tmp_path):
    site_packages = tmp_path / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    assert _EnvironmentConfigurator._resolve_site_packages(str(tmp_path)) == str(
        site_packages
    )


def test_resolve_site_packages_reconoce_linux(tmp_path):
    import sys

    site_packages = (
        tmp_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)

    assert _EnvironmentConfigurator._resolve_site_packages(str(tmp_path)) == str(
        site_packages
    )


def test_conector_exige_nombre_de_proyecto():
    with pytest.raises(ValueError, match="no vacío"):
        BrightwayConnector(project_name="   ")


def test_conector_limpia_estado_si_falla_el_importador():
    connector = BrightwayConnector(project_name="tesis")
    connector._importer = FakeImporter(error=ImportError("dependencia ausente"))

    with pytest.raises(ImportError, match="dependencia ausente"):
        connector.connect()

    assert connector.is_connected is False
    assert connector.bw is None
    assert connector.bd is None
    assert connector._project_manager is None


def test_available_databases_requiere_conexion():
    connector = BrightwayConnector(project_name="tesis")

    with pytest.raises(RuntimeError, match="no está conectado"):
        connector.available_databases()