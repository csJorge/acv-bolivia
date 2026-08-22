"""Tests aislados para BrightwayLCAProvider mediante fakes."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.infrastructure.brightway.adapters import BrightwayLCAProvider


class FakeActivity:
    def __init__(self, name: str, key: tuple[str, str] = ("db", "activity")):
        self.name = name
        self.key = key

    def __getitem__(self, key: str):
        if key == "name":
            return self.name
        raise KeyError(key)


class FakeDatabase:
    def __init__(self, activities: list[FakeActivity]):
        self.activities = activities

    def __iter__(self):
        return iter(self.activities)


class FakeDatabaseModule:
    def __init__(self, activities: list[FakeActivity]):
        self.activities = activities

    def Database(self, _name: str):
        return FakeDatabase(self.activities)


def _provider(activities: list[FakeActivity]) -> BrightwayLCAProvider:
    project = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=1000.0,
        exchanges=[
            Exchange(
                component_id="tower",
                quantity=Quantity(amount=25.0, unit="kg"),
                exchange_type="technosphere",
            )
        ],
    )
    return BrightwayLCAProvider(
        bc_module=SimpleNamespace(),
        bd_module=FakeDatabaseModule(activities),
        local_db_name="local",
        project=project,
        technical_map={},
    )


def test_get_nominal_parameters_rechaza_otro_proyecto():
    provider = _provider([])

    with pytest.raises(ValueError, match="configurado para el proyecto"):
        provider.get_nominal_parameters("Gas Natural")


def test_get_nominal_parameters_devuelve_solo_tecnosfera():
    provider = _provider([])

    assert provider.get_nominal_parameters("El Dorado") == {"tower": 25.0}


def test_resolve_activity_rechaza_actividad_inexistente():
    provider = _provider([])

    with pytest.raises(RuntimeError, match="no encontrada"):
        provider._resolve_activity("El Dorado")


def test_resolve_activity_rechaza_nombres_duplicados():
    provider = _provider([FakeActivity("El Dorado"), FakeActivity("El Dorado")])

    with pytest.raises(RuntimeError, match="debe ser único"):
        provider._resolve_activity("El Dorado")


def test_get_act_dict_acepta_dicts_activity():
    activity = FakeActivity("El Dorado")
    provider = _provider([activity])
    provider.bc = SimpleNamespace(
        LCA=lambda *_args: SimpleNamespace(
            dicts=SimpleNamespace(activity={activity.key: 3}),
            lci=lambda: None,
            lcia=lambda: None,
        )
    )

    assert provider._get_act_dict(activity, ("method",), 1.0) == {activity.key: 3}


def test_get_act_dict_acepta_activity_dict():
    activity = FakeActivity("El Dorado")
    provider = _provider([activity])
    provider.bc = SimpleNamespace(
        LCA=lambda *_args: SimpleNamespace(
            activity_dict={activity.key: 4},
            lci=lambda: None,
            lcia=lambda: None,
        )
    )

    assert provider._get_act_dict(activity, ("method",), 1.0) == {activity.key: 4}


def test_get_act_dict_rechaza_api_sin_diccionario_de_actividades():
    activity = FakeActivity("El Dorado")
    provider = _provider([activity])
    provider.bc = SimpleNamespace(
        LCA=lambda *_args: SimpleNamespace(
            lci=lambda: None,
            lcia=lambda: None,
        )
    )

    with pytest.raises(RuntimeError, match="no expone"):
        provider._get_act_dict(activity, ("method",), 1.0)
