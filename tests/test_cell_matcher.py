"""Pruebas aisladas del matching de componentes a celdas de matriz."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.infrastructure.brightway.cell_matcher import CellMatcher


class FakeInput(dict):
    def __init__(self, name: str, key: tuple[str, str], location: str = "GLO"):
        super().__init__(name=name, location=location)
        self.key = key


class FakeExchange(dict):
    def __init__(self, input_activity: FakeInput, amount: float, component: str | None = None):
        super().__init__(amount=amount)
        self.input = input_activity
        if component is not None:
            self["component"] = component


class FakeActivity:
    def __init__(self, exchanges: list[FakeExchange]):
        self._exchanges = exchanges

    def technosphere(self) -> list[FakeExchange]:
        return self._exchanges


def _project(*components: tuple[str, float]) -> Project:
    return Project(
        id="p1",
        name="El Dorado",
        generation_kwh=1000.0,
        exchanges=[
            Exchange(
                component_id=name,
                quantity=Quantity(amount=amount, unit="kg"),
                exchange_type="technosphere",
            )
            for name, amount in components
        ],
    )


def test_componentes_que_comparten_proceso_se_agrupan_en_una_celda():
    steel = FakeInput("steel", ("db", "steel"))
    activity = FakeActivity([
        FakeExchange(steel, 100.0, "tower"),
        FakeExchange(steel, 50.0, "foundation"),
    ])

    result = CellMatcher().match(
        _project(("tower", 100.0), ("foundation", 50.0)),
        activity,
        {steel.key: 4},
        output_idx=2,
        technical_map={"tower": "steel", "foundation": "steel"},
    )

    assert result.unmatched_components == []
    assert result.matrix_cell_to_comps == {(4, 2): ["tower", "foundation"]}
    assert result.shared_cells_groups == [["tower", "foundation"]]


def test_fallback_exige_monto_exacto():
    steel = FakeInput("steel", ("db", "steel"))
    activity = FakeActivity([FakeExchange(steel, 100.0)])

    result = CellMatcher().match(
        _project(("tower", 101.0)),
        activity,
        {steel.key: 4},
        output_idx=2,
        technical_map={"tower": "steel"},
    )

    assert result.unmatched_components == ["tower"]


def test_fallback_exige_ubicacion_cuando_se_proporciona():
    steel = FakeInput("steel", ("db", "steel"), location="RER")
    activity = FakeActivity([FakeExchange(steel, 100.0)])

    result = CellMatcher().match(
        _project(("tower", 100.0)),
        activity,
        {steel.key: 4},
        output_idx=2,
        technical_map={"tower": "steel"},
        location_map={"tower": "GLO"},
    )

    assert result.unmatched_components == ["tower"]


def test_metadata_con_monto_inconsistente_no_hace_fallback_ambiguo():
    steel_tagged = FakeInput("steel", ("db", "tagged"))
    steel_other = FakeInput("steel", ("db", "other"))
    activity = FakeActivity([
        FakeExchange(steel_tagged, 90.0, "tower"),
        FakeExchange(steel_other, 100.0),
    ])

    result = CellMatcher().match(
        _project(("tower", 100.0)),
        activity,
        {steel_tagged.key: 4, steel_other.key: 5},
        output_idx=2,
        technical_map={"tower": "steel"},
    )

    assert result.unmatched_components == ["tower"]


def test_fallback_ambiguo_no_elige_un_exchange_arbitrariamente():
    steel_a = FakeInput("steel", ("db", "steel_a"))
    steel_b = FakeInput("steel", ("db", "steel_b"))
    activity = FakeActivity([
        FakeExchange(steel_a, 100.0),
        FakeExchange(steel_b, 100.0),
    ])

    result = CellMatcher().match(
        _project(("tower", 100.0)),
        activity,
        {steel_a.key: 4, steel_b.key: 5},
        output_idx=2,
        technical_map={"tower": "steel"},
    )

    assert result.unmatched_components == ["tower"]
    assert result.matrix_cell_to_comps == {}


def test_rechaza_output_idx_negativo():
    with pytest.raises(ValueError, match="no negativo"):
        CellMatcher().match(_project(("tower", 100.0)), FakeActivity([]), {}, -1, {})
