"""Tests de EcoinventNameResolver: resolución por código, unidad y nombre.

Verifica la corrección de un bug real: las bases de Ecoinvent contienen
procesos homónimos con la misma ubicación pero unidades distintas (p. ej.
``cubic meter`` y ``kilogram``). El índice de diccionario simple
``{(name, location): activity}`` colisionaba en esa clave y el proceso
seleccionado dependía del orden de iteración de la BD, provocando builds no
reproducibles (a veces se enlazaba a un proceso con impacto nulo).

La estrategia prioriza la clave canónica de Brightway (``code``), luego
``(nombre, ubicación, unidad)`` y, en último término, desempata por nombre de
forma determinista advirtiendo de la ambigüedad.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from acv_bolivia.infrastructure.brightway.ei_name_resolver import (
    EcoinventNameResolver,
)
from tests.fakes import FakeActivity, FakeDatabase


def _mk(name: str, unit: str, code: str) -> FakeActivity:
    """Construye una FakeActivity homónima con unidad y code propios."""
    act = FakeActivity(code=code, name=name, unit=unit, location="RoW")
    act["location"] = "RoW"
    return act


def _resolver(*activities: FakeActivity) -> tuple[EcoinventNameResolver, FakeDatabase]:
    db = FakeDatabase("ecoinvent 3.12 cutoff", activities=list(activities))
    return EcoinventNameResolver(db), db


def test_resuelve_por_code_ignorando_nombre_y_unidad() -> None:
    """El code es la fuente de verdad: resuelve directo aunque haya homónimos."""
    stub = _mk("concreto", "kilogram", code="code_kg")
    real = _mk("concreto", "cubic meter", code="code_m3")
    resolver, _ = _resolver(stub, real)
    resolver.build_index({"fundacion": "concreto"}, {"fundacion": "RoW"})

    act = resolver.resolve("fundacion", "concreto", "RoW", code="code_m3")
    assert act == real
    assert resolver.warnings == []


def test_code_no_encontrado_devuelve_none() -> None:
    """Un code inexistente en la BD resuelve a None y advierte."""
    resolver, _ = _resolver(_mk("concreto", "kilogram", code="code_kg"))
    resolver.build_index({"f": "concreto"}, {"f": "RoW"})

    act = resolver.resolve("f", "concreto", "RoW", code="code_inexistente")
    assert act is None
    assert any("CÓDIGO NO ENCONTRADO" in w for w in resolver.warnings)


def test_unidad_discrimina_entre_homonimos() -> None:
    """Con la unidad declarada, elige el proceso de esa unidad sin warning."""
    kg = _mk("concreto", "kilogram", code="code_kg")
    m3 = _mk("concreto", "cubic meter", code="code_m3")
    resolver, _ = _resolver(kg, m3)
    resolver.build_index({"fundacion": "concreto"}, {"fundacion": "RoW"})

    act = resolver.resolve("fundacion", "concreto", "RoW", unit="cubic meter")
    assert act == m3
    assert resolver.warnings == []


def test_unidad_sin_coincidencia_devuelve_none() -> None:
    """Una unidad que no existe en ninguno de los homónimos resuelve a None."""
    kg = _mk("concreto", "kilogram", code="code_kg")
    m3 = _mk("concreto", "cubic meter", code="code_m3")
    resolver, _ = _resolver(kg, m3)
    resolver.build_index({"f": "concreto"}, {"f": "RoW"})

    assert resolver.resolve("f", "concreto", "RoW", unit="megajoule") is None


def test_sin_code_ni_unidad_desempata_determinista_y_advierte() -> None:
    """Sin code ni unidad, el desempate por nombre es estable y avisa."""
    kg = _mk("concreto", "kilogram", code="aaa")
    m3 = _mk("concreto", "cubic meter", code="zzz")

    forward, _ = _resolver(kg, m3)
    forward.build_index({"f": "concreto"}, {"f": "RoW"})
    reverse, _ = _resolver(m3, kg)
    reverse.build_index({"f": "concreto"}, {"f": "RoW"})

    chosen = forward.resolve("f", "concreto", "RoW")
    assert chosen == m3  # "aaa" < "zzz" -> el mayor code gana
    assert forward.warnings != []
    assert reverse.resolve("f", "concreto", "RoW") == chosen


def test_sin_ambiguedad_no_genera_advertencia() -> None:
    """Un único proceso por (nombre, ubicación) resuelve sin warnings."""
    unico = _mk("acero", "kilogram", code="solo")
    resolver, _ = _resolver(unico)
    resolver.build_index({"acero": "acero"}, {"acero": "RoW"})

    act = resolver.resolve("acero", "acero", "RoW")
    assert act == unico
    assert resolver.warnings == []


def test_no_encontrado_devuelve_none() -> None:
    """Sin coincidencia de (nombre, ubicación) devuelve None."""
    resolver, _ = _resolver(_mk("acero", "kilogram", code="solo"))
    resolver.build_index({"x": "acero"}, {"x": "RoW"})

    assert resolver.resolve("x", "acero", "GLO") is None
