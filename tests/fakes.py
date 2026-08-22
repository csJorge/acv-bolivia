"""Fakes reutilizables para testear acv_bolivia sin Brightway2/Ecoinvent real.

Filosofía: un FAKE tiene comportamiento real (aunque simplificado), no es un
`Mock()` que solo registra qué se llamó y devuelve valores prefabricados. Un
test contra un fake ejecuta la lógica real del código bajo prueba (indexado,
resolución de biosfera, cálculo de incertidumbre, etc.) contra estructuras de
datos que se comportan como las de Brightway2, sin necesitar la instalación
real ni una base de datos de Ecoinvent con licencia.

Por qué esto importa: acv_bolivia hace bastante aritmética
(matrices CSR, parámetros estadísticos de incertidumbre, indexado por
nombre+ubicación). Un mock que simplemente devuelve `Mock(name="acero")` no
puede delatar un error en esa aritmética, un fake con datos reales sí,
porque el código bajo prueba corre de verdad contra él.

Los fakes de este módulo imitan la superficie mínima de bw2data que
`infrastructure/brightway/*.py` realmente usa (confirmado leyendo el código
fuente, no adivinando la API completa de bw2data):

    - bd.Database(nombre)  -> objeto iterable de actividades
    - bd.databases          -> dict-like, soporta `in` y `del bd.databases[x]`
    - Database.register() / .process() / .new_activity(code, name, unit)
    - Activity: dict-like (acceso por .get()/[]), .new_exchange(**kw).save()
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


class FakeExchange(dict):
    """Imita un Exchange de bw2data: dict-like, con .save() no-op."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.saved = False

    def save(self) -> None:
        self.saved = True


class FakeActivity(dict):
    """Imita una Activity de bw2data: dict-like, con new_exchange()/.save().

    Registra en `exchanges_created` cada exchange creado, en orden. Útil
    para verificar qué se generó (biosfera vs tecnosfera, parámetros de
    incertidumbre, montos) sin depender de una BD real.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exchanges_created: list[FakeExchange] = []
        self.saved = False

    def new_exchange(self, **kwargs: Any) -> FakeExchange:
        exc = FakeExchange(**kwargs)
        self.exchanges_created.append(exc)
        return exc

    def save(self) -> None:
        self.saved = True

    def technosphere_exchanges(self) -> list[FakeExchange]:
        return [e for e in self.exchanges_created if e.get("type") == "technosphere"]

    def biosphere_exchanges(self) -> list[FakeExchange]:
        return [e for e in self.exchanges_created if e.get("type") == "biosphere"]

    def exchanges(self) -> list[FakeExchange]:
        """Imita Activity.exchanges(): todos los intercambios creados."""
        return list(self.exchanges_created)


class FakeDatabase:
    """Imita bw2data.Database(nombre): iterable de actividades + ciclo de vida."""

    def __init__(
        self, name: str, activities: Iterable[FakeActivity] | None = None
    ) -> None:
        self.name = name
        self._activities: list[FakeActivity] = list(activities) if activities else []
        self.registered = False
        self.processed = False

    def __iter__(self) -> Iterator[FakeActivity]:
        return iter(self._activities)

    def __len__(self) -> int:
        return len(self._activities)

    def register(self) -> None:
        self.registered = True

    def process(self) -> None:
        self.processed = True

    def new_activity(self, code: str, name: str, unit: str) -> FakeActivity:
        act = FakeActivity(code=code, name=name, unit=unit, location="")
        self._activities.append(act)
        return act


class FakeBW2Module:
    """Imita el módulo bd (bw2data) en su superficie mínima usada por acv_bolivia.

    Uso típico:

        eco = FakeDatabase("ecoinvent 3.12 cutoff", activities=[
            FakeActivity(name="steel production", location="GLO"),
        ])
        bio = FakeDatabase("biosphere3", activities=[
            FakeActivity(name="Carbon dioxide, fossil"),
        ])
        bd = FakeBW2Module({"ecoinvent 3.12 cutoff": eco, "biosphere3": bio})
    """

    def __init__(self, databases: dict[str, FakeDatabase] | None = None) -> None:
        self.databases: dict[str, FakeDatabase] = databases or {}

    def Database(self, name: str) -> FakeDatabase:
        if name not in self.databases:
            self.databases[name] = FakeDatabase(name=name)
        return self.databases[name]


# ==============================================================================
# Fake de la matriz tecnosfera (para tests de _matrix_utils.py y similares)
# ==============================================================================


class FakeLCA:
    """Doble mínimo de un objeto lca de bw2calc, solo los atributos que
    infrastructure/brightway/montecarlo/_matrix_utils.py realmente usa
    (technosphere_matrix, tech_params). Usa scipy/numpy reales, no valores
    prefabricados, para poder probar aritmética real de matrices dispersas.
    """

    technosphere_matrix: Any
    tech_params: Any
