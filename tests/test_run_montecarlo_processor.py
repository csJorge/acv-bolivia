"""Tests de la fusión de reglas DEP/MIX en el procesador de muestras de MC.

Verifica que _ProcessorFactoryAdapter (compositor MC) mezcla las reglas con
prioridad proyecto > GLOBAL > dependency_config/mix_config (flat), replicando
el comportamiento del análisis de sensibilidad.
"""

from __future__ import annotations

from types import SimpleNamespace

from acv_bolivia.infrastructure.brightway.montecarlo._sampling_rules import (
    DependencyRule,
    MixRule,
    PhysicalConstraintRule,
)
from acv_bolivia.infrastructure.composition.run_montecarlo_composer import (
    _ProcessorFactoryAdapter,
)


def _proj(name: str):
    return SimpleNamespace(
        name=name,
        exchanges=[
            SimpleNamespace(
                component_id="a",
                exchange_type="technosphere",
                quantity=SimpleNamespace(amount=10.0),
            )
        ],
    )


def _deps(processor):
    return {r.target_comp for r in processor.rules if isinstance(r, DependencyRule)}


def _mixes(processor):
    return {
        r.target_sum: tuple(r.components)
        for r in processor.rules
        if isinstance(r, MixRule)
    }


def test_merges_flat_global_and_project_rules_in_mc():
    adapter = _ProcessorFactoryAdapter()
    processors = adapter(
        mc_config={
            "GLOBAL": {
                "dependencies": {"g": {"base_comps": ["a"], "factor": 2.0}},
                "mixes": {1.0: ["a", "global"]},
            },
            "proy1": {"dependencies": {"p1": {"base_comps": ["a"], "factor": 3.0}}},
            "proy2": {"mixes": {1.0: ["a", "proy2"]}},
        },
        projects=[_proj("proy1"), _proj("proy2")],
        dependency_config={"flat": {"base_comps": ["a"], "factor": 1.0}},
        mix_config={1.0: ["flat_base", "a"]},
    )

    assert _deps(processors["proy1"]) == {"flat", "g", "p1"}
    assert _mixes(processors["proy1"]) == {1.0: ("a", "global")}

    assert _deps(processors["proy2"]) == {"flat", "g"}
    assert _mixes(processors["proy2"]) == {1.0: ("a", "proy2")}

    for name in ("proy1", "proy2"):
        assert any(
            isinstance(r, PhysicalConstraintRule) for r in processors[name].rules
        )


def test_flat_rules_as_fallback_when_no_mc_config():
    adapter = _ProcessorFactoryAdapter()
    processors = adapter(
        mc_config={},
        projects=[_proj("proy1")],
        dependency_config={"flat": {"base_comps": ["a"], "factor": 1.0}},
        mix_config={1.0: ["a", "base_plain"]},
    )

    assert _deps(processors["proy1"]) == {"flat"}
    assert _mixes(processors["proy1"]) == {1.0: ("a", "base_plain")}
