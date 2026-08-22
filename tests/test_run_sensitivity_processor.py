"""Tests del cableado del procesador DEP/MIX en el análisis de sensibilidad.

Verifica que RunSensitivityUseCase construye el SampleProcessingStrategy por
proyecto (via processor_factory inyectada), lo aplica sobre las muestras
sintéticas antes de los analizadores por muestra y lo entrega al proveedor LCA
para que los métodos vivos (Sobol, Morris, Delta) evalúen con las dependencias
activas.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from acv_bolivia.application.use_cases.run_sensitivity import (
    RunSensitivityUseCase,
    SensitivityRequest,
)
from acv_bolivia.config.app_config import AppConfig


def _make_fakes():
    proj = SimpleNamespace(
        name="proy1",
        exchanges=[
            SimpleNamespace(component_id="a", exchange_type="technosphere"),
            SimpleNamespace(component_id="bio", exchange_type="biosphere"),
        ],
    )
    build_result = SimpleNamespace(success=True, projects=[proj])
    lca_result = SimpleNamespace(methods=[("EF 3.0", "Climate change")])
    return proj, build_result, lca_result


def _derived_processor(spy):
    def processor(samples):
        spy.applied += 1
        samples["b"] = samples["a"] * 2.0
        return samples

    return processor


def _make_use_case(
    builder,
    sampler_results=None,
):
    spy = SimpleNamespace(
        builder_calls=[],
        provider_args=[],
        engine_samples=[],
    )
    processor_spy = SimpleNamespace(applied=0)

    def synthetic_sampler(project, n_samples):
        spy.sampler = (project.name, n_samples)
        return sampler_results or {"a": np.array([1.0, 2.0, 3.0])}

    def provider_factory(project, sample_processor):
        spy.provider_args.append((project, sample_processor))
        return object()

    class FakeEngine:
        def run(
            self, project_name, method_tuple, component_samples_raw, lca_scores_raw
        ):
            spy.engine_samples.append((project_name, component_samples_raw))
            return SimpleNamespace(
                method_name=method_tuple[1], project_name=project_name
            )

    def engine_factory(lca_provider, analyzers, exclude_methods):
        return FakeEngine()

    use_case = RunSensitivityUseCase(
        config=AppConfig({}),
        lca_provider_factory=provider_factory,
        engine_factory=engine_factory,
        synthetic_sampler=synthetic_sampler,
        processor_factory=builder,
    )
    return use_case, spy, processor_spy


def test_applies_dependency_processor_to_samples_and_provider():
    proj, build_result, lca_result = _make_fakes()
    built = {}

    def builder(project, dependency_config, mix_config, enforce_physical_constraints):
        spy.builder_calls.append(
            (project, dependency_config, mix_config, enforce_physical_constraints)
        )
        built["processor"] = _derived_processor(processor_spy)
        return built["processor"]

    use_case, spy, processor_spy = _make_use_case(builder)

    request = SensitivityRequest(
        dependency_config={"b": {"base_comps": ["a"], "factor": 2.0}},
        enforce_physical_constraints=False,
    )
    result = use_case.run(build_result, lca_result, request)

    assert result.success
    assert len(spy.builder_calls) == 1
    called_project, dep, mix, enforce = spy.builder_calls[0]
    assert called_project is proj
    assert dep == {"b": {"base_comps": ["a"], "factor": 2.0}}
    assert mix is None
    assert enforce is False

    assert processor_spy.applied == 1
    _, provider_processor = spy.provider_args[0]
    assert provider_processor is built["processor"]

    proj_name, samples_raw = spy.engine_samples[0]
    assert proj_name == "proy1"
    processed = {k: np.asarray(v) for k, v in samples_raw.items()}
    assert "b" in processed
    assert np.allclose(processed["b"], processed["a"] * 2.0)


def test_skips_processor_when_factory_returns_none():
    _proj, build_result, lca_result = _make_fakes()

    def builder(project, dependency_config, mix_config, enforce_physical_constraints):
        spy.builder_calls.append(
            (project, dependency_config, mix_config, enforce_physical_constraints)
        )
        return None

    use_case, spy, _processor_spy = _make_use_case(builder)

    result = use_case.run(build_result, lca_result, SensitivityRequest())

    assert result.success
    assert len(spy.builder_calls) == 1
    assert spy.provider_args[0][1] is None
    proj_name, samples_raw = spy.engine_samples[0]
    assert proj_name == "proy1"
    assert "b" not in samples_raw


def test_enforce_physical_constraints_defaults_to_true_for_parity():
    assert SensitivityRequest().enforce_physical_constraints is True


def test_merges_excel_mc_config_with_global_and_flat_rules():
    proj1 = SimpleNamespace(name="proy1", exchanges=[])
    proj2 = SimpleNamespace(name="proy2", exchanges=[])
    build_result = SimpleNamespace(success=True, projects=[proj1, proj2])
    lca_result = SimpleNamespace(methods=[("EF 3.0", "Climate change")])

    def builder(project, dependency_config, mix_config, enforce_physical_constraints):
        spy.builder_calls.append(
            (project, dependency_config, mix_config, enforce_physical_constraints)
        )
        return None

    use_case, spy, _processor_spy = _make_use_case(builder)

    request = SensitivityRequest(
        dependency_config={"flat": {"base_comps": ["a"], "factor": 1.0}},
        mix_config={1.0: ["a", "base_plain"]},
        mc_config={
            "GLOBAL": {
                "dependencies": {"g": {"base_comps": ["a"], "factor": 2.0}},
                "mixes": {1.0: ["a", "cobre_primary"]},
            },
            "proy1": {"dependencies": {"p1": {"base_comps": ["a"], "factor": 3.0}}},
            "proy2": {"mixes": {1.0: ["a", "cobre_reciclado"]}},
        },
        enforce_physical_constraints=False,
    )
    result = use_case.run(build_result, lca_result, request)

    assert result.success
    merged = {
        project.name: (dep, mix) for project, dep, mix, _enforce in spy.builder_calls
    }
    assert merged["proy1"][0] == {
        "flat": {"base_comps": ["a"], "factor": 1.0},
        "g": {"base_comps": ["a"], "factor": 2.0},
        "p1": {"base_comps": ["a"], "factor": 3.0},
    }
    assert merged["proy1"][1] == {1.0: ["a", "cobre_primary"]}
    assert merged["proy2"][0] == {
        "flat": {"base_comps": ["a"], "factor": 1.0},
        "g": {"base_comps": ["a"], "factor": 2.0},
    }
    assert merged["proy2"][1] == {1.0: ["a", "cobre_reciclado"]}
