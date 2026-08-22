"""Tests de la resolución de ajustes de los analizadores de sensibilidad.

Verifica la cadena "entrada de usuario > settings.json > fallback" aplicada a
los parámetros de cada solver y a exclude_methods, replicando la misma lógica
que el método de impacto del LCA (patron_metodo > lca.patron_metodo > default).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from acv_bolivia.application.use_cases.run_sensitivity import (
    RunSensitivityUseCase,
    SensitivityRequest,
)
from acv_bolivia.config.app_config import AppConfig


def _make_fake_analyzer(name, requires_variance=False):
    return SimpleNamespace(
        method_name=name,
        requires_variance=requires_variance,
    )


def _make_fakes():
    proj = SimpleNamespace(
        name="proy1",
        exchanges=[SimpleNamespace(component_id="a", exchange_type="technosphere")],
    )
    build_result = SimpleNamespace(success=True, projects=[proj])
    lca_result = SimpleNamespace(methods=[("EF 3.0", "Climate change")])
    return proj, build_result, lca_result


def _make_use_case(config):
    captured = {"settings": None, "exclude": None}

    STANDARD_METHODS = {
        "delta_lca",
        "morris",
        "sobol",
        "shap",
        "correlation",
        "regression",
    }
    VARIANCE_METHODS = {"correlation", "regression", "shap"}

    def analyzer_factory(settings):
        captured["settings"] = settings
        return [
            _make_fake_analyzer(name, name in VARIANCE_METHODS)
            for name in sorted(STANDARD_METHODS)
        ]

    def engine_factory(lca_provider, analyzers, exclude_methods):
        captured["exclude"] = set(exclude_methods)

        def _run(project_name, method_tuple, component_samples_raw, lca_scores_raw):
            return SimpleNamespace(
                method_name=method_tuple[1],
                project_name=project_name,
            )

        return SimpleNamespace(run=_run)

    def synthetic_sampler(project, n_samples):
        return {"a": np.array([1.0, 2.0, 3.0])}

    use_case = RunSensitivityUseCase(
        config=config,
        lca_provider_factory=lambda project, sample_processor: object(),
        engine_factory=engine_factory,
        synthetic_sampler=synthetic_sampler,
        analyzer_factory=analyzer_factory,
    )
    return use_case, captured


def test_user_settings_override_config_and_defaults():
    _proj, build_result, lca_result = _make_fakes()
    config = AppConfig(
        {
            "sensibilidad": {
                "exclude_methods": ["morris"],
                "analyzers": {"sobol": {"n_samples": 300}},
            }
        }
    )
    use_case, captured = _make_use_case(config)

    request = SensitivityRequest(
        analyzer_settings={"sobol": {"n_samples": 64}, "morris": {"n_levels": 5}}
    )
    result = use_case.run(build_result, lca_result, request)

    assert result.success
    resolved = captured["settings"]
    # Usuario > config > (default se fusiona dentro de la fábrica real).
    assert resolved["sobol"] == {"n_samples": 64}
    assert resolved["morris"] == {"n_levels": 5}
    assert resolved["shap"] == {}
    # exclude_methods sale de settings (el usuario no pasa ninguno).
    assert "morris" in captured["exclude"]


def test_config_settings_used_when_user_silent():
    _proj, build_result, lca_result = _make_fakes()
    config = AppConfig(
        {
            "sensibilidad": {
                "analyzers": {"sobol": {"n_samples": 300, "top_k_screening": 6}}
            }
        }
    )
    use_case, captured = _make_use_case(config)

    result = use_case.run(build_result, lca_result, SensitivityRequest())

    assert result.success
    resolved = captured["settings"]
    assert resolved["sobol"] == {"n_samples": 300, "top_k_screening": 6}
    assert resolved["morris"] == {}
    assert resolved["delta_lca"] == {}


def test_legacy_flat_aliases_still_read():
    _proj, build_result, lca_result = _make_fakes()
    config = AppConfig(
        {
            "sensibilidad": {
                "delta_values": [0.2, 0.5],
                "sobol_n_samples": 1024,
                "shap_explainer": "linear",
            }
        }
    )
    use_case, captured = _make_use_case(config)

    result = use_case.run(build_result, lca_result, SensitivityRequest())

    assert result.success
    resolved = captured["settings"]
    assert resolved["delta_lca"] == {"deltas": [0.2, 0.5]}
    assert resolved["sobol"] == {"n_samples": 1024}
    assert resolved["shap"] == {"explainer_type": "linear"}


def test_no_settings_no_user_falls_back_to_empty_merge():
    _proj, build_result, lca_result = _make_fakes()
    use_case, captured = _make_use_case(AppConfig({}))

    result = use_case.run(build_result, lca_result, SensitivityRequest())

    assert result.success
    resolved = captured["settings"]
    assert all(settings == {} for settings in resolved.values())
    # Sin exclude del usuario ni de settings: no se excluye morris (siempre se
    # excluyen los métodos por varianza al no haber MC previo).
    assert "morris" not in captured["exclude"]
    assert "shap" in captured["exclude"]


def test_user_exclude_methods_override_settings():
    _proj, build_result, lca_result = _make_fakes()
    config = AppConfig({"sensibilidad": {"exclude_methods": ["morris"]}})
    use_case, captured = _make_use_case(config)

    request = SensitivityRequest(exclude_methods={"shap"})
    result = use_case.run(build_result, lca_result, request)

    assert result.success
    # El exclude del usuario reemplaza al de settings: morris debería ejecutarse.
    assert "morris" not in captured["exclude"]
    assert "shap" in captured["exclude"]


def test_factory_adapter_merges_settings_over_defaults():
    from acv_bolivia.infrastructure.composition.run_sensitivity_composer import (
        _AnalyzerFactoryAdapter,
    )

    factory = _AnalyzerFactoryAdapter()
    analyzers = factory(
        {
            "sobol": {"n_samples": 256, "calc_second_order": True},
            "morris": {"n_trajectories": 32},
        }
    )
    by_name = {a.method_name: a for a in analyzers}
    assert by_name["sobol"]._n_samples == 256
    assert by_name["sobol"]._calc_second_order is True
    assert by_name["sobol"]._top_k == 8
    assert by_name["morris"]._n_trajectories == 32
    assert by_name["morris"]._n_levels == 4
    assert by_name["shap"]._explainer_type == "tree"
    assert by_name["delta_lca"]._deltas == [0.1, 0.2]
    assert by_name["regression"]._compute_src is True


def test_factory_adapter_defaults_without_settings():
    from acv_bolivia.infrastructure.composition.run_sensitivity_composer import (
        _AnalyzerFactoryAdapter,
    )

    analyzers = _AnalyzerFactoryAdapter()()
    by_name = {a.method_name: a for a in analyzers}
    assert by_name["sobol"]._n_samples == 512
    assert by_name["morris"]._n_trajectories == 20
    assert by_name["correlation"]._compute_pearson is True
