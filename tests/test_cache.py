"""Tests de la caché de resultados: persistencia + use_cache en ACVEngine.

Cubren el flujo completo sin Brightway2:
- ResultsFileRepository round-trip con los DTO actuales (run_lca/mc/sensitivity).
- find_latest_cache() entre carpetas fechadas.
- ACVEngine.run_lca/run_montecarlo/run_sensitivity cargan desde caché sin
  recalcular (short-circuit), incluso sin build() previo cuando hay caché.
"""

from __future__ import annotations

import os

import pytest

from acv_bolivia.application.dto.run_lca import RunLCAResult
from acv_bolivia.application.dto.run_montecarlo import RunMonteCarloResult
from acv_bolivia.application.dto.run_sensitivity import RunSensitivityResult
from acv_bolivia.config.app_config import AppConfig
from acv_bolivia.core.domain.models import LCAResult, SensitivityReport
from acv_bolivia.infrastructure.persistence import (
    ResultsFileRepository,
    find_latest_cache,
    load_latest_cache,
)
from acv_bolivia.interfaces.acv_engine import ACVEngine

METODO = ("ReCiPe 2016", "climate change", "kg CO2 eq")


# ==============================================================================
# Helpers
# ==============================================================================


def _make_engine(tmp_path, name: str = "resultados") -> ACVEngine:
    config = AppConfig({}, base_output_path=str(tmp_path / name))
    return ACVEngine(config)


def _save_cache(tmp_path, phase: str, date_dir: str, filename: str, data) -> None:
    folder = tmp_path / "resultados" / phase / date_dir
    folder.mkdir(parents=True, exist_ok=True)
    ResultsFileRepository(folder).save(data, filename)


def _stamp_mtime(path, timestamp: int) -> None:
    os.utime(path, (timestamp, timestamp))


# ==============================================================================
# ResultsFileRepository contra los DTO actuales
# ==============================================================================


def test_repository_roundtrip_guarda_y_carga_run_lca_result(tmp_path):
    repo = ResultsFileRepository(tmp_path / "cache")
    result = RunLCAResult(
        lca_results=[
            LCAResult(
                project_id="El Dorado",
                method_id=METODO,
                method_label="climate change",
                score=123.4,
                score_per_kwh=0.012,
            )
        ],
        methods=[METODO],
        success=True,
    )
    repo.save(result, "lca_results")

    loaded = repo.load("lca_results")
    assert isinstance(loaded, RunLCAResult)
    assert loaded == result
    assert loaded.lca_results[0].project_id == "El Dorado"


def test_repository_load_devuelve_none_si_no_existe(tmp_path):
    repo = ResultsFileRepository(tmp_path / "cache")
    assert repo.load("no_existe") is None


def test_repository_load_no_rechaza_dto_por_tipo(tmp_path):
    """Regresión: el load() anterior validaba un tipo legacy (get_lca_results)
    e hizo TypeError con los DTO actuales (RunLCAResult/RunMonteCarloResult)."""
    repo = ResultsFileRepository(tmp_path / "cache")
    repo.save(RunMonteCarloResult(iterations_completed=500), "montecarlo_results")
    loaded = repo.load("montecarlo_results")
    assert isinstance(loaded, RunMonteCarloResult)
    assert loaded.iterations_completed == 500


# ==============================================================================
# find_latest_cache entre carpetas fechadas
# ==============================================================================


def test_find_latest_cache_elige_el_mas_reciente(tmp_path):
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-28_09-00-00",
        "smc_bw",
        RunMonteCarloResult(iterations_completed=100),
    )
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-29_10-00-00",
        "smc_bw",
        RunMonteCarloResult(iterations_completed=200),
    )
    antiguo = (
        tmp_path / "resultados" / "montecarlo" / "2026-08-28_09-00-00" / "smc_bw.pkl.gz"
    )
    reciente = (
        tmp_path / "resultados" / "montecarlo" / "2026-08-29_10-00-00" / "smc_bw.pkl.gz"
    )
    _stamp_mtime(antiguo, 1_000_000)
    _stamp_mtime(reciente, 2_000_000)

    base = tmp_path / "resultados"
    assert find_latest_cache(base, "montecarlo", "smc_bw") == reciente

    loaded = load_latest_cache(base, "montecarlo", "smc_bw")
    assert loaded.iterations_completed == 200


def test_find_latest_cache_devuelve_none_si_no_hay_carpetas(tmp_path):
    assert find_latest_cache(tmp_path / "resultados", "montecarlo", "smc_bw") is None


# ==============================================================================
# ACVEngine: short-circuit por caché
# ==============================================================================


def test_run_lca_carga_desde_cache_sin_build_previo(tmp_path):
    _save_cache(
        tmp_path,
        "lca",
        "2026-08-29_10-00-00",
        "lca_results",
        RunLCAResult(
            lca_results=[
                LCAResult(
                    project_id="El Dorado",
                    method_id=METODO,
                    method_label="climate change",
                    score=1.0,
                    score_per_kwh=0.01,
                )
            ],
            methods=[METODO],
        ),
    )
    eng = _make_engine(tmp_path)
    eng.run_lca()  # use_cache=True por defecto → no exige build()

    assert eng.build_result is None
    assert eng.lca_result is not None
    assert eng.lca_result.methods == [METODO]


def test_run_lca_sin_cache_y_sin_build_requiere_build(tmp_path):
    eng = _make_engine(tmp_path)
    with pytest.raises(RuntimeError, match="build"):
        eng.run_lca()


def test_run_lca_use_cache_false_sin_build_requiere_build(tmp_path):
    _save_cache(
        tmp_path,
        "lca",
        "2026-08-29_10-00-00",
        "lca_results",
        RunLCAResult(methods=[METODO]),
    )
    eng = _make_engine(tmp_path)
    with pytest.raises(RuntimeError, match="build"):
        eng.run_lca(use_cache=False)  # desactivar caché → recalentar siempre


def test_run_montecarlo_carga_nombre_especifico(tmp_path):
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-29_09-00-00",
        "smc_bw",
        RunMonteCarloResult(iterations_completed=100, modes_run=["bw_mc"]),
    )
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-29_10-00-00",
        "smc_piv",
        RunMonteCarloResult(iterations_completed=200, modes_run=["piv"]),
    )
    eng = _make_engine(tmp_path)
    eng.run_montecarlo(use_cache=True, cache_filename="smc_piv")

    assert eng.build_result is None
    assert eng.mc_result is not None
    assert eng.mc_result.iterations_completed == 200
    assert eng.mc_result.modes_run == ["piv"]


def test_run_sensibilidad_carga_desde_cache(tmp_path):
    report = SensitivityReport(
        project_id="El Dorado",
        method_id=METODO,
        methods_run=["delta_lca"],
    )
    _save_cache(
        tmp_path,
        "sensibilidad",
        "2026-08-29_10-00-00",
        "sensitivity_results",
        RunSensitivityResult(reports=[report], success=True),
    )
    eng = _make_engine(tmp_path)
    eng.run_sensitivity()

    assert eng.build_result is None
    assert eng.sensitivity_result is not None
    assert eng.sensitivity_result.get_report("El Dorado", METODO) is not None


# ==============================================================================
# list_caches
# ==============================================================================


def test_list_caches_devuelve_nombres_disponibles(tmp_path):
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-28_09-00-00",
        "smc_bw",
        RunMonteCarloResult(iterations_completed=100),
    )
    _save_cache(
        tmp_path,
        "montecarlo",
        "2026-08-29_10-00-00",
        "smc_piv",
        RunMonteCarloResult(iterations_completed=200),
    )
    _stamp_mtime(
        tmp_path
        / "resultados"
        / "montecarlo"
        / "2026-08-28_09-00-00"
        / "smc_bw.pkl.gz",
        1_000_000,
    )
    _stamp_mtime(
        tmp_path
        / "resultados"
        / "montecarlo"
        / "2026-08-29_10-00-00"
        / "smc_piv.pkl.gz",
        2_000_000,
    )

    eng = _make_engine(tmp_path)
    caches = eng.list_caches("montecarlo")
    assert "smc_piv" in caches and "smc_bw" in caches
    assert caches.index("smc_piv") < caches.index("smc_bw")  # más reciente primero


def test_list_caches_fase_invalida_levanta_valueerror(tmp_path):
    eng = _make_engine(tmp_path)
    with pytest.raises(ValueError, match="montecarlo"):
        eng.list_caches("otra_cosa")
