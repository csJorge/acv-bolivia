"""Tests para interfaces.acv_engine.ACVEngine.

No requieren Brightway2/Ecoinvent reales: cubren construcción, las guardas
de orden de fases, y la lógica pura de transformación de datos entre
BuildInventoryResult (mapas planos) y lo que los compositores de
Monte Carlo/Sensibilidad esperan (mapas anidados por proyecto), esa
transformación vive inline en run_lca()/run_montecarlo()/run_sensitivity()
y es fácil de romper sin darse cuenta al tocar esos métodos, así que vale
la pena tener un test que la ejercite de forma aislada.
"""
from __future__ import annotations

import pytest

from acv_bolivia.application.dto.build_inventory import BuildInventoryResult
from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.interfaces.acv_engine import ACVEngine


class _FakeAppConfig:
    """Fake mínimo de AppConfig, solo lo que ACVEngine.__init__ toca."""

    BASE_OUTPUT_PATH = ""

    def get(self, key: str, default=None):
        return {"proyecto": "test_project", "rutas.bw2": None, "rutas.entorno": None}.get(key, default)


# ==============================================================================
# Construcción y guardas de orden
# ==============================================================================

def test_construccion_no_falla_y_no_conecta_a_brightway_todavia():
    """__init__ debe ser barato — sin conectar a BW2 hasta el primer uso real
    (conexión diferida, ver _get_connector)."""
    eng = ACVEngine(_FakeAppConfig())
    assert eng.build_result is None
    assert eng.lca_result is None
    assert eng.mc_result is None
    assert eng.sensitivity_result is None
    assert eng._connector is None  # todavía no se conectó


def test_run_lca_sin_build_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    with pytest.raises(RuntimeError, match="build"):
        eng.run_lca()


def test_run_montecarlo_sin_run_lca_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    eng._build_result = BuildInventoryResult()  # simula build() ya corrido
    with pytest.raises(RuntimeError, match="run_lca"):
        eng.run_montecarlo()


def test_run_sensitivity_sin_run_lca_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    eng._build_result = BuildInventoryResult()
    with pytest.raises(RuntimeError, match="run_lca"):
        eng.run_sensitivity()


def test_export_sin_run_lca_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    eng._build_result = BuildInventoryResult()
    with pytest.raises(RuntimeError, match="run_lca"):
        eng.export()


def test_plotter_sin_run_lca_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    eng._build_result = BuildInventoryResult()
    with pytest.raises(RuntimeError, match="run_lca"):
        _ = eng.plotter


def test_piv_plotter_sin_montecarlo_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    with pytest.raises(RuntimeError, match="run_montecarlo"):
        eng.piv_plotter("El Dorado")


def test_run_convergence_diagnostics_sin_montecarlo_previo_lanza_runtimeerror():
    eng = ACVEngine(_FakeAppConfig())
    with pytest.raises(RuntimeError, match="run_montecarlo"):
        eng.run_convergence_diagnostics("El Dorado", "climate change")


# ==============================================================================
# Lógica pura: inversión technical_map -> process_to_component (usada en run_lca)
# ==============================================================================

def test_inversion_technical_map_a_process_to_component():
    """Replica exactamente la línea de run_lca():
    process_to_component = {v: k for k, v in build_result.technical_map.items()}
    """
    build_result = BuildInventoryResult(
        technical_map={"torre": "steel production, converter, unalloyed", "palas": "GRP production"},
    )
    process_to_component = {v: k for k, v in build_result.technical_map.items()}

    assert process_to_component == {
        "steel production, converter, unalloyed": "torre",
        "GRP production": "palas",
    }


def test_inversion_con_technical_map_vacio_da_dict_vacio():
    build_result = BuildInventoryResult(technical_map={})
    process_to_component = {v: k for k, v in build_result.technical_map.items()}
    assert process_to_component == {}


# ==============================================================================
# Lógica pura: expansión de mapas planos a anidados por proyecto
# (usada en run_montecarlo() y run_sensitivity())
# ==============================================================================

def test_expansion_technical_map_plano_a_anidado_por_proyecto():
    """Replica: technical_maps = {name: build_result.technical_map for name in build_result.project_names}

    BuildInventoryResult solo produce UN technical_map/location_map (planos,
    sin distinguir por proyecto), pero los compositores de MC/Sensibilidad
    esperan un mapa anidado {project_name: {...}}, se asume que todos los
    proyectos comparten el mismo mapeo técnico (consistente con que
    BuildInventoryResult no ofrece uno distinto por proyecto)."""
    build_result = BuildInventoryResult(
        projects=[
            Project(id="p1", name="El Dorado", generation_kwh=1000.0),
            Project(id="p2", name="Cobija Solar", generation_kwh=500.0),
        ],
        technical_map={"torre": "steel production"},
    )

    technical_maps = {name: build_result.technical_map for name in build_result.project_names}

    assert set(technical_maps.keys()) == {"El Dorado", "Cobija Solar"}
    # Ambos proyectos comparten literalmente el mismo dict (mismo id) —
    # confirma que es una referencia compartida, no una copia por proyecto.
    assert technical_maps["El Dorado"] is technical_maps["Cobija Solar"]
    assert technical_maps["El Dorado"] == {"torre": "steel production"}


def test_expansion_con_cero_proyectos_da_dict_vacio():
    build_result = BuildInventoryResult(projects=[], technical_map={"torre": "steel"})
    technical_maps = {name: build_result.technical_map for name in build_result.project_names}
    assert technical_maps == {}


# ==============================================================================
# Errores de fase encadenados correctamente (build -> lca -> mc/sensitivity)
# ==============================================================================

def test_asegurar_lca_run_tambien_exige_build_transitivamente():
    """_assert_lca_run() llama primero a _assert_built(), confirma que el
    mensaje de error es sobre build(), no sobre run_lca(), si ninguna de
    las dos fases corrió."""
    eng = ACVEngine(_FakeAppConfig())
    with pytest.raises(RuntimeError, match=r"engine\.build\(\)"):
        eng.run_montecarlo()
