"""Tests para core.services.normalization.

Lógica pura (usa dataclasses.replace sobre LCAResult/HotspotResult), sin
Brightway2. Importante: normalize_by_generation() MUTA las listas que
recibe (reemplaza elementos por índice) — no retorna listas nuevas. Varios
tests dejan eso explícito porque no es obvio solo leyendo la firma.
"""
from __future__ import annotations

import math

import pytest

from acv_bolivia.core.domain.models import HotspotResult, LCAResult
from acv_bolivia.core.services.normalization import normalize_by_generation


def _lca(project_id: str, score: float, score_per_kwh: float | None = 0.0) -> LCAResult:
    return LCAResult(
        project_id=project_id, method_id=("climate change",), method_label="Climate Change",
        score=score, score_per_kwh=score_per_kwh,
    )


def _hotspot(project_id: str, impact: float) -> HotspotResult:
    return HotspotResult(
        project_id=project_id, method_id=("climate change",), component_id="torre",
        background_process_name="steel", impact=impact, impact_per_kwh=0.0, unit="kg CO2 eq",
    )


# ==============================================================================
# LCAResults
# ==============================================================================

def test_normaliza_correctamente_con_generacion_valida():
    resultados = [_lca("El Dorado", score=1000.0)]
    reporte = normalize_by_generation(resultados, [], generation_dict={"El Dorado": 500.0})

    assert resultados[0].score_per_kwh == pytest.approx(2.0)  # 1000/500
    assert reporte.normalized == 1
    assert reporte.errors_count == 0


def test_muta_la_lista_original_en_vez_de_retornar_una_nueva():
    """normalize_by_generation() no es puro respecto a sus argumentos — hace
    dataclasses.replace() y reasigna por índice en la MISMA lista que
    recibió. Si tu código en otro lado asume que la lista original queda
    intacta, este test documenta que no es así."""
    original = _lca("El Dorado", score=100.0, score_per_kwh=0.0)
    lista = [original]

    normalize_by_generation(lista, [], generation_dict={"El Dorado": 100.0})

    assert lista[0] is not original  # se reemplazó el objeto en el índice 0
    assert lista[0].score_per_kwh == pytest.approx(1.0)


def test_proyecto_sin_generacion_declarada_usa_1_0_y_agrega_warning():
    resultados = [_lca("Proyecto Nuevo", score=42.0)]
    reporte = normalize_by_generation(resultados, [], generation_dict={})

    assert resultados[0].score_per_kwh == pytest.approx(42.0)  # dividido por 1.0
    assert len(reporte.warnings) == 1
    assert "Proyecto Nuevo" in reporte.warnings[0]


def test_generacion_cero_asigna_0_0_y_registra_error():
    resultados = [_lca("Proyecto Roto", score=42.0)]
    reporte = normalize_by_generation(resultados, [], generation_dict={"Proyecto Roto": 0.0})

    assert resultados[0].score_per_kwh == 0.0
    assert len(reporte.errors) == 1


def test_errors_count_coincide_con_la_cantidad_de_errores():
    """El contador debe representar exactamente los errores registrados."""
    resultados = [_lca("Proyecto Roto", score=42.0)]
    reporte = normalize_by_generation(resultados, [], generation_dict={"Proyecto Roto": 0.0})

    assert len(reporte.errors) == 1
    assert reporte.errors_count == len(reporte.errors)


def test_overwrite_false_saltea_resultados_que_YA_tienen_score_per_kwh_asignado():
    """La condición real es `result.score_per_kwh is not None` — salta
    cuando el campo YA tiene un valor (cualquiera, incluso 0.0), no cuando
    está vacío. Con overwrite=False, un LCAResult con score_per_kwh=0.5
    (ya "normalizado" antes) se saltea sin tocar."""
    ya_calculado = _lca("El Dorado", score=100.0, score_per_kwh=0.5)
    resultados = [ya_calculado]

    reporte = normalize_by_generation(resultados, [], generation_dict={"El Dorado": 50.0}, overwrite=False)

    assert reporte.skipped == 1
    assert reporte.normalized == 0
    assert resultados[0] is ya_calculado  # no se tocó, sigue siendo el mismo objeto
    assert resultados[0].score_per_kwh == 0.5  # no se recalculó


def test_HALLAZGO_overwrite_false_es_poco_util_en_la_practica():
    """score_per_kwh es un campo `float` obligatorio en LCAResult (no
    Optional) — no hay forma normal de construir un LCAResult 'todavía sin
    normalizar' salvo pasando explícitamente algún valor. En la práctica,
    CUALQUIER LCAResult ya construido (incluso con el placeholder más común,
    0.0) cumple `score_per_kwh is not None`, así que con overwrite=False se
    saltea igual que uno 'ya normalizado' de verdad. Es decir: si en tu
    pipeline construís los LCAResult con score_per_kwh=0.0 como placeholder
    antes de normalizar, overwrite=False los salteará a TODOS, no solo a
    los que ya tienen un valor real distinto de cero.

    La única forma de que overwrite=False SÍ procese un resultado es
    construirlo con score_per_kwh=None explícitamente — lo cual contradice
    el type hint (float, no Optional[float]) aunque Python no lo impida en
    tiempo de ejecución."""
    placeholder_en_cero = _lca("El Dorado", score=100.0, score_per_kwh=0.0)
    resultados = [placeholder_en_cero]

    reporte = normalize_by_generation(resultados, [], generation_dict={"El Dorado": 50.0}, overwrite=False)

    assert reporte.skipped == 1, (
        "Un score_per_kwh=0.0 'placeholder' se trata igual que uno ya "
        "normalizado de verdad — overwrite=False no distingue entre ambos."
    )

    # Único camino que SÍ dispara el recálculo: bypassear el type hint.
    sin_normalizar_de_verdad = _lca("El Dorado", score=100.0, score_per_kwh=None)
    resultados2 = [sin_normalizar_de_verdad]
    reporte2 = normalize_by_generation(resultados2, [], generation_dict={"El Dorado": 50.0}, overwrite=False)
    assert reporte2.normalized == 1
    assert resultados2[0].score_per_kwh == pytest.approx(2.0)


# ==============================================================================
# HotspotResults — comportamiento asimétrico respecto a LCAResults
# ==============================================================================

def test_hotspots_se_normalizan_igual_que_lca_results():
    hotspots = [_hotspot("El Dorado", impact=500.0)]
    normalize_by_generation([], hotspots, generation_dict={"El Dorado": 100.0})
    assert hotspots[0].impact_per_kwh == pytest.approx(5.0)


def test_hotspots_sin_generacion_declarada_agrega_warning():
    """La generación ausente debe advertirse también para hotspots."""
    hotspots = [_hotspot("Proyecto Nuevo", impact=10.0)]
    reporte = normalize_by_generation([], hotspots, generation_dict={})

    assert hotspots[0].impact_per_kwh == pytest.approx(10.0)  # dividido por 1.0 igual
    assert len(reporte.warnings) == 1
    assert "Proyecto Nuevo" in reporte.warnings[0]


def test_hotspots_generacion_invalida_asigna_0_0_y_registra_error():
    """Una generación no válida no debe producir una división silenciosa."""
    hotspots = [_hotspot("Proyecto Roto", impact=10.0)]
    reporte = normalize_by_generation([], hotspots, generation_dict={"Proyecto Roto": 0.0})

    assert hotspots[0].impact_per_kwh == 0.0
    assert len(reporte.errors) == 1


def test_generacion_negativa_o_no_finita_se_trata_como_invalida():
    for generation in [-100.0, math.nan, math.inf]:
        resultados = [_lca("Proyecto Roto", score=42.0)]
        reporte = normalize_by_generation(
            resultados, [], generation_dict={"Proyecto Roto": generation}
        )

        assert resultados[0].score_per_kwh == 0.0
        assert len(reporte.errors) == 1


def test_overwrite_false_cuenta_hotspots_omitidos():
    hotspots = [_hotspot("El Dorado", impact=10.0)]
    reporte = normalize_by_generation(
        [], hotspots, generation_dict={"El Dorado": 100.0}, overwrite=False
    )

    assert reporte.skipped == 1
    assert reporte.normalized == 0
