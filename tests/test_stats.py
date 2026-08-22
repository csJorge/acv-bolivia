"""Tests para application.services.stats.calculate_mc_stats.

Módulo de lógica pura (numpy puro, sin Brightway2) — corre en cualquier
entorno. Ver ANALISIS_ACV_BOLIVIA.md §3.2 sobre el aislamiento de
Brightway2 en el proyecto, que es lo que hace posible testear esto así.
"""
from __future__ import annotations

import numpy as np
import pytest
from acv_bolivia.application.services.stats import calculate_mc_stats
from acv_bolivia.core.domain.contracts import MethodId

METHOD_CLIMA: MethodId = ("ReCiPe 2016", "climate change", "kg CO2 eq")


def test_calculo_basico_media_std_percentiles():
    """Verifica mean/std(ddof=1)/percentiles contra cálculo manual de NumPy."""
    valores = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5])
    scores = {METHOD_CLIMA: {"El Dorado": valores}}
    generation = {"El Dorado": 1.0}  # sin normalización real (gen=1)

    resultado = calculate_mc_stats(scores, generation)

    assert len(resultado) == 1
    stat = resultado[0]
    assert stat.project_id == "El Dorado"
    assert stat.method_id == METHOD_CLIMA
    assert stat.mean == pytest.approx(float(np.mean(valores)))
    assert stat.std == pytest.approx(float(np.std(valores, ddof=1)))
    assert stat.p2_5 == pytest.approx(float(np.percentile(valores, 2.5)))
    assert stat.p97_5 == pytest.approx(float(np.percentile(valores, 97.5)))
    assert stat.min_val == pytest.approx(float(np.min(valores)))
    assert stat.max_val == pytest.approx(float(np.max(valores)))


def test_cv_es_std_sobre_media_en_porcentaje():
    valores = np.array([100.0, 110.0, 90.0, 105.0, 95.0])
    scores = {METHOD_CLIMA: {"El Dorado": valores}}
    resultado = calculate_mc_stats(scores, generation_dict={})

    stat = resultado[0]
    cv_esperado = (np.std(valores, ddof=1) / abs(np.mean(valores))) * 100.0
    assert stat.cv == pytest.approx(cv_esperado)


def test_normalizacion_por_kwh_divide_todas_las_metricas():
    valores = np.array([1000.0, 1100.0, 900.0, 1050.0])
    gen_kwh = 500.0
    scores = {METHOD_CLIMA: {"El Dorado": valores}}
    generation = {"El Dorado": gen_kwh}

    resultado = calculate_mc_stats(scores, generation)
    stat = resultado[0]

    assert stat.mean == pytest.approx(float(np.mean(valores)) / gen_kwh)
    assert stat.std == pytest.approx(float(np.std(valores, ddof=1)) / gen_kwh)
    assert stat.p2_5 == pytest.approx(float(np.percentile(valores, 2.5)) / gen_kwh)


def test_proyecto_sin_generacion_declarada_no_normaliza():
    """Si el project_id no está en generation_dict, se usa 1.0 (no-op).

    Documenta el comportamiento actual de
    `gen = generation_dict.get(project_id, 1.0)` — no es un error, es un
    default silencioso. Vale la pena que sepas que existe: si a un proyecto
    nuevo se le olvida declarar su generación en el Excel, sus estadísticas
    MC no fallan ni avisan — simplemente no se normalizan.
    """
    valores = np.array([10.0, 20.0, 30.0])
    scores = {METHOD_CLIMA: {"Proyecto Nuevo": valores}}

    resultado = calculate_mc_stats(scores, generation_dict={})  # sin 'Proyecto Nuevo'
    stat = resultado[0]

    assert stat.mean == pytest.approx(float(np.mean(valores)))  # sin dividir


def test_array_vacio_se_omite_del_resultado():
    scores = {
        METHOD_CLIMA: {
            "El Dorado": np.array([10.0, 11.0, 12.0]),
            "Proyecto Sin Datos": np.array([]),
        }
    }
    resultado = calculate_mc_stats(scores, generation_dict={})

    project_ids = {s.project_id for s in resultado}
    assert "El Dorado" in project_ids
    assert "Proyecto Sin Datos" not in project_ids
    assert len(resultado) == 1


def test_un_solo_valor_produce_std_y_cv_cero():
    """Una sola observación no debe generar estadísticas NaN."""
    scores = {METHOD_CLIMA: {"El Dorado": np.array([42.0])}}
    resultado = calculate_mc_stats(scores, generation_dict={})

    stat = resultado[0]
    assert stat.std == 0.0
    assert stat.cv == 0.0
    assert stat.mean == pytest.approx(42.0)  # la media sí está bien definida


def test_filtra_scores_no_finitos_antes_de_calcular():
    valores = np.array([10.0, np.nan, np.inf, 20.0])
    resultado = calculate_mc_stats(
        {METHOD_CLIMA: {"El Dorado": valores}}, generation_dict={}
    )

    stat = resultado[0]
    assert stat.mean == pytest.approx(15.0)
    assert stat.min_val == 10.0
    assert stat.max_val == 20.0


def test_omite_series_sin_scores_finitos():
    valores = np.array([np.nan, np.inf, -np.inf])
    resultado = calculate_mc_stats(
        {METHOD_CLIMA: {"Proyecto Inválido": valores}}, generation_dict={}
    )

    assert resultado == []


@pytest.mark.parametrize("generation", [0.0, -1.0, np.nan, np.inf])
def test_omite_proyectos_con_generacion_invalida(generation):
    valores = np.array([10.0, 20.0])
    resultado = calculate_mc_stats(
        {METHOD_CLIMA: {"Proyecto Inválido": valores}},
        generation_dict={"Proyecto Inválido": generation},
    )

    assert resultado == []


def test_multiples_metodos_y_proyectos_generan_una_fila_cada_uno():
    scores = {
        METHOD_CLIMA: {
            "El Dorado": np.array([1.0, 2.0, 3.0]),
            "Gas Natural": np.array([5.0, 6.0, 7.0]),
        },
        ("ReCiPe 2016", "water consumption", "m3"): {
            "El Dorado": np.array([0.1, 0.2, 0.3]),
        },
    }
    resultado = calculate_mc_stats(scores, generation_dict={})

    assert len(resultado) == 3
    combos = {(s.method_id, s.project_id) for s in resultado}
    assert (METHOD_CLIMA, "El Dorado") in combos
    assert (METHOD_CLIMA, "Gas Natural") in combos
    assert (("ReCiPe 2016", "water consumption", "m3"), "El Dorado") in combos
