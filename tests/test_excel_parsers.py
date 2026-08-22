"""Tests para infrastructure.input.parsers y infrastructure.input.helpers.

Lógica pura sobre dicts planos — sin pandas, sin archivos .xlsx reales.
"""
from __future__ import annotations

import math

import pytest

from acv_bolivia.core.domain.uncertainty import DistributionType
from acv_bolivia.infrastructure.input.helpers import build_generation_dict
from acv_bolivia.infrastructure.input.parsers import parse_uncertainty_from_excel_row


# ==============================================================================
# parse_uncertainty_from_excel_row
# ==============================================================================

def test_distribucion_normal_valida():
    fila = {"distribucion": "normal", "parametro_1": None, "parametro_2": "0.15"}
    up = parse_uncertainty_from_excel_row(fila)

    assert up is not None
    assert up.distribution == DistributionType.NORMAL
    assert up.p2 == pytest.approx(0.15)


def test_case_insensitive_y_con_espacios():
    fila = {"distribucion": "  NORMAL  ", "parametro_2": "0.1"}
    up = parse_uncertainty_from_excel_row(fila)
    assert up is not None
    assert up.distribution == DistributionType.NORMAL


@pytest.mark.parametrize("alias", ["uniforme", "uniform", "UNIFORME", " Uniform "])
def test_alias_de_uniforme_mapean_al_mismo_tipo(alias):
    fila = {"distribucion": alias, "parametro_1": "0.8", "parametro_2": "1.2"}
    up = parse_uncertainty_from_excel_row(fila)
    assert up is not None
    assert up.distribution == DistributionType.UNIFORM


def test_distribucion_deterministica_retorna_none():
    """DETERMINISTIC.is_stochastic es False -> la función retorna None
    explícitamente (no crea un UncertaintyParams 'vacío')."""
    fila = {"distribucion": "deterministic"}
    assert parse_uncertainty_from_excel_row(fila) is None


def test_distribucion_vacia_o_ausente_retorna_none():
    assert parse_uncertainty_from_excel_row({}) is None
    assert parse_uncertainty_from_excel_row({"distribucion": ""}) is None


def test_distribucion_no_reconocida_retorna_none_silenciosamente():
    """'gaussiana' no está en el mapa de alias -> None, sin lanzar error.
    Vale la pena saberlo: un typo en el Excel no rompe la carga, simplemente
    ese componente queda sin incertidumbre (determinístico de facto)."""
    fila = {"distribucion": "gaussiana", "parametro_1": "1", "parametro_2": "2"}
    assert parse_uncertainty_from_excel_row(fila) is None


def test_parametro_numerico_corrupto_retorna_none_aunque_la_distribucion_sea_valida():
    """Si 'normal' es válida pero parametro_2 es texto no convertible a
    float, la función descarta TODA la fila (None), no solo el parámetro
    corrupto. Documenta que no hay recuperación parcial."""
    fila = {"distribucion": "normal", "parametro_2": "no-es-un-numero"}
    assert parse_uncertainty_from_excel_row(fila) is None


def test_parametro_ausente_se_interpreta_como_none_no_como_error():
    """Con 'normal' sin parametro_1 en absoluto (ni siquiera la clave existe
    en el dict), p1 queda en None sin lanzar KeyError — solo parametro_2 es
    realmente exigido para NORMAL (ver UncertaintyParams)."""
    fila = {"distribucion": "normal", "parametro_2": "0.2"}  # sin 'parametro_1'
    up = parse_uncertainty_from_excel_row(fila)
    assert up is not None
    assert up.p1 is None
    assert up.p2 == pytest.approx(0.2)


def test_triangular_con_p1_mayor_que_p2_propaga_el_error_de_uncertaintyparams():
    """parse_uncertainty_from_excel_row no valida p1<p2 por sí mismo —
    delega en UncertaintyParams.__post_init__, que si falla, NO está
    capturado por el except (solo ValueError/TypeError/KeyError de la
    conversión numérica) — espera, sí está dentro del mismo tipo de
    excepción (ValueError), así que en la práctica SÍ queda atrapado.
    Este test confirma cuál de los dos comportamientos ocurre realmente."""
    fila = {"distribucion": "triangular", "parametro_1": "1.5", "parametro_2": "1.0"}  # invertidos
    resultado = parse_uncertainty_from_excel_row(fila)
    assert resultado is None  # el ValueError de UncertaintyParams queda atrapado por el except


# ==============================================================================
# build_generation_dict
# ==============================================================================

def test_construye_diccionario_basico():
    records = [
        {"Nombre_Parque": "El Dorado", "Generacion_kWh": 43_800_000},
        {"Nombre_Parque": "Cobija Solar", "Generacion_kWh": 18_250_000},
    ]
    gen = build_generation_dict(records)
    assert gen == {"El Dorado": 43_800_000.0, "Cobija Solar": 18_250_000.0}


def test_fila_con_nombre_vacio_se_omite():
    records = [
        {"Nombre_Parque": "", "Generacion_kWh": 1000},
        {"Nombre_Parque": "   ", "Generacion_kWh": 2000},
        {"Nombre_Parque": "El Dorado", "Generacion_kWh": 3000},
    ]
    gen = build_generation_dict(records)
    assert gen == {"El Dorado": 3000.0}


def test_generacion_no_numerica_se_convierte_en_0_0():
    records = [{"Nombre_Parque": "Proyecto X", "Generacion_kWh": "no-numero"}]
    gen = build_generation_dict(records)
    assert gen["Proyecto X"] == 0.0


def test_generacion_nan_se_convierte_en_0_0():
    """float('nan') no lanza excepción al convertir, así que necesita un
    chequeo explícito aparte (nan != nan) — confirmamos que existe y funciona."""
    records = [{"Nombre_Parque": "Proyecto Y", "Generacion_kWh": float("nan")}]
    gen = build_generation_dict(records)
    assert gen["Proyecto Y"] == 0.0


def test_generacion_ausente_se_convierte_en_0_0():
    records = [{"Nombre_Parque": "Proyecto Z"}]  # sin columna Generacion_kWh
    gen = build_generation_dict(records)
    assert gen["Proyecto Z"] == 0.0


def test_columnas_personalizadas():
    records = [{"parque": "El Dorado", "kwh": 500.0}]
    gen = build_generation_dict(records, name_col="parque", gen_col="kwh")
    assert gen == {"El Dorado": 500.0}


def test_lista_vacia_da_diccionario_vacio():
    assert build_generation_dict([]) == {}
