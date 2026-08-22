"""Tests para core.domain.validators.

Módulo de lógica pura de dominio (sin Brightway2, sin I/O) — corre en
cualquier entorno.
"""
from __future__ import annotations

import math

from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.core.domain.validators import (
    ValidationReport,
    validate_method_id,
    validate_project_consistency,
    validate_sensitivity_components,
)

# ==============================================================================
# ValidationReport — Value Object básico
# ==============================================================================

def test_create_sin_errores_es_valido():
    reporte = ValidationReport.create()
    assert reporte.is_valid is True
    assert reporte.errors == ()
    assert reporte.warnings == ()


def test_create_con_errores_no_es_valido():
    reporte = ValidationReport.create(errors=["algo salió mal"])
    assert reporte.is_valid is False
    assert reporte.errors == ("algo salió mal",)


def test_create_con_solo_warnings_sigue_siendo_valido():
    """is_valid depende solo de errors, no de warnings — por diseño."""
    reporte = ValidationReport.create(warnings=["revisar esto"])
    assert reporte.is_valid is True
    assert reporte.warnings == ("revisar esto",)


def test_merge_NO_muta_el_original_retorna_uno_nuevo():
    """Documenta el hallazgo de ANALISIS_ACV_BOLIVIA.md §4.3.

    ValidationReport es frozen=True (inmutable) — merge() DEBE retornar un
    objeto nuevo, no puede mutar self. El docstring actual ("fusiona...en
    este") y el tipo de retorno declarado (-> None) sugieren lo contrario;
    este test deja registrado el comportamiento real para que quede
    protegido si alguien "corrige" merge() intentando hacerlo mutar in-place
    (cosa que además rompería frozen=True y lanzaría FrozenInstanceError).
    """
    reporte_a = ValidationReport.create(errors=["error A"])
    reporte_b = ValidationReport.create(errors=["error B"], warnings=["aviso B"])

    resultado = reporte_a.merge(reporte_b)

    # El original NO cambia (inmutable, tal como frozen=True exige):
    assert reporte_a.errors == ("error A",)
    assert reporte_a.warnings == ()

    # El valor de retorno es el que realmente trae la fusión — si se
    # llama a merge() sin capturar el retorno, la fusión se pierde:
    assert resultado.errors == ("error A", "error B")
    assert resultado.warnings == ("aviso B",)
    assert resultado.is_valid is False


# ==============================================================================
# validate_method_id
# ==============================================================================

def test_validate_method_id_tupla_no_vacia_es_valida():
    reporte = validate_method_id(("ReCiPe 2016", "climate change", "kg CO2 eq"))
    assert reporte.is_valid is True


def test_validate_method_id_tupla_vacia_es_invalida():
    reporte = validate_method_id(())
    assert reporte.is_valid is False
    assert len(reporte.errors) == 1


def test_validate_method_id_no_tupla_es_invalida():
    reporte = validate_method_id(["no", "es", "tupla"])
    assert reporte.is_valid is False


def test_validate_method_id_rechaza_partes_vacias_o_no_textuales():
    for method_id in [("", "climate change"), ("ReCiPe 2016", 42)]:
        reporte = validate_method_id(method_id)
        assert reporte.is_valid is False


# ==============================================================================
# validate_sensitivity_components
# ==============================================================================

def test_validate_sensitivity_components_todos_validos():
    reporte = validate_sensitivity_components(
        components=["torre", "palas"],
        valid_components={"torre", "palas", "nacelle", "cimentacion"},
    )
    assert reporte.is_valid is True
    assert reporte.errors == ()


def test_validate_sensitivity_components_detecta_componente_inexistente():
    reporte = validate_sensitivity_components(
        components=["torre", "componente_que_no_existe"],
        valid_components={"torre", "palas"},
    )
    assert reporte.is_valid is False
    assert len(reporte.errors) == 1
    assert "componente_que_no_existe" in reporte.errors[0]


# ==============================================================================
# validate_project_consistency
# ==============================================================================

def _make_exchange(component_id: str, amount: float, exchange_type: str = "technosphere") -> Exchange:
    return Exchange(
        component_id=component_id,
        quantity=Quantity(amount=amount, unit="kg"),
        exchange_type=exchange_type,
    )


def test_validate_project_consistency_proyecto_valido_sin_observaciones():
    proyecto = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=43_800_000.0,
        exchanges=[_make_exchange("torre", 120_000.0)],
    )
    reporte = validate_project_consistency(proyecto)
    assert reporte.is_valid is True
    assert reporte.warnings == ()


def test_validate_project_consistency_generacion_cero_es_error():
    proyecto = Project(
        id="p1", name="El Dorado", generation_kwh=0.0,
        exchanges=[_make_exchange("torre", 120_000.0)],
    )
    reporte = validate_project_consistency(proyecto)
    assert reporte.is_valid is False
    assert any("generación_kwh" in e for e in reporte.errors)


def test_validate_project_consistency_generacion_no_finita_es_error():
    for generation_kwh in [math.nan, math.inf, -math.inf]:
        proyecto = Project(
            id="p1",
            name="El Dorado",
            generation_kwh=generation_kwh,
            exchanges=[_make_exchange("torre", 120_000.0)],
        )
        reporte = validate_project_consistency(proyecto)
        assert reporte.is_valid is False


def test_validate_project_consistency_sin_exchanges_tecnosfera_es_solo_warning():
    """Sin exchanges de tecnosfera es un AVISO, no bloquea is_valid."""
    proyecto = Project(id="p1", name="El Dorado", generation_kwh=1000.0, exchanges=[])
    reporte = validate_project_consistency(proyecto)
    assert reporte.is_valid is True  # sigue siendo válido — es warning, no error
    assert len(reporte.warnings) == 1
