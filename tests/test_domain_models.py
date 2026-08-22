"""Tests para core.domain.models.

Lógica pura de dominio. Para SensitivityReport.top_components() usamos
objetos "fake" mínimos en vez de la clase real AnalyzerResult (que vive en
analysis/ y no hace falta importar aquí) — mismo principio que
tests/fakes.py: un objeto simple con los atributos que el código realmente
lee (.method_name, .error_message, .raw_results, .scores[].component/.score),
para poder probar el algoritmo de agregación de rankings con datos reales.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from acv_bolivia.core.domain.models import (
    Exchange,
    HotspotResult,
    LCAResult,
    MonteCarloResult,
    Project,
    Quantity,
    SensitivityReport,
)

# ==============================================================================
# Quantity
# ==============================================================================


def test_quantity_negativa_lanza_error():
    with pytest.raises(ValueError):
        Quantity(amount=-1.0, unit="kg")


def test_quantity_cero_es_valida():
    q = Quantity(amount=0.0, unit="kg")
    assert q.amount == 0.0


# ==============================================================================
# Exchange
# ==============================================================================


def test_exchange_is_uncertain_refleja_presencia_de_uncertainty():
    exc_cierto = Exchange(
        component_id="torre",
        quantity=Quantity(100.0, "kg"),
        exchange_type="technosphere",
    )
    assert exc_cierto.is_uncertain is False

    from acv_bolivia.core.domain.uncertainty import DistributionType, UncertaintyParams

    exc_incierto = Exchange(
        component_id="torre",
        quantity=Quantity(100.0, "kg"),
        exchange_type="technosphere",
        uncertainty=UncertaintyParams(distribution=DistributionType.NORMAL, p2=0.1),
    )
    assert exc_incierto.is_uncertain is True


# ==============================================================================
# Project
# ==============================================================================


def _exchange(
    component_id: str, amount: float, exchange_type: str = "technosphere"
) -> Exchange:
    return Exchange(
        component_id=component_id,
        quantity=Quantity(amount, "kg"),
        exchange_type=exchange_type,
    )


def test_project_add_exchange_valido_lo_agrega():
    p = Project(id="p1", name="El Dorado", generation_kwh=1000.0)
    p.add_exchange(_exchange("torre", 50.0))
    assert len(p.exchanges) == 1


def test_project_add_exchange_con_amount_cero_lanza_error():
    """Nota: Quantity ya rechaza amount<0 en su propio constructor, así que
    para cuando llegás a add_exchange() el único caso que le queda por
    filtrar es amount==0 exactamente — 'amount <= 0.0' es parcialmente
    redundante con la validación de Quantity, no un bug, pero vale la pena
    dejarlo documentado con un test explícito."""
    p = Project(id="p1", name="El Dorado", generation_kwh=1000.0)
    with pytest.raises(ValueError):
        p.add_exchange(_exchange("torre", 0.0))


def test_project_get_technosphere_exchanges_filtra_por_tipo():
    p = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=1000.0,
        exchanges=[
            _exchange("torre", 50.0, "technosphere"),
            _exchange("co2", 10.0, "biosphere"),
            _exchange("produccion", 1.0, "production"),
        ],
    )
    tecno = p.get_technosphere_exchanges()
    assert len(tecno) == 1
    assert tecno[0].component_id == "torre"


def test_project_has_uncertainty_true_si_algun_exchange_es_incierto():
    from acv_bolivia.core.domain.uncertainty import DistributionType, UncertaintyParams

    incierto = Exchange(
        component_id="torre",
        quantity=Quantity(50.0, "kg"),
        exchange_type="technosphere",
        uncertainty=UncertaintyParams(distribution=DistributionType.NORMAL, p2=0.1),
    )
    p = Project(id="p1", name="El Dorado", generation_kwh=1000.0, exchanges=[incierto])
    assert p.has_uncertainty is True


def test_project_has_uncertainty_false_sin_exchanges_inciertos():
    p = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=1000.0,
        exchanges=[_exchange("torre", 50.0)],
    )
    assert p.has_uncertainty is False


# ==============================================================================
# LCAResult / HotspotResult / MonteCarloResult
# ==============================================================================


def test_lca_result_is_normalized():
    con_kwh = LCAResult(
        project_id="p1",
        method_id=("m",),
        method_label="M",
        score=10.0,
        score_per_kwh=0.5,
    )
    assert con_kwh.is_normalized is True

    sin_kwh = LCAResult(
        project_id="p1",
        method_id=("m",),
        method_label="M",
        score=10.0,
        score_per_kwh=0.0,
    )
    assert sin_kwh.is_normalized is False


def test_hotspot_result_is_dominant_es_umbral_absoluto_no_porcentaje():
    """El docstring de is_dominant dice '>5% del impacto total', pero la
    implementación real solo chequea abs(impact) > 1e-6 : no tiene forma de
    saber el total sin que se lo pasen. Documentamos el comportamiento REAL,
    no el que describe el docstring, para que quede claro que hoy 'is_dominant'
    es en la práctica 'no es prácticamente cero', no 'supera el 5%'."""
    chico_pero_no_cero = HotspotResult(
        project_id="p1",
        method_id=("m",),
        component_id="tornillo",
        background_process_name="steel",
        impact=1e-5,
        impact_per_kwh=1e-8,
        unit="kg CO2 eq",
    )
    assert (
        chico_pero_no_cero.is_dominant is True
    )  # 1e-5 > 1e-6, aunque sea insignificante en % real

    practicamente_cero = HotspotResult(
        project_id="p1",
        method_id=("m",),
        component_id="tornillo",
        background_process_name="steel",
        impact=1e-7,
        impact_per_kwh=1e-10,
        unit="kg CO2 eq",
    )
    assert practicamente_cero.is_dominant is False


def test_montecarlo_result_n_iterations_y_mean_score():
    mc = MonteCarloResult(project_id="p1", method_id=("m",), scores=[10.0, 20.0, 30.0])
    assert mc.n_iterations == 3
    assert mc.mean_score == pytest.approx(20.0)


def test_montecarlo_result_mean_score_con_lista_vacia_es_cero_no_error():
    mc = MonteCarloResult(project_id="p1", method_id=("m",), scores=[])
    assert mc.n_iterations == 0
    assert mc.mean_score == 0.0  # división por cero evitada explícitamente


# ==============================================================================
# SensitivityReport — usa fakes locales en vez de la clase real AnalyzerResult
# ==============================================================================


@dataclass
class _FakeScoreItem:
    """Fake de un ítem de ranking individual (lo que SensitivityReport.top_components()
    espera encontrar en result.scores: algo con .component y .score)."""

    component: str
    score: float


@dataclass
class _FakeAnalyzerResult:
    """Fake de AnalyzerResult — solo los atributos que SensitivityReport
    realmente lee (confirmado leyendo core/domain/models.py), no la clase
    real completa."""

    method_name: str
    scores: list[_FakeScoreItem]
    raw_results: list | None = None
    error_message: str | None = None


def test_sensitivity_report_add_result_exitoso_lo_registra():
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    resultado = _FakeAnalyzerResult(
        method_name="Delta LCA", scores=[_FakeScoreItem("torre", 0.8)]
    )

    report.add_result(resultado)

    assert "Delta LCA" in report.results
    assert report.methods_run == ["Delta LCA"]
    assert report.has_errors is False
    assert report.methods_executed_count == 1


def test_sensitivity_report_add_result_con_error_lo_registra_como_error_no_resultado():
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    fallido = _FakeAnalyzerResult(
        method_name="Sobol", scores=[], error_message="no convergió"
    )

    report.add_result(fallido)

    assert "Sobol" not in report.results
    assert report.methods_run == []
    assert report.has_errors is True
    assert "Sobol: no convergió" in report.errors[0]


def test_sensitivity_report_get_raw_devuelve_lista_vacia_si_no_existe():
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    assert report.get_raw("metodo_inexistente") == []


def test_sensitivity_report_top_components_agrega_por_ranking_ponderado():
    """torre queda primera en Delta (peso 2) y primera en PRCC (peso 2);
    palas segunda en ambos (peso 1) -> torre debe encabezar el consenso."""
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    report.add_result(
        _FakeAnalyzerResult(
            method_name="Delta LCA",
            scores=[_FakeScoreItem("torre", 0.9), _FakeScoreItem("palas", 0.5)],
        )
    )
    report.add_result(
        _FakeAnalyzerResult(
            method_name="PRCC",
            scores=[_FakeScoreItem("torre", 0.7), _FakeScoreItem("palas", 0.3)],
        )
    )

    top = report.top_components(n=2)
    assert top == ["torre", "palas"]


def test_sensitivity_report_top_components_usa_valor_absoluto_del_score():
    """Un score fuertemente negativo (-0.95) debe pesar tanto como uno
    positivo grande — top_components ordena por abs(score), no por score."""
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    report.add_result(
        _FakeAnalyzerResult(
            method_name="PRCC",
            scores=[_FakeScoreItem("torre", -0.95), _FakeScoreItem("palas", 0.10)],
        )
    )

    top = report.top_components(n=1)
    assert top == ["torre"]


def test_sensitivity_report_top_components_vacio_sin_resultados():
    report = SensitivityReport(project_id="p1", method_id=("climate change",))
    assert report.top_components() == []
