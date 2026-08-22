"""Tests para application.dto.run_sensitivity.RunSensitivityResult.

Cubre la corrección del 18/08/2026: reports ahora está correctamente
tipado como List[SensitivityReport] (antes decía List[SensitivityMethodReport],
una clase que nunca se instanciaba en todo el proyecto), y agrega
get_report() para lookup indexado sin recorrer la lista a mano.
"""
from __future__ import annotations

import pytest

from acv_bolivia.application.dto.run_sensitivity import RunSensitivityResult
from acv_bolivia.core.domain.models import SensitivityReport

METHOD_CLIMA = ("ReCiPe 2016", "climate change", "kg CO2 eq")
METHOD_AGUA = ("ReCiPe 2016", "water consumption", "m3")


def _reports() -> list[SensitivityReport]:
    return [
        SensitivityReport(project_id="El Dorado", method_id=METHOD_CLIMA),
        SensitivityReport(project_id="El Dorado", method_id=METHOD_AGUA),
        SensitivityReport(project_id="Gas Natural", method_id=METHOD_CLIMA),
    ]


def test_get_report_encuentra_la_combinacion_correcta():
    result = RunSensitivityResult(reports=_reports())

    encontrado = result.get_report("El Dorado", METHOD_AGUA)

    assert encontrado is not None
    assert encontrado.project_id == "El Dorado"
    assert encontrado.method_id == METHOD_AGUA


def test_get_report_no_confunde_proyectos_con_el_mismo_metodo():
    result = RunSensitivityResult(reports=_reports())

    el_dorado = result.get_report("El Dorado", METHOD_CLIMA)
    gas_natural = result.get_report("Gas Natural", METHOD_CLIMA)

    assert el_dorado is not gas_natural
    assert el_dorado.project_id == "El Dorado"
    assert gas_natural.project_id == "Gas Natural"


def test_get_report_retorna_none_si_no_se_analizo_esa_combinacion():
    result = RunSensitivityResult(reports=_reports())
    assert result.get_report("Cobija Solar", METHOD_CLIMA) is None


def test_get_reports_for_project_sigue_funcionando_con_sensitivityreport_real():
    result = RunSensitivityResult(reports=_reports())
    reportes = result.get_reports_for_project("El Dorado")
    assert len(reportes) == 2
    assert all(r.project_id == "El Dorado" for r in reportes)


def test_n_projects_y_n_methods_analyzed():
    result = RunSensitivityResult(reports=_reports())
    assert result.n_projects == 2  # El Dorado, Gas Natural
    assert result.n_methods_analyzed == 2  # climate change, water consumption


def test_result_vacio_no_rompe_nada():
    result = RunSensitivityResult()
    assert result.n_projects == 0
    assert result.get_report("cualquiera", METHOD_CLIMA) is None
    assert result.get_reports_for_project("cualquiera") == []
