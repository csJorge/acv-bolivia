"""Pruebas aisladas de evaluadores LCA mediante un objeto LCA fake."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from acv_bolivia.infrastructure.brightway.evaluators import (
    MatrixLcaEvaluator,
    PivLcaEvaluator,
)


class FakeActivity:
    key = ("local", "project")


class FakeLCA:
    def __init__(self, demand, _method):
        self.demand = demand
        self.technosphere_matrix = sparse.csr_matrix([[100.0]])
        self.score = 100.0
        self.redo_lci_calls = 0
        self.redo_lcia_calls = 0

    def lci(self):
        pass

    def lcia(self):
        self.score = float(self.technosphere_matrix[0, 0])

    def redo_lci(self, demand):
        self.demand = demand
        self.redo_lci_calls += 1

    def redo_lcia(self):
        self.score = float(self.technosphere_matrix[0, 0])
        self.redo_lcia_calls += 1


class FakeBC:
    def LCA(self, demand, method):
        return FakeLCA(demand, method)


def _matrix_evaluator(sample_processor=None):
    activity = FakeActivity()
    return MatrixLcaEvaluator(
        bc_module=FakeBC(),
        bw_activity=activity,
        method_tuple=("method", "impact", "unit"),
        functional_unit=1.0,
        matrix_cell_to_comps={(0, 0): ["tower", "foundation"]},
        nominal_amounts={"tower": 100.0, "foundation": 50.0},
        sample_processor=sample_processor,
    )


def test_matrix_evaluator_agrupa_componentes_en_una_celda():
    evaluator = _matrix_evaluator()

    score = evaluator.evaluate({"tower": 130.0, "foundation": 20.0})

    assert score == pytest.approx(100.0)
    assert evaluator.lca.technosphere_matrix[0, 0] == pytest.approx(100.0)


def test_matrix_evaluator_aplica_sample_processor():
    def processor(samples):
        return {key: values * 2 for key, values in samples.items()}

    evaluator = _matrix_evaluator(processor)

    score = evaluator.evaluate({"tower": 60.0, "foundation": 20.0})

    assert score == pytest.approx(100.0 * (120.0 + 40.0) / 150.0)


def test_sample_processor_con_array_vacio_lanza_error():
    def processor(_samples):
        return {"tower": np.array([])}

    evaluator = _matrix_evaluator(processor)

    with pytest.raises(ValueError, match="array vacío"):
        evaluator.evaluate({"tower": 100.0})


def test_evaluator_rechaza_parametro_no_finito():
    evaluator = _matrix_evaluator()

    with pytest.raises(ValueError, match="debe ser finito"):
        evaluator.evaluate({"tower": np.nan})


def test_cleanup_impide_evaluar_el_evaluador():
    evaluator = _matrix_evaluator()
    evaluator.cleanup()

    with pytest.raises(RuntimeError, match="ya fue limpiado"):
        evaluator.evaluate({"tower": 100.0})


def test_piv_evaluator_usa_nominal_si_falta_parametro():
    evaluator = PivLcaEvaluator(
        h_vectors={"tower": 2.0, "foundation": -1.0},
        nominal_params={"foundation": 50.0},
    )

    assert evaluator.evaluate({"tower": 100.0}) == pytest.approx(150.0)


def test_piv_evaluator_rechaza_componente_sin_valor():
    evaluator = PivLcaEvaluator(h_vectors={"tower": 2.0}, nominal_params={})

    with pytest.raises(KeyError, match="tower"):
        evaluator.evaluate({})


def test_piv_evaluator_aplica_sample_processor():
    def processor(samples):
        return {key: values + 1 for key, values in samples.items()}

    evaluator = PivLcaEvaluator(
        h_vectors={"tower": 2.0},
        nominal_params={"tower": 10.0},
        sample_processor=processor,
    )

    assert evaluator.evaluate({"tower": 4.0}) == pytest.approx(10.0)
