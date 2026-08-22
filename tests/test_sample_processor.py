"""Pruebas del procesador de muestras vectorizadas."""
from __future__ import annotations

import numpy as np
import pytest
from acv_bolivia.infrastructure.brightway.montecarlo._sample_processor import (
    SampleProcessor,
    create_sample_processor,
)


def test_no_muta_los_arrays_de_entrada():
    processor = SampleProcessor()
    samples = {"tower": np.array([1.0, 2.0])}

    result = processor(samples)

    assert result["tower"] is not samples["tower"]
    np.testing.assert_array_equal(samples["tower"], [1.0, 2.0])


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ({"tower": np.array([[1.0, 2.0]])}, "vector 1D"),
        ({"tower": np.array([1.0]), "foundation": np.array([1.0, 2.0])}, "misma cantidad"),
        ({"tower": np.array([1.0, np.nan])}, "valores finitos"),
    ],
)
def test_rechaza_muestras_con_formato_invalido(samples, message):
    with pytest.raises(ValueError, match=message):
        SampleProcessor()(samples)


def test_factory_construye_reglas_de_dependencia_y_restriccion_fisica():
    processor = create_sample_processor(
        dependency_config={"transport": {"base_comps": ["tower"], "factor": 2.0}},
        nominal_values={"tower": 100.0, "transport": 1.0},
    )

    result = processor({"tower": np.array([-10.0, 20.0])})

    np.testing.assert_array_equal(result["tower"], [0.0, 20.0])
    np.testing.assert_array_equal(result["transport"], [0.02, 0.04])


def test_factory_puede_desactivar_restricciones_fisicas():
    processor = create_sample_processor(
        nominal_values={"tower": 100.0},
        enforce_physical_constraints=False,
    )

    result = processor({"tower": np.array([-10.0, 20.0])})

    np.testing.assert_array_equal(result["tower"], [-10.0, 20.0])
