"""Tests para infrastructure.brightway.montecarlo._sampling_rules.PhysicalConstraintRule.

Lógica pura (numpy vectorizado, sin Brightway2). Cubre tanto la clase en
aislamiento (para probar ambas ramas, incluida la de créditos de reciclaje
que la wiring actual no puede disparar — ver test dedicado más abajo) como
la construcción real de nominal_values tal como la arma
infrastructure/composition/run_montecarlo_composer.py, para demostrar con
código, no solo explicarlo, la limitación que encontré al revisar.
"""
from __future__ import annotations

import numpy as np
import pytest

from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.infrastructure.brightway.montecarlo._sampling_rules import PhysicalConstraintRule


# ==============================================================================
# PhysicalConstraintRule en aislamiento — ambas ramas
# ==============================================================================

def test_nominal_positivo_trunca_muestras_negativas_a_cero():
    regla = PhysicalConstraintRule(nominal_values={"torre_acero": 280_000.0})
    muestras = {"torre_acero": np.array([-50.0, 0.0, 100.0, 300_000.0])}

    resultado = regla.apply(muestras)

    np.testing.assert_array_equal(resultado["torre_acero"], [0.0, 0.0, 100.0, 300_000.0])


def test_nominal_negativo_trunca_muestras_positivas_a_cero():
    """La rama de 'crédito de reciclaje' — funciona correctamente EN AISLAMIENTO,
    pasándole un nominal negativo directamente a la clase. El problema no está
    acá (ver test siguiente)."""
    regla = PhysicalConstraintRule(nominal_values={"acero_reciclaje": -417_037.5})
    muestras = {"acero_reciclaje": np.array([-500_000.0, -100.0, 0.0, 50_000.0])}

    resultado = regla.apply(muestras)

    np.testing.assert_array_equal(resultado["acero_reciclaje"], [-500_000.0, -100.0, 0.0, 0.0])


def test_componente_ausente_de_nominal_values_no_se_toca():
    regla = PhysicalConstraintRule(nominal_values={"torre_acero": 280_000.0})
    muestras = {"otro_componente": np.array([-10.0, 10.0])}

    resultado = regla.apply(muestras)

    np.testing.assert_array_equal(resultado["otro_componente"], [-10.0, 10.0])  # sin cambios


def test_preserva_dtype_float64():
    regla = PhysicalConstraintRule(nominal_values={"x": 100.0})
    resultado = regla.apply({"x": np.array([-1.0, 1.0])})
    assert resultado["x"].dtype == np.float64


def test_es_vectorizado_no_iteracion_python_por_muestra():
    """No es un test de velocidad estricto (el harness no tiene tu escala real),
    pero confirma que opera sobre el array completo con una sola llamada
    numpy por componente, no con un bucle Python muestra-por-muestra —
    coherente con que el costo de ESTA regla en sí no explique una MC más
    lenta (son 1-2 llamadas vectorizadas por proyecto, no por iteración;
    ver el hallazgo de rendimiento en la revisión)."""
    n = 200_000
    regla = PhysicalConstraintRule(nominal_values={"x": 1.0})
    muestras = {"x": np.random.default_rng(0).normal(1.0, 5.0, size=n)}

    import time
    t0 = time.perf_counter()
    resultado = regla.apply(muestras)
    elapsed = time.perf_counter() - t0

    assert (resultado["x"] >= 0).all()
    assert elapsed < 0.1  # 200k elementos vectorizados: debe ser prácticamente instantáneo


# ==============================================================================
# HALLAZGO: con la wiring actual, nominal_values NUNCA puede ser negativo
# ==============================================================================

def test_HALLAZGO_wiring_actual_nunca_produce_nominal_negativo():
    """Reproduce EXACTAMENTE cómo infrastructure/composition/run_montecarlo_composer.py
    construye nominal_values (mismo dict comprehension, mismo filtro por
    exchange_type == 'technosphere') a partir de un Project real.

    Quantity.__post_init__ rechaza amount < 0 en su propio constructor, y
    Exchange no tiene ningún campo de signo aparte — así que
    float(exc.quantity.amount) jamás puede ser negativo. Esto significa que
    la rama 'elif nominal < 0' de PhysicalConstraintRule (créditos de
    reciclaje, con soporte correcto según el test de arriba) es HOY código
    muerto por como se arma nominal_values, no por un error en la regla
    misma.

    No es urgente si todavía no tenés componentes de reciclaje en el
    inventario — pero si planeás usarlos, vas a necesitar otra fuente para
    el signo (un flag explícito en Exchange, o inferirlo de
    background_process_name/exchange_type), porque por acá nunca va a llegar."""
    proyecto = Project(id="p1", name="El Dorado", generation_kwh=1000.0, exchanges=[
        Exchange(component_id="torre_acero", quantity=Quantity(280_000.0, "kg"), exchange_type="technosphere"),
        Exchange(component_id="co2_directo", quantity=Quantity(5.0, "kg"), exchange_type="biosphere"),
    ])

    # Réplica exacta de la comprehension en run_montecarlo_composer.py
    nominal_values = {
        exc.component_id: float(exc.quantity.amount)
        for exc in proyecto.exchanges
        if exc.exchange_type == "technosphere"
    }

    assert all(v >= 0 for v in nominal_values.values()), (
        "Si este assert alguna vez falla, felicidades: ahora hay una forma "
        "de que nominal_values traiga valores negativos, y la rama de "
        "créditos de reciclaje de PhysicalConstraintRule dejó de ser "
        "código muerto. Actualizá este test para reflejar el nuevo mecanismo."
    )


# ==============================================================================
# Motivación del fix: por qué las muestras negativas ocurren en primer lugar
# ==============================================================================

def test_motivacion_normal_sin_truncar_SI_puede_generar_muestras_negativas():
    """No es un test de acv_bolivia — es documentación ejecutable de POR QUÉ
    tu fix tiene sentido: con CV alto, una normal centrada en un nominal
    positivo perfectamente razonable igual pone masa de probabilidad del
    lado negativo. Confirma que el problema que atacaste es real."""
    rng = np.random.default_rng(42)
    nominal, cv = 1000.0, 0.4  # CV 40% — no descabellado para inventarios con poca data
    muestras = rng.normal(loc=nominal, scale=nominal * cv, size=100_000)

    fraccion_negativa = (muestras < 0).mean()
    assert fraccion_negativa > 0.001, (
        "Con este nominal/CV se esperaba una fracción no despreciable de "
        "muestras negativas — si esto falla, el ejemplo dejó de ser "
        "representativo del problema real."
    )
