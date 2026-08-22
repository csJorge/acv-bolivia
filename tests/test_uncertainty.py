"""Tests para core.domain.uncertainty.

Módulo de lógica pura (matemática, sin Brightway2) pero de alto impacto:
UncertaintyParams.get_statistical_properties() es la función que convierte
los coeficientes relativos del Excel (p1, p2) a los parámetros absolutos
que Brightway2 usa para muestrear en Monte Carlo. Un error acá se propaga
silenciosamente a todos tus resultados estocásticos.
"""
from __future__ import annotations

import math

import pytest

from acv_bolivia.core.domain.uncertainty import DistributionType, UncertaintyParams


# ==============================================================================
# DistributionType
# ==============================================================================

def test_deterministic_no_es_estocastica():
    assert DistributionType.DETERMINISTIC.is_stochastic is False


@pytest.mark.parametrize(
    "dist", [
        DistributionType.NORMAL, DistributionType.LOGNORMAL,
        DistributionType.UNIFORM, DistributionType.TRIANGULAR,
        DistributionType.WEIBULL,
    ],
)
def test_todas_las_demas_son_estocasticas(dist):
    assert dist.is_stochastic is True


def test_stats_arrays_id_coincide_con_el_estandar_brightway2():
    """Estos IDs no son arbitrarios — son el estándar stats_arrays que
    Brightway2 espera en el campo 'uncertainty type'. Si alguno cambia,
    Brightway2 va a interpretar mal la distribución sin avisar."""
    assert DistributionType.DETERMINISTIC.stats_arrays_id == 0
    assert DistributionType.LOGNORMAL.stats_arrays_id == 2
    assert DistributionType.NORMAL.stats_arrays_id == 3
    assert DistributionType.UNIFORM.stats_arrays_id == 4
    assert DistributionType.TRIANGULAR.stats_arrays_id == 5
    assert DistributionType.WEIBULL.stats_arrays_id == 8


# ==============================================================================
# UncertaintyParams — validación en construcción
# ==============================================================================

def test_deterministic_no_requiere_p1_ni_p2():
    up = UncertaintyParams()  # todos los defaults
    assert up.distribution == DistributionType.DETERMINISTIC
    assert up.is_stochastic is False


def test_normal_sin_p2_lanza_error():
    with pytest.raises(ValueError, match="p2"):
        UncertaintyParams(distribution=DistributionType.NORMAL, p1=1.0, p2=None)


def test_lognormal_sin_p2_lanza_error():
    with pytest.raises(ValueError, match="p2"):
        UncertaintyParams(distribution=DistributionType.LOGNORMAL, p2=None)


def test_normal_con_p2_es_valida_sin_necesitar_p1():
    """NORMAL no exige p1 explícitamente (a diferencia de triangular/uniforme) —
    documenta que la validación es asimétrica entre distribuciones."""
    up = UncertaintyParams(distribution=DistributionType.NORMAL, p2=0.1)
    assert up.p1 is None
    assert up.p2 == 0.1


@pytest.mark.parametrize("dist", [DistributionType.TRIANGULAR, DistributionType.UNIFORM])
def test_triangular_y_uniforme_requieren_p1_y_p2(dist):
    with pytest.raises(ValueError, match="p1 y p2"):
        UncertaintyParams(distribution=dist, p1=0.9, p2=None)
    with pytest.raises(ValueError, match="p1 y p2"):
        UncertaintyParams(distribution=dist, p1=None, p2=1.1)


@pytest.mark.parametrize("dist", [DistributionType.TRIANGULAR, DistributionType.UNIFORM])
def test_triangular_y_uniforme_exigen_p1_menor_que_p2(dist):
    with pytest.raises(ValueError, match="debe ser <"):
        UncertaintyParams(distribution=dist, p1=1.2, p2=0.8)  # invertidos
    with pytest.raises(ValueError, match="debe ser <"):
        UncertaintyParams(distribution=dist, p1=1.0, p2=1.0)  # iguales, tampoco vale


def test_weibull_requiere_p1_y_p2():
    with pytest.raises(ValueError, match="p1 y p2"):
        UncertaintyParams(distribution=DistributionType.WEIBULL, p1=None, p2=2.0)


# ==============================================================================
# get_statistical_properties — la conversión relativo -> absoluto
# ==============================================================================

def test_nominal_amount_cero_o_negativo_siempre_lanza_error():
    """Incluso para DETERMINISTIC — el chequeo va antes del switch por tipo."""
    up = UncertaintyParams()
    with pytest.raises(ValueError, match="debe ser > 0"):
        up.get_statistical_properties(nominal_amount=0.0)
    with pytest.raises(ValueError, match="debe ser > 0"):
        up.get_statistical_properties(nominal_amount=-5.0)


def test_normal_escala_el_scale_por_el_monto_nominal_y_preserva_loc():
    up = UncertaintyParams(distribution=DistributionType.NORMAL, p2=0.1)  # CV 10%
    props = up.get_statistical_properties(nominal_amount=200.0)

    assert props["type_id"] == 3
    assert props["amount"] == pytest.approx(200.0)
    assert props["loc"] == pytest.approx(200.0)          # la media es el propio monto
    assert props["scale"] == pytest.approx(200.0 * 0.1)  # 10% de 200 = 20
    assert props["minimum"] is None and props["maximum"] is None


def test_lognormal_usa_log_del_monto_como_loc_y_p2_SIN_escalar_como_scale():
    """A diferencia de NORMAL, el scale de LOGNORMAL NO se multiplica por el
    monto nominal — es la sigma del espacio logarítmico, adimensional por
    convención estadística. Vale la pena un test explícito porque es fácil
    'corregir' esto por error creyendo que debería escalar como NORMAL."""
    up = UncertaintyParams(distribution=DistributionType.LOGNORMAL, p2=0.25)
    props = up.get_statistical_properties(nominal_amount=50.0)

    assert props["loc"] == pytest.approx(math.log(50.0))
    assert props["scale"] == pytest.approx(0.25)  # NO 0.25*50


def test_uniforme_calcula_minimo_y_maximo_desde_p1_p2():
    up = UncertaintyParams(distribution=DistributionType.UNIFORM, p1=0.8, p2=1.2)
    props = up.get_statistical_properties(nominal_amount=100.0)

    assert props["minimum"] == pytest.approx(80.0)
    assert props["maximum"] == pytest.approx(120.0)
    assert props["scale"] is None and props["shape"] is None


def test_triangular_valido_cuando_el_nominal_cae_dentro_del_rango():
    up = UncertaintyParams(distribution=DistributionType.TRIANGULAR, p1=0.9, p2=1.1)
    props = up.get_statistical_properties(nominal_amount=100.0)  # 90 <= 100 <= 110

    assert props["minimum"] == pytest.approx(90.0)
    assert props["maximum"] == pytest.approx(110.0)


def test_triangular_INCONSISTENTE_cuando_el_nominal_queda_fuera_del_rango():
    """p1=1.5, p2=2.0 (ambos > 1) implica minimo=1.5*nominal > nominal — el
    propio 'valor más probable' del triángulo (el nominal) queda por fuera
    de su rango [minimo, maximo]. get_statistical_properties() lo detecta
    y lanza, en vez de generar silenciosamente una distribución sin sentido."""
    up = UncertaintyParams(distribution=DistributionType.TRIANGULAR, p1=1.5, p2=2.0)
    with pytest.raises(ValueError, match="Inconsistencia Triangular"):
        up.get_statistical_properties(nominal_amount=100.0)


def test_uniforme_NO_valida_consistencia_con_el_nominal_a_diferencia_de_triangular():
    """Mismo p1/p2 'fuera de rango' que en el test anterior, pero con UNIFORM
    en vez de TRIANGULAR — acá NO hay chequeo de consistencia, se calcula
    igual. Documenta la asimetría entre ambas ramas del código."""
    up = UncertaintyParams(distribution=DistributionType.UNIFORM, p1=1.5, p2=2.0)
    props = up.get_statistical_properties(nominal_amount=100.0)  # no lanza
    assert props["minimum"] == pytest.approx(150.0)
    assert props["maximum"] == pytest.approx(200.0)


def test_weibull_shape_es_p1_sin_escalar_y_scale_es_p2_escalado():
    up = UncertaintyParams(distribution=DistributionType.WEIBULL, p1=2.5, p2=1.1)
    props = up.get_statistical_properties(nominal_amount=40.0)

    assert props["shape"] == pytest.approx(2.5)          # p1 tal cual
    assert props["scale"] == pytest.approx(40.0 * 1.1)   # p2 * nominal
