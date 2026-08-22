"""Tests del patrón de truncamiento post-hoc de scores de BW MC.

ADVERTENCIA DE PROVENIENCIA: a diferencia de los demás archivos de esta
suite, este test NO corre contra un archivo real de tu ZIP — confirmé con
grep que `enforce_physical_constraints`/`np.maximum(scores`/`negative_count`
no aparecen en application/use_cases/run_montecarlo.py ni en
infrastructure/composition/run_montecarlo_composer.py del último ZIP que
subiste. El snippet que compartiste por chat parece existir en tu entorno
local pero todavía no llegó a un ZIP que me hayas pasado.

Este archivo documenta el PATRÓN tal como lo pegaste (aislado, como función
pura) para dejar registrada la discusión metodológica — pero en cuanto me
compartas el archivo real actualizado, hay que reemplazar esto por un test
que importe y ejercite tu función real, no esta reconstrucción.
"""
from __future__ import annotations

import numpy as np
import pytest


def _truncar_scores_post_hoc(scores: np.ndarray, enforce: bool = True) -> tuple[np.ndarray, int]:
    """Reconstrucción exacta del snippet que compartiste — MISMA lógica,
    aislada como función para poder testearla. Reemplazar por un import
    real en cuanto tengas el archivo actualizado."""
    if not enforce:
        return scores, 0
    negative_count = int(np.sum(scores < 0))
    return np.maximum(scores, 0.0), negative_count


def test_trunca_negativos_a_cero_y_cuenta_cuantos_afecto():
    scores = np.array([10.0, -5.0, 3.0, -0.2, 8.0])
    truncados, n_negativos = _truncar_scores_post_hoc(scores)

    np.testing.assert_array_equal(truncados, [10.0, 0.0, 3.0, 0.0, 8.0])
    assert n_negativos == 2


def test_con_enforce_false_no_toca_nada():
    scores = np.array([10.0, -5.0, 3.0])
    truncados, n_negativos = _truncar_scores_post_hoc(scores, enforce=False)

    np.testing.assert_array_equal(truncados, scores)  # sin cambios
    assert n_negativos == 0


def test_HALLAZGO_crea_masa_de_probabilidad_artificial_en_cero():
    """Este es el punto central de la crítica metodológica: truncar el
    score AGREGADO (no un flujo individual) convierte cualquier iteración
    negativa en un valor idéntico (0.0) — a diferencia de la incertidumbre
    real, que es continua. Con una fracción no trivial de iteraciones
    negativas, esto genera una moda artificial exactamente en 0 que no
    representa ninguna combinación física real de emisiones, solo el techo
    del truncamiento."""
    rng = np.random.default_rng(0)
    # Simula una distribución de scores BW MC con cola izquierda ancha
    # (como 'climate change' o 'human toxicity: non-carcinogenic' en tu tabla)
    scores = rng.normal(loc=0.0075, scale=0.0053, size=1000)  # calibrado como tu ejemplo real

    truncados, n_negativos = _truncar_scores_post_hoc(scores)

    fraccion_en_cero_exacto = (truncados == 0.0).mean()
    assert n_negativos > 0
    assert fraccion_en_cero_exacto == pytest.approx(n_negativos / len(scores))
    assert fraccion_en_cero_exacto > 0.05, (
        "Con esta calibración se espera una moda notable en cero — "
        "confirma que el truncamiento post-hoc no es un ajuste marginal, "
        "afecta una fracción visible de la distribución reportada."
    )


def test_HALLAZGO_infla_el_cv_reportado_respecto_a_la_distribucion_sin_truncar():
    """Cuantifica el efecto sobre el CV que mencioné: comparar el CV con y
    sin truncamiento post-hoc, para la MISMA muestra subyacente."""
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=0.0075, scale=0.0053, size=1000)

    truncados, _ = _truncar_scores_post_hoc(scores, enforce=True)
    sin_truncar = scores  # equivalente a enforce=False

    cv_truncado = np.std(truncados, ddof=1) / abs(np.mean(truncados)) * 100
    cv_sin_truncar = np.std(sin_truncar, ddof=1) / abs(np.mean(sin_truncar)) * 100

    # No afirmamos una dirección universal (depende de la calibración),
    # solo que el truncamiento CAMBIA el CV reportado de forma no trivial
    # respecto a los datos crudos — por eso conviene reportar ambos.
    assert cv_truncado != pytest.approx(cv_sin_truncar, rel=0.01)
