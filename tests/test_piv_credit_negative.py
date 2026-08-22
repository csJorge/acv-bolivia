"""Tests para infrastructure.brightway.montecarlo._piv_runner — específicamente
_compute_scores(), para documentar por qué PIV puede seguir mostrando valores
negativos aunque las muestras de entrada estén correctamente truncadas a >= 0.

No requieren Brightway2 real: _compute_scores() no toca bd/bc directamente,
así que se puede instanciar PIVMonteCarloRunner con módulos dummy y poblar
_h_vectors a mano (evitando PivVectorCalculator.calculate(), que sí necesita
Ecoinvent real).
"""
from __future__ import annotations

import numpy as np
import pytest

from acv_bolivia.core.domain.models import Project
from acv_bolivia.infrastructure.brightway.montecarlo._piv_runner import PIVMonteCarloRunner

METHOD_CLIMA = ("ReCiPe 2016", "climate change", "kg CO2 eq")


def _runner_minimo(project: Project) -> PIVMonteCarloRunner:
    """Construye un PIVMonteCarloRunner testeable sin Brightway2 real.

    El constructor solo guarda referencias (no llama a bd/bc), así que
    pasar None ahí es seguro siempre que el test solo ejercite
    _compute_scores(), que no los usa.
    """
    return PIVMonteCarloRunner(
        bc_module=None,
        bd_module=None,
        local_db_name="acv_bolivia_local",
        ecoinvent_db_name="ecoinvent-3.10-cutoff",
        methods=[METHOD_CLIMA],
        projects=[project],
        technical_maps={},
        location_maps={},
        sample_processor=lambda s: s,  # no usado por _compute_scores directamente
        seed=42,
    )


def test_muestras_100pct_no_negativas_con_h_negativo_dan_score_total_negativo():
    """Reproduce con el código REAL de _compute_scores() lo que expliqué en
    el chat: score = Σ(clean_sample[comp] * h[comp]). Si el h de un
    componente de crédito domina, el score total puede ser negativo aunque
    TODAS las muestras de entrada sean >= 0 — esto no es un fallo del
    filtro del sampler, es la firma matemática correcta de un crédito.
    """
    proyecto = Project(id="p1", name="El Dorado", generation_kwh=1000.0)
    runner = _runner_minimo(proyecto)

    # h negativo para 'acero_reciclado' (crédito domina en esta categoría) y
    # positivo pero modesto para 'torre_acero' (impacto directo).
    runner._h_vectors["El Dorado"] = {
        METHOD_CLIMA: {"torre_acero": 0.05, "acero_reciclado": -0.6},
    }

    iterations = 1000
    rng = np.random.default_rng(0)
    clean_samples = {
        "torre_acero": np.maximum(rng.normal(280_000, 280_000 * 0.15, iterations), 0.0),
        "acero_reciclado": np.maximum(rng.normal(50_000, 50_000 * 0.20, iterations), 0.0),
    }
    assert (clean_samples["torre_acero"] >= 0).all()
    assert (clean_samples["acero_reciclado"] >= 0).all()

    method_scores, proj_contribs = runner._compute_scores(proyecto, clean_samples, iterations)

    scores = method_scores[METHOD_CLIMA]
    fraccion_negativa = (scores < 0).mean()
    assert fraccion_negativa > 0.5, (
        "Se esperaba que la mayoría de las iteraciones dieran negativo con "
        "este h de crédito dominante — si esto falla, _compute_scores() "
        "cambió de fórmula y hay que revisar el mecanismo descrito arriba."
    )


def test_sin_componente_de_credito_el_score_se_mantiene_no_negativo():
    """Contraste: con SOLO componentes de h positivo, el score total nunca
    baja de cero — confirma que la negatividad viene específicamente del
    signo de h, no de algún resto de muestra negativa que se cuele."""
    proyecto = Project(id="p1", name="El Dorado", generation_kwh=1000.0)
    runner = _runner_minimo(proyecto)
    runner._h_vectors["El Dorado"] = {METHOD_CLIMA: {"torre_acero": 1.9}}

    iterations = 1000
    rng = np.random.default_rng(0)
    clean_samples = {"torre_acero": np.maximum(rng.normal(280_000, 280_000 * 0.15, iterations), 0.0)}

    method_scores, _ = runner._compute_scores(proyecto, clean_samples, iterations)

    assert (method_scores[METHOD_CLIMA] >= 0).all()


def test_contribucion_por_componente_conserva_el_signo_de_h():
    """proj_contribs (usada por show_piv_contributions en la Guía) debe
    mostrar la contribución del componente de crédito como negativa — si
    alguien filtrara esto a >=0 más adelante, se perdería la trazabilidad
    de qué componente es el que aporta el crédito."""
    proyecto = Project(id="p1", name="El Dorado", generation_kwh=1000.0)
    runner = _runner_minimo(proyecto)
    runner._h_vectors["El Dorado"] = {
        METHOD_CLIMA: {"torre_acero": 0.05, "acero_reciclado": -0.6},
    }

    iterations = 500
    rng = np.random.default_rng(1)
    clean_samples = {
        "torre_acero": np.maximum(rng.normal(280_000, 10_000, iterations), 0.0),
        "acero_reciclado": np.maximum(rng.normal(50_000, 5_000, iterations), 0.0),
    }

    _, proj_contribs = runner._compute_scores(proyecto, clean_samples, iterations)

    contrib_reciclado = proj_contribs[METHOD_CLIMA]["acero_reciclado"]
    assert (contrib_reciclado <= 0).all(), (
        "La contribución del componente de crédito debe quedar negativa — "
        "es la única forma de distinguirlo de un componente de impacto "
        "directo al leer piv_contributions."
    )
