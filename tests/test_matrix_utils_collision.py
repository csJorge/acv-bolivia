"""Tests para infrastructure.brightway.montecarlo._matrix_utils.

Estos tests NO requieren Brightway2 ni Ecoinvent instalados: construyen una
matriz tecnosfera sintética con scipy y un objeto `lca` mínimo (duck-typed)
que expone solo los atributos que _matrix_utils.py realmente usa
(technosphere_matrix, tech_params).

Contexto completo del hallazgo en ANALISIS_ACV_BOLIVIA.md §4.1 y §4.2.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from acv_bolivia.infrastructure.brightway.montecarlo._matrix_utils import (
    build_data_positions,
    patch_matrix,
)
from tests.fakes import FakeLCA as _FakeLCA


def _make_tech_params(filas: list[tuple[int, int, float, bool]]) -> np.ndarray:
    """Construye un tech_params sintético con el mismo dtype que usa Brightway2."""
    dtype = np.dtype([("row", "i4"), ("col", "i4"), ("amount", "f8"), ("flip", "?")])
    arr = np.zeros(len(filas), dtype=dtype)
    for i, fila in enumerate(filas):
        arr[i] = fila
    return arr


# ==============================================================================
# Caso base: un único componente, sin colisión — debe funcionar siempre
# ==============================================================================

def test_build_data_positions_caso_simple_sin_colision():
    """Un solo componente, celda única — comportamiento esperado sin ambigüedad.

    Este test también sirve como regresión implícita del bug de NumPy 2.x
    (ANALISIS_ACV_BOLIVIA.md §4.2): si alguna vez corre bajo NumPy >= 2.0
    sin que _matrix_utils.py:106 haya sido corregido, este test FALLARÁ con
    TypeError en vez de con un assert — esa es la señal de que el bug de
    compatibilidad con NumPy 2.x sigue presente.
    """
    csr = sp.coo_matrix(([100.0], ([0], [1])), shape=(2, 2)).tocsr()
    lca = _FakeLCA()
    lca.technosphere_matrix = csr
    lca.tech_params = _make_tech_params([(0, 1, 100.0, False)])

    positions = build_data_positions(lca, relevant_rows=[(0, "torre")])

    assert len(positions) == 1
    data_pos, flip_sign, baseline = positions[0]
    assert data_pos == 0
    assert flip_sign == 1
    assert baseline == pytest.approx(100.0)


def test_patch_matrix_caso_simple_actualiza_correctamente():
    csr = sp.coo_matrix(([100.0], ([0], [1])), shape=(2, 2)).tocsr()
    lca = _FakeLCA()
    lca.technosphere_matrix = csr
    lca.tech_params = _make_tech_params([(0, 1, 100.0, False)])

    relevant_rows = [(0, "torre")]
    positions = build_data_positions(lca, relevant_rows)

    patch_matrix(
        lca, relevant_rows, positions,
        new_values={"torre": 130.0},
        dom_nominals={"torre": 100.0},
    )

    assert lca.technosphere_matrix.data[positions[0][0]] == pytest.approx(130.0)


# ==============================================================================
# Caso de colisión: dos componentes comparten (row, col)
# ==============================================================================
# Escenario real que motivó este test (ver memoria del proyecto): dos
# componentes del inventario (ej. 'torre' y 'cimentacion') que se abastecen
# del MISMO proceso Ecoinvent ('market for steel, GLO') hacia la MISMA
# actividad consumidora colapsan a una única celda física en la matriz CSR,
# porque scipy fusiona automáticamente coordenadas (row,col) duplicadas al
# construir la matriz.

def _escenario_colision() -> tuple[_FakeLCA, list[tuple[int, str]], dict, dict]:
    """torre (100) y cimentacion (50) comparten proveedor Y consumidor."""
    rows, cols, amounts = [0, 0], [1, 1], [100.0, 50.0]
    csr = sp.coo_matrix((amounts, (rows, cols)), shape=(2, 2)).tocsr()
    assert csr.nnz == 1, "precondición: scipy debe fusionar (row,col) duplicados"
    assert csr[0, 1] == pytest.approx(150.0)

    lca = _FakeLCA()
    lca.technosphere_matrix = csr
    lca.tech_params = _make_tech_params([
        (0, 1, 100.0, False),  # torre
        (0, 1, 50.0, False),   # cimentacion — mismo (row,col) que torre
    ])
    relevant_rows = [(0, "torre"), (1, "cimentacion")]
    new_values = {"torre": 130.0, "cimentacion": 20.0}
    dom_nominals = {"torre": 100.0, "cimentacion": 50.0}
    return lca, relevant_rows, new_values, dom_nominals


def test_build_data_positions_detecta_celda_compartida():
    """Ambos componentes deben resolver al MISMO data_pos físico.

    Esto por sí solo no es el bug — es la precondición que lo dispara.
    build_data_positions() no tiene por qué saber que hay colisión; el
    problema está en patch_matrix() (siguiente test).
    """
    lca, relevant_rows, _, _ = _escenario_colision()
    positions = build_data_positions(lca, relevant_rows)

    assert positions[0][0] == positions[1][0], (
        "Se esperaba que 'torre' y 'cimentacion' compartieran data_pos "
        "físico — si esto cambia, revisa si el escenario sintético sigue "
        "siendo representativo del caso real."
    )


def test_patch_matrix_combina_ambos_componentes_en_la_celda_compartida():
    """CORREGIDO (18/08/2026, ver ANALISIS_ACV_BOLIVIA.md §4.1 y §9.2 del chat):
    patch_matrix() ahora agrupa las contribuciones por data_pos ANTES de
    escribir, y suma las de cada grupo — cuando dos componentes comparten
    celda, ambos aportes quedan reflejados (130+20=150), no solo el último
    procesado.

    Este test reemplaza al que antes estaba marcado xfail(strict=True)
    documentando el bug — ahora que está corregido, se convierte en test
    de regresión normal: si alguna vez alguien reintroduce el bug
    (por ejemplo, "simplificando" el código para volver a escribir directo
    sin acumular), este test debe fallar.
    """
    lca, relevant_rows, new_values, dom_nominals = _escenario_colision()
    positions = build_data_positions(lca, relevant_rows)

    patch_matrix(lca, relevant_rows, positions, new_values, dom_nominals)

    valor_final = lca.technosphere_matrix.data[positions[0][0]]
    assert valor_final == pytest.approx(130.0 + 20.0)


def test_patch_matrix_sin_colision_se_comporta_exactamente_igual_que_antes():
    """Confirma que la corrección NO cambió nada para el caso normal (un
    solo componente por celda, sin colisión) — sumar un único valor da el
    mismo resultado que asignarlo directamente. Ya cubierto por
    test_patch_matrix_caso_simple_actualiza_correctamente() más arriba;
    este test es una segunda confirmación explícita, con un valor distinto,
    de que la corrección es transparente para el caso sin colisión."""
    csr = sp.coo_matrix(([500.0], ([2], [3])), shape=(4, 4)).tocsr()
    lca = _FakeLCA()
    lca.technosphere_matrix = csr
    lca.tech_params = _make_tech_params([(2, 3, 500.0, False)])

    relevant_rows = [(0, "aluminio_nacelle")]
    positions = build_data_positions(lca, relevant_rows)
    patch_matrix(
        lca, relevant_rows, positions,
        new_values={"aluminio_nacelle": 620.0},
        dom_nominals={"aluminio_nacelle": 500.0},
    )

    assert lca.technosphere_matrix.data[positions[0][0]] == pytest.approx(620.0)
