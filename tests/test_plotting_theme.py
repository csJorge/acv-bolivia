"""
tests.test_plotting_theme: Pruebas del tema gráfico global.

Verifica que el tema limpio/minimalista compartido se aplica sin errores,
que fija los ``rcParams`` esperados y que no impide crear una figura básica
(regresión mínima para los plotters LCA/MC, sensibilidad y PIV).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from acv_bolivia.plotting.theme import (
    BG_FACE,
    INK,
    apply_theme,
    figsize_h,
)


def test_apply_theme_define_rcparams_clave() -> None:
    """apply_theme oculta las espinas superior/derecha y fija el color."""
    apply_theme()
    static_facecolor = plt.rcParams["figure.facecolor"]
    static_textcolor = plt.rcParams["text.color"]
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert static_facecolor.upper() == BG_FACE.upper()
    assert static_textcolor.upper() == INK.upper()


def test_apply_theme_es_idempotente() -> None:
    """apply_theme se puede llamar varias veces sin error."""
    apply_theme()
    apply_theme()


def test_apply_theme_permite_crear_figura() -> None:
    """Tras aplicar el tema se puede crear y cerrar una figura básica."""
    apply_theme()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    plt.close(fig)


def test_figsize_h_calcula_alto_por_filas() -> None:
    """figsize_h produce dimensiones crecientes con el número de filas."""
    ancho, alto = figsize_h(5, per_row=0.6, base=5.0)
    assert ancho == 11.0
    assert alto == max(5.0, 5 * 0.6)
