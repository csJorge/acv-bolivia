"""
plotting.theme: Estilo gráfico global, limpio y minimalista.

Tema único de matplotlib/seaborn usado por todos los plotters del proyecto.
Fija los ``rcParams`` por defecto (tipografía, colores, grid, espinas, DPI) y
expone una pequeña paleta semántica reutilizable.

Los plotters llaman a ``apply_theme()`` una vez por sesión; los overrides
explícitos de cada gráfico siguen teniendo prioridad sobre estos valores.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# ── Paleta semántica (clean / minimalista) ─────────────────────────────
# Tinta principal, secundaria y de acento usadas de forma consistente.
INK = "#22303C"  # texto / títulos
MUTED = "#5B6B77"  # texto secundario / ejes
ACCENT = "#2E6DA4"  # color de acento principal
EDGE = "#C9D2D9"  # bordes / espinas
BG_FACE = "#FFFFFF"  # fondo de la figura
BG_GRID = "#F3F5F7"  # fondo de los ejes (muy claro)
GRID_ALPHA = 0.35  # opacidad del grid

# Paletas numéricas (divergente y secuencial) para barras/heatmaps.
PALETTE_SEQ = ["#9EC3E6", "#5B9BD5", "#2E6DA4", "#1F4E79"]
PALETTE_DIVERGENT = ["#A9DFBF", "#FAD7A0", "#F1948A"]

# Paletas categóricas para distinguir MANY categorías/componentes en un mismo
# gráfico (barras apiladas, boxplots por insumo, etc.). Se basan en los mapas
# cualitativos de matplotlib, que maximizan la distancia de matiz entre colores
# (a diferencia de las secuenciales, que solo varían tono/luminosidad).
_PALETTE_TAB20: list[str] = sns.color_palette("tab20").as_hex()
_PALETTE_TAB20B: list[str] = sns.color_palette("tab20b").as_hex()
_PALETTE_TAB20C: list[str] = sns.color_palette("tab20c").as_hex()
PALETTE_CATEGORICAL: list[str] = _PALETTE_TAB20
_CAT_COMBINED: list[str] = _PALETTE_TAB20 + _PALETTE_TAB20B + _PALETTE_TAB20C


def categorical_palette(n_colors: int) -> list[str]:
    """Devuelve ``n_colors`` colores hex distinguibles entre sí.

    Los colores se toman de los mapas cualitativos ``tab20``/``tab20b``/
    ``tab20c`` (60 tonos) y se cíclan si se piden más. A diferencia de una
    paleta secuencial, cada categoría recibe un matiz claramente distinto, lo
    que permite diferenciar muchos componentes en un mismo gráfico.

    Parameters
    ----------
    n_colors : int
        Número de colores distintos requeridos.

    Returns
    -------
    list[str]
        Lista de ``n_colors`` códigos hexadecimales ``#RRGGBB``.
    """
    if n_colors <= 0:
        return []
    reps = (n_colors - 1) // len(_CAT_COMBINED) + 1
    return (_CAT_COMBINED * reps)[:n_colors]


def apply_theme() -> None:
    """Aplica el tema gráfico global (idempotente).

    Configura ``matplotlib`` y ``seaborn`` con un estilo clean/minimalista:
    fondo claro, espinas delgadas, grid sutil solo en un eje y tipografía
    de matplotlib. Es seguro llamarla varias veces.

    Returns
    -------
    None
    """
    sns.set_theme(style="white", palette="deep", font_scale=1.0)

    plt.rcParams.update(
        {
            # Tipografía
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Segoe UI",
                "Arial",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            # Colores y fondo
            "figure.facecolor": BG_FACE,
            "axes.facecolor": BG_FACE,
            "axes.edgecolor": EDGE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            # Grid / espinas
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": EDGE,
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.9,
            # Líneas / parches
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "patch.linewidth": 0.8,
            "patch.edgecolor": BG_FACE,
            "legend.frameon": False,
            # Figuras
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.autolayout": True,
        }
    )


def figsize_h(
    n_rows: int,
    per_row: float = 0.45,
    base: float = 6.0,
) -> tuple[float, float]:
    """Altura de figura razonable según el número de filas (barras horizontales).

    Parameters
    ----------
    n_rows : int
        Número de elementos/categorías a disposición vertical.
    per_row : float, optional
        Altura por fila en pulgadas. Por defecto 0.45.
    base : float, optional
        Altura mínima base. Por defecto 6.0.

    Returns
    -------
    tuple[float, float]
        Dimensiones ``(ancho, alto)`` recomendadas.
    """
    return (11.0, max(base, n_rows * per_row))


__all__ = [
    "ACCENT",
    "BG_FACE",
    "BG_GRID",
    "EDGE",
    "GRID_ALPHA",
    "INK",
    "MUTED",
    "PALETTE_CATEGORICAL",
    "PALETTE_DIVERGENT",
    "PALETTE_SEQ",
    "apply_theme",
    "categorical_palette",
    "figsize_h",
]
