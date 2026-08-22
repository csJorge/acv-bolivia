"""
acv_bolivia.plotting: Tema gráfico y utilidades de visualización compartidas.

Expone un tema matplotlib/seaborn único, limpio y consistente, que se aplica a
todos los plotters del proyecto (LCA/MC, sensibilidad y PIV) para que las
figuras tengan una apariencia uniforme y profesional.

Nota sobre alcance: el tema fija valores por defecto (``rcParams``). Cada
gráfico puede seguir sobreescribiendo estilos de forma ad-hoc — los ajustes
explícitos (``color=``, ``ax.set_facecolor(...)``, leyendas, etc.) tienen
prioridad sobre el tema.

Autor: Jorge Luis Corrales Suarez
"""

from .theme import (
    ACCENT,
    BG_FACE,
    BG_GRID,
    EDGE,
    GRID_ALPHA,
    INK,
    MUTED,
    PALETTE_CATEGORICAL,
    apply_theme,
    categorical_palette,
)

__all__ = [
    "ACCENT",
    "BG_FACE",
    "BG_GRID",
    "EDGE",
    "GRID_ALPHA",
    "INK",
    "MUTED",
    "PALETTE_CATEGORICAL",
    "apply_theme",
    "categorical_palette",
]
