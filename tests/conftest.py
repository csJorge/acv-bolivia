"""Configuración compartida de pytest para la suite de acv_bolivia.

Dos responsabilidades:

1. Agregar la RAÍZ DEL PROYECTO (padre de `acv_bolivia/`) a `sys.path`, para
   que `import acv_bolivia` / `from acv_bolivia.core.x import Y` funcione.

   IMPORTANTE — esto cambió respecto a versiones anteriores de este archivo:
   desde la migración a imports relativos (ver ANALISIS_ACV_BOLIVIA.md), el
   código YA NO admite el estilo plano antiguo (`sys.path.insert(acv_bolivia/);
   from core.x import y`) — los módulos internos ahora hacen imports
   relativos entre sí (`from ..core.x import y`), que solo resuelven si
   `acv_bolivia` se carga como paquete propio. Ver §5.4 del informe.

2. Definir el marcador `requires_brightway`: los tests así marcados se
   saltan automáticamente si `brightway2`/`bw2data` no están instalados,
   para que el resto de la suite corra en cualquier entorno (incluyendo
   CI de GitHub sin licencia de Ecoinvent).

Estructura de carpetas esperada (ajusta PROJECT_ROOT abajo si la tuya
difiere):

    <raíz del repo>/
    ├── acv_bolivia/        ← el paquete en sí
    ├── tests/
    │   ├── conftest.py     ← este archivo
    │   └── test_*.py
    └── pyproject.toml
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ── 1. sys.path ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACV_BOLIVIA_DIR = PROJECT_ROOT / "acv_bolivia"

if not ACV_BOLIVIA_DIR.is_dir():
    raise RuntimeError(
        f"No se encontró la carpeta 'acv_bolivia/' en {PROJECT_ROOT}. "
        "Este conftest.py asume que tests/ es hermana de acv_bolivia/ "
        "(ambas dentro de la raíz del repo). Si tu estructura es distinta, "
        "ajusta ACV_BOLIVIA_DIR arriba."
    )

# Solo la RAÍZ del proyecto (padre de acv_bolivia/) — NO acv_bolivia/ en sí.
# Con imports relativos, agregar también acv_bolivia/ a sys.path sería
# contraproducente: permitiría que algún módulo se cargue dos veces bajo
# dos identidades distintas (acv_bolivia.core.x y core.x como paquetes
# separados), que es exactamente el riesgo que ya habíamos marcado en el
# informe original antes de migrar a relativos.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 2. Marcador requires_brightway ──────────────────────────────────────
def _brightway_disponible() -> bool:
    return importlib.util.find_spec("bw2data") is not None


BRIGHTWAY_DISPONIBLE = _brightway_disponible()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_brightway: requiere brightway2/bw2data instalados y una "
        "BD Ecoinvent construida; se salta automáticamente si no están disponibles.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if BRIGHTWAY_DISPONIBLE:
        return
    skip_marker = pytest.mark.skip(
        reason="brightway2/bw2data no está instalado en este entorno."
    )
    for item in items:
        if "requires_brightway" in item.keywords:
            item.add_marker(skip_marker)
