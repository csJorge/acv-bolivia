"""Tests de regresión para el import circular resuelto en application/contracts.py,
application/use_cases/run_lca.py y application/use_cases/run_montecarlo.py.

Historial: hasta la v2, `from interfaces.colab_session import ColabSession`
(y cualquier otro import fresco de `interfaces`) fallaba con ImportError en
un intérprete recién iniciado, por una dependencia circular real entre
application.contracts y application.use_cases.build_inventory (mediada por
infrastructure.composition). La causa raíz eran imports de
infrastructure.brightway.dto a nivel de módulo que solo se usaban como type
hints (nunca en runtime, con `from __future__ import annotations` activo) —
se resolvió envolviéndolos en `if TYPE_CHECKING:`.

colab_session.py y cli.py (los dos archivos donde más se notaba el síntoma)
ya no existen — se eliminaron junto con notebook_adapter.py, colab.py,
html_templates.py y acv_facade.py en favor de ACVEngine como única fuente
de orquestación. Este archivo prueba la causa raíz directamente
(acv_bolivia.application/infrastructure en frío).

IMPORTANTE — estilo de import cambió con la migración a relativos: desde
entonces, el ÚNICO punto de entrada válido es `import acv_bolivia` (o
`from acv_bolivia.x.y import z`), con la RAÍZ DEL PROYECTO (padre de
acv_bolivia/) en sys.path — NO la carpeta acv_bolivia/ en sí. El estilo
plano antiguo (sys.path a acv_bolivia/ directamente) ya no funciona en
absoluto, para nada — ver el test dedicado al final que lo confirma.

Seguimos usando subprocess: sys.modules cachea el resultado del primer
intento de import dentro del mismo proceso, así que un test in-process
normal solo podría observar UN resultado — subprocess garantiza un
intérprete realmente en frío en cada test, replicando un reinicio de
kernel de Colab o `python3 -c "..."` desde cero.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ACV_BOLIVIA_DIR = Path(__file__).resolve().parent.parent / "acv_bolivia"
PROJECT_ROOT = ACV_BOLIVIA_DIR.parent


def _run_import_en_proceso_fresco(codigo: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_application_sola_en_frio_ya_no_falla():
    """Antes del fix, esto era el reproductor más simple del bug."""
    resultado = _run_import_en_proceso_fresco(
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import acv_bolivia.application; print('OK')"
    )
    assert resultado.returncode == 0, f"stderr:\n{resultado.stderr}"
    assert "OK" in resultado.stdout


def test_infrastructure_sola_en_frio_sigue_funcionando():
    """Ya funcionaba antes del fix (era el workaround) — confirma que
    sigue andando después del fix, sin regresión."""
    resultado = _run_import_en_proceso_fresco(
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import acv_bolivia.infrastructure; print('OK')"
    )
    assert resultado.returncode == 0, f"stderr:\n{resultado.stderr}"
    assert "OK" in resultado.stdout


def test_acv_engine_en_frio_SIN_workaround_funciona():
    """El caso que más importa: el punto de entrada principal (ACVEngine)
    tiene que poder importarse en frío, en cualquier orden, sin necesitar
    'import acv_bolivia.infrastructure' como paso previo manual."""
    resultado = _run_import_en_proceso_fresco(
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "from acv_bolivia.interfaces.acv_engine import ACVEngine; print('OK')"
    )
    assert resultado.returncode == 0, f"stderr:\n{resultado.stderr}"
    assert "OK" in resultado.stdout


def test_paquete_acv_bolivia_completo_en_frio_funciona():
    resultado = _run_import_en_proceso_fresco(
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import acv_bolivia; print(acv_bolivia.ACVEngine.__name__)"
    )
    assert resultado.returncode == 0, f"stderr:\n{resultado.stderr}"
    assert "ACVEngine" in resultado.stdout


def test_orden_de_import_ya_no_importa():
    """Antes del fix, el orden decidía si fallaba o no (application primero
    -> falla; infrastructure primero -> funciona). Confirma que ahora
    ambos órdenes dan exactamente el mismo resultado."""
    ordenes = [
        "import acv_bolivia.application; import acv_bolivia.infrastructure",
        "import acv_bolivia.infrastructure; import acv_bolivia.application",
    ]
    for orden in ordenes:
        resultado = _run_import_en_proceso_fresco(
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            f"{orden}; print('OK')"
        )
        assert resultado.returncode == 0, f"orden '{orden}' falló:\n{resultado.stderr}"


def test_HALLAZGO_estilo_plano_antiguo_ya_no_funciona_en_absoluto():
    """Documenta el cambio de comportamiento de la migración a imports
    relativos: sys.path apuntando a acv_bolivia/ directamente (el patrón
    de tu Guía docx original) ahora falla siempre, para cualquier módulo
    interno — no es un caso borde, es el 100% de los casos. Si estás
    actualizando notebooks viejos, este es exactamente el error que vas a
    ver hasta que cambies al patrón de acv_bolivia.* (ver README/informe)."""
    resultado = _run_import_en_proceso_fresco(
        f"import sys; sys.path.insert(0, {str(ACV_BOLIVIA_DIR)!r}); "
        "from core.domain.models import Project; print('esto no debería imprimirse')"
    )
    assert resultado.returncode != 0
    assert "attempted relative import beyond top-level package" in resultado.stderr
