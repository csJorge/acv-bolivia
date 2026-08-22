"""
infrastructure.brightway.ei_name_resolver: resolución de procesos Ecoinvent.

Responsabilidad única: resolver qué actividad de Ecoinvent corresponde a cada
componente del mapeo técnico, priorizando la clave canónica de Brightway.

Estrategia de resolución (en orden de prioridad):
1. Por ``code`` si está disponible en el mapeo: clave única e inmutable de
   Brightway (``Activity.key`` = ``(db, code)``). Exacta y sin ambigüedad.
2. Por ``(nombre, ubicación, unidad)`` si se especifica la unidad: descarta
   procesos homónimos con unidades distintas (p. ej. ``kilogram`` vs
   ``cubic meter``).
3. Por ``(nombre, ubicación)`` a secas: desempate determinista y advertencia,
   cuando el inventario no declara ni ``code`` ni ``unidad``.

Consumidores:
- ``ActivityRepository`` (construcción del inventario local y LCA determinístico).
- ``PivVectorCalculator`` (vectores h del método PIV en Monte Carlo).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

from ...infrastructure.brightway.constants import DEFAULT_LOCATION

logger = logging.getLogger(__name__)


class EcoinventNameResolver:
    """Resuelve procesos de Ecoinvent por ``code`` o por ``(nombre, ubicación)``.

    Ecoinvent puede contener varios procesos homónimos (mismo nombre y
    ubicación) con unidades distintas. La clave ``(name, location)`` de un
    diccionario simple colisionaría en ese caso y el resultado dependería del
    orden de iteración de la base de datos, dando builds no reproducibles.
    Esta clase indexa todos los candidatos y selecciona de forma determinista,
    usando el ``code`` o la ``unidad`` cuando están disponibles.
    """

    def __init__(self, ei_db: Any) -> None:
        self._ei_db = ei_db
        self._lookup: dict[tuple[str, str], list[Any]] = {}
        self._code_lookup: dict[str, Any] = {}
        self._warnings: list[str] = []

    @property
    def warnings(self) -> list[str]:
        """Advertencias de ambigüedad o de procesos sin coincidencia exacta."""
        return list(self._warnings)

    def build_index(
        self,
        technical_map: dict[str, str],
        location_map: dict[str, str],
    ) -> None:
        """Indexa en memoria los candidatos de Ecoinvent por ``(name, location)``
        para las claves solicitadas.

        Parameters
        ----------
        technical_map : dict[str, str]
            Mapeo {componente: proceso_ei}.
        location_map : dict[str, str]
            Mapeo {componente: ubicación_ei}.
        """
        self._lookup.clear()
        self._code_lookup.clear()
        self._warnings.clear()

        required_keys = {
            (str(v).strip().lower(), str(location_map.get(k, DEFAULT_LOCATION)).strip())
            for k, v in technical_map.items()
        }

        for act in self._ei_db:
            code = act.get("code")
            if code:
                self._code_lookup[str(code)] = act
            key = (act.get("name", "").lower(), act.get("location", ""))
            if key in required_keys:
                self._lookup.setdefault(key, []).append(act)

    def resolve(
        self,
        component: str,
        ei_name: str,
        location: str = DEFAULT_LOCATION,
        unit: str | None = None,
        code: str | None = None,
    ) -> Any | None:
        """Resuelve la actividad Ecoinvent para un componente.

        Parameters
        ----------
        component : str
            Identificador del componente en el dominio.
        ei_name : str
            Nombre del proceso Ecoinvent del mapeo técnico.
        location : str
            Ubicación esperada del proceso.
        unit : str or None
            Unidad del proceso requerida (opcional). Limita la búsqueda por
            nombre a los procesos que comparten esa unidad.
        code : str or None
            Código Ecoinvent (clave canónica). Si se aporta, se usa
            directamente y se ignora ``ei_name``/``location``.

        Returns
        -------
        Any | None
            Actividad resuelta, o None si no hay ninguna coincidencia.
        """
        if code:
            return self._resolve_by_code(component, code, ei_name, location)

        key = (str(ei_name).strip().lower(), str(location).strip())
        candidates = self._lookup.get(key, [])

        if unit:
            candidates = [
                c for c in candidates if str(c.get("unit", "")).strip() == unit
            ]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                self._record_warning(
                    f"{self._describe(component, ei_name, location)}: la unidad "
                    f"'{unit}' no identifica un único proceso "
                    f"({len(candidates)} coincidencias)."
                )
                return self._deterministic(component, ei_name, location, candidates)
            return None

        return self._deterministic(component, ei_name, location, candidates)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _resolve_by_code(
        self,
        component: str,
        code: str,
        ei_name: str,
        location: str,
    ) -> Any | None:
        """Resuelve por la clave canónica de Brightway (``code``).

        La búsqueda se hace contra el índice en memoria construido en
        ``build_index`` (``code`` -> actividad), que es O(1) y funciona tanto
        con la base de datos real como en pruebas.
        """
        act = self._code_lookup.get(str(code).strip())

        if act is None:
            self._record_warning(
                f"CÓDIGO NO ENCONTRADO: el componente '{component}' declaró el "
                f"código Ecoinvent '{code}' pero no existe ninguna actividad con "
                f"esa clave en '{self._ei_db.name}'. Se ignora el código."
            )
            return None

        exp_name = str(ei_name).strip().lower()
        act_name = str(act.get("name", "")).strip().lower()
        if exp_name and act_name != exp_name:
            self._record_warning(
                f"CÓDIGO NOMBRE DISCREPANTE: el componente '{component}' apunta al "
                f"código '{code}' ('{act_name}') pero el mapeo por nombre indica "
                f"'{ei_name}'. Se usa el código (fuente de verdad)."
            )
        return act

    def _deterministic(
        self,
        component: str,
        ei_name: str,
        location: str,
        candidates: list[Any],
    ) -> Any | None:
        """Desempate determinista entre candidatos homónimos.

        Solo se alcanza cuando el inventario no aporta ni ``code`` ni ``unidad``
        y existen varios procesos con el mismo ``(nombre, ubicación)``.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        chosen = sorted(candidates, key=lambda c: c.get("code", ""))[-1]
        units = sorted({str(c.get("unit", "")).strip() for c in candidates})
        self._record_warning(
            f"{self._describe(component, ei_name, location)} coincide con "
            f"{len(candidates)} procesos homónimos de Ecoinvent con unidades "
            f"distintas ({', '.join(units)}). Sin declarar 'code' ni 'unidad', "
            f"se seleccionó '{chosen.get('unit', '')}' ({chosen.get('code', '')}). "
            f"Declare el 'code' (o la 'unidad') en el Excel para eliminar la "
            f"ambigüedad."
        )
        return chosen

    @staticmethod
    def _describe(component: str, ei_name: str, location: str) -> str:
        return (
            f"El componente '{component}' -> proceso '{ei_name}' "
            f"(ubicación '{location}')"
        )

    def _record_warning(self, message: str) -> None:
        self._warnings.append(message)
        logger.warning(message)
