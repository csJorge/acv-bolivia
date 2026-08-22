"""
infrastructure.brightway.bw_lca_calculator: Calculador determinístico de
impactos y hotspots.

Resuelve el sistema lineal de inventario (A · s = f) y evalúa los vectores
caracterizados de impacto (h = C · B · s) para todas las actividades del
proyecto y categorías ambientales de Brightway2.

Algoritmo:
    1. Factoriza un LCA maestro una sola vez por método (demanda 0 sobre
       todos los proyectos).
    2. Para cada proyecto: redo_lci() + redo_lcia() → score determinístico.
    3. Para hotspots: aísla la contribución de cada exchange tecnosfera
       al score total, usando marcas de componente inyectadas en el exchange.

El orden correcto es siempre redo_lci() → redo_lcia() (nunca al revés).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, cast

from ...core.domain.contracts import MethodId
from ...infrastructure.brightway.constants import (
    DEFAULT_IMPACT_UNIT,
    EXCHANGE_COMPONENT_TAG,
)
from ...infrastructure.brightway.dto import (
    DeterministicScoreDTO,
    HotspotDTO,
    LCACalculationResult,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Componentes internos (SRP)
# ==============================================================================


class _ComponentResolver:
    """Resuelve la correspondencia entre procesos EI y componentes del dominio.

    Soporta dos formatos de mapeo:
    - v2: {(proceso_EI_name, amount): componente} → desambiguación exacta por monto
    - legacy: {proceso_EI_name: componente} → fallback cuando no hay match por monto

    Ante ausencia de match exacto, aproxima por monto más cercano.
    """

    def __init__(self, process_to_component: dict[Any, str] | None = None) -> None:
        self._by_amount: dict[tuple[str, float], str] = {}
        self._fallback: dict[str, str] = {}

        for k, v in (process_to_component or {}).items():
            if isinstance(k, tuple):
                self._by_amount[k] = v
            else:
                self._fallback[k] = v

    def resolve(self, process_name: str, amount: float) -> str | None:
        """Resuelve un componente por proceso Ecoinvent y monto.

        Parameters
        ----------
        process_name : str
            Nombre del proceso de Ecoinvent.
        amount : float
            Monto del exchange para desambiguar componentes.

        Returns
        -------
        str or None
            Identificador del componente o ``None`` si no hay coincidencia.
        """
        if not self._by_amount and not self._fallback:
            return None

        if self._by_amount:
            candidates = {
                k: v for k, v in self._by_amount.items() if k[0] == process_name
            }
            if candidates:
                exact_match = candidates.get((process_name, amount))
                if exact_match:
                    return exact_match
                closest_key = min(candidates.keys(), key=lambda k: abs(k[1] - amount))
                return candidates[closest_key]

        return self._fallback.get(process_name)


class _ScoreCache:
    """Caché de scores determinísticos por (actividad, método)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, str, float], float] = {}

    def get_or_compute(
        self,
        lca: Any,
        act: Any,
        functional_unit: float,
        method_label: str,
    ) -> float:
        """Obtiene un score cacheado o evalúa la actividad.

        Parameters
        ----------
        lca : Any
            Objeto LCA inicializado.
        act : Any
            Actividad que se evaluará.
        functional_unit : float
            Cantidad de unidad funcional.
        method_label : str
            Etiqueta del método de impacto.

        Returns
        -------
        float
            Score determinístico.
        """
        key = (act.key, method_label, functional_unit)
        if key not in self._cache:
            lca.redo_lci({act: functional_unit})
            lca.redo_lcia()
            self._cache[key] = float(lca.score)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()


class _HotspotCache:
    """Caché de contribuciones de hotspots por (actividad, método)."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[Any, str], list[tuple[str, float, float, str | None]]
        ] = {}

    def get_or_compute(
        self,
        lca: Any,
        act: Any,
        method_label: str,
        top_n: int | None,
    ) -> list[tuple[str, float, float, str | None]]:
        """Obtiene contribuciones de exchanges, usando el caché disponible.

        Parameters
        ----------
        lca : Any
            Objeto LCA inicializado.
        act : Any
            Actividad cuyos exchanges se evaluarán.
        method_label : str
            Etiqueta del método de impacto.
        top_n : int or None
            Máximo de contribuciones devueltas; ``None`` devuelve todas.

        Returns
        -------
        list of tuple
            Tuplas ``(nombre, impacto, monto, componente)`` ordenadas por impacto.
        """
        key = (act.key, method_label)
        if key not in self._cache:
            contributions: list[tuple[str, float, float, str | None]] = []
            for exc in act.technosphere():
                lca.redo_lci({exc.input: float(exc["amount"])})
                lca.redo_lcia()
                contributions.append(
                    (
                        str(exc.input["name"]),
                        float(lca.score),
                        float(exc["amount"]),
                        exc.get(EXCHANGE_COMPONENT_TAG),
                    )
                )
            contributions.sort(key=lambda x: x[1], reverse=True)
            self._cache[key] = contributions
        return self._cache[key] if top_n is None else self._cache[key][:top_n]

    def clear(self) -> None:
        self._cache.clear()


# ==============================================================================
# Calculador principal (Orquestador)
# ==============================================================================


class LCACalculator:
    """Ejecuta cálculos LCIA determinísticos con caché interno.

    NO persiste resultados: los retorna como DTOs. La persistencia
    es responsabilidad de la capa de aplicación.
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        methods: list[MethodId],
        process_to_component: dict[Any, str] | None = None,
    ) -> None:
        self.bc = bc_module
        self.bd = bd_module
        self.local_db = bd_module.Database(local_db_name)
        self.methods = methods

        # Componentes internos (SRP)
        self._resolver = _ComponentResolver(process_to_component)
        self._score_cache = _ScoreCache()
        self._hotspot_cache = _HotspotCache()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(
        self,
        top_n_hotspot: int | None = 50,
        functional_unit: float = 1.0,
    ) -> LCACalculationResult:
        """Ejecuta la evaluación de impacto completa y retorna DTOs.

        Factoriza un LCA maestro una sola vez por método (demanda 0 sobre
        todos los proyectos) y luego reutiliza esa factorización para
        acelerar los redo_lci()/redo_lcia() de cada proyecto individual.

        Returns
        -------
        LCACalculationResult
            DTO con scores, hotspots y tiempo de ejecución.

        Raises
        ------
        ValueError
            Si ``top_n_hotspot`` es menor que cero o ``functional_unit`` no
            es finita y positiva.
        """
        if top_n_hotspot is not None and top_n_hotspot < 0:
            raise ValueError("top_n_hotspot debe ser mayor o igual que cero.")
        if not (
            isinstance(functional_unit, (int, float))
            and math.isfinite(float(functional_unit))
            and functional_unit > 0
        ):
            raise ValueError("functional_unit debe ser un número positivo.")

        activities = list(self.local_db)
        if not activities or not self.methods:
            logger.warning(
                "No se encontraron actividades o métodos para el cálculo "
                "determinístico."
            )
            return LCACalculationResult(scores=[], hotspots=[], elapsed_seconds=0.0)

        logger.info(
            "Iniciando evaluaciones: %d procesos x %d dimensiones ambientales.",
            len(activities),
            len(self.methods),
        )
        global_start = time.time()

        all_scores: list[DeterministicScoreDTO] = []
        all_hotspots: list[HotspotDTO] = []

        for i, method in enumerate(self.methods):
            method_label = self._build_method_label(method)
            unit = self._get_method_unit(method)
            logger.info(
                "[%d/%d] Procesando dimensión: %s",
                i + 1,
                len(self.methods),
                method[1],
            )

            # Factorización unificada: demanda maestra con todos los proyectos a 0
            master_demand = {act: 0.0 for act in activities}
            lca = self.bc.LCA(master_demand, method)
            lca.lci()
            lca.lcia()

            for act in activities:
                project_name = act["name"]
                t0 = time.time()

                score = self._score_cache.get_or_compute(
                    lca, act, functional_unit, method_label
                )
                all_scores.append(
                    DeterministicScoreDTO(
                        project_id=project_name,
                        method_id=method,
                        method_label=method_label,
                        score=score,
                        unit=unit,
                    )
                )

                hotspots = self._compute_hotspots(
                    lca=lca,
                    act=act,
                    method_id=method,
                    method_label=method_label,
                    unit=unit,
                    top_n=top_n_hotspot,
                )
                all_hotspots.extend(hotspots)

                logger.info(
                    "  %s -> Score: %.4e | Tiempo: %.2fs",
                    project_name,
                    score,
                    time.time() - t0,
                )

        elapsed = time.time() - global_start
        logger.info("Ciclo determinístico concluido en %.1fs.", elapsed)

        return LCACalculationResult(
            scores=all_scores,
            hotspots=all_hotspots,
            elapsed_seconds=elapsed,
        )

    def clear_cache(self) -> None:
        """Libera la memoria de los cachés internos.

        Notes
        -----
        Útil al reutilizar una instancia para distintas corridas.
        """
        self._score_cache.clear()
        self._hotspot_cache.clear()
        logger.info("Cachés del calculador LCA liberados.")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _compute_hotspots(
        self,
        lca: Any,
        act: Any,
        method_id: MethodId,
        method_label: str,
        unit: str,
        top_n: int | None,
    ) -> list[HotspotDTO]:
        """Aísla y ordena la contribución individual de cada exchange.

        Parameters
        ----------
        lca : Any
            Objeto LCA de Brightway2 inicializado para el método actual.
        act : Any
            Actividad del proyecto que se está evaluando.
        method_id : MethodId
            Identificador completo del método de impacto.
        method_label : str
            Etiqueta legible del método.
        unit : str
            Unidad del impacto.
        top_n : int or None
            Número máximo de hotspots. ``None`` devuelve todos.

        Returns
        -------
        list of HotspotDTO
            Contribuciones resolubles ordenadas por impacto descendente.
        """
        project_name = act["name"]
        cached = self._hotspot_cache.get_or_compute(lca, act, method_label, top_n)

        # Obtenemos el score total del proyecto para calcular el porcentaje
        # de contribución
        total_score = self._score_cache.get_or_compute(lca, act, 1.0, method_label)

        hotspots = []
        for name, impact, amount, comp_tag in cached:
            component_id = (
                comp_tag
                if comp_tag is not None
                else self._resolver.resolve(name, amount)
            )
            if component_id is None:
                logger.warning(
                    "Se omite hotspot sin componente resoluble: '%s' (%s).",
                    name,
                    project_name,
                )
                continue
            percentage = (
                (abs(impact) / abs(total_score)) * 100 if total_score != 0 else 0.0
            )
            hotspots.append(
                HotspotDTO(
                    project_id=project_name,
                    method_id=method_id,
                    method_label=method_label,
                    background_process_name=name,
                    component_id=component_id,
                    impact=impact,
                    unit=unit,
                    percentage=percentage,
                )
            )

        return hotspots

    def _build_method_label(self, method: MethodId) -> str:
        """Construye la etiqueta legible del método: 'Nombre (unidad)'."""
        method_name = method[1] if len(method) > 1 else str(method)
        unit = self._get_method_unit(method)
        return f"{method_name} ({unit})"

    def _get_method_unit(self, method: MethodId) -> str:
        """Obtiene la unidad del método desde bw2data."""
        method_meta = self.bd.methods.get(method, {})
        return cast(str, method_meta.get("unit", DEFAULT_IMPACT_UNIT))
