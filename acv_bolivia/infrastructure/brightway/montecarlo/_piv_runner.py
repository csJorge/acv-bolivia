"""
infrastructure.brightway.montecarlo._piv_runner: PIV Monte Carlo (aproximación lineal).

Implementa la aproximación lineal del análisis de incertidumbre:

    score(iter) = Σᵢ masa_i(iter) × h(i)

donde h(i) = LCA determinístico por unidad del componente i (pre-calculado una vez
por proyecto), acelerado con soporte opcional de variación estocástica de fondo
por pedigrí (PedigreeSampler).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....core.domain.contracts import MethodId
from ....core.domain.models import Project
from ....infrastructure.brightway.dto import PIVSimulationResult
from ....infrastructure.brightway.montecarlo._distributions import sample_vectorized
from ....infrastructure.brightway.montecarlo._pedigree_sampler import PedigreeSampler
from ....infrastructure.brightway.montecarlo._piv_vector_calculator import (
    PivVectorCalculator,
)

logger = logging.getLogger(__name__)


class PIVMonteCarloRunner:
    """
    MC escalar ultrarrápido basado en la linealización local de la tecnosfera.

    NO persiste resultados: los retorna como DTO. La persistencia es
    responsabilidad de la capa de aplicación.

    Implementa el Protocol MonteCarloSimulator.
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        ecoinvent_db_name: str,
        methods: list[MethodId],
        projects: list[Project],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
        sample_processor: Callable | dict[str, Callable],
        code_maps: dict[str, dict[str, str]] | None = None,
        unit_maps: dict[str, dict[str, str]] | None = None,
        functional_unit: float = 1.0,
        seed: int | None = None,
        include_pedigree: bool = False,
        h_pedigree_n: int = 1000,
        correlate_pedigree: bool = False,
    ) -> None:
        """
        Inicializa el runner inyectando dependencias.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado.
        bd_module : Any
            Módulo bw2data inyectado.
        local_db_name : str
            Nombre de la base de datos local.
        ecoinvent_db_name : str
            Nombre de la base de datos de fondo.
        methods : List[MethodId]
            Lista de métodos de impacto.
        projects : List[Project]
            Lista de proyectos del dominio.
        technical_maps : Dict[str, Dict[str, str]]
            Diccionario de mapeos técnicos {project_name: {component: proceso_ei}}.
        location_maps : Dict[str, Dict[str, str]]
            Diccionario de mapeos de ubicación
            {project_name: {component: ubicación_ei}}.
        code_maps : Dict[str, Dict[str, str]], optional
            Diccionario de mapeos de código Ecoinvent
            {project_name: {component: código_ei}}.
        unit_maps : Dict[str, Dict[str, str]], optional
            Diccionario de mapeos de unidad
            {project_name: {component: unidad_ei}}.
        sample_processor : Union[Callable, Dict[str, Callable]]
            Procesador de dependencias (global o por proyecto).
        functional_unit : float, optional
            Unidad funcional. Por defecto 1.0.
        seed : Optional[int], optional
            Semilla pseudoaleatoria. Por defecto None (aleatorio).
        include_pedigree : bool, optional
            Si True, incluye variabilidad del background por pedigrí.
        h_pedigree_n : int, optional
            Número de muestras unitarias por proceso de fondo.
        correlate_pedigree : bool, optional
            Si True, preserva correlación física en el muestreo de pedigrí.
        """
        self.bc = bc_module
        self.bd = bd_module
        self.local_db_name = local_db_name
        self.ecoinvent_db_name = ecoinvent_db_name
        self.methods = methods
        self.projects = projects
        self.technical_maps = technical_maps
        self.location_maps = location_maps
        self.code_maps = code_maps or {}
        self.unit_maps = unit_maps or {}
        self.functional_unit = functional_unit
        self.rng = np.random.default_rng(seed)
        self.include_pedigree = include_pedigree
        self.h_pedigree_n = h_pedigree_n
        self.correlate_pedigree = correlate_pedigree

        # Componentes internos (SRP)
        self._h_calculator = PivVectorCalculator(
            bc_module, bd_module, ecoinvent_db_name
        )
        self._pedigree_sampler: PedigreeSampler | None = None

        # Procesador de muestras
        self.processors: dict[str, Callable] | None = None
        self.processor: Callable | None = None
        if isinstance(sample_processor, dict):
            self.processors = sample_processor
            self.processor = None
        else:
            self.processors = None
            self.processor = sample_processor

        # Estado interno (se limpia con cleanup())
        self._h_vectors: dict[str, dict[MethodId, dict[str, float]]] = {}
        self._ei_caches: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, iterations: int) -> list[PIVSimulationResult]:
        """
        Ejecuta el ciclo PIV y retorna DTOs.

        Parameters
        ----------
        iterations : int
            Número de iteraciones de la simulación.

        Returns
        -------
        List[PIVSimulationResult]
            Lista de DTOs, uno por proyecto.
        """
        start_time = time.time()
        logger.info(
            "Iniciando PIV MC: %d iteraciones × %d proyectos × %d dimensiones.",
            iterations,
            len(self.projects),
            len(self.methods),
        )

        if self.include_pedigree:
            logger.info(
                "Modo pedigrí activo: %d simulaciones unitarias por proceso de fondo.",
                self.h_pedigree_n,
            )
            if self._pedigree_sampler is None:
                self._pedigree_sampler = PedigreeSampler(
                    bc_module=self.bc,
                    bd_module=self.bd,
                    ei_db_name=self.ecoinvent_db_name,
                    methods=self.methods,
                    n_samples=self.h_pedigree_n,
                )
            self._pedigree_sampler.build_cache(
                self.projects, self.technical_maps, self.location_maps
            )

        results: list[PIVSimulationResult] = []

        for project in self.projects:
            logger.info(
                "[%s] Extrayendo coeficientes lineales unitarios...", project.name
            )

            # 1. Calcular h-vectors si no están en caché
            if project.name not in self._h_vectors:
                self._h_vectors[project.name] = self._h_calculator.calculate(
                    project=project,
                    methods=self.methods,
                    technical_map=self.technical_maps.get(project.name, {}),
                    location_map=self.location_maps.get(project.name, {}),
                    code_map=self.code_maps.get(project.name, {}),
                    unit_map=self.unit_maps.get(project.name, {}),
                )
                # Poblar/actualizar caché de actividades EI del proyecto
                self._build_ei_cache(project)

            # 2. Muestrear foreground
            raw_samples = self._sample_foreground(project, iterations)

            proc: (
                Callable[[dict[str, NDArray[Any]]], dict[str, NDArray[Any]]] | None
            ) = None

            if self.processors:
                proc = self.processors.get(project.name)
                # Solo intenta extraer el primer valor si el diccionario NO está vacío
                if proc is None and len(self.processors) > 0:
                    proc = next(iter(self.processors.values()))
            else:
                proc = self.processor

            # Llamada segura: si proc es None, preserva las muestras puras sin fallar
            if proc is not None:
                clean_samples = proc(raw_samples)
            else:
                clean_samples = raw_samples

            # 3. Calcular scores por producto escalar (Preservando correlación)
            method_scores, proj_contribs = self._compute_scores(
                project, clean_samples, iterations
            )

            result = PIVSimulationResult(
                project_id=project.name,
                method_scores=method_scores,
                component_samples=clean_samples,
                piv_contributions=proj_contribs,
                iterations_completed=iterations,
            )
            results.append(result)

        elapsed = time.time() - start_time
        logger.info(
            "Simulación PIV concluida para %d proyectos en %.1fs.",
            len(self.projects),
            elapsed,
        )

        return results

    def cleanup(self) -> None:
        """
        Libera recursos de memoria (h-vectors, cachés, pedigrí).
        """
        self._h_vectors.clear()
        self._ei_caches.clear()
        self._pedigree_sampler = None
        logger.info("Recursos de PIVMonteCarloRunner liberados.")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _build_ei_cache(self, project: Project) -> None:
        """
        Construye la caché de actividades de Ecoinvent para el proyecto e
        inspecciona omisiones.

        Parameters
        ----------
        project : Project
            Instancia del proyecto del dominio para la cual se construye la caché.
        """
        tech_map = self.technical_maps.get(project.name, {})
        loc_map = self.location_maps.get(project.name, {})
        ei_db = self.bd.Database(self.ecoinvent_db_name)

        # Construcción segura de claves requeridas (manejo de valores None)
        required_keys = {
            ((tech_map.get(c) or "").strip().lower(), (loc_map.get(c) or "GLO").strip())
            for c in tech_map
        }

        ei_index: dict[tuple[str, str], Any] = {}
        for act in ei_db:
            name = act.get("name")
            if not name:
                continue
            key = (name.lower(), act.get("location", ""))
            if key in required_keys:
                ei_index[key] = act
                if len(ei_index) == len(required_keys):
                    break

        ei_cache: dict[str, Any] = {}
        for comp in tech_map:
            ei_name = (tech_map.get(comp) or "").strip().lower()
            ei_loc = (loc_map.get(comp) or "GLO").strip()
            act = ei_index.get((ei_name, ei_loc))
            if act:
                ei_cache[comp] = act

        self._ei_caches[project.name] = ei_cache

        # Detección y log de advertencias críticas
        all_domain_components = sorted(
            {
                e.component_id
                for e in project.exchanges
                if e.exchange_type == "technosphere"
            }
        )
        missing_tech_map = [c for c in all_domain_components if c not in tech_map]
        missing_ei_match = [
            c for c in all_domain_components if c in tech_map and c not in ei_cache
        ]

        if missing_tech_map:
            logger.warning(
                "[%s] [AVISO CRÍTICO PIV] %d componente(s) sin entrada en "
                "technical_map: %s",
                project.name,
                len(missing_tech_map),
                missing_tech_map,
            )
        if missing_ei_match:
            logger.warning(
                "[%s] [AVISO CRÍTICO PIV] %d componente(s) sin coincidencia en "
                "Ecoinvent: %s",
                project.name,
                len(missing_ei_match),
                missing_ei_match,
            )

    def _sample_foreground(
        self, project: Project, iterations: int
    ) -> dict[str, NDArray[Any]]:
        """
        Muestrea distribuciones del foreground.

        Parameters
        ----------
        project : Project
            Instancia del proyecto del dominio.
        iterations : int
            Número de iteraciones a muestrear.

        Returns
        -------
        Dict[str, NDArray[Any]]
            Diccionario con arrays de muestras crudas por componente.
        """
        raw_samples: dict[str, NDArray[Any]] = {}
        for exc in project.exchanges:
            if exc.exchange_type != "technosphere":
                continue
            raw_samples[exc.component_id] = sample_vectorized(
                exc.uncertainty, float(exc.quantity.amount), self.rng, iterations
            )
        return raw_samples

    def _compute_scores(
        self,
        project: Project,
        clean_samples: dict[str, NDArray[Any]],
        iterations: int,
    ) -> tuple[dict[MethodId, NDArray[Any]], dict[MethodId, dict[str, NDArray[Any]]]]:
        """
        Calcula los scores de impacto por producto escalar preservando la
        correlación de pedigrí.

        Parameters
        ----------
        project : Project
            Instancia del proyecto del dominio.
        clean_samples : Dict[str, NDArray[Any]]
            Muestras limpias del foreground por componente.
        iterations : int
            Número de iteraciones de la simulación.

        Returns
        -------
        Tuple[Dict[MethodId, NDArray[Any]], Dict[MethodId, Dict[str, NDArray[Any]]]]
            Tupla con dos diccionarios:
            - Diccionario de scores totales por método.
            - Diccionario de contribuciones por componente y método.
        """
        method_scores: dict[MethodId, NDArray[Any]] = {}
        proj_contribs: dict[MethodId, dict[str, NDArray[Any]]] = {}

        # Pre-generación de índices estocásticos compartidos por proyecto y clave
        # de actividad.
        # Esto garantiza que la misma perturbación de pedigrí se aplique a todos los
        # componentes que comparten la misma actividad de fondo (ei_act.key).
        shared_idx: dict[Any, NDArray[Any]] = {}
        if self.include_pedigree and self.correlate_pedigree and self._pedigree_sampler:
            ei_cache = self._ei_caches.get(project.name, {})
            n_samples = self._pedigree_sampler.n_samples
            for comp, ei_act in ei_cache.items():
                key = ei_act.key
                if key not in shared_idx:
                    shared_idx[key] = self.rng.integers(0, n_samples, size=iterations)

        for method in self.methods:
            h_map = self._h_vectors[project.name].get(method, {})
            scores = np.zeros(iterations, dtype=np.float64)
            comp_names = list(h_map.keys())
            contrib_matrix = np.zeros((len(comp_names), iterations), dtype=np.float64)

            for c_idx, comp in enumerate(comp_names):
                h_val = h_map[comp]

                if comp not in clean_samples:
                    nom = float(
                        next(
                            (
                                e.quantity.amount
                                for e in project.exchanges
                                if e.component_id == comp
                            ),
                            0.0,
                        )
                    )
                    c_vec = np.full(iterations, h_val * nom, dtype=np.float64)
                    contrib_matrix[c_idx] = c_vec
                    scores += c_vec
                    continue

                mass_vec = clean_samples[comp]

                if self.include_pedigree and self._pedigree_sampler:
                    ei_act = self._ei_caches.get(project.name, {}).get(comp)
                    h_dist = (
                        self._pedigree_sampler.get_samples(ei_act.key, method)
                        if ei_act
                        else None
                    )

                    if h_dist is not None and len(h_dist) > 0:
                        # Selección del índice: correlacionado si aplica,
                        # independiente si no.
                        if (
                            self.correlate_pedigree
                            and ei_act is not None
                            and ei_act.key in shared_idx
                        ):
                            idx = shared_idx[ei_act.key]
                        else:
                            idx = self.rng.integers(0, len(h_dist), size=iterations)

                        c_vec = mass_vec * h_dist[idx]
                    else:
                        c_vec = mass_vec * h_val
                else:
                    c_vec = mass_vec * h_val

                contrib_matrix[c_idx] = c_vec
                scores += c_vec

            method_scores[method] = scores
            proj_contribs[method] = {
                c: contrib_matrix[idx] for idx, c in enumerate(comp_names)
            }

        return method_scores, proj_contribs
