"""
infrastructure.brightway.montecarlo._fg_runner: MC Estático del Primer Plano.

Varía exclusivamente los parámetros de inventario del Excel (UncertaintyParams),
manteniendo la base de datos de fondo (Ecoinvent) fija en sus valores nominales.
Acelera el cálculo mediante parches in-place en el array de datos de la matriz CSR.

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
from ....infrastructure.brightway.dto import ForegroundSimulationResult
from ....infrastructure.brightway.montecarlo._distributions import sample_vectorized
from ....infrastructure.brightway.montecarlo._fg_matrix_patcher import (
    ForegroundMatrixPatcher,
)

logger = logging.getLogger(__name__)


class ForegroundMCRunner:
    """Motor de simulación estocástica enfocado exclusivamente en el primer plano.

    NO persiste resultados: los retorna como DTO. La persistencia es
    responsabilidad de la capa de aplicación.

    Implementa el Protocol MonteCarloSimulator.
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        methods: list[MethodId],
        projects: list[Project],
        sample_processor: Callable | dict[str, Callable],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
        functional_unit: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Inicializa el runner Foreground MC con todos sus parámetros.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado desde la capa de infraestructura.
        bd_module : Any
            Módulo bw2data inyectado desde la capa de infraestructura.
        local_db_name : str
            Nombre de la base de datos local del inventario.
        methods : List[MethodId]
            Lista de métodos de impacto a evaluar.
        projects : List[Project]
            Lista de proyectos del dominio a simular.
        sample_processor : Callable | dict[str, Callable]
            Procesador de muestras (o diccionario por proyecto) para aplicar
            reglas de dependencias y mezclas.
        technical_maps : Dict[str, Dict[str, str]]
            Mapeo {project_name: {componente: proceso_ecoinvent}} por proyecto.
        location_maps : Dict[str, Dict[str, str]]
            Mapeo {project_name: {componente: ubicación_ecoinvent}} por proyecto.
        functional_unit : float, optional
            Cantidad de la unidad funcional. Por defecto 1.0.
        seed : Optional[int], optional
            Semilla para generación pseudoaleatoria. Por defecto None.
        """
        self.bc = bc_module
        self.bd = bd_module
        self.local_db_name = local_db_name
        self.methods = methods
        self.projects = projects
        self.functional_unit = functional_unit
        self.technical_maps = technical_maps
        self.location_maps = location_maps
        self.rng = np.random.default_rng(seed)

        # Procesador de muestras
        self.processors: dict[str, Callable] | None = None
        self.processor: Callable | None = None
        if isinstance(sample_processor, dict):
            self.processors = sample_processor
            self.processor = None
        else:
            self.processors = None
            self.processor = sample_processor

        # Componente interno (SRP)
        self._patcher = ForegroundMatrixPatcher(
            bc_module, bd_module, methods, functional_unit
        )

    def run(self, iterations: int) -> list[ForegroundSimulationResult]:
        """Ejecuta el ciclo Foreground MC y retorna DTOs.

        Parameters
        ----------
        iterations : int
            Número de iteraciones de la simulación.

        Returns
        -------
        List[ForegroundSimulationResult]
            Lista de DTOs, uno por proyecto.
        """
        local_db = self.bd.Database(self.local_db_name)
        start_time = time.time()

        logger.info(
            "Iniciando Foreground MC: %d iteraciones × %d proyectos × %d dimensiones.",
            iterations,
            len(self.projects),
            len(self.methods),
        )

        results: list[ForegroundSimulationResult] = []

        for project in self.projects:
            logger.info("[%s] Ejecutando perturbaciones de inventario...", project.name)

            # 1. Muestreo vectorizado del foreground
            raw_samples = self._sample_foreground(project, iterations)

            # 2. Aplicar grafo de dependencias físicas
            proc: (
                Callable[[dict[str, NDArray[Any]]], dict[str, NDArray[Any]]] | None
            ) = None

            if self.processors:
                # Si es un diccionario de procesadores por proyecto
                proc = self.processors.get(project.name)
                if proc is None and len(self.processors) > 0:
                    # Fallback al primer procesador disponible si no coincide el nombre
                    # exacto
                    proc = next(iter(self.processors.values()))
            else:
                # Si es un procesador único global
                proc = self.processor

            # Aplicar el procesador si existe; si es None, mantiene las muestras puras
            if proc is not None:
                clean_samples = proc(raw_samples)
            else:
                clean_samples = raw_samples

            # 3. Ejecutar ciclo matricial
            try:
                result = self._run_project(local_db, project, iterations, clean_samples)
                results.append(result)
            except Exception:
                logger.exception("Falló el cálculo matricial en %s.", project.name)

        elapsed = time.time() - start_time
        logger.info("Simulación Foreground MC concluida en %.1fs.", elapsed)

        return results

    def cleanup(self) -> None:
        """Libera recursos."""
        self._patcher.cleanup()
        logger.info("Recursos de ForegroundMCRunner liberados.")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _sample_foreground(
        self, project: Project, iterations: int
    ) -> dict[str, NDArray[Any]]:
        """Muestrea distribuciones del foreground."""
        raw_samples: dict[str, np.ndarray] = {}
        for exc in project.exchanges:
            if exc.exchange_type != "technosphere":
                continue
            raw_samples[exc.component_id] = sample_vectorized(
                exc.uncertainty, float(exc.quantity.amount), self.rng, iterations
            )
        return raw_samples

    def _run_project(
        self,
        local_db: Any,
        project: Project,
        iterations: int,
        processed_samples: dict[str, NDArray[Any]],
    ) -> ForegroundSimulationResult:
        """Ejecuta el ciclo matricial para un proyecto específico.

        Parameters
        ----------
        local_db : Any
            Base de datos local de Brightway2 con las actividades del inventario.
        project : Project
            Entidad del dominio que representa el proyecto a simular.
        iterations : int
            Número de iteraciones de Monte Carlo a ejecutar.
        processed_samples : Dict[str, NDArray[Any]]
            Muestras procesadas por componente, con shape (n_componentes, iterations).

        Returns
        -------
        ForegroundSimulationResult
            DTO con los scores por método, muestras de componentes e iteraciones
            completadas.
        """
        # Obtener los mapas técnicos específicos para este proyecto
        tech_map = self.technical_maps.get(project.name, {})
        loc_map = self.location_maps.get(project.name, {})

        if not tech_map:
            logger.warning(
                "No hay technical_map para el proyecto '%s'. "
                "El parcheador no podrá mapear componentes.",
                project.name,
            )

        if not self._patcher.setup(local_db, project, tech_map, loc_map):
            return ForegroundSimulationResult(
                project_id=project.name,
                method_scores={},
                component_samples=processed_samples,
                iterations_completed=0,
            )

        scores_fg: dict[int, list[float]] = {
            m_idx: [] for m_idx in range(len(self.methods))
        }

        for it in range(iterations):
            current_samples = {
                comp: float(processed_samples[comp][it])
                for _, comp in self._patcher._relevant_rows
            }
            scores_iter = self._patcher.patch_and_solve(current_samples)

            for m_idx in range(len(self.methods)):
                scores_fg[m_idx].append(float(scores_iter[m_idx]))

        self._patcher.restore()
        self._log_convergence(scores_fg, iterations)

        method_scores = {
            method: np.array(scores_fg[m_idx], dtype=np.float64)
            for m_idx, method in enumerate(self.methods)
        }

        return ForegroundSimulationResult(
            project_id=project.name,
            method_scores=method_scores,
            component_samples=processed_samples,
            iterations_completed=iterations,
        )

    def _log_convergence(
        self, scores_fg: dict[int, list[float]], iterations: int
    ) -> None:
        """Registra métricas de convergencia (CV)."""
        valid = [s for s in scores_fg[0] if not np.isnan(s)]
        if len(valid) > 1:
            cv = np.std(valid) / abs(np.mean(valid)) * 100
            logger.info(
                "Integridad: %d/%d iteraciones válidas | CV (Método Principal) = "
                "%.2f%%",
                len(valid),
                iterations,
                cv,
            )
