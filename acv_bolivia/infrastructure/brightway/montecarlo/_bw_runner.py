"""
infrastructure.brightway.montecarlo._bw_runner: BW Montecarlo Completo Secuencial.

Perturba simultáneamente el inventario del proyecto (foreground) y la base de
datos Ecoinvent (background) de forma lineal y secuencial en un único proceso,
optimizando el consumo de memoria y la estabilidad del entorno de ejecución.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ....core.domain.contracts import MethodId
from ....infrastructure.brightway.dto import MonteCarloSimulationResult
from ....infrastructure.brightway.montecarlo._matrix_utils import build_c_stack
from ....infrastructure.brightway.montecarlo._simulation_loop import SimulationLoop

logger = logging.getLogger(__name__)


class MonteCarloRunner:
    """Motor de simulación estocástica secuencial de alto rendimiento.

    Este motor no persiste resultados por sí mismo; los retorna como un objeto
    de transferencia de datos (DTO). La persistencia es responsabilidad exclusiva
    de la capa de aplicación o composición.

    Implementa el protocolo MonteCarloSimulator definido en el dominio.
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        local_db_name: str,
        methods: list[MethodId],
        functional_unit: float = 1.0,
    ) -> None:
        """Inicializa el runner inyectando los módulos de Brightway2.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado desde la capa de infraestructura.
        bd_module : Any
            Módulo bw2data inyectado desde la capa de infraestructura.
        local_db_name : str
            Nombre de la base de datos local SQLite que contiene el inventario.
        methods : List[MethodId]
            Lista de métodos de impacto a evaluar, representados como tuplas
            jerárquicas de Brightway2.
        functional_unit : float, optional
            Cantidad de la unidad funcional para el cálculo. El valor por
            defecto es 1.0.
        """
        self.bc = bc_module
        self.bd = bd_module
        self.local_db_name = local_db_name
        self.methods = methods
        self.functional_unit = functional_unit

        # Estado interno del objeto, gestionado durante el ciclo de vida y
        # liberado explícitamente mediante el método cleanup().
        self._mc_object: Any = None
        self._last_result: MonteCarloSimulationResult | None = None

        logger.info(
            "MonteCarloRunner inicializado en modo secuencial. Resolvedor activo: %s",
            self._get_solver_name(),
        )

    def run(
        self,
        iterations: int,
        functional_unit: float | None = None,
    ) -> MonteCarloSimulationResult:
        """Ejecuta el ciclo estocástico completo y retorna los resultados.

        Parameters
        ----------
        iterations : int
            Número de iteraciones de la simulación de Monte Carlo.
        functional_unit : float, optional
            Valor para sobrescribir la unidad funcional. Si es None, se utiliza
            el valor configurado en el constructor.

        Returns
        -------
        MonteCarloSimulationResult
            Objeto DTO que contiene los scores simulados, metadatos de la
            ejecución y el tiempo transcurrido.
        """
        fu = functional_unit if functional_unit is not None else self.functional_unit

        # 1. Cargar entorno de Brightway2
        local_db = self.bd.Database(self.local_db_name)
        activities = list(local_db)

        if not activities or not self.methods:
            logger.warning("Sin actividades o métodos configurados para la simulación.")
            return MonteCarloSimulationResult(
                scores=np.array([]),
                project_ids=[],
                method_ids=[],
                method_labels=[],
                elapsed_seconds=0.0,
                iterations_completed=0,
                solver_name=self._get_solver_name(),
            )

        act_keys = [act.key for act in activities]
        project_ids = [act.get("name", str(act.key)) for act in activities]
        method_labels = [
            m[1] if isinstance(m, tuple) and len(m) > 1 else str(m)
            for m in self.methods
        ]

        # 2. Inicializar objeto MonteCarloLCA con la primera actividad
        first_act_key = act_keys[0][1]
        first_act = local_db.get(first_act_key)

        self._mc_object = self.bc.MonteCarloLCA({first_act: fu}, self.methods[0])
        self._mc_object.load_data()

        # 3. Construir la matriz de caracterización C (operación realizada una sola vez)
        c_matrix = build_c_stack(
            lca=self._mc_object,
            methods=self.methods,
            bd_module=self.bd,
        )

        # 4. Ejecutar el bucle de simulación estocástica
        loop = SimulationLoop(
            mc_object=self._mc_object,
            act_keys=act_keys,
            c_matrix=c_matrix,
            methods=self.methods,
        )
        scores, elapsed = loop.run(iterations, fu)

        result = MonteCarloSimulationResult(
            scores=scores,
            project_ids=project_ids,
            method_ids=list(self.methods),
            method_labels=method_labels,
            elapsed_seconds=elapsed,
            iterations_completed=iterations,
            solver_name=loop.solver_name,
        )

        self._last_result = result
        return result

    def cleanup(self) -> None:
        """Libera recursos de memoria asociados a matrices y objetos de Brightway2.

        Este método debe ser llamado explícitamente al finalizar el uso del
        runner para evitar fugas de memoria en sesiones prolongadas o notebooks.
        """
        self._mc_object = None
        self._last_result = None
        logger.info("Recursos de MonteCarloRunner liberados.")

    @staticmethod
    def _get_solver_name() -> str:
        """Obtiene el nombre del solucionador de matrices configurado.

        Returns
        -------
        str
            Nombre del solucionador activo (ej. 'scipy_umfpack', 'scipy_superlu').
        """
        from ....infrastructure.brightway.montecarlo._solver import _SOLVER_NAME

        return _SOLVER_NAME
