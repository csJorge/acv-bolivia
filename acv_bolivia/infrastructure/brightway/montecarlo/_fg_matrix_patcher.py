"""
infrastructure.brightway.montecarlo._fg_matrix_patcher: Parcheo in-place
de la matriz de tecnosfera.

Gestiona el ciclo de vida de la matriz parcheada para Foreground MC:
construcción inicial, parcheo iterativo durante el loop de simulación,
y restauración final al estado nominal.

Responsabilidad única: orquestar el mapeo de componentes, el parcheo de la
matriz CSR y la resolución del sistema lineal en cada iteración.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp

from ....core.domain.models import Project
from ....infrastructure.brightway.montecarlo._component_mapper import (
    ComponentToMatrixMapper,
)
from ....infrastructure.brightway.montecarlo._matrix_utils import (
    build_c_stack,
    build_data_positions,
    patch_matrix,
)

logger = logging.getLogger(__name__)


class ForegroundMatrixPatcher:
    """Gestiona el parcheo in-place de la matriz de tecnosfera para Foreground MC.

    Este componente coordina tres responsabilidades:
        1. Mapear los componentes del dominio a posiciones en la matriz CSR.
        2. Parchear in-place los valores de la matriz en cada iteración.
        3. Resolver el sistema lineal y calcular los scores de impacto.

    El parcheo in-place evita el costo O(nnz) de reconstruir la matriz completa
    en cada iteración, reduciéndolo a O(k) donde k es el número de componentes
    con incertidumbre (típicamente 5-15).
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        methods: list[tuple[str, ...]],
        functional_unit: float = 1.0,
    ) -> None:
        """Inicializa el parcheador con los módulos de Brightway2 y los métodos.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado desde la capa de infraestructura.
        bd_module : Any
            Módulo bw2data inyectado desde la capa de infraestructura.
        methods : List[Tuple[str, ...]]
            Lista de métodos de impacto a evaluar, representados como tuplas
            jerárquicas de Brightway2.
        functional_unit : float, optional
            Cantidad de la unidad funcional para el cálculo. Por defecto 1.0.
        """
        self.bc = bc_module
        self.bd = bd_module
        self.methods = methods
        self.functional_unit = functional_unit

        self._mapper = ComponentToMatrixMapper()
        self._lca: Any = None
        self._bw_activity: Any = None
        self._relevant_rows: list[tuple[int, str]] = []
        self._data_positions: list[tuple[int, int, float]] = []
        self._c_stack: sp.csr_matrix | None = None
        self._dom_nominals: dict[str, float] = {}

    def setup(
        self,
        local_db: Any,
        project: Project,
        technical_map: dict[str, str],
        location_map: dict[str, str],
    ) -> bool:
        """Inicializa el parcheador para un proyecto específico.

        Este método debe llamarse una sola vez por proyecto antes de iniciar
        el loop de simulación. Factoriza el LCA maestro, construye el índice
        de componentes y pre-calcula las posiciones en la matriz CSR.

        Parameters
        ----------
        local_db : Any
            Base de datos local de Brightway2 con las actividades del inventario.
        project : Project
            Entidad del dominio que representa el proyecto a simular.
        technical_map : Dict[str, str]
            Mapeo {componente: proceso_ecoinvent} para resolver actividades de fondo.
        location_map : Dict[str, str]
            Mapeo {componente: ubicación_ecoinvent} para resolver actividades de fondo.

        Returns
        -------
        bool
            True si la inicialización fue exitosa, False si la actividad raíz
            no se encontró en la base de datos local.
        """
        self._bw_activity = next(
            (act for act in local_db if act.get("name") == project.name), None
        )
        if self._bw_activity is None:
            logger.warning(
                "Actividad raíz '%s' no localizada en la base de datos.",
                project.name,
            )
            return False

        # Factorizar una sola vez (background fijo para todo el loop)
        self._lca = self.bc.LCA(
            {self._bw_activity: self.functional_unit}, self.methods[0]
        )
        self._lca.lci()
        self._lca.lcia()

        act_dict = (
            self._lca.dicts.activity
            if hasattr(self._lca, "dicts") and hasattr(self._lca.dicts, "activity")
            else self._lca.activity_dict
        )

        # Mapear componentes del dominio a posiciones en la matriz CSR
        self._relevant_rows, _baseline = self._mapper.map(
            lca=self._lca,
            bw_activity=self._bw_activity,
            project=project,
            act_dict=act_dict,
            technical_map=technical_map,
            location_map=location_map,
        )

        # Construir la matriz de caracterización C apilada (una sola vez)
        self._c_stack = build_c_stack(self._lca, self.methods, self.bd)

        # Pre-calcular posiciones en CSR .data para parcheo O(k)
        self._data_positions = build_data_positions(self._lca, self._relevant_rows)

        # Capturar valores nominales de los exchanges del dominio
        self._dom_nominals = {
            exc.component_id: float(exc.quantity.amount)
            for exc in project.exchanges
            if exc.exchange_type == "technosphere"
        }

        logger.info(
            "Patcher configurado para '%s': %d componentes mapeados, %d métodos.",
            project.name,
            len(self._relevant_rows),
            len(self.methods),
        )
        return True

    def patch_and_solve(self, current_samples: dict[str, float]) -> np.ndarray:
        """Parchea la matriz con las muestras actuales y resuelve el sistema.

        Este método se llama en cada iteración del loop de Monte Carlo.
        Modifica in-place los valores de la matriz CSR, recalcula el inventario
        y el impacto, y retorna los scores por método.

        Parameters
        ----------
        current_samples : Dict[str, float]
            Diccionario {component_id: valor_muestreado} para esta iteración.
            Los componentes no presentes en el diccionario se restauran a su
            valor nominal.

        Returns
        -------
        np.ndarray
            Array unidimensional con los scores de impacto por método, en el
            mismo orden que la lista de métodos proporcionada al constructor.
        """
        patch_matrix(
            self._lca,
            self._relevant_rows,
            self._data_positions,
            current_samples,
            self._dom_nominals,
        )

        self._lca.redo_lci({self._bw_activity: self.functional_unit})
        supply = np.asarray(self._lca.supply_array).flatten()
        bio_vec = self._lca.biosphere_matrix @ supply
        scores = np.asarray(self._c_stack @ bio_vec).flatten()

        return scores

    def restore(self) -> None:
        """Restaura la matriz a su estado nominal tras finalizar la simulación.

        Debe llamarse al finalizar el loop de Monte Carlo para garantizar que
        la base de datos local quede en un estado consistente para otros cálculos.
        """
        patch_matrix(
            self._lca,
            self._relevant_rows,
            self._data_positions,
            {},
            self._dom_nominals,
        )

    def cleanup(self) -> None:
        """Libera todos los recursos internos (matrices, objetos BW2, cachés).

        Debe llamarse explícitamente al finalizar el uso del runner para evitar
        fugas de memoria en sesiones prolongadas o notebooks.
        """
        self._lca = None
        self._bw_activity = None
        self._relevant_rows = []
        self._data_positions = []
        self._c_stack = None
        self._dom_nominals = {}
        logger.info("Recursos de ForegroundMatrixPatcher liberados.")
