"""
infrastructure.brightway.montecarlo._pedigree_sampler: Incertidumbre por
pedigrí de fondo.

Muestrea los factores de incertidumbre adicional (pedigrí) de la metodología
Ecoinvent, que cuantifica la calidad analítica del dato de inventario en cinco
dimensiones (fiabilidad, integridad, correlación temporal, geográfica y
tecnológica).

Se usa cuando include_pedigree=True en PIVMonteCarloRunner, añadiendo una capa
de incertidumbre epistémica de fondo sobre la incertidumbre paramétrica de
primer plano ya muestreada en los exchanges.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ....core.domain.contracts import MethodId
from ....core.domain.models import Project
from ....infrastructure.brightway.montecarlo._matrix_utils import build_c_stack
from ....infrastructure.brightway.montecarlo._pedigree_stats import PedigreeSamplerStats

logger = logging.getLogger(__name__)


class PedigreeSampler:
    """Indexador estadístico que precarga el caché de variabilidad de fondo por pedigrí.

    La instancia se crea UNA VEZ y se reutiliza para todos los proyectos: procesa
    la unión de procesos únicos de Ecoinvent en una única pasada, reutilizando las
    descomposiciones matriciales CSR entre las tres instalaciones de la tesis.

    Cada proceso se muestrea con MonteCarloLCA sobre su incertidumbre paramétrica
    real y se cachean los coeficientes de impacto unitario por (proceso, método).
    """

    def __init__(
        self,
        bc_module: Any,
        bd_module: Any,
        ei_db_name: str,
        methods: list[MethodId],
        n_samples: int = 1000,
    ) -> None:
        """Configura el entorno del muestreador de pedigrí de fondo.

        Parameters
        ----------
        bc_module : Any
            Módulo bw2calc inyectado.
        bd_module : Any
            Módulo bw2data inyectado.
        ei_db_name : str
            Nombre de la base de datos de Ecoinvent.
        methods : List[MethodId]
            Lista de métodos de impacto.
        n_samples : int, optional
            Número de muestras por proceso. Por defecto 1000.
        """
        self.bc = bc_module
        self.bd = bd_module
        self.ei_db_name = ei_db_name
        self.methods = methods
        self.n_samples = n_samples

        # Caché indexado por (act_key, method_tuple)
        self._cache: dict[tuple[Any, MethodId], NDArray[Any]] = {}
        self._built = False

        # Matriz C reutilizable (se construye una sola vez)
        self._c_stack: sp.csr_matrix | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def n_samples_count(self) -> int:
        """Retorna la dimensión del vector de iteraciones configurado."""
        return self.n_samples

    def build_cache(
        self,
        projects: list[Project],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
    ) -> None:
        """Construye el caché de pedigrí para todos los proyectos.

        Parameters
        ----------
        projects : List[Project]
            Proyectos del dominio.
        technical_maps : Dict[str, Dict[str, str]]
            {project_name: {component: proceso_ei}}.
        location_maps : Dict[str, Dict[str, str]]
            {project_name: {component: ubicación_ei}}.
        """
        if self._built and self._cache:
            logger.info("Caché ya construido. Reutilizando instancias.")
            return

        t0 = time.time()
        logger.info(
            "Compilando caché analítico de pedigrí: %d proyectos | %d métodos | %d "
            "iteraciones.",
            len(projects),
            len(self.methods),
            self.n_samples,
        )

        # 1. Recopilar procesos únicos
        unique_processes = self._collect_unique_processes(
            projects, technical_maps, location_maps
        )
        logger.info(
            "Procesos Ecoinvent identificados de forma única: %d", len(unique_processes)
        )

        # 2. Construir C_stack una sola vez (reutilizable)
        self._c_stack = self._build_c_stack_once()

        # 3. Muestrear cada proceso
        n_ok = 0
        for idx, ((ei_name, ei_loc), ei_act) in enumerate(unique_processes.items()):
            t_proc = time.time()
            ok = self._sample_process(ei_act)
            elapsed = time.time() - t_proc

            if ok:
                n_ok += 1
                logger.debug(
                    "[%d/%d] %s | %s - CONVERGIDO (%.1fs)",
                    idx + 1,
                    len(unique_processes),
                    ei_name[:50],
                    ei_loc,
                    elapsed,
                )
            else:
                logger.warning(
                    "[%d/%d] %s | %s - FALLIDO (%.1fs)",
                    idx + 1,
                    len(unique_processes),
                    ei_name[:50],
                    ei_loc,
                    elapsed,
                )

        self._built = True
        total_time = time.time() - t0
        logger.info(
            "Estructuración concluida: %d/%d procesos consolidados | %d dimensiones "
            "en caché | %.1fs total.",
            n_ok,
            len(unique_processes),
            len(self._cache),
            total_time,
        )

    def get_samples(self, act_key: Any, method_id: MethodId) -> NDArray[Any] | None:
        """Retorna el arreglo estocástico para (proceso, método) o None si no
        fue muestreado.

        Parameters
        ----------
        act_key : Any
            Clave de la actividad de Ecoinvent.
        method_id : MethodId
            Tupla completa del método de impacto.

        Returns
        -------
        Optional[np.ndarray]
            Arreglo de muestras o None.
        """
        return self._cache.get((act_key, method_id))

    def is_cached(self, act_key: Any, method_id: MethodId) -> bool:
        """Valida la presencia del vector numérico dentro del caché."""
        return (act_key, method_id) in self._cache

    def stats(self) -> PedigreeSamplerStats:
        """Retorna estadísticas descriptivas del estado del sampler."""
        if not self._cache:
            return PedigreeSamplerStats(
                built=False,
                n_entries=0,
                n_processes=0,
                n_methods=0,
                n_samples=self.n_samples,
            )
        return PedigreeSamplerStats(
            built=self._built,
            n_entries=len(self._cache),
            n_processes=len({k[0] for k in self._cache}),
            n_methods=len({k[1] for k in self._cache}),
            n_samples=self.n_samples,
        )

    def cleanup(self) -> None:
        """Libera el caché y la matriz C de la memoria."""
        self._cache.clear()
        self._c_stack = None
        self._built = False
        logger.info("Recursos de PedigreeSampler liberados.")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _collect_unique_processes(
        self,
        projects: list[Project],
        technical_maps: dict[str, dict[str, str]],
        location_maps: dict[str, dict[str, str]],
    ) -> dict[tuple[str, str], Any]:
        """Recopila los procesos de fondo únicos de todos los proyectos."""
        required_keys: set[tuple[str, str]] = set()
        for project in projects:
            tech_map = technical_maps.get(project.name, {})
            loc_map = location_maps.get(project.name, {})
            for comp in tech_map:
                ei_name = str(tech_map.get(comp, "")).strip().lower()
                ei_loc = str(loc_map.get(comp, "GLO")).strip()
                if ei_name:
                    required_keys.add((ei_name, ei_loc))

        if not required_keys:
            logger.warning("Matriz de correspondencia vacía. Verifique technical_maps.")
            return {}

        ei_db = self.bd.Database(self.ei_db_name)
        unique_index: dict[tuple[str, str], Any] = {}

        for act in ei_db:
            key = (act.get("name", "").lower(), act.get("location", ""))
            if key in required_keys and key not in unique_index:
                unique_index[key] = act
                if len(unique_index) == len(required_keys):
                    break

        missing_keys = required_keys - set(unique_index.keys())
        if missing_keys:
            logger.warning(
                "Detectados %d procesos huérfanos ausentes en Ecoinvent. Primeros 5: "
                "%s",
                len(missing_keys),
                sorted(missing_keys)[:5],
            )

        return unique_index

    def _build_c_stack_once(self) -> sp.csr_matrix:
        """Construye la matriz C apilada una sola vez (reutilizable)."""
        # Crear un LCA temporal solo para obtener biosphere_dict
        ei_db = self.bd.Database(self.ei_db_name)
        first_act = next(iter(ei_db))
        lca_temp = self.bc.LCA({first_act: 1.0}, self.methods[0])
        lca_temp.lci()
        return build_c_stack(lca_temp, self.methods, self.bd)

    def _sample_process(self, ei_act: Any) -> bool:
        """Muestrea las iteraciones de la grilla de pedigrí para un proceso de fondo."""
        try:
            lca_mc = self.bc.MonteCarloLCA({ei_act: 1.0}, self.methods[0])
            lca_mc.load_data()

            h_matrix = np.zeros((len(self.methods), self.n_samples), dtype=np.float64)

            for i in range(self.n_samples):
                next(lca_mc)
                supply = np.asarray(lca_mc.supply_array).flatten()
                bio_vec = lca_mc.biosphere_matrix @ supply
                h_matrix[:, i] = np.asarray(self._c_stack @ bio_vec).flatten()

            # Inyección hacia el caché
            act_key = ei_act.key
            for m_idx, method in enumerate(self.methods):
                self._cache[(act_key, method)] = h_matrix[m_idx].copy()

            return True

        except Exception:
            logger.exception(
                "Falló el cálculo estocástico en %s.", ei_act.get("name", "?")
            )
            return False
