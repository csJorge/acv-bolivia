"""
infrastructure.brightway.bw_activity_repository: Repositorio Concreto de Actividades.

Compila el inventario de ciclo de vida (LCI) interactuando con Brightway2 (SQLite).
Encapsula la creación de Activities e Exchanges, aplicando auditorías estrictas
para interceptar componentes huérfanos antes de la factorización matricial.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.domain.models import Exchange, Project
from ...core.domain.uncertainty import UncertaintyParams
from ...infrastructure.brightway.constants import (
    BIOSPHERE_DB_NAME,
    BIOSPHERE_EXCHANGE_TYPE,
    DEFAULT_LOCATION,
    DEFAULT_UNIT,
    PRODUCTION_EXCHANGE_TYPE,
    TECHNOSPHERE_EXCHANGE_TYPE,
)
from ...infrastructure.brightway.ei_name_resolver import EcoinventNameResolver
from ...infrastructure.brightway.validators import validate_ecoinvent_db

logger = logging.getLogger(__name__)


# ==============================================================================
# Componentes internos (SRP: cada clase tiene una sola responsabilidad)
# ==============================================================================


class _EcoinventIndexer:
    """Indexa la base de datos de Ecoinvent en memoria para búsqueda rápida."""

    def __init__(self, bd_module: Any, ecoinvent_db_name: str) -> None:
        self.bd = bd_module
        self.ecoinvent_db_name = ecoinvent_db_name
        self._cache: dict[str, Any] = {}
        self._mapping_errors: list[str] = []
        self._warnings: list[str] = []
        self._resolver: EcoinventNameResolver | None = None

    @property
    def cache(self) -> dict[str, Any]:
        return self._cache

    @property
    def mapping_errors(self) -> list[str]:
        return list(self._mapping_errors)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def build_index(
        self,
        technical_map: dict[str, str],
        location_map: dict[str, str],
        code_map: dict[str, str] | None = None,
        unit_map: dict[str, str] | None = None,
    ) -> None:
        """Construye el índice en memoria a partir de los mapeos técnicos."""
        self._cache.clear()
        self._mapping_errors.clear()
        self._warnings.clear()
        logger.info(
            "Cargando mapa asociativo de '%s' en memoria...", self.ecoinvent_db_name
        )
        ei_db = self.bd.Database(self.ecoinvent_db_name)

        code_map = code_map or {}
        unit_map = unit_map or {}

        self._resolver = EcoinventNameResolver(ei_db)
        self._resolver.build_index(technical_map, location_map)
        self._warnings.extend(self._resolver.warnings)

        for comp, ei_name in technical_map.items():
            loc = str(location_map.get(comp, DEFAULT_LOCATION)).strip()
            found_activity = self._resolver.resolve(
                comp,
                ei_name,
                loc,
                unit=unit_map.get(comp),
                code=code_map.get(comp),
            )
            cache_key = str(comp).strip().lower()

            if found_activity is not None:
                self._cache[cache_key] = found_activity
            else:
                self._mapping_errors.append(
                    f"Componente huérfano: '{comp}' -> Proceso '{ei_name}' "
                    f"no localizado con ubicación '{loc}'."
                )

        logger.info(
            "[Caché] %d procesos indexados | %d desalineaciones críticas | "
            "%d advertencias.",
            len(self._cache),
            len(self._mapping_errors),
            len(self._warnings),
        )

    def get_activity(self, component: str) -> Any | None:
        """Retorna la actividad de Ecoinvent asociada al componente o None."""
        return self._cache.get(str(component).strip().lower())


class _BiosphereResolver:
    """Resuelve flujos elementales de la biosfera."""

    def __init__(self, bd_module: Any) -> None:
        self.bd = bd_module
        self._cache: dict[str, Any] = {}

    def resolve(self, component_name: str) -> Any | None:
        """Busca un flujo elemental en biosphere3 por nombre normalizado."""
        key = str(component_name).strip().lower()
        if key in self._cache:
            return self._cache[key]

        bio_search = [
            f for f in self.bd.Database(BIOSPHERE_DB_NAME) if f["name"].lower() == key
        ]
        if bio_search:
            self._cache[key] = bio_search[0]
            return bio_search[0]
        return None


class _ExchangeFactory:
    """Factory para crear intercambios de Brightway2 con parámetros correctos."""

    def __init__(
        self,
        indexer: _EcoinventIndexer,
        biosphere_resolver: _BiosphereResolver,
    ) -> None:
        self.indexer = indexer
        self.biosphere_resolver = biosphere_resolver
        self.errors: list[str] = []

    def create_and_attach(self, bw_activity: Any, exchange: Exchange) -> None:
        """Crea un intercambio BW2 y lo adjunta a la actividad padre."""
        comp_key = str(exchange.component_id).strip().lower()

        if exchange.exchange_type == BIOSPHERE_EXCHANGE_TYPE:
            bio_act = self.biosphere_resolver.resolve(comp_key)
            if bio_act is not None:
                bw_activity.new_exchange(
                    input=bio_act,
                    amount=float(exchange.quantity.amount),
                    type=BIOSPHERE_EXCHANGE_TYPE,
                ).save()
                return
            self._record_error(
                f"DIVERGENCIA CRÍTICA: El flujo de biosfera "
                f"'{exchange.component_id}' no existe en biosphere3."
            )
            return

        if exchange.exchange_type != TECHNOSPHERE_EXCHANGE_TYPE:
            self._record_error(
                f"Tipo de exchange inválido para '{exchange.component_id}': "
                f"'{exchange.exchange_type}'."
            )
            return

        ei_act = self.indexer.get_activity(comp_key)
        if ei_act is None:
            self._record_error(
                f"DIVERGENCIA CRÍTICA: El componente '{exchange.component_id}' "
                f"no posee un proceso asignado en Ecoinvent."
            )
            return

        params: dict[str, Any] = {
            "input": ei_act,
            "amount": float(exchange.quantity.amount),
            "type": TECHNOSPHERE_EXCHANGE_TYPE,
            "component": exchange.component_id,
        }

        if exchange.uncertainty and isinstance(exchange.uncertainty, UncertaintyParams):
            stats = exchange.uncertainty.get_statistical_properties(
                float(exchange.quantity.amount)
            )
            bw_uncertainty_dict = {
                "uncertainty type": stats["type_id"],
                "loc": stats["loc"],
                "scale": stats["scale"],
            }
            if stats["minimum"] is not None:
                bw_uncertainty_dict["minimum"] = stats["minimum"]
            if stats["maximum"] is not None:
                bw_uncertainty_dict["maximum"] = stats["maximum"]

            params.update(bw_uncertainty_dict)

        bw_activity.new_exchange(**params).save()

    def _record_error(self, message: str) -> None:
        logger.error(message)
        self.errors.append(message)


# ==============================================================================
# Repositorio principal (Orquestador)
# ==============================================================================


class ActivityRepository:
    """Repositorio de infraestructura que compila el tecnosfera en Brightway2.

    Orquesta:
    - Validación de Ecoinvent
    - Indexación del fondo en memoria
    - Creación de actividades locales
    - Persistencia de intercambios
    """

    def __init__(
        self,
        bd_module: Any,
        local_db_name: str,
        ecoinvent_db_name: str,
        error_output_dir: str | None = None,
    ) -> None:
        self.bd = bd_module
        self.local_db_name = local_db_name
        self.ecoinvent_db_name = ecoinvent_db_name
        self.error_output_dir = error_output_dir

        # Componentes internos (SRP)
        self._indexer = _EcoinventIndexer(bd_module, ecoinvent_db_name)
        self._biosphere_resolver = _BiosphereResolver(bd_module)
        self._exchange_factory = _ExchangeFactory(
            self._indexer, self._biosphere_resolver
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def validate_ecoinvent(self) -> bool:
        """Verifica la disponibilidad de la base de datos de fondo de Ecoinvent."""
        report = validate_ecoinvent_db(
            available_databases=set(self.bd.databases),
            expected_db_name=self.ecoinvent_db_name,
        )
        if not report.is_valid:
            logger.warning("Falló la validación estructural de fondo:\n%s", report)
        return report.is_valid

    def build(
        self,
        projects: list[Project],
        location_map: dict[str, str],
        technical_map: dict[str, str],
        code_map: dict[str, str] | None = None,
        unit_map: dict[str, str] | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """Compila y sincroniza la base de datos local con las actividades del
        dominio."""
        self._exchange_factory.errors.clear()
        if not self.validate_ecoinvent():
            raise RuntimeError(
                f"La base de datos de fondo de Ecoinvent '{self.ecoinvent_db_name}' "
                f"no se encuentra registrada en el entorno activo de Brightway2."
            )

        # Control del ciclo de vida de la BD local
        if self.local_db_name in self.bd.databases:
            if force_rebuild:
                logger.info(
                    "Purgando base de datos local existente: '%s'...",
                    self.local_db_name,
                )
                del self.bd.databases[self.local_db_name]
            else:
                db = self.bd.Database(self.local_db_name)
                if len(db) > 0:
                    logger.info(
                        "Base de datos local '%s' detectada. Cargando caché.",
                        self.local_db_name,
                    )
                    self._indexer.build_index(
                        technical_map, location_map, code_map, unit_map
                    )
                    self._report_warnings()
                    if self._indexer.mapping_errors:
                        self._export_errors()
                        raise RuntimeError(
                            "No se puede reutilizar el inventario: existen "
                            f"componentes sin mapeo en Ecoinvent "
                            f"({len(self._indexer.mapping_errors)})."
                        )
                    return

        # Construcción del caché
        self._indexer.build_index(technical_map, location_map, code_map, unit_map)
        self._report_warnings()

        if self._indexer.mapping_errors:
            self._export_errors()
            raise RuntimeError(
                "No se puede construir el inventario: existen componentes "
                f"sin mapeo en Ecoinvent ({len(self._indexer.mapping_errors)})."
            )

        # Inserción de nodos físicos
        self._create_activities(projects)

        if self._exchange_factory.errors:
            self._export_errors()
            raise RuntimeError(
                "La construcción del inventario contiene errores de exchanges: "
                f"{len(self._exchange_factory.errors)}."
            )

    def get_ecoinvent_activity(self, component: str) -> Any | None:
        """Retorna el nodo de actividad de Ecoinvent asociado al componente o None."""
        return self._indexer.get_activity(component)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _report_warnings(self) -> None:
        """Registra en el logger las advertencias de mapeo no bloqueantes."""
        for warning in self._indexer.warnings:
            logger.warning(warning)

    def _create_activities(self, projects: list[Project]) -> None:
        """Registra y procesa los nodos de actividades en la BD local SQLite."""
        db_local = self.bd.Database(self.local_db_name)
        db_local.register()

        logger.info(
            "Compilando %d actividades raíz en '%s'...",
            len(projects),
            self.local_db_name,
        )

        for project in projects:
            self._create_single_activity(db_local, project)

        db_local.process()
        logger.info(
            "Compilación concluida. Base de datos '%s' lista para cálculo.",
            self.local_db_name,
        )

    def _create_single_activity(self, db_local: Any, project: Project) -> None:
        """Crea una actividad Brightway desde una entidad Project del dominio."""
        act = db_local.new_activity(
            code=str(project.id),
            name=project.name,
            unit=DEFAULT_UNIT,
        )
        act.save()

        # Auto-intercambio de producción
        act.new_exchange(input=act, amount=1.0, type=PRODUCTION_EXCHANGE_TYPE).save()

        # Intercambios de tecnosfera y biosfera
        for exchange in project.exchanges:
            self._exchange_factory.create_and_attach(act, exchange)

        act.save()
        logger.info(
            "Actividad '%s' consolidada con %d enlaces.",
            project.name,
            len(project.exchanges),
        )

    def _export_errors(self) -> None:
        """Exporta los errores de mapeo a un archivo si hay directorio configurado."""
        if not self.error_output_dir:
            return

        import os

        os.makedirs(self.error_output_dir, exist_ok=True)
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(
            self.error_output_dir, f"mapping_errors_{timestamp}.txt"
        )

        with open(error_file, "w", encoding="utf-8") as f:
            f.write("ERRORES DE MAPEO\n")
            f.write("=" * 60 + "\n\n")
            f.write("ADVERTENCIAS (no bloqueantes)\n")
            f.write("-" * 60 + "\n")
            if self._indexer.warnings:
                f.write("\n".join(self._indexer.warnings))
            else:
                f.write("(ninguna)")
            f.write("\n\nERRORES CRÍTICOS\n")
            f.write("-" * 60 + "\n")
            f.writelines(
                f"{err}\n"
                for err in self._indexer.mapping_errors + self._exchange_factory.errors
            )

        logger.warning(
            "Se exportaron %d errores de mapeo a: %s",
            len(self._indexer.mapping_errors) + len(self._exchange_factory.errors),
            error_file,
        )
