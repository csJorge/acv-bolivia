"""
application.use_cases.build_inventory - Caso de uso: construcción del inventario.

Orquesta la lectura del inventario, la validación y la construcción del modelo
de dominio. No conoce implementaciones concretas de infraestructura.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging

from ...application.contracts import (
    BrightwayActivityBuilder,
    BrightwayConnectionManager,
    InventoryDataAuditor,
    InventoryLoader,
)
from ...application.dto.build_inventory import BuildInventoryResult
from ...config.app_config import AppConfig

logger = logging.getLogger(__name__)


class BuildInventoryUseCase:
    """Construye el inventario ACV orquestando abstracciones inyectadas.

    No conoce implementaciones concretas (Excel, Brightway2, etc.).
    """

    def __init__(
        self,
        loader: InventoryLoader,
        auditor: InventoryDataAuditor,
        connection_manager: BrightwayConnectionManager,
        activity_builder: BrightwayActivityBuilder,
        config: AppConfig,
    ) -> None:
        """Inyección de dependencias mediante constructor.

        Parameters
        ----------
        loader : InventoryLoader
            Cargador de inventario (Excel, JSON, etc.)
        auditor : InventoryDataAuditor
            Auditor de calidad de datos
        connection_manager : BrightwayConnectionManager
            Gestor de conexión a Brightway2
        activity_builder : BrightwayActivityBuilder
            Constructor de actividades en BW2
        config : AppConfig
            Configuración de la aplicación
        """
        self.loader = loader
        self.auditor = auditor
        self.connection_manager = connection_manager
        self.activity_builder = activity_builder
        self.config = config

    def run(self, force_rebuild: bool = False) -> BuildInventoryResult:
        """Ejecuta la construcción del inventario.

        Parameters
        ----------
        force_rebuild : bool
            Si True, borra la BD local existente y la reconstruye.

        Returns
        -------
        BuildInventoryResult
            Resultado con proyectos cargados, mapeos, configuración MC y nombre de
            la BD local.
        """
        # 1. Cargar inventario desde fuente abstracta
        logger.info("Iniciando carga del inventario...")
        inventory = self.loader.load()

        # 2. Auditar calidad de datos
        data_quality = self.auditor.audit(
            project_rows=inventory.raw_data.project_rows,
            mc_rows=inventory.raw_data.mc_rows,
            mc_mode=self.config.get("mc_mode", "full"),
        )

        if data_quality.warnings:
            logger.warning("Advertencias de calidad de datos:\n%s", data_quality)

        if not inventory.validation.is_valid:
            logger.error("Validación de inventario fallida: %s", inventory.validation)
            return BuildInventoryResult(
                projects=[],
                technical_map=inventory.technical_map,
                location_map=inventory.location_map,
                code_map=inventory.code_map,
                unit_map=inventory.unit_map,
                generation_dict=inventory.generation_dict,
                local_db_name="",
                mc_config=inventory.mc_config,
                data_quality=data_quality,
                success=False,
                error_message=str(inventory.validation),
            )

        # 3. Conectar a Brightway2 (abstracción)
        try:
            logger.info("Conectando a Brightway2...")
            self.connection_manager.connect()
            local_db_name = self.config.get("inventario_nombre")
        except RuntimeError as e:
            logger.error("Error de conexión a Brightway2: %s", e)
            return BuildInventoryResult(
                projects=inventory.projects,
                technical_map=inventory.technical_map,
                location_map=inventory.location_map,
                code_map=inventory.code_map,
                unit_map=inventory.unit_map,
                generation_dict=inventory.generation_dict,
                local_db_name="",
                mc_config=inventory.mc_config,
                data_quality=data_quality,
                success=False,
                error_message=f"Error de conexión: {e!s}",
            )

        # 4. Construir actividades en Brightway2 (abstracción)
        try:
            logger.info(
                "Construyendo actividades en Brightway2 (force_rebuild=%s)...",
                force_rebuild,
            )
            self.activity_builder.build(
                projects=inventory.projects,
                location_map=inventory.location_map,
                technical_map=inventory.technical_map,
                code_map=inventory.code_map,
                unit_map=inventory.unit_map,
                force_rebuild=force_rebuild,
            )
        except RuntimeError as e:
            logger.error("Error construyendo actividades en Brightway2: %s", e)
            return BuildInventoryResult(
                projects=inventory.projects,
                technical_map=inventory.technical_map,
                location_map=inventory.location_map,
                code_map=inventory.code_map,
                unit_map=inventory.unit_map,
                generation_dict=inventory.generation_dict,
                local_db_name=local_db_name,
                mc_config=inventory.mc_config,
                data_quality=data_quality,
                success=False,
                error_message=f"Error construyendo actividades: {e!s}",
            )

        logger.info(
            "Inventario construido exitosamente: %d proyectos en '%s'",
            len(inventory.projects),
            local_db_name,
        )

        return BuildInventoryResult(
            projects=inventory.projects,
            technical_map=inventory.technical_map,
            location_map=inventory.location_map,
            code_map=inventory.code_map,
            unit_map=inventory.unit_map,
            generation_dict=inventory.generation_dict,
            local_db_name=local_db_name,
            mc_config=inventory.mc_config,
            data_quality=data_quality,
            success=True,
        )
