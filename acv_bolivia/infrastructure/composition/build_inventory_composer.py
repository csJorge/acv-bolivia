"""
infrastructure.composition.build_inventory_composer: Compositor del caso
de uso BuildInventoryUseCase.

Este modulo es el punto de ensamblaje donde se inyectan las implementaciones
concretas de infraestructura en el caso de uso BuildInventoryUseCase, satisfaciendo
los protocolos definidos en application/contracts.py.

El compositor orquesta:
    - El cargador de inventario (ExcelInventoryLoader)
    - El auditor de datos (InventoryDataAuditorAdapter)
    - El conector a Brightway2 (BrightwayConnector)
    - El constructor de actividades (ActivityRepository)

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...application.use_cases.build_inventory import BuildInventoryUseCase
from ...config.app_config import AppConfig
from ...infrastructure.brightway import (
    ActivityRepository,
    BrightwayConnector,
)
from ...infrastructure.input import ExcelInventoryLoader
from ...infrastructure.input.auditors import InventoryDataAuditorAdapter

logger = logging.getLogger(__name__)


def create_build_inventory_use_case(
    config: AppConfig,
) -> BuildInventoryUseCase:
    """Factory method que compone todas las dependencias del caso de uso.

    Parameters
    ----------
    config : AppConfig
        Configuracion de la aplicacion con las rutas y nombres necesarios.

    Returns
    -------
    BuildInventoryUseCase
        Caso de uso listo para ejecutar con todas las dependencias inyectadas.

    Raises
    ------
    KeyError
        Si la configuracion no tiene las claves necesarias.

    Example
    -------
    >>> from config.app_config import AppConfig
    >>> from infrastructure.composition import create_build_inventory_use_case
    >>>
    >>> config = AppConfig()
    >>> use_case = create_build_inventory_use_case(config)
    >>> result = use_case.run(force_rebuild=False)
    """
    # 1. Instanciar implementaciones concretas
    excel_path = Path(config.get("rutas.inventario"))
    logger.info("Creando cargador de inventario desde: %s", excel_path)
    loader = ExcelInventoryLoader(path=excel_path)

    logger.info("Creando auditor de datos de inventario")
    auditor = InventoryDataAuditorAdapter()

    bw_project_name = config.get("proyecto")
    bw_dir = config.get("rutas.bw2")
    env_path = config.get("rutas.entorno")
    logger.info(
        "Creando conector a Brightway2: proyecto=%s, dir=%s",
        bw_project_name,
        bw_dir,
    )
    connection_manager = BrightwayConnector(
        project_name=bw_project_name,
        brightway_dir=bw_dir,
        env_path=env_path,
    )

    connection_manager.connect()

    local_db_name = config.get("inventario_nombre")
    ecoinvent_db_name = config.get("ecoinvent_source_db_name")
    error_output_dir = config.get_dated_output_folder("errores_mapeo")
    logger.info(
        "Creando repositorio de actividades: local_db=%s, ecoinvent_db=%s",
        local_db_name,
        ecoinvent_db_name,
    )
    activity_builder = ActivityRepository(
        bd_module=connection_manager.get_data_module(),
        local_db_name=local_db_name,
        ecoinvent_db_name=ecoinvent_db_name,
        error_output_dir=error_output_dir,
    )

    # 2. Inyectar en el caso de uso
    logger.info("Ensamblando BuildInventoryUseCase con todas las dependencias")
    return BuildInventoryUseCase(
        loader=loader,
        auditor=auditor,
        connection_manager=connection_manager,
        activity_builder=activity_builder,
        config=config,
    )
