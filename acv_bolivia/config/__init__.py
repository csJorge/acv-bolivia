"""
config: Configuración centralizada del sistema ACV Bolivia.

Provee AppConfig: único punto de acceso a los parámetros del archivo
settings.json con soporte de claves anidadas y rutas con timestamp.

Uso:

    >>> from acv_bolivia.config import AppConfig
    >>> config = AppConfig.load_from_json("config/settings.json")
    >>> config.get("brightway.ecoinvent_db_name")
    'ecoinvent-3.9-cutoff'
"""

from .app_config import AppConfig

__all__ = ["AppConfig"]
